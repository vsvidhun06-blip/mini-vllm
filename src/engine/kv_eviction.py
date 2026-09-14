"""
H2O KV-cache eviction -- serve sequences longer than the cache budget.

THE PROBLEM
-----------
The KV cache grows by one entry per generated token, at every layer. A fixed
memory budget therefore caps the sequence length: once the cache is full you
either stop, or you drop context. For long-context generation that cap is the
bottleneck, not compute.

H2O (Zhang et al. 2023, "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of LLMs") observes that attention is extremely sparse: a small set of
"heavy hitter" tokens receive the bulk of the attention mass across the whole
generation, and the rest contribute almost nothing. So you can EVICT the
low-attention tokens from the cache and keep quality nearly intact -- as long as
you also always keep a window of the most RECENT tokens (recency is critical:
the newest tokens dominate local prediction and their scores haven't had time to
accumulate yet, so a pure top-k by score would wrongly drop them).

Keep set after eviction  =  top-k heavy hitters (by cumulative attention)
                            +  the last `recent_window` tokens (always).

With a budget of B tokens you can then generate sequences far longer than B:
each time the cache fills, drop the cold tokens and keep going. This module
gives ~2x+ effective context in the same memory.

WHAT'S HERE
-----------
  * AttentionScoreTracker -- accumulates per-position attention mass (summed
    over heads, queries, and layers) and decides which positions to evict.
  * EvictingPagedKVCache  -- a per-request KV cache (drop-in for
    PagedRequestCache) wrapping a PagedKVCache pool. It tracks scores, and when
    occupancy crosses a threshold it compacts the survivors into the leading
    blocks, frees the rest, and keeps RoPE positions correct.
  * make_evicting_cache / generate_with_eviction -- convenience builders for
    the tests and the benchmark.

HOW SCORES ARE CAPTURED
-----------------------
attention.py checks whether the cache it was handed carries a `score_tracker`.
If so it computes attention via an un-fused softmax (the only way to see the
weights -- SDPA/FA2 fuse and discard them) and calls `tracker.update(layer_idx,
weights)` as a side effect. So just passing an EvictingPagedKVCache as the
model's `kv_cache` turns tracking on; nothing in model.py changes.

POSITION REMAPPING (why RoPE stays correct)
-------------------------------------------
Eviction removes tokens from the MIDDLE of the sequence and compacts the rest,
so the cache now holds fewer entries than tokens generated. Two counters per
layer keep this honest:
  * entry count (PagedRequestCache._seq_lens) -- how many entries are physically
    cached; drives block layout and the gather length. Drops on eviction.
  * RoPE offset (_rope_offset) -- the TRUE absolute position of the next token =
    total tokens ever appended. Monotonic; NEVER drops on eviction.
attention reads the RoPE offset via `seq_len()` to rotate the new token, so the
new token is placed at its real absolute position even though the cache is
compacted. Surviving keys keep the rotation they were cached with (RoPE is
relative -- rotations at absolute m and n give a dot product depending only on
m - n), so every surviving pair stays correct.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from src.engine.kv_cache import PagedKVCache, PagedRequestCache

if TYPE_CHECKING:
    from src.engine.model import LlamaModel


class AttentionScoreTracker:
    """Accumulates per-token attention mass and picks H2O eviction victims.

    The score of position p is the total attention weight it RECEIVES, summed
    over all query positions, all heads, and all layers, accumulated across
    every forward pass since the last eviction realigned the indices. Scores are
    indexed by CACHE SLOT and kept aligned with the cache's entries: a new token
    extends the score vector by one; an eviction drops the same slots from both.
    """

    def __init__(self, num_layers: int, recent_window: int = 32) -> None:
        self.num_layers = num_layers
        self.recent_window = recent_window
        # 1-D cumulative score per slot, on CPU (eviction logic is host-side;
        # the per-step reduction below moves the small (S_k,) vector to CPU).
        self.scores = torch.zeros(0, dtype=torch.float64)

    def __len__(self) -> int:
        return int(self.scores.numel())

    def update(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        """Add one layer's attention weights into the running per-slot scores.

        Args:
            layer_idx: which layer produced these weights. H2O sums across
                layers, so this is informational (kept for the documented API
                and possible per-layer policies); all layers feed one
                accumulator.
            attn_weights: (..., S_k) -- typically (B, NQ, S_q, S_k). We sum over
                every axis except the last (the key/slot axis) to get the mass
                each cached position received this call.
        """
        # Reduce to per-key mass, in fp64 on CPU for stable accumulation.
        reduce_axes = tuple(range(attn_weights.dim() - 1))
        per_key = attn_weights.detach().to(torch.float64).sum(dim=reduce_axes).cpu()
        n = per_key.numel()
        # Grow the accumulator if the sequence got longer (a new token slot).
        if n > self.scores.numel():
            pad = torch.zeros(n - self.scores.numel(), dtype=torch.float64)
            self.scores = torch.cat([self.scores, pad])
        self.scores[:n] += per_key

    def get_eviction_candidates(self, keep_budget: int) -> list[int]:
        """Slot indices to EVICT so that the survivors are the top scorers plus
        the recency window.

        Survivors = the last `recent_window` slots (always kept)
                    + the highest-scoring of the remaining slots, up to
                      `keep_budget` survivors total.
        Returns the complement (the slots to drop), or [] if we're already at or
        under budget.
        """
        n = int(self.scores.numel())
        if n <= keep_budget:
            return []

        recent = min(self.recent_window, n)
        survivors: set[int] = set(range(n - recent, n))  # recency: always keep

        # Fill the rest of the budget with heavy hitters drawn from the
        # non-recent region (recent slots are already guaranteed survivors).
        budget_for_heavy = max(0, keep_budget - recent)
        n_non_recent = n - recent
        if budget_for_heavy > 0 and n_non_recent > 0:
            non_recent_scores = self.scores[:n_non_recent]
            k = min(budget_for_heavy, n_non_recent)
            top = torch.topk(non_recent_scores, k).indices.tolist()
            survivors.update(int(i) for i in top)

        return [i for i in range(n) if i not in survivors]

    def evict(self, evict_positions: list[int]) -> None:
        """Drop the given slot indices from the score vector and compact it,
        preserving the ascending order of survivors (so it stays aligned with
        the cache, which compacts the same way)."""
        if not evict_positions:
            return
        ev = set(evict_positions)
        keep = [i for i in range(self.scores.numel()) if i not in ev]
        self.scores = self.scores[keep].contiguous()


class EvictingPagedKVCache(PagedRequestCache):
    """Per-request KV cache with H2O eviction, wrapping a PagedKVCache pool.

    Drop-in for PagedRequestCache (same seq_len/append/get contract), so the
    model can use it as `kv_cache` directly. The presence of `score_tracker`
    flips attention.py onto the weight-capturing path. After each decode step
    the owner calls `maybe_evict()`; when occupancy crosses the threshold it
    compacts the survivors and frees blocks.
    """

    def __init__(
        self,
        pool: PagedKVCache,
        request_id: str,
        num_layers: int,
        capacity_tokens: int,
        keep_budget: int | None = None,
        recent_window: int = 32,
        evict_threshold: float = 0.8,
        eviction_observer: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(pool, request_id, num_layers)
        # PagedRequestCache stores per-layer state but not the layer count;
        # _evict_slots iterates over it, so keep our own copy.
        self.num_layers = num_layers
        self.capacity_tokens = capacity_tokens
        self.recent_window = recent_window
        self.evict_threshold = evict_threshold
        # Optional observability hook: called with the number of tokens dropped
        # each time an eviction fires. Stays None in the engine/test path (no
        # prometheus dependency here); the server/benchmark wires
        # metrics.observe_eviction into it.
        self.eviction_observer = eviction_observer
        # How many tokens to keep after an eviction. Default: half the budget,
        # but never below the recency window (otherwise recency is impossible).
        if keep_budget is None:
            keep_budget = max(recent_window, capacity_tokens // 2)
        self.keep_budget = keep_budget

        self.score_tracker = AttentionScoreTracker(num_layers, recent_window)

        # RoPE offset per layer = true absolute position of the next token.
        # Monotonic; survives eviction (unlike _seq_lens, the entry count).
        self._rope_offset: list[int] = [0] * num_layers
        # Absolute position of each cached slot (for inspection / the position-
        # remapping test). Updated once per token (on the layer-0 append).
        self._abs_positions: list[int] = []
        self._tokens_seen = 0

        # Stats.
        self.num_evictions = 0
        self.evicted_tokens = 0

    # ---- interface overrides (decouple RoPE position from entry count) ------

    def seq_len(self, layer_idx: int = 0) -> int:
        """RoPE position offset for the next token == true absolute position.

        This is what attention reads to rotate the new token, so the token
        lands at its real position even though the cache holds fewer entries.
        """
        return self._rope_offset[layer_idx]

    def entry_count(self, layer_idx: int = 0) -> int:
        """Number of physically cached entries (post-compaction)."""
        return self._seq_lens[layer_idx]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        S_new = k_new.shape[1]
        # Record absolute positions once per token, on the first layer's append.
        if layer_idx == 0:
            for j in range(S_new):
                self._abs_positions.append(self._tokens_seen + j)
            self._tokens_seen += S_new
        # Base class writes K/V at slot _seq_lens[layer] and advances it (the
        # entry count). NOTE: base append reads self._seq_lens directly, not
        # seq_len(), so our seq_len() override does not disturb the write slot.
        super().append(layer_idx, k_new, v_new)
        # Advance the monotonic RoPE position separately.
        self._rope_offset[layer_idx] += S_new

    # ---- eviction -----------------------------------------------------------

    def should_evict(self) -> bool:
        return self._seq_lens[0] >= self.evict_threshold * self.capacity_tokens

    def maybe_evict(self) -> int:
        """Evict if occupancy has crossed the threshold. Returns #tokens evicted.

        Call once per decode step (after the forward, between tokens) so the
        per-layer caches are in lockstep when we compact them.
        """
        if not self.should_evict():
            return 0
        evict = self.score_tracker.get_eviction_candidates(self.keep_budget)
        if not evict:
            return 0
        self._evict_slots(evict)
        self.num_evictions += 1
        self.evicted_tokens += len(evict)
        if self.eviction_observer is not None:
            self.eviction_observer(len(evict))
        return len(evict)

    def _evict_slots(self, evict: list[int]) -> None:
        """Remove the given slots from every layer's K/V, compact survivors into
        the leading blocks, free the emptied blocks, and realign the tracker and
        position map."""
        ev = set(evict)
        n = self._seq_lens[0]
        survivors = [i for i in range(n) if i not in ev]  # ascending == temporal
        keep = len(survivors)
        device = self.pool.K_pool.device
        idx = torch.tensor(survivors, dtype=torch.long, device=device)
        bs = self.pool.block_size

        for layer in range(self.num_layers):
            # get() returns a COPY (advanced index + reshape), so selecting and
            # writing back into the same blocks does not alias.
            k_full, v_full = self.get(layer)              # (1, n, NKV, D)
            k_keep = k_full.index_select(1, idx)          # (1, keep, NKV, D)
            v_keep = v_full.index_select(1, idx)
            self._write_compacted(layer, k_keep, v_keep)
            self._seq_lens[layer] = keep

        # Free the now-unused trailing blocks (block table is shared by layers).
        keep_blocks = (keep + bs - 1) // bs
        self.pool.trim_request_blocks(self.request_id, keep_blocks)
        # The cached block-table tensor is now stale (shorter table); drop it so
        # the next get() rebuilds it.
        self._bt_tensor = None

        # Realign the score vector and the absolute-position map identically.
        self.score_tracker.evict(evict)
        self._abs_positions = [self._abs_positions[i] for i in survivors]

    def _write_compacted(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write the survivor K/V (1, keep, NKV, D) into the leading slots
        (0..keep-1) of this layer's blocks, block by block -- the same slicing
        append() uses, but starting from slot 0."""
        keep = k.shape[1]
        bs = self.pool.block_size
        block_table = self.pool.get_block_table(self.request_id)
        written = 0
        while written < keep:
            pos = written
            phys = block_table[pos // bs]
            slot = pos % bs
            n_w = min(bs - slot, keep - written)
            self.pool.K_pool[layer_idx, phys, slot:slot + n_w] = k[0, written:written + n_w]
            self.pool.V_pool[layer_idx, phys, slot:slot + n_w] = v[0, written:written + n_w]
            written += n_w


# ---------------------------------------------------------------------------
# Convenience builders for tests + the benchmark.
# ---------------------------------------------------------------------------


def make_evicting_cache(
    model: "LlamaModel",
    capacity_tokens: int,
    keep_budget: int | None = None,
    recent_window: int = 32,
    evict_threshold: float = 0.8,
    block_size: int = 16,
    request_id: str = "h2o",
    eviction_observer: Callable[[int], None] | None = None,
) -> EvictingPagedKVCache:
    """Build an EvictingPagedKVCache backed by a fresh pool sized to
    `capacity_tokens` (NOT the full sequence length -- eviction is what lets a
    long sequence live inside this bounded budget).

    The request is admitted reserving the WHOLE budget up front. As generation
    grows the cache toward the budget, blocks are drawn from that reservation;
    eviction frees blocks back into it; the request never needs more physical
    blocks than the budget, no matter how many tokens it ultimately generates.
    """
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    capacity_blocks = (capacity_tokens + block_size - 1) // block_size
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=capacity_blocks,
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    # Reserve the whole budget: prefill_blocks=0 (nothing cached yet), the rest
    # reserved for JIT growth + post-eviction regrowth.
    pool.admit_request(
        request_id=request_id,
        prefill_blocks_needed=0,
        total_blocks_needed=capacity_blocks,
    )
    return EvictingPagedKVCache(
        pool=pool,
        request_id=request_id,
        num_layers=cfg.num_hidden_layers,
        capacity_tokens=capacity_tokens,
        keep_budget=keep_budget,
        recent_window=recent_window,
        evict_threshold=evict_threshold,
        eviction_observer=eviction_observer,
    )


@torch.no_grad()
def run_sequence_with_eviction(
    model: "LlamaModel",
    token_ids: torch.Tensor,
    capacity_tokens: int,
    keep_budget: int | None = None,
    recent_window: int = 32,
    evict_threshold: float = 0.8,
) -> dict:
    """Teacher-force `token_ids` through the model using an evicting cache, one
    token at a time (so eviction can fire between steps), accumulating the
    negative log-likelihood of each next actual token.

    Returns a dict with perplexity, token count, eviction stats, and the final
    resident cache size -- everything the benchmark prints. Teacher forcing (vs
    free generation) makes perplexity comparable across cache policies: every
    policy is scored on predicting the SAME reference tokens.
    """
    device = next(model.parameters()).device
    ids = token_ids.to(device).reshape(1, -1)
    T = ids.shape[1]
    cache = make_evicting_cache(
        model, capacity_tokens, keep_budget, recent_window, evict_threshold
    )

    nll_sum = 0.0
    nll_count = 0
    for t in range(T):
        step_input = ids[:, t:t + 1]                       # (1, 1)
        logits = model(step_input, kv_cache=cache)         # (1, 1, V)
        cache.maybe_evict()
        if t + 1 < T:
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            nll_sum += -logp[int(ids[0, t + 1])].item()
            nll_count += 1

    ppl = float(torch.tensor(nll_sum / max(1, nll_count)).exp())
    return {
        "tokens": T,
        "resident_tokens": cache.entry_count(),
        "num_evictions": cache.num_evictions,
        "evicted_tokens": cache.evicted_tokens,
        "perplexity": ppl,
    }
