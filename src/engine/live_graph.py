"""
CUDA graph capture + replay for the LIVE decode step (continuous batching).

WHY THIS MODULE EXISTS ALONGSIDE cuda_graph.py
----------------------------------------------
`src/engine/cuda_graph.CUDAGraphRunner` is a MICROBENCHMARK runner. It builds
its own dedicated `PagedKVCache`, seeds it to a fixed `seq_len`, captures a
decode step against those cache objects, and `can_replay` enforces cache
IDENTITY. Two properties make it unusable for live serving:

  1. It is bound to caches it created. The scheduler decodes against `r.cache`
     objects it created for real requests, which are never the runner's caches,
     so `can_replay(B, live_caches)` is False by identity.
  2. Its graphs freeze `seq_len`. Live decode increments every request's
     seq_len every step, so even a matching batch size never matches twice.

(The proximate cause of `cuda_graph_hits == 0` was simpler still: nothing in the
repository ever assigned `scheduler._graph_runner`, so the graph branch in
`_decode_forward` was unreachable regardless of the above.)

THE DESIGN: STATIC SHAPES + ON-DEVICE METADATA (the vLLM approach)
------------------------------------------------------------------
A CUDA graph replays a fixed sequence of kernels against fixed addresses, with
every host-side value that shaped the launch baked in. To serve a growing,
churning decode batch from a captured graph, every host-dependent quantity in
the decode forward has to become the CONTENTS of a device tensor at a STABLE
address, and every tensor SHAPE has to become constant.

There are exactly four such quantities in this engine's batched-decode path:

  a) the new token ids                -> `static_input`  (B, 1)
  b) each row's RoPE position         -> `static_ctx`    (B,)     [pre-append]
  c) each row's KV block table        -> `static_bt`     (B, NB)
  d) the slot each row writes K/V to  -> `static_write`  (B,)

and one shape that used to be host-dependent:

  e) `PagedRequestCache.get()` returned `k_flat[:S]` -- S a Python int, so the
     attention shape changed every step. We instead gather the WHOLE padded
     block table, `(NB * block_size, NKV, D)`, a constant shape, and suppress
     the out-of-context tail with an ADDITIVE -inf bias

         `static_bias` (B, NB * block_size)   0.0 where pos <= ctx, -inf else

     which the FA2 kernel already supports (`attn_mask`, an (S_q, S_k) additive
     float bias). Padded columns contribute `exp(-inf - m) == 0` exactly, and a
     fully-masked FA2 KV tile folds in as `p == 0, alpha == 1`, leaving the
     online-softmax accumulator untouched -- so on the CUDA path graph and eager
     decode agree BITWISE. (Position 0 is always valid, so `m_i` is never -inf
     when a masked tile is folded in and no NaN can arise.) On CPU the
     attention is `F.scaled_dot_product_attention`, which chooses its own
     blocking per key length and disagrees with itself at ~1e-7 between a
     masked and a trimmed call on identical q/k/v; the CPU test therefore
     asserts a tight tolerance plus exact agreement on the emitted token, and
     the GPU test asserts bitwise equality. See
     tests/test_engine/test_live_graph.py.

  Item (d) replaces `PagedRequestCache.append`'s host-indexed slice write
  (`K_pool[layer, phys, slot] = ...`, which would bake `phys`/`slot` into the
  graph and rewrite the same slot forever) with an on-device `index_copy_` into
  a flattened view of the pool. The index is read from `static_write`, so each
  replay writes wherever the live block table says.

Nothing here is bound to particular request objects: a captured graph is bound
to the scheduler's ONE shared `PagedKVCache` and to its own static buffers. Any
set of B live requests whose contexts fit the bucket can be served by it.

DYNAMIC SHAPES: WHAT IS CAPTURED, AND WHAT IS NOT
-------------------------------------------------
Two dimensions vary at run time, and each is handled EXPLICITLY -- never by
capturing one shape and pretending it covers the others:

  * DECODE BATCH SIZE. One graph per batch size. We capture every batch size in
    `1..min(max_batch_size, max_graph_batch)`. There is NO batch padding: a
    decode batch of 3 runs the graph captured for 3, or it runs eager. Padding a
    batch of 3 up to a captured 4 would inflate that row's work by 33% in a
    batch-size-dependent way, which is precisely the slope the Phase 16
    experiment exists to measure.

  * KV CONTEXT LENGTH. One graph per (batch size, block-table length) bucket.
    Buckets are block counts, so the padding a bucket introduces is bounded by
    the bucket spacing (default: 4 buckets spanning the workload's maximum
    context). A decode step whose longest row needs more blocks than the largest
    bucket falls back to eager and is COUNTED as a fallback.

Any (batch size, bucket) pair that was not captured -- including because capture
itself failed, e.g. OOM -- is an eager fallback, counted separately and never
reported as a graph hit. `fallback_reasons` records why.

  * LORA ROUTING PLAN. A third dimension, and the one that is easiest to get
    wrong, because it is invisible in the tensor shapes. `LoRALinear.forward`
    branches on `self._active`: base rows take the zero-overhead path (one
    GEMM), a single active adapter takes `_apply_single`, and a mixed batch
    takes `_apply_per_row`. Capture freezes whichever branch was live, together
    with that adapter's weight tensors and, for `_apply_per_row`, the row-index
    tensors. A graph captured under plan P therefore computes P's arithmetic no
    matter what the scheduler later routes -- replaying it under plan Q would
    silently serve Q's requests with P's adapters.

    So the routing plan is part of GRAPH IDENTITY, not a property checked
    alongside it:

        key = (batch_size, n_blocks, canonical_routing_plan)

    canonicalised by `src.engine.lora.normalize_routing_plan` -- the same
    function `model_routing_plan` uses, so the scheduler and this module cannot
    disagree about what plan is live. `None` is base, a `str` is one adapter for
    every row, and a tuple is per-row routing; an all-`None` tuple collapses to
    `None`, because "every row is base" IS base. Tuples (never lists) so the
    plan can be part of a dict key.

    A T4 established that capture under an active per-row plan SUCCEEDS
    (16/16 graphs, 1.324s, zero capture failures), which is what makes routed
    graphs worth having at all; see `test_cuda_capture_under_active_per_row_routing`.

ON-DEMAND CAPTURE OF ROUTED PLANS
---------------------------------
`capture_all()` runs before any request is admitted, so it necessarily captures
the UNROUTED plan: nothing is routed yet. That is the right default -- base
decode is the common case -- but it cannot cover routing, because which adapter
combinations will share a decode batch is a property of the arrival pattern and
the scheduler's batching, not something a caller can enumerate up front.

So a routed plan is captured the first time it is actually encountered, from the
very step that needs it, and reused for every later step with the same plan.

The capture-time hazard that makes `capture_all()` demand an empty pool does not
apply here, and the reason is worth being precise about.

`capture_all()` stages PLACEHOLDER metadata -- block 0, slot 0, context 0 --
because it has no real batch to point at. Those writes land on whoever owns
block 0, which is why it may only run on an empty pool.

An on-demand capture stages the LIVE batch's metadata instead, so its writes go
to the slot THIS decode step is already going to overwrite. The token ids staged
for capture are placeholder zeros (`can_replay` is not given the real ones, and
a graph does not bake token VALUES anyway -- only the address of
`static_input`), so during capture that slot transiently holds the K/V of token
0 rather than of the real token. That is safe because of what happens next, and
only because of it: this step then either replays the graph just captured or --
if the capture failed -- runs eager, and both write the correct K/V to that same
slot, computed from the same pre-append `_seq_lens`, before anything reads it.
The transient value is never observable.

`commit()` is NOT called by capture; only `replay` advances the host-side
`_seq_lens`, exactly once, so the context bookkeeping cannot double-step.

Routed graphs are capped by `max_routed_graphs`. Adapter combinations grow
combinatorially and each graph pins a static-buffer set plus its share of the
memory pool; past the cap, further plans are counted eager fallbacks
(`routed_graph_budget_exhausted`) rather than being allowed to exhaust device
memory mid-serve. A capture that fails for any other reason (OOM included) is
caught, counted as `routed_capture_failed`, and the step runs eager -- an
on-demand capture must never be able to take down a live server.

CAPTURE ORDERING CONSTRAINT
---------------------------
`capture_all()` must run BEFORE any request is admitted to the pool. Capture and
its warmup forwards genuinely execute the decode step, writing K/V into the pool
at whatever slots the placeholder metadata points at (block 0, slot 0). On an
empty pool that is harmless scratch; on a live pool it would corrupt a real
request's KV. The constructor enforces this.

MEMORY POOL
-----------
All graphs share one CUDA graph memory pool (vLLM does the same). Sharing is
safe here because the only tensor a caller ever reads out of graph-private
memory is `static_logits`, and `replay()` clones it before returning, so no
graph-private tensor stays live across another graph's replay.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from src.engine.lora import normalize_routing_plan

if TYPE_CHECKING:
    from src.engine.kv_cache import PagedKVCache, PagedRequestCache
    from src.engine.model import LlamaModel

NEG_INF = float("-inf")

# "argument omitted", which is not the same as the routing plan `None` (base).
_UNSET = object()


class GraphCaptureError(RuntimeError):
    """A CUDA graph capture failed. Never downgrade this to eager execution."""


class GraphSelfTestError(RuntimeError):
    """Capture succeeded but replay could not be demonstrated against live
    request caches. The graph arm must not run."""


class _StaticRowCache:
    """One decode row, duck-typed as a `PagedRequestCache` for the model.

    `MultiHeadAttention._forward_decode_batched` needs exactly three things from
    a cache -- `seq_len_tensor`, `append`, `get` -- plus (new) an optional
    `decode_attn_bias`. All four are backed here by views into the owning
    `StaticDecodeBatch`'s static buffers, so their ADDRESSES never change and a
    captured graph keeps reading the right memory while the CONTENTS change per
    replay.

    Deliberately NOT implemented: `seq_len()` (the host-side int). Nothing on
    the batched-decode path reads it, and returning a stale value would be a
    silent correctness bug, so it raises instead.
    """

    __slots__ = ("_batch", "_row", "_ctx", "_bt", "_write", "decode_attn_bias")

    def __init__(self, batch: "StaticDecodeBatch", row: int) -> None:
        self._batch = batch
        self._row = row
        # 0-dim view -- attention stacks these into the (B,) RoPE position vector.
        self._ctx = batch.static_ctx[row]
        self._bt = batch.static_bt[row]                          # (n_blocks,)
        self._write = batch.static_write[row:row + 1]            # (1,) scatter index
        self.decode_attn_bias = batch.static_bias[row:row + 1]   # (1, L_pad)

    # -- PagedRequestCache interface (the decode subset) ---------------------

    def seq_len_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """Pre-append context length, as a device scalar, for per-row RoPE.

        Every layer of a request shares one context length (the scheduler
        advances them in lockstep), so one buffer serves all layers -- and the
        graph reads the same address for every layer, which is what we need.
        """
        return self._ctx

    def seq_len(self, layer_idx: int = 0) -> int:
        raise RuntimeError(
            "_StaticRowCache has no host-side seq_len: the context length lives "
            "on the device in static_ctx so a captured graph can read it. "
            "Nothing on the batched-decode path should need the Python int."
        )

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Scatter this row's new K/V into the pool at an ON-DEVICE slot index.

        `PagedRequestCache.append` resolves `block_table[pos // bs]` and
        `pos % bs` in Python and writes a host-indexed slice. Both would be
        frozen at capture, so every replay would rewrite the capture-time slot.
        Here the destination is `static_write`, whose contents `stage()` refills
        from the live block table before each replay.
        """
        b = self._batch
        b.k_flat[layer_idx].index_copy_(
            0, self._write, k_new.reshape(1, b.num_kv_heads, b.head_dim))
        b.v_flat[layer_idx].index_copy_(
            0, self._write, v_new.reshape(1, b.num_kv_heads, b.head_dim))

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the FULL padded block table -- a constant shape.

        `PagedRequestCache.get` trims to `[:seq_len]`, which makes the attention
        shape host-dependent and therefore uncapturable. We return
        `(1, n_blocks * block_size, NKV, D)` every time and let
        `decode_attn_bias` mask the tail.
        """
        b = self._batch
        k = b.pool.K_pool[layer_idx, self._bt].reshape(-1, b.num_kv_heads, b.head_dim)
        v = b.pool.V_pool[layer_idx, self._bt].reshape(-1, b.num_kv_heads, b.head_dim)
        return k.unsqueeze(0), v.unsqueeze(0)


class StaticDecodeBatch:
    """Static-shape decode state for one (batch_size, n_blocks) pair.

    Owns every buffer a captured graph reads, plus the host-side staging that
    refills them from live `PagedRequestCache` objects. Usable WITHOUT CUDA
    graphs (and without CUDA): `run_eager` drives the same static-shape forward
    through the model, which is how the CPU parity test checks the masking,
    gather and scatter logic on a machine with no GPU.
    """

    def __init__(self, pool: "PagedKVCache", batch_size: int, n_blocks: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        if n_blocks > pool.num_blocks:
            raise ValueError(
                f"n_blocks={n_blocks} exceeds the pool's {pool.num_blocks} blocks")
        self.pool = pool
        self.batch_size = batch_size
        self.n_blocks = n_blocks
        self.block_size = pool.block_size
        self.num_kv_heads = pool.num_kv_heads
        self.head_dim = pool.head_dim
        self.num_layers = pool.num_layers
        self.padded_len = n_blocks * pool.block_size

        device = pool.K_pool.device
        self.device = device

        # Flattened (block, slot) -> position views of the pool, so `append` can
        # address a token by a single on-device index. `.view` (not `.reshape`)
        # so this is asserted to be a genuine alias of the pool rather than a
        # silent copy that would drop the writes.
        flat = (pool.num_layers, pool.num_blocks * pool.block_size,
                pool.num_kv_heads, pool.head_dim)
        self.k_flat = pool.K_pool.view(flat)
        self.v_flat = pool.V_pool.view(flat)

        self.static_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.static_ctx = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.static_bt = torch.zeros((batch_size, n_blocks), dtype=torch.long, device=device)
        self.static_write = torch.zeros((batch_size,), dtype=torch.long, device=device)
        # fp32: the FA2 wrapper promotes every input to fp32 anyway, and an fp16
        # -inf would survive the cast but buys nothing.
        self.static_bias = torch.zeros((batch_size, self.padded_len),
                                       dtype=torch.float32, device=device)
        self._arange = torch.arange(self.padded_len, device=device).unsqueeze(0)

        # Host staging. Pinned on CUDA so the per-step refill is a single async
        # H2D per buffer rather than three pageable copies that each sync.
        pin = device.type == "cuda"
        self._h_ctx = torch.zeros((batch_size,), dtype=torch.long, pin_memory=pin)
        self._h_bt = torch.zeros((batch_size, n_blocks), dtype=torch.long, pin_memory=pin)
        self._h_write = torch.zeros((batch_size,), dtype=torch.long, pin_memory=pin)

        self.rows: list[_StaticRowCache] = [
            _StaticRowCache(self, i) for i in range(batch_size)
        ]

    # ---- staging -----------------------------------------------------------

    def blocks_needed(self, ctx: int) -> int:
        """Blocks required to hold logical positions [0, ctx] inclusive."""
        return (ctx + 1 + self.block_size - 1) // self.block_size

    def stage(self, input_ids: torch.Tensor, caches: list["PagedRequestCache"]) -> None:
        """Refill every static buffer from the live decode batch.

        Runs on the host OUTSIDE any captured region. Also performs the block
        allocation that `PagedRequestCache.append` would otherwise do lazily
        mid-forward -- a graph cannot grow a block table, so the table must be
        long enough BEFORE the replay. The allocation is exactly the one the
        eager path would have made on this step.
        """
        if len(caches) != self.batch_size:
            raise ValueError(
                f"staged {len(caches)} caches into a batch_size={self.batch_size} "
                f"graph state")
        if tuple(input_ids.shape) != (self.batch_size, 1):
            raise ValueError(
                f"input_ids must be ({self.batch_size}, 1), got {tuple(input_ids.shape)}")

        bs = self.block_size
        bt_rows: list[list[int]] = []
        for i, c in enumerate(caches):
            ctx = c._seq_lens[0]  # noqa: SLF001
            need = self.blocks_needed(ctx)
            if need > self.n_blocks:
                raise ValueError(
                    f"row {i} needs {need} blocks but this graph state holds "
                    f"{self.n_blocks}; the caller must pick a larger bucket or "
                    f"fall back to eager")
            table = self.pool.get_block_table(c.request_id)
            while len(table) < need:
                self.pool.allocate_block(c.request_id)
                table = self.pool.get_block_table(c.request_id)
            self._h_ctx[i] = ctx
            self._h_write[i] = table[ctx // bs] * bs + (ctx % bs)
            # Pad the tail of the block table with the row's own last live
            # block. Those columns are masked to -inf, so their contents are
            # irrelevant -- but they must be a VALID physical index, because the
            # gather reads them unconditionally.
            bt_rows.append(list(table[:need])
                           + [table[need - 1]] * (self.n_blocks - need))

        self._h_bt.copy_(torch.tensor(bt_rows, dtype=torch.long))
        nb = self.device.type == "cuda"
        self.static_ctx.copy_(self._h_ctx, non_blocking=nb)
        self.static_write.copy_(self._h_write, non_blocking=nb)
        self.static_bt.copy_(self._h_bt, non_blocking=nb)
        self.static_input.copy_(input_ids)
        self._refresh_bias()

    def stage_placeholder(self) -> None:
        """Neutral contents for warmup + capture: block 0, slot 0, context 0.

        Only legal on an empty pool (enforced by `LiveDecodeGraphRunner`): the
        capture pass really does run the decode step and really does write K/V
        to the slot `static_write` names.
        """
        self._h_ctx.zero_()
        self._h_bt.zero_()
        self._h_write.zero_()
        self.static_ctx.zero_()
        self.static_bt.zero_()
        self.static_write.zero_()
        self.static_input.zero_()
        self._refresh_bias()

    def _refresh_bias(self) -> None:
        """Additive mask: 0.0 for positions <= ctx, -inf beyond.

        Built on-device from `static_ctx` so no per-step host-side mask
        construction leaks into the step time this experiment measures.
        """
        valid = self._arange <= self.static_ctx.unsqueeze(1)
        self.static_bias.fill_(NEG_INF)
        self.static_bias.masked_fill_(valid, 0.0)

    def commit(self, caches: list["PagedRequestCache"]) -> None:
        """Advance the live caches' host bookkeeping by the one token the graph
        just wrote. The graph performed the device-side K/V write; only the
        Python `_seq_lens` are left, and they must move exactly as
        `PagedRequestCache.append` would have moved them.
        """
        for c in caches:
            sl = c._seq_lens  # noqa: SLF001
            for layer_idx in range(len(sl)):
                sl[layer_idx] += 1

    # ---- eager driver (capture warmup, and the CPU parity test) ------------

    def run_eager(self, model: "LlamaModel") -> torch.Tensor:
        with torch.no_grad():
            return model(self.static_input, kv_cache=self.rows)


class DecodeGraphRouter:
    """Decides, for one live decode batch, WHICH captured graph can serve it.

    Split out of `LiveDecodeGraphRunner` deliberately. Routing is where the
    scientific hazard lives -- this is the code that must never round a decode
    batch up into a shape it did not capture and call the result a graph hit --
    and it is pure host-side bookkeeping with no CUDA in it. Keeping it separate
    means the routing that runs on the T4 is the same routing a CPU-only host
    can execute and test end to end through a real scheduler.

    Subclasses supply the execution: `LiveDecodeGraphRunner` replays a captured
    CUDA graph. Nothing here ever reports a hit; a hit is only what the caller
    observes when `replay` actually runs.
    """

    def __init__(
        self,
        pool: "PagedKVCache",
        *,
        max_batch_size: int,
        max_context_tokens: int,
        num_seq_buckets: int = 4,
        max_graph_batch: int = 32,
    ) -> None:
        self.pool = pool
        self.max_context_tokens = int(max_context_tokens)

        n_max = max(1, min(int(max_batch_size), int(max_graph_batch)))
        self.batch_sizes: tuple[int, ...] = tuple(range(1, n_max + 1))

        max_blocks = max(
            1, (self.max_context_tokens + pool.block_size - 1) // pool.block_size)
        step = max(1, (max_blocks + num_seq_buckets - 1) // num_seq_buckets)
        self.buckets: tuple[int, ...] = tuple(sorted({
            min(max_blocks, k) for k in range(step, max_blocks + step, step)
        }))

        # key -> whatever the subclass needs to execute that shape.
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.capture_failures: list[dict] = []
        self.fallback_reasons: dict[str, int] = {}
        self._capture_seconds: float = 0.0

    # ---- routing -----------------------------------------------------------

    def _note(self, reason: str) -> None:
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def _bucket_for(self, batch_size: int, caches: list, *, note: bool = True):
        """The block bucket that fits this decode batch, or None.

        Split out of `_resolve` so on-demand capture can ask "which bucket would
        this batch use?" WITHOUT recording a fallback -- a capture that is about
        to succeed must not leave a refusal in `fallback_reasons`, or the
        diagnostics stop summing to one entry per decode forward.
        """
        def _refuse(reason: str):
            if note:
                self._note(reason)
            return None

        if batch_size not in self.batch_sizes:
            return _refuse("batch_size_not_captured")
        bs = self.pool.block_size
        need = 1
        for c in caches:
            if getattr(c, "pool", None) is not self.pool:
                return _refuse("foreign_kv_pool")
            seq_lens = getattr(c, "_seq_lens", None)
            if not seq_lens:
                return _refuse("no_seq_lens")
            lo, hi = min(seq_lens), max(seq_lens)
            if lo != hi:
                # Mid-forward state (some layers appended, some not). A graph
                # writes every layer at one context, so eager is the only
                # correct option here.
                return _refuse("ragged_layer_seq_lens")
            need = max(need, (hi + 1 + bs - 1) // bs)
        for nb in self.buckets:
            if nb >= need:
                return nb
        return _refuse("context_exceeds_largest_bucket")

    def _lookup(self, batch_size: int, n_blocks: int, plan):
        """The key into `self.graphs` for this shape AND plan, or None.

        Two key layouts are supported on purpose:

          * ROUTED (what `LiveDecodeGraphRunner` writes): a 3-tuple
            `(batch_size, n_blocks, plan)`, so a plan mismatch is simply a
            missing key and no graph captured under another plan is even
            reachable by lookup.
          * LEGACY: a 2-tuple `(batch_size, n_blocks)` whose plan lives in the
            optional `graph_plans` side table. CPU test doubles that only
            populate `self.graphs` use this, and the absent-table case degrades
            to "captured under None", which is what an unrouted double means.

        `plan` must already be canonical.
        """
        k3 = (batch_size, n_blocks, plan)
        if k3 in self.graphs:
            return k3
        k2 = (batch_size, n_blocks)
        if k2 in self.graphs:
            captured = getattr(self, "graph_plans", {}).get(k2, None)
            return k2 if normalize_routing_plan(captured) == plan else None
        return None

    def _resolve(self, batch_size: int, caches: list, routing_plan=None):
        """The graph key for this decode batch, or None.

        None means "run eager"; the reason is recorded. Returning a key is a
        promise that `replay` will succeed for this exact batch -- INCLUDING
        that the graph behind it was captured under this exact routing plan.
        """
        nb = self._bucket_for(batch_size, caches)
        if nb is None:
            return None
        plan = normalize_routing_plan(routing_plan)
        key = self._lookup(batch_size, nb, plan)
        if key is not None:
            return key
        # Nothing serves this batch. Distinguish "this shape was never
        # captured" from "this shape exists, but only under other routing
        # plans" -- they have completely different fixes, and conflating them
        # is what made the original zero-hit runs unreadable.
        shape_exists = any(
            k[0] == batch_size and k[1] == nb for k in self.graphs)
        self._note("lora_routing_mismatch" if shape_exists
                   else "bucket_capture_failed")
        return None

    def can_replay(self, batch_size: int, caches: "list | None" = None, routing_plan=None) -> bool:
        """True iff a captured graph can serve exactly this decode batch."""
        if caches is None:
            return any(k[0] == batch_size for k in self.graphs)
        return self._resolve(batch_size, caches, routing_plan) is not None

    # ---- reporting ---------------------------------------------------------

    @staticmethod
    def _plan_str(plan) -> str:
        """A routing plan as a stable, JSON-safe, sortable string.

        Graph keys mix `None`, `str` and `tuple` plans, which are not mutually
        orderable in Python 3, so neither `sorted(self.graphs)` nor a JSON dump
        of the raw keys works. Every artifact consumer reads this instead.
        """
        if plan is None:
            return "base"
        if isinstance(plan, str):
            return plan
        return "[" + ",".join("base" if p is None else str(p) for p in plan) + "]"

    def _key_str(self, key) -> list:
        """One graph key as `[batch_size, n_blocks, plan_str]`."""
        plan = key[2] if len(key) > 2 else getattr(self, "graph_plans", {}).get(key)
        return [key[0], key[1], self._plan_str(normalize_routing_plan(plan))]

    def report(self) -> dict:
        keys = [self._key_str(k) for k in self.graphs]
        keys.sort(key=lambda k: (k[0], k[1], k[2]))
        plans = sorted({k[2] for k in keys})
        return {
            "graphs_captured": len(self.graphs),
            "graph_keys": keys,
            "routing_plans_captured": plans,
            "graphs_per_routing_plan": {
                p: sum(1 for k in keys if k[2] == p) for p in plans},
            "batch_sizes": list(self.batch_sizes),
            "block_buckets": list(self.buckets),
            "block_size": self.pool.block_size,
            "max_context_tokens": self.max_context_tokens,
            "capture_seconds": self._capture_seconds,
            "capture_failures": list(self.capture_failures),
            # On-demand routed capture; absent on the base router and on CPU
            # doubles, which never capture anything.
            "routed_plans_captured": [
                self._plan_str(p) for p in getattr(self, "routed_plans", [])],
            "max_routed_graphs": getattr(self, "max_routed_graphs", None),
            "routed_capture_failures": list(
                getattr(self, "routed_capture_failures", [])),
            # Promoted to the top of the report on purpose. A routed capture
            # that throws every step produces a zero hit rate that is
            # indistinguishable, in the hit count alone, from the routing guard
            # doing its job -- so the first exception is carried where a reader
            # cannot miss it rather than left at the bottom of a failure list.
            "first_routed_capture_error": (
                getattr(self, "routed_capture_failures", [None])[0]["error"]
                if getattr(self, "routed_capture_failures", None) else None),
        }


class LiveDecodeGraphRunner(DecodeGraphRouter):
    """Captures and replays real decode steps for a live scheduler.

    Implements the `can_replay(batch_size, caches, routing_plan)` /
    `replay(input_ids, caches, routing_plan)` protocol
    `ContinuousBatchScheduler._decode_forward` already calls, so attaching one
    to `scheduler._graph_runner` is the whole integration.

    The routing plan is threaded through both calls rather than re-derived from
    the model here. Re-deriving would reintroduce exactly the gap this guard
    exists to close: the scheduler sets the batch's routing, reads the plan, and
    then calls in, and only a plan carried unchanged across that boundary is
    guaranteed to describe the batch that is about to run.
    """

    WARMUP_ITERS = 3

    def __init__(
        self,
        model: "LlamaModel",
        pool: "PagedKVCache",
        *,
        max_batch_size: int,
        max_context_tokens: int,
        num_seq_buckets: int = 4,
        max_graph_batch: int = 32,
        max_routed_graphs: int = 16,
    ) -> None:
        device = pool.K_pool.device
        if device.type != "cuda":
            raise RuntimeError(
                "LiveDecodeGraphRunner requires a CUDA pool; CUDA graphs are a "
                "GPU feature. Use StaticDecodeBatch directly to exercise the "
                "static-shape path on CPU.")
        if next(model.parameters()).device != device:
            raise RuntimeError("model and KV pool must be on the same device")
        if pool._blocks:  # noqa: SLF001
            raise RuntimeError(
                "capture must happen before any request is admitted: the warmup "
                "and capture passes write scratch K/V into the pool and would "
                "corrupt live requests")

        super().__init__(pool, max_batch_size=max_batch_size,
                         max_context_tokens=max_context_tokens,
                         num_seq_buckets=num_seq_buckets,
                         max_graph_batch=max_graph_batch)
        self.model = model
        self.device = device
        self._mempool = None
        self.max_routed_graphs = int(max_routed_graphs)
        # Plans captured on demand, in first-seen order. `None` is captured by
        # capture_all() and is not counted against the routed budget.
        self.routed_plans: list = []
        self.routed_capture_failures: list[dict] = []

    # ---- capture -----------------------------------------------------------

    def capture_all(self, *, require_all: bool = True, routing_plan=_UNSET) -> dict:
        """Capture the full (batch size x context bucket) grid for ONE plan.

        `routing_plan` defaults to THE MODEL'S CURRENT PLAN, read here rather
        than assumed to be `None`. Called at the documented point -- before any
        request is admitted -- that is base, because nothing is routed yet. But
        a caller may route the model first and capture under that plan, and a
        default of `None` would then record "base" for graphs that actually
        froze an adapter's arithmetic: precisely the mislabelling this key is
        supposed to prevent, reintroduced at the capture site. What is recorded
        must be what was executed, so it is derived from the same source the
        scheduler reads.

        The plan is canonicalised and recorded as part of every key this call
        writes, so a later batch routed differently cannot reach these graphs.

        FAILS LOUDLY BY DEFAULT. The previous version of this method recorded
        every capture exception and carried on, which meant a grid where EVERY
        capture failed degraded silently to eager execution and produced a
        "CUDA graph" arm with zero hits -- the exact failure mode this whole
        module exists to eliminate, reintroduced one level down. A capture
        failure is a hard error unless the caller explicitly opts out with
        `require_all=False`, and even then the uncaptured pairs become counted
        eager fallbacks that drag the hit rate below the validity threshold.

        Nothing is ever labelled a graph hit because a capture was attempted.
        """
        from src.engine.lora_model import model_routing_plan
        if routing_plan is _UNSET:
            routing_plan = model_routing_plan(self.model)
        plan = normalize_routing_plan(routing_plan)
        t0 = time.perf_counter()
        for b in self.batch_sizes:
            for nb in self.buckets:
                try:
                    self._capture(b, nb, plan)
                except Exception as exc:                      # noqa: BLE001
                    self.capture_failures.append(
                        {"batch_size": b, "n_blocks": nb,
                         "routing_plan": self._plan_str(plan),
                         "error": f"{type(exc).__name__}: {exc}"})
                    if require_all:
                        raise GraphCaptureError(
                            f"CUDA graph capture failed for batch_size={b}, "
                            f"n_blocks={nb}, routing_plan="
                            f"{self._plan_str(plan)} after {len(self.graphs)} successful "
                            f"captures: {type(exc).__name__}: {exc}\n"
                            f"Refusing to continue: an uncaptured shape becomes "
                            f"an eager decode step, and a graph arm assembled "
                            f"from eager steps is not CUDA-graph evidence. "
                            f"Lower the grid (num_seq_buckets / max_graph_batch) "
                            f"if this is memory pressure."
                        ) from exc
        self._capture_seconds = time.perf_counter() - t0
        if not self.graphs:
            raise GraphCaptureError(
                "no CUDA graphs were captured; the graph arm cannot run")
        return self.report()

    def self_test(self, *, tolerance: float = 0.0) -> dict:
        """Prove -- before any measurement -- that a captured graph really
        replays against LIVE request caches and reproduces eager decode.

        Two T4 smokes reported zero graph hits, and in neither case was there
        anything between "capture returned without raising" and the measured
        run that would have caught it. This closes that gap: it builds throwaway
        requests on the real pool, prefills them, routes one decode step through
        `can_replay`/`replay`, compares against eager, and frees everything.

        Raises `GraphSelfTestError` on any failure -- including a `can_replay`
        refusal, which is reported WITH the runner's reason, so a failure names
        its own cause instead of surfacing as an unexplained zero hit rate
        thirty minutes later.
        """
        results = []
        for batch_size in sorted({1, min(2, self.batch_sizes[-1]),
                                  self.batch_sizes[-1]}):
            results.append(self._self_test_one(batch_size, tolerance))
        return {"checked_batch_sizes": [r["batch_size"] for r in results],
                "max_abs_logit_delta": max(r["max_abs_delta"] for r in results),
                "per_batch": results}

    def _self_test_one(self, batch_size: int, tolerance: float) -> dict:
        from src.engine.kv_cache import PagedRequestCache
        from src.engine.lora_model import model_routing_plan

        pool = self.pool
        cfg = self.model.config
        # A short prompt: enough to exercise prefill + block-table growth, small
        # enough to land in the smallest bucket.
        prompt_len = max(1, min(pool.block_size, self.max_context_tokens // 2))
        blocks = (prompt_len + 2 + pool.block_size - 1) // pool.block_size + 1
        rids = [f"__graph_selftest_{batch_size}_{i}__" for i in range(batch_size)]
        caches = []
        try:
            for rid in rids:
                if not pool.can_admit(blocks):
                    raise GraphSelfTestError(
                        f"pool cannot admit the self-test request ({blocks} "
                        f"blocks); the KV pool is too small for this grid")
                pool.admit_request(request_id=rid, prefill_blocks_needed=blocks,
                                   total_blocks_needed=blocks)
                c = PagedRequestCache(pool, rid, num_layers=pool.num_layers)
                ids = torch.randint(0, cfg.vocab_size, (1, prompt_len),
                                    device=self.device)
                with torch.no_grad():
                    self.model(ids, kv_cache=c)
                caches.append(c)

            step_ids = torch.randint(0, cfg.vocab_size, (batch_size, 1),
                                     device=self.device)
            seq_lens_before = [list(c._seq_lens) for c in caches]  # noqa: SLF001
            k_before = pool.K_pool.clone()
            v_before = pool.V_pool.clone()

            # Resolve with the model's live plan, exactly as the scheduler
            # does -- a self-test that used a different plan than production
            # would prove nothing about production.
            plan = model_routing_plan(self.model)
            if not self.can_replay(batch_size, caches, plan):
                raise GraphSelfTestError(
                    f"can_replay() refused a live decode batch of size "
                    f"{batch_size} at context {seq_lens_before[0][0]}. Runner "
                    f"reasons so far: {dict(self.fallback_reasons)}. Captured "
                    f"keys: {sorted(self.graphs)}")

            with torch.no_grad():
                eager = self.model(step_ids, kv_cache=caches)

            # Rewind to the pre-eager state and replay the same step.
            pool.K_pool.copy_(k_before)
            pool.V_pool.copy_(v_before)
            for c, sl in zip(caches, seq_lens_before):
                c._seq_lens[:] = list(sl)          # noqa: SLF001
                c._bt_tensor = None                # noqa: SLF001
                c._seqlen_tensors.clear()          # noqa: SLF001
                c._seqlen_vals.clear()             # noqa: SLF001

            graphed = self.replay(step_ids, caches, plan)

            delta = (graphed.float() - eager.float()).abs().max().item()
            if delta > tolerance:
                raise GraphSelfTestError(
                    f"graph replay disagreed with eager decode at batch_size="
                    f"{batch_size}: max|delta| = {delta:.3e} > {tolerance:.3e}. "
                    f"The graph arm would be measuring a different computation "
                    f"than the eager arm.")
            if not torch.equal(graphed[:, -1].argmax(-1), eager[:, -1].argmax(-1)):
                raise GraphSelfTestError(
                    f"graph replay emitted a different token than eager decode "
                    f"at batch_size={batch_size}")
            return {"batch_size": batch_size, "context": seq_lens_before[0][0],
                    "max_abs_delta": delta}
        finally:
            for rid in rids:
                pool.free_request(rid)

    def _capture(self, batch_size: int, n_blocks: int, plan=None,
                 caches: "list | None" = None) -> None:
        """Capture one graph for (batch_size, n_blocks) under `plan`.

        `caches` is the live decode batch when this is an on-demand capture, and
        None during `capture_all()`. With a live batch we stage ITS metadata, so
        the warmup and capture writes land on the slot this step is about to
        overwrite anyway, rather than on block 0 and whichever request owns it.
        The staged token ids are zeros; see the module docstring for why that is
        safe (the caller always rewrites the slot before it is read).

        The caller is responsible for having the model routed to `plan` already:
        capture freezes whatever branch `LoRALinear.forward` takes, so the plan
        recorded in the key is only honest if the model was actually routed that
        way while the capture ran.
        """
        state = StaticDecodeBatch(self.pool, batch_size, n_blocks)
        if caches is None:
            state.stage_placeholder()
        else:
            state.stage(input_ids=torch.zeros((batch_size, 1), dtype=torch.long,
                                              device=self.device),
                        caches=caches)

        # Warmup on a side stream is required by the capture protocol, and the
        # FIRST CUDA decode forward also JIT-compiles/autotunes the Triton RoPE
        # and FA2 kernels -- that must not happen mid-capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.WARMUP_ITERS):
                state.run_eager(self.model)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        kwargs = {} if self._mempool is None else {"pool": self._mempool}
        with torch.no_grad(), torch.cuda.graph(graph, **kwargs):
            static_logits = self.model(state.static_input, kv_cache=state.rows)
        if self._mempool is None:
            self._mempool = graph.pool()

        # B2: the plan is part of the KEY, not a side table. A batch routed
        # differently cannot reach this graph by lookup at all.
        self.graphs[(batch_size, n_blocks, plan)] = (graph, state, static_logits)

    # ---- replay ------------------------------------------------------------

    def can_replay(self, batch_size: int, caches: "list | None" = None,
                   routing_plan=None) -> bool:
        """True iff a captured graph can serve exactly this decode batch.

        Extends the base router with ON-DEMAND capture: when the batch is
        servable in every respect except that no graph exists yet for its
        routing plan, capture one now, from this batch, and serve it. See the
        module docstring for why capturing against the live batch is safe here
        while `capture_all()` requires an empty pool.
        """
        if caches is None:
            return any(k[0] == batch_size for k in self.graphs)
        plan = normalize_routing_plan(routing_plan)
        # Probe WITHOUT recording a reason: if the capture below succeeds this
        # step is a hit, and a hit must not also leave a fallback behind.
        nb = self._bucket_for(batch_size, caches, note=False)
        if nb is not None and self._lookup(batch_size, nb, plan) is None:
            if not self._capture_on_demand(batch_size, nb, plan, caches):
                # Capture was attempted for this exact key and did not produce a
                # graph. The specific reason is already recorded, so return here
                # rather than falling through to `_resolve`.
                #
                # Falling through was a real defect, and an expensive one. It
                # added a SECOND reason for one decode forward -- breaking this
                # module's "exactly one reason per batched-decode forward"
                # contract -- and the reason it added was `lora_routing_mismatch`,
                # which means "a graph exists for this shape under a DIFFERENT
                # plan and we refused to reuse it". That is the guard working. A
                # CUDA capture that threw is not the guard working, and labelling
                # it that way sent a T4 investigation after a routing bug that
                # did not exist while the actual exception sat unread in
                # `routed_capture_failures`.
                return False
        return self._resolve(batch_size, caches, routing_plan) is not None

    def _capture_on_demand(self, batch_size: int, n_blocks: int, plan,
                           caches: list) -> bool:
        """Capture the graph this decode batch needs, or leave it uncaptured.

        Returns True when the caller should go on to resolve normally, and
        False when this call has already recorded the reason this step cannot
        be a graph hit. That distinction is what keeps exactly one reason per
        batched-decode forward.

        Never raises: an on-demand capture happens inside a live serving loop,
        and a failure there must degrade to an eager step, not take the server
        down. But it is never silent either -- the exception text is kept
        verbatim in `routed_capture_failures` and surfaced through `report()`,
        because "the graph arm is slow" and "every routed capture is throwing"
        look identical in a hit rate alone.
        """
        if plan is None:
            # The unrouted grid is capture_all()'s job. Reaching here means that
            # shape genuinely failed to capture; let `_resolve` say so.
            return True
        if plan not in self.routed_plans:
            if len(self.routed_plans) >= self.max_routed_graphs:
                self._note("routed_graph_budget_exhausted")
                return False
            self.routed_plans.append(plan)
        try:
            torch.cuda.synchronize()
            self._capture(batch_size, n_blocks, plan, caches=caches)
            return True
        except Exception as exc:                              # noqa: BLE001
            self.routed_capture_failures.append(
                {"batch_size": batch_size, "n_blocks": n_blocks,
                 "routing_plan": self._plan_str(plan),
                 "error": f"{type(exc).__name__}: {exc}"})
            self._note("routed_capture_failed")
            return False

    def replay(self, input_ids: torch.Tensor, caches: list,
               routing_plan=None) -> torch.Tensor:
        """Serve one decode step from a captured graph. Returns (B, 1, vocab).

        B3: `routing_plan` is the plan the caller has already established for
        this batch, and it is resolved with -- not recomputed. Resolving without
        it would look up a DIFFERENT key than the `can_replay` that authorised
        this call, which is how a routed batch could be served by the unrouted
        graph.
        """
        key = self._resolve(len(caches), caches, routing_plan)
        if key is None:
            raise RuntimeError(
                f"replay() called for a decode batch no captured graph can "
                f"serve (batch_size={len(caches)}, routing_plan="
                f"{self._plan_str(normalize_routing_plan(routing_plan))}); "
                f"the caller must consult can_replay() first")
        graph, state, static_logits = self.graphs[key]
        state.stage(input_ids, caches)
        graph.replay()
        # Clone before returning: static_logits lives in the shared graph memory
        # pool and is only valid until the next replay.
        out = static_logits.clone()
        state.commit(caches)
        return out
