"""
Speculative decoding via self-speculation with early-exit drafting.

What problem this solves:

  Standard greedy decode runs the full 1.1B model once per emitted token.
  On bandwidth-bound hardware (anything sub-A100 with a 1B-class model),
  most of that step is loading the 2.2 GB of fp32 weights through cache.
  Actual arithmetic on a single token is negligible. The GPU spends most
  of every step idle on memory traffic.

  Speculative decoding (Leviathan et al. 2023, "Fast Inference from
  Transformers via Speculative Decoding") exploits this. A cheap "draft"
  model proposes K candidate tokens. The base model then runs ONCE on
  all K+1 positions in parallel -- same memory-bandwidth cost as a single
  decode step, but it returns logits for K+1 positions. We accept draft
  tokens as long as the base model's greedy choice agrees with them; on
  the first disagreement we emit the base model's pick.

  Worst case: 0 drafts accepted, cost = K cheap drafts + 1 base forward,
  output = 1 token (base's pick at position 0). Slower than vanilla.
  Best case: K accepted, cost = K cheap drafts + 1 base forward, output
  = K+1 tokens. ~Kx speedup minus draft overhead.

Why "self-speculation with early-exit" is the v0.3 draft strategy:

  A dedicated draft model would be ideal (vLLM, TensorRT-LLM, EAGLE all
  do this). But it needs separately trained weights matching TinyLlama's
  tokenizer, which we don't have and can't easily produce. The trade-off:

    (a) Train a smaller LlamaModel  -- needs a training pipeline. No.
    (b) Bolt on EAGLE / Medusa head -- needs trained adapter weights. No.
    (c) Self-speculation: take the SAME model but stop the forward pass
        early (here, after layer 8 of 22) and feed that partial residual
        stream through the existing final RMSNorm + lm_head.

  Option (c) gives us a working speculative decoder with no new weights.
  The cost is acceptance rate: layer-8 predictions agree with layer-22
  predictions only ~40-60% of the time (vs ~70-80% for a properly trained
  draft head). The expected decode speedup on TinyLlama + RTX 4060 is
  1.3-1.7x, not the 2-3x of a real draft model. We document this as a
  v0.3 trade-off; a future v0.4 could add an EAGLE-style trained head.

Why this still produces BYTE-IDENTICAL output under greedy:

  Greedy speculative is exact. Argmax is deterministic, so:
    * If draft token d_i matches base's argmax at position i, then a
      non-speculative greedy run that had been at that context would
      have emitted the same token.
    * On first mismatch, we emit base's argmax -- exactly the token
      non-speculative greedy would have produced.
  The output sequence is identical to non-speculative greedy. Only the
  wall-clock time changes. The parity test enforces this byte-for-byte.

  (Speculative decoding with temperature sampling requires a more careful
  rejection-sampling acceptance step to preserve the base distribution;
  we don't sample, so we don't need that machinery.)

Scope cuts for v0.3 (documented honestly):

  * Single-request spec decode only. Batched speculative decoding needs
    per-request K/V truncation in the middle of a batched forward, which
    is non-trivial layout work. v0.3 falls back to vanilla batched decode
    whenever 2+ requests are in DECODE simultaneously.
  * Fixed K (default 4). Adaptive K based on recent acceptance rate is a
    well-known improvement; we keep it simple here.
  * No KV truncation amortization across spec rounds: each round
    pessimistically allocates blocks for K+1 positions, and frees any
    excess on partial rejection. Correct but a touch wasteful.

Two implementations live in this file:

  A. The v0.3 GREEDY self-speculation path (the function surface below).
     This is what the ContinuousBatchScheduler drives. It is byte-exact
     under greedy decoding -- argmax agreement, no rejection sampling.

  B. The TRUE draft/target speculative decoder (the class surface below),
     added in v0.5. A small DRAFT model proposes K tokens *with their
     probabilities*; the large TARGET model verifies all K in one forward;
     the exact acceptance-rejection rule of Leviathan et al. 2023 (their
     Algorithm 1) accepts a prefix and, on the first rejection, resamples
     from the adjusted residual distribution max(0, p_target - p_draft).
     This preserves the TARGET model's sampling distribution exactly -- not
     just under greedy, but at any temperature -- which is the property
     greedy self-speculation cannot give you.

  The two coexist: A stays the scheduler's path (it needs no second model
  and integrates with the paged cache); B is the general algorithm, wired
  to the server via `speculative_k` and exercised by test_spec_decode.py.

Public surface (A -- greedy self-spec functions):
  early_exit_forward(model, input_ids, kv_cache, n_layers) -> logits
  draft_k_tokens(model, request_cache, last_token_id, k, n_draft_layers) -> list[int]
  verify_full_forward(model, request_cache, last_token_id, draft_tokens) -> logits
  spec_decode_step(model, request_cache, last_token_id, k, eos_token_id, max_emit, n_draft_layers) -> (list[int], int)

Public surface (B -- draft/target acceptance-rejection):
  DraftModel  (Protocol):  propose(input_ids, k, kv_cache) -> (token_ids, draft_probs)
  TargetModel (Protocol):  verify(input_ids, draft_tokens, kv_cache) -> target_probs
  speculative_sample(draft_tokens, draft_probs, target_probs, generator) -> (tokens, n_accepted)
  SpeculativeDecoder(draft_model, target_model, k).decode_step(input_ids, kv_cache) -> tokens
  TinyDraftModel(small_model)         -- a smaller LlamaModel as the draft
  FullModelTarget(model)              -- a LlamaModel as the verifying target
  SelfSpecDraftModel(model, n_layers) -- the early-exit draft, as a DraftModel
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.engine.kv_cache import PagedRequestCache
    from src.engine.model import LlamaModel


# Default early-exit point. TinyLlama has 22 layers; layer 8 (the first 8
# blocks) gives a residual stream that's "deep enough" for the lm_head to
# produce sensible predictions but cheap enough to be a useful draft.
# Roughly 8/22 = 36% of the base-model forward cost per draft token.
DEFAULT_N_DRAFT_LAYERS = 8


# ---------------------------------------------------------------------------
# early_exit_forward
# ---------------------------------------------------------------------------
#
# A surgical reimplementation of LlamaModel.forward that stops after
# `n_layers` blocks instead of running all 22, then applies the existing
# final RMSNorm and lm_head.
#
# Why a separate function instead of a forward(... early_exit=N) kwarg on
# LlamaModel:
#   * Keeps the model module unaware of speculative decoding. The base
#     forward path remains untouched -- no parity-test risk to the
#     existing 20 tests.
#   * Easier to read: the speculative-decoding logic lives in one file.
#
# The function intentionally mirrors LlamaModel.forward line-by-line so a
# future reader can diff them and see exactly what changed (just the loop
# bound).
# ---------------------------------------------------------------------------


@torch.no_grad()
def early_exit_forward(
    model: "LlamaModel",
    input_ids: torch.Tensor,
    kv_cache: "PagedRequestCache | None",
    n_layers: int = DEFAULT_N_DRAFT_LAYERS,
) -> torch.Tensor:
    """Forward pass through the first `n_layers` blocks, then final norm + lm_head.

    Args:
        model: the LlamaModel. We touch only model.embed, model.layers[:n],
            model.final_norm, model.lm_head.
        input_ids: (B, S) int64. In draft loops B=1, S=1.
        kv_cache: PagedRequestCache. Will be MUTATED -- each visited layer
            appends K/V at its current seq_len. Caller is responsible for
            snapshotting seq_lens before draft and rewinding after.
        n_layers: how many transformer blocks to traverse before exiting.
            Must be in [1, len(model.layers)]. The default of 8 (out of 22)
            balances draft fidelity vs draft cost.

    Returns:
        logits: (B, S, vocab_size). Same shape as a full forward.
    """
    if n_layers < 1 or n_layers > len(model.layers):
        raise ValueError(
            f"n_layers={n_layers} out of range [1, {len(model.layers)}]"
        )
    # Device-transparent input handling, matching LlamaModel.forward.
    input_ids = input_ids.to(model.embed.weight.device)
    x = model.embed(input_ids)
    # Walk the first n_layers blocks; the cache append happens inside each
    # block's attention. Crucially we pass the SAME layer_idx the base
    # forward would have used, so K/V land at the right pool slot.
    for i in range(n_layers):
        x = model.layers[i](x, kv_cache=kv_cache, layer_idx=i)
    # Final norm + LM head. These are trained on the layer-22 residual but
    # applied to layer-`n_layers` here. That's the "off-distribution" cost
    # of self-speculation -- the price we pay for not training a new head.
    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits


# ---------------------------------------------------------------------------
# draft_k_tokens
# ---------------------------------------------------------------------------
#
# Greedy autoregressive loop that generates K candidate tokens using the
# early-exit forward. Each step:
#   * feed the current token at the current cache position
#   * early-exit forward through `n_draft_layers` blocks
#   * argmax the last position's logits to get the next candidate
#   * the model's attention auto-appended K/V at layers [0, n_draft_layers)
#     for this position, so the next step's RoPE offset is correct
#
# IMPORTANT: this leaves the cache in a half-mutated state. Layers
# [0, n_draft_layers) have written K/V at K new positions; the deeper
# layers haven't. The caller (spec_decode_step) MUST roll back seq_lens
# before running verify -- otherwise verify's append() would write at
# positions [N+K, N+2K] instead of [N, N+K].
# ---------------------------------------------------------------------------


@torch.no_grad()
def draft_k_tokens(
    model: "LlamaModel",
    request_cache: "PagedRequestCache",
    last_token_id: int,
    k: int,
    n_draft_layers: int = DEFAULT_N_DRAFT_LAYERS,
) -> list[int]:
    """Generate K draft tokens via early-exit greedy decoding.

    Args:
        model: LlamaModel.
        request_cache: per-request paged cache view. WILL BE MUTATED at
            layers [0, n_draft_layers); caller is responsible for rewinding
            seq_lens before any subsequent full forward.
        last_token_id: the most recently emitted token. This is the token
            that would be fed in a vanilla decode step.
        k: number of draft tokens to produce. Typically 4.
        n_draft_layers: how many layers the draft path traverses.

    Returns:
        List of K candidate token ids, in order.
    """
    if k <= 0:
        return []
    device = model.embed.weight.device
    draft_tokens: list[int] = []
    cur = last_token_id
    for _ in range(k):
        # Single-token forward at the current cache position. Shape (1, 1).
        input_ids = torch.tensor([[cur]], dtype=torch.long, device=device)
        logits = early_exit_forward(
            model, input_ids, request_cache, n_layers=n_draft_layers,
        )
        # Greedy pick at the (only) input position. logits shape: (1, 1, V).
        cur = int(torch.argmax(logits[0, -1, :]))
        draft_tokens.append(cur)
    return draft_tokens


# ---------------------------------------------------------------------------
# verify_full_forward
# ---------------------------------------------------------------------------
#
# Run the FULL 22-layer model on [last_token, d_0, d_1, ..., d_{K-1}] -- a
# K+1-token sequence -- in one parallel forward pass. The causal mask
# ensures each position attends only to its prior positions.
#
# After this call, the cache contains FRESH K/V at all 22 layers for the
# K+1 new positions [N, N+1, ..., N+K]. seq_len for every layer has
# advanced by K+1.
#
# Returns logits of shape (1, K+1, V):
#   logits[0]  -> prediction for position N+1   (compare argmax to d_0)
#   logits[1]  -> prediction for position N+2   (compare argmax to d_1)
#   ...
#   logits[K-1]-> prediction for position N+K   (compare argmax to d_{K-1})
#   logits[K]  -> prediction for position N+K+1 (the "bonus" -- the token
#                 we'd emit next if every draft was accepted)
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_full_forward(
    model: "LlamaModel",
    request_cache: "PagedRequestCache",
    last_token_id: int,
    draft_tokens: list[int],
) -> torch.Tensor:
    """Run the full model on the K+1 token candidate window.

    Caller MUST have rewound request_cache._seq_lens so that all layers
    point at the position where `last_token_id` should land (i.e., the
    seq_len BEFORE draft_k_tokens mutated anything).

    Args:
        model: LlamaModel.
        request_cache: per-request cache. WILL BE MUTATED -- K/V for K+1
            new positions get appended at every layer.
        last_token_id: the token to feed at the cache's current head
            position.
        draft_tokens: K candidate tokens proposed by the draft path.

    Returns:
        logits: (1, K+1, vocab_size).
    """
    device = model.embed.weight.device
    # Concatenate [last_token, d_0, ..., d_{K-1}] into one batched input.
    # Shape (1, K+1).
    seq = [last_token_id] + list(draft_tokens)
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    # Full forward. The cache's append writes K/V for all K+1 positions at
    # every layer. RoPE positions are read from each layer's seq_len before
    # appending, exactly matching vanilla decode.
    logits = model(input_ids, kv_cache=request_cache)
    return logits


# ---------------------------------------------------------------------------
# Cache rewind helper
# ---------------------------------------------------------------------------
#
# After draft + verify, we know how many positions to keep. This helper
# reverts the per-layer seq_lens to the target and, if the request now
# holds more physical blocks than necessary, returns the excess to the
# pool so subsequent admissions can use them.
#
# Why we cannot just "leak" the excess block:
#   The block was drawn from the pool's free list AND consumed a slot of
#   the request's reservation (allocate_block decrements _reserved by 1).
#   If we don't put it back, every spec_decode round that partially
#   rejects could permanently siphon capacity. Over many rounds the pool
#   would silently starve.
# ---------------------------------------------------------------------------


def rewind_cache(
    request_cache: "PagedRequestCache",
    target_seq_len: int,
) -> None:
    """Set every layer's seq_len to `target_seq_len` and free any block
    that is now beyond the request's needed footprint.

    Args:
        request_cache: the per-request cache view to roll back.
        target_seq_len: the seq_len we want every layer to hold. Must be
            <= the current seq_len at every layer (this is purely a
            rewind; growing is the cache's normal append-driven path).
    """
    pool = request_cache.pool
    request_id = request_cache.request_id
    # Rewind every layer in lockstep. Note: per-layer seq_lens may differ
    # transiently (e.g. mid-forward during a draft step), but in the
    # spec_decode call sites we always invoke this when all layers are
    # equal -- either after draft (uneven, but we're resetting to the
    # pre-draft value across the board) or after verify (uniform).
    for layer_idx in range(len(request_cache._seq_lens)):
        request_cache._seq_lens[layer_idx] = target_seq_len  # noqa: SLF001
    # Compute how many blocks the request truly needs to cover positions
    # [0, target_seq_len).
    bs = pool.block_size
    needed_blocks = (target_seq_len + bs - 1) // bs
    block_table = pool._blocks[request_id]  # noqa: SLF001
    # Pop excess physical blocks off the tail of the request's table and
    # return them to the pool. Tail pops are correct because cache.append
    # always appends in logical-position order, so the excess is at the
    # end.
    while len(block_table) > needed_blocks:
        excess_phys = block_table.pop()
        pool.ref_count[excess_phys] -= 1
        if pool.ref_count[excess_phys] < 0:
            raise RuntimeError(
                f"rewind_cache: refcount underflow on block {excess_phys}; "
                f"accounting bug"
            )
        if pool.ref_count[excess_phys] == 0:
            del pool.ref_count[excess_phys]
            pool._free_blocks.add(excess_phys)  # noqa: SLF001
            # Defensive hash-mapping cleanup. Decode-time allocations
            # don't register hashes (allocate_block omits them) so this
            # is almost always a no-op; we keep it for symmetry with
            # PagedKVCache.free_request.
            h = pool.block_hashes.pop(excess_phys, None)
            if h is not None and pool.hash_to_block.get(h) == excess_phys:
                del pool.hash_to_block[h]
        # Each freed block restores one unit of the request's reservation
        # so future cache growth has budget. Symmetric with allocate_block's
        # `_reserved[request_id] -= 1`.
        pool._reserved[request_id] += 1  # noqa: SLF001

    # The per-request cached block-table device tensor (`_bt_tensor`, added in
    # 9a5a4c9 for CUDA-graph capture) keys its validity ONLY on len(block_table)
    # and assumes the table grows monotonically with tail entries never moving.
    # The block-free loop above violates that: we pop tail blocks and return
    # them to the pool's free set, so the next allocate_block can hand the same
    # logical tail slot a DIFFERENT physical block at the SAME length -- a stale
    # `_bt_tensor` would then gather K/V from the wrong block and silently
    # corrupt decode. Drop it so the next get() rebuilds, exactly mirroring the
    # H2O-eviction path (kv_eviction.py, after trim_request_blocks).
    request_cache._bt_tensor = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# spec_decode_step (the orchestrator)
# ---------------------------------------------------------------------------
#
# One full speculative-decoding round for ONE request. The contract:
#
#   In:  cache at seq_len = N for every layer (consistent post-step state).
#        last_token_id is the most recent emitted token.
#
#   Out: list of newly emitted tokens (1 to K+1 of them) AND the new
#        seq_len after the round. The cache is left in a consistent state
#        with seq_len equal to N + len(emitted).
#
# Step by step:
#   1. Snapshot N = cache.seq_len(layer 0).
#   2. Draft K tokens via early-exit. Layers [0, n_draft_layers) now have
#      seq_len = N+K; deeper layers still at N.
#   3. Rewind ALL layers to N. (No block-table cleanup here -- verify is
#      about to use those blocks.)
#   4. Verify: full forward on [last_token, d_0, ..., d_{K-1}]. All layers
#      now at seq_len = N+K+1 with fresh K/V written.
#   5. Compute acceptance count m by walking argmax(logits[i]) vs d_i.
#      Build emit list = [d_0, ..., d_{m-1}, argmax(logits[m])], length m+1.
#   6. Truncate emit list against EOS and max_emit. The truncated length
#      is e (1 <= e <= m+1; can be less than m+1 if EOS or budget hits).
#   7. Final rewind: cache to N + e. Free excess blocks back to pool.
#   8. Return (emit_list[:e], m_for_metric).
#
# We return BOTH the (possibly EOS-truncated) emit list and the raw
# acceptance count m, so the scheduler can advance request state cleanly
# AND record the un-truncated acceptance rate for metrics (truncation by
# EOS shouldn't artificially deflate the speedup signal).
# ---------------------------------------------------------------------------


@torch.no_grad()
def spec_decode_step(
    model: "LlamaModel",
    request_cache: "PagedRequestCache",
    last_token_id: int,
    k: int,
    eos_token_id: int | None,
    max_emit: int,
    n_draft_layers: int = DEFAULT_N_DRAFT_LAYERS,
) -> tuple[list[int], int]:
    """One speculative-decoding round for a single request.

    Args:
        model: LlamaModel.
        request_cache: per-request paged cache. WILL BE MUTATED, but left
            in a consistent state on return.
        last_token_id: most recently emitted token for this request.
        k: number of draft tokens. Typically 4.
        eos_token_id: emit list is truncated at (and including) the first
            EOS, if any. None disables EOS truncation.
        max_emit: emit list is truncated to at most this many tokens. The
            caller passes (request.max_new_tokens - len(generated_token_ids))
            so we never overshoot the request's budget.
        n_draft_layers: depth of the early-exit draft forward.

    Returns:
        (emitted_tokens, accepted_count). emitted_tokens has length
        1 <= e <= min(k+1, max_emit), possibly EOS-truncated.
        accepted_count is the raw m before truncation -- the value the
        metrics layer wants for acceptance-rate histograms.
    """
    if max_emit <= 0:
        return [], 0

    # --- 1. Snapshot pre-step seq_len ---------------------------------
    # We assume all layers are at the same value going in (the post-step
    # invariant of vanilla decode AND of any prior spec_decode_step).
    pre_seq_len = request_cache.seq_len(0)

    # --- 2. Draft K tokens (mutates layers [0, n_draft_layers)) -------
    # Cap K against the remaining emission budget so we never produce
    # drafts that would have to be thrown away. If only 2 emissions
    # remain, K=4 is wasteful; clamp to max_emit-1 so verify still has
    # the "+1 bonus" slot. The actual emission count cap below handles
    # the truncation either way.
    effective_k = max(0, min(k, max_emit - 1))
    if effective_k == 0:
        # No room for any draft; this round emits exactly one token from
        # a single vanilla decode forward. Fall back to a plain forward
        # to avoid the verify K+1 ceremony for the K=0 case.
        device = model.embed.weight.device
        input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=device)
        logits = model(input_ids, kv_cache=request_cache)  # advances seq_len by 1
        next_id = int(torch.argmax(logits[0, -1, :]))
        return [next_id], 0

    draft_tokens = draft_k_tokens(
        model, request_cache, last_token_id, effective_k,
        n_draft_layers=n_draft_layers,
    )

    # --- 3. Rewind layers [0, n_draft_layers) so verify writes fresh --
    # The deeper layers are already at pre_seq_len; rewind_cache will set
    # every layer to pre_seq_len uniformly. We do NOT free any blocks
    # here -- the draft may have grown the block table, and verify is
    # about to need those same slots. Block-table cleanup happens AFTER
    # verify when we know the true final seq_len.
    #
    # Note: rewind_cache's block-free logic uses (target_seq_len, bs) to
    # compute needed_blocks. With target = pre_seq_len, needed_blocks is
    # at most the current block_table length (since draft only grew
    # things), so the while-loop bound holds -- but in the partial-K
    # case some blocks may already be poppable. That's still safe: any
    # block popped here that verify needs will be re-allocated by the
    # cache's append. To avoid the redundant pop+realloc, we open-code
    # the seq_len reset without the block-free pass.
    for layer_idx in range(len(request_cache._seq_lens)):  # noqa: SLF001
        request_cache._seq_lens[layer_idx] = pre_seq_len  # noqa: SLF001
    # The draft loop above may have grown the block table (and rebuilt the
    # cached `_bt_tensor`) at a seq_len we've now rewound past. Drop the cached
    # tensor so verify's full forward rebuilds it against the true post-rewind
    # block table -- same rationale as rewind_cache / the eviction path.
    request_cache._bt_tensor = None  # noqa: SLF001

    # --- 4. Verify: full forward on [last_token, d_0, ..., d_{K-1}] ---
    # Writes fresh K/V at all 22 layers for the K+1 new positions.
    logits = verify_full_forward(
        model, request_cache, last_token_id, draft_tokens,
    )  # (1, K+1, V)

    # --- 5. Acceptance loop -------------------------------------------
    # logits[0, i, :] predicts the token at position pre_seq_len + i + 1.
    # That's what d_i should be (for i < K), or the bonus token (for i = K).
    base_preds = torch.argmax(logits[0], dim=-1).tolist()  # length K+1

    accepted = 0
    for i in range(effective_k):
        if base_preds[i] == draft_tokens[i]:
            accepted += 1
        else:
            break
    # m = accepted. The fix-up / bonus token is always base_preds[m].
    fix_up_token = base_preds[accepted]

    # Build the un-truncated emit list: m accepted drafts + 1 base token.
    emit: list[int] = list(draft_tokens[:accepted]) + [fix_up_token]
    # Length is accepted + 1, between 1 (m=0) and effective_k + 1 (all
    # accepted, K+1 tokens total).

    # --- 6. Truncate emit list against EOS and max_emit ---------------
    # Walk the emit list; stop at (and INCLUDE) the first EOS, or at
    # max_emit, whichever comes first. The truncated count is `e`.
    e = 0
    for tok in emit:
        e += 1
        if e >= max_emit:
            break
        if eos_token_id is not None and tok == eos_token_id:
            break
    emit_truncated = emit[:e]

    # --- 7. Final rewind: cache to pre_seq_len + e, free excess blocks
    rewind_cache(request_cache, pre_seq_len + e)

    return emit_truncated, accepted


# ===========================================================================
# PART B: true draft/target speculative decoding (Leviathan et al. 2023).
# ===========================================================================
#
# Everything above is GREEDY self-speculation: one model, early exit, argmax
# agreement. The code below is the GENERAL algorithm with a separate draft
# model and exact acceptance-rejection sampling. It does not touch the paged
# cache or the scheduler; it works on plain (1, S) token tensors so it can be
# unit-tested on CPU with random-weight tiny models and a RandomDraftModel.
#
# The contract between the two model roles:
#
#   draft.propose(context, k)  -> (x_1..x_K, q_1..q_K)
#       The draft autoregressively samples K tokens. q_i is the FULL
#       categorical distribution (over the whole vocab) it sampled x_i from.
#       We need the full q_i, not just q_i(x_i), to build the residual
#       distribution on rejection.
#
#   target.verify(context, x_1..x_K) -> p_1..p_{K+1}
#       The target runs ONCE on [context, x_1, ..., x_K] (K+1 new logit
#       positions thanks to the causal mask) and returns p_i, the target's
#       distribution for the i-th slot. p_{K+1} is the "bonus" distribution
#       used only if every draft token is accepted.
#
# Why this is more than greedy self-spec can do: the accept/resample rule
# below makes the emitted tokens distributed EXACTLY as if sampled directly
# from the target at the chosen temperature (Leviathan Theorem 1). Greedy
# self-spec only matches greedy (temperature 0) decoding.
# ---------------------------------------------------------------------------


@runtime_checkable
class DraftModel(Protocol):
    """A cheap model that proposes K candidate tokens with their probabilities.

    Implementations: TinyDraftModel (a smaller LlamaModel), SelfSpecDraftModel
    (early-exit of the target), and RandomDraftModel (draft_model.py, tests).
    """

    def propose(
        self, input_ids: torch.Tensor, k: int, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given context `input_ids` (1, S), propose `k` next tokens.

        Returns:
            token_ids:   (K,) int64 -- the proposed tokens, in order.
            draft_probs: (K, V) float -- row i is the categorical q_i over the
                whole vocabulary that token_ids[i] was sampled from.
        """
        ...


@runtime_checkable
class TargetModel(Protocol):
    """The large model that verifies a draft in a single forward pass."""

    def verify(
        self, input_ids: torch.Tensor, draft_tokens: torch.Tensor, kv_cache=None
    ) -> torch.Tensor:
        """Run the target on [input_ids, draft_tokens] and return the per-slot
        target distributions.

        Returns:
            target_probs: (K+1, V) float -- row i is p_i, the target's
                distribution for slot i. Rows 0..K-1 line up with the K draft
                tokens; row K is the bonus distribution (used iff all accepted).
        """
        ...


# ---------------------------------------------------------------------------
# speculative_sample -- the acceptance-rejection core (Leviathan Algorithm 1).
# ---------------------------------------------------------------------------
#
# Pulled out as a free function (rather than buried in SpeculativeDecoder) so
# the math is testable in isolation with hand-built p/q tensors -- which is
# exactly what test_spec_decode.py does to confirm the emitted distribution.
#
# The rule, for each draft token x_i (i = 0..K-1):
#   r ~ Uniform[0, 1)
#   accept x_i  iff  r < min(1, p_i(x_i) / q_i(x_i))
#   on the FIRST rejection at i: emit a token sampled from the residual
#       distribution  norm(max(0, p_i - q_i))  and STOP.
#   if all K accepted: emit a bonus token sampled from p_{K+1}.
#
# Guaranteed progress: even a position-0 rejection still emits the resampled
# token, so the return list always has length n_accepted + 1 >= 1.
# ---------------------------------------------------------------------------


def speculative_sample(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """Run Leviathan acceptance-rejection over a drafted block.

    Args:
        draft_tokens: (K,) int64, the draft's proposed tokens.
        draft_probs:  (K, V) float, q_i over the vocab for each draft token.
        target_probs: (K+1, V) float, p_1..p_{K+1} from the target's verify.
        generator: optional torch.Generator for reproducible sampling (CPU
            tests pass one; the server leaves it None to use the global RNG).

    Returns:
        (emitted_tokens, n_accepted) where emitted_tokens has length
        n_accepted + 1 (>= 1, guaranteed progress) and n_accepted is the
        number of draft tokens accepted before the first rejection (== K if
        all were accepted and the bonus token was appended).
    """
    k = int(draft_tokens.shape[0])
    emitted: list[int] = []

    for i in range(k):
        xi = int(draft_tokens[i])
        p_xi = float(target_probs[i, xi])
        q_xi = float(draft_probs[i, xi])
        # ratio = p/q clamped into [0, 1]. q_xi > 0 because the draft sampled
        # xi from q_i; the guard is belt-and-braces for degenerate inputs.
        accept_prob = 1.0 if q_xi <= 0.0 else min(1.0, p_xi / q_xi)
        r = float(torch.rand((), generator=generator))
        if r < accept_prob:
            emitted.append(xi)
            continue
        # --- rejection at position i: resample from the residual ----------
        # residual = max(0, p_i - q_i). This is the distribution that, mixed
        # with the accept step, makes the marginal of the emitted token equal
        # p_i exactly (Leviathan's correction term).
        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        total = float(residual.sum())
        if total <= 0.0:
            # p_i and q_i agree everywhere they overlap (no positive residual).
            # Falling back to p_i is distribution-preserving and avoids a divide
            # by zero. In practice this only happens with contrived inputs.
            dist = target_probs[i]
        else:
            dist = residual / total
        new_tok = int(torch.multinomial(dist, num_samples=1, generator=generator))
        emitted.append(new_tok)
        return emitted, i  # i drafts accepted, then one resampled token

    # All K draft tokens accepted: emit the bonus from p_{K+1} (row K).
    bonus = int(torch.multinomial(target_probs[k], num_samples=1, generator=generator))
    emitted.append(bonus)
    return emitted, k


# ---------------------------------------------------------------------------
# _autoregressive_draft -- shared K-step sampling loop for model-based drafts.
# ---------------------------------------------------------------------------
#
# TinyDraftModel and SelfSpecDraftModel differ only in HOW they turn a context
# into next-token logits (full small-model forward vs early-exit of the
# target). Everything else -- the K-step loop, temperature, softmax, sampling,
# and stacking the q_i rows -- is identical, so it lives here once.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _autoregressive_draft(
    logits_fn,
    input_ids: torch.Tensor,
    k: int,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample k tokens by repeatedly calling `logits_fn(context) -> (1, S, V)`.

    Returns (token_ids (K,), draft_probs (K, V)). No KV cache: the context
    grows by one token per step and is re-fed in full. That is O(k * S) work,
    which is fine for tiny draft models and keeps this path cache-agnostic.
    """
    device = input_ids.device
    context = input_ids
    tokens: list[int] = []
    prob_rows: list[torch.Tensor] = []
    for _ in range(k):
        logits = logits_fn(context)                 # (1, S, V)
        last = logits[0, -1, :].to(torch.float32)   # (V,)
        q = F.softmax(last / temperature, dim=-1)    # categorical for this slot
        tok = int(torch.multinomial(q, num_samples=1, generator=generator))
        tokens.append(tok)
        prob_rows.append(q)
        nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
        context = torch.cat([context, nxt], dim=1)
    token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    draft_probs = torch.stack(prob_rows, dim=0)      # (K, V)
    return token_ids, draft_probs


# ---------------------------------------------------------------------------
# TinyDraftModel -- a smaller LlamaModel used as the draft.
# ---------------------------------------------------------------------------


class TinyDraftModel:
    """Wrap a smaller LlamaModel as a DraftModel.

    The "smaller" is the whole point: a 2-4 layer model proposes tokens far
    cheaper than the full target, and the target verifies K of them in one
    pass. This wrapper just adapts the model's forward to the propose()
    contract (autoregressive sampling, returning the full q_i per token).
    """

    def __init__(self, model: "LlamaModel", temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature

    @torch.no_grad()
    def propose(
        self, input_ids: torch.Tensor, k: int, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # kv_cache is accepted for protocol symmetry but unused: the no-cache
        # recompute keeps the draft self-contained (see SmallModelDraft in
        # draft_model.py for the KV-cached variant).
        return _autoregressive_draft(
            lambda ctx: self.model(ctx), input_ids, k, self.temperature, _gen(self)
        )


# ---------------------------------------------------------------------------
# SelfSpecDraftModel -- the v0.3 early-exit draft, exposed as a DraftModel.
# ---------------------------------------------------------------------------


class SelfSpecDraftModel:
    """Adapt early-exit self-speculation to the DraftModel interface.

    Reuses `early_exit_forward` (the same first-`n_layers`-blocks + final norm
    + lm_head path the scheduler's greedy spec uses), but returns the softmax
    distributions so the general acceptance-rejection sampler can drive it.
    No new weights -- the draft IS the target, run shallow.
    """

    def __init__(self, model: "LlamaModel", n_layers: int = DEFAULT_N_DRAFT_LAYERS,
                 temperature: float = 1.0) -> None:
        self.model = model
        self.n_layers = n_layers
        self.temperature = temperature

    @torch.no_grad()
    def propose(
        self, input_ids: torch.Tensor, k: int, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # No-cache early exit: pass kv_cache=None so each block does a full
        # (1, S) forward over the growing context (the layers handle None).
        return _autoregressive_draft(
            lambda ctx: early_exit_forward(self.model, ctx, None, n_layers=self.n_layers),
            input_ids, k, self.temperature, _gen(self),
        )


# ---------------------------------------------------------------------------
# FullModelTarget -- a LlamaModel as the verifying target.
# ---------------------------------------------------------------------------


class FullModelTarget:
    """Wrap a LlamaModel as a TargetModel: one forward verifies the whole block."""

    def __init__(self, model: "LlamaModel", temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature

    @torch.no_grad()
    def verify(
        self, input_ids: torch.Tensor, draft_tokens: torch.Tensor, kv_cache=None
    ) -> torch.Tensor:
        device = input_ids.device
        dt = draft_tokens.to(device).view(1, -1)
        seq = torch.cat([input_ids, dt], dim=1)      # (1, S+K)
        logits = self.model(seq)                     # (1, S+K, V), one pass
        s = input_ids.shape[1]
        k = dt.shape[1]
        # Position s-1 (last context token) predicts slot 1; position s+k-1
        # (last draft token) predicts the bonus slot K+1. The causal mask
        # makes each of these depend only on the tokens up to and including it,
        # so this matches the autoregressive draft positions exactly.
        window = logits[0, s - 1 : s + k, :].to(torch.float32)   # (K+1, V)
        return F.softmax(window / self.temperature, dim=-1)


def _gen(obj) -> torch.Generator | None:
    """Return a per-object torch.Generator if one was set, else None.

    Draft wrappers don't take a generator directly (keeps their __init__ to the
    spec); SpeculativeDecoder injects one by setting obj._generator so a seeded
    decode is reproducible end-to-end. None falls back to the global RNG.
    """
    return getattr(obj, "_generator", None)


# ---------------------------------------------------------------------------
# SpeculativeDecoder -- orchestrates one draft -> verify -> accept round.
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """Drive draft/target speculative decoding and track the acceptance rate.

    One `decode_step` = one speculative round: the draft proposes K tokens, the
    target verifies them in a single forward, and `speculative_sample` accepts a
    prefix (resampling the first rejection). Emits between 1 and K+1 tokens.
    """

    def __init__(self, draft_model, target_model, k: int = 4,
                 generator: torch.Generator | None = None) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = k
        self.generator = generator
        # Share the generator with the model wrappers so the WHOLE round
        # (draft sampling + acceptance + resample) is reproducible from one
        # seed. _gen() reads this attribute off each wrapper.
        for m in (draft_model, target_model):
            try:
                m._generator = generator
            except AttributeError:
                pass  # Protocol objects we don't own; they'll use global RNG.

        # Metrics. acceptance_rate is the LAST step's rate; the running totals
        # feed mean_acceptance_rate across the whole generation.
        self.acceptance_rate: float = 0.0
        self.last_accepted: int = 0
        self._total_accepted: int = 0
        self._total_drafted: int = 0
        self._steps: int = 0

    @torch.no_grad()
    def decode_step(self, input_ids: torch.Tensor, kv_cache=None) -> list[int]:
        """One speculative round over context `input_ids` (1, S).

        Returns the list of newly accepted/emitted token ids (length 1..K+1).
        Updates `acceptance_rate` and the running mean.
        """
        draft_tokens, draft_probs = self.draft_model.propose(input_ids, self.k, kv_cache=kv_cache)
        target_probs = self.target_model.verify(input_ids, draft_tokens, kv_cache=kv_cache)
        emitted, n_accepted = speculative_sample(
            draft_tokens, draft_probs, target_probs, generator=self.generator
        )
        self.last_accepted = n_accepted
        self.acceptance_rate = n_accepted / self.k
        self._total_accepted += n_accepted
        self._total_drafted += self.k
        self._steps += 1
        return emitted

    @property
    def mean_acceptance_rate(self) -> float:
        """Accepted draft tokens / total drafted, across all decode_steps."""
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted
