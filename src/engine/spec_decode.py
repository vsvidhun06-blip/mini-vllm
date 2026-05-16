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

Public surface:
  early_exit_forward(model, input_ids, kv_cache, n_layers) -> logits
  draft_k_tokens(model, request_cache, last_token_id, k, n_draft_layers) -> list[int]
  verify_full_forward(model, request_cache, last_token_id, draft_tokens) -> logits
  spec_decode_step(model, request_cache, last_token_id, k, eos_token_id, max_emit, n_draft_layers) -> (list[int], int)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
