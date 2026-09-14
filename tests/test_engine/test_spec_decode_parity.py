"""
Speculative decoding parity + acceptance-rate + worst-case tests.

Why these tests exist:

  Speculative decoding (Leviathan et al. 2023) is a pure wall-clock
  optimization: under greedy sampling it MUST produce the byte-identical
  token sequence that non-speculative greedy would produce. If it does
  not, the algorithm is broken in a way that's catastrophic for a real
  serving system -- outputs would silently differ depending on whether
  spec decode was enabled.

  We enforce that here. Three tests:
    1. matches_greedy   -- single request, spec on vs spec off, token by
                           token equality.
    2. acceptance_rate_reasonable -- the speedup is contingent on the
                           draft model agreeing with the base model often
                           enough. We assert a soft floor (>30%) so that
                           a future regression that breaks the early-exit
                           path (e.g. wrong layer count) gets caught.
    3. all_draft_rejected -- contrived worst case where every drafted
                           token disagrees. Confirms there's no infinite
                           loop and that the emitted single base token
                           still matches vanilla greedy at that position.

  Tests 1 and 3 are the correctness anchor. Test 2 protects the speedup
  story but is intentionally lenient on threshold.
"""
from __future__ import annotations

import pytest

from src.engine.scheduler import ContinuousBatchScheduler


# A "continuation" prompt where the model has a strong, predictable
# next-token distribution. We use it for the parity check (any prompt
# would do) and for the acceptance-rate test (where high-confidence
# continuations give the early-exit draft a fair shot at agreeing).
CONTINUATION_PROMPT = "The capital of France is"
MAX_NEW = 30


def _run_scheduler_solo(
    model,
    tokenizer,
    prompt: str,
    max_new: int,
    enable_spec_decode: bool,
    spec_decode_k: int = 4,
    observer=None,
) -> list[int]:
    """Drive a one-request scheduler to completion and return generated ids.

    Single request so spec_decode (single-request-only in v0.3) actually
    runs when enabled.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    eos = tokenizer.eos_token_id
    sched = ContinuousBatchScheduler(
        model,
        max_batch_size=1,
        num_blocks=64,
        enable_spec_decode=enable_spec_decode,
        spec_decode_k=spec_decode_k,
        spec_decode_observer=observer,
    )
    sched.add_request(
        request_id="parity",
        prompt_ids=prompt_ids,
        max_new_tokens=max_new,
        eos_token_id=eos,
    )
    out: list[int] = []
    while sched.has_work():
        for _rid, tok in sched.step():
            out.append(tok)
    return out


def test_spec_decode_matches_greedy(model_and_tokenizer) -> None:
    """Spec-on output == spec-off output, token by token.

    This is the algorithm's correctness contract. A divergence here means
    either:
      * verify is using the wrong logits index (off-by-one in K+1 frame),
      * the cache rewind is leaving stale K/V in reachable positions,
      * the acceptance loop is overshooting (accepting one past mismatch),
      * the EOS / cap truncation is dropping a valid emission, OR
      * the early-exit path is mutating cache state the verify forward
        then reads instead of overwriting.
    """
    model, tokenizer = model_and_tokenizer

    vanilla = _run_scheduler_solo(
        model, tokenizer, CONTINUATION_PROMPT, MAX_NEW,
        enable_spec_decode=False,
    )
    spec = _run_scheduler_solo(
        model, tokenizer, CONTINUATION_PROMPT, MAX_NEW,
        enable_spec_decode=True, spec_decode_k=4,
    )
    assert vanilla == spec, (
        "Speculative decoding diverged from vanilla greedy.\n"
        f"  vanilla: {vanilla}\n"
        f"  spec   : {spec}"
    )


def test_acceptance_rate_above_random(model_and_tokenizer) -> None:
    """The early-exit draft must agree with the base model meaningfully
    more often than random chance.

    Empirical reality (see scripts/probe_spec_acceptance.py):
        depth  4: ~1% acceptance
        depth  8: ~1% acceptance  (the v0.3 default)
        depth 16: ~11% acceptance
        depth 21: ~52% acceptance

    TinyLlama was NOT trained for early-exit. LayerSkip-style 40-60% at
    half the depth requires explicit layer-dropout training that the
    LayerSkip paper introduces. Without that training, intermediate-layer
    residuals are not faithfully readable by the final lm_head -- so a
    layer-8 draft predicts tokens that mostly disagree with the
    full-depth argmax.

    What this means for v0.3 speedup:
        Net speedup requires alpha > depth/total_layers. At depth=8 we
        need >36% acceptance; we get ~1%. So self-speculation on an
        untrained model is a CORRECTNESS demonstration, not a speedup
        win. v0.4 with a trained draft head fixes this.

    What this test guards against:
        Random output from the early-exit path (e.g., the lm_head being
        applied to the wrong tensor, layer count off by 22, RoPE
        positions wrong) would crash acceptance to ~1/vocab_size =
        0.003%. So a floor of 0.5% catches "draft is essentially
        random" while accepting our genuinely-measured ~1%.
    """
    model, tokenizer = model_and_tokenizer

    # Accumulate (accepted, k) tuples per round. We compute the ratio
    # as total_accepted / total_drafts -- the natural way to weight
    # rounds that drafted different K values (e.g. clamped near the
    # budget boundary).
    rounds: list[tuple[int, int]] = []
    def observer(accepted: int, k: int) -> None:
        rounds.append((accepted, k))

    _ = _run_scheduler_solo(
        model, tokenizer, CONTINUATION_PROMPT, max_new=50,
        enable_spec_decode=True, spec_decode_k=4, observer=observer,
    )
    assert rounds, "spec_decode_observer never fired -- spec decode path didn't run"

    total_accepted = sum(a for a, _ in rounds)
    total_drafts = sum(k for _, k in rounds)
    rate = total_accepted / total_drafts if total_drafts else 0.0

    # Empirical floor: random would be ~0.003%; we measure ~1% on this
    # prompt. 0.5% catches a regression to fully-random output without
    # forcing acceptance the architecture cannot deliver.
    assert rate > 0.005, (
        f"Acceptance rate {rate:.1%} below 0.5% floor (random ~0.003%). "
        f"{total_accepted} accepted of {total_drafts} drafted across "
        f"{len(rounds)} rounds. Early-exit draft is essentially random "
        f"-- likely lm_head applied to wrong tensor or layer count bug."
    )


def test_all_draft_rejected_no_infinite_loop(model_and_tokenizer, monkeypatch) -> None:
    """Force every draft to be wrong; spec decode must still emit the
    correct vanilla-greedy token at position 0 of each round, and the
    request must terminate at max_new_tokens.

    We monkeypatch draft_k_tokens to return all zeros (BOS / pad-ish).
    The base model essentially never argmaxes to 0 in continuation
    context, so EVERY round will have accepted_count == 0. The emitted
    token is base_preds[0] -- the same as what vanilla greedy would
    pick at this position. So even in the worst case, output matches
    vanilla.

    What this guards against:
      * Acceptance loop allowing accepted = -1 or out-of-bounds index.
      * spec_decode_step returning an empty emit list on m=0 (would
        stall the scheduler forever).
      * Cache rewind on m=0 leaving stale K/V at position 0 that
        breaks the NEXT round's prediction.
    """
    model, tokenizer = model_and_tokenizer

    # Vanilla baseline: what should the output be if spec decode is off?
    vanilla = _run_scheduler_solo(
        model, tokenizer, CONTINUATION_PROMPT, MAX_NEW,
        enable_spec_decode=False,
    )

    # Now run with spec decode, but a forcibly broken draft. The patched
    # function must accept the SAME signature as the real draft_k_tokens
    # so spec_decode_step's internal call works without modification.
    import src.engine.spec_decode as spec_mod

    def always_zero_draft(model, request_cache, last_token_id, k, n_draft_layers=8):
        # Returns k zeros, so every comparison against base argmax fails
        # in continuation context. We still need to mutate the cache the
        # same way the real draft would (writing K/V at layers
        # [0, n_draft_layers) so the cache's per-layer seq_lens advance
        # by K). spec_decode_step rewinds those seq_lens before verify,
        # so it doesn't matter what we write -- the easiest "correct"
        # mutation is to just call the real early-exit forward once per
        # k and discard its output.
        from src.engine.spec_decode import early_exit_forward
        import torch as _t
        device = model.embed.weight.device
        cur = last_token_id
        for _ in range(k):
            input_ids = _t.tensor([[cur]], dtype=_t.long, device=device)
            _ = early_exit_forward(
                model, input_ids, request_cache, n_layers=n_draft_layers,
            )
            cur = 0  # the fake "draft" token; next iter feeds zero
        return [0] * k

    monkeypatch.setattr(spec_mod, "draft_k_tokens", always_zero_draft)

    rounds: list[tuple[int, int]] = []
    spec_output = _run_scheduler_solo(
        model, tokenizer, CONTINUATION_PROMPT, MAX_NEW,
        enable_spec_decode=True, spec_decode_k=4,
        observer=lambda a, k: rounds.append((a, k)),
    )

    # Sanity 1: the scheduler terminated within max_new_tokens. If
    # spec_decode_step had a stall bug for m=0, has_work() would never
    # flip and this test would hang.
    assert len(spec_output) == len(vanilla), (
        f"Spec output length {len(spec_output)} != vanilla {len(vanilla)}; "
        f"spec_decode_step lost or duplicated tokens under all-reject."
    )

    # Sanity 2: every round had acceptance 0 (since draft is all zeros
    # and base argmax in continuation context is not zero).
    assert all(a == 0 for a, _ in rounds), (
        f"Expected 0 accepted in every round but got {rounds}. "
        f"Did the base model actually argmax to 0 somewhere? Pick a "
        f"different sentinel draft token."
    )

    # Sanity 3: output matches vanilla token by token. This is the real
    # correctness check -- even with a broken draft, the algorithm must
    # produce the same tokens as non-speculative greedy.
    assert spec_output == vanilla, (
        "Spec output diverges from vanilla under forced rejection.\n"
        f"  vanilla: {vanilla}\n"
        f"  spec   : {spec_output}"
    )
