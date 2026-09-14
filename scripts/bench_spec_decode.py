"""
Speculative decoding benchmark (Day 15).

What we measure:
    For each of four prompt types (continuation, Q&A, code, list), run
    the scheduler twice: once with vanilla decode, once with self-
    speculative decode (early-exit at layer 8, K=4 draft tokens). For
    each run, capture:
        * tokens/sec (max_new_tokens / decode wall time)
        * acceptance rate (with spec on, otherwise N/A)

What we honestly EXPECT this benchmark to show:

    On a model that was NOT trained for early-exit (TinyLlama is not),
    intermediate-layer residuals don't decode well via the final
    lm_head, so layer-8 acceptance rate is ~1% on natural text. The
    breakeven formula (alpha > depth/total_layers) requires >36%
    acceptance at depth 8; we don't get there.

    So the expected outcome is a SLOWDOWN of roughly 0.4-0.6x. That is
    the honest engineering result for this configuration of the
    algorithm. The infrastructure (KV cache rewind, scheduler hook,
    metrics) is real and correct; the parity test enforces byte-
    identical output to vanilla greedy. Picking up a real speedup
    requires either:
        (a) a trained early-exit head (LayerSkip-style),
        (b) a dedicated draft head (EAGLE / Medusa), or
        (c) a separately trained smaller draft model.
    All three are out of scope for v0.3. We document this here so the
    benchmark numbers don't blindside a reader.

Why this isn't a pytest:
    Wallclock numbers are hardware-dependent and noisy. The correctness
    guarantee is in tests/test_engine/test_spec_decode_parity.py
    (byte-identical to vanilla under greedy); the benchmark just
    surfaces the speed picture so the v0.3 writeup has data.

Run:
    python scripts/bench_spec_decode.py > docs/benchmarks/v0.3_spec_decode.txt
"""
from __future__ import annotations

import time
from typing import Iterable

import torch

from src.engine.device import DEVICE, DTYPE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler


# Four prompts spanning different generation regimes. The continuation
# one is the easiest for early-exit drafting (model has a strong
# distribution); the code and list prompts have more entropy.
PROMPTS = [
    ("continuation", "The capital of France is"),
    ("qa",           "Q: What is the boiling point of water in Celsius? A:"),
    ("code",         "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
    ("list",         "Three popular Python web frameworks are: 1."),
]
MAX_NEW = 50
NUM_BLOCKS = 64
SPEC_K = 4


def _run(
    model,
    tokenizer,
    prompt: str,
    *,
    enable_spec_decode: bool,
) -> dict:
    """One full request through a fresh single-request scheduler.

    Returns:
        dict with:
          * decode_wall_s: wall clock from first decode step to last
            emission (excludes prefill).
          * total_wall_s:  wall clock from add_request to has_work=False.
          * generated_tokens: count of emitted tokens (sanity-check
            against max_new_tokens).
          * accepted_total / drafts_total: spec-only acceptance counts.
    """
    eos = tokenizer.eos_token_id
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Spec-decode round counts. Captured per round via the observer
    # callback so the bench is independent of /metrics scrape timing.
    accepted_total = 0
    drafts_total = 0

    def observer(accepted: int, k: int) -> None:
        nonlocal accepted_total, drafts_total
        accepted_total += accepted
        drafts_total += k

    sched = ContinuousBatchScheduler(
        model,
        max_batch_size=1,
        num_blocks=NUM_BLOCKS,
        block_size=16,
        enable_spec_decode=enable_spec_decode,
        spec_decode_k=SPEC_K,
        spec_decode_observer=observer if enable_spec_decode else None,
    )
    sched.add_request(
        request_id="bench",
        prompt_ids=prompt_ids,
        max_new_tokens=MAX_NEW,
        eos_token_id=eos,
    )

    # Walk steps; the FIRST step is prefill (emits 1 token from prefill
    # logits), every later step is decode. We start the decode-wall timer
    # AFTER the prefill step so the speedup numbers are decode-only --
    # spec decode doesn't affect prefill.
    generated = 0
    decode_t0 = None
    total_t0 = time.perf_counter()
    while sched.has_work():
        if decode_t0 is None and generated > 0:
            # First emission already happened during prefill; the next
            # step is the first decode step.
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            decode_t0 = time.perf_counter()
        emitted = sched.step()
        generated += len(emitted)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    total_wall = time.perf_counter() - total_t0
    decode_wall = (time.perf_counter() - decode_t0) if decode_t0 else 0.0

    return {
        "decode_wall_s": decode_wall,
        "total_wall_s": total_wall,
        "generated_tokens": generated,
        "accepted_total": accepted_total,
        "drafts_total": drafts_total,
    }


def main() -> None:
    from transformers import AutoTokenizer

    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Spec decode config: K={SPEC_K}, draft_layers=8/22 (~36% of base cost)")
    print()

    print("Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=DTYPE)
    model.eval()

    # Warmup -- kernel JIT, CUDA graph capture, allocator stabilisation.
    # The first inference call in a process is always anomalously slow;
    # we do a throwaway pair so the timed runs reflect steady state.
    print("Warmup (vanilla)...")
    _ = _run(model, tokenizer, "Warmup", enable_spec_decode=False)
    print("Warmup (spec)...")
    _ = _run(model, tokenizer, "Warmup", enable_spec_decode=True)
    print()

    print("=" * 78)
    print(f"  Prompt-type benchmark   (max_new_tokens={MAX_NEW})")
    print("=" * 78)
    print(f"{'prompt':<14} {'mode':<8} {'decode_s':>10} {'tok/s':>9} {'tokens':>7} {'accept':>8}")
    print("-" * 78)

    for label, prompt in PROMPTS:
        van = _run(model, tokenizer, prompt, enable_spec_decode=False)
        spec = _run(model, tokenizer, prompt, enable_spec_decode=True)

        # tokens/sec in the decode-only window (excludes prefill).
        # We use (generated - 1) for decode-only token count because the
        # very first emission came from prefill, not from a decode step.
        van_decode_tokens = max(0, van["generated_tokens"] - 1)
        spec_decode_tokens = max(0, spec["generated_tokens"] - 1)
        van_tps = van_decode_tokens / van["decode_wall_s"] if van["decode_wall_s"] > 0 else 0.0
        spec_tps = spec_decode_tokens / spec["decode_wall_s"] if spec["decode_wall_s"] > 0 else 0.0
        accept_rate = (
            spec["accepted_total"] / spec["drafts_total"]
            if spec["drafts_total"] > 0 else 0.0
        )

        print(
            f"{label:<14} {'vanilla':<8} {van['decode_wall_s']:>10.3f} "
            f"{van_tps:>9.2f} {van['generated_tokens']:>7d} {'-':>8}"
        )
        print(
            f"{label:<14} {'spec':<8} {spec['decode_wall_s']:>10.3f} "
            f"{spec_tps:>9.2f} {spec['generated_tokens']:>7d} {accept_rate*100:>7.1f}%"
        )
        speedup = spec_tps / van_tps if van_tps > 0 else 0.0
        print(f"{'':<14} {'speedup':<8} {'':>10} {speedup:>8.2f}x")
        print()

    print("=" * 78)
    print(
        "Reminder: parity test (tests/test_engine/test_spec_decode_parity.py)\n"
        "enforces byte-identical output to vanilla greedy. Numbers above are\n"
        "wall-clock only; correctness is independent."
    )


if __name__ == "__main__":
    main()
