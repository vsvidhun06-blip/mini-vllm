"""
Day 16 Phase 1: probe spec decode at three configs.

Hypothesis: Day 15's K=4 + exit-8 default left obvious knobs unturned.
Try K=2 (less draft waste per round) and exit-18 (higher acceptance per
draft step) to see if any combination crosses breakeven.

The breakeven inequality from Day 15:
    alpha > exit_layer / total_layers
For total_layers = 22:
    exit=8  -> need alpha > 36%
    exit=18 -> need alpha > 82%

What K does to the formula:
    speedup = (1 + alpha*K) / (K*c + 1)   where c = exit_layer/22
Lower K = less wasted draft when acceptance is low, but caps upside
when acceptance is high. We pick K in {2, 4} so the matrix is small.

Three configs run on the same single-request continuation prompt --
the easiest case for spec decode. If even this prompt doesn't show
speedup we're done with Phase 1. If one config crosses 1.1x with
acceptance >20%, we adopt it as v0.4 default.
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE, DTYPE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler


PROMPT = "The capital of France is"
MAX_NEW = 50
NUM_BLOCKS = 64

# (label, K, exit_layer)
CONFIGS = [
    ("Day15 default (K=4, exit=8)", 4, 8),
    ("(a) K=2, exit=8",             2, 8),
    ("(b) K=4, exit=18",            4, 18),
    ("(c) K=2, exit=18",            2, 18),
]


def _run(model, tokenizer, *, enable_spec, k, exit_layer):
    """Single-request run; return (decode_wall_s, tokens_total, accept_rate)."""
    accepted_total = 0
    drafts_total = 0
    def obs(a, kk):
        nonlocal accepted_total, drafts_total
        accepted_total += a
        drafts_total += kk

    sched = ContinuousBatchScheduler(
        model,
        max_batch_size=1,
        num_blocks=NUM_BLOCKS,
        enable_spec_decode=enable_spec,
        spec_decode_k=k,
        spec_decode_exit_layer=exit_layer,
        spec_decode_observer=obs if enable_spec else None,
    )
    sched.add_request(
        request_id="probe",
        prompt_ids=tokenizer(PROMPT, return_tensors="pt")["input_ids"],
        max_new_tokens=MAX_NEW,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = 0
    decode_t0 = None
    while sched.has_work():
        if decode_t0 is None and generated > 0:
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            decode_t0 = time.perf_counter()
        emitted = sched.step()
        generated += len(emitted)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    decode_wall = (time.perf_counter() - decode_t0) if decode_t0 else 0.0
    rate = accepted_total / drafts_total if drafts_total > 0 else 0.0
    return decode_wall, generated, rate


def main():
    from transformers import AutoTokenizer

    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompt: {PROMPT!r}   max_new_tokens={MAX_NEW}")
    print()

    print("Loading TinyLlama...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=DTYPE)
    model.eval()

    # Warmup so the first config doesn't pay JIT/kernel-launch overhead.
    print("Warmup...")
    _ = _run(model, tok, enable_spec=False, k=4, exit_layer=8)
    _ = _run(model, tok, enable_spec=True,  k=4, exit_layer=8)
    print()

    # Baseline vanilla once for the speedup denominators below.
    van_decode_s, van_tokens, _ = _run(model, tok, enable_spec=False, k=4, exit_layer=8)
    van_decode_tokens = max(0, van_tokens - 1)
    van_tps = van_decode_tokens / van_decode_s if van_decode_s > 0 else 0.0

    print(f"Vanilla baseline: {van_decode_s:.3f}s for {van_tokens} tokens = {van_tps:.2f} tok/s")
    print()
    print("=" * 80)
    print(f"{'config':<32} {'decode_s':>10} {'tok/s':>9} {'speedup':>9} {'accept':>8}")
    print("-" * 80)

    for label, k, exit_layer in CONFIGS:
        decode_s, n_tokens, rate = _run(
            model, tok, enable_spec=True, k=k, exit_layer=exit_layer,
        )
        decode_tokens = max(0, n_tokens - 1)
        tps = decode_tokens / decode_s if decode_s > 0 else 0.0
        speedup = tps / van_tps if van_tps > 0 else 0.0
        print(
            f"{label:<32} {decode_s:>10.3f} {tps:>9.2f} "
            f"{speedup:>8.2f}x {rate*100:>7.1f}%"
        )

    print("=" * 80)
    print()
    print(
        "Decision rule: adopt a config as v0.4 default if speedup > 1.10x\n"
        "AND acceptance > 20%. Otherwise Phase 1 is conclusive negative\n"
        "and we proceed to Phase 2 (real draft model)."
    )


if __name__ == "__main__":
    main()
