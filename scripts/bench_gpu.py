"""
GPU-vs-CPU benchmark: solo decode + 4-way batched scheduler.

Run:
    python scripts/bench_gpu.py
    CUDA_VISIBLE_DEVICES="" python scripts/bench_gpu.py    # force CPU

What we report:
    * Device, dtype.
    * Single-request: total time, average ms / decode token.
    * 4-request batched: total wall time via the scheduler,
      tokens/sec aggregate, tokens/sec per-request.
    * Speedup of batched vs sequential solo.

Why this isn't a pytest:
    Benchmarks are sensitive to whatever else is on the machine. Pytest
    runs them in a noisy environment alongside other tests and reports
    pass/fail rather than numbers; that's the wrong shape. This is
    explicit, single-purpose, and prints absolute numbers + speedups.
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE, DTYPE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler


PROMPTS = [
    "The capital of France is",
    "Python is a programming language designed",
    "In 1969, the first humans landed on",
    "The largest ocean on Earth is the",
]
MAX_NEW = 32


def _solo(model, tokenizer) -> tuple[float, list[int]]:
    """Generate each prompt sequentially. Return total time + per-prompt token counts."""
    eos = tokenizer.eos_token_id
    counts: list[int] = []
    t0 = time.perf_counter()
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt")["input_ids"]
        out = model.generate(ids, max_new_tokens=MAX_NEW, eos_token_id=eos)
        counts.append(out.shape[1] - ids.shape[1])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, counts


def _batched(model, tokenizer, num_blocks: int) -> tuple[float, int, dict[str, int]]:
    """Drive all prompts through the scheduler. Return time, n_steps, per-rid token count."""
    eos = tokenizer.eos_token_id
    scheduler = ContinuousBatchScheduler(
        model,
        max_batch_size=len(PROMPTS),
        num_blocks=num_blocks,
    )
    for i, p in enumerate(PROMPTS):
        ids = tokenizer(p, return_tensors="pt")["input_ids"]
        scheduler.add_request(
            request_id=f"req-{i}",
            prompt_ids=ids,
            max_new_tokens=MAX_NEW,
            eos_token_id=eos,
        )

    counts: dict[str, int] = {f"req-{i}": 0 for i in range(len(PROMPTS))}
    n_steps = 0
    t0 = time.perf_counter()
    while scheduler.has_work():
        for rid, _tok in scheduler.step():
            counts[rid] += 1
        n_steps += 1
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, n_steps, counts


def main() -> None:
    from transformers import AutoTokenizer

    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("  TF32 matmul: off (preserve fp32 parity)")
    print()

    print("Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=DTYPE)
    model.eval()

    # Warm up: first forward pass on CUDA is much slower than steady-state
    # because of kernel JIT, allocator priming, and cuDNN benchmarking.
    print("Warmup...")
    ids = tokenizer("warmup", return_tensors="pt")["input_ids"]
    _ = model.generate(ids, max_new_tokens=4)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print()

    # ---- Solo --------------------------------------------------------
    solo_time, solo_counts = _solo(model, tokenizer)
    total_solo_tokens = sum(solo_counts)
    print(f"Solo (sequential): {solo_time:.2f}s")
    print(f"  {total_solo_tokens} tokens total across {len(PROMPTS)} prompts")
    print(f"  {total_solo_tokens / solo_time:.1f} tokens/sec aggregate")
    print(f"  {solo_time / total_solo_tokens * 1000:.1f} ms / token avg")
    print()

    # ---- Scheduler, ample blocks ------------------------------------
    sched_time, n_steps, sched_counts = _batched(model, tokenizer, num_blocks=64)
    total_sched_tokens = sum(sched_counts.values())
    speedup = solo_time / sched_time if sched_time > 0 else float("inf")
    print(f"Scheduler (ample blocks={64}): {sched_time:.2f}s  steps={n_steps}")
    print(f"  {total_sched_tokens} tokens total")
    print(f"  {total_sched_tokens / sched_time:.1f} tokens/sec aggregate")
    print(f"  {sched_time * 1000 / n_steps:.1f} ms / step avg")
    print(f"  speedup vs solo: {speedup:.2f}x")
    print()

    # ---- Scheduler, tight blocks ------------------------------------
    tight_time, tight_steps, tight_counts = _batched(model, tokenizer, num_blocks=6)
    total_tight_tokens = sum(tight_counts.values())
    tight_speedup = solo_time / tight_time if tight_time > 0 else float("inf")
    print(f"Scheduler (tight blocks=6, forces admission control): {tight_time:.2f}s  steps={tight_steps}")
    print(f"  {total_tight_tokens} tokens total")
    print(f"  {total_tight_tokens / tight_time:.1f} tokens/sec aggregate")
    print(f"  speedup vs solo: {tight_speedup:.2f}x")


if __name__ == "__main__":
    main()
