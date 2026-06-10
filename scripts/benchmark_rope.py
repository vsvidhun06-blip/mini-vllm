"""
Microbenchmark: PyTorch apply_rope vs the fused Triton RoPE kernel.

Run:
    python scripts/benchmark_rope.py

This is GPU-only -- the whole point is the fused CUDA kernel, so on a CPU-only
host we bail with a message rather than print meaningless numbers.

Method:
    * Per config we build a (B, H, S, D) Q-shaped tensor (TinyLlama head
      geometry: H=32 query heads, D=64) plus its cos/sin tables.
    * We time the two implementations with CUDA events (the only honest way
      to time async GPU work), 100 warmup iters to settle clocks/caches +
      autotuning, then 1000 timed iters.
    * We report per-iteration milliseconds and the speedup.

Why a microbench and not a pytest: benchmarks want absolute numbers in a quiet
process, not pass/fail in a noisy test run. Same rationale as bench_gpu.py.
"""
from __future__ import annotations

import torch

from src.engine.attention import apply_rope, build_rope_cache

# TinyLlama head geometry. We benchmark the Q layout (the larger of Q/K).
NUM_HEADS = 32
HEAD_DIM = 64

# (batch, seq_len) configs to sweep.
CONFIGS = [
    (1, 512),
    (4, 128),
    (8, 256),
]

WARMUP = 100
TIMED = 1000


def _time_ms(fn, iters: int) -> float:
    """Average ms/iter for `fn` over `iters` runs, timed with CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # elapsed_time is already ms


def _bench_config(B: int, S: int) -> tuple[float, float]:
    """Return (pytorch_ms, triton_ms) for one (B, S) config."""
    from src.engine.kernels.rope import fused_rope

    x = torch.randn(B, NUM_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float32)
    cos, sin = build_rope_cache(HEAD_DIM, S)
    cos = cos.to("cuda")
    sin = sin.to("cuda")

    # Single-tensor rotation for both paths so the comparison is apples-to-apples.
    def pytorch_path():
        apply_rope(x, x, cos, sin)

    def triton_path():
        fused_rope(x, cos, sin)

    # Warmup (also triggers Triton JIT/autotune on the first call).
    for _ in range(WARMUP):
        pytorch_path()
        triton_path()

    pytorch_ms = _time_ms(pytorch_path, TIMED)
    triton_ms = _time_ms(triton_path, TIMED)
    return pytorch_ms, triton_ms


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- the fused RoPE kernel is GPU-only. Skipping.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Head geometry: H={NUM_HEADS}, D={HEAD_DIM} | warmup={WARMUP}, timed={TIMED}\n")

    header = f"{'config':>14} | {'pytorch ms':>11} | {'triton ms':>10} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for B, S in CONFIGS:
        pytorch_ms, triton_ms = _bench_config(B, S)
        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
        cfg = f"B={B},S={S}"
        print(f"{cfg:>14} | {pytorch_ms:>11.4f} | {triton_ms:>10.4f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
