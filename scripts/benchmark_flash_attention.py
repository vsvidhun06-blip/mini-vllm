"""
Microbenchmark: PyTorch SDPA vs the from-scratch FA2 Triton kernel.

Run:
    python scripts/benchmark_flash_attention.py

GPU-only. We model TTFT (time-to-first-token): the prefill step processes the
whole prompt at once with causal attention, and that attention op is the part
this kernel changes -- so we time a single causal prefill forward at each
sequence length as a TTFT proxy.

Method:
    * (B=1, NH=32, D=64) -- TinyLlama head geometry -- with S in {512,1024,2048}.
    * CUDA-event timing (the only honest way to time async GPU work):
      30 warmup iters (settles clocks + triggers Triton JIT/autotune), then
      100 timed iters.
    * Peak memory via reset_peak_memory_stats / max_memory_allocated around a
      single call, to show FA2's O(S) vs the naive O(S^2) score-matrix cost.

Table: seq_len | sdpa ms | flash_attn ms | speedup | memory saved
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_HEADS = 32
HEAD_DIM = 64
BATCH = 1
SEQ_LENS = [512, 1024, 2048]

WARMUP = 30
TIMED = 100


def _time_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_mb(fn) -> float:
    """Peak CUDA memory (MB) allocated during one call to `fn`."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024 * 1024)


def _bench(S: int):
    from src.engine.kernels.flash_attention import flash_attention_forward

    q = torch.randn(BATCH, NUM_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def sdpa_path():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def flash_path():
        return flash_attention_forward(q, k, v, causal=True)

    for _ in range(WARMUP):
        sdpa_path()
        flash_path()

    sdpa_ms = _time_ms(sdpa_path, TIMED)
    flash_ms = _time_ms(flash_path, TIMED)
    sdpa_mem = _peak_mb(sdpa_path)
    flash_mem = _peak_mb(flash_path)
    return sdpa_ms, flash_ms, sdpa_mem, flash_mem


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- flash attention is GPU-only. Skipping.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"B={BATCH}, NH={NUM_HEADS}, D={HEAD_DIM} | warmup={WARMUP}, timed={TIMED}\n")

    header = (
        f"{'seq_len':>8} | {'sdpa ms':>9} | {'flash_attn ms':>13} | "
        f"{'speedup':>8} | {'memory saved':>13}"
    )
    print(header)
    print("-" * len(header))

    for S in SEQ_LENS:
        sdpa_ms, flash_ms, sdpa_mem, flash_mem = _bench(S)
        speedup = sdpa_ms / flash_ms if flash_ms > 0 else float("inf")
        saved = sdpa_mem - flash_mem
        print(
            f"{S:>8} | {sdpa_ms:>9.4f} | {flash_ms:>13.4f} | "
            f"{speedup:>7.2f}x | {saved:>10.1f} MB"
        )


if __name__ == "__main__":
    main()
