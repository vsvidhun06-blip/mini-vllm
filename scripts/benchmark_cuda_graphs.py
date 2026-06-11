"""
Benchmark: eager decode vs CUDA-graph decode.

Run:
    python scripts/benchmark_cuda_graphs.py

Needs a CUDA GPU and TinyLlama in the HF cache (run `python -m src.engine.model`
once). CPU-only hosts can't run this -- CUDA graphs are a GPU feature -- so it
prints a notice and exits.

WHAT IT MEASURES
----------------
A decode step is launch-bound: hundreds of tiny kernels, one new token per
request, so the GPU finishes each kernel faster than Python can queue the next.
A CUDA graph records the whole step once and replays it with a single launch,
removing that per-launch overhead.

We capture a graph per batch size in {1, 2, 4, 8} at a FIXED KV context length
(SEQ_LEN) and time many replays of that single decode step, against the eager
path timed the same way. Holding seq_len fixed is the standard decode-step
microbenchmark: it isolates per-step launch overhead from the (separate) cost
of a growing KV cache. The metric is TPOT -- time per output token -- which for
a batch is the per-step latency amortised over the batch_size tokens it emits.

Table: batch_size | eager TPOT (ms) | graph TPOT (ms) | speedup
"""
from __future__ import annotations

import time

import torch

from src.engine.cuda_graph import CUDAGraphRunner
from src.engine.device import DEVICE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf

SEQ_LEN = 256          # fixed KV context length each captured step decodes at
WARMUP_STEPS = 200     # untimed steps to settle clocks / caches before timing
TIMED_STEPS = 500      # timed decode steps
POOL = [1, 2, 4, 8]    # batch sizes (must be CUDAGraphRunner.POOL_SIZES)


def _time_loop(step_fn, iters: int) -> float:
    """Total wall time (ms) to run `step_fn` `iters` times, GPU-synchronised at
    both ends so we measure real device time, not just queue-submission time."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- CUDA graphs are a GPU feature; nothing to do.")
        return

    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    vocab = model.config.vocab_size

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Decode-step microbenchmark @ seq_len={SEQ_LEN} "
          f"({WARMUP_STEPS} warmup + {TIMED_STEPS} timed steps)\n")

    runner = CUDAGraphRunner(model)
    rows = []
    for bs in POOL:
        # Capture (also builds the dedicated KV caches seeded to SEQ_LEN).
        runner.capture(model, bs, SEQ_LEN)
        caches = runner.caches_for(bs)
        input_ids = torch.randint(0, vocab, (bs, 1), device=DEVICE)

        def eager_step():
            # Re-seed so every step decodes the same fixed-context token (and
            # the KV cache never grows past its allocation).
            runner.reset(bs)
            with torch.no_grad():
                model(input_ids, kv_cache=caches)

        def graph_step():
            runner.reset(bs)
            runner.replay(input_ids, caches)

        # Warm both paths, then time.
        _time_loop(eager_step, WARMUP_STEPS)
        _time_loop(graph_step, WARMUP_STEPS)
        eager_ms = _time_loop(eager_step, TIMED_STEPS)
        graph_ms = _time_loop(graph_step, TIMED_STEPS)

        # TPOT = per-step latency / batch_size (each step emits `bs` tokens).
        tpot_eager = eager_ms / (TIMED_STEPS * bs)
        tpot_graph = graph_ms / (TIMED_STEPS * bs)
        speedup = (tpot_eager / tpot_graph) if tpot_graph else float("nan")
        rows.append((bs, tpot_eager, tpot_graph, speedup))

    header = (f"{'batch_size':>10} | {'eager TPOT (ms)':>16} | "
              f"{'graph TPOT (ms)':>16} | {'speedup':>8}")
    print(header)
    print("-" * len(header))
    for bs, te, tg, sp in rows:
        print(f"{bs:>10} | {te:>16.4f} | {tg:>16.4f} | {sp:>7.2f}x")

    print(
        "\nCUDA graphs help most at SMALL batch sizes, where the step is most "
        "launch-bound (the GPU starves waiting on Python). As batch size grows, "
        "each kernel does more work and the per-launch overhead matters less, so "
        "the speedup shrinks."
    )


if __name__ == "__main__":
    main()
