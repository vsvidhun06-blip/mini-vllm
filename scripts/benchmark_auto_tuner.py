"""
Benchmark: the auto-tuner reacting to a workload that shifts phase dominance.

Run:
    python scripts/benchmark_auto_tuner.py

This is a self-contained CONTROL-LOOP simulation (no model, no GPU): we feed the
StepProfiler synthetic per-step phase timings that start PREFILL-heavy and then
flip to DECODE-heavy, drive the AutoTuner over a stub scheduler, and show it
detecting each regime and adjusting the matching parameter.

Expected behaviour:
  * While prefill dominates, the tuner ramps chunk_size up (256 -> ... -> 512),
    one bump per cooldown window.
  * After the shift to decode-heavy, it starts reducing max_batch_size.
The printed tuning log is the evidence: step, detected bottleneck, action, and
old -> new value for every change.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from src.engine.auto_tuner import AutoTuner, TuningConfig
from src.engine.profiler import StepProfiler

TOTAL_STEPS = 600
SHIFT_AT = 300          # prefill-heavy before this step, decode-heavy after


def _phase_timings(step: int) -> dict:
    """Synthetic per-phase seconds for `step`. A little noise keeps the rolling
    average realistic; the dominant phase flips at SHIFT_AT."""
    jitter = 0.001 * ((step * 7) % 5)        # deterministic pseudo-noise
    base = {"prefill": 0.004, "decode": 0.004, "kv_alloc": 0.002, "overhead": 0.001}
    if step < SHIFT_AT:
        base["prefill"] = 0.020 + jitter      # prefill dominates
    else:
        base["decode"] = 0.020 + jitter       # decode dominates
    return base


def main() -> None:
    profiler = StepProfiler(window=100)
    # Short cooldown so the demo shows several adjustments within 600 steps.
    tuner = AutoTuner(profiler, config=TuningConfig(tune_interval=50, cooldown=100))
    scheduler = SimpleNamespace(
        chunk_size=256, max_batch_size=8, use_cuda_graphs=False, evict_threshold=0.8,
    )

    print(f"Simulating {TOTAL_STEPS} steps: PREFILL-heavy for the first "
          f"{SHIFT_AT}, then DECODE-heavy.\n")

    bottleneck_at_tune: list[tuple[int, str]] = []
    t0 = time.perf_counter()
    for step in range(1, TOTAL_STEPS + 1):
        ph = _phase_timings(step)
        profiler.record_step(ph["prefill"], ph["decode"], ph["kv_alloc"], ph["overhead"])
        applied = tuner.observe(scheduler, step=step)
        if applied is not None:
            bottleneck_at_tune.append((step, applied[1]))
    wall = time.perf_counter() - t0

    print("Final scheduler params:")
    print(f"  chunk_size      = {scheduler.chunk_size}")
    print(f"  max_batch_size  = {scheduler.max_batch_size}")
    print(f"  use_cuda_graphs = {scheduler.use_cuda_graphs}")
    print(f"  evict_threshold = {scheduler.evict_threshold}")
    print(f"\nSimulated {TOTAL_STEPS} steps in {wall * 1e3:.1f} ms\n")

    print("Tuning log:")
    header = f"  {'step':>5} | {'bottleneck':>10} | {'action':>22} | {'old -> new':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for step, bottleneck, action, old, new in tuner.tuning_log:
        print(f"  {step:>5} | {bottleneck:>10} | {action:>22} | "
              f"{str(old):>6} -> {str(new):<6}")

    # Sanity narration: the tuner should have chased prefill first, decode later.
    pre = [s for s, b in bottleneck_at_tune if b == "prefill"]
    dec = [s for s, b in bottleneck_at_tune if b == "decode"]
    if pre and dec:
        print(f"\nDetected prefill bottleneck at steps {pre}, then decode at "
              f"{dec} -- the auto-tuner tracked the workload shift.")


if __name__ == "__main__":
    main()
