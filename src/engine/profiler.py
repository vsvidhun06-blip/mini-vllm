"""
Continuous step profiling: measure where each scheduler iteration spends time.

A serving engine's bottleneck moves around at runtime. A burst of long prompts
makes prefill dominate; a big decode batch makes the decode forward dominate; a
nearly-full KV pool makes block allocation / eviction dominate; and at small
batch sizes Python/launch overhead can dominate everything else. You cannot tune
the engine sensibly without knowing which of these is currently the limiter --
so we measure it, every step, with a cheap rolling window.

StepProfiler splits each iteration into four phases:

    prefill   -- the prompt forward pass(es)
    decode    -- the batched decode forward
    kv_alloc  -- admission + block allocation (the "memory" phase)
    overhead  -- everything else (eviction sweep, event emission, Python glue)

It is driven by the scheduler via a tiny mark-based API (begin_step / lap /
end_step) that records a timestamp at each phase boundary, so instrumenting the
scheduler costs four one-line calls and zero block re-indentation. The window
keeps the last `window` steps; `bottleneck()` returns whichever phase has the
largest rolling-average time, and `to_dict()` exports the stats the auto-tuner
and the /profiler endpoint consume.

This module is import-time torch-free: the GPU-accurate clock
(CUDAEventProfiler) imports torch lazily and falls back to a wall-clock timer
when CUDA is unavailable, so the profiler and its tests run anywhere.
"""
from __future__ import annotations

import time
from collections import deque


# ---------------------------------------------------------------------------
# Clocks. A clock turns "measure the interval between two boundary marks" into
# either a wall-clock delta or a GPU-event delta. mark() returns an opaque
# handle; delta(a, b) returns seconds; finalize() is called once per step
# before any delta() so a GPU clock can synchronise.
# ---------------------------------------------------------------------------


class PerfCounterClock:
    """Wall-clock timing via time.perf_counter (CPU-accurate, no sync)."""

    def mark(self):
        return time.perf_counter()

    def delta(self, a, b) -> float:
        return b - a

    def finalize(self) -> None:
        pass


class CUDAEventProfiler:
    """GPU-accurate clock using torch.cuda.Event, with a perf_counter fallback.

    GPU kernels run asynchronously, so a host-side perf_counter around a forward
    measures launch time, not execution time. CUDA events are recorded INTO the
    stream and their elapsed_time (after a synchronise) reflects real GPU work.
    When CUDA isn't available -- or torch isn't installed -- this transparently
    degrades to wall-clock timing, so it is a safe default everywhere.
    """

    def __init__(self) -> None:
        self._torch = None
        self.use_cuda = False
        try:
            import torch  # lazy: keeps this module importable without torch
            self._torch = torch
            self.use_cuda = bool(torch.cuda.is_available())
        except Exception:
            self.use_cuda = False

    def mark(self):
        if self.use_cuda:
            ev = self._torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def delta(self, a, b) -> float:
        # Called only after finalize() has synchronised, so elapsed_time is safe.
        if self.use_cuda:
            return a.elapsed_time(b) / 1000.0   # CUDA reports milliseconds
        return b - a

    def finalize(self) -> None:
        if self.use_cuda:
            self._torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# StepProfiler
# ---------------------------------------------------------------------------


class StepProfiler:
    """Per-step phase timing over a rolling window, with bottleneck detection."""

    # Internal phase keys. "kv_alloc" surfaces to callers as the "memory"
    # bottleneck (it's the memory-management phase); the other three keep their
    # names. Order is the deterministic tie-break for bottleneck().
    PHASES = ("prefill", "decode", "kv_alloc", "overhead")

    def __init__(self, window: int = 100, clock=None) -> None:
        self.window: deque = deque(maxlen=window)
        self.clock = clock or PerfCounterClock()
        self._marks = None  # list[(label, handle)] for the in-progress step

    # ---- mark-based instrumentation (driven by the scheduler) -----------

    def begin_step(self) -> None:
        """Start timing a step. Records the opening boundary mark."""
        self._marks = [(None, self.clock.mark())]

    def lap(self, label: str) -> None:
        """Close the current phase and label it. `label` is one of PHASES;
        the elapsed time since the previous mark is attributed to it."""
        if self._marks is not None:
            self._marks.append((label, self.clock.mark()))

    def end_step(self) -> dict | None:
        """Finish the step: the trailing interval is 'overhead'. Resolves all
        boundary deltas (after a clock finalize/sync) and appends one record to
        the rolling window. Returns that record (phase -> seconds)."""
        if self._marks is None:
            return None
        self._marks.append(("overhead", self.clock.mark()))
        self.clock.finalize()
        rec = {p: 0.0 for p in self.PHASES}
        for i in range(1, len(self._marks)):
            label = self._marks[i][0]
            dt = self.clock.delta(self._marks[i - 1][1], self._marks[i][1])
            rec[label if label in rec else "overhead"] += dt
        self._marks = None
        self.window.append(rec)
        return rec

    # ---- direct recording (tests / external callers) --------------------

    def record_step(self, prefill: float, decode: float,
                    kv_alloc: float, overhead: float) -> None:
        """Append a step's phase times directly (seconds). Used by tests and by
        callers that measure phases themselves."""
        self.window.append({
            "prefill": prefill, "decode": decode,
            "kv_alloc": kv_alloc, "overhead": overhead,
        })

    # ---- analysis -------------------------------------------------------

    def averages(self) -> dict:
        """Rolling mean time per phase (seconds). Zeros when the window is empty."""
        if not self.window:
            return {p: 0.0 for p in self.PHASES}
        n = len(self.window)
        return {p: sum(s[p] for s in self.window) / n for p in self.PHASES}

    def bottleneck(self) -> str | None:
        """The dominant phase by rolling-average time.

        Returns "prefill" | "decode" | "memory" | "overhead", or None when no
        steps have been recorded (or every phase is zero). "memory" is the
        external name for the kv_alloc phase.
        """
        if not self.window:
            return None
        avg = self.averages()
        ranked = [
            ("prefill", avg["prefill"]),
            ("decode", avg["decode"]),
            ("memory", avg["kv_alloc"]),
            ("overhead", avg["overhead"]),
        ]
        name, value = max(ranked, key=lambda kv: kv[1])
        return name if value > 0 else None

    def to_dict(self) -> dict:
        """Export rolling stats for the auto-tuner and the /profiler endpoint."""
        avg = self.averages()
        total = sum(avg.values()) or 1.0
        return {
            "n_steps": len(self.window),
            "window": self.window.maxlen,
            "bottleneck": self.bottleneck(),
            "avg_ms": {
                "prefill": avg["prefill"] * 1000.0,
                "decode": avg["decode"] * 1000.0,
                "memory": avg["kv_alloc"] * 1000.0,
                "overhead": avg["overhead"] * 1000.0,
            },
            # Fractions feed the Grafana "bottleneck distribution" pie.
            "fractions": {
                "prefill": avg["prefill"] / total,
                "decode": avg["decode"] / total,
                "memory": avg["kv_alloc"] / total,
                "overhead": avg["overhead"] / total,
            },
        }
