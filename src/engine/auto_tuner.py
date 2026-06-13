"""
Auto-tuner: read the live bottleneck and adjust serving parameters to chase it.

The StepProfiler tells us WHICH phase dominates each scheduler iteration. The
AutoTuner closes the loop: every N steps it reads StepProfiler.bottleneck() and
nudges the one parameter most likely to relieve that bottleneck.

  bottleneck == "prefill"  -> chunk_size up (bigger prefill chunks = fewer,
                              fatter prefill passes; cap 512)
  bottleneck == "decode"   -> max_batch_size down (smaller decode batches cut
                              per-step decode latency; floor 1)
  bottleneck == "memory"   -> evict_threshold down (start KV eviction sooner so
                              admission stops stalling on a full pool)
  bottleneck == "overhead" -> enable CUDA graphs (collapse hundreds of kernel
                              launches per decode into one replay)

Two guards keep it from thrashing:
  * It only acts every `tune_interval` steps (default 50) -- enough history for
    a stable bottleneck read.
  * A per-parameter `cooldown` (default 200 steps) blocks re-tuning the same
    knob too soon, so a single sustained bottleneck ramps a parameter gradually
    instead of slamming it to its bound in three consecutive reads.

Every applied change is appended to `tuning_log` as
(step, bottleneck, action, old_val, new_val). The tuner mutates LIVE scheduler
attributes (chunk_size, max_batch_size, use_cuda_graphs, evict_threshold), all
of which the scheduler reads fresh each step, so changes take effect on the very
next iteration.

Torch-free: this is pure control logic over scalar parameters, so it (and its
tests) run anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TuningConfig:
    """Bounds and step sizes for each tuning rule."""
    # prefill rule
    chunk_size_max: int = 512
    chunk_size_increment: int = 128
    # decode rule
    max_batch_floor: int = 1
    max_batch_decrement: int = 1
    # memory rule
    evict_threshold_factor: float = 0.8     # multiplicative reduction
    evict_threshold_floor: float = 0.3
    # cadence
    tune_interval: int = 50
    cooldown: int = 200


# One place to map a bottleneck name to (parameter, human-readable action).
_PARAM_FOR = {
    "prefill": ("chunk_size", "increase_chunk_size"),
    "decode": ("max_batch_size", "reduce_max_batch_size"),
    "memory": ("evict_threshold", "reduce_eviction_threshold"),
    "overhead": ("use_cuda_graphs", "enable_cuda_graphs"),
}


class AutoTuner:
    """Reads StepProfiler.bottleneck() periodically and tunes scheduler params."""

    def __init__(self, profiler, config: TuningConfig | None = None,
                 tune_interval: int | None = None, cooldown: int | None = None) -> None:
        self.profiler = profiler
        self.config = config or TuningConfig()
        self.tune_interval = (tune_interval if tune_interval is not None
                              else self.config.tune_interval)
        self.cooldown = cooldown if cooldown is not None else self.config.cooldown
        # (step, bottleneck, action, old_val, new_val) per applied change.
        self.tuning_log: list[tuple] = []
        # param name -> step at which it was last tuned (for cooldown).
        self._last_tuned: dict[str, int] = {}
        # Current step, advanced by observe(); also settable directly in tests.
        self.step_count: int = 0

    # ---- driving ---------------------------------------------------------

    def observe(self, scheduler, step: int | None = None):
        """Call once per scheduler step. Tunes only on `tune_interval` boundaries.

        Returns the tuning_log entry applied this step, or None.
        """
        self.step_count = step if step is not None else self.step_count + 1
        if self.tune_interval <= 0 or self.step_count % self.tune_interval != 0:
            return None
        return self.apply_tuning(scheduler)

    def apply_tuning(self, scheduler, config: TuningConfig | None = None):
        """Read the current bottleneck and apply its rule, honouring cooldown.

        Mutates `scheduler` in place. Returns the (step, bottleneck, action,
        old, new) log entry if a change was made, else None.
        """
        cfg = config or self.config
        bottleneck = self.profiler.bottleneck()
        if bottleneck is None:
            return None

        proposal = self._propose(bottleneck, scheduler, cfg)
        if proposal is None:
            return None
        param, action, old, new = proposal

        # No-op: already at the bound (or nothing to change). Don't log or
        # start a cooldown for a change we didn't make.
        if new == old:
            return None

        # Cooldown: don't re-tune the same parameter too soon.
        last = self._last_tuned.get(param)
        if last is not None and (self.step_count - last) < self.cooldown:
            return None

        setattr(scheduler, param, new)
        self._last_tuned[param] = self.step_count
        entry = (self.step_count, bottleneck, action, old, new)
        self.tuning_log.append(entry)
        return entry

    # ---- rules -----------------------------------------------------------

    def _propose(self, bottleneck: str, scheduler, cfg: TuningConfig):
        """Compute (param, action, old_val, new_val) for a bottleneck, or None."""
        mapping = _PARAM_FOR.get(bottleneck)
        if mapping is None:
            return None
        param, action = mapping

        if bottleneck == "prefill":
            old = getattr(scheduler, "chunk_size", None)
            if old is None:
                return None
            new = min(cfg.chunk_size_max, old + cfg.chunk_size_increment)

        elif bottleneck == "decode":
            old = getattr(scheduler, "max_batch_size", None)
            if old is None:
                return None
            new = max(cfg.max_batch_floor, old - cfg.max_batch_decrement)

        elif bottleneck == "memory":
            # Default 0.8 matches EvictingPagedKVCache's evict_threshold; if the
            # scheduler has no such knob yet we set one (harmless on the base
            # scheduler, picked up by an eviction-capable cache).
            old = getattr(scheduler, "evict_threshold", 0.8)
            new = max(cfg.evict_threshold_floor,
                      round(old * cfg.evict_threshold_factor, 4))

        elif bottleneck == "overhead":
            old = bool(getattr(scheduler, "use_cuda_graphs", False))
            new = True

        else:  # pragma: no cover - guarded by _PARAM_FOR above
            return None

        return (param, action, old, new)

    # ---- export ----------------------------------------------------------

    def log_as_dicts(self) -> list[dict]:
        """tuning_log as JSON-friendly dicts (for the /tuning-log endpoint)."""
        return [
            {"step": s, "bottleneck": b, "action": a, "old": old, "new": new}
            for (s, b, a, old, new) in self.tuning_log
        ]
