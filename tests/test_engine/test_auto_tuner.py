"""
Profiler + auto-tuner tests. Pure control logic over scalar timings/params --
no torch, no GPU, no model.

  1. StepProfiler picks the dominant phase from synthetic step timings, and maps
     kv_alloc -> "memory".
  2. AutoTuner applies the correct rule for each bottleneck type.
  3. Cooldown blocks re-tuning the same parameter too soon.
  4. tuning_log records every applied action with old/new values.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.engine.auto_tuner import AutoTuner, TuningConfig
from src.engine.profiler import StepProfiler


def _profiler_with(prefill=0.0, decode=0.0, kv_alloc=0.0, overhead=0.0, n=5):
    """A StepProfiler whose window is filled with n identical synthetic steps."""
    p = StepProfiler(window=100)
    for _ in range(n):
        p.record_step(prefill=prefill, decode=decode,
                      kv_alloc=kv_alloc, overhead=overhead)
    return p


def _scheduler(**kw):
    """A stand-in scheduler exposing the tunable attributes."""
    defaults = dict(chunk_size=256, max_batch_size=8,
                    use_cuda_graphs=False, evict_threshold=0.8)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1. StepProfiler bottleneck detection.
# ---------------------------------------------------------------------------


def test_profiler_identifies_each_bottleneck():
    assert _profiler_with(prefill=0.10, decode=0.01).bottleneck() == "prefill"
    assert _profiler_with(decode=0.10, prefill=0.01).bottleneck() == "decode"
    # kv_alloc surfaces as "memory".
    assert _profiler_with(kv_alloc=0.10, decode=0.01).bottleneck() == "memory"
    assert _profiler_with(overhead=0.10, decode=0.01).bottleneck() == "overhead"


def test_profiler_empty_window_has_no_bottleneck():
    assert StepProfiler().bottleneck() is None
    # All-zero timings also report no bottleneck.
    assert _profiler_with().bottleneck() is None


def test_profiler_window_is_bounded():
    p = StepProfiler(window=100)
    for _ in range(250):
        p.record_step(0.01, 0.0, 0.0, 0.0)
    assert len(p.window) == 100


def test_profiler_to_dict_fractions_sum_to_one():
    p = _profiler_with(prefill=0.06, decode=0.02, kv_alloc=0.01, overhead=0.01)
    d = p.to_dict()
    assert d["bottleneck"] == "prefill"
    assert d["n_steps"] == 5
    fr = d["fractions"]
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert abs(fr["prefill"] - 0.6) < 1e-9     # 0.06 / 0.10


def test_profiler_mark_api_attributes_phases():
    """begin/lap/end attribute each interval to its phase via a fake clock."""
    # Fake clock: mark() returns a manually-advanced counter value.
    times = iter([0.0, 0.5, 1.5, 1.8, 2.0])   # 5 marks
    clock = SimpleNamespace(
        mark=lambda: next(times),
        delta=lambda a, b: b - a,
        finalize=lambda: None,
    )
    p = StepProfiler(clock=clock)
    p.begin_step()           # mark 0.0
    p.lap("kv_alloc")        # mark 0.5  -> kv_alloc = 0.5
    p.lap("prefill")         # mark 1.5  -> prefill  = 1.0
    p.lap("decode")          # mark 1.8  -> decode   = 0.3
    rec = p.end_step()       # mark 2.0  -> overhead = 0.2
    assert abs(rec["kv_alloc"] - 0.5) < 1e-9
    assert abs(rec["prefill"] - 1.0) < 1e-9
    assert abs(rec["decode"] - 0.3) < 1e-9
    assert abs(rec["overhead"] - 0.2) < 1e-9
    assert p.bottleneck() == "prefill"


# ---------------------------------------------------------------------------
# 2. AutoTuner rules.
# ---------------------------------------------------------------------------


def test_prefill_bottleneck_increases_chunk_size():
    tuner = AutoTuner(_profiler_with(prefill=0.1, decode=0.01))
    sched = _scheduler(chunk_size=256)
    tuner.step_count = 50
    entry = tuner.apply_tuning(sched)
    assert sched.chunk_size == 384                 # 256 + 128
    assert entry == (50, "prefill", "increase_chunk_size", 256, 384)


def test_chunk_size_capped_at_max():
    tuner = AutoTuner(_profiler_with(prefill=0.1))
    sched = _scheduler(chunk_size=500)
    tuner.step_count = 50
    tuner.apply_tuning(sched)
    assert sched.chunk_size == 512                 # min(512, 500+128)


def test_decode_bottleneck_reduces_batch_size():
    tuner = AutoTuner(_profiler_with(decode=0.1, prefill=0.01))
    sched = _scheduler(max_batch_size=8)
    tuner.step_count = 50
    entry = tuner.apply_tuning(sched)
    assert sched.max_batch_size == 7
    assert entry[1] == "decode" and entry[2] == "reduce_max_batch_size"


def test_batch_size_floored_at_one():
    tuner = AutoTuner(_profiler_with(decode=0.1))
    sched = _scheduler(max_batch_size=1)
    tuner.step_count = 50
    entry = tuner.apply_tuning(sched)
    assert sched.max_batch_size == 1               # already at floor -> no-op
    assert entry is None


def test_memory_bottleneck_reduces_eviction_threshold():
    tuner = AutoTuner(_profiler_with(kv_alloc=0.1, decode=0.01))
    sched = _scheduler(evict_threshold=0.8)
    tuner.step_count = 50
    entry = tuner.apply_tuning(sched)
    assert abs(sched.evict_threshold - 0.64) < 1e-9   # 0.8 * 0.8
    assert entry[1] == "memory" and entry[2] == "reduce_eviction_threshold"


def test_overhead_bottleneck_enables_cuda_graphs():
    tuner = AutoTuner(_profiler_with(overhead=0.1, decode=0.01))
    sched = _scheduler(use_cuda_graphs=False)
    tuner.step_count = 50
    entry = tuner.apply_tuning(sched)
    assert sched.use_cuda_graphs is True
    assert entry[1] == "overhead" and entry[2] == "enable_cuda_graphs"


def test_overhead_noop_when_graphs_already_on():
    tuner = AutoTuner(_profiler_with(overhead=0.1))
    sched = _scheduler(use_cuda_graphs=True)
    tuner.step_count = 50
    assert tuner.apply_tuning(sched) is None       # already on -> no change
    assert tuner.tuning_log == []


# ---------------------------------------------------------------------------
# 3. Cooldown.
# ---------------------------------------------------------------------------


def test_cooldown_prevents_over_tuning():
    tuner = AutoTuner(_profiler_with(prefill=0.1), cooldown=200)
    sched = _scheduler(chunk_size=128)

    tuner.step_count = 50
    assert tuner.apply_tuning(sched) is not None    # 128 -> 256, logged
    assert sched.chunk_size == 256

    # Within cooldown (50 -> 100, only 50 steps): blocked.
    tuner.step_count = 100
    assert tuner.apply_tuning(sched) is None
    assert sched.chunk_size == 256
    assert len(tuner.tuning_log) == 1

    # Past cooldown (50 -> 300 >= 200): allowed again.
    tuner.step_count = 300
    assert tuner.apply_tuning(sched) is not None    # 256 -> 384
    assert sched.chunk_size == 384
    assert len(tuner.tuning_log) == 2


def test_observe_only_tunes_on_interval():
    tuner = AutoTuner(_profiler_with(prefill=0.1), tune_interval=50)
    sched = _scheduler(chunk_size=128)
    # Steps 1..49: no tuning. Step 50: tunes once.
    applied = None
    for step in range(1, 51):
        result = tuner.observe(sched, step=step)
        if result is not None:
            applied = (step, result)
    assert applied is not None and applied[0] == 50
    assert sched.chunk_size == 256
    assert len(tuner.tuning_log) == 1


# ---------------------------------------------------------------------------
# 4. tuning_log records actions.
# ---------------------------------------------------------------------------


def test_tuning_log_records_all_actions():
    # A profiler whose bottleneck we can flip between reads.
    p = _profiler_with(prefill=0.1, decode=0.01)
    tuner = AutoTuner(p, cooldown=10)
    sched = _scheduler(chunk_size=128, max_batch_size=8)

    tuner.step_count = 50
    tuner.apply_tuning(sched)                       # prefill -> chunk_size

    # Switch the workload to decode-heavy and tune again.
    p.window.clear()
    for _ in range(5):
        p.record_step(prefill=0.01, decode=0.1, kv_alloc=0.0, overhead=0.0)
    tuner.step_count = 100
    tuner.apply_tuning(sched)                       # decode -> max_batch_size

    assert len(tuner.tuning_log) == 2
    actions = [e[2] for e in tuner.tuning_log]
    assert actions == ["increase_chunk_size", "reduce_max_batch_size"]
    # Each entry carries (step, bottleneck, action, old, new).
    for entry in tuner.tuning_log:
        assert len(entry) == 5
    dicts = tuner.log_as_dicts()
    assert dicts[0]["old"] == 128 and dicts[0]["new"] == 256
    assert dicts[1]["old"] == 8 and dicts[1]["new"] == 7
