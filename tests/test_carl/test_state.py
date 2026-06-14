"""
Tests for the runtime state observer, regime classifier, and metrics tracker.

All CPU, torch-free, model-free: RuntimeState reads its inputs through getattr,
so we drive it with SimpleNamespace stubs (the same pattern test_auto_tuner uses
for the scheduler). That keeps these fast and dependency-light while exercising
the real observe()/classify_regime()/MetricsTracker code paths.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.carl.state import (
    FEATURE_DIM,
    MetricsTracker,
    RuntimeState,
    WorkloadRegime,
    classify_regime,
)


# ---------------------------------------------------------------------------
# RuntimeState feature vector.
# ---------------------------------------------------------------------------


def test_feature_vector_length_matches_dim():
    vec = RuntimeState().to_feature_vector()
    assert len(vec) == FEATURE_DIM
    assert all(isinstance(v, float) for v in vec)


def test_feature_names_align_with_vector():
    names = RuntimeState.feature_names()
    assert len(names) == FEATURE_DIM
    # The two views must enumerate the same features in the same order.
    s = RuntimeState(queue_depth=64, gpu_utilization=1.0)
    vec = s.to_feature_vector()
    assert names[0] == "queue_depth" and vec[0] == pytest.approx(1.0)  # 64/64
    assert names[2] == "gpu_utilization" and vec[2] == pytest.approx(1.0)


def test_feature_vector_normalization():
    # Each feature divided by its characteristic scale.
    s = RuntimeState(
        queue_depth=32,            # /64  -> 0.5
        avg_prompt_len=512,        # /1024 -> 0.5
        cache_hit_rate=0.25,       # /1   -> 0.25
        p50_ttft_ms=250,           # /500 -> 0.5
        throughput_tps=50,         # /100 -> 0.5
    )
    v = dict(zip(RuntimeState.feature_names(), s.to_feature_vector()))
    assert v["queue_depth"] == pytest.approx(0.5)
    assert v["avg_prompt_len"] == pytest.approx(0.5)
    assert v["cache_hit_rate"] == pytest.approx(0.25)
    assert v["p50_ttft_ms"] == pytest.approx(0.5)
    assert v["throughput_tps"] == pytest.approx(0.5)


def test_as_dict_roundtrips_features():
    s = RuntimeState(queue_depth=5, avg_prompt_len=100.0, active_requests=3)
    d = s.as_dict()
    assert d["queue_depth"] == 5
    assert d["avg_prompt_len"] == 100.0
    assert d["active_requests"] == 3
    assert set(d) == set(RuntimeState.feature_names())


# ---------------------------------------------------------------------------
# observe() against stubs.
# ---------------------------------------------------------------------------


def _req(prompt_len: int):
    return SimpleNamespace(prompt_len=prompt_len)


def test_observe_reads_all_components():
    scheduler = SimpleNamespace(
        waiting=[_req(10), _req(20)],         # queue depth 2
        active=[_req(30)],                    # 1 in flight
    )
    spec = SimpleNamespace(mean_acceptance_rate=0.4)
    kv = SimpleNamespace(cache_hit_rate=0.6)
    metrics = MetricsTracker()
    metrics.record_request(ttft_ms=120, tpot_ms=40)
    metrics.record_throughput(55.0)
    metrics.record_batch(3)

    state = RuntimeState.observe(
        scheduler=scheduler, spec_decoder=spec, kv_cache=kv, metrics=metrics,
        gpu_utilization=0.75,
    )
    assert state.queue_depth == 2
    assert state.active_requests == 1
    # avg prompt len over waiting+active = (10+20+30)/3 = 20
    assert state.avg_prompt_len == pytest.approx(20.0)
    assert state.gpu_utilization == pytest.approx(0.75)
    assert state.cache_hit_rate == pytest.approx(0.6)
    assert state.spec_acceptance_rate == pytest.approx(0.4)
    assert state.p50_ttft_ms == pytest.approx(120.0)
    assert state.p99_tpot_ms == pytest.approx(40.0)
    assert state.throughput_tps == pytest.approx(55.0)
    assert state.batch_size_mean == pytest.approx(3.0)


def test_observe_all_none_is_safe():
    # The defensive contract: nothing wired in -> a zeroed-but-valid state.
    state = RuntimeState.observe()
    assert state.queue_depth == 0
    assert state.active_requests == 0
    assert state.gpu_utilization == 0.0     # pynvml absent on the CI box
    assert len(state.to_feature_vector()) == FEATURE_DIM


def test_observe_reads_prompt_ids_shape():
    # Real Requests carry a (1, S) tensor-like prompt_ids; we read shape[-1].
    fake_tensor = SimpleNamespace(shape=(1, 48))
    scheduler = SimpleNamespace(
        waiting=[], active=[SimpleNamespace(prompt_ids=fake_tensor)]
    )
    state = RuntimeState.observe(scheduler=scheduler)
    assert state.avg_prompt_len == pytest.approx(48.0)


def test_observe_spec_acceptance_falls_back_to_last():
    spec = SimpleNamespace(acceptance_rate=0.2)   # no mean_acceptance_rate
    state = RuntimeState.observe(spec_decoder=spec)
    assert state.spec_acceptance_rate == pytest.approx(0.2)


def test_observe_cache_hit_rate_from_counters():
    kv = SimpleNamespace(cache_hits=3, cache_lookups=12)   # 0.25
    state = RuntimeState.observe(kv_cache=kv)
    assert state.cache_hit_rate == pytest.approx(0.25)


def test_observe_batch_mean_falls_back_to_active():
    # No batch samples recorded -> batch_size_mean uses the instantaneous count.
    scheduler = SimpleNamespace(waiting=[], active=[_req(1), _req(1)])
    state = RuntimeState.observe(scheduler=scheduler, metrics=MetricsTracker())
    assert state.batch_size_mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# classify_regime: one synthetic state per regime.
# ---------------------------------------------------------------------------


def test_classify_long_context():
    # avg_prompt_len > 512 wins regardless of a deep queue.
    s = RuntimeState(avg_prompt_len=600, queue_depth=50, active_requests=1)
    assert classify_regime(s) is WorkloadRegime.LONG_CONTEXT


def test_classify_cache_heavy():
    s = RuntimeState(cache_hit_rate=0.7, avg_prompt_len=40, queue_depth=2)
    assert classify_regime(s) is WorkloadRegime.CACHE_HEAVY


def test_classify_burst():
    # Deep backlog (30) that the in-flight batch (2) isn't absorbing.
    s = RuntimeState(queue_depth=30, active_requests=2, avg_prompt_len=40,
                     cache_hit_rate=0.1)
    assert classify_regime(s) is WorkloadRegime.BURST


def test_classify_batch_via_queue():
    # Steady queue (10) being absorbed (active 8) -> not burst, but BATCH.
    s = RuntimeState(queue_depth=10, active_requests=8, avg_prompt_len=40,
                     cache_hit_rate=0.1)
    assert classify_regime(s) is WorkloadRegime.BATCH


def test_classify_batch_via_prompt_len():
    # Moderately long prompts (300) with no queue still read as throughput regime.
    s = RuntimeState(queue_depth=0, avg_prompt_len=300, cache_hit_rate=0.1)
    assert classify_regime(s) is WorkloadRegime.BATCH


def test_classify_interactive_default():
    s = RuntimeState(queue_depth=1, active_requests=1, avg_prompt_len=30,
                     cache_hit_rate=0.0)
    assert classify_regime(s) is WorkloadRegime.INTERACTIVE


def test_classify_covers_every_regime():
    states = {
        WorkloadRegime.LONG_CONTEXT: RuntimeState(avg_prompt_len=600),
        WorkloadRegime.CACHE_HEAVY: RuntimeState(cache_hit_rate=0.6, avg_prompt_len=40),
        WorkloadRegime.BURST: RuntimeState(queue_depth=30, active_requests=2, avg_prompt_len=40),
        WorkloadRegime.BATCH: RuntimeState(queue_depth=12, active_requests=10, avg_prompt_len=40),
        WorkloadRegime.INTERACTIVE: RuntimeState(queue_depth=0, avg_prompt_len=20),
    }
    for expected, s in states.items():
        assert classify_regime(s) is expected


# ---------------------------------------------------------------------------
# WorkloadRegime transitions over a sequence of observations.
# ---------------------------------------------------------------------------


def test_regime_transition_sequence():
    # Simulate a workload shift: interactive -> batch -> burst -> long_context.
    timeline = [
        RuntimeState(queue_depth=1, avg_prompt_len=20),                     # interactive
        RuntimeState(queue_depth=12, active_requests=10, avg_prompt_len=40),  # batch
        RuntimeState(queue_depth=40, active_requests=3, avg_prompt_len=40),   # burst
        RuntimeState(avg_prompt_len=700),                                   # long context
    ]
    regimes = [classify_regime(s) for s in timeline]
    assert regimes == [
        WorkloadRegime.INTERACTIVE,
        WorkloadRegime.BATCH,
        WorkloadRegime.BURST,
        WorkloadRegime.LONG_CONTEXT,
    ]


# ---------------------------------------------------------------------------
# MetricsTracker.
# ---------------------------------------------------------------------------


def test_metrics_percentiles_and_throughput():
    m = MetricsTracker()
    for ttft in range(1, 101):                 # 1..100 ms
        m.record_request(ttft_ms=ttft, tpot_ms=ttft / 2.0)
    # Nearest-rank P50 of 1..100 is 50; P99 of the tpot (0.5..50) is ~49.5.
    assert m.p50_ttft_ms() == pytest.approx(50.0, abs=1.0)
    assert m.p99_tpot_ms() == pytest.approx(49.5, abs=1.0)
    m.record_throughput(10.0)
    m.record_throughput(30.0)
    assert m.throughput_tps() == pytest.approx(20.0)


def test_metrics_violation_rates():
    m = MetricsTracker()
    # 4 requests; TTFTs 50/150/50/250 vs a 100ms SLO -> 2 of 4 violate.
    for ttft, tpot in [(50, 10), (150, 10), (50, 80), (250, 120)]:
        m.record_request(ttft_ms=ttft, tpot_ms=tpot)
    assert m.ttft_violation_rate(100.0) == pytest.approx(0.5)
    # TPOTs 10/10/80/120 vs a 50ms SLO -> 2 of 4 violate.
    assert m.tpot_violation_rate(50.0) == pytest.approx(0.5)


def test_metrics_empty_windows_are_zero():
    m = MetricsTracker()
    assert m.p50_ttft_ms() == 0.0
    assert m.p99_tpot_ms() == 0.0
    assert m.throughput_tps() == 0.0
    assert m.ttft_violation_rate(100.0) == 0.0


def test_metrics_window_is_bounded():
    m = MetricsTracker(window=10)
    for i in range(50):
        m.record_request(ttft_ms=float(i), tpot_ms=float(i))
    assert len(m._ttft) == 10
