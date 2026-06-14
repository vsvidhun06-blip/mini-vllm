"""
Tests for the CARLController control loop.

CPU, torch-free, model-free: the controller drives its components through
attribute writes, so SimpleNamespace stubs stand in for the scheduler / spec
decoder / router / KV cache (mirroring test_auto_tuner's approach). We verify:
  * _apply() writes every declared knob across all four components.
  * step() selects then updates the bandit in the right (delayed-reward) order.
  * controller_log records each decision.
  * stats() reports the regime distribution and adaptation count.
  * override + reset behave.
  * concurrent step() calls don't corrupt the log or bandit state.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from src.carl.bandit import PerRegimeBandit
from src.carl.config import CARLConfig, all_arm_sets
from src.carl.controller import CARLController, SLO
from src.carl.state import FEATURE_DIM, MetricsTracker, RuntimeState, WorkloadRegime


# ---------------------------------------------------------------------------
# Component stubs carrying exactly the attributes CARL drives.
# ---------------------------------------------------------------------------


def _scheduler_stub():
    return SimpleNamespace(
        max_batch_size=8, chunk_size=256, preemption_enabled=True,
        use_cuda_graphs=True, spec_decode_k=4, enable_spec_decode=False,
        waiting=[], active=[],
    )


def _spec_stub():
    return SimpleNamespace(k=4, mean_acceptance_rate=0.3)


def _router_stub():
    return SimpleNamespace(routing_threshold=0.5, cache_affinity_weight=0.0)


def _kv_stub():
    return SimpleNamespace(
        evict_threshold=0.8, recent_window=32, cache_hit_rate=0.1,
        score_tracker=SimpleNamespace(recent_window=32),
    )


def _bandit():
    return PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM, alpha=0.5)


def _controller(**kw):
    defaults = dict(
        scheduler=_scheduler_stub(), spec_decoder=_spec_stub(),
        router=_router_stub(), kv_cache=_kv_stub(), bandit=_bandit(),
        observe_interval=50, slo=SLO(),
    )
    defaults.update(kw)
    return CARLController(**defaults)


# ---------------------------------------------------------------------------
# _apply()
# ---------------------------------------------------------------------------


def test_apply_updates_all_components():
    c = _controller()
    cfg = CARLConfig(
        max_batch_size=12, chunk_size=384, preemption_enabled=False, spec_k=3,
        routing_threshold=0.7, cache_affinity_weight=0.9,
        eviction_threshold=0.75, eviction_window=48, use_cuda_graphs=False,
    )
    c._apply(cfg)

    assert c.scheduler.max_batch_size == 12
    assert c.scheduler.chunk_size == 384
    assert c.scheduler.preemption_enabled is False
    assert c.scheduler.use_cuda_graphs is False
    assert c.scheduler.spec_decode_k == 3
    assert c.scheduler.enable_spec_decode is True       # spec_k > 0
    assert c.spec_decoder.k == 3
    assert c.router.routing_threshold == pytest.approx(0.7)
    assert c.router.cache_affinity_weight == pytest.approx(0.9)
    assert c.kv_cache.evict_threshold == pytest.approx(0.75)
    assert c.kv_cache.recent_window == 48
    assert c.kv_cache.score_tracker.recent_window == 48


def test_apply_spec_k_zero_disables_speculation():
    c = _controller()
    c._apply(CARLConfig(spec_k=0))
    assert c.scheduler.enable_spec_decode is False
    assert c.spec_decoder.k == 0
    # spec_decode_k floored at 1 so the scheduler never reads k=0 as a draft len.
    assert c.scheduler.spec_decode_k == 1


def test_apply_clamps_out_of_range():
    c = _controller()
    c._apply(CARLConfig(max_batch_size=999, spec_k=99))
    assert c.scheduler.max_batch_size == 32        # clamped to range high
    assert c.spec_decoder.k == 8


def test_apply_skips_absent_components():
    # No router / kv_cache wired in -> those writes are silently skipped.
    c = CARLController(scheduler=_scheduler_stub(), bandit=_bandit())
    c._apply(CARLConfig(max_batch_size=5))         # must not raise
    assert c.scheduler.max_batch_size == 5


# ---------------------------------------------------------------------------
# step() ordering + bandit interaction.
# ---------------------------------------------------------------------------


def test_step_selects_and_applies():
    c = _controller()
    state = RuntimeState(queue_depth=1, avg_prompt_len=20)   # interactive
    entry = c.step(step_idx=0, state=state)
    assert entry.regime is WorkloadRegime.INTERACTIVE
    # The applied config is the selected arm; scheduler reflects it.
    assert c.scheduler.max_batch_size == entry.config.max_batch_size


def test_step_updates_previous_arm_in_order():
    # Spy bandit records the call order of select/update.
    calls: list[tuple] = []

    class SpyBandit:
        def __init__(self, inner):
            self.inner = inner
        def select(self, regime, ctx):
            arm, cfg = self.inner.select(regime, ctx)
            calls.append(("select", regime, arm))
            return arm, cfg
        def update(self, regime, arm, reward, ctx):
            calls.append(("update", regime, arm))
            self.inner.update(regime, arm, reward, ctx)
        def selection_counts(self):
            return self.inner.selection_counts()
        def arms(self, r):
            return self.inner.arms(r)

    c = _controller(bandit=SpyBandit(_bandit()))
    s = RuntimeState(queue_depth=1, avg_prompt_len=20)
    c.step(step_idx=0, state=s)     # first step: select only (no prev to update)
    c.step(step_idx=50, state=s)    # second step: update(prev) AFTER select

    kinds = [k[0] for k in calls]
    # First cycle: a select with no preceding update.
    assert kinds[0] == "select"
    # The second cycle issues both a select and an update for the previous arm.
    assert "update" in kinds
    # The update credits the arm selected in the FIRST cycle.
    first_arm = calls[0][2]
    update_calls = [k for k in calls if k[0] == "update"]
    assert update_calls[0][2] == first_arm


def test_step_no_update_on_first_cycle():
    c = _controller()
    c.step(step_idx=0, state=RuntimeState(queue_depth=1, avg_prompt_len=20))
    # Exactly one log entry, and no bandit arm has accumulated an update yet
    # (every arm still at the identity prior across all regimes).
    import numpy as np
    for reg_bandit in c.bandit.bandits.values():
        for A in reg_bandit.A:
            assert np.allclose(A, np.identity(FEATURE_DIM))
    assert len(c.controller_log) == 1


# ---------------------------------------------------------------------------
# Logging + stats.
# ---------------------------------------------------------------------------


def test_controller_log_records_each_decision():
    c = _controller()
    states = [
        RuntimeState(queue_depth=1, avg_prompt_len=20),    # interactive
        RuntimeState(queue_depth=12, active_requests=10),  # batch
        RuntimeState(avg_prompt_len=700),                  # long context
    ]
    for i, s in enumerate(states):
        c.step(step_idx=i, state=s)
    assert len(c.controller_log) == 3
    assert [e.regime for e in c.controller_log] == [
        WorkloadRegime.INTERACTIVE, WorkloadRegime.BATCH, WorkloadRegime.LONG_CONTEXT,
    ]
    # Each entry carries a JSON-friendly view.
    d = c.controller_log[0].as_dict()
    assert set(d) == {"step", "regime", "config", "reward", "state_features"}
    assert len(d["state_features"]) == FEATURE_DIM


def test_stats_regime_distribution():
    c = _controller()
    for _ in range(3):
        c.step(state=RuntimeState(queue_depth=1, avg_prompt_len=20))   # interactive
    for _ in range(2):
        c.step(state=RuntimeState(avg_prompt_len=700))                 # long context
    stats = c.stats()
    assert stats["regime_distribution"]["interactive"] == 3
    assert stats["regime_distribution"]["long_context"] == 2
    assert set(stats) == {
        "regime_distribution", "config_distribution", "mean_reward_per_regime",
        "total_adaptations", "best_config_per_regime",
    }


def test_stats_counts_adaptations_on_change():
    c = _controller()
    # Force a known config sequence via overrides applied through step()? Use
    # direct _apply to drive adaptation counting deterministically.
    c._apply(CARLConfig(max_batch_size=4))
    c._apply(CARLConfig(max_batch_size=4))     # identical -> not an adaptation
    c._apply(CARLConfig(max_batch_size=8))     # changed -> adaptation
    # First apply counts as the initial adaptation; identical repeat does not;
    # the change does. So total = 2.
    assert c.stats()["total_adaptations"] == 2


def test_mean_reward_per_regime_present():
    c = _controller()
    for _ in range(4):
        c.step(state=RuntimeState(queue_depth=1, avg_prompt_len=20))
    mrr = c.stats()["mean_reward_per_regime"]
    assert "interactive" in mrr
    assert 0.0 <= mrr["interactive"] <= 1.0


# ---------------------------------------------------------------------------
# Override + reset.
# ---------------------------------------------------------------------------


def test_override_applies_and_sticks():
    c = _controller()
    c.apply_override(CARLConfig(max_batch_size=3, spec_k=0))
    assert c.scheduler.max_batch_size == 3
    # A subsequent step keeps the override (no bandit selection).
    c.step(state=RuntimeState(queue_depth=1, avg_prompt_len=20))
    assert c.scheduler.max_batch_size == 3


def test_reset_clears_state():
    c = _controller()
    for _ in range(3):
        c.step(state=RuntimeState(queue_depth=1, avg_prompt_len=20))
    c.reset()
    assert c.controller_log == []
    assert c.stats()["regime_distribution"] == {}
    assert c.stats()["total_adaptations"] == 0


# ---------------------------------------------------------------------------
# maybe_step cadence.
# ---------------------------------------------------------------------------


def test_maybe_step_only_on_interval():
    c = _controller(observe_interval=50)
    # Drive observe() (no explicit state) -- components are empty stubs.
    fired = [c.maybe_step(i) is not None for i in range(101)]
    # Fires at 0, 50, 100.
    assert [i for i, f in enumerate(fired) if f] == [0, 50, 100]


# ---------------------------------------------------------------------------
# Thread-safety.
# ---------------------------------------------------------------------------


def test_concurrent_steps_dont_corrupt_state():
    c = _controller()
    s = RuntimeState(queue_depth=12, active_requests=10)   # batch regime

    def worker():
        for _ in range(50):
            c.step(state=s)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 4 threads * 50 steps = 200 well-formed log entries, no exceptions, and the
    # applied config is always a valid BATCH arm.
    assert len(c.controller_log) == 200
    batch_arms = {a.max_batch_size for a in c.bandit.arms(WorkloadRegime.BATCH)}
    assert c.scheduler.max_batch_size in batch_arms
