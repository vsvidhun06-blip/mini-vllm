"""The live reward-v2 adapter, its frozen-scales contract, and the abort gates.

These tests encode the two failures that produced the 2026-09-02 artifact:

  * the live controller silently used reward v1 and nobody noticed, and
  * the resulting constant reward was not detected until the run was over.

So: the injection is asserted, the v1 default is asserted to be UNCHANGED (the
reward-v1 run must stay reproducible), and both gates are asserted to REJECT the
exact reward-v1 signature that was actually observed on hardware.
"""
from __future__ import annotations

import pytest

from src.carl.bandit import DEFAULT_UTILITY_WEIGHTS, utility
from src.carl.controller import CARLController
from src.carl.live_reward import (
    MIN_ARM_REWARD_SPREAD, MIN_LIVE_DISTINCT, MIN_LIVE_REWARD_SPREAD,
    LiveRewardV2, gate_candidate_arm_spread, gate_live_reward_stream, spread,
)
from src.carl.reward import DegenerateRewardError, RewardScales
from src.carl.state import FEATURE_DIM, MetricsTracker, RuntimeState

# The operating point actually measured on the T4 on 2026-09-02.
_T4_SCALES = RewardScales(t_half=75.0, ttft_target=29000.0, tpot_target=120.0)
_T4_STATE = RuntimeState(queue_depth=40, avg_prompt_len=180.0,
                         throughput_tps=96.3, p50_ttft_ms=21084.0,
                         p99_tpot_ms=330.1)


def _tracker(ttfts) -> MetricsTracker:
    t = MetricsTracker(window=200)
    for v in ttfts:
        t.record_request(v, 330.0)
    return t


# --- the frozen-scales contract -------------------------------------------

def test_scales_must_be_supplied_explicitly():
    """No default scales, no from_measurements path: deriving scales from the
    evaluation runs would move the yardstick with the measurement."""
    with pytest.raises(TypeError):
        LiveRewardV2(scales=None)
    with pytest.raises(TypeError):
        LiveRewardV2(scales={"t_half": 60.0})


def test_as_dict_reports_exactly_what_was_frozen():
    r = LiveRewardV2(_T4_SCALES)
    d = r.as_dict()
    assert d["reward_version"] == "v2"
    assert d["scales"] == {"t_half": 75.0, "ttft_target": 29000.0,
                           "tpot_target": 120.0, "sharpness": 2.0}
    assert d["weights"] == DEFAULT_UTILITY_WEIGHTS


# --- it scores the TAIL, and does not touch the context vector -------------

def test_uses_ttft_p99_from_the_tracker_not_the_state_p50():
    r = LiveRewardV2(_T4_SCALES)
    m = r.raw_metrics(_T4_STATE, _tracker([100.0] * 5 + [9000.0] * 5))
    assert m["ttft_p99_ms"] == pytest.approx(9000.0)
    assert m["ttft_p99_ms"] != _T4_STATE.p50_ttft_ms


def test_falls_back_to_state_p50_without_a_tracker():
    r = LiveRewardV2(_T4_SCALES)
    assert r.raw_metrics(_T4_STATE, None)["ttft_p99_ms"] == _T4_STATE.p50_ttft_ms


def test_context_vector_is_untouched():
    """Widening _FEATURE_SCALES would change the bandit's input dimension and
    break the controlled pair against the reward-v1 run."""
    assert FEATURE_DIM == 10
    assert len(RuntimeState().to_feature_vector()) == 10
    assert "p99_ttft_ms" not in RuntimeState.feature_names()


# --- v2 discriminates exactly where v1 saturated ---------------------------

def test_v1_saturates_and_v2_does_not_at_the_measured_operating_point():
    """The decisive regression test. At the T4 operating point, reward v1
    returns the same 0.3 for wildly different configurations; v2 separates them."""
    from src.carl.controller import SLO
    slo = SLO(ttft_ms=200.0, tpot_ms=50.0, throughput_ref=50.0)

    def v1(tps):
        return utility({"throughput_norm": min(1.0, tps / slo.throughput_ref),
                        "ttft_violation_rate": 1.0, "tpot_violation_rate": 1.0,
                        "cache_hit_rate": 0.0}, DEFAULT_UTILITY_WEIGHTS)

    r = LiveRewardV2(_T4_SCALES)

    def v2(tps, ttft, tpot):
        st = RuntimeState(throughput_tps=tps, p50_ttft_ms=ttft, p99_tpot_ms=tpot)
        return r(st, _tracker([ttft]))[0]

    # Three genuinely different operating points from the T4 batch sweep, all
    # at or above the 50 tok/s v1 reference -- which is where the live run sat
    # (~96 tok/s). Below that reference v1's throughput term is NOT yet clipped;
    # the saturation claim is specifically about the measured operating range.
    points = [(75.71, 29367.0, 122.6), (89.09, 22254.0, 171.7),
              (99.72, 10879.0, 283.0)]
    v1_vals = [v1(p[0]) for p in points]
    v2_vals = [v2(*p) for p in points]

    assert spread(v1_vals) == pytest.approx(0.0, abs=1e-9), (
        "reward v1 must be shown to saturate here; that is the premise")
    assert v1_vals[0] == pytest.approx(0.3)
    assert spread(v2_vals) >= MIN_ARM_REWARD_SPREAD, v2_vals


# --- the gates -------------------------------------------------------------

def test_candidate_gate_rejects_the_observed_v1_constant():
    """The exact signature from the 2026-09-02 artifact must abort."""
    with pytest.raises(DegenerateRewardError, match="ABORTING before evaluation"):
        gate_candidate_arm_spread({f"arm{i}": 0.29999999999999993
                                   for i in range(6)}, context="test")


def test_candidate_gate_accepts_the_measured_non_degenerate_floor():
    """long_context measured 0.0219 on the repaired substrate -- the floor the
    threshold was derived from. It must pass."""
    rep = gate_candidate_arm_spread({"lo": 0.21088, "hi": 0.23281}, context="test")
    assert rep["passed"] is True
    assert rep["reward_spread"] == pytest.approx(0.02193, abs=1e-4)
    assert rep["threshold"] == MIN_ARM_REWARD_SPREAD


def test_live_gate_rejects_both_historical_failures():
    # 2026-09-02: one distinct value.
    with pytest.raises(DegenerateRewardError, match="ABORTING"):
        gate_live_reward_stream([0.3] * 39, context="test")
    # decisions_042.csv: two distinct values (0.8 then 0.3 forever).
    with pytest.raises(DegenerateRewardError, match="ABORTING"):
        gate_live_reward_stream([0.8, 0.8, 0.8] + [0.3] * 36, context="test")


def test_live_gate_accepts_a_varying_stream():
    rep = gate_live_reward_stream([0.30, 0.33, 0.36, 0.41], context="test")
    assert rep["passed"] is True
    assert rep["n_distinct"] >= MIN_LIVE_DISTINCT
    assert rep["reward_spread"] >= MIN_LIVE_REWARD_SPREAD


def test_empty_streams_abort_rather_than_pass_vacuously():
    with pytest.raises(DegenerateRewardError):
        gate_live_reward_stream([], context="test")
    with pytest.raises(DegenerateRewardError):
        gate_candidate_arm_spread({}, context="test")


# --- controller injection --------------------------------------------------

def test_controller_defaults_to_reward_v1_unchanged():
    """The reward-v1 T4 run must stay reproducible: the default path is v1 and
    reports an empty term breakdown."""
    # The tracker must hold the SLO-violating samples the live run had; v1's
    # violation terms read the tracker, and an empty window reports no
    # violations (0.8), which is not the operating point being reproduced.
    c = CARLController(observe_interval=1, metrics=_tracker([21084.0] * 10))
    assert c.reward_fn is None
    e = c.step(state=_T4_STATE)
    assert e.reward_terms == {}
    assert e.reward == pytest.approx(0.3), (
        "default path must reproduce the saturated v1 value measured on the T4")


def test_controller_uses_the_injected_reward_when_given_one():
    tr = _tracker([21084.0, 30000.0, 61848.0])
    c = CARLController(observe_interval=1, metrics=tr,
                       reward_fn=LiveRewardV2(_T4_SCALES))
    e = c.step(state=_T4_STATE)
    assert e.reward != pytest.approx(0.3)
    assert 0.0 < e.reward < 1.0
    for k in ("throughput_term", "ttft_term", "tpot_term", "cache_term", "total"):
        assert k in e.reward_terms
    assert e.reward_terms["total"] == pytest.approx(e.reward)
    assert e.reward_terms["raw_metrics"]["throughput_tps"] == pytest.approx(96.3)


def test_log_entry_carries_everything_the_analysis_needs():
    """Requirement: per-cycle timestamp, regime, arm, config, context, reward,
    reward components, throughput, TTFT, TPOT, realised batch, queue depth."""
    tr = _tracker([500.0, 900.0, 4000.0])
    c = CARLController(observe_interval=1, metrics=tr,
                       reward_fn=LiveRewardV2(_T4_SCALES))
    e = c.step(state=RuntimeState(queue_depth=7, avg_prompt_len=50.0,
                                  throughput_tps=80.0, batch_size_mean=6.5,
                                  p50_ttft_ms=900.0, p99_tpot_ms=120.0))
    assert e.t_monotonic_s > 0.0
    assert e.regime is not None
    assert isinstance(e.arm, int) and isinstance(e.rewarded_arm, int)
    assert e.config is not None
    assert len(e.state_features) == FEATURE_DIM
    assert e.reward_terms
    for key, want in (("throughput_tps", 80.0), ("tpot_p99_ms", 120.0),
                      ("realised_batch_mean", 6.5), ("queue_depth", 7)):
        assert e.observed[key] == want
    assert e.observed["ttft_p99_ms"] == pytest.approx(4000.0)


def test_delayed_reward_attribution_is_recorded_not_inferred():
    """`reward` at cycle t scores the arm chosen at t-1. Both indices are
    recorded so a trace never requires the reader to re-derive that."""
    from src.carl.bandit import PerRegimeBandit, RepairedLinUCBBandit
    from src.carl.config import all_arm_sets
    b = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                        bandit_cls=RepairedLinUCBBandit, alpha=0.5)
    # A WARM tracker. Attribution and the cold-start gate are separate
    # properties, and a controller with an empty tracker withholds its reward on
    # purpose (see tests/test_carl/test_reward_validity.py), so credit would
    # never be assigned and this test would pass for the wrong reason.
    c = CARLController(bandit=b, observe_interval=1, metrics=_tracker([1500.0]),
                       reward_fn=LiveRewardV2(_T4_SCALES))
    st = RuntimeState(queue_depth=3, avg_prompt_len=40.0, throughput_tps=70.0)
    first = c.step(state=st)
    second = c.step(state=st)
    assert first.rewarded_arm == -1, "nothing to credit on the first cycle"
    assert second.rewarded_arm == first.arm
