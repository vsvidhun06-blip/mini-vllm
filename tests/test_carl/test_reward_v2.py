"""Reward v2: non-degeneracy, monotonicity, and agreement with serving utility.

These tests encode the properties whose absence made reward v1 unusable. v1
would fail every one of them at the operating point the project actually ran at.
"""
from __future__ import annotations

import pytest

from src.carl.bandit import utility as utility_v1
from src.carl.reward import (
    DegenerateRewardError, RewardScales, check_non_degenerate, latency_term,
    term_breakdown, throughput_term, utility_v2,
)

SCALES = RewardScales(t_half=60.0, ttft_target=2000.0, tpot_target=60.0)


def _m(tps=60.0, ttft=2000.0, tpot=60.0, cache=0.0):
    return {"throughput_tps": tps, "ttft_p99_ms": ttft,
            "tpot_p99_ms": tpot, "cache_hit_rate": cache}


# --- non-saturation --------------------------------------------------------

def test_throughput_term_never_saturates():
    """The v1 defect: min(1, tps/50) is EXACTLY 1.0 for every tps >= 50, so all
    such configs are indistinguishable. v2 must keep a positive gradient."""
    prev = throughput_term(0.0, SCALES)
    for tps in (10, 50, 85, 200, 1000, 10_000):
        cur = throughput_term(tps, SCALES)
        assert cur > prev, f"not strictly increasing at {tps}"
        assert cur < 1.0, f"saturated at {tps}"
        prev = cur


def test_latency_term_distinguishes_configs_that_both_miss_the_slo():
    """v1's violation RATE pinned at 1.0 once every request missed, making a 2x
    miss identical to a 10x miss. That is the exact GPU failure mode."""
    at_target = latency_term(2000.0, 2000.0)
    assert at_target == pytest.approx(0.5), "target must be the 0.5 crossing"
    two_x = latency_term(4000.0, 2000.0)
    ten_x = latency_term(20000.0, 2000.0)
    assert two_x > ten_x > 0.0
    assert two_x != ten_x


def test_v1_is_degenerate_at_the_measured_gpu_operating_point():
    """Reproduces the documented failure: at the real T4 operating point every
    v1 term pins, so distinct configurations score identically."""
    # throughput 85 and 200 tok/s; both above throughput_ref=50 -> both clip to 1.
    a = utility_v1({"throughput_norm": min(1.0, 85 / 50), "ttft_violation_rate": 1.0,
                    "tpot_violation_rate": 1.0, "cache_hit_rate": 0.0})
    b = utility_v1({"throughput_norm": min(1.0, 200 / 50), "ttft_violation_rate": 1.0,
                    "tpot_violation_rate": 1.0, "cache_hit_rate": 0.0})
    assert a == b == pytest.approx(0.3), "v1 collapses to a constant 0.3"

    # v2 must separate them.
    a2 = utility_v2(_m(tps=85, ttft=2200, tpot=120), None, SCALES)
    b2 = utility_v2(_m(tps=200, ttft=2200, tpot=120), None, SCALES)
    assert b2 > a2


# --- monotonicity ----------------------------------------------------------

@pytest.mark.parametrize("field,better,worse", [
    ("throughput_tps", 120.0, 60.0),
    ("ttft_p99_ms", 500.0, 5000.0),
    ("tpot_p99_ms", 30.0, 300.0),
    ("cache_hit_rate", 0.9, 0.1),
])
def test_each_term_is_strictly_monotone(field, better, worse):
    good, bad = _m(), _m()
    good[field], bad[field] = better, worse
    assert utility_v2(good, None, SCALES) > utility_v2(bad, None, SCALES)


def test_ordering_agrees_with_serving_utility():
    """Phase 3 requirement: rank a poor point, a baseline point and the best
    measured point in the order an operator would."""
    poor = _m(tps=30, ttft=9000, tpot=300)
    baseline = _m(tps=60, ttft=2000, tpot=60)
    best = _m(tps=95, ttft=300, tpot=35)
    r = [utility_v2(x, None, SCALES) for x in (poor, baseline, best)]
    assert r[0] < r[1] < r[2], f"ordering violated: {r}"


def test_bounded():
    assert 0.0 < utility_v2(_m(tps=1e6, ttft=1e-6, tpot=1e-6, cache=1.0), None, SCALES) <= 1.0
    assert utility_v2(_m(tps=0, ttft=1e9, tpot=1e9, cache=0.0), None, SCALES) >= 0.0


def test_term_breakdown_sums_to_total():
    tb = term_breakdown(_m(tps=85, ttft=1500, tpot=45, cache=0.3), None, SCALES)
    parts = (tb["throughput_contrib"] + tb["ttft_contrib"]
             + tb["tpot_contrib"] + tb["cache_contrib"])
    assert parts == pytest.approx(tb["total"])


# --- degeneracy detection --------------------------------------------------

def test_check_non_degenerate_raises_on_constant_reward():
    with pytest.raises(DegenerateRewardError, match="degenerate"):
        check_non_degenerate([0.3] * 20, context="synthetic")


def test_check_non_degenerate_raises_on_the_real_gpu_trace():
    """The actual reward sequence from docs/eval/raw/adaptation/decisions_042.csv."""
    observed = [0.8, 0.8, 0.8] + [0.3] * 8
    with pytest.raises(DegenerateRewardError):
        check_non_degenerate(observed, context="decisions_042", min_distinct=3)


def test_check_non_degenerate_passes_on_a_real_spread():
    rep = check_non_degenerate([0.51, 0.55, 0.62, 0.58, 0.49], context="ok")
    assert rep["degenerate"] is False
    assert rep["n_distinct"] == 5


def test_scales_must_be_positive():
    with pytest.raises(ValueError):
        RewardScales(t_half=0.0)
