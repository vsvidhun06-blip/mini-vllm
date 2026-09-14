"""SLO accounting: units, polarity, statistical unit, and internal consistency.

Written against the specific contradictions found in the quarantined artifacts:
a 61-second TTFT p99 reported alongside `slo_violations: 0`, and a `slo_rate`
field that is a fraction in one row and a percentage in another.
"""
from __future__ import annotations

import pytest

from src.carl.slo import SLOTargets, account, check_consistency

T = SLOTargets(ttft_ms=200.0, tpot_ms=50.0)


def _r(ttft, tpot=10.0):
    return {"ttft_ms": ttft, "tpot_ms": tpot}


def test_meeting_the_deadline_exactly_is_satisfaction():
    assert T.satisfied(200.0, 50.0) is True
    assert T.satisfied(200.01, 50.0) is False


def test_missing_ttft_is_not_a_pass():
    """A request with no first token has not met the deadline. Treating a
    missing measurement as success is how dropped requests vanish from the
    accounting."""
    assert T.satisfied(None, 10.0) is False


def test_single_token_request_has_no_tpot_violation():
    assert T.satisfied(100.0, None) is True


def test_rates_are_fractions_and_pct_is_percent():
    blk = account([_r(100), _r(100), _r(9999), _r(9999)], T)
    assert blk["slo_satisfaction_rate"] == pytest.approx(0.5)
    assert blk["slo_satisfaction_pct"] == pytest.approx(50.0)
    assert blk["slo_violation_rate"] == pytest.approx(0.5)
    assert blk["slo_violations"] == 2
    assert blk["rate_convention"] == "fraction_0_to_1"


def test_statistical_unit_is_the_request():
    blk = account([_r(100)] * 9 + [_r(9999)], T)
    assert blk["accounting_unit"] == "request"
    assert blk["slo_satisfaction_rate"] == pytest.approx(0.9)


def test_unfinished_requests_are_reported_not_absorbed():
    blk = account([_r(100), _r(100)], T, n_submitted=10)
    assert blk["n_completed"] == 2
    assert blk["unfinished"] == 8
    # The satisfaction rate is over COMPLETED requests and must not silently
    # credit the 8 that never finished.
    assert blk["slo_satisfaction_rate"] == pytest.approx(1.0)


# --- the contradiction detector -------------------------------------------

def test_detects_the_reconstructed_artifact_contradiction():
    """ablation_live_results.RECONSTRUCTED.json: ttft_p99 61386ms, violations 0,
    target 200ms. check_consistency must refuse this."""
    bad = {"n_completed": 50, "slo_satisfaction_rate": 1.0,
           "slo_violations": 0, "slo_ttft_target_ms": 200.0}
    problems = check_consistency(bad, ttft_p99_ms=61386.1)
    assert problems, "must flag zero violations against a 61s p99"
    assert any("ttft_p99" in p for p in problems)


def test_detects_percentage_in_a_rate_field():
    """failure_cases_results: slo_rate_mean 100.0 where a fraction was meant."""
    bad = {"n_completed": 30, "slo_satisfaction_rate": 100.0,
           "slo_violations": 0, "slo_ttft_target_ms": 200.0}
    assert any("outside [0,1]" in p for p in check_consistency(bad))


def test_detects_count_rate_mismatch():
    bad = {"n_completed": 100, "slo_satisfaction_rate": 0.5,
           "slo_violations": 3, "slo_ttft_target_ms": 200.0}
    assert any("disagrees" in p for p in check_consistency(bad))


def test_a_well_formed_block_has_no_problems():
    blk = account([_r(100)] * 8 + [_r(9999)] * 2, T)
    assert check_consistency(blk, ttft_p99_ms=9999.0) == []
