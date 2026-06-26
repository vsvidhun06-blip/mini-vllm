"""
Unit tests for the pure adaptation-analysis layer (src/carl/adaptation.py).

These are deterministic and torch-free: we hand-build synthetic controller logs
(no GPU, no serving) and assert the per-decision rows and the summary. Five
scenarios exercise the convergence / regret / event-marker logic:

  1. immediate convergence  -- one arm per regime, settles at the regime flip
  2. delayed convergence    -- explores, then locks onto a final arm
  3. oscillation            -- alternates arms to the very end (never settles)
  4. no arm changes         -- a single constant arm throughout
  5. multiple regime transitions -- regime flips several times

A "log entry" only needs .step/.regime/.config/.reward, and a "bandit" only
needs .arms(regime); we use the real WorkloadRegime enum and real CARLConfig
arms so arm-index recovery is exercised exactly as in production.
"""
from dataclasses import dataclass

from src.carl.adaptation import decision_rows, summarize
from src.carl.config import CARLConfig
from src.carl.state import WorkloadRegime

I = WorkloadRegime.INTERACTIVE
B = WorkloadRegime.BATCH


# --- synthetic fixtures -----------------------------------------------------


@dataclass
class _Entry:
    """Minimal ControllerLogEntry stand-in."""
    step: int
    regime: WorkloadRegime
    config: CARLConfig
    reward: float


# Distinct arms per regime: index i == a config with max_batch_size encoding i,
# so _arm_index recovers the intended index by value-equality.
_ARMS = {
    I: [CARLConfig(max_batch_size=mb) for mb in (1, 2, 3, 4)],
    B: [CARLConfig(max_batch_size=mb, chunk_size=512) for mb in (1, 2, 3, 4)],
}


class _FakeBandit:
    def arms(self, regime):
        return _ARMS[regime]


def _log(seq):
    """seq: list of (regime, arm_index, reward) -> list[_Entry] at cadence 10."""
    return [
        _Entry(step=i * 10, regime=rg, config=_ARMS[rg][arm], reward=rew)
        for i, (rg, arm, rew) in enumerate(seq)
    ]


def _rows(seq, oracle=None):
    oracle = oracle if oracle is not None else {"interactive": 1.0, "batch": 1.0}
    return decision_rows(_log(seq), _FakeBandit(), oracle)


# --- arm-index recovery + regret math --------------------------------------


def test_arm_index_and_regret_math():
    # Two interactive cycles: arm 2 then arm 3; rewards 0.6 and 1.2 (clamps).
    rows = _rows([(I, 2, 0.6), (I, 3, 1.2)], oracle={"interactive": 1.0})
    assert [r["arm"] for r in rows] == [2, 3]
    # instant_regret = max(0, oracle - reward): 0.4 then clamped to 0.0.
    assert rows[0]["instant_regret"] == 0.4
    assert rows[1]["instant_regret"] == 0.0
    # cumulative is the running (monotonic) sum.
    assert rows[0]["cumulative_regret"] == 0.4
    assert rows[1]["cumulative_regret"] == 0.4
    # knobs of the selected arm are exported.
    assert rows[0]["max_batch_size"] == 3   # _ARMS[I][2] -> mb=3
    # step is carried from the entry.
    assert [r["step"] for r in rows] == [0, 10]


def test_unknown_arm_is_minus_one():
    # A config not in the arm set -> arm index -1 (override sentinel).
    log = [_Entry(step=0, regime=I, config=CARLConfig(max_batch_size=99), reward=0.5)]
    rows = decision_rows(log, _FakeBandit(), {"interactive": 1.0})
    assert rows[0]["arm"] == -1


# --- scenario 1: immediate convergence -------------------------------------


def test_immediate_convergence():
    # arm 1 for 3 interactive cycles, then arm 0 for 3 batch cycles.
    seq = [(I, 1, 0.9)] * 3 + [(B, 0, 0.9)] * 3
    rows = _rows(seq)
    s = summarize(rows)
    # The only (regime,arm) change is the regime flip at cycle 3.
    assert s["arm_switch_count"] == 1
    assert s["convergence_point"]["cycle"] == 3
    assert s["convergence_point"]["converged_within_window"] is True
    # No WITHIN-regime exploration in either regime.
    assert s["time_to_first_adaptation"] == {"interactive": None, "batch": None}
    assert s["final_arm_per_regime"] == {"interactive": 1, "batch": 0}
    assert s["unique_arm_count"] == 2
    # One regime transition event, exported with from/to context.
    assert len(s["events"]["regime_transitions"]) == 1
    assert s["events"]["regime_transitions"][0]["from_regime"] == "interactive"
    assert s["events"]["regime_transitions"][0]["to_regime"] == "batch"
    # Convergence marker present on exactly one row in the CSV-bound rows.
    assert sum(1 for r in rows if r["is_convergence_point"]) == 1


# --- scenario 2: delayed convergence ---------------------------------------


def test_delayed_convergence():
    # Explore 0->1->2->2->3 then hold 3: last switch at cycle 4, stable after.
    seq = [(I, a, 0.8) for a in (0, 1, 2, 2, 3, 3, 3, 3)]
    rows = _rows(seq)
    s = summarize(rows)
    assert s["arm_switch_count"] == 3                  # cycles 1, 2, 4
    assert s["convergence_point"]["cycle"] == 4
    assert s["convergence_point"]["converged_within_window"] is True
    # First within-regime switch is at cycle 1 (entered at cycle 0).
    assert s["time_to_first_adaptation"]["interactive"] == 1
    assert s["final_arm_per_regime"]["interactive"] == 3
    assert len(s["events"]["arm_switches"]) == 3


# --- scenario 3: oscillation -----------------------------------------------


def test_oscillation_never_converges():
    # Alternate 0/1 every cycle, ending on a switch -> not converged in-window.
    seq = [(I, a, 0.5) for a in (0, 1, 0, 1, 0, 1)]
    rows = _rows(seq)
    s = summarize(rows)
    assert s["arm_switch_count"] == 5                  # every cycle after the 1st
    assert s["convergence_point"]["cycle"] == 5        # last switch == final cycle
    assert s["convergence_point"]["converged_within_window"] is False
    assert s["unique_arm_count"] == 2
    assert s["time_to_first_adaptation"]["interactive"] == 1


# --- scenario 4: no arm changes --------------------------------------------


def test_no_arm_changes():
    # A single constant arm: zero switches, immediate convergence, no adaptation.
    seq = [(B, 0, 0.7)] * 5
    rows = _rows(seq)
    s = summarize(rows)
    assert s["arm_switch_count"] == 0
    assert s["convergence_point"]["cycle"] == 0
    assert s["convergence_point"]["converged_within_window"] is True
    assert s["time_to_first_adaptation"] == {"batch": None}
    assert s["final_arm_per_regime"] == {"batch": 0}
    assert s["events"]["regime_transitions"] == []
    assert s["events"]["arm_switches"] == []
    # Regret still accrues against the oracle even with no switching.
    # oracle 1.0, reward 0.7 -> 0.3 per cycle * 5 = 1.5.
    assert abs(s["total_cumulative_regret"] - 1.5) < 1e-9
    assert abs(s["per_regime_cumulative_regret"]["batch"] - 1.5) < 1e-9


# --- scenario 5: multiple regime transitions -------------------------------


def test_multiple_regime_transitions():
    # regimes I,I,B,B,I,B with arms 0,0,1,2,0,2. The batch block at cycles 2-3
    # has a genuine WITHIN-regime change (1->2); the cross-block changes are
    # regime re-entries, which must NOT count as within-regime adaptation.
    seq = [(I, 0, 0.9), (I, 0, 0.9), (B, 1, 0.9), (B, 2, 0.9),
           (I, 0, 0.9), (B, 2, 0.9)]
    rows = _rows(seq)
    s = summarize(rows)
    # Transitions at cycles 2 (I->B), 4 (B->I), 5 (I->B).
    assert len(s["events"]["regime_transitions"]) == 3
    assert [t["cycle"] for t in s["events"]["regime_transitions"]] == [2, 4, 5]
    # Arm switches: 2 (regime+arm), 3 (within-regime 1->2), 4 (regime),
    # 5 (regime) -> 4 total.
    assert s["arm_switch_count"] == 4
    # Still switching at the final cycle -> not converged in-window.
    assert s["convergence_point"]["cycle"] == 5
    assert s["convergence_point"]["converged_within_window"] is False
    # Interactive never changes its arm within-regime -> None. Batch's first
    # within-regime change is at cycle 3 and it entered at cycle 2 -> ttfa 1.
    assert s["time_to_first_adaptation"]["interactive"] is None
    assert s["time_to_first_adaptation"]["batch"] == 1
    # Final arm per regime: interactive's last is cycle 4 (arm 0); batch's
    # last is cycle 5 (arm 2).
    assert s["final_arm_per_regime"] == {"interactive": 0, "batch": 2}
    # Distinct (regime, arm) pairs: (I,0), (B,1), (B,2).
    assert s["unique_arm_count"] == 3


# --- edge case: empty log ---------------------------------------------------


def test_empty_log():
    rows = decision_rows([], _FakeBandit(), {})
    assert rows == []
    s = summarize(rows)
    assert s["n_decisions"] == 0
    assert s["arm_switch_count"] == 0
    assert s["convergence_point"]["cycle"] is None
    assert s["convergence_point"]["converged_within_window"] is False
    assert s["events"]["arm_switches"] == []
