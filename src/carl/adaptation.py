"""
Offline adaptation analysis for CARL's decision history.

WHAT THIS IS (and what it deliberately is NOT)
----------------------------------------------
This is a PURE, POST-HOC analysis layer. It reads a CARLController's already-
recorded `controller_log` (a list of ControllerLogEntry) AFTER a run has
finished and turns it into:

  * a flat per-decision table (decision_rows)  -- one row per control cycle, and
  * a compact summary (summarize)              -- convergence, regret, switches.

It touches NONE of the serving hot path. The existing evaluation pipeline
(`_serve`, `ablation_live.py`, the result generators) stays byte-identical;
this module only consumes data those harnesses already keep in memory. That is
the whole reason it lives apart from them and imports nothing from `scripts/`.

GRANULARITY -- read this before interpreting a "cycle"
------------------------------------------------------
CARL does not decide per request. CARLController.step() runs once every
`observe_interval` scheduler steps, with many requests in flight per cycle. So
the unit of every row here is a CONTROL CYCLE (a bandit decision), not a
request. The request-level table (TTFT etc.) is a separate artifact built by
the driver from the serve loop's per-request records; this module is only about
the controller's decision dynamics.

TORCH-FREE / DETERMINISTIC
--------------------------
Everything here is plain arithmetic over duck-typed inputs:
  * a "log entry" is anything exposing `.step`, `.regime`, `.config`, `.reward`
    (ControllerLogEntry satisfies this; so do the synthetic stubs in the tests),
  * a "bandit" is anything exposing `.arms(regime) -> list[config-like]`,
  * a "config" is anything whose knob attributes can be read with getattr.
No numpy, no torch, no RNG -- given the same log it always returns the same rows.

REGRET MODEL (stated plainly so it can be defended)
---------------------------------------------------
The oracle is a STATIC best-arm-per-regime oracle: for each regime we are handed
the mean reward of the arm that was best in hindsight (the same quantity the
ablation's DynOracle computes from CARL-Full's recorded rewards). Per-cycle
regret is therefore

    instant_regret = max(0, oracle_reward[regime] - reward)

clamped at 0 because a single noisy realised reward of the chosen arm can exceed
the oracle's MEAN; clamping yields the conventional non-negative pseudo-regret
and keeps cumulative_regret monotonic (so a flat tail == converged, which is
exactly what the plots want to show).

EVENT MARKERS
-------------
Every row carries explicit boolean markers so a plotter never has to re-derive
them: `is_regime_transition`, `is_arm_switch`, `is_within_regime_switch`,
`is_convergence_point`. summarize() additionally exports the same events as
explicit lists (with from/to context) in its "events" block.
"""
from __future__ import annotations

import csv
import json
import os


# The full CARLConfig knob set, in as_dict() order. Read with getattr so any
# config-like object (real CARLConfig or a test stub) works; a missing knob
# simply records None rather than raising.
_KNOBS = [
    "max_batch_size",
    "chunk_size",
    "preemption_enabled",
    "spec_k",
    "routing_threshold",
    "cache_affinity_weight",
    "eviction_threshold",
    "eviction_window",
    "use_cuda_graphs",
]

# CSV column order for the per-decision table. Knobs are appended after the
# analysis columns so the markers stay near the metrics they annotate.
DECISION_COLUMNS = [
    "cycle",
    "step",
    "regime",
    "arm",
    "reward",
    "oracle_reward",
    "instant_regret",
    "cumulative_regret",
    "arm_changed",
    "is_regime_transition",
    "is_arm_switch",
    "is_within_regime_switch",
    "is_convergence_point",
] + _KNOBS


# ---------------------------------------------------------------------------
# Small duck-typed readers (kept module-level so the public funcs stay short).
# ---------------------------------------------------------------------------


def _regime_value(regime) -> str:
    """The string label of a regime, accepting either a WorkloadRegime enum
    (use .value) or a bare string (used by synthetic test logs)."""
    return regime.value if hasattr(regime, "value") else str(regime)


def _arm_index(arms: list, config) -> int:
    """Index of `config` within an arm list by value-equality; -1 if absent.

    CARLConfig is a dataclass (equality by field), and every logged config is
    one of the bandit's arms, so this recovers exactly the arm the bandit chose.
    -1 is the same sentinel the controller uses for an override config.
    """
    for i, a in enumerate(arms):
        if a == config:
            return i
    return -1


def _knob_values(config) -> dict:
    """The selected arm's active knobs as a flat dict (getattr, defaulting None)."""
    return {k: getattr(config, k, None) for k in _KNOBS}


def _oracle_for(oracle_reward_by_regime: dict, regime) -> float:
    """Oracle reward for `regime`, tolerant of enum- or string-keyed maps.

    Returns 0.0 when the regime has no oracle entry, which (after the regret
    clamp) makes its regret contribution zero rather than crashing -- a regime
    with no recorded oracle simply doesn't accrue regret.
    """
    if regime in oracle_reward_by_regime:
        return float(oracle_reward_by_regime[regime])
    val = _regime_value(regime)
    if val in oracle_reward_by_regime:
        return float(oracle_reward_by_regime[val])
    return 0.0


def _detect_convergence(rows: list) -> tuple:
    """(cycle, step, converged_within_window) for the controller's settle point.

    Convergence = the controller stops switching arms for the remainder of the
    log. Concretely the cycle of the LAST arm switch (the arm it lands on there
    is held to the end). Special cases:
      * no rows           -> (None, None, False)
      * no switches at all -> the first cycle, converged (it never moved)
      * last switch is the final cycle -> still switching at the end, NOT
        converged within the observed window.
    """
    if not rows:
        return (None, None, False)
    switches = [i for i, r in enumerate(rows) if r["is_arm_switch"]]
    if not switches:
        return (rows[0]["cycle"], rows[0]["step"], True)
    last = switches[-1]
    return (rows[last]["cycle"], rows[last]["step"], last < len(rows) - 1)


# ---------------------------------------------------------------------------
# decision_rows: the per-cycle table.
# ---------------------------------------------------------------------------


def decision_rows(controller_log: list, bandit, oracle_reward_by_regime: dict) -> list:
    """Flatten a controller log into per-decision rows (one per control cycle).

    Args:
        controller_log: ordered ControllerLogEntry-likes (`.step`, `.regime`,
            `.config`, `.reward`).
        bandit: exposes `.arms(regime) -> list`, used to recover the arm index
            of each logged config.
        oracle_reward_by_regime: {regime (enum or value) -> oracle mean reward},
            the static best-arm-per-regime oracle used for regret.

    Returns:
        A list of dicts with keys = DECISION_COLUMNS. Rows are fully annotated
        with event markers (regime transition / arm switch / within-regime
        switch / convergence point), so the CSV is plot-ready as written.
    """
    rows: list = []
    cum_regret = 0.0
    prev_regime_val = None
    prev_arm = None

    for cycle, entry in enumerate(controller_log):
        regime = entry.regime
        regime_val = _regime_value(regime)
        arm = _arm_index(bandit.arms(regime), entry.config)
        reward = float(entry.reward)
        oracle = _oracle_for(oracle_reward_by_regime, regime)

        instant_regret = max(0.0, oracle - reward)
        cum_regret += instant_regret

        is_regime_transition = prev_regime_val is not None and regime_val != prev_regime_val
        # An arm switch is a change of the (regime, arm) pair -- a regime flip or
        # a within-regime config change both count (matches the eval harnesses).
        is_arm_switch = prev_arm is not None and (regime_val, arm) != prev_arm
        is_within_regime_switch = is_arm_switch and not is_regime_transition

        row = {
            "cycle": cycle,
            "step": getattr(entry, "step", None) if getattr(entry, "step", None) is not None else cycle,
            "regime": regime_val,
            "arm": arm,
            "reward": reward,
            "oracle_reward": oracle,
            "instant_regret": instant_regret,
            "cumulative_regret": cum_regret,
            "arm_changed": is_arm_switch,
            "is_regime_transition": is_regime_transition,
            "is_arm_switch": is_arm_switch,
            "is_within_regime_switch": is_within_regime_switch,
            "is_convergence_point": False,
        }
        row.update(_knob_values(entry.config))
        rows.append(row)

        prev_regime_val = regime_val
        prev_arm = (regime_val, arm)

    # Stamp the single convergence marker so the CSV alone carries every event.
    conv_cycle, _conv_step, _converged = _detect_convergence(rows)
    if conv_cycle is not None:
        for r in rows:
            if r["cycle"] == conv_cycle:
                r["is_convergence_point"] = True
                break
    return rows


# ---------------------------------------------------------------------------
# summarize: the compact, plot-and-paper-ready digest.
# ---------------------------------------------------------------------------


def summarize(rows: list) -> dict:
    """Reduce per-decision rows to the adaptation summary.

    Returns a dict with:
        n_decisions
        convergence_point          {cycle, step, converged_within_window}
        total_cumulative_regret
        per_regime_cumulative_regret   {regime: sum of instant_regret}
        arm_switch_count
        unique_arm_count           distinct (regime, arm) pairs
        time_to_first_adaptation   {regime: cycles from regime entry to first
                                    WITHIN-regime arm switch, or None}
        final_arm_per_regime       {regime: arm of the last cycle in that regime}
        events                     explicit regime_transitions / arm_switches /
                                    convergence_point lists (so the plotter never
                                    has to infer a marker)
    """
    conv_cycle, conv_step, converged = _detect_convergence(rows)
    convergence_point = {
        "cycle": conv_cycle,
        "step": conv_step,
        "converged_within_window": converged,
    }

    if not rows:
        return {
            "n_decisions": 0,
            "convergence_point": convergence_point,
            "total_cumulative_regret": 0.0,
            "per_regime_cumulative_regret": {},
            "arm_switch_count": 0,
            "unique_arm_count": 0,
            "time_to_first_adaptation": {},
            "final_arm_per_regime": {},
            "events": {
                "regime_transitions": [],
                "arm_switches": [],
                "convergence_point": convergence_point,
            },
        }

    arm_switch_count = sum(1 for r in rows if r["is_arm_switch"])
    unique_arm_count = len({(r["regime"], r["arm"]) for r in rows})
    total_regret = rows[-1]["cumulative_regret"]

    # Per-regime aggregates, walking in order so "first/last" are well-defined.
    per_regime_regret: dict = {}
    final_arm: dict = {}
    first_cycle: dict = {}
    ttfa: dict = {}
    for r in rows:
        rv = r["regime"]
        per_regime_regret[rv] = per_regime_regret.get(rv, 0.0) + r["instant_regret"]
        final_arm[rv] = r["arm"]
        if rv not in first_cycle:
            first_cycle[rv] = r["cycle"]
        # First within-regime switch fixes the regime's time-to-first-adaptation.
        if rv not in ttfa and r["is_within_regime_switch"]:
            ttfa[rv] = r["cycle"] - first_cycle[rv]
    # Regimes that never adapted within-regime report None explicitly.
    for rv in first_cycle:
        ttfa.setdefault(rv, None)

    # Explicit event lists with from/to context.
    regime_transitions: list = []
    arm_switches: list = []
    for i, r in enumerate(rows):
        if r["is_regime_transition"]:
            regime_transitions.append({
                "cycle": r["cycle"], "step": r["step"],
                "from_regime": rows[i - 1]["regime"], "to_regime": r["regime"],
            })
        if r["is_arm_switch"]:
            arm_switches.append({
                "cycle": r["cycle"], "step": r["step"], "regime": r["regime"],
                "from_arm": rows[i - 1]["arm"], "to_arm": r["arm"],
            })

    return {
        "n_decisions": len(rows),
        "convergence_point": convergence_point,
        "total_cumulative_regret": total_regret,
        "per_regime_cumulative_regret": per_regime_regret,
        "arm_switch_count": arm_switch_count,
        "unique_arm_count": unique_arm_count,
        "time_to_first_adaptation": ttfa,
        "final_arm_per_regime": final_arm,
        "events": {
            "regime_transitions": regime_transitions,
            "arm_switches": arm_switches,
            "convergence_point": convergence_point,
        },
    }


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------


def write_decision_csv(rows: list, path: str) -> None:
    """Write per-decision rows to `path` as CSV (header = DECISION_COLUMNS)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in DECISION_COLUMNS})


def write_summary_json(summary: dict, path: str) -> None:
    """Write a summary dict (or aggregate of summaries) to `path` as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
