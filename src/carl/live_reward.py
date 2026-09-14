"""Reward v2 for the LIVE controller, with frozen scales and a hard gate.

WHY THIS MODULE EXISTS
----------------------
`src/carl/reward.py` defines reward v2, but nothing on the live path ever used
it. `CARLController._reward_for_state` computes reward v1 (`bandit.utility`),
and v1 saturates on real hardware: in the 2026-09-02 T4 run every arm in every
regime scored exactly 0.29999999999999993 (see the `dynoracle.per_arm_mean_reward`
block of that run's artifact). The repair reached the simulation harness
(`scripts/eval/repair/*`) and never reached the controller.

This module is the missing adapter. It does four things and nothing else:

  1. Maps an observed `RuntimeState` + `MetricsTracker` onto the RAW metric dict
     `utility_v2` expects (v2 takes raw tok/s and raw milliseconds, because
     normalisation is precisely the thing being fixed and must not be done by
     the caller).
  2. Holds the `RewardScales` FROZEN. They are passed in, never derived here.
  3. Reports the per-term breakdown alongside the scalar, so a trace can show
     WHICH term went flat rather than only that the total did.
  4. REFUSES TO SCORE UNMEASURED METRICS. Until the tracker holds a real TTFT
     and TPOT observation, the reward is None and the learner is not updated.
     See the COLD START section of `LiveRewardV2` for what this repairs.

THE CALIBRATION RULE (non-negotiable)
-------------------------------------
`RewardScales` must come from held-out calibration measurements taken BEFORE
evaluation and frozen. Deriving them from the evaluation runs would let the
yardstick move with the thing being measured: `t_half` set to the median of the
test runs' own throughput guarantees a 0.5 mid-point by construction, which
manufactures discrimination instead of measuring it. This module therefore has
no `from_measurements` path and no default scales that silently apply --
`scales` is a required constructor argument and must already be frozen.

TTFT: p99, NOT p50
------------------
`utility_v2` scores `ttft_p99_ms`, so this adapter feeds it the p99. Note that
`RuntimeState` exposes `p50_ttft_ms` (it is a *context feature*, chosen for the
bandit's benefit) and NOT a TTFT p99 -- so the p99 is read from the
`MetricsTracker` directly. The context feature vector is deliberately untouched:
changing `state._FEATURE_SCALES` would change `FEATURE_DIM` and the bandit's
input dimension, which would break the controlled pair against the reward-v1 run.
"""
from __future__ import annotations

from src.carl.reward import (
    DEFAULT_WEIGHTS, DegenerateRewardError, RewardScales, term_breakdown,
    utility_v2,
)

REWARD_VERSION = "v2"

# ---------------------------------------------------------------------------
# Documented non-degeneracy thresholds.
# ---------------------------------------------------------------------------
#
# These are NOT round numbers. They are read off the per-arm reward spreads
# measured on the repaired, calibrated simulation substrate, where the reward is
# known to be non-degenerate -- docs/eval/reward_diagnostics.md, 8 workloads x
# 30 arms:
#
#     interactive 0.1838 | batch 0.1148 | long_context 0.0219
#     interactive_heavy 0.1026 | batch_heavy 0.1000 | long_context_heavy 0.1463
#     burst_dump 0.0981 | overload 0.1032
#
# The observed FLOOR across all eight is 0.0219 (long_context). A run whose
# candidate arms spread less than that is outside the range in which v2 has ever
# been demonstrated to discriminate, so:
#
#   MIN_ARM_REWARD_SPREAD = 0.02 -- applied to the CALIBRATION sweep, which
#       evaluates the full candidate axis. Sits just under the observed floor,
#       so any substrate on which v2 has been shown to work passes, and a
#       saturated one (spread ~1e-16, as reward v1 produced on the T4) fails by
#       many orders of magnitude. There is no calibrated setting of this
#       constant that would let the observed v1 failure through.
#
#   MIN_LIVE_REWARD_SPREAD = 0.01 -- applied to a SINGLE run's reward stream.
#       Half the calibration floor, because one episode visits a subset of the
#       candidate axis and legitimately spans less of it.
#
#   MIN_LIVE_DISTINCT = 3 -- reward v1 produced exactly TWO distinct values in
#       docs/eval/raw/adaptation/decisions_042.csv (0.8 then 0.3) and ONE in the
#       2026-09-02 run. Requiring three makes both historical failures fail this
#       gate.
# ---------------------------------------------------------------------------

MIN_ARM_REWARD_SPREAD = 0.02
MIN_LIVE_REWARD_SPREAD = 0.01
MIN_LIVE_DISTINCT = 3

THRESHOLD_PROVENANCE = (
    "Thresholds derived from the measured per-arm reward spread on the repaired "
    "calibrated substrate (docs/eval/reward_diagnostics.md, 8 workloads x 30 "
    "arms, spreads 0.0219-0.1838). MIN_ARM_REWARD_SPREAD sits just below the "
    "observed floor of 0.0219; MIN_LIVE_REWARD_SPREAD is half of it because a "
    "single episode visits a subset of the candidate axis."
)


class LiveRewardV2:
    """Callable reward v2 with frozen scales, for injection into CARLController.

    Usage:
        reward_fn = LiveRewardV2(scales=frozen_scales)
        CARLController(..., reward_fn=reward_fn)

    Call signature is `(state, metrics) -> (reward, terms)`, which is the
    contract `CARLController._reward_for_state` uses when `reward_fn` is set.
    `reward` is None when the reward could not be computed from MEASURED data;
    see COLD START below.

    COLD START -- MISSING LATENCY IS NOT ZERO LATENCY
    -------------------------------------------------
    The reward-v2 T4 artifact shows, in EVERY seed, that the first three control
    cycles logged `ttft_p99_ms = 0` and `tpot_p99_ms = 0`. No request had
    completed yet, so the tracker's windows were empty and `_percentile`
    returned its NaN-free 0.0. Reward v2 then scored those zeros through
    `latency_term`, which maps 0 ms to 1.0 -- PERFECT latency -- and awarded the
    full TTFT and TPOT weight (0.5 of the total) for latencies that had never
    been observed. Those rewards were handed to LinUCB, which credits them to
    the previous cycle's arm; the first arm played is arm 0 in every regime by
    construction, so the bias is directional and inflates the hand-tuned default
    before any evidence exists.

    The fix is semantic, not numerical. There is no correct fill value: zero
    fabricates a perfect measurement, a large penalty fabricates a bad one, and
    the mid-point fabricates an average one. So this adapter refuses to produce
    a number at all until every metric the reward consumes is MEASURED:

        reward_valid = False  ->  reward is None  ->  the learner is not updated

    `require_valid_metrics=False` restores the ORIGINAL, optimistic behaviour
    verbatim. It exists for exactly one purpose: `reward_v2_live.py` must remain
    able to reproduce the artifact that exposed this bug. Nothing else should
    set it.
    """

    version = REWARD_VERSION

    def __init__(self, scales: RewardScales, weights: dict | None = None, *,
                 require_valid_metrics: bool = True,
                 min_latency_samples: int = 1) -> None:
        if not isinstance(scales, RewardScales):
            raise TypeError(
                "LiveRewardV2 requires an explicit frozen RewardScales; scales "
                "must be calibrated on held-out measurements before evaluation, "
                "never derived from the test runs.")
        if min_latency_samples < 1:
            raise ValueError(
                "min_latency_samples must be >= 1: a reward computed from zero "
                "observations is the bug this parameter exists to prevent.")
        self.scales = scales
        self.weights = weights or DEFAULT_WEIGHTS
        self.require_valid_metrics = bool(require_valid_metrics)
        self.min_latency_samples = int(min_latency_samples)

    def raw_metrics(self, state, metrics) -> dict:
        """The RAW metric dict `utility_v2` consumes.

        `ttft_p99_ms` is read from the MetricsTracker because RuntimeState
        carries only the p50 (a context feature). Falls back to the state's p50
        only when no tracker is available, which happens in unit tests that
        drive the controller with synthetic states.
        """
        ttft_p99 = None
        if metrics is not None and hasattr(metrics, "p99_ttft_ms"):
            ttft_p99 = metrics.p99_ttft_ms()
        if not ttft_p99:
            ttft_p99 = getattr(state, "p50_ttft_ms", 0.0)
        return {
            "throughput_tps": getattr(state, "throughput_tps", 0.0),
            "ttft_p99_ms": ttft_p99,
            "tpot_p99_ms": getattr(state, "p99_tpot_ms", 0.0),
            "cache_hit_rate": getattr(state, "cache_hit_rate", 0.0),
        }

    def validity(self, metrics) -> dict:
        """Which reward metrics are measured, from the tracker's sample counts.

        A tracker that predates this change (or a duck-typed stub in a unit test)
        has no `metric_validity`. Those callers drive the controller with
        synthetic states rather than a live engine, so there is no cold start to
        protect them from, and treating them as valid is the behaviour-preserving
        choice. The record says so explicitly rather than leaving a reader to
        guess why `all_valid` is True with no counts behind it.
        """
        if metrics is not None and hasattr(metrics, "metric_validity"):
            record = metrics.metric_validity(self.min_latency_samples)
            record["source"] = "MetricsTracker.metric_validity"
            return record
        return {
            "min_samples_required": self.min_latency_samples,
            "valid": {}, "missing": [], "all_valid": True,
            "source": "no tracker (synthetic state); validity not observable",
        }

    def __call__(self, state, metrics) -> tuple:
        """(reward, terms). `reward` is None when the metrics are not measured.

        The per-term breakdown is returned EVEN WHEN INVALID, with
        `reward_valid=False` next to it, so a trace records what the reward
        WOULD have been and why it was refused. That is what makes the
        cold-start optimism auditable in the artifact instead of merely absent
        from it -- `would_be_reward` is diagnostic output, is clearly labelled,
        and is never fed to the learner.
        """
        m = self.raw_metrics(state, metrics)
        validity = self.validity(metrics)
        terms = term_breakdown(m, self.weights, self.scales)
        terms["raw_metrics"] = m
        terms["reward_validity"] = validity

        if self.require_valid_metrics and not validity["all_valid"]:
            terms["reward_valid"] = False
            terms["would_be_reward"] = utility_v2(m, self.weights, self.scales)
            terms["invalid_reason"] = (
                "no measured observation yet for: "
                + ", ".join(validity["missing"])
                + ". Missing latency is NOT zero latency; the reward is withheld "
                  "and the learner is not updated.")
            return None, terms

        terms["reward_valid"] = True
        return utility_v2(m, self.weights, self.scales), terms

    def as_dict(self) -> dict:
        """Exactly what was frozen, for the provenance block."""
        return {
            "reward_version": self.version,
            "weights": dict(self.weights),
            "scales": {
                "t_half": self.scales.t_half,
                "ttft_target": self.scales.ttft_target,
                "tpot_target": self.scales.tpot_target,
                "sharpness": self.scales.sharpness,
            },
            "require_valid_metrics": self.require_valid_metrics,
            "min_latency_samples": self.min_latency_samples,
            "cold_start_policy": (
                "reward withheld (None) until every consumed metric has a "
                "measured observation; the learner is not updated on a withheld "
                "reward and no fill value is substituted"
                if self.require_valid_metrics else
                "NONE -- missing latency percentiles are scored as 0 ms, i.e. "
                "as PERFECT latency. This is the pre-repair behaviour and is "
                "retained only to reproduce the artifact that exposed it."),
        }


# ---------------------------------------------------------------------------
# The hard gates.
# ---------------------------------------------------------------------------


def spread(values) -> float:
    """max - min, or 0.0 for an empty sequence."""
    vals = [float(v) for v in values]
    return (max(vals) - min(vals)) if vals else 0.0


def gate_candidate_arm_spread(arm_rewards: dict, *, context: str = "",
                              threshold: float = MIN_ARM_REWARD_SPREAD) -> dict:
    """ABORT the experiment unless the reward separates the candidate arms.

    This is the PRE-RUN gate. `arm_rewards` maps a candidate label to the reward
    that candidate earned on the held-out calibration sweep. If the spread is
    below `threshold`, the reward cannot distinguish the operating points the
    controller will choose among, so no learning result from the run could mean
    anything -- and we stop BEFORE spending GPU hours rather than discovering it
    afterwards, which is what happened on 2026-09-02.

    Raises:
        DegenerateRewardError, carrying the full per-candidate table.
    """
    vals = list(arm_rewards.values())
    s = spread(vals)
    report = {
        "n_candidates": len(vals),
        "reward_min": min(vals) if vals else None,
        "reward_max": max(vals) if vals else None,
        "reward_spread": s,
        "threshold": threshold,
        "per_candidate": dict(arm_rewards),
        "passed": bool(vals) and s >= threshold,
        "threshold_provenance": THRESHOLD_PROVENANCE,
    }
    if not report["passed"]:
        raise DegenerateRewardError(
            "{}: candidate-arm reward spread {:.6g} < threshold {} over {} "
            "candidates {!r}. The reward cannot distinguish the configurations "
            "the controller selects among; ABORTING before evaluation. {}".format(
                context, s, threshold, len(vals), arm_rewards,
                THRESHOLD_PROVENANCE))
    return report


def gate_live_reward_stream(rewards, *, context: str = "",
                            threshold: float = MIN_LIVE_REWARD_SPREAD,
                            min_distinct: int = MIN_LIVE_DISTINCT) -> dict:
    """ABORT unless one run's reward stream actually varied.

    This is the PER-RUN gate, applied to the rewards the controller logged. It
    catches the failure the pre-run gate cannot: scales that discriminate across
    the calibration sweep but collapse at the operating point an individual run
    happens to reach.
    """
    vals = [float(r) for r in rewards]
    distinct = sorted(set(vals))
    s = spread(vals)
    report = {
        "n": len(vals),
        "n_distinct": len(distinct),
        "reward_spread": s,
        "reward_min": min(vals) if vals else None,
        "reward_max": max(vals) if vals else None,
        "threshold": threshold,
        "min_distinct": min_distinct,
        "distinct_sample": distinct[:10],
        "passed": bool(vals) and s >= threshold and len(distinct) >= min_distinct,
        "threshold_provenance": THRESHOLD_PROVENANCE,
    }
    if not report["passed"]:
        raise DegenerateRewardError(
            "{}: live reward stream is degenerate -- spread {:.6g} (threshold "
            "{}), {} distinct value(s) {} (minimum {}) over {} cycles. The "
            "controller could not distinguish configurations; ABORTING. {}".format(
                context, s, threshold, len(distinct), distinct[:5], min_distinct,
                len(vals), THRESHOLD_PROVENANCE))
    return report
