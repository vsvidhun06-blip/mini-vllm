"""Reward v2 -- a non-saturating multi-objective serving utility.

WHY v1 HAD TO BE REPLACED
-------------------------
`bandit.utility()` (v1) computes

    u = 0.3 * min(1, tps / 50)
      + 0.3 * (1 - ttft_violation_rate)
      + 0.2 * (1 - tpot_violation_rate)
      + 0.2 * cache_hit_rate

Every one of those four terms is a HARD CLIP or an INDICATOR RATE, and on the
project's own T4 runs all four pinned simultaneously. From the committed trace
`docs/eval/raw/adaptation/decisions_042.csv`, the reward takes exactly two values
across an entire run -- 0.8 for three cycles, then 0.3 forever:

    0.3 = 0.3*(1.0)      <- tps 85 >= throughput_ref 50, clipped to 1
        + 0.3*(1 - 1.0)  <- TTFT ~2200ms vs 200ms SLO: 100% violation
        + 0.2*(1 - 1.0)  <- TPOT vs 50ms SLO: 100% violation
        + 0.2*(0.0)      <- no prefix reuse in the synthetic workload

A bandit cannot learn from a constant. Worse, the degeneracy is invisible in the
aggregate: mean reward looks fine, and only the per-cycle trace reveals that no
configuration is distinguishable from any other.

THE FIX, AND WHY IT IS NOT TUNING
---------------------------------
v2 keeps the same four objectives, the same weights, and the same monotonicity
(more throughput is better; more latency is worse). It changes only the SHAPE of
each term, from clipped/indicator to smooth-and-strictly-monotone:

  throughput   tps / (tps + T_half)          Michaelis-Menten. 0.5 at T_half,
                                             asymptotes to 1 but NEVER reaches it,
                                             so the gradient is positive for all
                                             finite tps. No clipping.

  ttft, tpot   1 / (1 + (x / target)^2)      Smooth relaxation of the indicator
                                             1[x <= target]. Equals 0.5 exactly
                                             at the SLO target, is strictly
                                             decreasing, and is never exactly 0,
                                             so a config that misses the SLO by
                                             2x is still distinguishable from one
                                             that misses it by 10x. Under v1 both
                                             scored identically (violation = 1.0).

  cache        hit_rate                      Unchanged; already in [0,1] and
                                             already non-degenerate when the
                                             workload actually reuses prefixes.

Two properties matter and are asserted by tests in
`tests/test_carl/test_reward_v2.py`:

  1. STRICT MONOTONICITY. Improving any single metric strictly increases the
     reward, at every operating point. There is no region where the reward is
     flat, which is exactly the failure v1 exhibited.
  2. ORDERING AGREES WITH SERVING UTILITY. On a deliberately chosen triple --
     a poor point, a baseline point, and the best measured point -- the reward
     ranks them in the order an operator would.

WHAT IS *NOT* CLAIMED
---------------------
v2 does not make CARL win. It makes configurations DISTINGUISHABLE. Whether the
bandit then finds a better one than a static rule is the open question that
`scripts/eval/repair/rule_only_ablation.py` answers, and v2 was fixed before that
experiment was run precisely so the answer would not be a function of the reward
design.

CHOOSING THE SCALES
-------------------
`T_half` must be set to the middle of the ACHIEVABLE operating range for the
(model, hardware, workload) under test -- not to a round number. Setting it far
below the operating range reproduces v1's saturation in smooth clothing; setting
it far above compresses all configs near 0. `RewardScales.from_measurements()`
derives it from an observed throughput range, which is the intended path once
`batch_intervention.py` has run on a GPU.

`ttft_target` / `tpot_target` remain genuine SLOs -- they are the point where the
term crosses 0.5, so they keep their operational meaning.
"""
from __future__ import annotations

from dataclasses import dataclass

REWARD_VERSION = "v2"

# Same weights as v1: the objective did not change, only the shape of the terms.
DEFAULT_WEIGHTS = {
    "throughput": 0.3,
    "ttft": 0.3,
    "tpot": 0.2,
    "cache": 0.2,
}


@dataclass(frozen=True)
class RewardScales:
    """Normalisation scales for reward v2. All must be > 0.

    t_half:      throughput (tok/s) at which the throughput term scores 0.5.
                 Set to the midpoint of the achievable range for this setup.
    ttft_target: TTFT (ms) at which the TTFT term scores 0.5 -- i.e. the SLO.
    tpot_target: TPOT (ms) at which the TPOT term scores 0.5 -- i.e. the SLO.
    sharpness:   exponent on the latency terms. 2.0 gives a gentle knee; larger
                 values approach the v1 hard indicator (and reintroduce
                 saturation), so raising it above ~4 defeats the purpose.
    """

    t_half: float = 60.0
    ttft_target: float = 2000.0
    tpot_target: float = 60.0
    sharpness: float = 2.0

    def __post_init__(self) -> None:
        for name in ("t_half", "ttft_target", "tpot_target", "sharpness"):
            if getattr(self, name) <= 0:
                raise ValueError(f"RewardScales.{name} must be > 0")

    @classmethod
    def from_measurements(cls, throughput_samples, ttft_p99_samples,
                          tpot_p99_samples, **kw) -> "RewardScales":
        """Derive scales from an observed operating range.

        t_half is the MEDIAN of observed throughput, so roughly half the
        candidate configurations score above 0.5 and half below -- the maximum
        discriminative setting, and the opposite of picking a round number that
        happens to sit outside the range (v1's failure).

        The latency targets default to the observed medians too, so that they
        also sit inside the range rather than being unreachable constants. A
        caller with a real contractual SLO should pass it explicitly instead.
        """
        def _median(xs):
            s = sorted(float(x) for x in xs if x is not None)
            if not s:
                return None
            return s[len(s) // 2]

        t = _median(throughput_samples)
        ttft = _median(ttft_p99_samples)
        tpot = _median(tpot_p99_samples)
        return cls(
            t_half=kw.pop("t_half", t if t and t > 0 else cls.t_half),
            ttft_target=kw.pop("ttft_target", ttft if ttft and ttft > 0 else cls.ttft_target),
            tpot_target=kw.pop("tpot_target", tpot if tpot and tpot > 0 else cls.tpot_target),
            **kw,
        )


DEFAULT_SCALES = RewardScales()


def throughput_term(tps: float, scales: RewardScales = DEFAULT_SCALES) -> float:
    """tps / (tps + t_half). Strictly increasing on [0, inf), never reaches 1."""
    tps = max(0.0, float(tps))
    return tps / (tps + scales.t_half)


def latency_term(value_ms: float, target_ms: float, sharpness: float = 2.0) -> float:
    """1 / (1 + (value/target)^sharpness). Strictly decreasing, never exactly 0.

    Equals 0.5 exactly at value == target, so `target` keeps its meaning as the
    SLO. A config at 2x the SLO scores 0.2; at 10x it scores 0.0099. Under v1
    both scored 0. That difference is the whole point.
    """
    value_ms = max(0.0, float(value_ms))
    if value_ms == 0.0:
        return 1.0
    return 1.0 / (1.0 + (value_ms / float(target_ms)) ** sharpness)


def utility_v2(metrics: dict, weights: dict | None = None,
               scales: RewardScales = DEFAULT_SCALES) -> float:
    """Reward v2 from RAW metrics (not pre-normalised rates).

    Args:
        metrics: keys (all optional, each defaulting to a neutral value):
            throughput_tps  float, raw tokens/sec
            ttft_p99_ms     float, raw milliseconds
            tpot_p99_ms     float, raw milliseconds
            cache_hit_rate  float in [0,1]
        weights: term weights; defaults to DEFAULT_WEIGHTS.
        scales:  normalisation scales; see RewardScales.

    Returns:
        Weighted sum in (0, sum(weights)). With default weights: (0, 1).

    Note the signature difference from v1: v2 takes RAW metrics, because
    normalisation is the thing being fixed and must not be done by the caller.
    """
    w = weights or DEFAULT_WEIGHTS
    return (
        w["throughput"] * throughput_term(metrics.get("throughput_tps", 0.0), scales)
        + w["ttft"] * latency_term(metrics.get("ttft_p99_ms", 0.0),
                                   scales.ttft_target, scales.sharpness)
        + w["tpot"] * latency_term(metrics.get("tpot_p99_ms", 0.0),
                                   scales.tpot_target, scales.sharpness)
        + w["cache"] * float(metrics.get("cache_hit_rate", 0.0))
    )


def term_breakdown(metrics: dict, weights: dict | None = None,
                   scales: RewardScales = DEFAULT_SCALES) -> dict:
    """Per-term contributions, for the reward diagnostic (Phase 4).

    Returns raw term values AND their weighted contributions, so a diagnostic can
    show *which* term went flat rather than only that the total did.
    """
    w = weights or DEFAULT_WEIGHTS
    t = throughput_term(metrics.get("throughput_tps", 0.0), scales)
    ttft = latency_term(metrics.get("ttft_p99_ms", 0.0), scales.ttft_target, scales.sharpness)
    tpot = latency_term(metrics.get("tpot_p99_ms", 0.0), scales.tpot_target, scales.sharpness)
    cache = float(metrics.get("cache_hit_rate", 0.0))
    return {
        "throughput_term": t, "ttft_term": ttft, "tpot_term": tpot, "cache_term": cache,
        "throughput_contrib": w["throughput"] * t,
        "ttft_contrib": w["ttft"] * ttft,
        "tpot_contrib": w["tpot"] * tpot,
        "cache_contrib": w["cache"] * cache,
        "total": (w["throughput"] * t + w["ttft"] * ttft
                  + w["tpot"] * tpot + w["cache"] * cache),
    }


# ---------------------------------------------------------------------------
# Degeneracy detection.
# ---------------------------------------------------------------------------


class DegenerateRewardError(RuntimeError):
    """Raised when a reward carries no information across the arms under test."""


def check_non_degenerate(rewards, *, context: str = "",
                         min_variance: float = 1e-9,
                         min_distinct: int = 2) -> dict:
    """Assert a reward sequence actually discriminates; return a report.

    This exists because the v1 degeneracy went unnoticed for the entire project:
    aggregate statistics looked healthy while the per-cycle reward was constant.
    Any experiment that expects learning should call this and FAIL LOUDLY.

    Raises:
        DegenerateRewardError if the reward has fewer than `min_distinct`
        distinct values or variance below `min_variance`.
    """
    vals = [float(r) for r in rewards]
    if not vals:
        raise DegenerateRewardError(f"{context}: empty reward sequence")
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    distinct = sorted(set(vals))
    report = {
        "n": n, "mean": mean, "variance": var,
        "n_distinct": len(distinct), "min": min(vals), "max": max(vals),
        "distinct_sample": distinct[:10],
        "degenerate": len(distinct) < min_distinct or var < min_variance,
    }
    if report["degenerate"]:
        raise DegenerateRewardError(
            f"{context}: reward is degenerate -- {len(distinct)} distinct "
            f"value(s) {distinct[:5]}, variance {var:.3e} over {n} samples. "
            "The controller cannot distinguish configurations; any learning "
            "result from this run is meaningless."
        )
    return report
