"""
CARL alpha (LinUCB exploration) sensitivity sweep.

WHAT THIS MEASURES
------------------
LinUCB scores each arm by  theta^T x  +  alpha * sqrt(x^T A^{-1} x): alpha is the
single knob that trades EXPLOITATION (trust the learned reward) against
EXPLORATION (probe under-tried arms). This script sweeps alpha over
{0.1, 0.25, 0.5, 1.0, 2.0} on the NON-STATIONARY workload and reports how the
controller's behaviour moves with it, so the paper can justify the alpha=0.5
default rather than asserting it.

Because arm 0 of every regime is that regime's hand-tuned oracle config (CARL's
warm start), MORE exploration (large alpha) means the controller spends more
cycles probing AWAY from an already-good operating point -- we therefore expect
regret to RISE and oracle-capture to FALL as alpha grows, while adaptation
latency (which is detection-bound in this simulation) stays roughly flat. The
sweep confirms the shape and locates the sweet spot.

Metrics (mean +/- std over 3 seeds), per alpha:
  * throughput            mean tok/s over the run.
  * cumulative_regret     sum of max(0, oracle_reward[regime] - reward) per cycle,
                          against the static best-arm-per-regime (DynOracle).
  * time_to_adaptation    mean requests after a regime transition until CARL's
                          detected regime matches the new one (the suite's
                          "adaptation latency"; detection-bound here).
  * oracle_capture_pct    100 * sum(reward) / sum(oracle_reward[regime]).

CONTROL-LOOP SIMULATION (see _harness.SIM_NOTE): drives the REAL CARL
controller/LinUCB over benchmark_carl's analytical cost model. Comparisons across
alpha are the result, not absolute values. Imports only the shared suite modules;
modifies no existing script.

Run:
  python scripts/eval/alpha_sensitivity.py                # 5 alphas, 3 seeds, 50 reqs
  python scripts/eval/alpha_sensitivity.py --seeds 5 --requests 100
Outputs the sweep table and docs/eval/alpha_sensitivity_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

# --- path bootstrap so `python scripts/eval/alpha_sensitivity.py` runs standalone.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # scripts/eval
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)                    # scripts
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)                   # repo root
for _p in (_REPO_ROOT, _SCRIPTS_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _harness as h  # noqa: E402
import benchmark_carl as bc  # noqa: E402
from src.carl.bandit import DEFAULT_UTILITY_WEIGHTS, LinUCBBandit, utility  # noqa: E402
from src.carl.config import all_arm_sets  # noqa: E402
from src.carl.controller import SLO  # noqa: E402
from src.carl.state import WorkloadRegime, classify_regime  # noqa: E402

# The sweep + experiment size (the spec's parameters).
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]
DEFAULT_SEEDS = 3
DEFAULT_REQUESTS = 50

# DynOracle estimation (best arm per regime by mean reward, in hindsight).
ORACLE_SAMPLES = 500
ORACLE_SEED = 99_999


def round_reward(metrics: dict, slo: SLO) -> float:
    """Scalar utility for one control cycle's realised metrics.

    Mirrors CARLController._reward_for_state: throughput normalized to the
    reference, TTFT/TPOT violation rates over the cycle's samples, and the cache
    hit rate, combined with the bandit's default weights -- the SAME objective
    CARL optimizes, so regret/oracle-capture are scored on CARL's own reward.
    """
    ttft = metrics["ttft_samples"]
    tpot = metrics["tpot_samples"]
    n = len(ttft) or 1
    tps_norm = (min(1.0, metrics["throughput"] / slo.throughput_ref)
                if slo.throughput_ref > 0 else 0.0)
    ttft_viol = sum(1 for t in ttft if t > slo.ttft_ms) / n
    tpot_viol = sum(1 for p in tpot if p > slo.tpot_ms) / n
    return utility({
        "throughput_norm": tps_norm,
        "ttft_violation_rate": ttft_viol,
        "tpot_violation_rate": tpot_viol,
        "cache_hit_rate": metrics["cache_hit"],
    }, DEFAULT_UTILITY_WEIGHTS)


def dynoracle_rewards(slo: SLO) -> dict:
    """Best achievable mean reward per regime (the DynOracle reward target).

    For each regime, simulate every arm over ORACLE_SAMPLES single-request cycles
    and keep the arm with the highest mean reward -- the best-arm-per-regime an
    omniscient operator would pick in hindsight. No online alpha can beat this, so
    it is the denominator for oracle_capture and the reference for regret.
    """
    rng = random.Random(ORACLE_SEED)
    model = bc.WorkloadModel(rng)
    out: dict[WorkloadRegime, float] = {}
    for regime, arms in all_arm_sets().items():
        best = float("-inf")
        for cfg in arms:
            rewards = [round_reward(model.simulate(cfg, regime, 1), slo)
                       for _ in range(ORACLE_SAMPLES)]
            best = max(best, statistics.fmean(rewards))
        out[regime] = best
    return out


def nonstationary_phases(n_requests: int) -> list[tuple]:
    """The canonical NON-STATIONARY phase plan: INTERACTIVE -> BATCH -> BURST.

    Splits `n_requests` into three near-equal phases so the run contains two
    regime transitions to measure adaptation against (matches benchmark_carl's
    scenario-3 definition of NON-STATIONARY).
    """
    a = n_requests // 3
    b = n_requests // 3
    c = n_requests - a - b
    return [(WorkloadRegime.INTERACTIVE, a), (WorkloadRegime.BATCH, b),
            (WorkloadRegime.BURST, c)]


def run_one(alpha: float, regimes: list, boundaries: list[int], slo: SLO,
            seed: int, oracle: dict) -> dict:
    """One (alpha, seed) run over the per-request NON-STATIONARY regime sequence.

    Drives the REAL CarlAgent (LinUCB at this alpha) one control cycle per
    request -- identical timing to _harness.run_once -- while capturing the
    detected-regime series (for adaptation lag) and the per-cycle reward (for
    regret / oracle capture).
    """
    rng = random.Random(seed)
    model = bc.WorkloadModel(rng)
    agent = bc.CarlAgent(LinUCBBandit, f"carl_a{alpha}", slo, alpha=alpha)

    tps_series: list[float] = []
    detected: list = []
    reward_sum = oracle_sum = regret = 0.0
    prev_metrics: dict | None = None

    for true_regime in regimes:
        state = bc._synth_state(true_regime, prev_metrics, rng)
        det = classify_regime(state)
        detected.append(det)

        config = agent.choose(true_regime, state)
        metrics = model.simulate(config, true_regime, 1)   # one request per cycle.
        agent.note_realised(metrics)

        reward = round_reward(metrics, slo)
        oracle_r = oracle[true_regime]
        reward_sum += reward
        oracle_sum += oracle_r
        regret += max(0.0, oracle_r - reward)
        tps_series.append(metrics["throughput"])
        prev_metrics = metrics

    # Adaptation latency: requests after each transition until detection catches
    # up to the new regime, averaged across the boundaries.
    lags = [h.adaptation_lag(detected, b, regimes[b]) for b in boundaries]
    time_to_adaptation = statistics.fmean(lags) if lags else 0.0

    return {
        "throughput_tps": statistics.fmean(tps_series) if tps_series else 0.0,
        "cumulative_regret": regret,
        "time_to_adaptation": time_to_adaptation,
        "oracle_capture_pct": (100.0 * reward_sum / oracle_sum) if oracle_sum > 0 else 0.0,
        "adaptations": float(agent.adaptations),
    }


METRIC_KEYS = ["throughput_tps", "cumulative_regret", "time_to_adaptation",
               "oracle_capture_pct"]


def run_sweep(n_seeds: int, n_requests: int) -> dict:
    # Same SLO the other NON-STATIONARY evals use (TTFT-only, ttft<200, ref=50);
    # CARL's controller and our external reward both score against it.
    slo = h.slo_ttft_only()
    phases = nonstationary_phases(n_requests)
    regimes = h.phased(*phases)
    boundaries = h.boundaries_of(phases)     # request indices of each transition.
    oracle = dynoracle_rewards(slo)
    seeds = list(range(n_seeds))

    per_alpha: dict = {}
    for alpha in h.tqdm(ALPHAS, desc="alpha"):
        runs = [run_one(alpha, regimes, boundaries, slo, seed, oracle)
                for seed in seeds]
        agg: dict = {}
        for k in METRIC_KEYS:
            mean, std = h.mean_std([r[k] for r in runs])
            agg[k] = {"mean": mean, "std": std}
        agg["adaptations_mean"] = h.mean_std([r["adaptations"] for r in runs])[0]
        per_alpha[str(alpha)] = agg

    # Pick the recommended alpha: the lowest cumulative regret (tie-break to the
    # higher oracle capture). Reported so the default has an explicit basis.
    best_alpha = min(
        ALPHAS,
        key=lambda a: (per_alpha[str(a)]["cumulative_regret"]["mean"],
                       -per_alpha[str(a)]["oracle_capture_pct"]["mean"]),
    )

    return {
        "note": ("CONTROL-LOOP SIMULATION (see _harness.SIM_NOTE). Real CARL "
                 "controller/LinUCB over benchmark_carl's analytical cost model; "
                 "the alpha-to-alpha comparison is the result, not absolute "
                 "values. Oracle is near-optimal by construction, so regret / "
                 "oracle-capture measure how well each alpha tracks a known-good "
                 "target online."),
        "settings": {
            "alphas": ALPHAS,
            "seeds": seeds,
            "requests": n_requests,
            "workload": "NON-STATIONARY (INTERACTIVE->BATCH->BURST)",
            "phase_sizes": [c for _r, c in phases],
            "boundaries": boundaries,
            "slo_ttft_ms": h.SLO_TTFT_MS,
            "dynoracle_reward_per_regime": {r.value: v for r, v in oracle.items()},
        },
        "per_alpha": per_alpha,
        "recommended_alpha_min_regret": best_alpha,
    }


def _print(results: dict) -> None:
    print(h.SIM_NOTE)
    headers = ["alpha", "tok/s", "cumRegret", "adaptLag(req)", "oracleCap%"]
    rows = []
    for alpha in ALPHAS:
        a = results["per_alpha"][str(alpha)]
        rows.append([
            f"{alpha}",
            h.fmt_pm(a["throughput_tps"]["mean"], a["throughput_tps"]["std"], 1),
            h.fmt_pm(a["cumulative_regret"]["mean"], a["cumulative_regret"]["std"], 2),
            h.fmt_pm(a["time_to_adaptation"]["mean"], a["time_to_adaptation"]["std"], 1),
            h.fmt_pm(a["oracle_capture_pct"]["mean"], a["oracle_capture_pct"]["std"], 1),
        ])
    h.print_pipe_table("ALPHA SENSITIVITY (NON-STATIONARY, mean +/- std over seeds)",
                       headers, rows)
    print(f"\nLowest-regret alpha: {results['recommended_alpha_min_regret']} "
          f"(default is 0.5).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL LinUCB alpha sensitivity sweep (simulation).")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = run_sweep(args.seeds, args.requests)
    _print(results)

    out_path = args.out or os.path.join(h.eval_docs_dir(), "alpha_sensitivity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved alpha sensitivity results to {out_path}")


if __name__ == "__main__":
    main()
