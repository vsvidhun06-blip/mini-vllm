"""
100k-request long-horizon INTERACTIVE stability experiment (simulation).

WHAT THIS IS
------------
A CONTROL-LOOP SIMULATION (torch-free, CPU-only), consistent with the rest of
scripts/eval/. It drives the REAL CARL controller / LinUCB bandit / regime
classifier / reward over benchmark_carl's analytical WorkloadModel for 100,000
requests, on a workload calibrated to stay INTERACTIVE (arrival_probe: rho=0.6,
single-turn, lognormal(mean=48, std=24), seed 42). The question is ENDURANCE, not
a cross-seed effect: does CARL converge and STAY near-optimal over a long horizon,
or does something (regret, oracle-capture, SLO, or the bandit's own linear algebra)
drift?

NO controller / scheduler / reward / classifier changes. The only new thing vs
trace_replay is (a) the calibrated INTERACTIVE workload (from arrival_probe) and
(b) long-horizon instrumentation: LinUCB numerical health (condition number,
||theta||) and arm-switch rate sampled at every 1k-request checkpoint, plus a
within-run statistical analysis (single seed -> trend tests, not cross-seed CIs).

REUSE (nothing duplicated)
--------------------------
  arrival_probe.gen_stream / derive  -- the calibrated INTERACTIVE arrival model.
  trace_replay.dynoracle_rewards / checkpoint_metrics / round_reward / _config_sig
                                     -- the cost-model oracle + checkpoint metrics.
  benchmark_carl.WorkloadModel / _synth_state -- the analytical cost model.
  _harness.make_agent / best_static_config    -- the agents (unchanged).

RUNTIME: CPU-only, ~1-5 min single-seed (10k cycles; the only compute is 10x10
LinUCB inverses). GPU is unnecessary -- there is no NN forward pass.

Run (design default is the full 100k; --smoke does a tiny dry-run):
  python scripts/eval/stability_100k.py                    # seed 42, 100k
  python scripts/eval/stability_100k.py --smoke            # ~2k dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

# --- path bootstrap ---------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import _harness as h  # noqa: E402
import arrival_probe as ap  # noqa: E402
import benchmark_carl as bc  # noqa: E402
import trace_replay as tr  # noqa: E402
from src.carl.config import all_arm_sets  # noqa: E402
from src.carl.state import WorkloadRegime, classify_regime  # noqa: E402

ROUND_SIZE = tr.ROUND_SIZE                 # 10 requests per control cycle.
DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
DEFAULT_OUT = os.path.join(DOCS_EVAL, "stability_100k_results.json")

# Calibrated INTERACTIVE workload (from arrival_probe, seed 42).
RHO = 0.6
LEN_MEAN = 48.0
LEN_STD = 24.0

# Pre-registered validity gate (from the protocol's failure criteria).
INTERACTIVE_MIN_FRAC = 0.99
QUEUE_P99_MAX = 8.0

# Methods: CARL-Full is the subject; Static-Best is the fixed-policy reference;
# UCB1 / EpsilonGreedy are cheap context-free contrasts. DynOracle is the ceiling
# (its reward is the denominator of oracle-capture, computed separately).
METHODS = ["CARL-Full", "Static-Best", "UCB1", "EpsilonGreedy"]


# ===========================================================================
# Workload: 100k INTERACTIVE requests -> per-cycle regime stream.
# ===========================================================================


def build_workload(seed: int, n: int, rho: float, len_mean: float,
                   len_std: float) -> dict:
    """Generate the calibrated INTERACTIVE stream and group into control cycles.

    Returns per-request regimes, per-cycle (ROUND_SIZE majority) regimes, and the
    validity stats the pre-registered gate checks (INTERACTIVE fraction, queue
    percentiles). Grouping-by-majority matches trace_replay.derive_regimes so the
    controller fires on the identical cadence as the rest of the suite.
    """
    reqs = ap.gen_stream(seed, n, len_mean=len_mean, len_std=len_std,
                         single_turn=True)
    per_req, qinfo = ap.derive(reqs, rho)
    rounds = []
    for i in range(0, len(per_req), ROUND_SIZE):
        chunk = per_req[i:i + ROUND_SIZE]
        rounds.append(max(set(chunk), key=chunk.count))

    n_req = len(per_req) or 1
    mix: dict = {}
    for reg in per_req:
        mix[reg.value] = mix.get(reg.value, 0) + 1
    qd = qinfo.get("qdepths", [])
    return {
        "round_regimes": rounds,
        "n_requests": len(reqs),
        "n_cycles": len(rounds),
        "interactive_frac": mix.get("interactive", 0) / n_req,
        "regime_mix": {k: v / n_req for k, v in mix.items()},
        "queue_p99": ap._pct(qd, 99),
        "queue_max": max(qd) if qd else 0.0,
    }


# ===========================================================================
# Long-horizon instrumentation (the new bits vs trace_replay).
# ===========================================================================


def linucb_health(agent, regime: WorkloadRegime) -> dict:
    """LinUCB numerical health for one regime: condition number + ||theta||.

    Each arm holds a d x d design matrix A and vector b; theta = A^-1 b is the
    arm's learned weight. Over 100k updates A must stay well-conditioned and
    ||theta|| bounded -- unbounded growth is the classic long-horizon failure of
    an online accumulator. Reported as the max/mean across the regime's arms.
    """
    sub = agent.controller.bandit.bandits[regime]
    conds, norms = [], []
    for A, b in zip(sub.A, sub.b):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        conds.append(float(np.linalg.cond(A)))
        try:
            theta = np.linalg.solve(A, b)
            norms.append(float(np.linalg.norm(theta)))
        except np.linalg.LinAlgError:
            norms.append(float("inf"))
    return {"cond_max": max(conds), "cond_mean": statistics.fmean(conds),
            "theta_norm_max": max(norms), "theta_norm_mean": statistics.fmean(norms)}


def run_carl_with_checkpoints(round_regimes: list, slo, seed: int,
                              oracle_rewards: dict, ckpt_cycles: int) -> tuple:
    """Drive CARL-Full over the full horizon, snapshotting at each checkpoint.

    Mirrors trace_replay.run_agent's CARL path (same delayed-reward loop, same
    reward) but additionally records, at every `ckpt_cycles` cycles: the standard
    checkpoint metrics (cumulative + trailing window), the LinUCB health for the
    INTERACTIVE regime, the trailing-window arm-switch rate, and the incremental
    regret. Returns (records, checkpoints).
    """
    rng = random.Random(seed)
    model = bc.WorkloadModel(rng)
    agent = h.make_agent("CARL-Full", slo)

    records: list = []
    checkpoints: list = []
    prev_metrics = None
    last_sig: dict = {}
    for ci, true_regime in enumerate(round_regimes):
        state = bc._synth_state(true_regime, prev_metrics, rng)
        detected = classify_regime(state)
        config = agent.choose(true_regime, state)
        metrics = model.simulate(config, true_regime, ROUND_SIZE)
        reward = tr.round_reward(metrics, slo)
        agent.note_realised(metrics)

        sig = tr._config_sig(config)
        switched = detected in last_sig and sig != last_sig[detected]
        last_sig[detected] = sig

        oracle_r = oracle_rewards.get(true_regime, 0.0)
        records.append({
            "true_regime": true_regime.value, "detected": detected.value,
            "sig": sig, "reward": reward, "oracle_reward": oracle_r,
            "instant_regret": max(0.0, oracle_r - reward),
            "throughput": metrics["throughput"],
            "ttft": metrics["ttft_samples"], "tpot": metrics["tpot_samples"],
            "switched": bool(switched),
        })
        prev_metrics = metrics

        if (ci + 1) % ckpt_cycles == 0:
            checkpoints.append(_make_checkpoint(records, ci + 1, slo, agent,
                                                ckpt_cycles, oracle_rewards))
    return records, checkpoints


def _make_checkpoint(records: list, cycles_done: int, slo, agent,
                     window: int, oracle_rewards: dict) -> dict:
    """One checkpoint: cumulative + trailing-window metrics + bandit health."""
    cum = tr.checkpoint_metrics(records, cycles_done, slo)
    win_records = records[-window:]
    win = tr.checkpoint_metrics(win_records, len(win_records), slo)
    health = linucb_health(agent, WorkloadRegime.INTERACTIVE)
    switch_rate = sum(1 for r in win_records if r["switched"]) / max(1, len(win_records))
    inc_regret = sum(r["instant_regret"] for r in win_records)
    return {
        "requests": cycles_done * ROUND_SIZE,
        "cycle": cycles_done,
        "cumulative": cum,
        "window": win,
        "linucb": health,
        "arm_switch_rate": switch_rate,
        "incremental_regret": inc_regret,
    }


def run_baseline_with_checkpoints(name: str, round_regimes: list, slo, seed: int,
                                  oracle_rewards: dict, ckpt_cycles: int,
                                  static_best_cfg=None) -> list:
    """Reference series for a non-CARL method (Static-Best / UCB1 / EpsilonGreedy).

    Reuses trace_replay.run_agent to produce the per-cycle records, then slices
    the cumulative checkpoint metrics at each checkpoint. No bandit-health/
    arm-switch instrumentation (only CARL is the subject of those).
    """
    records = tr.run_agent(name, round_regimes, slo, seed,
                           static_best_cfg=static_best_cfg,
                           oracle_rewards=oracle_rewards)
    out = []
    for ci in range(ckpt_cycles, len(records) + 1, ckpt_cycles):
        cum = tr.checkpoint_metrics(records, ci, slo)
        out.append({"requests": ci * ROUND_SIZE, "cycle": ci,
                    "oracle_capture_pct": cum.get("oracle_capture_pct"),
                    "cumulative_regret": cum.get("cumulative_regret"),
                    "slo_rate": cum.get("slo_rate"),
                    "ttft_p99_ms": cum.get("ttft_p99_ms")})
    return out


# ===========================================================================
# Within-run statistical analysis (single seed -> trend, not cross-seed CIs).
# ===========================================================================


def ols_slope_ci(xs: list, ys: list, alpha: float = 0.05) -> dict:
    """OLS slope of ys vs xs with a (1-alpha) CI (H0: slope = 0).

    Used on the checkpoint series (e.g. oracle-capture vs checkpoint index). A
    non-degrading metric has a slope CI that excludes the "bad" direction. df is
    small enough (n_checkpoints) that we use the normal-ish critical value 1.96
    for the 95% CI; for 100 checkpoints this is accurate.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    n = len(pairs)
    if n < 3:
        return {"slope": None, "ci95": [None, None], "n": n}
    mx = statistics.fmean([x for x, _ in pairs])
    my = statistics.fmean([y for _, y in pairs])
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    if sxx == 0:
        return {"slope": None, "ci95": [None, None], "n": n}
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in pairs]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    z = 1.96
    return {"slope": slope, "se": se, "ci95": [slope - z * se, slope + z * se],
            "n": n}


def mann_kendall(ys: list) -> dict:
    """Mann-Kendall trend test on a series (H0: no monotone trend).

    Used on the per-checkpoint INCREMENTAL regret: a healthy converged controller
    shows NO upward trend (S <= 0 / p > 0.05). Returns S, the normal z, and a
    two-sided p; positive z with p<0.05 => a significant upward (worsening) trend.
    """
    vals = [y for y in ys if y is not None]
    n = len(vals)
    if n < 3:
        return {"S": None, "z": None, "p_value": None, "trend": "insufficient"}
    S = sum(np.sign(vals[j] - vals[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if S > 0:
        z = (S - 1) / math.sqrt(var)
    elif S < 0:
        z = (S + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))       # two-sided normal p.
    trend = "increasing" if (z > 0 and p < 0.05) else \
            "decreasing" if (z < 0 and p < 0.05) else "none"
    return {"S": float(S), "z": z, "p_value": p, "trend": trend}


def block_bootstrap_diff(early: list, late: list, *, n_boot: int = 1000,
                         block: int = 10, seed: int = 12345) -> dict:
    """Block-bootstrap 95% CI for mean(late) - mean(early) over per-cycle values.

    Resamples contiguous blocks (respecting autocorrelation) to compare an early
    vs a late window (e.g. cycles 0-1k vs 9k-10k). A CI that excludes 0 means the
    late window genuinely differs from the early one.
    """
    rng = random.Random(seed)

    def boot_mean(series: list) -> list:
        nb = max(1, len(series) // block)
        means = []
        for _ in range(n_boot):
            vals = []
            for _b in range(nb):
                start = rng.randrange(max(1, len(series) - block))
                vals.extend(series[start:start + block])
            means.append(statistics.fmean(vals) if vals else 0.0)
        return means
    if len(early) < block or len(late) < block:
        return {"mean_diff": None, "ci95": [None, None]}
    de = boot_mean(early)
    dl = boot_mean(late)
    diffs = sorted(l - e for e, l in zip(de, dl))
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return {"mean_diff": statistics.fmean(late) - statistics.fmean(early),
            "ci95": [lo, hi]}


# ===========================================================================
# Failure-criteria evaluation (pre-registered).
# ===========================================================================


def evaluate_failure_criteria(validity: dict, carl_ckpts: list,
                              analysis: dict) -> dict:
    """Return a pass/fail verdict per pre-registered failure criterion."""
    v = {}
    # Validity gate.
    v["workload_interactive"] = validity["interactive_frac"] >= INTERACTIVE_MIN_FRAC
    v["workload_queue"] = validity["queue_p99"] < QUEUE_P99_MAX
    # Regret sub-linearity: no upward trend in incremental regret.
    v["regret_sublinear"] = analysis["incremental_regret_mk"]["trend"] != "increasing"
    # No capture degradation: slope not significantly negative.
    cap = analysis["oracle_capture_slope"]
    v["capture_not_degrading"] = (cap["slope"] is None) or (cap["ci95"][1] is not None
                                                            and cap["ci95"][1] >= 0)
    # Arm-switch decay: late-window switch rate below early-window.
    if len(carl_ckpts) >= 4:
        early = statistics.fmean([c["arm_switch_rate"] for c in carl_ckpts[:len(carl_ckpts)//4]])
        late = statistics.fmean([c["arm_switch_rate"] for c in carl_ckpts[-len(carl_ckpts)//4:]])
        v["arm_switch_decays"] = late <= early
    else:
        v["arm_switch_decays"] = None
    # Numerical bound: max condition number / ||theta|| finite and not exploding.
    conds = [c["linucb"]["cond_max"] for c in carl_ckpts]
    norms = [c["linucb"]["theta_norm_max"] for c in carl_ckpts]
    v["numerically_bounded"] = (all(math.isfinite(x) for x in conds + norms)
                                and (len(conds) < 2 or conds[-1] < 1e12))
    v["all_pass"] = all(x for x in v.values() if x is not None)
    return v


# ===========================================================================
# Driver.
# ===========================================================================


def run(seed: int, n: int, ckpt_every: int, out_path: str) -> dict:
    slo = tr.make_slo()
    ckpt_cycles = max(1, ckpt_every // ROUND_SIZE)

    print(f"Building calibrated INTERACTIVE workload: seed={seed}, n={n}, "
          f"rho={RHO}, len=({LEN_MEAN},{LEN_STD}), single-turn...", flush=True)
    wl = build_workload(seed, n, RHO, LEN_MEAN, LEN_STD)
    print(f"  INTERACTIVE={wl['interactive_frac']*100:.2f}%  "
          f"queue_p99={wl['queue_p99']:.0f}  cycles={wl['n_cycles']}", flush=True)

    # Pre-registered validity GATE: abort before the heavy loop if contaminated.
    if wl["interactive_frac"] < INTERACTIVE_MIN_FRAC or wl["queue_p99"] >= QUEUE_P99_MAX:
        print("!!! VALIDITY GATE FAILED (workload not INTERACTIVE) -- aborting; "
              "recalibrate arrival_probe before reporting.", flush=True)
        return {"aborted": True, "validity": wl}

    round_regimes = wl["round_regimes"]
    oracle_rewards = tr.dynoracle_rewards(slo, all_arm_sets())
    static_best = h.best_static_config(round_regimes[:max(1, 100 // ROUND_SIZE)], slo,
                                       seed=seed)

    t0 = time.perf_counter()
    print("Running CARL-Full (subject) with checkpoint instrumentation...", flush=True)
    carl_records, carl_ckpts = run_carl_with_checkpoints(
        round_regimes, slo, seed, oracle_rewards, ckpt_cycles)

    baseline_series: dict = {}
    for name in [m for m in METHODS if m != "CARL-Full"]:
        print(f"Running reference: {name}...", flush=True)
        baseline_series[name] = run_baseline_with_checkpoints(
            name, round_regimes, slo, seed, oracle_rewards, ckpt_cycles,
            static_best_cfg=static_best if name == "Static-Best" else None)
    wall = time.perf_counter() - t0

    # ---- Within-run statistical analysis. ------------------------------------
    xs = [c["cycle"] for c in carl_ckpts]
    analysis = {
        "oracle_capture_slope": ols_slope_ci(
            xs, [c["cumulative"].get("oracle_capture_pct") for c in carl_ckpts]),
        "slo_rate_slope": ols_slope_ci(
            xs, [c["window"].get("slo_rate") for c in carl_ckpts]),
        "ttft_p99_slope": ols_slope_ci(
            xs, [c["window"].get("ttft_p99_ms") for c in carl_ckpts]),
        "incremental_regret_mk": mann_kendall(
            [c["incremental_regret"] for c in carl_ckpts]),
    }
    # Early-vs-late steady-state capture (block bootstrap over per-cycle reward
    # captured, using window oracle-capture as the per-checkpoint proxy).
    if len(carl_ckpts) >= 8:
        q = len(carl_ckpts) // 4
        analysis["capture_early_vs_late"] = block_bootstrap_diff(
            [c["window"]["oracle_capture_pct"] for c in carl_ckpts[:q]],
            [c["window"]["oracle_capture_pct"] for c in carl_ckpts[-q:]], block=5)

    verdict = evaluate_failure_criteria(wl, carl_ckpts, analysis)

    results = {
        "description": ("100k long-horizon INTERACTIVE stability (simulation). "
                        "Real CARL controller/bandit/classifier/reward over "
                        "benchmark_carl's analytical cost model; calibrated "
                        "INTERACTIVE workload (arrival_probe rho=0.6, single-turn, "
                        "lognormal(48,24)). Single seed -> endurance/stability, "
                        "not a cross-seed effect."),
        "settings": {"seed": seed, "n_requests": n, "rho": RHO,
                     "len_mean": LEN_MEAN, "len_std": LEN_STD, "single_turn": True,
                     "round_size": ROUND_SIZE, "checkpoint_every_requests": ckpt_every,
                     "slo": {"ttft_ms": tr.SLO_TTFT_MS, "tpot_ms": tr.SLO_TPOT_MS,
                             "throughput_ref": tr.SLO_THROUGHPUT_REF},
                     "methods": METHODS},
        "validity": {k: wl[k] for k in ("interactive_frac", "regime_mix",
                                        "queue_p99", "queue_max", "n_cycles")},
        "carl_checkpoints": carl_ckpts,
        "baseline_checkpoints": baseline_series,
        "analysis": analysis,
        "failure_criteria": verdict,
        "runtime_s": wall,
        "figures": _figure_data(carl_ckpts, baseline_series),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    print(f"\nWall: {wall:.1f}s (CPU, torch-free). Saved to {out_path}", flush=True)
    return results


def _figure_data(carl_ckpts: list, baseline_series: dict) -> dict:
    """Extract the plotting series for the paper figures (A-F)."""
    x = [c["requests"] for c in carl_ckpts]
    return {
        "x_requests": x,
        "A_oracle_capture_pct": [c["cumulative"].get("oracle_capture_pct") for c in carl_ckpts],
        "B_cumulative_regret": [c["cumulative"].get("cumulative_regret") for c in carl_ckpts],
        "C_slo_rate": [c["window"].get("slo_rate") for c in carl_ckpts],
        "C_ttft_p99_ms": [c["window"].get("ttft_p99_ms") for c in carl_ckpts],
        "D_arm_switch_rate": [c["arm_switch_rate"] for c in carl_ckpts],
        "E_cond_max": [c["linucb"]["cond_max"] for c in carl_ckpts],
        "E_theta_norm_max": [c["linucb"]["theta_norm_max"] for c in carl_ckpts],
        "baseline_capture": {k: [c["oracle_capture_pct"] for c in v]
                             for k, v in baseline_series.items()},
    }


def _print_summary(results: dict) -> None:
    print("\n=== 100k STABILITY SUMMARY (seed "
          f"{results['settings']['seed']}) ===")
    v = results["validity"]
    print(f"workload: INTERACTIVE={v['interactive_frac']*100:.2f}%  "
          f"queue_p99={v['queue_p99']:.0f}")
    a = results["analysis"]
    if results["carl_checkpoints"]:
        last = results["carl_checkpoints"][-1]
        print(f"final: oracle_capture={last['cumulative'].get('oracle_capture_pct'):.1f}%  "
              f"cum_regret={last['cumulative'].get('cumulative_regret'):.2f}  "
              f"arm_switch_rate={last['arm_switch_rate']:.3f}  "
              f"cond_max={last['linucb']['cond_max']:.1f}")
    print(f"capture slope/ckpt: {a['oracle_capture_slope']['slope']}  "
          f"CI95={a['oracle_capture_slope']['ci95']}")
    print(f"incremental-regret trend (Mann-Kendall): {a['incremental_regret_mk']['trend']} "
          f"(p={a['incremental_regret_mk']['p_value']})")
    print("\nFAILURE CRITERIA:")
    for k, val in results["failure_criteria"].items():
        print(f"  {k:24}: {'PASS' if val else ('n/a' if val is None else 'FAIL')}")


def main() -> None:
    p = argparse.ArgumentParser(description="100k INTERACTIVE stability (simulation).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--smoke", action="store_true",
                   help="tiny dry-run (n=2000) to validate the pipeline")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    n = 2000 if args.smoke else args.n
    run(args.seed, n, args.checkpoint_every, args.out)


if __name__ == "__main__":
    main()
