"""
CARL Statistical Validation -- is CARL's throughput win real or noise?

Runs CARL-Full and Static-Best on the NON-STATIONARY workload over N=30
independent seeds (each seed is a different cost-model noise draw -- a different
"request ordering"). Because both methods see the SAME seed on each run, this is
a PAIRED comparison, so we use a paired t-test on the per-seed throughput
difference d = CARL - Static.

  Null hypothesis  H0: mean(d) = 0  (CARL and Static have equal mean throughput)
  Alt  hypothesis  H1: mean(d) != 0
  Reject H0 (claim a significant difference) when p < 0.05.

We report each method's mean / std / 95% CI, the paired difference with its 95%
CI, the t-statistic, the two-sided p-value, and Cohen's d (paired effect size).
The t-distribution CDF is computed from scratch (regularised incomplete beta), so
this script needs no scipy -- only the standard library and the eval harness.

CONTROL-LOOP SIMULATION (see _harness.SIM_NOTE): the test establishes that, IN
SIMULATION, CARL's mean throughput differs from the best static config by more
than seed noise -- not a measured-hardware claim.

Run:
  python scripts/eval/statistical_validation.py          # 30 runs, 40 requests
Outputs the statistical summary and docs/eval/stats_results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

import _harness as h


# ---------------------------------------------------------------------------
# Student's t distribution helpers (no scipy): regularised incomplete beta.
# Standard Numerical-Recipes continued-fraction evaluation; betai gives the
# two-sided t p-value via the A&S identity p = I_{df/(df+t^2)}(df/2, 1/2).
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    res = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        res *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        res *= delta
        if abs(delta - 1.0) < EPS:
            break
    return res


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for Student's t with `df` dof."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def t_critical(df: float, alpha: float = 0.05) -> float:
    """Two-sided critical t for confidence 1-alpha (bisection on the p-value)."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # two-sided p decreases as |t| grows; find where it equals alpha.
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# The experiment.
# ---------------------------------------------------------------------------


def run_stats(runs: int, n_requests: int) -> dict:
    slo = h.slo_ttft_only()
    half = n_requests // 2
    regimes = h.nonstationary(half, n_requests - half)
    static_best = h.best_static_config(regimes, slo)

    carl_tps: list[float] = []
    static_tps: list[float] = []
    for seed in h.tqdm(range(runs), desc="paired runs"):
        carl_tps.append(h.run_once(
            h.make_agent("CARL-Full", slo, static_best_cfg=static_best),
            regimes, slo, seed)["throughput"])
        static_tps.append(h.run_once(
            h.make_agent("Static-Best", slo, static_best_cfg=static_best),
            regimes, slo, seed)["throughput"])

    n = runs
    df = n - 1
    diffs = [c - s for c, s in zip(carl_tps, static_tps)]
    mean_d = statistics.fmean(diffs)
    std_d = statistics.stdev(diffs) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n) if n > 0 else 0.0
    t_stat = mean_d / se_d if se_d > 0 else float("inf")
    p_value = t_two_sided_p(t_stat, df) if se_d > 0 else 0.0
    tcrit = t_critical(df)
    ci_d = (mean_d - tcrit * se_d, mean_d + tcrit * se_d)
    cohens_d = mean_d / std_d if std_d > 0 else float("inf")

    def summary(vals: list[float]) -> dict:
        m = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        se = sd / math.sqrt(len(vals)) if vals else 0.0
        return {"mean": m, "std": sd,
                "ci95": (m - tcrit * se, m + tcrit * se)}

    return {
        "settings": {"runs": runs, "requests": n_requests,
                     "workload": "NON-STATIONARY", "slo_ttft_ms": h.SLO_TTFT_MS,
                     "static_best_config": static_best.as_dict()},
        "carl_full": summary(carl_tps),
        "static_best": summary(static_tps),
        "paired_difference": {
            "mean": mean_d, "std": std_d, "ci95": ci_d,
            "t_statistic": t_stat, "df": df, "p_value": p_value,
            "cohens_d": cohens_d, "significant_at_0.05": bool(p_value < 0.05),
        },
        "_carl_tps": carl_tps, "_static_tps": static_tps,
    }


def _print(r: dict) -> None:
    c, s, pd = r["carl_full"], r["static_best"], r["paired_difference"]
    headers = ["method", "mean tok/s", "std", "95% CI"]
    rows = [
        ["CARL-Full", f"{c['mean']:.2f}", f"{c['std']:.2f}",
         f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"],
        ["Static-Best", f"{s['mean']:.2f}", f"{s['std']:.2f}",
         f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"],
    ]
    h.print_pipe_table("STATISTICAL VALIDATION (NON-STATIONARY, paired over seeds)",
                       headers, rows)

    headers = ["quantity", "value"]
    rows = [
        ["mean difference (CARL - Static)", f"{pd['mean']:+.2f} tok/s"],
        ["95% CI of difference", f"[{pd['ci95'][0]:+.2f}, {pd['ci95'][1]:+.2f}]"],
        ["t-statistic", f"{pd['t_statistic']:.3f}"],
        ["degrees of freedom", str(pd["df"])],
        ["p-value (two-sided)", f"{pd['p_value']:.3e}"],
        ["Cohen's d (paired)", f"{pd['cohens_d']:.2f}"],
        ["significant at p<0.05", "YES" if pd["significant_at_0.05"] else "NO"],
    ]
    h.print_pipe_table("PAIRED t-TEST  (H0: equal mean throughput)", headers, rows)

    verdict = ("REJECT H0: CARL's mean throughput differs from Static-Best by more "
               "than seed noise (statistically significant in simulation)."
               if pd["significant_at_0.05"] else
               "FAIL TO REJECT H0: no significant difference at p<0.05.")
    print(f"\n{verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL statistical validation (simulation).")
    parser.add_argument("--runs", type=int, default=30, help="independent paired runs")
    parser.add_argument("--requests", type=int, default=40, help="requests per run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(h.SIM_NOTE)
    results = run_stats(args.runs, args.requests)
    _print(results)

    out_path = args.out or (h.eval_docs_dir() / "stats_results.json")
    # Drop the bulky raw-sample arrays' underscore keys? Keep them: they let a
    # reader re-run the test or plot the distribution. They are small (N<=30).
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved stats results to {out_path}")


if __name__ == "__main__":
    main()
