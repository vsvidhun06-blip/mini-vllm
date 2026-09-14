"""
Statistical validation SUMMARY -- an offline meta-analysis across the eval suite.

This is a PURE-JSON post-processor (no GPU, no torch, no simulation). It reads
the already-generated eval artifacts and attaches the inferential statistics a
reviewer asks for to every major throughput / TTFT claim:

  * 95% confidence intervals          (mean +/- t_{.975,df} * std/sqrt(n))
  * Cohen's d effect size             (CARL-Full vs Static-Best)
  * paired t-tests across seeds        (where per-seed raw data exists)

It is DELIBERATELY a separate file from scripts/eval/statistical_validation.py,
which is the SIMULATION that GENERATES docs/eval/stats_results.json (one of THIS
script's inputs). Overwriting that generator would delete the source of our
strongest paired sample, so this meta-analysis lives on its own and only reads.

INPUTS  (docs/eval/)
  stats_results.json        headline NON-STATIONARY run; carries the per-seed
                            raw arrays _carl_tps / _static_tps (n=30) -> the one
                            place we can run a TRUE paired t-test + paired d.
  workload_results.json     per-workload mean/std per method (n=3).  summary-stat
  ablation_results.json     per-scenario mean/std per config (n=5).  CIs + Welch
  sensitivity_results.json  per-sweep/setting carl vs static (n=3).  + Cohen's d
  oracle_results.json       per-phase CARL-vs-oracle throughput gap.

OUTPUT  docs/eval/statistical_validation_results.json

HONESTY ON TEST CHOICE
----------------------
A paired t-test needs per-seed PAIRED observations. Only stats_results.json keeps
the raw per-seed arrays, so that is the only genuinely paired test here and it is
labelled paired=true. For workload / ablation / sensitivity we have only summary
mean/std/n, so we report (a) CIs from those summaries and (b) a Welch two-sample
t-test + pooled-SD Cohen's d computed FROM the summary stats, each labelled
paired=false / method="welch_from_summary_stats". We never dress a summary-stat
test up as a paired one.

The Student-t CDF is computed from scratch via the regularised incomplete beta
(Numerical-Recipes continued fraction), so the script needs no scipy.

Run:
  python scripts/eval/statistical_validation_summary.py
"""
from __future__ import annotations

import json
import math
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
OUTPUT_PATH = os.path.join(DOCS_EVAL, "statistical_validation_results.json")


# ===========================================================================
# Student's t distribution from scratch (regularised incomplete beta).
# ===========================================================================


def _gammaln(x: float) -> float:
    """Log-gamma via the Lanczos approximation (g=5, n=6); good to ~1e-10."""
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
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
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(_gammaln(a + b) - _gammaln(a) - _gammaln(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for Student-t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    if math.isinf(t):
        return 0.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def t_cdf(t: float, df: float) -> float:
    """CDF of Student-t at t."""
    ib = _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of Student-t via bisection on the monotone CDF."""
    if df <= 0:
        return float("nan")
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# Small stats helpers operating on summary (mean, std, n) or raw samples.
# ===========================================================================


def ci95_from_summary(mean: float, std: float, n: int) -> dict:
    """95% CI for a mean from summary stats; df = n-1, t critical, se = std/sqrt(n)."""
    if n is None or n < 2:
        return {"mean": mean, "ci95": None, "n": n,
                "note": "n<2: no CI"}
    df = n - 1
    tcrit = t_ppf(0.975, df)
    se = std / math.sqrt(n)
    half = tcrit * se
    return {"mean": mean, "std": std, "n": n, "df": df,
            "se": se, "t_crit_0.975": tcrit,
            "ci95": [mean - half, mean + half]}


def ci95_from_raw(samples: list) -> dict:
    """95% CI for the mean of a raw sample list."""
    n = len(samples)
    if n < 2:
        return {"mean": (samples[0] if samples else None), "ci95": None, "n": n}
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    std = math.sqrt(var)
    return ci95_from_summary(mean, std, n)


def cohens_d_pooled(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    """Pooled-SD Cohen's d from summary stats (independent groups)."""
    if n1 < 2 or n2 < 2:
        # Fall back to a simple average-SD denominator when df is unavailable.
        denom = math.sqrt((s1 * s1 + s2 * s2) / 2.0)
    else:
        denom = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if denom == 0:
        return float("inf") if m1 != m2 else 0.0
    return (m1 - m2) / denom


def welch_ttest(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> dict:
    """Welch's unequal-variance two-sample t-test from summary stats (NOT paired)."""
    if n1 < 2 or n2 < 2:
        return {"t_statistic": None, "df": None, "p_value": None,
                "note": "need n>=2 per group"}
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    denom = v1 + v2
    if denom == 0:
        # Both groups deterministic: separation is exact if means differ.
        t = float("inf") if m1 != m2 else 0.0
        df = n1 + n2 - 2
        return {"t_statistic": t, "df": df,
                "p_value": (0.0 if m1 != m2 else 1.0),
                "note": "zero pooled variance"}
    t = (m1 - m2) / math.sqrt(denom)
    df = denom * denom / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    p = t_sf_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p, "significant_at_0.05": p < 0.05}


def paired_ttest(a: list, b: list) -> dict:
    """Paired t-test on a-b across seeds; also returns the paired Cohen's d."""
    n = min(len(a), len(b))
    if n < 2:
        return {"note": "need n>=2 paired observations"}
    diffs = [a[i] - b[i] for i in range(n)]
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        t = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
        d = float("inf") if mean != 0 else 0.0
    else:
        se = std / math.sqrt(n)
        t = mean / se
        df = n - 1
        p = t_sf_two_sided(t, df)
        d = mean / std   # paired Cohen's d (d_z)
    df = n - 1
    half = t_ppf(0.975, df) * (std / math.sqrt(n)) if std > 0 else 0.0
    return {
        "n": n, "mean_difference": mean, "std_difference": std, "df": df,
        "ci95_difference": [mean - half, mean + half],
        "t_statistic": t, "p_value": p, "cohens_d_paired": d,
        "significant_at_0.05": (p < 0.05),
        "paired": True,
    }


# ===========================================================================
# Per-artifact analyses.
# ===========================================================================


def _load(name: str):
    path = os.path.join(DOCS_EVAL, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _carl_vs_static_from_summary(carl: dict, static: dict, n: int, metric: str) -> dict:
    """CIs + Cohen's d + Welch test for one metric, CARL-Full vs Static-Best."""
    cm, cs = carl.get(f"{metric}_mean"), carl.get(f"{metric}_std")
    sm, ss = static.get(f"{metric}_mean"), static.get(f"{metric}_std")
    if cm is None or sm is None:
        return {"note": f"{metric} absent"}
    return {
        "carl_full": ci95_from_summary(cm, cs, n),
        "static_best": ci95_from_summary(sm, ss, n),
        "difference_mean": cm - sm,
        "cohens_d_summary": cohens_d_pooled(cm, cs, n, sm, ss, n),
        "welch_ttest": welch_ttest(cm, cs, n, sm, ss, n),
        "note": "Cohen's d / Welch test from summary stats (paired=false).",
    }


def analyze_headline(stats: dict) -> dict:
    """The central NON-STATIONARY claim: recomputed from per-seed raw arrays."""
    out: dict = {"source": "stats_results.json",
                 "workload": stats.get("settings", {}).get("workload"),
                 "n_seeds": stats.get("settings", {}).get("runs")}
    carl_raw = stats.get("_carl_tps")
    static_raw = stats.get("_static_tps")
    if carl_raw and static_raw:
        out["carl_full_throughput_ci95"] = ci95_from_raw(carl_raw)
        out["static_best_throughput_ci95"] = ci95_from_raw(static_raw)
        out["paired_ttest_throughput"] = paired_ttest(carl_raw, static_raw)
        # Cross-check our from-scratch t-math against the value the generator
        # stored, so a reviewer can trust the inferential numbers.
        ref = stats.get("paired_difference", {})
        rec = out["paired_ttest_throughput"]
        out["crosscheck_vs_generator"] = {
            "reported_t": ref.get("t_statistic"),
            "recomputed_t": rec.get("t_statistic"),
            "reported_p": ref.get("p_value"),
            "recomputed_p": rec.get("p_value"),
            "reported_cohens_d": ref.get("cohens_d"),
            "recomputed_cohens_d": rec.get("cohens_d_paired"),
            "t_matches": (ref.get("t_statistic") is not None
                          and abs(ref["t_statistic"] - rec["t_statistic"])
                          < 1e-6 * max(1.0, abs(ref["t_statistic"]))),
        }
    else:
        out["note"] = "raw per-seed arrays (_carl_tps/_static_tps) absent"
    return out


def analyze_workloads(workload: dict) -> dict:
    n = workload.get("settings", {}).get("runs")
    out: dict = {}
    for wname, methods in workload.get("workloads", {}).items():
        carl = methods.get("CARL-Full")
        static = methods.get("Static-Best")
        if not carl or not static:
            continue
        out[wname] = {
            "throughput": _carl_vs_static_from_summary(carl, static, n, "throughput"),
            "ttft_p99": _carl_vs_static_from_summary(carl, static, n, "ttft_p99"),
        }
    return {"n_seeds": n, "per_workload": out}


def analyze_ablation(ablation: dict) -> dict:
    n = ablation.get("settings", {}).get("runs")
    out: dict = {}
    for sname, configs in ablation.get("scenarios", {}).items():
        carl = configs.get("CARL-Full")
        static = configs.get("Static-Best")
        if not carl or not static:
            continue
        out[sname] = {
            "throughput": _carl_vs_static_from_summary(carl, static, n, "throughput"),
            "ttft_p99": _carl_vs_static_from_summary(carl, static, n, "ttft_p99"),
        }
    return {"n_seeds": n, "per_scenario": out}


def analyze_sensitivity(sens: dict) -> dict:
    n = sens.get("settings", {}).get("runs")
    out: dict = {}
    for sweep, settings in sens.get("sweeps", {}).items():
        sweep_out: dict = {}
        for setting, methods in settings.items():
            carl = methods.get("carl")
            static = methods.get("static")
            if not carl:
                continue
            entry = {"carl_throughput_ci95": ci95_from_summary(
                carl.get("throughput_mean"), carl.get("throughput_std"), n)}
            if static:
                entry["static_throughput_ci95"] = ci95_from_summary(
                    static.get("throughput_mean"), static.get("throughput_std"), n)
                entry["difference_mean"] = (carl.get("throughput_mean", 0.0)
                                            - static.get("throughput_mean", 0.0))
                entry["cohens_d_summary"] = cohens_d_pooled(
                    carl.get("throughput_mean"), carl.get("throughput_std"), n,
                    static.get("throughput_mean"), static.get("throughput_std"), n)
            sweep_out[setting] = entry
        out[sweep] = sweep_out
    return {"n_seeds": n, "per_sweep": out,
            "note": "CARL vs Static throughput per setting (summary-stat Cohen's d)."}


def analyze_oracle(oracle: dict) -> dict:
    """CARL's throughput gap to the per-phase oracle, with a CI over phases."""
    gaps = oracle.get("gaps", {})
    phases = oracle.get("phase_names", [])
    out: dict = {"phase_names": phases}
    for method, g in gaps.items():
        tg = g.get("throughput_gap_pct")
        if not tg:
            continue
        out[method] = {
            "throughput_gap_pct_per_phase": tg,
            "throughput_gap_pct_ci95": ci95_from_raw(tg),
        }
    out["note"] = ("Gap = (oracle - method)/oracle * 100 per phase; CI is across "
                   "the phases, not seeds.")
    return out


# ===========================================================================
# Driver.
# ===========================================================================


def run() -> dict:
    stats = _load("stats_results.json")
    workload = _load("workload_results.json")
    oracle = _load("oracle_results.json")
    ablation = _load("ablation_results.json")
    sensitivity = _load("sensitivity_results.json")

    missing = [n for n, d in (
        ("stats_results.json", stats), ("workload_results.json", workload),
        ("oracle_results.json", oracle), ("ablation_results.json", ablation),
        ("sensitivity_results.json", sensitivity)) if d is None]

    results: dict = {
        "description": ("Offline statistical validation of CARL's throughput / "
                        "TTFT claims: 95% CIs, Cohen's d, and t-tests across the "
                        "simulation eval suite. Pure-JSON post-processing -- no GPU."),
        "method_notes": {
            "ci95": "mean +/- t_{.975, n-1} * std / sqrt(n)",
            "paired_ttest": ("per-seed paired t-test on CARL-Static throughput; "
                             "only stats_results.json keeps the raw per-seed arrays"),
            "welch_ttest": ("two-sample unequal-variance t-test from summary "
                            "mean/std/n; used where raw seeds are absent (paired=false)"),
            "cohens_d_paired": "mean(diff) / std(diff) over seeds",
            "cohens_d_summary": "pooled-SD Cohen's d from summary stats (not paired)",
            "tdist": "Student-t CDF via regularised incomplete beta; no scipy",
        },
        "inputs_missing": missing,
    }

    if stats is not None:
        results["headline_non_stationary"] = analyze_headline(stats)
    if workload is not None:
        results["workloads"] = analyze_workloads(workload)
    if ablation is not None:
        results["ablation"] = analyze_ablation(ablation)
    if sensitivity is not None:
        results["sensitivity"] = analyze_sensitivity(sensitivity)
    if oracle is not None:
        results["oracle_gap"] = analyze_oracle(oracle)

    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved statistical validation to {OUTPUT_PATH}", flush=True)
    return results


def _print(r: dict) -> None:
    print("=== STATISTICAL VALIDATION SUMMARY (offline, simulation suite) ===")
    if r.get("inputs_missing"):
        print("  MISSING INPUTS:", ", ".join(r["inputs_missing"]))

    h = r.get("headline_non_stationary", {})
    pt = h.get("paired_ttest_throughput")
    if pt:
        cf = h["carl_full_throughput_ci95"]["ci95"]
        sb = h["static_best_throughput_ci95"]["ci95"]
        print(f"\nHEADLINE ({h.get('workload')}, n={pt['n']} seeds, PAIRED):")
        print(f"  CARL-Full   throughput 95% CI: [{cf[0]:.3f}, {cf[1]:.3f}]")
        print(f"  Static-Best throughput 95% CI: [{sb[0]:.3f}, {sb[1]:.3f}]")
        print(f"  paired diff {pt['mean_difference']:+.3f} "
              f"CI [{pt['ci95_difference'][0]:.3f}, {pt['ci95_difference'][1]:.3f}], "
              f"t={pt['t_statistic']:.2f}, df={pt['df']}, p={pt['p_value']:.2e}, "
              f"Cohen's d={pt['cohens_d_paired']:.2f}, "
              f"{'SIGNIFICANT' if pt['significant_at_0.05'] else 'n.s.'}")
        cc = h.get("crosscheck_vs_generator", {})
        print(f"  crosscheck vs generator: t_matches={cc.get('t_matches')} "
              f"(reported p={cc.get('reported_p'):.2e}, recomputed p={cc.get('recomputed_p'):.2e})")

    def _dump_pairs(title, block, key):
        if not block:
            return
        print(f"\n{title} (n={block.get('n_seeds')} seeds, summary-stat Welch + d):")
        for name, m in block.get(key, {}).items():
            tp = m.get("throughput", {})
            d = tp.get("cohens_d_summary")
            w = tp.get("welch_ttest", {})
            diff = tp.get("difference_mean")
            if d is None or diff is None:
                continue
            p = w.get("p_value")
            pstr = f"{p:.2e}" if isinstance(p, float) else str(p)
            print(f"  {name:<16} tput diff {diff:+7.2f}  d={d:7.2f}  Welch p={pstr}")

    _dump_pairs("WORKLOADS", r.get("workloads"), "per_workload")
    _dump_pairs("ABLATION", r.get("ablation"), "per_scenario")

    og = r.get("oracle_gap", {})
    if og:
        print("\nORACLE GAP (CARL throughput gap %, CI across phases):")
        for method, m in og.items():
            if not isinstance(m, dict) or "throughput_gap_pct_ci95" not in m:
                continue
            ci = m["throughput_gap_pct_ci95"].get("ci95")
            mean = m["throughput_gap_pct_ci95"].get("mean")
            if ci:
                print(f"  {method:<16} mean {mean:.2f}%  CI [{ci[0]:.2f}, {ci[1]:.2f}]")


if __name__ == "__main__":
    run()
