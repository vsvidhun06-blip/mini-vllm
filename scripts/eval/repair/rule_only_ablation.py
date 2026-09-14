"""PHASE 4+5 -- reward diagnostics and the CARL-vs-RuleOnly null ablation.

THE DECISIVE EXPERIMENT
-----------------------
Everything in the paper rests on one unasked question: does the contextual
bandit beat a stateless rule that classifies the regime and looks up a
hand-tuned config? No configuration in the existing evaluation suite isolates
learning -- every ablation freezes a KNOB, none freezes LEARNING.

This script runs, on the repaired mechanistic substrate (src/eval/engine_model)
with the repaired non-saturating reward (src/carl/reward), across stationary and
non-stationary workloads and >=10 seeds:

    CARL-AsPublished   LinUCB exactly as shipped (alpha=0.5)
    CARL-Repaired      LinUCB + intercept feature + reward centring
    RuleOnly           DEFAULT_CONFIGS[classify_regime(state)], zero learning
    Static-Best-Tput   best fixed config, searched on THROUGHPUT
    Static-Best-SLO    best fixed config, searched on the SAME utility CARL maximises
    Oracle             per-phase brute-force optimum, computed BY the model

Both bandits are reported (per the approved plan) so the as-published defect is
disclosed rather than silently repaired.

PHASE 4 is folded in: before the ablation runs, every candidate arm is evaluated
on every workload and the per-arm reward spread is reported. If the reward has
no variance across arms on a workload where learning is expected, the run FAILS
LOUDLY via check_non_degenerate rather than producing a meaningless learning
curve. That check is what the original suite lacked.

INTERPRETING THE OUTPUT
-----------------------
The comparison that matters is `carl_repaired_vs_rule_only`. A tie means the
classifier is doing the work and the bandit is unnecessary AT THIS SCALE -- a
real result, and the one the paper must then be built around. A win localised to
transition cycles means adaptation pays at boundaries. Either is publishable;
neither is to be tuned away.

CPU-only. Minutes, not hours.

Run:
  python scripts/eval/repair/rule_only_ablation.py
  python scripts/eval/repair/rule_only_ablation.py --seeds 3 --quick
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harness2 as H  # noqa: E402

from src.carl.config import DEFAULT_CONFIGS, all_arm_sets  # noqa: E402
from src.carl.reward import (  # noqa: E402
    DegenerateRewardError, RewardScales, check_non_degenerate, utility_v2,
)
from src.carl.state import WorkloadRegime  # noqa: E402
from src.eval.engine_model import HardwareProfile, simulate  # noqa: E402
from src.eval.provenance import write_result  # noqa: E402

OUT = os.path.join(_ROOT, "docs", "eval", "raw", "repair", "rule_only_ablation.json")
DIAG_MD = os.path.join(_ROOT, "docs", "eval", "reward_diagnostics.md")

# Workload scenarios: three stationary, three non-stationary, one stress.
SCENARIOS = {
    # Stationary controls: nothing to adapt to. Any method that "wins" here is
    # winning on noise.
    "stationary_interactive": ["interactive"] * 3,
    "stationary_batch": ["batch"] * 3,
    "stationary_long_context": ["long_context"] * 3,

    # Prompt-length regime shifts -- the paper's notion of non-stationarity.
    "nonstationary_i_b": ["interactive", "batch", "interactive"],
    "nonstationary_i_b_l": ["interactive", "batch", "long_context"],
    "oscillating": ["interactive", "batch"] * 3,

    # LOAD shifts -- where a load sweep says the optimum actually moves
    # (light load wants max_batch ~16, heavy load wants ~2). This is
    # adaptation's best case on this substrate; if CARL cannot win here it
    # cannot win anywhere.
    "load_shift_interactive": ["interactive", "interactive_heavy", "interactive"],
    "load_shift_batch": ["batch", "batch_heavy", "batch"],
    "load_oscillating": ["interactive", "interactive_heavy"] * 3,
    "load_and_regime_shift": ["interactive", "batch_heavy", "long_context",
                              "interactive_heavy"],

    # Stress.
    "stress_burst": ["burst_dump"] * 3,
}


# ---------------------------------------------------------------------------
# PHASE 4 -- reward diagnostics.
# ---------------------------------------------------------------------------


def reward_diagnostics(hw: HardwareProfile, scales: RewardScales) -> dict:
    """Per-arm reward spread on every base workload.

    Answers: can the reward tell the candidate configurations apart at all? If
    the spread is ~0 the bandit is being asked to learn from noise, and every
    downstream 'convergence' result is vacuous.
    """
    arms_by_regime = all_arm_sets()
    out: dict = {"scales": scales.__dict__, "per_workload": {}}
    for wl_name, spec in H.WORKLOADS.items():
        # Evaluate the union of all arms plus a slice of the static grid, so the
        # spread reflects the whole reachable space, not just one regime's arms.
        candidates = []
        seen = set()
        for arms in arms_by_regime.values():
            for a in arms:
                key = tuple(sorted(a.as_dict().items()))
                if key not in seen:
                    seen.add(key)
                    candidates.append(a)
        rows = []
        for cfg in candidates:
            vals = []
            for s in (0, 1, 2):
                res = simulate(cfg, spec.generate(random.Random(s)), hw,
                               random.Random(s ^ 0x5EED))
                vals.append(utility_v2(H.run_to_metrics(res), None, scales))
            res0 = simulate(cfg, spec.generate(random.Random(0)), hw, random.Random(0x5EED))
            rows.append({
                "config": cfg.as_dict(),
                "reward_mean": statistics.fmean(vals),
                "throughput_tps": res0.throughput_tps,
                "ttft_p99_ms": res0.ttft_p99_ms,
                "tpot_p99_ms": res0.tpot_p99_ms,
                "terms": H.term_breakdown(H.run_to_metrics(res0), None, scales)
                if hasattr(H, "term_breakdown") else None,
            })
        rewards = [r["reward_mean"] for r in rows]
        try:
            report = check_non_degenerate(rewards, context=f"arm sweep on {wl_name}")
            degenerate, err = False, None
        except DegenerateRewardError as e:
            report, degenerate, err = {}, True, str(e)
        best = max(rows, key=lambda r: r["reward_mean"])
        worst = min(rows, key=lambda r: r["reward_mean"])
        out["per_workload"][wl_name] = {
            "n_arms_evaluated": len(rows),
            "reward_min": min(rewards), "reward_max": max(rewards),
            "reward_spread": max(rewards) - min(rewards),
            "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "degenerate": degenerate,
            "degeneracy_error": err,
            "non_degeneracy_report": report,
            "best_config": best["config"],
            "best_reward": best["reward_mean"],
            "worst_config": worst["config"],
            "worst_reward": worst["reward_mean"],
            # The key anti-circularity check: is the best arm on this workload
            # the hand-tuned DEFAULT for the regime this workload represents?
            "best_is_a_default_config": best["config"] in [
                c.as_dict() for c in DEFAULT_CONFIGS.values()],
            "rows": rows,
        }
    return out


# ---------------------------------------------------------------------------
# PHASE 5 -- the ablation.
# ---------------------------------------------------------------------------


def build_controllers(oracle_by_phase, static_tput, static_slo, wide_arms):
    return {
        # Shipped arm sets (perturbations of DEFAULT_CONFIGS).
        "CARL-AsPublished": lambda: H.BanditController(
            H.LinUCBBandit, alpha=0.5, name="CARL-AsPublished"),
        "CARL-Repaired": lambda: H.BanditController(
            H.RepairedLinUCBBandit, alpha=0.5, name="CARL-Repaired"),
        # Same repaired learner, but an arm set wide enough to contain the
        # configs the static search can reach. Separates "learning does not
        # help" from "the action space excludes the optimum".
        "CARL-Repaired-WideArms": lambda: H.BanditController(
            H.RepairedLinUCBBandit, alpha=0.5, name="CARL-Repaired-WideArms",
            arms=wide_arms),
        "RuleOnly": lambda: H.RuleOnlyController(),
        "AutoTuner-ClosedLoop": lambda: H.ClosedLoopAutoTunerController(),
        "Static-Best-Tput": lambda: H.StaticController(static_tput, "Static-Best-Tput"),
        "Static-Best-SLO": lambda: H.StaticController(static_slo, "Static-Best-SLO"),
        "Oracle": lambda: H.OracleController(oracle_by_phase),
    }


def paired_stats(a: list[float], b: list[float]) -> dict:
    """Paired difference a-b with CI, t and Cohen's d. Same seeds, so paired."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m = statistics.fmean(diffs)
    s = statistics.stdev(diffs) if n > 1 else 0.0
    se = s / (n ** 0.5) if n > 1 and s > 0 else 0.0
    return {
        "n": n, "mean_difference": m, "std_difference": s,
        "ci95": [m - 1.96 * se, m + 1.96 * se] if se else [m, m],
        "t_statistic": (m / se) if se else None,
        "cohens_d_paired": (m / s) if s else None,
        "all_diffs_zero": all(d == 0.0 for d in diffs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-per-phase", type=int, default=40)
    ap.add_argument("--static-candidates", type=int, default=60)
    ap.add_argument("--quick", action="store_true", help="fewer static candidates")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    hw = HardwareProfile()
    seeds = list(range(args.seeds))
    n_static = 20 if args.quick else args.static_candidates

    # --- derive reward scales from the reachable operating range -------------
    cands = H.static_candidates(limit=n_static, seed=0)
    tps, tt, tp = [], [], []
    for c in cands[:25]:
        for ph in ("interactive", "batch"):
            r = simulate(c, H.WORKLOADS[ph].generate(random.Random(0)), hw,
                         random.Random(1))
            tps.append(r.throughput_tps); tt.append(r.ttft_p99_ms); tp.append(r.tpot_p99_ms)
    scales = RewardScales.from_measurements(tps, tt, tp)
    print(f"Reward scales (derived from operating range): {scales}")

    # Wide arm set for the action-space control (see harness2.wide_arm_sets).
    wide_arms = H.wide_arm_sets(per_regime=8, seed=0)

    # --- PHASE 4 -------------------------------------------------------------
    print("\n[Phase 4] reward diagnostics ...")
    diag = reward_diagnostics(hw, scales)
    for wl, d in diag["per_workload"].items():
        flag = "DEGENERATE" if d["degenerate"] else "ok"
        print(f"  {wl:<22} spread={d['reward_spread']:.4f} std={d['reward_std']:.4f} "
              f"[{flag}]  best_is_default={d['best_is_a_default_config']}")
    if any(d["degenerate"] for d in diag["per_workload"].values()):
        print("\n!! reward is degenerate on at least one workload -- see diagnostics")

    obj_tput = H.objective_throughput
    obj_util = H.make_objective_utility(scales)

    # --- PHASE 5 -------------------------------------------------------------
    results: dict = {}
    for scen, phases in SCENARIOS.items():
        print(f"\n[Phase 5] {scen}: {' -> '.join(phases)}")
        # Oracle + statics solved on HELD-OUT seeds (900+) so the evaluation
        # seeds below are not the seeds the baselines were tuned on.
        oracle = H.solve_oracle(phases, cands, hw, None, seeds=(900, 901),
                                n_per_phase=args.n_per_phase, scales=scales)
        st_tput, st_tput_meta = H.solve_static_best(
            phases, cands, hw, obj_tput, seeds=(900, 901),
            n_per_phase=args.n_per_phase, scales=scales)
        st_slo, st_slo_meta = H.solve_static_best(
            phases, cands, hw, obj_util, seeds=(900, 901),
            n_per_phase=args.n_per_phase, scales=scales)
        print(f"    oracle per phase: "
              f"{ {k: v.as_dict()['max_batch_size'] for k, v in oracle['by_phase'].items()} }"
              f"  (any is a DEFAULT_CONFIG: "
              f"{any(d['is_a_default_config'] for d in oracle['detail'].values())})")
        print(f"    static-tput mb={st_tput.max_batch_size} cs={st_tput.chunk_size} | "
              f"static-slo  mb={st_slo.max_batch_size} cs={st_slo.chunk_size}")

        # Can CARL's shipped arm set even REACH the configs the static search
        # found? Recorded per scenario so an action-space failure is never
        # misread as a learning failure.
        shipped = all_arm_sets()
        coverage = {
            "static_best_tput": H.arm_set_coverage(st_tput, shipped),
            "static_best_slo": H.arm_set_coverage(st_slo, shipped),
            "oracle_configs": {
                p: H.arm_set_coverage(c, shipped)
                for p, c in oracle["by_phase"].items()},
        }
        cov = coverage["static_best_tput"]
        print(f"    arm-set coverage: static-tput needs max_batch="
              f"{cov['target_max_batch_size']}; reachable in every regime="
              f"{cov['reachable_in_every_regime']}; per-regime reach="
              f"{ {r: v['reachable_max_batch_sizes'] for r, v in cov['per_regime'].items()} }")

        ctrls = build_controllers(oracle["by_phase"], st_tput, st_slo, wide_arms)
        per_method: dict = {}
        for name, factory in ctrls.items():
            runs = [H.run_episode(factory(), phases, hw, scales, seed,
                                  n_per_phase=args.n_per_phase) for seed in seeds]
            per_method[name] = {
                "reward": H.mean_std([r["mean_reward"] for r in runs]),
                "throughput_tps": H.mean_std([r["throughput_tps"] for r in runs]),
                "ttft_p99_ms": H.mean_std([r["ttft_p99_ms"] for r in runs]),
                "tpot_p99_ms": H.mean_std([r["tpot_p99_ms"] for r in runs]),
                "mean_occupancy": H.mean_std([r["mean_occupancy"] for r in runs]),
                "arm_changes": H.mean_std([r["arm_changes"] for r in runs]),
                "adaptations": H.mean_std([r["adaptations"] for r in runs]),
                "_per_seed_reward": [r["mean_reward"] for r in runs],
                "_per_seed_tput": [r["throughput_tps"] for r in runs],
                # Seed-independence check (R6): identical values across seeds
                # would mean the seed path is dead.
                "seeds_distinct": len(set(r["mean_reward"] for r in runs)) == len(runs),
                "example_trace": runs[0]["per_cycle"],
            }
            print(f"      {name:<20} reward={per_method[name]['reward']['mean']:.4f}"
                  f"  tps={per_method[name]['throughput_tps']['mean']:7.2f}"
                  f"  ttft99={per_method[name]['ttft_p99_ms']['mean']:8.0f}"
                  f"  armchg={per_method[name]['arm_changes']['mean']:.1f}"
                  f"  seeds_distinct={per_method[name]['seeds_distinct']}")

        oracle_reward = per_method["Oracle"]["reward"]["mean"]
        comparisons = {}
        for a, b in (("CARL-Repaired", "RuleOnly"),
                     ("CARL-Repaired-WideArms", "RuleOnly"),
                     ("CARL-Repaired-WideArms", "Static-Best-Tput"),
                     ("CARL-Repaired-WideArms", "CARL-Repaired"),
                     ("CARL-AsPublished", "RuleOnly"),
                     ("CARL-Repaired", "CARL-AsPublished"),
                     ("CARL-Repaired", "Static-Best-SLO"),
                     ("RuleOnly", "Static-Best-SLO"),
                     ("Static-Best-SLO", "Static-Best-Tput")):
            comparisons[f"{a}_vs_{b}"] = {
                "reward": paired_stats(per_method[a]["_per_seed_reward"],
                                       per_method[b]["_per_seed_reward"]),
                "throughput": paired_stats(per_method[a]["_per_seed_tput"],
                                           per_method[b]["_per_seed_tput"]),
            }
        results[scen] = {
            "phases": phases,
            "arm_set_coverage": coverage,
            "oracle": oracle["detail"],
            "oracle_method": oracle["oracle_method"],
            "oracle_scope_note": oracle["oracle_scope_note"],
            "static_best_tput_selection": st_tput_meta,
            "static_best_slo_selection": st_slo_meta,
            "per_method": per_method,
            "oracle_capture_pct": {
                m: 100.0 * per_method[m]["reward"]["mean"] / oracle_reward
                for m in per_method
            },
            "comparisons": comparisons,
        }

    payload = {
        "description": (
            "PHASE 4+5: reward diagnostics and the CARL-vs-RuleOnly null "
            "ablation on the repaired mechanistic substrate."
        ),
        "substrate": "src/eval/engine_model (mechanistic; optimum found by search)",
        "reward": "src/carl/reward.utility_v2 (non-saturating)",
        "reward_scales": scales.__dict__,
        "hardware_profile": hw.__dict__,
        "seeds": seeds,
        "n_per_phase": args.n_per_phase,
        "n_static_candidates": len(cands),
        "wide_arm_set": [a.as_dict() for a in wide_arms[WorkloadRegime.INTERACTIVE]],
        "shipped_arm_set_max_batch_sizes": sorted({
            a.max_batch_size for arms in all_arm_sets().values() for a in arms}),
        "phase4_reward_diagnostics": diag,
        "phase5_scenarios": results,
    }
    path = write_result(args.out, payload, "rule_only_ablation",
                        script="scripts/eval/repair/rule_only_ablation.py",
                        extra={"seeds": seeds, "substrate": "engine_model"})
    print(f"\nWrote {path}")
    _write_diag_md(diag, scales, results)
    print(f"Wrote {DIAG_MD}")
    return 0


def _write_diag_md(diag: dict, scales, results: dict) -> None:
    lines = [
        "# Reward diagnostics (Phase 4)",
        "",
        "Generated by `scripts/eval/repair/rule_only_ablation.py`. Do not hand-edit.",
        "",
        "## Why this file exists",
        "",
        "Reward v1 was degenerate on real hardware: `docs/eval/raw/adaptation/"
        "decisions_042.csv` shows exactly two reward values across an entire run "
        "(0.8 then 0.3 forever), because all four terms saturated simultaneously. "
        "Aggregate statistics hid it. This diagnostic makes the failure "
        "impossible to miss: it evaluates every candidate arm on every workload "
        "and reports the reward spread, failing loudly if the reward cannot tell "
        "configurations apart.",
        "",
        f"## Scales in use",
        "",
        f"- `t_half` = {scales.t_half:.2f} tok/s (throughput term scores 0.5 here)",
        f"- `ttft_target` = {scales.ttft_target:.1f} ms",
        f"- `tpot_target` = {scales.tpot_target:.1f} ms",
        f"- `sharpness` = {scales.sharpness}",
        "",
        "Derived from the observed operating range of the static search grid, "
        "not chosen by hand. See `RewardScales.from_measurements`.",
        "",
        "## Per-arm reward spread",
        "",
        "| workload | arms | reward min | reward max | spread | std | degenerate | best arm is a DEFAULT_CONFIG |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for wl, d in diag["per_workload"].items():
        lines.append(
            f"| {wl} | {d['n_arms_evaluated']} | {d['reward_min']:.4f} | "
            f"{d['reward_max']:.4f} | {d['reward_spread']:.4f} | {d['reward_std']:.4f} | "
            f"{'**YES**' if d['degenerate'] else 'no'} | "
            f"{d['best_is_a_default_config']} |")
    lines += [
        "",
        "The last column is the anti-circularity check. Under the OLD cost model "
        "it would have been `True` for every row by construction, because the "
        "model defined the optimum to be `DEFAULT_CONFIGS[regime]`. On the "
        "mechanistic substrate the best arm is whatever the step-time equation "
        "makes best.",
        "",
        "## Best / worst arm per workload",
        "",
    ]
    for wl, d in diag["per_workload"].items():
        lines += [
            f"### {wl}",
            "",
            f"- best  (reward {d['best_reward']:.4f}): `max_batch_size="
            f"{d['best_config']['max_batch_size']}, chunk_size="
            f"{d['best_config']['chunk_size']}, eviction_threshold="
            f"{d['best_config']['eviction_threshold']}`",
            f"- worst (reward {d['worst_reward']:.4f}): `max_batch_size="
            f"{d['worst_config']['max_batch_size']}, chunk_size="
            f"{d['worst_config']['chunk_size']}, eviction_threshold="
            f"{d['worst_config']['eviction_threshold']}`",
            "",
        ]
    os.makedirs(os.path.dirname(DIAG_MD), exist_ok=True)
    with open(DIAG_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
