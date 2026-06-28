"""
Regime-classifier robustness analysis for CARL (GPU).

QUESTION
--------
classify_regime (src/carl/state.py) is a rule-based classifier with FIXED
thresholds (avg_prompt_len > 512 -> long context, queue_depth >= 8 -> batch,
etc.). Those numbers were hand-picked. How sensitive is CARL's end-to-end
behaviour to getting them "right"? If a +/-30% wobble on every threshold barely
moves throughput / regret / adaptation, the classifier is a robust, low-stakes
component (the bandit underneath recovers from a mislabelled regime); if it
swings, the thresholds are load-bearing and must be tuned carefully.

INTERVENTION
------------
We scale ALL classifier thresholds by a common factor and re-run CARL on the
non-stationary mixed workload:

    factor in { 0.7, 0.8, 0.9, [1.0 baseline], 1.1, 1.2, 1.3 }   (i.e. -30%..+30%)

For each level we report, over 3 seeds (42, 43, 44):
  * throughput               (mean +/- std, tok/s -- the regime-independent signal)
  * cumulative regret        (vs a FIXED default-threshold oracle)
  * time-to-adaptation       (control cycles to convergence)
and the delta vs the default-threshold baseline.

The thresholds are perturbed by MONKEYPATCHING the module globals
classify_regime reads at call time -- nothing on disk is modified, and the
originals are restored in a finally. classify_regime is the ONLY thing that
changes; the workload, scheduler, bandit, reward, and SLO are byte-identical to
ablation_live's CARL-Full.

REGRET CAVEAT (stated plainly)
------------------------------
The oracle is the static best-arm-per-regime mean reward computed from the
DEFAULT-threshold baseline (the ablation's DynOracle), defined only for the
regimes the default classifier visits here (INTERACTIVE, BATCH). A perturbation
aggressive enough to reclassify cycles into OTHER regimes (e.g. BURST) accrues no
modelled regret for those cycles, so cumulative regret is a LOWER bound under
large perturbations. Throughput is the clean, regime-independent robustness
metric; read it as primary and regret/t2a as supporting.

REUSE (nothing existing is modified)
------------------------------------
Imports the live CARL-Full path (run_config), the oracle builder
(compute_dynoracle_arms), and OBSERVE_INTERVAL / _REGIMES / capture_environment
from ablation_live; the pure regret/convergence analysis from src/carl/adaptation;
and ControllerLogEntry from the controller. Writes ONLY
docs/eval/classifier_robustness_results.json.

SCOPE: same single-model caveat as ablation_live (scheduler-only, spec off).

Run:
  python scripts/eval/classifier_robustness.py                 # seeds 42,43,44 x 50
  python scripts/eval/classifier_robustness.py --seeds 42 --limit 30   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
from datetime import datetime

# --- path bootstrap so `python scripts/eval/classifier_robustness.py` finds src/ -
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

# Reuse the ablation's CARL-Full serving + oracle machinery verbatim.
from scripts.eval.ablation_live import (  # noqa: E402
    OBSERVE_INTERVAL, _REGIMES, capture_environment, compute_dynoracle_arms, run_config,
)
import src.carl.state as _state  # noqa: E402  (monkeypatch target for thresholds)
from src.carl.adaptation import decision_rows, summarize  # noqa: E402
from src.carl.config import all_arm_sets  # noqa: E402
from src.carl.controller import ControllerLogEntry  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RESULTS_PATH = os.path.join(DOCS_EVAL, "classifier_robustness_results.json")

DEFAULT_SEEDS = [42, 43, 44]
SCENARIO = "NON-STATIONARY"

# The fixed thresholds classify_regime reads (the perturbation targets). Captured
# once at import so we always scale from and restore to the true defaults.
_THRESHOLD_NAMES = [
    "_LONG_CONTEXT_PROMPT_TOKENS",
    "_CACHE_HEAVY_HIT_RATE",
    "_BURST_QUEUE_DEPTH",
    "_BURST_BACKLOG_RATIO",
    "_BATCH_QUEUE_DEPTH",
    "_BATCH_PROMPT_TOKENS",
]
_DEFAULT_THRESHOLDS = {n: getattr(_state, n) for n in _THRESHOLD_NAMES}

# Perturbation levels: -30%..+30% on EVERY threshold; 1.0 is the default baseline.
_PERTURBATIONS = [
    ("-30%", 0.7), ("-20%", 0.8), ("-10%", 0.9),
    ("+10%", 1.1), ("+20%", 1.2), ("+30%", 1.3),
]


# ===========================================================================
# Threshold monkeypatch helpers (restore in finally -- never leak a perturbation).
# ===========================================================================


def _apply_factor(factor: float) -> dict:
    """Scale every classifier threshold by `factor`; return the applied values."""
    applied = {}
    for n in _THRESHOLD_NAMES:
        v = _DEFAULT_THRESHOLDS[n] * factor
        setattr(_state, n, v)
        applied[n] = v
    return applied


def _restore_thresholds() -> None:
    for n in _THRESHOLD_NAMES:
        setattr(_state, n, _DEFAULT_THRESHOLDS[n])


# ===========================================================================
# Analysis helpers (pure; reuse src/carl/adaptation).
# ===========================================================================


class _ArmsView:
    """Minimal `.arms(regime)` adapter over the static per-regime arm sets."""

    def __init__(self) -> None:
        self._arms = all_arm_sets()

    def arms(self, regime):
        return self._arms[regime]


def _log_from_decisions(decisions: list) -> list:
    """Rebuild a ControllerLogEntry list from run_config's `decisions` records.

    Step is synthesised at the controller cadence (cycle * OBSERVE_INTERVAL), the
    same reconstruction adaptation_analysis.py uses.
    """
    return [
        ControllerLogEntry(step=i * OBSERVE_INTERVAL, regime=d["regime"],
                           config=d["config"], reward=d["reward"], state_features=[])
        for i, d in enumerate(decisions)
    ]


def _mean(vals: list) -> float:
    return statistics.fmean(vals) if vals else 0.0


def _std(vals: list) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _run_level(model, tokenizer, n, seeds, factor, arms_view,
               oracle_reward_by_regime) -> tuple:
    """Run all seeds at one threshold factor; return (aggregate, pooled_decisions).

    pooled_decisions is only consumed for the baseline (factor 1.0) to build the
    oracle; for perturbed levels it is ignored.
    """
    throughputs, regrets, t2as = [], [], []
    per_seed: dict = {}
    pooled: list = []
    for seed in seeds:
        try:
            out = run_config("CARL-Full", model, tokenizer, n, seed)
        except Exception:
            print(f"    seed {seed}: FAILED", flush=True)
            traceback.print_exc()
            continue
        decs = out.get("decisions", [])
        pooled.extend(decs)
        tput = out["throughput_tps"]
        # Regret / convergence vs the shared oracle (None for the baseline's own
        # pre-oracle pass -> filled on the analysis pass below).
        if oracle_reward_by_regime is not None:
            rows = decision_rows(_log_from_decisions(decs), arms_view,
                                 oracle_reward_by_regime)
            s = summarize(rows)
            regret = s["total_cumulative_regret"]
            t2a = s["convergence_point"]["cycle"]
        else:
            regret, t2a = None, None
        throughputs.append(tput)
        if regret is not None:
            regrets.append(regret)
        if t2a is not None:
            t2as.append(float(t2a))
        per_seed[str(seed)] = {"throughput_tps": tput, "cumulative_regret": regret,
                               "time_to_adaptation": t2a}
        print(f"    seed {seed}: {tput:6.1f} tok/s"
              + (f", regret {regret:.3f}, t2a {t2a}" if regret is not None else ""),
              flush=True)
    agg = {
        "throughput_tps_mean": _mean(throughputs), "throughput_tps_std": _std(throughputs),
        "cumulative_regret_mean": _mean(regrets), "cumulative_regret_std": _std(regrets),
        "time_to_adaptation_mean": _mean(t2as), "time_to_adaptation_std": _std(t2as),
        "per_seed": per_seed,
    }
    return agg, pooled


# ===========================================================================
# Driver.
# ===========================================================================


def run_all(seeds: list, n: int) -> dict:
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | classifier robustness on {SCENARIO} | "
          f"seeds {seeds} x {n} requests | {len(_PERTURBATIONS)} levels + baseline",
          flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab T4 for real numbers.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    arms_view = _ArmsView()

    try:
        # --- Baseline (default thresholds, factor 1.0). Two-step: run once to pool
        #     decisions for the oracle, then score the SAME runs against it. ------
        print("\n[baseline] default thresholds (factor 1.0)", flush=True)
        _apply_factor(1.0)   # == defaults; explicit for symmetry
        # First pass: gather decisions (no oracle yet) to build it.
        _pre, pooled = _run_level(model, tokenizer, n, seeds, 1.0, arms_view,
                                  oracle_reward_by_regime=None)
        _oracle_arms, oracle_meta = compute_dynoracle_arms(pooled)
        oracle_reward_by_regime = {r.value: oracle_meta[r.value]["mean_reward"]
                                   for r in _REGIMES}
        print(f"  oracle (best-arm-per-regime mean reward): "
              f"{ {k: round(v, 4) for k, v in oracle_reward_by_regime.items()} }",
              flush=True)
        # Second pass: re-run the baseline scored against the oracle (so baseline
        # regret/t2a are measured on the same footing as the perturbations).
        baseline, _ = _run_level(model, tokenizer, n, seeds, 1.0, arms_view,
                                 oracle_reward_by_regime)
        baseline["factor"] = 1.0
        baseline["thresholds"] = dict(_DEFAULT_THRESHOLDS)

        # --- Perturbations. -----------------------------------------------------
        perturbations: dict = {}
        for label, factor in _PERTURBATIONS:
            print(f"\n[{label}] all thresholds x {factor}", flush=True)
            applied = _apply_factor(factor)
            try:
                agg, _ = _run_level(model, tokenizer, n, seeds, factor, arms_view,
                                    oracle_reward_by_regime)
            finally:
                _restore_thresholds()   # restore before the next level no matter what
            agg["factor"] = factor
            agg["thresholds"] = applied
            agg["delta_throughput_tps_mean"] = (
                agg["throughput_tps_mean"] - baseline["throughput_tps_mean"])
            agg["delta_cumulative_regret_mean"] = (
                agg["cumulative_regret_mean"] - baseline["cumulative_regret_mean"])
            agg["delta_time_to_adaptation_mean"] = (
                agg["time_to_adaptation_mean"] - baseline["time_to_adaptation_mean"])
            perturbations[label] = agg
            print(f"  tput {agg['throughput_tps_mean']:.1f} "
                  f"(delta {agg['delta_throughput_tps_mean']:+.1f}), "
                  f"regret {agg['cumulative_regret_mean']:.3f} "
                  f"(delta {agg['delta_cumulative_regret_mean']:+.3f}), "
                  f"t2a {agg['time_to_adaptation_mean']:.1f}", flush=True)
    finally:
        _restore_thresholds()   # belt-and-braces: never leave thresholds perturbed

    results = {
        "experiment": "classifier_robustness",
        "scenario": SCENARIO,
        "method": ("CARL-Full with all classify_regime thresholds scaled by a "
                   "common factor; everything else identical to ablation_live."),
        "seeds": list(baseline["per_seed"].keys()),
        "requests_per_run": n,
        "observe_interval": OBSERVE_INTERVAL,
        "environment": env,
        "default_thresholds": dict(_DEFAULT_THRESHOLDS),
        "perturbation_levels": [f for _, f in _PERTURBATIONS],
        "regret_model": ("static best-arm-per-regime oracle (DynOracle mean reward "
                         "from the DEFAULT-threshold baseline, pooled over seeds); "
                         "reused for every level. Under large perturbations regret "
                         "is a LOWER bound (cycles reclassified into regimes absent "
                         "from the oracle accrue no modelled regret)."),
        "oracle_reward_by_regime": oracle_reward_by_regime,
        "oracle_arms_per_regime": oracle_meta,
        "baseline": baseline,
        "perturbations": perturbations,
        "scope_note": ("Single-model live harness: CARL wired to the scheduler "
                       "only, speculation off, no router, KV eviction inactive. "
                       "cache/spec features are ~0 so cache/spec thresholds are "
                       "inert here; throughput is the primary robustness signal. "
                       "See docs/eval/README.md."),
        "generated": datetime.now().isoformat(),
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved classifier-robustness results to {RESULTS_PATH}", flush=True)
    return results


def _print(results: dict) -> None:
    print("\n=== REGIME-CLASSIFIER ROBUSTNESS (all thresholds scaled; vs default) ===")
    b = results["baseline"]
    print(f"baseline (1.0): {b['throughput_tps_mean']:.1f} +/- "
          f"{b['throughput_tps_std']:.1f} tok/s, regret {b['cumulative_regret_mean']:.3f}, "
          f"t2a {b['time_to_adaptation_mean']:.1f}\n")
    headers = ["level", "factor", "tput", "d_tput", "regret", "d_regret", "t2a"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    # Order rows from -30% up through +30% for a readable sweep.
    order = sorted(results["perturbations"].items(), key=lambda kv: kv[1]["factor"])
    for label, a in order:
        print("| " + " | ".join([
            label, f"{a['factor']:.1f}",
            f"{a['throughput_tps_mean']:.1f}+/-{a['throughput_tps_std']:.1f}",
            f"{a['delta_throughput_tps_mean']:+.1f}",
            f"{a['cumulative_regret_mean']:.3f}",
            f"{a['delta_cumulative_regret_mean']:+.3f}",
            f"{a['time_to_adaptation_mean']:.1f}",
        ]) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL regime-classifier threshold robustness (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else 50
    run_all(seeds, n)


if __name__ == "__main__":
    main()
