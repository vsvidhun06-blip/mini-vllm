"""
Feature-importance ablation for CARL's context vector (GPU).

QUESTION
--------
Which of the ten features in phi(s_t) actually drive CARL's ARM-SELECTION
decisions? The bandit scores every arm with a linear UCB over phi(s_t); if a
feature carries no discriminative signal, removing it should barely move CARL's
behaviour, whereas knocking out a feature the policy leans on should hurt. This
is the §5 "why CARL works" evidence.

INTERVENTION (one feature at a time)
------------------------------------
For each feature i in {0..9} we ZERO it -- phi_i := 0 -- in every context the
bandit sees, and compare against the unmasked baseline on:

  * cumulative regret        (vs a FIXED static best-arm-per-regime oracle)
  * time-to-adaptation       (control cycles to convergence)
  * final arm selected       (per regime; did the converged policy change?)

The mask is injected by wrapping the bandit (FeatureMaskedBandit), NOT by
touching the controller, the reward, the state observer, or the bandit math.
Crucially the mask hits ONLY the bandit's context: classify_regime still reads
the RAW state, so a masked run is assigned the SAME regimes as the baseline and
we are measuring within-regime ARM discrimination, exactly as asked -- not regime
classification.

WHY A FIXED ORACLE
------------------
Regret is scored against ONE oracle (the best-arm-per-regime mean reward computed
from the UNMASKED baseline runs, the same quantity the ablation's DynOracle uses)
and reused for every masked run. A shared reference is what makes "change in
cumulative regret" comparable across features: a feature whose removal makes CARL
pick worse arms accrues more regret against the same yardstick.

REUSE (nothing existing is modified)
------------------------------------
Mirrors scripts/eval/adaptation_analysis.py: it imports the live serving + oracle
machinery from ablation_live (_serve / _new_scheduler / _arm_index /
compute_dynoracle_arms / _SLO / OBSERVE_INTERVAL / _REGIMES / capture_environment),
the NON-STATIONARY workload from src/carl/live, and the pure regret/convergence
analysis from src/carl/adaptation. It writes ONLY
docs/eval/feature_importance_results.json. The baseline path is byte-identical to
ablation_live's CARL-Full (FeatureMaskedBandit with mask_idx=None behaves exactly
like PerRegimeBandit), so the only variable between baseline and a masked run is
the single zeroed feature.

SCOPE (same single-model caveat as ablation_live)
-------------------------------------------------
Single-TinyLlama harness: CARL is wired to the scheduler only, speculation off,
no router, KV eviction inactive. Several context features (cache_hit_rate,
spec_acceptance_rate) are structurally ~0 here, so their ablation is expected to
be a no-op -- which is itself useful evidence. See docs/eval/README.md.

Run:
  python scripts/eval/feature_importance.py                 # seeds 42,43,44 x 50
  python scripts/eval/feature_importance.py --seeds 42 --limit 30   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
from datetime import datetime

# --- path bootstrap so `python scripts/eval/feature_importance.py` finds src/ ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

# Reuse the ablation's serving + oracle machinery verbatim (no modification).
from scripts.eval.ablation_live import (  # noqa: E402
    OBSERVE_INTERVAL, _REGIMES, _SLO, _arm_index, _new_scheduler, _serve,
    capture_environment, compute_dynoracle_arms,
)
from src.carl.adaptation import decision_rows, summarize  # noqa: E402
from src.carl.bandit import LinUCBBandit, PerRegimeBandit  # noqa: E402
from src.carl.config import all_arm_sets  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.live import _build_workload  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker, RuntimeState  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RESULTS_PATH = os.path.join(DOCS_EVAL, "feature_importance_results.json")

DEFAULT_SEEDS = [42, 43, 44]
SCENARIO = "NON-STATIONARY"      # same workload as ablation_live
FEATURE_NAMES = RuntimeState.feature_names()   # length == FEATURE_DIM, order-locked


# ===========================================================================
# FeatureMaskedBandit -- the whole intervention, in one wrapper.
# ===========================================================================
#
# A PerRegimeBandit that zeroes feature `mask_idx` in EVERY context it is handed,
# on both select() and update(). The controller still computes the full phi(s_t),
# logs it, and classifies the regime from the raw state; only the value the bandit
# actually scores arms with is masked. mask_idx=None is a transparent pass-through
# (the baseline), guaranteeing baseline and masked runs differ by exactly one
# zeroed coordinate and nothing else.
# ===========================================================================


class FeatureMaskedBandit(PerRegimeBandit):
    """PerRegimeBandit with one context coordinate forced to 0 (mask_idx)."""

    def __init__(self, arms_by_regime, d, mask_idx=None, bandit_cls=LinUCBBandit,
                 **bandit_kwargs) -> None:
        super().__init__(arms_by_regime, d, bandit_cls=bandit_cls, **bandit_kwargs)
        self.mask_idx = mask_idx

    def _masked(self, context):
        if self.mask_idx is None:
            return context
        x = list(context)                      # copy; never mutate the caller's vector
        if 0 <= self.mask_idx < len(x):
            x[self.mask_idx] = 0.0
        return x

    def select(self, regime, context):
        return super().select(regime, self._masked(context))

    def update(self, regime, arm, reward, context) -> None:
        return super().update(regime, arm, reward, self._masked(context))


# ===========================================================================
# One CARL-Full run with a given feature mask. Mirrors ablation_live's CARL-Full
# path exactly, swapping in the masked bandit.
# ===========================================================================


def run_carl(model, tokenizer, n, seed, mask_idx) -> CARLController:
    """Serve the NON-STATIONARY workload once with feature `mask_idx` zeroed.

    Returns the controller, whose controller_log carries the per-cycle decisions
    the analysis consumes. mask_idx=None == the unmasked baseline.
    """
    import random
    specs = _build_workload(tokenizer, SCENARIO, n, random.Random(seed))
    sched = _new_scheduler(model)
    tracker = MetricsTracker(window=max(50, n))
    bandit = FeatureMaskedBandit(all_arm_sets(), d=FEATURE_DIM, mask_idx=mask_idx,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
    controller = CARLController(scheduler=sched, bandit=bandit,
                                observe_interval=OBSERVE_INTERVAL, slo=_SLO,
                                metrics=tracker)
    _serve(sched, specs, controller=controller, tracker=tracker)
    return controller


# ===========================================================================
# Analysis helpers (pure; reuse src/carl/adaptation).
# ===========================================================================


class _ArmsView:
    """Minimal `.arms(regime)` adapter over the static per-regime arm sets.

    decision_rows only needs to map a logged config back to its arm index; masking
    never changes the arm sets, so the static lists are exactly what CARL chose
    among in every run.
    """

    def __init__(self) -> None:
        self._arms = all_arm_sets()

    def arms(self, regime):
        return self._arms[regime]


def _mean(vals: list) -> float:
    return statistics.fmean(vals) if vals else 0.0


def _std(vals: list) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _decisions(log, arms_view) -> list:
    """Per-cycle decision dicts ({regime, arm, reward, config}) for the oracle."""
    return [
        {"regime": e.regime, "arm": _arm_index(arms_view.arms(e.regime), e.config),
         "reward": e.reward, "config": e.config}
        for e in log
    ]


def _analyze(logs_by_seed: dict, arms_view, oracle_reward_by_regime: dict) -> dict:
    """Per-seed regret/convergence/final-arm + the mean/std aggregate.

    cumulative_regret, time_to_adaptation (convergence cycle), and
    final_arm_per_regime all come from src/carl/adaptation.summarize, scored
    against the shared oracle.
    """
    per_seed: dict = {}
    for seed, log in logs_by_seed.items():
        rows = decision_rows(log, arms_view, oracle_reward_by_regime)
        s = summarize(rows)
        per_seed[str(seed)] = {
            "cumulative_regret": s["total_cumulative_regret"],
            "time_to_adaptation": s["convergence_point"]["cycle"],
            "converged_within_window": s["convergence_point"]["converged_within_window"],
            "final_arm_per_regime": s["final_arm_per_regime"],
            "arm_switch_count": s["arm_switch_count"],
            "n_decisions": s["n_decisions"],
        }
    regrets = [v["cumulative_regret"] for v in per_seed.values()]
    t2as = [v["time_to_adaptation"] for v in per_seed.values()
            if v["time_to_adaptation"] is not None]
    return {
        "per_seed": per_seed,
        "cumulative_regret_mean": _mean(regrets),
        "cumulative_regret_std": _std(regrets),
        "time_to_adaptation_mean": _mean([float(x) for x in t2as]),
        "time_to_adaptation_std": _std([float(x) for x in t2as]),
    }


def _feature_entry(i: int, masked: dict, baseline: dict) -> dict:
    """Build a feature's result block, including matched-seed deltas vs baseline.

    Deltas are computed PER SEED (same workload, same seed -> the only difference
    is the zeroed feature) then averaged: a positive delta_cumulative_regret means
    removing the feature made CARL choose worse arms (the feature matters).
    """
    d_reg, d_t2a, arm_changed = [], [], {}
    for seed, mv in masked["per_seed"].items():
        bv = baseline["per_seed"][seed]
        d_reg.append(mv["cumulative_regret"] - bv["cumulative_regret"])
        if mv["time_to_adaptation"] is not None and bv["time_to_adaptation"] is not None:
            d_t2a.append(float(mv["time_to_adaptation"] - bv["time_to_adaptation"]))
        arm_changed[seed] = mv["final_arm_per_regime"] != bv["final_arm_per_regime"]
    return {
        "feature_index": i,
        "feature_name": FEATURE_NAMES[i],
        "cumulative_regret_mean": masked["cumulative_regret_mean"],
        "cumulative_regret_std": masked["cumulative_regret_std"],
        "time_to_adaptation_mean": masked["time_to_adaptation_mean"],
        "time_to_adaptation_std": masked["time_to_adaptation_std"],
        "delta_cumulative_regret_mean": _mean(d_reg),
        "delta_cumulative_regret_std": _std(d_reg),
        "delta_time_to_adaptation_mean": _mean(d_t2a),
        "delta_time_to_adaptation_std": _std(d_t2a),
        "final_arm_change_rate": (sum(arm_changed.values()) / len(arm_changed)
                                  if arm_changed else 0.0),
        "final_arm_changed_per_seed": arm_changed,
        "per_seed": masked["per_seed"],
    }


# ===========================================================================
# Driver.
# ===========================================================================


def run_all(seeds: list, n: int) -> dict:
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | feature-importance on {SCENARIO} | "
          f"seeds {seeds} x {n} requests | {FEATURE_DIM} features + baseline", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab T4 for real numbers.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    arms_view = _ArmsView()

    # --- Pass 1: the unmasked baseline. Its pooled decisions define the oracle. --
    print("\n[baseline] CARL-Full (no feature masked)", flush=True)
    baseline_logs: dict = {}
    pooled: list = []
    for seed in seeds:
        try:
            ctrl = run_carl(model, tokenizer, n, seed, mask_idx=None)
            baseline_logs[seed] = ctrl.controller_log
            pooled.extend(_decisions(ctrl.controller_log, arms_view))
            print(f"  baseline seed {seed}: {len(ctrl.controller_log)} cycles", flush=True)
        except Exception:
            print(f"  baseline seed {seed}: FAILED", flush=True)
            traceback.print_exc()

    # Fixed static best-arm-per-regime oracle from the baseline (DynOracle reward).
    _oracle_arms, oracle_meta = compute_dynoracle_arms(pooled)
    oracle_reward_by_regime = {r.value: oracle_meta[r.value]["mean_reward"] for r in _REGIMES}
    print(f"  oracle (best-arm-per-regime mean reward): "
          f"{ {k: round(v, 4) for k, v in oracle_reward_by_regime.items()} }", flush=True)

    baseline = _analyze(baseline_logs, arms_view, oracle_reward_by_regime)
    print(f"  baseline regret {baseline['cumulative_regret_mean']:.3f} +/- "
          f"{baseline['cumulative_regret_std']:.3f}, "
          f"t2a {baseline['time_to_adaptation_mean']:.1f} cycles", flush=True)

    # --- Pass 2: zero each feature in turn, score against the SAME oracle. -------
    features: dict = {}
    for i in range(FEATURE_DIM):
        print(f"\n[mask {i}: {FEATURE_NAMES[i]}]", flush=True)
        logs: dict = {}
        for seed in seeds:
            try:
                ctrl = run_carl(model, tokenizer, n, seed, mask_idx=i)
                logs[seed] = ctrl.controller_log
            except Exception:
                print(f"  feat {i} seed {seed}: FAILED", flush=True)
                traceback.print_exc()
        if not logs:
            continue
        masked = _analyze(logs, arms_view, oracle_reward_by_regime)
        entry = _feature_entry(i, masked, baseline)
        features[str(i)] = entry
        print(f"  regret {entry['cumulative_regret_mean']:.3f} "
              f"(delta {entry['delta_cumulative_regret_mean']:+.3f}), "
              f"d_t2a {entry['delta_time_to_adaptation_mean']:+.1f}, "
              f"final-arm change rate {entry['final_arm_change_rate']:.2f}", flush=True)

    # Importance ranking: most-damaging-to-remove first (delta regret desc, then
    # how often the converged arm changed).
    ranking = sorted(
        (
            {
                "feature_index": e["feature_index"],
                "feature_name": e["feature_name"],
                "delta_cumulative_regret_mean": e["delta_cumulative_regret_mean"],
                "delta_time_to_adaptation_mean": e["delta_time_to_adaptation_mean"],
                "final_arm_change_rate": e["final_arm_change_rate"],
            }
            for e in features.values()
        ),
        key=lambda d: (d["delta_cumulative_regret_mean"], d["final_arm_change_rate"]),
        reverse=True,
    )

    results = {
        "experiment": "feature_importance_ablation",
        "scenario": SCENARIO,
        "method": ("CARL-Full with one context feature zeroed in the bandit's "
                   "phi(s_t); regime classification uses the unmasked state."),
        "seeds": list(baseline_logs.keys()),
        "requests_per_run": n,
        "observe_interval": OBSERVE_INTERVAL,
        "slo_ttft_ms": _SLO.ttft_ms,
        "environment": env,
        "feature_names": FEATURE_NAMES,
        "regret_model": ("static best-arm-per-regime oracle (DynOracle mean reward "
                         "from the UNMASKED baseline, pooled over seeds); reused for "
                         "every masked run so deltas are comparable. "
                         "instant_regret = max(0, oracle - reward)."),
        "oracle_reward_by_regime": oracle_reward_by_regime,
        "oracle_arms_per_regime": oracle_meta,
        "baseline": baseline,
        "features": features,
        "feature_importance_ranking": ranking,
        "scope_note": ("Single-model live harness: CARL wired to the scheduler "
                       "only, speculation off, no router, KV eviction inactive. "
                       "Features that are structurally ~0 here (cache_hit_rate, "
                       "spec_acceptance_rate) are expected to ablate to ~no-op. "
                       "Masking affects ONLY the bandit context, not "
                       "classify_regime. See docs/eval/README.md."),
        "generated": datetime.now().isoformat(),
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved feature-importance results to {RESULTS_PATH}", flush=True)
    return results


def _print(results: dict) -> None:
    print("\n=== FEATURE IMPORTANCE (ablate one phi feature; vs unmasked baseline) ===")
    b = results["baseline"]
    print(f"baseline: regret {b['cumulative_regret_mean']:.3f}, "
          f"t2a {b['time_to_adaptation_mean']:.1f} cycles\n")
    headers = ["rank", "feat", "name", "d_regret", "d_t2a", "armChange%"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for rank, d in enumerate(results["feature_importance_ranking"], 1):
        print("| " + " | ".join([
            str(rank), str(d["feature_index"]), d["feature_name"],
            f"{d['delta_cumulative_regret_mean']:+.3f}",
            f"{d['delta_time_to_adaptation_mean']:+.1f}",
            f"{100 * d['final_arm_change_rate']:.0f}",
        ]) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL context-feature importance ablation (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else 50
    run_all(seeds, n)


if __name__ == "__main__":
    main()
