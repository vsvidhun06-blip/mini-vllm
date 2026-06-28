"""
Feature-importance ablation for CARL's context vector (GPU).

QUESTION
--------
Which of the ten features in phi(s_t) actually drive CARL's ARM-SELECTION
decisions? The bandit scores every arm with a linear UCB over phi(s_t); if a
feature carries no discriminative signal, removing it should barely move CARL's
behaviour, whereas knocking out a feature the policy leans on should hurt. This
is the §5 "why CARL works" evidence.

TWO INTERVENTIONS (one feature at a time), selected by --mode
------------------------------------------------------------
  * --mode zero  : ZERO the feature -- phi_i := 0 (FeatureMaskedBandit).
                   -> docs/eval/feature_importance_results.json
  * --mode noise : CORRUPT one feature with N(0, sigma), sigma in {0.1, 0.2, 0.5}
                   times the feature's CHARACTERISTIC SCALE (NoisyFeatureBandit).
                   -> docs/eval/feature_noise_results.json
  * --mode global_noise : N(0, sigma) on ALL features simultaneously, same sigma
                   levels (GlobalNoiseBandit). Also reports throughput per level.
                   -> docs/eval/global_noise_results.json
  * --mode both  : run zero + noise (baseline is re-measured for each).

Both compare against the unperturbed baseline on:

  * cumulative regret        (vs a FIXED static best-arm-per-regime oracle)
  * time-to-adaptation       (control cycles to convergence)
  * final arm selected       (per regime; did the converged policy change?)

The perturbation is injected by WRAPPING the bandit, NOT by touching the
controller, the reward, the state observer, or the bandit math. Crucially it hits
ONLY the bandit's context: classify_regime still reads the RAW state, so a
perturbed run is assigned the SAME regimes as the baseline and we are measuring
within-regime ARM discrimination, exactly as asked -- not regime classification.

NOISE CALIBRATION
-----------------
The bandit consumes the NORMALIZED context (raw / characteristic_scale), so adding
N(0, sigma*scale_i) in raw units is identically N(0, sigma) on the normalized
coordinate. The noise ablation therefore injects N(0, sigma) directly, with
sigma in {0.1, 0.2, 0.5} == 10/20/50% of feature i's characteristic scale.

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
analysis from src/carl/adaptation. It writes ONLY its own result JSONs. The
baseline path is byte-identical to ablation_live's CARL-Full (a wrapped bandit
with no perturbation behaves exactly like PerRegimeBandit), so the only variable
between baseline and a perturbed run is the single feature touched.

SCOPE (same single-model caveat as ablation_live)
-------------------------------------------------
Single-TinyLlama harness: CARL is wired to the scheduler only, speculation off,
no router, KV eviction inactive. Several context features (cache_hit_rate,
spec_acceptance_rate) are structurally ~0 here, so their ablation is expected to
be a no-op -- which is itself useful evidence. See docs/eval/README.md.

Run:
  python scripts/eval/feature_importance.py                      # zeroing ablation
  python scripts/eval/feature_importance.py --mode noise         # per-feature noise
  python scripts/eval/feature_importance.py --mode global_noise  # all-feature noise
  python scripts/eval/feature_importance.py --mode both --seeds 42 --limit 30  # quick
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

import numpy as np  # noqa: E402
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
# _FEATURE_SCALES: the per-feature characteristic scale state.to_feature_vector
# divides by. The noise ablation is calibrated to it (see SIGMA_LEVELS below).
from src.carl.state import (  # noqa: E402
    FEATURE_DIM, MetricsTracker, RuntimeState, _FEATURE_SCALES,
)
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RESULTS_PATH = os.path.join(DOCS_EVAL, "feature_importance_results.json")
NOISE_RESULTS_PATH = os.path.join(DOCS_EVAL, "feature_noise_results.json")
GLOBAL_NOISE_RESULTS_PATH = os.path.join(DOCS_EVAL, "global_noise_results.json")

DEFAULT_SEEDS = [42, 43, 44]
SCENARIO = "NON-STATIONARY"      # same workload as ablation_live
FEATURE_NAMES = RuntimeState.feature_names()   # length == FEATURE_DIM, order-locked

# Noise ablation: sigma as a fraction of each feature's CHARACTERISTIC SCALE.
# The bandit sees the NORMALIZED context (raw / scale), so adding N(0, level*scale)
# in raw units is exactly N(0, level) in normalized units -- we therefore inject
# N(0, level) straight onto the normalized coordinate. level in {0.1, 0.2, 0.5}.
SIGMA_LEVELS = [0.1, 0.2, 0.5]


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
# NoisyFeatureBandit -- the Gaussian-noise counterpart of the mask.
# ===========================================================================
#
# Instead of zeroing feature `noise_idx`, it CORRUPTS it with N(0, sigma) on every
# context the bandit reads (select + update). sigma is in NORMALIZED units, which
# (because the context is raw/scale) equals sigma * characteristic_scale in raw
# units -- so sigma in {0.1, 0.2, 0.5} is "10/20/50% of the feature's scale", as
# specified. Noise is drawn from a seeded RNG so each run is reproducible, and
# only this one coordinate is perturbed; everything else matches the baseline.
#
# Noise is applied independently on select and update (a "noisy sensor" model: the
# feature is read with fresh noise each time), the exact analogue of how the mask
# zeroes both reads.
# ===========================================================================


class NoisyFeatureBandit(PerRegimeBandit):
    """PerRegimeBandit that adds N(0, sigma) to one context coordinate."""

    def __init__(self, arms_by_regime, d, noise_idx=None, sigma=0.0, seed=None,
                 bandit_cls=LinUCBBandit, **bandit_kwargs) -> None:
        super().__init__(arms_by_regime, d, bandit_cls=bandit_cls, **bandit_kwargs)
        self.noise_idx = noise_idx
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

    def _noisy(self, context):
        if self.noise_idx is None or self.sigma <= 0.0:
            return context
        x = list(context)                      # copy; never mutate the caller's vector
        if 0 <= self.noise_idx < len(x):
            x[self.noise_idx] = x[self.noise_idx] + float(self._rng.normal(0.0, self.sigma))
        return x

    def select(self, regime, context):
        return super().select(regime, self._noisy(context))

    def update(self, regime, arm, reward, context) -> None:
        return super().update(regime, arm, reward, self._noisy(context))


# ===========================================================================
# GlobalNoiseBandit -- N(0, sigma) on ALL features at once.
# ===========================================================================
#
# The per-feature NoisyFeatureBandit corrupts a single coordinate; this corrupts
# EVERY coordinate of phi(s_t) simultaneously with independent N(0, sigma). Same
# calibration: sigma is in normalized units, == sigma * characteristic_scale per
# feature in raw units. sigma=0 is a transparent pass-through (the baseline).
# ===========================================================================


class GlobalNoiseBandit(PerRegimeBandit):
    """PerRegimeBandit that adds independent N(0, sigma) to every context coord."""

    def __init__(self, arms_by_regime, d, sigma=0.0, seed=None,
                 bandit_cls=LinUCBBandit, **bandit_kwargs) -> None:
        super().__init__(arms_by_regime, d, bandit_cls=bandit_cls, **bandit_kwargs)
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

    def _noisy(self, context):
        if self.sigma <= 0.0:
            return context
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        return list(x + self._rng.normal(0.0, self.sigma, size=x.shape))

    def select(self, regime, context):
        return super().select(regime, self._noisy(context))

    def update(self, regime, arm, reward, context) -> None:
        return super().update(regime, arm, reward, self._noisy(context))


# ===========================================================================
# One CARL-Full run. Mirrors ablation_live's CARL-Full path exactly, swapping in
# whichever wrapped bandit the caller built.
# ===========================================================================


def _serve_carl_out(model, tokenizer, n, seed, bandit) -> tuple:
    """Serve the NON-STATIONARY workload once with `bandit`.

    Returns (controller, serve_metrics) -- the controller carries the decision
    log; serve_metrics is _serve's dict (throughput_tps, latency percentiles, ...).
    """
    import random
    specs = _build_workload(tokenizer, SCENARIO, n, random.Random(seed))
    sched = _new_scheduler(model)
    tracker = MetricsTracker(window=max(50, n))
    controller = CARLController(scheduler=sched, bandit=bandit,
                                observe_interval=OBSERVE_INTERVAL, slo=_SLO,
                                metrics=tracker)
    out = _serve(sched, specs, controller=controller, tracker=tracker)
    return controller, out


def _serve_carl(model, tokenizer, n, seed, bandit) -> CARLController:
    """Serve once with `bandit`; return just the controller (regret/convergence)."""
    return _serve_carl_out(model, tokenizer, n, seed, bandit)[0]


def run_carl(model, tokenizer, n, seed, mask_idx) -> CARLController:
    """CARL-Full with feature `mask_idx` zeroed (mask_idx=None == baseline)."""
    bandit = FeatureMaskedBandit(all_arm_sets(), d=FEATURE_DIM, mask_idx=mask_idx,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
    return _serve_carl(model, tokenizer, n, seed, bandit)


def run_carl_noisy(model, tokenizer, n, seed, noise_idx, sigma) -> CARLController:
    """CARL-Full with N(0, sigma) added to feature `noise_idx` (normalized units)."""
    bandit = NoisyFeatureBandit(all_arm_sets(), d=FEATURE_DIM, noise_idx=noise_idx,
                                sigma=sigma, seed=seed, bandit_cls=LinUCBBandit,
                                alpha=0.5)
    return _serve_carl(model, tokenizer, n, seed, bandit)


def run_carl_global_noise(model, tokenizer, n, seed, sigma) -> tuple:
    """CARL-Full with N(0, sigma) added to ALL features; return (controller, out).

    sigma=0.0 -> the unperturbed baseline (GlobalNoiseBandit passes through).
    """
    bandit = GlobalNoiseBandit(all_arm_sets(), d=FEATURE_DIM, sigma=sigma, seed=seed,
                               bandit_cls=LinUCBBandit, alpha=0.5)
    return _serve_carl_out(model, tokenizer, n, seed, bandit)


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


def _deltas(perturbed: dict, baseline: dict) -> dict:
    """Matched-seed deltas of a perturbed run vs baseline (regret, t2a, final arm).

    Same per-seed differencing as _feature_entry but without the feature labels --
    used by the noise ablation, where each (feature, sigma) cell is one perturbed
    run to compare against the shared baseline.
    """
    d_reg, d_t2a, arm_changed = [], [], {}
    for seed, mv in perturbed["per_seed"].items():
        bv = baseline["per_seed"][seed]
        d_reg.append(mv["cumulative_regret"] - bv["cumulative_regret"])
        if mv["time_to_adaptation"] is not None and bv["time_to_adaptation"] is not None:
            d_t2a.append(float(mv["time_to_adaptation"] - bv["time_to_adaptation"]))
        arm_changed[seed] = mv["final_arm_per_regime"] != bv["final_arm_per_regime"]
    return {
        "cumulative_regret_mean": perturbed["cumulative_regret_mean"],
        "cumulative_regret_std": perturbed["cumulative_regret_std"],
        "time_to_adaptation_mean": perturbed["time_to_adaptation_mean"],
        "time_to_adaptation_std": perturbed["time_to_adaptation_std"],
        "delta_cumulative_regret_mean": _mean(d_reg),
        "delta_cumulative_regret_std": _std(d_reg),
        "delta_time_to_adaptation_mean": _mean(d_t2a),
        "delta_time_to_adaptation_std": _std(d_t2a),
        "final_arm_change_rate": (sum(arm_changed.values()) / len(arm_changed)
                                  if arm_changed else 0.0),
        "per_seed": perturbed["per_seed"],
    }


def _baseline_and_oracle(model, tokenizer, seeds: list, n: int, arms_view) -> tuple:
    """Run the unmasked CARL-Full baseline; return (baseline, oracle_by_regime, meta).

    The baseline's pooled decisions define the FIXED static best-arm-per-regime
    oracle that every perturbed run (masked OR noised) is scored against, so all
    deltas share one reference.
    """
    print("\n[baseline] CARL-Full (unperturbed)", flush=True)
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

    _oracle_arms, oracle_meta = compute_dynoracle_arms(pooled)
    oracle_reward_by_regime = {r.value: oracle_meta[r.value]["mean_reward"] for r in _REGIMES}
    print(f"  oracle (best-arm-per-regime mean reward): "
          f"{ {k: round(v, 4) for k, v in oracle_reward_by_regime.items()} }", flush=True)

    baseline = _analyze(baseline_logs, arms_view, oracle_reward_by_regime)
    print(f"  baseline regret {baseline['cumulative_regret_mean']:.3f} +/- "
          f"{baseline['cumulative_regret_std']:.3f}, "
          f"t2a {baseline['time_to_adaptation_mean']:.1f} cycles", flush=True)
    return baseline, oracle_reward_by_regime, oracle_meta


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

    # Unmasked baseline + the fixed oracle every masked run is scored against.
    baseline, oracle_reward_by_regime, oracle_meta = _baseline_and_oracle(
        model, tokenizer, seeds, n, arms_view)

    # --- Zero each feature in turn, score against the SAME oracle. ---------------
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


# ===========================================================================
# Gaussian-noise ablation driver.
# ===========================================================================


def run_noise_all(seeds: list, n: int) -> dict:
    """For each feature, add N(0, sigma) (sigma in SIGMA_LEVELS, as a fraction of
    the feature's characteristic scale) and measure delta_regret / delta_t2a."""
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | feature-NOISE on {SCENARIO} | "
          f"seeds {seeds} x {n} requests | {FEATURE_DIM} features x "
          f"{len(SIGMA_LEVELS)} sigmas + baseline", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab T4 for real numbers.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    arms_view = _ArmsView()

    # Unperturbed baseline + the fixed oracle every noised run is scored against.
    baseline, oracle_reward_by_regime, oracle_meta = _baseline_and_oracle(
        model, tokenizer, seeds, n, arms_view)

    # For each feature, sweep the noise levels; score against the SAME oracle.
    features: dict = {}
    for i in range(FEATURE_DIM):
        print(f"\n[noise {i}: {FEATURE_NAMES[i]}]", flush=True)
        noise: dict = {}
        for sigma in SIGMA_LEVELS:
            logs: dict = {}
            for seed in seeds:
                try:
                    ctrl = run_carl_noisy(model, tokenizer, n, seed, i, sigma)
                    logs[seed] = ctrl.controller_log
                except Exception:
                    print(f"  feat {i} sigma {sigma} seed {seed}: FAILED", flush=True)
                    traceback.print_exc()
            if not logs:
                continue
            perturbed = _analyze(logs, arms_view, oracle_reward_by_regime)
            cell = _deltas(perturbed, baseline)
            noise[str(sigma)] = cell
            print(f"  sigma {sigma}: regret {cell['cumulative_regret_mean']:.3f} "
                  f"(delta {cell['delta_cumulative_regret_mean']:+.3f}), "
                  f"d_t2a {cell['delta_time_to_adaptation_mean']:+.1f}, "
                  f"final-arm change {cell['final_arm_change_rate']:.2f}", flush=True)
        features[str(i)] = {
            "feature_index": i,
            "feature_name": FEATURE_NAMES[i],
            "characteristic_scale": _FEATURE_SCALES[FEATURE_NAMES[i]],
            "noise": noise,
        }

    # Sensitivity ranking: most-damaging-to-noise first, judged at the LARGEST
    # sigma (delta regret desc, then how often the converged arm changed).
    top = str(SIGMA_LEVELS[-1])
    ranking = sorted(
        (
            {
                "feature_index": e["feature_index"],
                "feature_name": e["feature_name"],
                "sigma": SIGMA_LEVELS[-1],
                "delta_cumulative_regret_mean": e["noise"][top]["delta_cumulative_regret_mean"],
                "delta_time_to_adaptation_mean": e["noise"][top]["delta_time_to_adaptation_mean"],
                "final_arm_change_rate": e["noise"][top]["final_arm_change_rate"],
            }
            for e in features.values() if top in e["noise"]
        ),
        key=lambda d: (d["delta_cumulative_regret_mean"], d["final_arm_change_rate"]),
        reverse=True,
    )

    results = {
        "experiment": "feature_noise_ablation",
        "scenario": SCENARIO,
        "method": ("CARL-Full with N(0, sigma) added to one context feature in the "
                   "bandit's phi(s_t); regime classification uses the unmasked state."),
        "seeds": list(baseline["per_seed"].keys()),
        "requests_per_run": n,
        "observe_interval": OBSERVE_INTERVAL,
        "slo_ttft_ms": _SLO.ttft_ms,
        "environment": env,
        "feature_names": FEATURE_NAMES,
        "feature_characteristic_scales": dict(_FEATURE_SCALES),
        "sigma_levels": SIGMA_LEVELS,
        "noise_model": ("sigma is a fraction of each feature's characteristic scale; "
                        "since the bandit context is raw/scale, N(0, sigma*scale) in "
                        "raw units == N(0, sigma) on the normalized coordinate, which "
                        "is what we inject (seeded per run, independent draw per read "
                        "on select and update)."),
        "regret_model": ("static best-arm-per-regime oracle (DynOracle mean reward "
                         "from the UNPERTURBED baseline, pooled over seeds); reused "
                         "for every noised run. instant_regret = max(0, oracle - reward)."),
        "oracle_reward_by_regime": oracle_reward_by_regime,
        "oracle_arms_per_regime": oracle_meta,
        "baseline": baseline,
        "features": features,
        "noise_sensitivity_ranking": ranking,
        "scope_note": ("Single-model live harness: CARL wired to the scheduler only, "
                       "speculation off, no router, KV eviction inactive. Features "
                       "structurally ~0 here (cache_hit_rate, spec_acceptance_rate) "
                       "are insensitive to noise by construction. See "
                       "docs/eval/README.md."),
        "generated": datetime.now().isoformat(),
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(NOISE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print_noise(results)
    print(f"\nSaved feature-noise results to {NOISE_RESULTS_PATH}", flush=True)
    return results


def _print_noise(results: dict) -> None:
    print(f"\n=== FEATURE NOISE (add N(0, sigma*scale); ranked at sigma="
          f"{SIGMA_LEVELS[-1]}, vs baseline) ===")
    b = results["baseline"]
    print(f"baseline: regret {b['cumulative_regret_mean']:.3f}, "
          f"t2a {b['time_to_adaptation_mean']:.1f} cycles\n")
    headers = ["rank", "feat", "name", "d_regret", "d_t2a", "armChange%"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for rank, d in enumerate(results["noise_sensitivity_ranking"], 1):
        print("| " + " | ".join([
            str(rank), str(d["feature_index"]), d["feature_name"],
            f"{d['delta_cumulative_regret_mean']:+.3f}",
            f"{d['delta_time_to_adaptation_mean']:+.1f}",
            f"{100 * d['final_arm_change_rate']:.0f}",
        ]) + " |")


# ===========================================================================
# Global-noise ablation driver (ALL features noised at once).
# ===========================================================================


def run_global_noise_all(seeds: list, n: int) -> dict:
    """Add N(0, sigma) to ALL features at once for sigma in SIGMA_LEVELS, and
    measure throughput / regret / t2a / final-arm change vs the unperturbed run.

    Throughput is tracked here (unlike the per-feature path) because global noise
    degrades the realised serving rate, not just the bandit's choices.
    """
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | GLOBAL-NOISE on {SCENARIO} | "
          f"seeds {seeds} x {n} requests | {len(SIGMA_LEVELS)} sigmas + baseline "
          f"(all {FEATURE_DIM} features noised)", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab T4 for real numbers.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    arms_view = _ArmsView()

    # --- Baseline: sigma=0 (all features intact). Capture throughput + the oracle.
    print("\n[baseline] CARL-Full (sigma=0)", flush=True)
    base_logs: dict = {}
    base_tput: list = []
    pooled: list = []
    for seed in seeds:
        try:
            ctrl, out = run_carl_global_noise(model, tokenizer, n, seed, sigma=0.0)
            base_logs[seed] = ctrl.controller_log
            base_tput.append(out["throughput_tps"])
            pooled.extend(_decisions(ctrl.controller_log, arms_view))
            print(f"  baseline seed {seed}: {out['throughput_tps']:6.1f} tok/s, "
                  f"{len(ctrl.controller_log)} cycles", flush=True)
        except Exception:
            print(f"  baseline seed {seed}: FAILED", flush=True)
            traceback.print_exc()

    _oracle_arms, oracle_meta = compute_dynoracle_arms(pooled)
    oracle_reward_by_regime = {r.value: oracle_meta[r.value]["mean_reward"] for r in _REGIMES}
    base_analysis = _analyze(base_logs, arms_view, oracle_reward_by_regime)
    baseline = {
        "throughput_tps_mean": _mean(base_tput), "throughput_tps_std": _std(base_tput),
        "cumulative_regret_mean": base_analysis["cumulative_regret_mean"],
        "cumulative_regret_std": base_analysis["cumulative_regret_std"],
        "time_to_adaptation_mean": base_analysis["time_to_adaptation_mean"],
        "time_to_adaptation_std": base_analysis["time_to_adaptation_std"],
        "per_seed": base_analysis["per_seed"],
        "per_seed_throughput": {str(s): t for s, t in zip(base_logs.keys(), base_tput)},
    }
    print(f"  oracle: { {k: round(v, 4) for k, v in oracle_reward_by_regime.items()} } | "
          f"baseline {baseline['throughput_tps_mean']:.1f} tok/s, "
          f"regret {baseline['cumulative_regret_mean']:.3f}, "
          f"t2a {baseline['time_to_adaptation_mean']:.1f}", flush=True)

    # --- Sweep sigma levels, all features noised; score vs the SAME oracle. ------
    levels: dict = {}
    for sigma in SIGMA_LEVELS:
        print(f"\n[global noise sigma={sigma}]", flush=True)
        logs: dict = {}
        tputs: list = []
        for seed in seeds:
            try:
                ctrl, out = run_carl_global_noise(model, tokenizer, n, seed, sigma=sigma)
                logs[seed] = ctrl.controller_log
                tputs.append(out["throughput_tps"])
                print(f"  sigma {sigma} seed {seed}: {out['throughput_tps']:6.1f} tok/s",
                      flush=True)
            except Exception:
                print(f"  sigma {sigma} seed {seed}: FAILED", flush=True)
                traceback.print_exc()
        if not logs:
            continue
        analysis = _analyze(logs, arms_view, oracle_reward_by_regime)
        d = _deltas(analysis, base_analysis)    # regret/t2a deltas + final-arm change
        cell = {
            "sigma": sigma,
            "throughput_tps_mean": _mean(tputs), "throughput_tps_std": _std(tputs),
            "cumulative_regret_mean": d["cumulative_regret_mean"],
            "cumulative_regret_std": d["cumulative_regret_std"],
            "time_to_adaptation_mean": d["time_to_adaptation_mean"],
            "time_to_adaptation_std": d["time_to_adaptation_std"],
            "delta_throughput_tps_mean": _mean(tputs) - baseline["throughput_tps_mean"],
            "delta_cumulative_regret_mean": d["delta_cumulative_regret_mean"],
            "delta_cumulative_regret_std": d["delta_cumulative_regret_std"],
            "delta_time_to_adaptation_mean": d["delta_time_to_adaptation_mean"],
            "delta_time_to_adaptation_std": d["delta_time_to_adaptation_std"],
            "final_arm_change_rate": d["final_arm_change_rate"],
            "per_seed": d["per_seed"],
            "per_seed_throughput": {str(s): t for s, t in zip(logs.keys(), tputs)},
        }
        levels[str(sigma)] = cell
        print(f"  -> {cell['throughput_tps_mean']:.1f} tok/s "
              f"(delta {cell['delta_throughput_tps_mean']:+.1f}), "
              f"regret {cell['cumulative_regret_mean']:.3f} "
              f"(delta {cell['delta_cumulative_regret_mean']:+.3f}), "
              f"d_t2a {cell['delta_time_to_adaptation_mean']:+.1f}, "
              f"final-arm change {cell['final_arm_change_rate']:.2f}", flush=True)

    results = {
        "experiment": "global_noise_ablation",
        "scenario": SCENARIO,
        "method": ("CARL-Full with N(0, sigma) added to ALL context features "
                   "simultaneously; regime classification uses the unmasked state."),
        "seeds": list(base_logs.keys()),
        "requests_per_run": n,
        "observe_interval": OBSERVE_INTERVAL,
        "slo_ttft_ms": _SLO.ttft_ms,
        "environment": env,
        "feature_names": FEATURE_NAMES,
        "feature_characteristic_scales": dict(_FEATURE_SCALES),
        "sigma_levels": SIGMA_LEVELS,
        "noise_model": ("independent N(0, sigma) on every normalized coordinate; "
                        "since context is raw/scale, that is N(0, sigma*scale_i) per "
                        "feature in raw units. Seeded per run, fresh draw per read."),
        "regret_model": ("static best-arm-per-regime oracle (DynOracle mean reward "
                         "from the sigma=0 baseline, pooled over seeds); reused for "
                         "every level. instant_regret = max(0, oracle - reward)."),
        "oracle_reward_by_regime": oracle_reward_by_regime,
        "oracle_arms_per_regime": oracle_meta,
        "baseline": baseline,
        "levels": levels,
        "scope_note": ("Single-model live harness: CARL wired to the scheduler only, "
                       "speculation off, no router, KV eviction inactive. See "
                       "docs/eval/README.md."),
        "generated": datetime.now().isoformat(),
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(GLOBAL_NOISE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print_global(results)
    print(f"\nSaved global-noise results to {GLOBAL_NOISE_RESULTS_PATH}", flush=True)
    return results


def _print_global(results: dict) -> None:
    print("\n=== GLOBAL NOISE (N(0, sigma) on all features; vs sigma=0 baseline) ===")
    b = results["baseline"]
    print(f"baseline: {b['throughput_tps_mean']:.1f} +/- {b['throughput_tps_std']:.1f} "
          f"tok/s, regret {b['cumulative_regret_mean']:.3f}, "
          f"t2a {b['time_to_adaptation_mean']:.1f}\n")
    headers = ["sigma", "tput", "d_tput", "regret", "d_regret", "t2a", "armChange%"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for sigma in SIGMA_LEVELS:
        a = results["levels"].get(str(sigma))
        if a is None:
            continue
        print("| " + " | ".join([
            str(sigma),
            f"{a['throughput_tps_mean']:.1f}+/-{a['throughput_tps_std']:.1f}",
            f"{a['delta_throughput_tps_mean']:+.1f}",
            f"{a['cumulative_regret_mean']:.3f}",
            f"{a['delta_cumulative_regret_mean']:+.3f}",
            f"{a['time_to_adaptation_mean']:.1f}",
            f"{100 * a['final_arm_change_rate']:.0f}",
        ]) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL context-feature importance + noise ablations (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    parser.add_argument("--mode", choices=["zero", "noise", "global_noise", "both"],
                        default="zero",
                        help="zero = feature_importance_results.json (default); "
                             "noise = feature_noise_results.json (per-feature); "
                             "global_noise = global_noise_results.json (all features "
                             "at once); both = zero + noise")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else 50
    if args.mode in ("zero", "both"):
        run_all(seeds, n)
    if args.mode in ("noise", "both"):
        run_noise_all(seeds, n)
    if args.mode == "global_noise":
        run_global_noise_all(seeds, n)


if __name__ == "__main__":
    main()
