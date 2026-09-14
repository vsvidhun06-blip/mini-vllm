"""
CARL adaptation analysis driver (GPU) -- generates the offline decision data the
plots and the paper's adaptation section consume.

WHAT IT DOES
------------
Runs CARL-Full on the EXISTING NON-STATIONARY scenario for seeds 42-44 (the same
serving setup as scripts/eval/ablation_live.py -- byte-identical scheduler,
knobs, SLO, workload), then for each run turns the controller's recorded
decision history into:

  * docs/eval/raw/adaptation/decisions_<seed>.csv  -- per control cycle, with
    reward, oracle reward, instant/cumulative regret, the selected arm + its
    knobs, and explicit event markers (regime transition / arm switch /
    convergence point).
  * docs/eval/adaptation_results.json              -- per-seed summaries
    (convergence point, regret totals, switch/unique-arm counts, time-to-first
    adaptation, final arm per regime) plus the oracle table and environment.

PURE REUSE -- NOTHING EXISTING IS MODIFIED
------------------------------------------
It imports the serving primitives from ablation_live (run_config builds and runs
the exact CARL-Full path; compute_dynoracle_arms builds the oracle from recorded
rewards) and the pure analysis from src/carl/adaptation. It writes ONLY to the
new adaptation artifacts above; it never opens, overwrites, or re-runs any
existing eval result (ablation_live_results.json, failure_cases_results.json,
the raw/ablation and raw/failure_cases trees, etc.).

ORACLE / REGRET
---------------
The oracle is the static best-arm-per-regime oracle that the ablation's DynOracle
already defines: compute_dynoracle_arms aggregates the best arm per regime by
MEAN recorded reward across ALL CARL-Full runs. We therefore gather every seed's
decisions first, build one oracle table from the pooled rewards, then score each
seed's per-cycle regret against it (see src/carl/adaptation for the clamp).

GPU note: like the rest of the eval suite this loads TinyLlama, so it needs a
GPU/Colab box to produce representative numbers (it will run on CPU as a slow
smoke test). It is NOT run as part of this offline-layer implementation.

Run:
  python scripts/eval/adaptation_analysis.py                 # seeds 42,43,44 x 50
  python scripts/eval/adaptation_analysis.py --seeds 42 --limit 30   # quick
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime

# --- path bootstrap so `python scripts/eval/adaptation_analysis.py` finds src/ -
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

# Reuse the ablation's serving + oracle machinery verbatim (no modification).
from scripts.eval.ablation_live import (  # noqa: E402
    OBSERVE_INTERVAL, _REGIMES, capture_environment, compute_dynoracle_arms, run_config,
)
from src.carl.adaptation import (  # noqa: E402
    decision_rows, summarize, write_decision_csv, write_summary_json,
)
from src.carl.config import all_arm_sets  # noqa: E402
from src.carl.controller import ControllerLogEntry  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "adaptation")
RESULTS_PATH = os.path.join(DOCS_EVAL, "adaptation_results.json")

DEFAULT_SEEDS = [42, 43, 44]
SCENARIO = "NON-STATIONARY"


class _ArmsView:
    """Minimal `.arms(regime)` adapter over the static per-regime arm sets.

    decision_rows only needs to map a logged config back to its arm index, which
    depends solely on the (static) arm lists -- CARL-Full uses all_arm_sets()
    unfrozen, so these are exactly the arms the controller chose among. Using a
    tiny view avoids reconstructing a numpy-backed bandit just to read arms.
    """

    def __init__(self) -> None:
        self._arms = all_arm_sets()

    def arms(self, regime):
        return self._arms[regime]


def _log_from_decisions(decisions: list) -> list:
    """Rebuild a ControllerLogEntry list from run_config's `decisions` records.

    run_config returns per-cycle decision dicts ({regime, arm, reward, config})
    in controller-log order but drops the controller handle. We reconstruct the
    log entries the pure analysis expects; step is synthesised at the controller
    cadence (cycle * OBSERVE_INTERVAL), which is exactly the scheduler-step
    spacing maybe_step() fires on.
    """
    return [
        ControllerLogEntry(
            step=i * OBSERVE_INTERVAL,
            regime=d["regime"],
            config=d["config"],
            reward=d["reward"],
            state_features=[],
        )
        for i, d in enumerate(decisions)
    ]


def run_all(seeds: list, n: int) -> dict:
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | CARL-Full on {SCENARIO} | "
          f"seeds {seeds} x {n} requests", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab GPU for real numbers.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    # Pass 1: run CARL-Full per seed, keep each run's decisions (pooled for the
    # oracle, per-seed for the CSVs).
    decisions_by_seed: dict = {}
    pooled_decisions: list = []
    for seed in seeds:
        try:
            out = run_config("CARL-Full", model, tokenizer, n, seed)
            decs = out.get("decisions", [])
            decisions_by_seed[seed] = decs
            pooled_decisions.extend(decs)
            print(f"  CARL-Full seed {seed}: {out['throughput_tps']:6.1f} tok/s, "
                  f"{len(decs)} decisions", flush=True)
        except Exception:
            print(f"  CARL-Full seed {seed}: FAILED", flush=True)
            traceback.print_exc()

    # Build the static best-arm-per-regime oracle from the pooled rewards, then
    # reduce it to a {regime_value -> oracle mean reward} table for regret.
    _oracle_arms, oracle_meta = compute_dynoracle_arms(pooled_decisions)
    oracle_reward_by_regime = {
        r.value: oracle_meta[r.value]["mean_reward"] for r in _REGIMES
    }
    print(f"\nOracle (best-arm-per-regime mean reward): "
          f"{ {k: round(v, 4) for k, v in oracle_reward_by_regime.items()} }", flush=True)

    # Pass 2: per-seed decision CSV + summary, scored against the shared oracle.
    arms_view = _ArmsView()
    os.makedirs(RAW_DIR, exist_ok=True)
    per_seed_summaries: dict = {}
    for seed, decs in decisions_by_seed.items():
        log = _log_from_decisions(decs)
        rows = decision_rows(log, arms_view, oracle_reward_by_regime)
        summary = summarize(rows)
        csv_path = os.path.join(RAW_DIR, f"decisions_{seed:03d}.csv")
        write_decision_csv(rows, csv_path)
        per_seed_summaries[str(seed)] = summary
        cp = summary["convergence_point"]
        print(f"  seed {seed}: {summary['n_decisions']} cycles, "
              f"{summary['arm_switch_count']} switches, "
              f"converge@cycle {cp['cycle']} (converged={cp['converged_within_window']}), "
              f"total_regret {summary['total_cumulative_regret']:.3f} -> {csv_path}",
              flush=True)

    results = {
        "scenario": SCENARIO,
        "method": "CARL-Full",
        "seeds": list(decisions_by_seed.keys()),
        "requests_per_run": n,
        "observe_interval": OBSERVE_INTERVAL,
        "environment": env,
        "regret_model": ("static best-arm-per-regime oracle (DynOracle mean "
                         "reward); instant_regret = max(0, oracle - reward)"),
        "oracle_reward_by_regime": oracle_reward_by_regime,
        "oracle_arms_per_regime": oracle_meta,
        "per_seed_summary": per_seed_summaries,
        "generated": datetime.now().isoformat(),
        "note": ("Offline post-processing of CARL-Full's recorded controller log; "
                 "the live serving pipeline is unmodified. Per-cycle data is in "
                 "docs/eval/raw/adaptation/decisions_<seed>.csv."),
    }
    write_summary_json(results, RESULTS_PATH)
    print(f"\nSaved adaptation summary to {RESULTS_PATH}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL offline adaptation analysis (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else 50
    run_all(seeds, n)


if __name__ == "__main__":
    main()
