"""
Cross-model CARL validation: does the SAME controller, with the SAME
hyperparameters and the SAME DEFAULT_CONFIGS, generalize across model families
with ZERO retuning?

  >>> Same CARLConfig DEFAULT_CONFIGS used for all models without modification. <<<

We load each model through the from-scratch engine (src/engine/model.py, now
architecture-aware: LLaMA + Qwen2) and serve the identical NON-STATIONARY
workload under three methods -- CARL-Full, Static-Best (held-out LHS validation
search), and AutoTuner -- reusing the exact serving harness from
scripts/eval/ablation_live.py so the metrics and raw schema match the in-paper
ablation. CARL's arm sets and per-regime DEFAULT_CONFIGS come from
src/carl/config.py and are NEVER touched per model: that invariance is the whole
claim being tested.

PRIMARY METRIC (per model): normalized_performance = CARL-Full throughput /
Static-Best throughput, computed PER RUN (paired at the same seed), then
aggregated mean / std / 95% CI across the N runs. The HEADLINE averages the
per-model means ACROSS MODELS (not across runs).

Models the engine cannot represent exactly (e.g. Gemma's GeGLU + embedding
scaling) or that don't fit in VRAM are skipped gracefully (loaded=false with a
skip_reason) and excluded from the aggregation. We deliberately DO NOT compute
any oracle gap here -- that analysis lives in the ablation, not the cross-model
study.

CPU note: this needs torch + the model weights, so it runs on a GPU/Colab box
(cell 6f), not in CI.

Run:
  python scripts/eval/cross_model.py                       # N=3 seeds, 50 requests
  python scripts/eval/cross_model.py --seeds 42 --limit 30 # quick smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import traceback
from datetime import datetime

# --- path bootstrap so `python scripts/eval/cross_model.py` finds src/ -------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

# Reuse the EXACT serving harness + validation search from the ablation, so the
# numbers and raw-data schema are directly comparable and DEFAULT_CONFIGS/arm
# sets are shared verbatim (zero retuning is enforced by construction).
from scripts.eval.ablation_live import (  # noqa: E402
    N_LHS_CANDIDATES, SEARCH_SPACE, VALIDATION_SEED, latin_hypercube, run_config,
)
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import (  # noqa: E402
    UnsupportedArchitectureError, load_model_from_hf,
)

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "cross_model")
RESULTS_PATH = os.path.join(DOCS_EVAL, "cross_model_results.json")
ENV_PATH = os.path.join(DOCS_EVAL, "environment.json")

DEFAULT_SEEDS = [42, 43, 44]
N_REQUESTS = 50
METHODS = ["CARL-Full", "Static-Best", "AutoTuner"]

# Each model: HF id, a short slug for filenames, the minimum FREE VRAM (GB) we
# require before even attempting the load, and an approximate parameter count
# used only for the skipped-model report row.
MODELS = [
    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "short": "tinyllama",
     "min_vram_gb": 0.0, "approx_params_b": 1.1},
    {"name": "Qwen/Qwen2-0.5B-Instruct", "short": "qwen2",
     "min_vram_gb": 0.0, "approx_params_b": 0.5},
    {"name": "google/gemma-2b-it", "short": "gemma",
     "min_vram_gb": 3.0, "approx_params_b": 2.5},
]


# ===========================================================================
# Environment + VRAM helpers.
# ===========================================================================


def capture_environment() -> dict:
    """Reuse docs/eval/environment.json if present; otherwise create it."""
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                env = json.load(f)
            print(f"Environment: reused {ENV_PATH}", flush=True)
            return env
        except Exception:
            pass
    env = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python": sys.version,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"Environment: {env['gpu']} | torch {env['torch']} -> {ENV_PATH}", flush=True)
    return env


def _free_vram_gb() -> float:
    """Free VRAM in GB, or +inf on CPU (so the CPU smoke path never VRAM-skips)."""
    if not torch.cuda.is_available():
        return float("inf")
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def _model_weight_gb(model) -> float:
    """Resident weight footprint (params + buffers) in GB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1e9


def _params_billion(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


# ===========================================================================
# Per-model load (graceful: returns loaded=False with a reason on any failure).
# ===========================================================================


def load_one(model_cfg: dict, dtype: torch.dtype):
    """Try to load a model. Returns (model, tokenizer, load_info).

    load_info matches the required schema: model / parameters_billion / vram_gb /
    loaded / skip_reason. On any failure (VRAM gate, unsupported architecture,
    missing shard, download error) model/tokenizer are None and loaded=False.
    """
    name = model_cfg["name"]
    info = {"model": name, "parameters_billion": model_cfg["approx_params_b"],
            "vram_gb": 0.0, "loaded": False, "skip_reason": None}

    free = _free_vram_gb()
    if free < model_cfg["min_vram_gb"]:
        info["skip_reason"] = "insufficient VRAM"
        print(f"  [skip] {name}: insufficient VRAM "
              f"({free:.1f} GB free < {model_cfg['min_vram_gb']:.1f} GB needed)",
              flush=True)
        return None, None, info

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        model, _config = load_model_from_hf(name, dtype=dtype)
        model.eval()
        info["loaded"] = True
        info["parameters_billion"] = round(_params_billion(model), 4)
        info["vram_gb"] = round(_model_weight_gb(model), 4)
        print(f"  [ok]   {name}: {info['parameters_billion']:.2f}B params, "
              f"{info['vram_gb']:.2f} GB", flush=True)
        return model, tokenizer, info
    except UnsupportedArchitectureError as exc:
        info["skip_reason"] = f"unsupported architecture: {exc}"
    except Exception as exc:  # noqa: BLE001 -- any load failure is a graceful skip
        info["skip_reason"] = f"load failed: {type(exc).__name__}: {exc}"
    print(f"  [skip] {name}: {info['skip_reason']}", flush=True)
    return None, None, info


# ===========================================================================
# Per-model Static-Best via held-out LHS validation (reuses ablation search).
# ===========================================================================


def select_static_best(model, tokenizer, val_n: int, short: str) -> tuple:
    """LHS search over the full 5-D space on the held-out validation seed.

    Identical procedure to the ablation's Static-Best, run PER MODEL so each
    model gets its own best fixed config (the fair non-adaptive baseline). The
    CARL DEFAULT_CONFIGS are untouched; this only tunes the STATIC competitor.
    """
    candidates = latin_hypercube(N_LHS_CANDIDATES, SEARCH_SPACE, VALIDATION_SEED)
    tputs = []
    for cfg in candidates:
        m = run_config("Static-Best", model, tokenizer, val_n, VALIDATION_SEED,
                       static_cfg=cfg)
        tputs.append(m["throughput_tps"])
    win = max(range(len(candidates)), key=lambda i: tputs[i])
    winner = candidates[win]
    print(f"    [{short}] Static-Best: mb={winner.max_batch_size} "
          f"cs={winner.chunk_size} ({tputs[win]:.1f} tok/s)", flush=True)
    return winner, {"validation_throughputs": tputs, "winner": winner.as_dict(),
                    "validation_seed": VALIDATION_SEED}


# ===========================================================================
# Raw data (same schema as the ablation, plus model + provenance).
# ===========================================================================


def _save_raw(short: str, method: str, seed: int, run: dict, model_name: str,
              seeds_used: list, load_info: dict) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = {
        "model": model_name, "config": method, "seed": seed,
        "requests": run["requests"],
        "throughput_tps": run["throughput_tps"],
        "ttft_p50": run["ttft_p50"], "ttft_p99": run["ttft_p99"],
        "tpot_p50": run["tpot_p50"], "tpot_p99": run["tpot_p99"],
        "slo_rate": run["slo_rate"],
        "seeds_used": seeds_used, "model_load_info": load_info,
    }
    path = os.path.join(RAW_DIR, f"{short}_{method}_run_{seed:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ===========================================================================
# Aggregation helpers.
# ===========================================================================


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _ci95(mean: float, std: float, n: int) -> list:
    if n <= 1:
        return [mean, mean]
    half = 1.96 * std / math.sqrt(n)
    return [mean - half, mean + half]


# ===========================================================================
# Run all three methods over all seeds for ONE model.
# ===========================================================================


def run_model(model_cfg: dict, model, tokenizer, seeds: list, n: int,
              load_info: dict) -> dict:
    name, short = model_cfg["name"], model_cfg["short"]
    print(f"\n[{short}] serving {len(seeds)} seeds x {n} requests, methods={METHODS}",
          flush=True)

    # Per-model Static-Best (validation uses ~half the eval size, like the ablation).
    static_cfg, selection = select_static_best(model, tokenizer, max(10, n // 2), short)

    # method -> {seed -> run-metrics dict}
    per_method: dict = {m: {} for m in METHODS}
    for method in METHODS:
        for seed in seeds:
            try:
                run = run_config(method, model, tokenizer, n, seed,
                                 static_cfg=static_cfg if method == "Static-Best" else None)
                per_method[method][seed] = run
                _save_raw(short, method, seed, run, name, seeds, load_info)
                print(f"    {method:<12} seed {seed}: {run['throughput_tps']:6.1f} tok/s, "
                      f"ttftP99={run['ttft_p99']:6.1f}ms", flush=True)
            except Exception:
                print(f"    {method:<12} seed {seed}: FAILED", flush=True)
                traceback.print_exc()

    # Aggregate each method's throughput + ttft_p99 (mean +/- std over seeds).
    methods_agg: dict = {}
    for method in METHODS:
        runs = list(per_method[method].values())
        tmean, tstd = _mean_std([r["throughput_tps"] for r in runs])
        f99m, f99s = _mean_std([r["ttft_p99"] for r in runs])
        methods_agg[method] = {
            "throughput_tps_mean": tmean, "throughput_tps_std": tstd,
            "ttft_p99_mean": f99m, "ttft_p99_std": f99s,
            "per_seed_throughput": {str(s): per_method[method][s]["throughput_tps"]
                                    for s in seeds if s in per_method[method]},
        }

    # PRIMARY METRIC: normalized_performance per RUN, paired at the same seed.
    per_run_norm = []
    for seed in seeds:
        carl = per_method["CARL-Full"].get(seed)
        static = per_method["Static-Best"].get(seed)
        if carl and static and static["throughput_tps"] > 0:
            per_run_norm.append(carl["throughput_tps"] / static["throughput_tps"])
    npm, nps = _mean_std(per_run_norm)
    norm = {
        "definition": "CARL-Full throughput / Static-Best throughput, per run",
        "per_run": per_run_norm,
        "mean": npm, "std": nps, "n": len(per_run_norm),
        "ci95": _ci95(npm, nps, len(per_run_norm)),
    }

    # Honest per-model note.
    if not per_run_norm:
        note = "no paired runs completed"
    elif npm < 0.99:
        note = (f"CARL underperforms Static-Best by {(1 - npm) * 100:.1f}% here "
                f"-- on a single model with a well-validated static config, online "
                f"learning pays an exploration cost it cannot fully amortize over "
                f"{n} requests.")
    elif npm <= 1.01:
        note = ("CARL matches Static-Best (within +/-1%) with zero per-model "
                "tuning, despite Static-Best being validation-tuned for THIS model.")
    else:
        note = (f"CARL beats the per-model-tuned Static-Best by {(npm - 1) * 100:.1f}% "
                f"by adapting across the regime shift mid-stream.")

    return {
        "model": name, "short": short, "load_info": load_info,
        "static_best_selection": selection,
        "methods": methods_agg,
        "normalized_performance": norm,
        "note": note,
    }


# ===========================================================================
# Driver.
# ===========================================================================


def run_all(seeds: list, n: int) -> dict:
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {len(seeds)} seeds {seeds} x {n} requests",
          flush=True)
    print("INVARIANT: same CARLConfig DEFAULT_CONFIGS used for all models without "
          "modification (zero retuning).", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab GPU for real numbers.\n",
              flush=True)

    load_table: list = []
    per_model: list = []
    for model_cfg in MODELS:
        print(f"\n=== {model_cfg['name']} ===", flush=True)
        model, tokenizer, load_info = load_one(model_cfg, dtype)
        load_table.append(load_info)
        if not load_info["loaded"]:
            per_model.append({"model": model_cfg["name"], "short": model_cfg["short"],
                              "load_info": load_info, "skipped": True})
            continue
        try:
            per_model.append(run_model(model_cfg, model, tokenizer, seeds, n, load_info))
        finally:
            # Free VRAM before the next (larger) model so they fit sequentially.
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results = _finalize(env, seeds, n, load_table, per_model)
    return results


def _finalize(env, seeds, n, load_table, per_model) -> dict:
    # HEADLINE: average the per-model normalized-performance MEANS across the
    # models that actually loaded (average across MODELS, not across runs).
    loaded = [m for m in per_model if not m.get("skipped")
              and m["normalized_performance"]["n"] > 0]
    per_model_means = [m["normalized_performance"]["mean"] for m in loaded]
    across_mean, across_std = _mean_std(per_model_means)

    headline = (
        f"CARL achieves {across_mean * 100:.1f}+/-{across_std * 100:.1f}% of "
        f"Static-Best across {len(loaded)} model families (mean+/-std across "
        f"{len(loaded)} models, zero retuning, identical parameters)."
    )

    results = {
        "seeds": seeds, "requests": n, "scenario": "NON-STATIONARY",
        "methods": METHODS,
        "invariant": "Same CARLConfig DEFAULT_CONFIGS used for all models without modification.",
        "environment": env,
        "model_load_info": load_table,
        "models": per_model,
        "headline_aggregation": {
            "definition": ("average of per-model mean_normalized_performance, "
                           "averaged across MODELS (not across runs)"),
            "loaded_models": [m["model"] for m in loaded],
            "per_model_mean_normalized_performance": {
                m["model"]: m["normalized_performance"]["mean"] for m in loaded},
            "mean_normalized_performance_across_models": across_mean,
            "std_normalized_performance_across_models": across_std,
        },
        "headline": headline,
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved cross-model results to {RESULTS_PATH}", flush=True)
    return results


def _print(results: dict) -> None:
    print("\n=== MODEL LOAD ===")
    print("| model | params_B | vram_gb | loaded |")
    print("| --- | --- | --- | --- |")
    for li in results["model_load_info"]:
        print(f"| {li['model']} | {li['parameters_billion']:.2f} | "
              f"{li['vram_gb']:.2f} | {'yes' if li['loaded'] else 'no'} |")

    print("\n=== CROSS-MODEL RESULTS (NON-STATIONARY, mean +/- std over seeds) ===")
    print("| model | method | throughput | ttft_p99 | norm_perf_mean | norm_perf_ci95 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for m in results["models"]:
        if m.get("skipped"):
            print(f"| {m['short']} | (skipped: {m['load_info']['skip_reason']}) | - | - | - | - |")
            continue
        norm = m["normalized_performance"]
        ci = norm["ci95"]
        for i, method in enumerate(METHODS):
            a = m["methods"][method]
            npmean = f"{norm['mean']:.3f}" if (i == 0) else ""
            npci = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if (i == 0) else ""
            print(f"| {m['short']} | {method} | "
                  f"{a['throughput_tps_mean']:.1f}+/-{a['throughput_tps_std']:.1f} | "
                  f"{a['ttft_p99_mean']:.1f}+/-{a['ttft_p99_std']:.1f} | {npmean} | {npci} |")

    print(f"\nHEADLINE: {results['headline']}")
    print("\nPer-model notes:")
    for m in results["models"]:
        if m.get("skipped"):
            print(f"  - {m['short']}: skipped ({m['load_info']['skip_reason']})")
        else:
            print(f"  - {m['short']}: {m['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model CARL validation (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=N_REQUESTS, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else N_REQUESTS
    run_all(seeds, n)


if __name__ == "__main__":
    main()
