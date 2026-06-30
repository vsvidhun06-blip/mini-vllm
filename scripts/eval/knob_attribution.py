"""
Per-knob attribution WITHIN the scheduler subsystem (live TinyLlama, GPU).

WHY THIS SCRIPT EXISTS
----------------------
The live ablation (scripts/eval/ablation_live.py) measures CARL-NoSched as a
SINGLE block: freeze max_batch_size and the whole "scheduler" contribution
collapses by -28.51 tok/s vs CARL-Full. That tells us the scheduler is where
CARL's live gain comes from, but it does NOT tell us WHICH scheduler knob earns
it. This script decomposes that one number into per-knob marginal contributions.

THE EXPERIMENT (leave-one-knob-frozen)
--------------------------------------
For each knob k in:

    max_batch_size, preemption_enabled, chunk_size,
    routing_threshold, cache_affinity_weight, eviction_threshold

we run CARL with knob k PINNED at its Static-Best value while CARL adapts every
OTHER knob normally (the bandit's arm set still varies the rest). The throughput
loss vs CARL-Full (all knobs adaptive) is knob k's marginal contribution: it is
the cost of taking away CARL's freedom on that ONE axis and nothing else. This
is the dual of ablation_live's "freeze a subsystem" -- here we freeze a single
knob and let the subsystem otherwise adapt.

Setup is IDENTICAL to ablation_live.py and is imported from it, not duplicated:
the same NON-STATIONARY workload builder, the same fresh ContinuousBatchScheduler
per run, the same _serve loop, the same LinUCB PerRegimeBandit + CARLController
(observe_interval, SLO), the same 10 seeds (42..51), and the same held-out LHS
validation to pick Static-Best. Freezing a knob reuses ablation_live._frozen_arms,
which rewrites that knob in every bandit arm so it can never change.

!!! HONEST SCOPE -- READ BEFORE INTERPRETING THE TABLE !!!
----------------------------------------------------------
This is the SAME single-model live harness as ablation_live.py, so the same
caveat applies, now at knob granularity. The CARLController here is wired to the
SCHEDULER ONLY (router / kv_cache / spec_decoder are None), and its _apply()
only writes knobs the component actually declares. The live ContinuousBatchScheduler
declares max_batch_size and chunk_size (and use_cuda_graphs) but NOT
preemption_enabled, and there is no router and no KV cache. Therefore:

  * max_batch_size, chunk_size  -> change what the GPU does -> real deltas.
  * preemption_enabled, routing_threshold, cache_affinity_weight,
    eviction_threshold           -> knobs the live engine does not act on here;
                                    pinning them only reshuffles otherwise-
                                    equivalent bandit arms, so they measure ~= 0
                                    (flagged live_effective=false).

A near-zero contribution for those four is EXPECTED and honest -- it is the
per-knob restatement of why CARL-NoSpec/NoCache/NoRouter ~= CARL-Full in the
ablation. We measure the decomposition empirically anyway rather than assert it,
and we flag which knobs are live-effective. See docs/eval/README.md.

REPORTING
---------
For each knob: throughput_tps (mean +/- std over seeds), delta_vs_full
(CARL-Full - frozen), and the knob's share of the scheduler gain. The raw deltas
are NORMALISED to sum to the ablation's headline +28.51 tok/s so the per-knob
numbers are directly comparable to that single block (pct_of_scheduler_gain =
raw_delta / sum(raw_deltas) * 100; normalized_delta_tps = that share of 28.51).

Run (on a GPU box / Colab T4, like ablation_live.py):
  python scripts/eval/knob_attribution.py                       # N=10, 50 reqs
  python scripts/eval/knob_attribution.py --seeds 42,43 --limit 30   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime

# --- path bootstrap so `python scripts/eval/knob_attribution.py` finds src/ ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# scripts/eval onto the path so we can import its sibling harness as a module.
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import torch  # noqa: E402

# Reuse ablation_live.py wholesale -- single source of truth for the live serving
# setup. Importing the module runs only its import-time code (its main() is
# guarded by __main__), so nothing executes here. We do NOT modify it.
import ablation_live as abl  # noqa: E402

from src.carl.bandit import LinUCBBandit, PerRegimeBandit  # noqa: E402
from src.carl.config import CARLConfig  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

DOCS_EVAL = abl.DOCS_EVAL
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "knob_attribution")
RESULTS_PATH = os.path.join(DOCS_EVAL, "knob_attribution_results.json")

# The scheduler-subsystem knobs we attribute, in report order. These are exactly
# the six the task names; they are the CARLConfig fields routed (in principle) to
# the scheduler/router/cache subsystems, excluding spec_k (speculation, pinned
# off in this harness) and the always-on eviction_window / use_cuda_graphs.
KNOBS = [
    "max_batch_size",
    "preemption_enabled",
    "chunk_size",
    "routing_threshold",
    "cache_affinity_weight",
    "eviction_threshold",
]

# Which knobs the LIVE engine in this harness actually acts on. The controller is
# wired to the scheduler only and _set() writes a knob only if the target object
# declares the attribute (see CARLController._apply / scheduler.py). max_batch_size
# and chunk_size are real scheduler attributes; the rest target a missing attr
# (preemption_enabled) or a None component (router / kv_cache), so pinning them is
# a no-op for inference and the measured delta is bandit-shuffle noise ~= 0.
_LIVE_EFFECTIVE_KNOBS = {"max_batch_size", "chunk_size"}

# The single-block scheduler contribution this decomposition normalises against.
# Source: docs/eval/ablation_live_results.json -> subsystem_contributions ->
# {"subsystem": "Sched", "delta_tps": 28.51} (CARL-Full - CARL-NoSched).
SCHEDULER_ABLATION_DELTA_TPS = 28.51


# ===========================================================================
# One CARL run, with an optional single knob frozen at a fixed value.
# ===========================================================================


def _run_carl(model, tokenizer, n: int, seed: int, freeze: dict | None) -> dict:
    """Serve one NON-STATIONARY workload with CARL driving the scheduler.

    `freeze=None` is CARL-Full (every knob adaptive). `freeze={knob: value}` pins
    that one knob across all bandit arms (via ablation_live._frozen_arms) while
    CARL adapts the rest. This mirrors ablation_live.run_config's CARL branch
    exactly -- same workload builder, scheduler, bandit, controller, SLO and
    cadence -- so CARL-Full here is the same configuration as in the ablation.
    """
    specs = abl._build_workload(tokenizer, "NON-STATIONARY", n, random.Random(seed))
    sched = abl._new_scheduler(model)

    tracker = MetricsTracker(window=max(50, n))
    bandit = PerRegimeBandit(
        abl._frozen_arms(freeze), d=FEATURE_DIM,
        bandit_cls=LinUCBBandit, alpha=0.5,
    )
    controller = CARLController(
        scheduler=sched, bandit=bandit,
        observe_interval=abl.OBSERVE_INTERVAL, slo=abl._SLO, metrics=tracker,
    )
    return abl._serve(sched, specs, controller=controller, tracker=tracker)


# ===========================================================================
# Aggregation helpers.
# ===========================================================================


def _aggregate(per_run: list) -> dict:
    """Mean +/- std of the headline metrics over a config's per-seed runs."""
    agg = {}
    for key, _label in abl._METRICS:
        mean, std = abl._mean_std([m[key] for m in per_run])
        agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
    return agg


def _save_raw(tag: str, seed: int, run: dict) -> None:
    """Persist one run's per-request records + summary, like ablation_live."""
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = {
        "config": tag, "seed": seed,
        "requests": run["requests"],
        "throughput_tps": run["throughput_tps"],
        "ttft_p50": run["ttft_p50"], "ttft_p99": run["ttft_p99"],
        "tpot_p50": run["tpot_p50"], "tpot_p99": run["tpot_p99"],
        "slo_rate": run["slo_rate"],
    }
    with open(os.path.join(RAW_DIR, f"{tag}_run_{seed:03d}.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_config_over_seeds(tag: str, model, tokenizer, seeds: list, n: int,
                           freeze: dict | None) -> list:
    """Run one configuration across all seeds; return the list of _serve dicts."""
    per_run = []
    for r_idx, seed in enumerate(seeds):
        try:
            run = _run_carl(model, tokenizer, n, seed, freeze)
            per_run.append(run)
            _save_raw(tag, seed, run)
            print(f"  {tag:<22} {r_idx + 1}/{len(seeds)} (seed {seed}): "
                  f"{run['throughput_tps']:6.1f} tok/s, "
                  f"ttftP99={run['ttft_p99']:7.1f}ms", flush=True)
        except Exception:
            print(f"  {tag:<22} {r_idx + 1}/{len(seeds)} (seed {seed}): FAILED",
                  flush=True)
            traceback.print_exc()
    return per_run


def _verify_raw_files(seeds: list) -> list:
    """Guardrail: confirm every per-seed raw file per_seed_attribution.py needs
    actually landed on disk -- one CARL-Full_run_NNN.json AND one
    freeze_<knob>_run_NNN.json per seed. Read-only (a filesystem check; it writes
    nothing and does not affect the results). Returns the list of missing paths
    and logs loudly so a partial run is caught before the pairing analysis, not
    silently mis-read as a missing baseline. A file can be absent only if that
    (config, seed) run raised -- in which case _run_config_over_seeds already
    printed FAILED above -- so this surfaces exactly those gaps."""
    expected_tags = ["CARL-Full"] + [f"freeze_{k}" for k in KNOBS]
    missing = [
        os.path.join(RAW_DIR, f"{tag}_run_{seed:03d}.json")
        for tag in expected_tags for seed in seeds
        if not os.path.exists(os.path.join(RAW_DIR, f"{tag}_run_{seed:03d}.json"))
    ]
    if missing:
        print(f"\n!!! GUARDRAIL: {len(missing)} expected raw file(s) MISSING from "
              f"{RAW_DIR} -- per_seed_attribution.py pairing will be incomplete:",
              flush=True)
        for p in missing:
            print(f"    MISSING {os.path.basename(p)}", flush=True)
    else:
        print(f"\nGuardrail OK: all {len(expected_tags) * len(seeds)} per-seed raw "
              f"files present (CARL-Full + {len(KNOBS)} freeze configs x "
              f"{len(seeds)} seeds).", flush=True)
    return missing


# ===========================================================================
# Driver.
# ===========================================================================


def run_all(seeds: list, n: int) -> dict:
    env = abl.capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {len(seeds)} runs x {n} requests "
          f"| seeds {seeds}", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA device -- CPU smoke test only; run on a Colab T4.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    # Static-Best from the SAME held-out LHS validation ablation_live uses, so the
    # value we pin each knob at is exactly the ablation's Static-Best operating
    # point (and recomputed on THIS hardware, so it is consistent with the
    # CARL-Full runs we compare against). Validation uses ~half the eval size.
    static_cfg, _selection = abl.select_static_best(model, tokenizer, max(10, n // 2))
    static_dict = static_cfg.as_dict()
    print(f"\n[static-best] pinning values: "
          + ", ".join(f"{k}={static_dict[k]}" for k in KNOBS), flush=True)

    # 1. CARL-Full (every knob adaptive) -- the reference for every delta.
    print("\n[CARL-Full] all knobs adaptive (reference)", flush=True)
    full_runs = _run_config_over_seeds("CARL-Full", model, tokenizer, seeds, n,
                                       freeze=None)
    full_agg = _aggregate(full_runs)
    full_tput = full_agg["throughput_tps_mean"]

    # 2. One config per knob: pin THAT knob at Static-Best, adapt the rest.
    knob_aggs: dict = {}
    for knob in KNOBS:
        frozen_val = static_dict[knob]
        print(f"\n[freeze {knob}={frozen_val}] CARL adapts the other 5 knobs",
              flush=True)
        tag = f"freeze_{knob}"
        runs = _run_config_over_seeds(tag, model, tokenizer, seeds, n,
                                      freeze={knob: frozen_val})
        if runs:
            knob_aggs[knob] = (frozen_val, _aggregate(runs))

    # 3. Deltas + normalisation onto the scheduler ablation's headline number.
    raw_deltas = {
        k: full_tput - agg["throughput_tps_mean"] for k, (_v, agg) in knob_aggs.items()
    }
    delta_sum = sum(raw_deltas.values())

    knobs_out: dict = {}
    for knob in KNOBS:
        if knob not in knob_aggs:
            continue
        frozen_val, agg = knob_aggs[knob]
        raw_delta = raw_deltas[knob]
        # Share of the (signed) total. Guard a ~0 denominator so a degenerate run
        # reports null shares instead of dividing by zero.
        if abs(delta_sum) > 1e-9:
            pct = raw_delta / delta_sum * 100.0
            normalized = raw_delta / delta_sum * SCHEDULER_ABLATION_DELTA_TPS
        else:
            pct = normalized = None
        knobs_out[knob] = {
            "frozen_value": frozen_val,
            "live_effective": knob in _LIVE_EFFECTIVE_KNOBS,
            "throughput_tps_mean": agg["throughput_tps_mean"],
            "throughput_tps_std": agg["throughput_tps_std"],
            "ttft_p99_mean": agg["ttft_p99_mean"],
            "ttft_p99_std": agg["ttft_p99_std"],
            "slo_rate_mean": agg["slo_rate_mean"],
            "delta_vs_full": raw_delta,
            "pct_of_scheduler_gain": pct,
            "normalized_delta_tps": normalized,
        }

    ranked = sorted(knobs_out.items(),
                    key=lambda kv: kv[1]["delta_vs_full"], reverse=True)

    results = {
        "description": ("Per-knob marginal attribution within the scheduler "
                        "subsystem: each knob is pinned at Static-Best while CARL "
                        "adapts the rest; delta_vs_full is the throughput cost of "
                        "that pin. Decomposes ablation_live's single -28.51 tok/s "
                        "CARL-NoSched block into per-knob shares."),
        "method": ("leave-one-knob-frozen vs CARL-Full on the NON-STATIONARY "
                   "workload; setup imported from scripts/eval/ablation_live.py"),
        "environment": env,
        "scenario": ("NON-STATIONARY (1-25 INTERACTIVE prompt16-64/max32, "
                     "26-50 BATCH prompt128-256/max64)"),
        "seeds": seeds, "runs": len(seeds), "requests_per_seed": n,
        "slo_ttft_ms": abl.SLO_TTFT_MS, "validation_seed": abl.VALIDATION_SEED,
        "static_best": static_dict,
        "scheduler_ablation_delta_tps": SCHEDULER_ABLATION_DELTA_TPS,
        "scheduler_ablation_source": ("docs/eval/ablation_live_results.json -> "
                                      "subsystem_contributions -> Sched"),
        "carl_full": {
            "throughput_tps_mean": full_agg["throughput_tps_mean"],
            "throughput_tps_std": full_agg["throughput_tps_std"],
            "ttft_p99_mean": full_agg["ttft_p99_mean"],
            "ttft_p99_std": full_agg["ttft_p99_std"],
            "slo_rate_mean": full_agg["slo_rate_mean"],
        },
        "knobs": knobs_out,
        "ranked_by_delta": [
            {"knob": k, "delta_vs_full": v["delta_vs_full"],
             "pct_of_scheduler_gain": v["pct_of_scheduler_gain"],
             "normalized_delta_tps": v["normalized_delta_tps"],
             "live_effective": v["live_effective"]}
            for k, v in ranked
        ],
        "raw_delta_sum_tps": delta_sum,
        "live_effective_knobs": sorted(_LIVE_EFFECTIVE_KNOBS),
        "scope_note": ("Single-model live harness (CARL wired to the scheduler "
                       "only; router/kv_cache/spec_decoder are None; speculation "
                       "pinned off). The live scheduler acts on max_batch_size and "
                       "chunk_size only -- it has no preemption_enabled attribute "
                       "and there is no router or KV cache here -- so pinning "
                       "routing_threshold/cache_affinity_weight/eviction_threshold/"
                       "preemption_enabled is a no-op for inference and their "
                       "deltas measure bandit-shuffle noise (~=0, live_effective="
                       "false). This is the per-knob restatement of why "
                       "CARL-NoSpec/NoCache/NoRouter ~= CARL-Full in ablation_live. "
                       "See docs/eval/README.md."),
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved knob-attribution results to {RESULTS_PATH}", flush=True)
    _verify_raw_files(seeds)
    return results


def _print(results: dict) -> None:
    full = results["carl_full"]
    print("\n=== PER-KNOB ATTRIBUTION: NON-STATIONARY on real TinyLlama "
          "(mean +/- std) ===")
    print(f"CARL-Full (all knobs adaptive): "
          f"{full['throughput_tps_mean']:.1f} +/- {full['throughput_tps_std']:.1f} tok/s")
    headers = ["knob (pinned@static-best)", "live?", "tput", "delta", "%sched", "normTPS"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for k, v in results["knobs"].items():
        live = "yes" if v["live_effective"] else "no*"
        pct = "n/a" if v["pct_of_scheduler_gain"] is None else f"{v['pct_of_scheduler_gain']:.1f}"
        norm = "n/a" if v["normalized_delta_tps"] is None else f"{v['normalized_delta_tps']:+.2f}"
        print("| " + " | ".join([
            f"{k}={v['frozen_value']}", live,
            f"{v['throughput_tps_mean']:.1f} +/- {v['throughput_tps_std']:.1f}",
            f"{v['delta_vs_full']:+.2f}", pct, norm,
        ]) + " |")
    print(f"\nraw delta sum = {results['raw_delta_sum_tps']:+.2f} tok/s; "
          f"normalised to scheduler ablation block "
          f"{results['scheduler_ablation_delta_tps']:+.2f} tok/s.")
    print("* 'no' = pinned knob has no live effect in this single-model harness "
          "(scheduler lacks the attr / no router / no KV cache); delta ~= noise.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-knob attribution within the scheduler (real TinyLlama, GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, abl.DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42..51)")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else 50
    run_all(seeds, n)


if __name__ == "__main__":
    main()
