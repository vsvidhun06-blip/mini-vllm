"""
Reconciliation probe for the CARL live-overhead discrepancy.

WHY THIS EXISTS
---------------
docs/eval/overhead_results.json records two very different "P99 CARL decision
latency" numbers measured inside the real serving loop:

    earlier live run : ~2527 us  (LIVE_ABLATION_P99_US in overhead.py)
    later   live run : ~16753 us (reported, never committed to a JSON)

A 6.6x swing between two runs of the SAME measurement is suspicious. This probe
runs ONLY the CARL-Full / NON-STATIONARY configuration from ablation_live.py
(skipping the LHS validation search and the other nine configs, which are
irrelevant to the overhead question) and captures the FULL per-decision latency
list per run -- not just the aggregate P99 the harness keeps.

With the raw distribution we can test the two hypotheses directly:

  H1 (sample size): a control cycle fires once every OBSERVE_INTERVAL scheduler
      steps, so a 30-request run yields only ~tens of decisions. P99 over so few
      samples is just the single slowest decision -- an unstable statistic.
  H2 (warmup):      the first decision of a process is cold (numpy linalg / lazy
      import warmup). Excluding it should collapse the tail toward steady state.

This runs on CPU (slow, but the n_decisions count and the warmup STRUCTURE are
identical to GPU; only the absolute microseconds differ). It is a diagnostic,
not an eval artifact -- it does not overwrite ablation_live_results.json.

Run:
  python scripts/eval/overhead_warmup_probe.py --seeds 42,43,44 --limit 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from src.carl.live import _percentile  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
import scripts.eval.ablation_live as abl  # noqa: E402

OUT = os.path.join(_REPO_ROOT, "docs", "eval", "raw", "overhead", "warmup_probe.json")


def main() -> None:
    p = argparse.ArgumentParser(description="CARL live-overhead reconciliation probe.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device {DEVICE} | dtype {dtype} | CARL-Full only | "
          f"{len(seeds)} seeds {seeds} x {n} requests", flush=True)
    if DEVICE.type != "cuda":
        print("NOTE: CPU run -- absolute us are CPU-inflated, but n_decisions and "
              "the warmup structure match GPU.", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    per_run = []          # one decision_us list per seed
    for seed in seeds:
        out = abl.run_config("CARL-Full", model, tokenizer, n, seed)
        dus = list(out.get("decision_us", []))
        per_run.append(dus)
        cold = dus[0] if dus else 0.0
        steady = dus[1:]
        print(f"  seed {seed}: n_decisions={len(dus)}  cold(1st)={cold:.1f}us  "
              f"steady P50={_percentile(steady,50):.1f} P99={_percentile(steady,99):.1f}us  "
              f"raw-run-P99={_percentile(dus,99):.1f}us", flush=True)

    flat = [v for d in per_run for v in d]
    steady_flat = [v for d in per_run for v in d[1:]]
    cold_starts = [d[0] for d in per_run if d]

    report = {
        "device": str(DEVICE),
        "dtype": str(dtype),
        "scenario": "NON-STATIONARY",
        "seeds": seeds,
        "requests": n,
        "observe_interval": abl.OBSERVE_INTERVAL,
        "n_decisions_per_run": [len(d) for d in per_run],
        "n_decisions_total": len(flat),
        "decision_us_per_run": per_run,
        "cold_start_us_per_run": cold_starts,
        "raw_p99_us": _percentile(flat, 99),
        "raw_p99_interpretation": (
            f"P99 over {len(flat)} samples is index "
            f"{max(0, min(len(flat)-1, round(0.99*(len(flat)-1))))} of the sorted "
            f"list == effectively the max sample -- not a stable tail statistic."),
        "steady_state_p50_us": _percentile(steady_flat, 50),
        "steady_state_p99_us": _percentile(steady_flat, 99),
        "steady_state_n": len(steady_flat),
        "cold_start_max_us": max(cold_starts) if cold_starts else 0.0,
        "cold_start_to_steady_ratio": (
            (max(cold_starts) / _percentile(steady_flat, 50))
            if cold_starts and steady_flat and _percentile(steady_flat, 50) > 0 else 0.0),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nraw P99 (all {report['n_decisions_total']} decisions): "
          f"{report['raw_p99_us']:.1f} us", flush=True)
    print(f"steady-state P50/P99 (warmup excluded, n={report['steady_state_n']}): "
          f"{report['steady_state_p50_us']:.1f} / {report['steady_state_p99_us']:.1f} us",
          flush=True)
    print(f"cold-start first decision per run: "
          f"{[round(c,1) for c in cold_starts]} us "
          f"(max {report['cold_start_max_us']:.1f}us = "
          f"{report['cold_start_to_steady_ratio']:.1f}x steady P50)", flush=True)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
