"""
CARL Workload Diversity Suite -- does CARL win across many workload shapes?

Compares three methods -- CARL-Full vs Static-Best (best single fixed config for
that workload) vs AutoTuner (independent per-component tuning) -- across eight
workload types, each 50 requests, three seeds, reported as mean +/- std.

The eight workloads exercise different corners of the regime space. Where a
workload is defined by prompt length we map length -> regime through the REAL
classify_regime (not by hand); where it is defined by arrival pattern (bursty /
queued / interactive) we emit the regime that pattern induces directly.

CONTROL-LOOP SIMULATION (see _harness.SIM_NOTE). The cost model has no notion of
max_tokens or wall-clock inter-arrival, so workloads differing only in those
(e.g. short_prompts vs interactive_only) map to the same regime here; the
diversity that matters for the controller is the regime mix, which does vary.

Run:
  python scripts/eval/workload_suite.py                 # 3 runs, 50 requests
  python scripts/eval/workload_suite.py --real          # try real LMSYS prompts
Outputs a workload x metric table and docs/eval/workload_results.json.
"""
from __future__ import annotations

import argparse
import json
import random

import _harness as h

METHODS = ["Static-Best", "AutoTuner", "CARL-Full"]

# (key, label, higher_is_better) -- the metrics reported per workload.
_METRICS = [
    ("throughput", "tok/s", True),
    ("ttft_p99", "ttftP99", False),
    ("slo_sat", "SLO%", True),
]


def _workload_regimes(name: str, n: int, real: bool) -> list:
    """Per-request regime sequence for a named workload (deterministic).

    Fixed internal seed so the sequence is identical for every method and seed
    within a workload -- the run_once seed varies only the cost-model noise, so
    method differences are not confounded by a different request stream.
    """
    rng = random.Random(1234)
    R = h.R
    if name == "short_prompts":
        return h.regimes_from_prompt_lengths([rng.randint(16, 32) for _ in range(n)])
    if name == "long_prompts":
        return h.regimes_from_prompt_lengths([rng.randint(256, 512) for _ in range(n)])
    if name == "mixed_lengths":
        return h.regimes_from_prompt_lengths([rng.randint(16, 512) for _ in range(n)])
    if name == "bursty":
        # 10 dense arrivals (a backlog -> BURST) then 10 sparse (-> INTERACTIVE),
        # repeated. The dense/sparse alternation is the defining feature.
        out = []
        while len(out) < n:
            out += [R.BURST] * 10 + [R.INTERACTIVE] * 10
        return out[:n]
    if name == "interactive_only":
        return [R.INTERACTIVE] * n
    if name == "batch_only":
        # Queued arrival -> a sustained deep queue -> BATCH throughput regime.
        return [R.BATCH] * n
    if name == "long_context":
        return h.regimes_from_prompt_lengths([rng.randint(513, 1024) for _ in range(n)])
    if name == "lmsys_sample":
        return _lmsys_regimes(n, real)
    raise ValueError(f"unknown workload: {name!r}")


def _lmsys_regimes(n: int, real: bool) -> list:
    """Regimes from real LMSYS-Chat-1M prompt lengths, or a synthetic fallback.

    Gracefully degrades: if --real is off, or the dataset is gated / `datasets`
    is missing / HF auth is absent, we fall back to a synthetic length mix with
    the same shape (mostly short, some long) instead of failing the suite.
    """
    if real:
        try:
            prompts = h.bc._load_lmsys_prompts(n)
            lengths = [max(1, len(p) // 4) for p in prompts]   # ~4 chars/token
            if lengths:
                print(f"  lmsys_sample: using {len(lengths)} real LMSYS prompts")
                return h.regimes_from_prompt_lengths(lengths)
        except Exception as exc:
            print(f"  lmsys_sample: real LMSYS unavailable ({exc}); using synthetic")
    rng = random.Random(7)
    pool = [24] * 6 + [320] * 3 + [700] * 1   # mostly short, some batch, a little long
    return h.regimes_from_prompt_lengths([rng.choice(pool) for _ in range(n)])


WORKLOADS = ["short_prompts", "long_prompts", "mixed_lengths", "bursty",
             "interactive_only", "batch_only", "long_context", "lmsys_sample"]


def run_suite(runs: int, n_requests: int, real: bool, skip: set) -> dict:
    slo = h.slo_ttft_only()
    seeds = list(range(runs))
    out: dict = {"settings": {"runs": runs, "requests": n_requests, "seeds": seeds,
                              "real_lmsys": real, "slo_ttft_ms": h.SLO_TTFT_MS},
                 "workloads": {}}

    for wl in h.tqdm(WORKLOADS, desc="workloads"):
        if wl in skip:
            print(f"  skipping {wl} (requested)")
            continue
        try:
            regimes = _workload_regimes(wl, n_requests, real)
            static_best = h.best_static_config(regimes, slo)
            per_method: dict = {}
            for method in METHODS:
                runs_for = [
                    h.run_once(h.make_agent(method, slo, static_best_cfg=static_best),
                               regimes, slo, seed)
                    for seed in seeds
                ]
                per_method[method] = h.aggregate_runs(runs_for)
            out["workloads"][wl] = per_method
        except Exception as exc:
            # Treat any per-workload failure (e.g. an OOM on long_context in a
            # real-inference variant, or a dataset error) as skip-and-continue,
            # so one bad workload never sinks the whole suite.
            print(f"  WARNING: {wl} failed ({exc}); skipping")
    return out


def _improvement(carl: float, static: float, higher_better: bool) -> float:
    """Signed % improvement of CARL over Static-Best (positive = CARL better)."""
    if static == 0:
        return 0.0
    if higher_better:
        return (carl - static) / static * 100.0
    return (static - carl) / static * 100.0   # lower is better -> shrink wins


def _print(results: dict) -> None:
    headers = ["workload", "metric", "Static-Best", "AutoTuner", "CARL-Full",
               "CARL improv%"]
    rows = []
    for wl, per_method in results["workloads"].items():
        for key, label, higher in _METRICS:
            s = per_method["Static-Best"]
            a = per_method["AutoTuner"]
            c = per_method["CARL-Full"]
            improv = _improvement(c[f"{key}_mean"], s[f"{key}_mean"], higher)
            rows.append([
                wl, label,
                h.fmt_pm(s[f"{key}_mean"], s[f"{key}_std"]),
                h.fmt_pm(a[f"{key}_mean"], a[f"{key}_std"]),
                h.fmt_pm(c[f"{key}_mean"], c[f"{key}_std"]),
                f"{improv:+.1f}",
            ])
    h.print_pipe_table("WORKLOAD DIVERSITY (CARL vs Static-Best vs AutoTuner)",
                       headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL workload diversity suite (simulation).")
    parser.add_argument("--runs", type=int, default=3, help="seeds per method")
    parser.add_argument("--requests", type=int, default=50, help="requests per workload")
    parser.add_argument("--real", action="store_true",
                        help="use real LMSYS-Chat-1M for lmsys_sample (needs HF auth)")
    parser.add_argument("--skip", default="",
                        help="comma-separated workloads to skip (e.g. long_context)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(h.SIM_NOTE)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    results = run_suite(args.runs, args.requests, args.real, skip)
    _print(results)

    out_path = args.out or (h.eval_docs_dir() / "workload_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved workload results to {out_path}")


if __name__ == "__main__":
    main()
