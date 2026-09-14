"""
CARL Ablation Study -- what does each adaptive subsystem contribute?

Runs nine configurations across three scenarios and reports, per metric,
mean +/- std over N independent seeds. The five "CARL-NoX" rows each freeze ONE
adaptive subsystem at the global default while the rest stay bandit-adaptive, so
the throughput a subsystem buys is read directly off

    delta_throughput(X) = CARL-Full - CARL-NoX

A positive delta means removing that subsystem hurt -- i.e. it was pulling
weight. Static-Best (best single fixed config), AutoTuner (independent
per-component tuning) and Oracle (perfect regime knowledge) bracket the result.

This is a CONTROL-LOOP SIMULATION (see _harness.SIM_NOTE); the comparisons
between rows are the contribution, not the absolute tok/s.

Run:
  python scripts/eval/ablation.py                  # 5 runs, 30 requests
  python scripts/eval/ablation.py --runs 3 --requests 20   # fast preview
Outputs a table per scenario, a per-subsystem contribution summary, and
docs/eval/ablation_results.json.
"""
from __future__ import annotations

import argparse
import json

import _harness as h


def _scenarios(n_requests: int) -> dict:
    """The three ablation scenarios as per-request regime sequences.

    INTERACTIVE / BATCH each get `n_requests`; NON-STATIONARY splits them half
    INTERACTIVE then half BATCH (a regime flip at the midpoint).
    """
    half = n_requests // 2
    return {
        "INTERACTIVE": h.interactive(n_requests),
        "BATCH": h.batch(n_requests),
        "NON-STATIONARY": h.nonstationary(half, n_requests - half),
    }


def run_ablation(runs: int, n_requests: int) -> dict:
    """Run every config x scenario over `runs` seeds; return the results dict."""
    slo = h.slo_ttft_only()
    seeds = list(range(runs))
    out: dict = {"settings": {"runs": runs, "requests": n_requests,
                              "seeds": seeds, "slo_ttft_ms": h.SLO_TTFT_MS},
                 "scenarios": {}, "contributions": {}}

    for scenario, regimes in _scenarios(n_requests).items():
        # Resolve the single best static config for THIS workload once.
        static_best = h.best_static_config(regimes, slo)
        per_config: dict = {}

        for name in h.tqdm(h.ABLATION_CONFIGS, desc=f"{scenario:>14}"):
            runs_for_config = []
            for seed in seeds:
                agent = h.make_agent(name, slo, static_best_cfg=static_best)
                runs_for_config.append(h.run_once(agent, regimes, slo, seed))
            per_config[name] = h.aggregate_runs(runs_for_config)

        out["scenarios"][scenario] = per_config

        # Per-subsystem contribution: CARL-Full throughput minus each ablation's.
        full_tps = per_config["CARL-Full"]["throughput_mean"]
        out["contributions"][scenario] = {
            name: full_tps - per_config[name]["throughput_mean"]
            for name in ("CARL-NoSched", "CARL-NoSpec", "CARL-NoCache",
                         "CARL-NoRouter", "CARL-NoChunk")
        }
    return out


def _print(results: dict) -> None:
    for scenario, per_config in results["scenarios"].items():
        headers = ["config", "tok/s", "ttftP50", "ttftP95", "ttftP99",
                   "tpotP99", "SLO%", "adapt"]
        rows = []
        for name in h.ABLATION_CONFIGS:
            a = per_config[name]
            rows.append([
                name,
                h.fmt_pm(a["throughput_mean"], a["throughput_std"]),
                h.fmt_pm(a["ttft_p50_mean"], a["ttft_p50_std"]),
                h.fmt_pm(a["ttft_p95_mean"], a["ttft_p95_std"]),
                h.fmt_pm(a["ttft_p99_mean"], a["ttft_p99_std"]),
                h.fmt_pm(a["tpot_p99_mean"], a["tpot_p99_std"]),
                h.fmt_pm(a["slo_sat_mean"], a["slo_sat_std"]),
                h.fmt_pm(a["adaptations_mean"], a["adaptations_std"], prec=0),
            ])
        h.print_pipe_table(f"ABLATION: {scenario} (mean +/- std)", headers, rows)

    # Contribution summary: how much throughput each subsystem adds.
    print("\n=== Per-subsystem contribution (delta tok/s = CARL-Full - CARL-NoX) ===")
    subsystems = ["CARL-NoSched", "CARL-NoSpec", "CARL-NoCache",
                  "CARL-NoRouter", "CARL-NoChunk"]
    headers = ["scenario"] + [s.replace("CARL-No", "") for s in subsystems]
    rows = []
    for scenario, contrib in results["contributions"].items():
        rows.append([scenario] + [f"{contrib[s]:+.1f}" for s in subsystems])
    h.print_pipe_table("", headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL ablation study (simulation).")
    parser.add_argument("--runs", type=int, default=5, help="seeds per config")
    parser.add_argument("--requests", type=int, default=30, help="requests per scenario")
    parser.add_argument("--out", default=None, help="results JSON path")
    args = parser.parse_args()

    print(h.SIM_NOTE)
    results = run_ablation(args.runs, args.requests)
    _print(results)

    out_path = args.out or (h.eval_docs_dir() / "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {out_path}")


if __name__ == "__main__":
    main()
