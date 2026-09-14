"""
CARL Oracle Gap Analysis -- how close does online CARL get to perfect knowledge?

The Oracle applies DEFAULT_CONFIGS[true_regime] every request: it is TOLD the
regime and uses the hand-tuned optimal config for it. No online method can beat
it (it has information they must infer), so the gap to the oracle is the natural
upper-bound metric: a small gap means CARL learns the right operating point fast.

Five methods -- oracle / carl_linucb / carl_thompson / autotuner / static_default
-- are run over a four-phase non-stationary stream with three regime transitions:

    P1 requests  1-20  INTERACTIVE
    P2 requests 21-40  BATCH
    P3 requests 41-60  LONG_CONTEXT
    P4 requests 61-80  BURST

Per phase we report the throughput gap (oracle - method)/oracle and the SLO gap
(oracle_slo - method_slo). At each transition we report CARL's adaptation lag:
requests after the boundary until CARL's detected regime first matches the new
one (the per-regime bandit swaps policy the instant detection flips, so this lag
is detection-bound and is what the paper reports as adaptation latency).

CONTROL-LOOP SIMULATION (see _harness.SIM_NOTE). The oracle is near-optimal BY
CONSTRUCTION, so "oracle gap" measures online learning quality in simulation, not
a hardware result.

Run:
  python scripts/eval/oracle_comparison.py            # 5 runs, 80 requests
Outputs the oracle-gap table, the adaptation-lag table, and
docs/eval/oracle_results.json.
"""
from __future__ import annotations

import argparse
import json

import _harness as h

METHODS = ["oracle", "carl_linucb", "carl_thompson", "AutoTuner", "static_default"]
# Display names map our factory names to the spec's labels.
_LABEL = {"oracle": "Oracle", "carl_linucb": "carl_linucb",
          "carl_thompson": "carl_thompson", "AutoTuner": "autotuner",
          "static_default": "static_default"}


def _phase_spec() -> list:
    R = h.R
    return [(R.INTERACTIVE, 20), (R.BATCH, 20), (R.LONG_CONTEXT, 20), (R.BURST, 20)]


def _phase_metrics(run: dict, phase_slices: list) -> tuple:
    """Per-phase (throughput, SLO%) from one run's per-request series."""
    tps_series = run["_tps_series"]
    ttft_series = run["_ttft_series"]
    tps_per_phase, slo_per_phase = [], []
    for start, end in phase_slices:
        tps = tps_series[start:end]
        ttft = ttft_series[start:end]
        tps_per_phase.append(sum(tps) / len(tps) if tps else 0.0)
        sat = sum(1 for t in ttft if t <= h.SLO_TTFT_MS)
        slo_per_phase.append(100.0 * sat / len(ttft) if ttft else 0.0)
    return tps_per_phase, slo_per_phase


def run_oracle(runs: int, requests_per_phase: int) -> dict:
    slo = h.slo_ttft_only()
    seeds = list(range(runs))
    phases = [(r, requests_per_phase) for r, _ in _phase_spec()]
    regimes = h.phased(*phases)
    boundaries = h.boundaries_of(phases)              # [20, 40, 60] by default
    # Phase [start, end) request-index slices.
    slices, idx = [], 0
    for _r, count in phases:
        slices.append((idx, idx + count))
        idx += count
    phase_names = [f"P{i+1}-{r.value}" for i, (r, _c) in enumerate(phases)]

    # Per method: average per-phase throughput / SLO across seeds. CARL methods
    # also carry per-transition adaptation lag.
    per_method: dict = {}
    lag_by_method: dict = {}
    for method in h.tqdm(METHODS, desc="methods"):
        tps_acc = [[] for _ in phases]
        slo_acc = [[] for _ in phases]
        lags_acc = [[] for _ in boundaries]
        for seed in seeds:
            run = h.run_once(h.make_agent(method, slo, thompson_seed=seed),
                             regimes, slo, seed)
            tps_ph, slo_ph = _phase_metrics(run, slices)
            for i in range(len(phases)):
                tps_acc[i].append(tps_ph[i])
                slo_acc[i].append(slo_ph[i])
            if method in ("carl_linucb", "carl_thompson"):
                for j, b in enumerate(boundaries):
                    lags_acc[j].append(h.adaptation_lag(run["_detected"], b, regimes[b]))
        per_method[method] = {
            "throughput": [h.mean_std(x)[0] for x in tps_acc],
            "slo": [h.mean_std(x)[0] for x in slo_acc],
        }
        if method in ("carl_linucb", "carl_thompson"):
            lag_by_method[method] = [round(h.mean_std(x)[0], 1) for x in lags_acc]

    # Gaps vs the oracle, per phase.
    oracle = per_method["oracle"]
    gaps: dict = {}
    for method in METHODS:
        m = per_method[method]
        tp_gap, slo_gap = [], []
        for i in range(len(phases)):
            o_tps = oracle["throughput"][i]
            tp_gap.append((o_tps - m["throughput"][i]) / o_tps * 100.0 if o_tps else 0.0)
            slo_gap.append(oracle["slo"][i] - m["slo"][i])
        gaps[method] = {"throughput_gap_pct": tp_gap, "slo_gap_pct": slo_gap}

    return {
        "settings": {"runs": runs, "requests_per_phase": requests_per_phase,
                     "seeds": seeds, "slo_ttft_ms": h.SLO_TTFT_MS},
        "phase_names": phase_names,
        "boundaries": boundaries,
        "per_method": per_method,
        "gaps": gaps,
        "adaptation_lag_requests": lag_by_method,
    }


def _print(results: dict) -> None:
    phase_names = results["phase_names"]

    # Oracle-gap table: one row per (method, phase).
    headers = ["method", "phase", "throughput_gap%", "slo_gap%"]
    rows = []
    for method in METHODS:
        g = results["gaps"][method]
        for i, phase in enumerate(phase_names):
            rows.append([_LABEL[method], phase,
                         f"{g['throughput_gap_pct'][i]:+.1f}",
                         f"{g['slo_gap_pct'][i]:+.1f}"])
    h.print_pipe_table("ORACLE GAP (per phase; 0% = matches perfect knowledge)",
                       headers, rows)

    # Adaptation-lag table: one row per transition.
    boundaries = results["boundaries"]
    lag = results["adaptation_lag_requests"]
    headers = ["transition", "carl_linucb", "carl_thompson"]
    rows = []
    for j, b in enumerate(boundaries):
        trans = f"{phase_names[j]} -> {phase_names[j+1]}"
        rows.append([trans,
                     lag.get("carl_linucb", ["-"] * len(boundaries))[j],
                     lag.get("carl_thompson", ["-"] * len(boundaries))[j]])
    h.print_pipe_table("ADAPTATION LAG (requests after a transition to track the "
                       "new regime)", headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL oracle gap analysis (simulation).")
    parser.add_argument("--runs", type=int, default=5, help="seeds per method")
    parser.add_argument("--requests-per-phase", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(h.SIM_NOTE)
    results = run_oracle(args.runs, args.requests_per_phase)
    _print(results)

    out_path = args.out or (h.eval_docs_dir() / "oracle_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved oracle results to {out_path}")


if __name__ == "__main__":
    main()
