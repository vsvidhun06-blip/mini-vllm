"""
CARL Sensitivity Analysis -- does CARL stay effective as conditions change?

Sweeps one parameter at a time and compares CARL-Full vs Static-Best (the best
single fixed config for that setting) on throughput and TTFT P99.

HOW THE PARAMETERS MAP INTO THE SIMULATION (important, read this)
----------------------------------------------------------------
This is a control-loop simulation whose cost model is parameterised by WORKLOAD
REGIME, not by wall-clock arrival rate or by a KV-cache byte budget. So two of
the three swept parameters are expressed THROUGH the real regime classifier --
the only honest lever the cost model exposes:

  * REQUEST_RATES  -> queue backlog: a higher arrival rate means a deeper backlog
    relative to in-flight work, which classify_regime reads as BATCH then BURST.
  * PROMPT_LENGTHS -> avg_prompt_len fed straight to classify_regime.
  * CONTEXT_LENGTHS-> the effective context length fed to classify_regime
    (longer context -> LONG_CONTEXT).

Each setting runs a TWO-regime alternating stream (a fixed INTERACTIVE baseline
interleaved with the parameter-driven regime), so the controller always has a
non-stationary signal to adapt to; the improvement therefore reflects how much
joint adaptation buys at that setting. When the parameter lands in the same
regime as the baseline there is nothing to adapt to and CARL correctly ties
Static-Best (improvement ~0) -- "remains effective, never worse".

Run:
  python scripts/eval/sensitivity.py                  # 3 runs, 40 requests/setting
Outputs three sensitivity tables and docs/eval/sensitivity_results.json.
"""
from __future__ import annotations

import argparse
import json

import _harness as h

REQUEST_RATES = [1, 2, 5, 10, 20]            # requests/sec (modelled as backlog)
PROMPT_LENGTHS = [16, 64, 128, 256, 512]     # tokens
CONTEXT_LENGTHS = [128, 256, 512, 1024]      # max KV-cache tokens


def _alternate(regime_a, regime_b, n: int) -> list:
    """An a,b,a,b,... two-regime stream of length n (the non-stationary base)."""
    return [regime_a if i % 2 == 0 else regime_b for i in range(n)]


def _rate_to_regime(rate: int):
    """Map an arrival rate to the regime its backlog induces (documented proxy)."""
    R = h.R
    if rate <= 2:
        return R.INTERACTIVE      # arrivals keep up with service: shallow queue
    if rate <= 5:
        return R.BATCH            # a sustained queue: throughput regime
    return R.BURST                # arrivals outrun service: a growing backlog


def _length_to_regime(length: int):
    """Map a prompt/context length to its regime via the REAL classifier."""
    return h.regimes_from_prompt_lengths([length])[0]


def _eval_setting(regimes: list, runs: int, slo) -> dict:
    """CARL-Full vs Static-Best on `regimes`; aggregate over `runs` seeds."""
    static_best = h.best_static_config(regimes, slo)
    seeds = list(range(runs))
    carl = h.aggregate_runs([
        h.run_once(h.make_agent("CARL-Full", slo, static_best_cfg=static_best),
                   regimes, slo, s) for s in seeds])
    static = h.aggregate_runs([
        h.run_once(h.make_agent("Static-Best", slo, static_best_cfg=static_best),
                   regimes, slo, s) for s in seeds])
    return {"carl": carl, "static": static}


def _improv(carl: float, static: float, higher_better: bool) -> float:
    if static == 0:
        return 0.0
    return ((carl - static) if higher_better else (static - carl)) / static * 100.0


def run_sensitivity(runs: int, n_requests: int) -> dict:
    slo = h.slo_ttft_only()
    R = h.R
    out: dict = {"settings": {"runs": runs, "requests_per_setting": n_requests,
                              "slo_ttft_ms": h.SLO_TTFT_MS},
                 "sweeps": {}}

    sweeps = {
        "request_rate_rps": [(r, _alternate(R.INTERACTIVE, _rate_to_regime(r), n_requests))
                             for r in REQUEST_RATES],
        "prompt_length_tokens": [(L, _alternate(R.INTERACTIVE, _length_to_regime(L), n_requests))
                                 for L in PROMPT_LENGTHS],
        "context_length_tokens": [(C, _alternate(R.INTERACTIVE, _length_to_regime(C), n_requests))
                                  for C in CONTEXT_LENGTHS],
    }

    for sweep_name, settings in sweeps.items():
        per_setting: dict = {}
        for value, regimes in h.tqdm(settings, desc=f"{sweep_name:>22}"):
            try:
                per_setting[str(value)] = _eval_setting(regimes, runs, slo)
            except Exception as exc:
                # e.g. an OOM at the largest context length on a memory-bound box.
                print(f"  WARNING: {sweep_name}={value} failed ({exc}); skipping")
        out["sweeps"][sweep_name] = per_setting
    return out


def _print(results: dict) -> None:
    titles = {
        "request_rate_rps": "SENSITIVITY: request rate (req/s, modelled as backlog)",
        "prompt_length_tokens": "SENSITIVITY: prompt length (tokens)",
        "context_length_tokens": "SENSITIVITY: context length (max KV tokens)",
    }
    for sweep_name, per_setting in results["sweeps"].items():
        headers = ["setting", "CARL tok/s", "Static tok/s", "tok/s improv%",
                   "CARL ttftP99", "Static ttftP99", "ttftP99 improv%"]
        rows = []
        for value, d in per_setting.items():
            c, s = d["carl"], d["static"]
            rows.append([
                value,
                h.fmt_pm(c["throughput_mean"], c["throughput_std"]),
                h.fmt_pm(s["throughput_mean"], s["throughput_std"]),
                f"{_improv(c['throughput_mean'], s['throughput_mean'], True):+.1f}",
                h.fmt_pm(c["ttft_p99_mean"], c["ttft_p99_std"]),
                h.fmt_pm(s["ttft_p99_mean"], s["ttft_p99_std"]),
                f"{_improv(c['ttft_p99_mean'], s['ttft_p99_mean'], False):+.1f}",
            ])
        h.print_pipe_table(titles[sweep_name], headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL sensitivity analysis (simulation).")
    parser.add_argument("--runs", type=int, default=3, help="seeds per setting")
    parser.add_argument("--requests", type=int, default=40, help="requests per setting")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(h.SIM_NOTE)
    results = run_sensitivity(args.runs, args.requests)
    _print(results)

    out_path = args.out or (h.eval_docs_dir() / "sensitivity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sensitivity results to {out_path}")


if __name__ == "__main__":
    main()
