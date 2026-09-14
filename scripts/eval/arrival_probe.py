"""
2k-request INTERACTIVE arrival-model calibration probe (simulation, no inference).

PURPOSE
-------
Before the 100k long-horizon stability run we must set the workload-generator
parameters so the derived regime stream is genuinely INTERACTIVE -- otherwise a
"stability" experiment is ill-defined (an overloaded queue is trivially unstable,
and long/multi-turn prompts leak into BATCH/LONG_CONTEXT/CACHE_HEAVY). This probe
sweeps candidate generator settings over a 2k-request stream (seed 42) and reports
which keep classify_regime == INTERACTIVE, WITHOUT touching the controller,
scheduler, reward, or the classifier.

WHAT IT REUSES (the parts that must stay authoritative)
-------------------------------------------------------
  src.carl.state.classify_regime / RuntimeState / WorkloadRegime -- the REAL
      classifier, unchanged. The probe's only job is to feed it a workload whose
      generator parameters we vary; the regime verdict is the engine's own rule.
  trace_replay._lognormal_params -- the same length-distribution parameterisation
      the existing trace replay uses.

WHAT IT PARAMETERISES (the approved "change the workload generator" knobs)
-------------------------------------------------------------------------
This mirrors trace_replay.synthetic_window + derive_regimes but exposes the three
levers the earlier design identified, so the 100k run can reuse the calibrated
values:

  * rho  -- offered load = arrival_rate / service_rate. trace_replay hardwires
            service = 0.9 * arrival (rho ~= 1.11 > 1 -> unbounded queue -> BURST).
            rho < 1 (sub-critical) keeps the virtual queue shallow -> INTERACTIVE
            on the queue axis. This is the smallest change vs capping queue depth.
  * len_mean / len_std -- prompt token length. Must stay < 256 (>=256 -> BATCH,
            >512 -> LONG_CONTEXT).
  * single_turn -- if True every conversation is one turn, so cache_hit stays at
            the cold-prompt floor (< 0.5 -> never CACHE_HEAVY). Multi-turn context
            growth is what pushes deep chats into CACHE_HEAVY / LONG_CONTEXT.

INTERACTIVE requires ALL of: effective_prompt_len < 256, cache_hit_rate < 0.5,
queue_depth < 8 (and < 24 for BURST) -- see src/carl/state.classify_regime.

Run (deterministic, CPU, < a few seconds):
  python scripts/eval/arrival_probe.py                 # seed 42, n=2000, full sweep
  python scripts/eval/arrival_probe.py --n 2000 --seed 42
Writes a candidate x metric table + JSON; nothing else is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass

# --- path bootstrap ---------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The REAL classifier (unchanged) + the same lognormal parameterisation the trace
# replay uses. We import classify_regime rather than reimplement the thresholds so
# the probe's verdict is exactly the engine's rule.
from src.carl.state import RuntimeState, WorkloadRegime, classify_regime  # noqa: E402
import trace_replay as tr  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
DEFAULT_OUT = os.path.join(DOCS_EVAL, "arrival_probe_results.json")

# INTERACTIVE-keeping thresholds, straight from classify_regime (for the report).
LONG_CTX_TOKENS = 512.0
BATCH_TOKENS = 256.0
CACHE_HEAVY_HIT = 0.5
BATCH_QUEUE = 8.0
BURST_QUEUE = 24.0
INTERACTIVE_MIN_FRAC = 0.99   # the calibration pass/fail bar for the sweep.

# Idle-gap / active cap mirror trace_replay.derive_regimes so the queue dynamics
# match the existing arrival model (only rho / length / turns are the new levers).
IDLE_GAP_S = tr.IDLE_GAP_S
MAX_ACTIVE = tr.MAX_ACTIVE


@dataclass
class Req:
    input_len: int
    context_len: int      # accumulated prior context (0 for single-turn).
    arrival: float
    turn: int

    @property
    def eff_prompt(self) -> float:
        return float(self.context_len + self.input_len)


# ===========================================================================
# Parameterised generator (the calibratable synthetic_window).
# ===========================================================================


def gen_stream(seed: int, n: int, *, len_mean: float, len_std: float,
               single_turn: bool, within_mean: float = 0.25,
               between_mean: float = 0.4, conv_mean_turns: float = 2.7,
               conv_std_turns: float = 2.5) -> list:
    """A synthetic request stream with explicit length / turn knobs.

    Mirrors trace_replay.synthetic_window (log-normal lengths, exponential
    think-time arrivals, multi-turn context accumulation) but exposes len_mean/
    len_std and a single_turn switch. single_turn=True emits one turn per
    conversation (context stays 0, cache stays cold); otherwise conversations grow
    a log-normal number of turns and context accumulates (the BATCH/CACHE_HEAVY
    leak we want to characterise). Absolute arrival timing only rescales time, so
    it does not affect the utilization rho applied in derive() -- rho is the real
    lever.
    """
    rng = random.Random(seed * 1_000_003 + 7)
    mu_len, sigma_len = tr._lognormal_params(len_mean, len_std)
    mu_turns, sigma_turns = tr._lognormal_params(conv_mean_turns, conv_std_turns)

    reqs: list = []
    t = 0.0
    while len(reqs) < n:
        conv_len = 1 if single_turn else min(
            40, max(1, int(round(rng.lognormvariate(mu_turns, sigma_turns)))))
        t += rng.expovariate(1.0 / between_mean)
        context = 0
        for turn in range(conv_len):
            if len(reqs) >= n:
                break
            in_len = max(1, int(round(rng.lognormvariate(mu_len, sigma_len))))
            out_len = max(1, int(round(rng.lognormvariate(mu_len, sigma_len))))
            reqs.append(Req(input_len=in_len, context_len=context, arrival=t, turn=turn))
            context += in_len + out_len
            if turn < conv_len - 1:
                t += rng.expovariate(1.0 / within_mean)
    return reqs[:n]


# ===========================================================================
# Regime derivation with an explicit utilization rho (the offered-load lever).
# ===========================================================================


def derive(reqs: list, rho: float) -> tuple[list, dict]:
    """Per-request regimes via a virtual queue at offered load `rho`.

    Utilization rho = arrival_rate / service_rate, so service_rate =
    mean_arrival_rate / rho. rho < 1 => service faster than arrival => the fluid
    queue drains and stays shallow (INTERACTIVE); rho > 1 (trace_replay's 1.11)
    => the queue grows without bound (BURST). Everything else (idle drain, active
    cap, the classify_regime call) matches trace_replay.derive_regimes.
    """
    if len(reqs) < 2:
        return [WorkloadRegime.INTERACTIVE], {}
    span = reqs[-1].arrival - reqs[0].arrival
    mean_rate = (len(reqs) - 1) / span if span > 0 else float(len(reqs))
    mu = max(1e-6, mean_rate / rho)     # service rate for this utilization.

    queue = 0.0
    last_t = reqs[0].arrival
    regimes: list = []
    qdepths: list = []
    for r in reqs:
        dt = max(0.0, r.arrival - last_t)
        last_t = r.arrival
        queue = max(0.0, queue - mu * dt) + 1.0
        qd = int(queue)
        qdepths.append(qd)
        active = max(1, min(qd, MAX_ACTIVE))
        cache_hit = min(0.9, 0.1 + 0.2 * r.turn)
        st = RuntimeState(avg_prompt_len=r.eff_prompt, queue_depth=qd,
                          active_requests=active, cache_hit_rate=cache_hit)
        regimes.append(classify_regime(st))
    return regimes, {"qdepths": qdepths, "mean_rate_rps": mean_rate, "service_rps": mu}


# ===========================================================================
# Characterisation.
# ===========================================================================


def _pct(xs: list, p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return float(s[k])


def characterise(reqs: list, regimes: list, qinfo: dict) -> dict:
    """Regime mix + the INTERACTIVE-driver distributions for one stream."""
    mix: dict = {}
    for reg in regimes:
        mix[reg.value] = mix.get(reg.value, 0) + 1
    n = len(regimes) or 1
    eff = [r.eff_prompt for r in reqs]
    qd = qinfo.get("qdepths", [])
    return {
        "n": n,
        "interactive_frac": mix.get("interactive", 0) / n,
        "regime_mix": {k: v / n for k, v in mix.items()},
        "eff_prompt_p50": _pct(eff, 50), "eff_prompt_p90": _pct(eff, 90),
        "eff_prompt_p99": _pct(eff, 99), "eff_prompt_max": max(eff) if eff else 0.0,
        "frac_turn_ge2": sum(1 for r in reqs if r.turn >= 2) / n,
        "queue_p50": _pct(qd, 50), "queue_p99": _pct(qd, 99),
        "queue_max": max(qd) if qd else 0.0,
        "mean_rate_rps": qinfo.get("mean_rate_rps"),
        "service_rps": qinfo.get("service_rps"),
    }


# ===========================================================================
# The sweep: baseline (current defaults) + INTERACTIVE candidates.
# ===========================================================================

# Each candidate = (label, rho, len_mean, len_std, single_turn). The baseline
# reproduces trace_replay's current settings (multi-turn, len 128/256, rho~1.11)
# to demonstrate it is NOT interactive; the rest are the calibration candidates.
CANDIDATES = [
    ("baseline_tracereplay",     1.111, 128.0, 256.0, False),
    ("rho0.7_len64_wide_single", 0.7,    64.0,  64.0, True),  # wide tail: p99>256.
    ("rho0.7_len48_tight_single", 0.7,   48.0,  24.0, True),  # tight len, rho at queue edge.
    ("rho0.6_len48_tight_single", 0.6,   48.0,  24.0, True),
    ("rho0.5_len48_tight_single", 0.5,   48.0,  24.0, True),
    ("rho0.6_len64_tight_single", 0.6,   64.0,  32.0, True),
    ("rho0.5_len64_tight_single", 0.5,   64.0,  32.0, True),
    ("rho0.6_len64_tight_multi",  0.6,   64.0,  32.0, False),  # turn-only stress.
]


def run_sweep(seed: int, n: int) -> dict:
    rows = []
    for label, rho, lm, ls, single in CANDIDATES:
        reqs = gen_stream(seed, n, len_mean=lm, len_std=ls, single_turn=single)
        regimes, qinfo = derive(reqs, rho)
        c = characterise(reqs, regimes, qinfo)
        c.update({"label": label, "rho": rho, "len_mean": lm, "len_std": ls,
                  "single_turn": single,
                  "passes": bool(c["interactive_frac"] >= INTERACTIVE_MIN_FRAC)})
        rows.append(c)
    # Recommend the LOWEST-load passing candidate that still uses realistic
    # lengths (prefer the least-conservative rho that clears the bar, to keep the
    # server busy-but-stable rather than trivially idle).
    passing = [r for r in rows if r["passes"]]
    passing.sort(key=lambda r: (-r["rho"], r["len_mean"]))  # highest rho first.
    recommended = passing[0]["label"] if passing else None
    return {
        "description": ("2k-request INTERACTIVE arrival-model calibration probe. "
                        "Sweeps workload-generator parameters (offered load rho, "
                        "prompt length, single- vs multi-turn) and reports which "
                        "keep classify_regime == INTERACTIVE. No controller / "
                        "scheduler / reward / classifier changes."),
        "seed": seed, "n_requests": n,
        "interactive_bar": INTERACTIVE_MIN_FRAC,
        "thresholds": {"long_ctx_tokens": LONG_CTX_TOKENS, "batch_tokens": BATCH_TOKENS,
                       "cache_heavy_hit": CACHE_HEAVY_HIT, "batch_queue": BATCH_QUEUE,
                       "burst_queue": BURST_QUEUE},
        "candidates": rows,
        "recommended": recommended,
    }


def _print(results: dict) -> None:
    print(f"\n=== ARRIVAL PROBE (seed {results['seed']}, n={results['n_requests']}) ===")
    hdr = ["candidate", "rho", "len", "1turn", "INTER%", "prompt_p99",
           "turn>=2%", "queue_p99", "pass"]
    print("| " + " | ".join(hdr) + " |")
    print("| " + " | ".join("---" for _ in hdr) + " |")
    for r in results["candidates"]:
        print("| " + " | ".join([
            r["label"], f"{r['rho']:.2f}", f"{r['len_mean']:.0f}",
            "Y" if r["single_turn"] else "N",
            f"{r['interactive_frac']*100:.1f}", f"{r['eff_prompt_p99']:.0f}",
            f"{r['frac_turn_ge2']*100:.1f}", f"{r['queue_p99']:.0f}",
            "PASS" if r["passes"] else "fail",
        ]) + " |")
    rec = results["recommended"]
    if rec:
        rr = next(r for r in results["candidates"] if r["label"] == rec)
        print(f"\nRecommended: {rec} -> rho={rr['rho']}, len_mean={rr['len_mean']}, "
              f"single_turn={rr['single_turn']} "
              f"(INTERACTIVE {rr['interactive_frac']*100:.1f}%, "
              f"queue_p99={rr['queue_p99']:.0f}, prompt_p99={rr['eff_prompt_p99']:.0f}).")
    else:
        print("\nNo candidate cleared the INTERACTIVE bar -- widen the sweep.")


def main() -> None:
    p = argparse.ArgumentParser(description="2k INTERACTIVE arrival-model probe.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    results = run_sweep(args.seed, args.n)
    _print(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved arrival-probe results to {args.out}")


if __name__ == "__main__":
    main()
