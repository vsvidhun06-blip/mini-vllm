"""B7 -- does the reward-optimal max_batch_size move with OFFERED LOAD?

THE EVIDENCE GAP THIS CLOSES
----------------------------
`docs/paper/CLAIM_LEDGER.md` B7 asserts that the reward-optimal
`max_batch_size` is driven by load rather than prompt length ("light load wants
~16, heavy load wants ~2") and cites "load sweep in rule_only_ablation.py". No
such sweep exists: the phrase occurs only in two source comments, and the only
executable evidence is the Phase 4 arm sweep, which

  * varies prompt shape and arrival rate TOGETHER (light and heavy workloads
    differ in lambda, so a shape-vs-load attribution is not identified),
  * searches only CARL's 30 shipped arms, so `max_batch_size = 12` -- which the
    static grid selects in three scenarios -- is not even reachable, and
  * reports a single argmax over what is often an EXACT TIE: on the calibrated
    substrate `interactive` scores 0.471835622152 at max_batch 8, 16, 24 and 32
    alike, because below saturation the cap never binds.

This script holds the prompt/output shape FIXED and sweeps lambda alone, over
the full static `max_batch_size` axis, and reports tie sets rather than an
arbitrary argmax.

WHAT IT IS NOT
--------------
Not a CARL evaluation. No controller, no bandit, no regime classifier, no
episode structure. Every run here is one static configuration serving one
stationary stream.

WHY `evaluate_config` AND NOT `run_episode`
-------------------------------------------
`harness2.run_episode` re-bases every 10-request slice to t=0, which resets the
queue at each slice boundary. That is correct for a controller episode -- one
decision per slice -- but it destroys the steady-state backlog that "offered
load" is DEFINED by, which would be a first-order confounder here. So this uses
`engine_model.evaluate_config`: one continuous stationary stream per run, with
the request stream held identical across configurations at a given seed (paired
comparison).

Everything else is the existing infrastructure: `harness2.STATIC_GRID_AXES` for
the search space, `harness2.WORKLOADS` shapes, `src.carl.reward.utility_v2` for
the objective, `src.eval.provenance.write_result` for the artifact.

ON THE THREE OBJECTIVES
-----------------------
Analyses of the calibrated ablation showed a config ranking that FLIPS with the
reward scales (`RewardScales` are derived from each substrate's operating range,
so the yardstick moves when the substrate does). A load-vs-batch finding that
only holds under one yardstick is not a finding, so every cell is scored three
ways and the agreement between them is part of the result:

  primary      throughput_tps                     -- scale-free
  secondary    utility_v2 @ ablation scales       -- continuity with the
                                                     calibrated CARL rerun
  sensitivity  utility_v2 @ scales derived from
               this sweep's SUB-CRITICAL rows     -- the repaired evaluation's
                                                     own methodology, at this
                                                     experiment's scope

The sensitivity scales are derived from sub-critical rows ONLY on purpose:
pooling overloaded rows drags the observed TTFT median from ~10^2 to ~10^4 ms,
which would make the yardstick a function of the lambda grid's composition
rather than of the engine.

ON OVERLOAD
-----------
Above the measured saturation rate mu_max, lambda stops being the independent
variable -- the queue never drains, and the run is governed by the request COUNT
instead. Measured on the calibrated profile: interactive 5.56 req/s (CUDA
graphs) / 2.96 (eager); batch 2.57 / 1.41. Every row therefore carries
`rho = lambda / mu_max` and a `regime` label, and rows at rho > 1 are reported
as the overload regime rather than as distinct operating points.

Run:
  python scripts/eval/repair/load_sweep.py --smoke
  python scripts/eval/repair/load_sweep.py
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harness2 as H  # noqa: E402

from src.carl.config import CARLConfig  # noqa: E402
from src.carl.reward import RewardScales, utility_v2  # noqa: E402
from src.eval.engine_model import (  # noqa: E402
    HardwareProfile, WorkloadSpec, evaluate_config,
)
from src.eval.provenance import write_result  # noqa: E402

OUT = os.path.join(_ROOT, "docs", "eval", "raw", "repair", "load_sweep_b7.json")
ABLATION = os.path.join(_ROOT, "docs", "eval", "raw", "repair",
                        "rule_only_ablation_calibrated.json")

# Two shapes, taken verbatim from harness2.WORKLOADS. Only `rate_rps` and
# `n_requests` are overridden below; the length distributions are the
# repository's own, not new ones invented here.
SHAPES = {
    "interactive": "interactive",   # prompt 32+-8,  output 32+-8
    "batch": "batch",               # prompt 220+-40, output 64+-16
}

DEFAULT_LAMBDAS = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6,
                   8, 12, 16, 24, 32]

# The controlled operating point for the primary reduction: mid-grid chunk, and
# the eviction/preemption defaults. Chosen before seeing results.
CONTROL = {"chunk_size": 256, "eviction_threshold": 0.75, "preemption_enabled": True}

TIE_REL = 0.001   # 0.1% band for the "practically tied" set


# ---------------------------------------------------------------------------
# Grid selection: drop an axis only when it PROVABLY cannot bind.
# ---------------------------------------------------------------------------


def kv_worst_case(spec: WorkloadSpec, hw: HardwareProfile, seeds) -> int:
    """Largest per-request KV footprint (block-rounded) in the real streams.

    Uses the actual generated requests rather than a distributional bound, so
    the decision below is about the streams this sweep will really serve.
    """
    worst = 0
    for seed in seeds:
        for r in spec.generate(random.Random(seed)):
            blocks = math.ceil(max(1, r.prompt_len + r.output_len) / hw.block_size)
            worst = max(worst, blocks * hw.block_size)
    return worst


def axes_for_shape(spec: WorkloadSpec, hw: HardwareProfile, seeds) -> dict:
    """Which STATIC_GRID_AXES dimensions can bind for this shape?

    `eviction_threshold` and `preemption_enabled` act only through the KV
    admission cap. If the largest possible concurrent footprint -- every one of
    `max(max_batch_size)` rows holding the worst request in the stream -- still
    fits under the TIGHTEST cap, then neither knob can change any recorded
    metric, and sweeping them would multiply the run by six for nothing.

    The test is deliberately worst-case. An axis is kept whenever it MIGHT
    bind, even if a spot check happens not to trigger it: "did not bind in the
    cells I sampled" is not a proof, and dropping an axis on that basis would be
    dropping it for convenience.
    """
    axes = {k: list(v) for k, v in H.STATIC_GRID_AXES.items()}
    worst_req = kv_worst_case(spec, hw, seeds)
    max_mb = max(axes["max_batch_size"])
    peak = worst_req * max_mb
    tightest_cap = int(hw.kv_capacity_tokens() * min(axes["eviction_threshold"]))
    inert = peak <= tightest_cap
    justification = {
        "worst_request_kv_tokens": worst_req,
        "max_batch_size": max_mb,
        "peak_concurrent_kv_tokens": peak,
        "tightest_admit_cap_tokens": tightest_cap,
        "kv_pool_tokens": hw.kv_capacity_tokens(),
        "kv_can_bind": not inert,
    }
    if inert:
        axes["eviction_threshold"] = [CONTROL["eviction_threshold"]]
        axes["preemption_enabled"] = [CONTROL["preemption_enabled"]]
        justification["decision"] = (
            "eviction_threshold and preemption_enabled DROPPED: the peak "
            "concurrent KV footprint cannot reach even the tightest admission "
            "cap, so neither knob can alter any recorded metric for this shape.")
    else:
        justification["decision"] = (
            "eviction_threshold and preemption_enabled KEPT: the peak "
            "concurrent KV footprint exceeds the tightest admission cap, so "
            "admission can bind and the knobs are not provably inert.")
    justification["axes_used"] = axes
    return axes, justification


def grid(axes: dict) -> list[CARLConfig]:
    import itertools
    return [
        CARLConfig(max_batch_size=mb, chunk_size=cs, eviction_threshold=ev,
                   preemption_enabled=pe).clamp()
        for mb, cs, ev, pe in itertools.product(
            axes["max_batch_size"], axes["chunk_size"],
            axes["eviction_threshold"], axes["preemption_enabled"])
    ]


# ---------------------------------------------------------------------------
# Saturation.
# ---------------------------------------------------------------------------


def measure_saturation(spec_shape: WorkloadSpec, hw: HardwareProfile,
                       graphs: bool, n_requests: int, seeds=(0, 1, 2)) -> float:
    """Max sustainable completion rate: a burst dump at the largest batch.

    This is the denominator for rho. It is measured, not assumed, because the
    calibrated eager and CUDA-graph arms differ by roughly 2x.
    """
    burst = WorkloadSpec(spec_shape.name + "_burst", n_requests,
                         spec_shape.prompt_mean, spec_shape.prompt_std,
                         spec_shape.output_mean, spec_shape.output_std,
                         arrival="burst")
    cfg = CARLConfig(max_batch_size=max(H.STATIC_GRID_AXES["max_batch_size"]),
                     chunk_size=CONTROL["chunk_size"],
                     use_cuda_graphs=graphs).clamp()
    rates = []
    for s in seeds:
        r = evaluate_config(cfg, burst, hw, s)
        rates.append(r.n_completed / max(r.wall_s, 1e-9))
    return statistics.fmean(rates)


def regime_for(rho: float) -> str:
    if rho < 0.95:
        return "subcritical"
    if rho < 1.05:
        return "critical"
    return "overload"


# ---------------------------------------------------------------------------
# Tie sets.
# ---------------------------------------------------------------------------


def tie_sets(per_mb: dict) -> dict:
    """Argmax plus the two tie sets, over {max_batch_size: value}.

    Reporting a bare argmax here would be the exact defect this experiment
    exists to correct: below saturation the cap does not bind, so every
    max_batch above the binding point returns a BIT-IDENTICAL value and the
    argmax is decided by dict ordering.
    """
    best_mb = max(per_mb, key=lambda k: (per_mb[k], -k))
    best = per_mb[best_mb]
    exact = sorted(k for k, v in per_mb.items() if v == best)
    near = sorted(k for k, v in per_mb.items()
                  if best == 0 or (best - v) <= abs(best) * TIE_REL)
    return {
        "argmax_max_batch_size": best_mb,
        "best_value": best,
        "exact_tie_set": exact,
        "n_exact_ties": len(exact),
        "tie_set_within_0p1pct": near,
        "n_ties_within_0p1pct": len(near),
        # The honest headline: with an exact tie set of size > 1 there is no
        # identified optimum, only a smallest sufficient batch size.
        "optimum_identified": len(exact) == 1,
        "smallest_sufficient_max_batch_size": min(exact),
        "per_max_batch": {str(k): per_mb[k] for k in sorted(per_mb)},
    }


# ---------------------------------------------------------------------------
# Sweep.
# ---------------------------------------------------------------------------


def sweep(shapes: dict, lambdas: list, seeds: list, n_requests: int,
          hw: HardwareProfile, arms=(True, False)) -> tuple[list, dict, dict]:
    rows: list[dict] = []
    grid_decisions: dict = {}
    saturation: dict = {}

    for shape_key, wl_name in shapes.items():
        base = H.WORKLOADS[wl_name]
        probe = WorkloadSpec(shape_key, n_requests, base.prompt_mean,
                             base.prompt_std, base.output_mean, base.output_std,
                             arrival="poisson", rate_rps=1.0)
        axes, justification = axes_for_shape(probe, hw, seeds)
        cfgs = grid(axes)
        grid_decisions[shape_key] = dict(
            justification,
            n_configs=len(cfgs),
            shape={"workload": wl_name, "prompt_mean": base.prompt_mean,
                   "prompt_std": base.prompt_std, "output_mean": base.output_mean,
                   "output_std": base.output_std})

        for graphs in arms:
            mu = measure_saturation(probe, hw, graphs, n_requests)
            saturation["%s@%s" % (shape_key, "graph" if graphs else "eager")] = {
                "mu_max_req_per_s": mu, "method": "burst dump at max_batch=32",
                "n_seeds": 3, "n_requests": n_requests}
            print("  [%s/%s] mu_max = %.3f req/s; %d configs x %d lambda x %d seeds"
                  % (shape_key, "graph" if graphs else "eager", mu, len(cfgs),
                     len(lambdas), len(seeds)), flush=True)

            for lam in lambdas:
                spec = WorkloadSpec(shape_key, n_requests, base.prompt_mean,
                                    base.prompt_std, base.output_mean,
                                    base.output_std, arrival="poisson",
                                    rate_rps=lam)
                rho = lam / mu if mu > 0 else float("inf")
                for cfg in cfgs:
                    c = CARLConfig(**dict(cfg.as_dict(), use_cuda_graphs=graphs)).clamp()
                    runs = [evaluate_config(c, spec, hw, s) for s in seeds]
                    rows.append({
                        "shape": shape_key,
                        "arm": "graph" if graphs else "eager",
                        "lambda_rps": lam,
                        "rho": rho,
                        "regime": regime_for(rho),
                        "max_batch_size": c.max_batch_size,
                        "chunk_size": c.chunk_size,
                        "eviction_threshold": c.eviction_threshold,
                        "preemption_enabled": c.preemption_enabled,
                        "n_seeds": len(seeds),
                        "throughput_tps": statistics.fmean(r.throughput_tps for r in runs),
                        "throughput_tps_std": (statistics.stdev(
                            [r.throughput_tps for r in runs]) if len(runs) > 1 else 0.0),
                        "ttft_p99_ms": statistics.fmean(r.ttft_p99_ms for r in runs),
                        "tpot_p99_ms": statistics.fmean(r.tpot_p99_ms for r in runs),
                        "ttft_p50_ms": statistics.fmean(r.ttft_p50_ms for r in runs),
                        "tpot_p50_ms": statistics.fmean(r.tpot_p50_ms for r in runs),
                        # Utilisation / cap binding.
                        "mean_batch": statistics.fmean(r.mean_batch for r in runs),
                        "mean_occupancy": statistics.fmean(r.mean_occupancy for r in runs),
                        "kv_utilisation": statistics.fmean(r.kv_utilisation for r in runs),
                        "preemptions": statistics.fmean(r.preemptions for r in runs),
                        # Proxy only: RunResult records the MEAN batch, not the
                        # per-step maximum, so this cannot prove the cap was hit
                        # on some step. The exact tie sets below are the direct
                        # evidence of non-binding and should be read first.
                        "cap_binding_proxy": statistics.fmean(
                            r.mean_occupancy for r in runs) >= 0.95,
                        "n_completed": statistics.fmean(r.n_completed for r in runs),
                        "truncated_any": any(r.truncated for r in runs),
                        "livelocked_any": any(r.livelocked for r in runs),
                    })
    return rows, grid_decisions, saturation


# ---------------------------------------------------------------------------
# Scales and objectives.
# ---------------------------------------------------------------------------


def ablation_scales(path: str) -> tuple[RewardScales, dict]:
    """The scales the calibrated CARL rerun used, read from its artifact.

    Read rather than transcribed so the two experiments cannot silently drift
    apart, and so the artifact records where the numbers came from.
    """
    import json
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    s = payload["reward_scales"]
    prov = payload.get("_provenance", {})
    return RewardScales(**s), {
        "source_artifact": os.path.relpath(path, _ROOT).replace("\\", "/"),
        "source_git_sha": prov.get("git_sha"),
        "source_timestamp_utc": prov.get("timestamp_utc"),
        "values": s,
    }


def derived_scales(rows: list, shape: str, arm: str) -> tuple[RewardScales, dict]:
    """RewardScales.from_measurements over this sweep's SUB-CRITICAL rows only.

    Same methodology as `rule_only_ablation.py`; different scope. Overloaded
    rows are excluded because their TTFT p99 is two orders of magnitude larger,
    so including them would make the derived targets depend on how many
    overloaded lambdas the grid happens to contain.
    """
    sub = [r for r in rows
           if r["shape"] == shape and r["arm"] == arm and r["regime"] == "subcritical"]
    if not sub:
        return None, {"derived": False, "reason": "no sub-critical rows"}
    sc = RewardScales.from_measurements(
        [r["throughput_tps"] for r in sub], [r["ttft_p99_ms"] for r in sub],
        [r["tpot_p99_ms"] for r in sub])
    return sc, {"derived": True, "n_rows": len(sub),
                "scope": "subcritical rows only (rho < 0.95)",
                "values": {"t_half": sc.t_half, "ttft_target": sc.ttft_target,
                           "tpot_target": sc.tpot_target,
                           "sharpness": sc.sharpness}}


def _utility(row: dict, scales: RewardScales) -> float:
    return utility_v2({"throughput_tps": row["throughput_tps"],
                       "ttft_p99_ms": row["ttft_p99_ms"],
                       "tpot_p99_ms": row["tpot_p99_ms"],
                       "cache_hit_rate": 0.0}, None, scales)


def analyse(rows: list, saturation: dict, abl: RewardScales) -> dict:
    """Per shape x arm x objective x reduction x lambda: tie sets over max_batch."""
    out: dict = {}
    shapes = sorted({r["shape"] for r in rows})
    for shape in shapes:
        out[shape] = {}
        for arm in ("graph", "eager"):
            sub = [r for r in rows if r["shape"] == shape and r["arm"] == arm]
            if not sub:
                continue
            der, der_meta = derived_scales(rows, shape, arm)
            objectives = {
                "throughput_tps": (lambda r: r["throughput_tps"]),
                "utility_ablation_scales": (lambda r, s=abl: _utility(r, s)),
            }
            if der is not None:
                objectives["utility_derived_subcritical_scales"] = (
                    lambda r, s=der: _utility(r, s))

            arm_out = {"derived_scales": der_meta, "by_objective": {}}
            for oname, fn in objectives.items():
                arm_out["by_objective"][oname] = {"controlled": {}, "joint": {}}
                for lam in sorted({r["lambda_rps"] for r in sub}):
                    at_lam = [r for r in sub if r["lambda_rps"] == lam]
                    ctrl_rows = [r for r in at_lam
                                 if r["chunk_size"] == CONTROL["chunk_size"]
                                 and r["eviction_threshold"] == CONTROL["eviction_threshold"]
                                 and r["preemption_enabled"] == CONTROL["preemption_enabled"]]
                    meta = {
                        "rho": at_lam[0]["rho"], "regime": at_lam[0]["regime"],
                        "mean_batch_at_argmax": None,
                    }
                    # Controlled: one config per max_batch.
                    ctrl = {r["max_batch_size"]: fn(r) for r in ctrl_rows}
                    ts = dict(tie_sets(ctrl), **meta)
                    ts["mean_batch_at_argmax"] = next(
                        r["mean_batch"] for r in ctrl_rows
                        if r["max_batch_size"] == ts["argmax_max_batch_size"])
                    ts["mean_occupancy_by_max_batch"] = {
                        str(r["max_batch_size"]): r["mean_occupancy"] for r in
                        sorted(ctrl_rows, key=lambda x: x["max_batch_size"])}
                    arm_out["by_objective"][oname]["controlled"][str(lam)] = ts

                    # Joint: best over every other axis, per max_batch.
                    joint: dict = {}
                    best_cfg: dict = {}
                    for r in at_lam:
                        v = fn(r)
                        mb = r["max_batch_size"]
                        if mb not in joint or v > joint[mb]:
                            joint[mb] = v
                            best_cfg[mb] = {"chunk_size": r["chunk_size"],
                                            "eviction_threshold": r["eviction_threshold"],
                                            "preemption_enabled": r["preemption_enabled"],
                                            "mean_batch": r["mean_batch"],
                                            "mean_occupancy": r["mean_occupancy"]}
                    tj = dict(tie_sets(joint), **meta)
                    tj["best_other_axes_by_max_batch"] = {
                        str(k): best_cfg[k] for k in sorted(best_cfg)}
                    tj["mean_batch_at_argmax"] = best_cfg[
                        tj["argmax_max_batch_size"]]["mean_batch"]
                    arm_out["by_objective"][oname]["joint"][str(lam)] = tj
            out[shape][arm] = arm_out
    return out


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ablation-scales-from", default=ABLATION)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid for wiring verification; NOT a result")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = list(range(args.seeds))
    shapes = {k: SHAPES[k] for k in args.shapes.split(",") if k in SHAPES}
    n_requests = args.requests
    if args.smoke:
        lambdas = [0.5, 2.0, 32.0]
        seeds = [0, 1]
        n_requests = 60

    hw = HardwareProfile()
    abl, abl_meta = ablation_scales(args.ablation_scales_from)
    print("Ablation scales (%s): t_half=%.4f ttft=%.4f tpot=%.4f"
          % (abl_meta["source_artifact"], abl.t_half, abl.ttft_target, abl.tpot_target))
    print("Sweep: shapes=%s lambdas=%d seeds=%d requests=%d"
          % (list(shapes), len(lambdas), len(seeds), n_requests), flush=True)

    t0 = time.perf_counter()
    rows, grid_decisions, saturation = sweep(shapes, lambdas, seeds, n_requests, hw)
    elapsed = time.perf_counter() - t0
    print("\n%d rows in %.1f s" % (len(rows), elapsed), flush=True)

    for k, d in grid_decisions.items():
        print("  grid[%s]: %d configs -- %s" % (k, d["n_configs"], d["decision"]))

    analysis = analyse(rows, saturation, abl)

    payload = {
        "description": (
            "B7: reward-optimal max_batch_size versus offered load, at FIXED "
            "prompt/output shape, on the calibrated mechanistic substrate. Not "
            "a CARL evaluation -- static configurations only, no controller."),
        "substrate": "src/eval/engine_model (calibrated HardwareProfile)",
        "hardware_profile": hw.__dict__,
        "calibration_source": ("docs/eval/raw/repair/"
                               "batch_intervention_phase15_16_s42_s43_s44.json"),
        "run_primitive": "engine_model.evaluate_config (one continuous stream)",
        "run_primitive_note": (
            "harness2.run_episode is deliberately NOT used: it re-bases each "
            "10-request slice to t=0, resetting the queue and destroying the "
            "steady-state backlog that offered load is defined by."),
        "shapes": {k: {"workload": v} for k, v in shapes.items()},
        "lambdas_rps": lambdas,
        "seeds": seeds,
        "n_requests_per_run": n_requests,
        "arms": ["graph", "eager"],
        "control_point": CONTROL,
        "grid_axes_full": {k: list(v) for k, v in H.STATIC_GRID_AXES.items()},
        "grid_decision": grid_decisions,
        "saturation": saturation,
        "objectives": {
            "primary": "throughput_tps (scale-free)",
            "secondary": "utility_v2 @ ablation RewardScales",
            "sensitivity": "utility_v2 @ RewardScales derived from subcritical rows",
        },
        "ablation_scales": abl_meta,
        "tie_policy": {
            "exact_tie_set": "values equal to the maximum exactly",
            "tie_set_within_0p1pct": "values within 0.1% of the maximum",
            "note": ("A bare argmax is never the result. Below saturation the "
                     "cap does not bind and every sufficiently large batch "
                     "returns a bit-identical value; the argmax is then decided "
                     "by iteration order, not by the engine."),
        },
        "smoke": bool(args.smoke),
        "elapsed_s": elapsed,
        "rows": rows,
        "analysis": analysis,
    }
    path = write_result(args.out, payload, "load_sweep_b7",
                        script="scripts/eval/repair/load_sweep.py",
                        extra={"shapes": list(shapes), "lambdas": lambdas,
                               "seeds": seeds, "n_requests": n_requests,
                               "substrate": "engine_model_calibrated"})
    print("\nWrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
