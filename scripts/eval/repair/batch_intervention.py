"""PHASE 15+16 -- the batch-size intervention, and the host-overhead question.

REQUIRES A GPU. This host has none (torch CPU build); the script is implemented
and import-tested here and must be RUN on a T4/A100 box. See
docs/eval/REPRODUCIBILITY.md for the hand-off command. It fabricates nothing: if
CUDA is unavailable it refuses to run rather than emitting placeholder numbers.

WHY THIS IS THE MOST IMPORTANT MISSING EXPERIMENT
--------------------------------------------------
The paper wants to claim a mechanism:

    workload shift -> queue composition -> batch size -> GPU occupancy -> throughput

Not one link in that chain has ever been measured. The evidence offered is an
ablation that pins `max_batch_size` to 4 and observes that throughput falls --
which establishes only that a small batch is bad for batch-shaped traffic. It is
an observational comparison between two policies, not an intervention on the
proposed causal variable.

This script intervenes directly. It holds the workload FIXED, forces
`max_batch_size` across a sweep, and measures every link:

    forced max_batch_size
      -> REALISED mean batch (do we actually get the batch we asked for?)
      -> GPU occupancy / utilisation
      -> throughput, TTFT, TPOT
      -> host (Python dispatch) share of step time

Only with the middle links measured can "batch size causes throughput" be said
at all. Note the first link is not guaranteed: below saturation the realised
batch is bounded by ARRIVALS, not by the cap, so raising `max_batch_size` changes
nothing. The mechanistic simulation predicts exactly that (mean batch ~1.8
regardless of cap at rho well below 1). Whether the real engine agrees is the
question.

PHASE 16 -- IS IT JUST PYTHON DISPATCH?
---------------------------------------
The repository's own `docs/vllm_comparison.md` states mini-vLLM is "roughly an
order of magnitude slower on TPOT" than vLLM because of "hundreds of separate
CUDA kernel launches per decode step, each an eager Python->C++ dispatch", and
that "the throughput gap widens from batch=4 to batch=8". In a host-bound engine,
raising the batch amortises a large fixed per-step Python cost over more rows --
so a big throughput-vs-batch slope may be a property of THIS engine rather than
of LLM serving.

So the sweep is run twice, with CUDA graphs off and on. If the throughput(batch)
slope survives graph capture, the effect is not merely dispatch amortisation and
the external-validity argument is much stronger. If it flattens, that is a
boundary condition on the whole paper and must be reported as one.

WHAT THIS ALSO PRODUCES
-----------------------
A calibrated `HardwareProfile` for `src/eval/engine_model`. The defaults there
are anchored to two batch=1 datapoints and are explicitly labelled an assumption
about SHAPE. This sweep is the measurement that replaces them, at which point
every simulation result in the repair pass can be re-run against real hardware
constants.

Run (on a GPU box):
  python scripts/eval/repair/batch_intervention.py
  python scripts/eval/repair/batch_intervention.py --batches 1,2,4,8,16,32 --seeds 42,43,44
  python scripts/eval/repair/batch_intervention.py --no-cuda-graph-arm   # skip Phase 16
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
for _p in (_ROOT, os.path.dirname(_THIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_ROOT, "docs", "eval", "raw", "repair", "batch_intervention.json")

DEFAULT_BATCHES = [1, 2, 4, 8, 12, 16, 24, 32]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_RATES = [None, 2.0, 8.0]   # None = burst (legacy bulk dump)

# A requested-graphs arm below this replay fraction is a MIXTURE of graph and
# eager execution; its throughput is attributable to neither, so Phase 16's
# eager-vs-graph slope comparison must not use it. See graph_accounting().
GRAPH_CLEAN_HIT_RATE = 0.95


def _require_cuda() -> None:
    """Refuse to produce numbers on CPU.

    This project already has two hand-reconstructed 'GPU' result files in
    quarantine (docs/eval/legacy/). The failure mode that produced them was a
    GPU run whose artifact never made it back. The mitigation is to make
    non-GPU execution a hard error and to write the artifact incrementally.
    """
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "batch_intervention.py requires CUDA. This is a MEASUREMENT script; "
            "running it on CPU would produce numbers that do not describe the "
            "hardware under study. Run it on the T4/A100 box.\n"
            "See docs/eval/REPRODUCIBILITY.md."
        )


def graph_repair_provenance() -> dict:
    """Fingerprint the CUDA-graph implementation this process is running.

    WHY THIS EXISTS. The second T4 smoke returned byte-identical fallback counts
    to the first (2350 / 627) and zero graph hits. The cause was not a logic bug
    at all: the GPU box was running a bundle built BEFORE the repair, so
    `src/engine/live_graph.py` was not present and the runner was never
    attached. Nothing in the artifact recorded which implementation produced it,
    so two runs of different code were indistinguishable in the results.

    Every artifact now carries this block, and `require_graph_repair()` turns a
    stale checkout into an immediate, explicit failure instead of a plausible
    looking table of eager numbers.
    """
    import hashlib

    info = {"live_graph_present": False, "live_graph_sha256": None,
            "live_graph_bytes": None, "scheduler_has_reason_counts": False,
            "git_describe": None}
    path = os.path.join(_ROOT, "src", "engine", "live_graph.py")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            blob = fh.read()
        info["live_graph_present"] = True
        info["live_graph_bytes"] = len(blob)
        info["live_graph_sha256"] = hashlib.sha256(blob).hexdigest()[:16]
    try:
        from src.engine.scheduler import ContinuousBatchScheduler
        info["scheduler_has_reason_counts"] = hasattr(
            ContinuousBatchScheduler, "graph_diagnostics")
    except Exception:                                          # noqa: BLE001
        pass
    try:
        import subprocess
        info["git_describe"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:                                          # noqa: BLE001
        pass
    return info


def require_graph_repair() -> dict:
    """Refuse to run the graph arm from a checkout that predates the repair.

    A missing `live_graph.py` or a scheduler without the fallback-reason
    counters means this process CANNOT produce CUDA-graph execution. Running
    anyway yields eager rows in a slot labelled "graphs", which is what both
    failed smokes did. Fail here instead, and say exactly what to re-sync.
    """
    prov = graph_repair_provenance()
    missing = []
    if not prov["live_graph_present"]:
        missing.append("src/engine/live_graph.py is absent")
    if not prov["scheduler_has_reason_counts"]:
        missing.append("src/engine/scheduler.py has no graph_diagnostics() "
                       "(pre-repair scheduler)")
    if missing:
        raise SystemExit(
            "REFUSING TO RUN THE CUDA-GRAPH ARM: this checkout predates the "
            "graph repair.\n  - " + "\n  - ".join(missing) + "\n\n"
            "This is exactly what produced the two zero-hit smoke runs: the GPU "
            "box was executing a bundle built before the repair, so no graph "
            "runner was ever attached and every decode fell back to eager.\n"
            "Re-sync the working tree to the GPU box (all four of "
            "src/engine/live_graph.py, src/engine/scheduler.py, "
            "src/engine/attention.py, scripts/eval/repair/batch_intervention.py) "
            "and re-run. Use --no-cuda-graph-arm to collect eager rows only."
        )
    return prov


def graph_accounting(requested: bool, hits: int, fallbacks: int) -> dict:
    """The graph-arm self-verification block, as a pure function.

    Split out of `run_one` so the contract can be tested WITHOUT a GPU. The
    contract, which is the whole point of this experiment's Phase 16 arm:

      * `use_cuda_graphs` reports what EXECUTED, not what was asked for. A
        requested-graphs run that never replayed a graph ran eager, and saying
        otherwise is how the first T4 smoke produced four eager rows labelled as
        two eager and two CUDA-graph arms.
      * `cuda_graph_hits` counts replays only. Eager fallbacks are a separate
        counter and are never folded into it.
      * `cuda_graph_arm_valid` is False for a requested-graphs arm with zero
        hits. The eager arm is always valid -- it makes no graph claim.
      * `cuda_graph_arm_clean` additionally requires that the arm was
        OVERWHELMINGLY graph-executed. An arm at, say, 0.4 hit rate is a
        mixture of two execution modes and its throughput attributes to
        neither; it is `valid` (it did use graphs) but not `clean`, and the
        Phase 16 slope comparison must use clean arms only.
    """
    total = hits + fallbacks
    hit_rate = (hits / total) if total else None
    valid = (hits > 0) if requested else True
    return {
        "use_cuda_graphs_requested": bool(requested),
        "use_cuda_graphs": bool(requested and hits > 0),
        "cuda_graph_hits": hits,
        "cuda_graph_eager_fallbacks": fallbacks,
        "cuda_graph_decode_steps": total,
        "cuda_graph_hit_rate": hit_rate,
        "cuda_graph_arm_valid": valid,
        "cuda_graph_arm_clean": (
            bool(valid and hit_rate is not None
                 and hit_rate >= GRAPH_CLEAN_HIT_RATE)
            if requested else True),
        "cuda_graph_clean_threshold": GRAPH_CLEAN_HIT_RATE,
    }


def run_one(model, tokenizer, max_batch: int, chunk: int, seed: int,
            n_requests: int, arrival: str, rate_rps: float,
            use_cuda_graphs: bool, graph_seq_buckets: int = 4,
            graph_max_batch: int = 32) -> dict:
    """Serve one fixed workload under a FORCED max_batch_size; measure everything.

    The controller is absent by design: this is an intervention on the causal
    variable, so nothing is allowed to change it mid-run.
    """
    import random

    import torch

    from src.carl.live import BLOCK_SIZE, NUM_BLOCKS, _build_workload, _percentile
    from src.engine.scheduler import ContinuousBatchScheduler

    rng = random.Random(seed)
    specs = _build_workload(tokenizer, "NON-STATIONARY", n_requests, rng,
                            arrival=arrival, rate_rps=rate_rps)

    # CUDA-graph accounting. scheduler._decode_forward falls back to eager
    # whenever `_graph_runner is None` or `can_replay()` says no graph was
    # captured for this exact decode batch. Setting the flag is therefore NOT
    # sufficient: without counting hits, a graphs=True arm that silently ran
    # eager would be recorded as `use_cuda_graphs: true` and the Phase 16
    # comparison ("does the batch effect survive graph capture?") would be
    # answered by an arm that never used graphs. We count and report instead.
    graph_hits = {"hit": 0, "eager": 0}

    def _observe_graph(hit: bool) -> None:
        graph_hits["hit" if hit else "eager"] += 1

    sched = ContinuousBatchScheduler(
        model, max_batch_size=max_batch, num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE, chunk_size=chunk, enable_spec_decode=False,
        use_cuda_graphs=bool(use_cuda_graphs),
        cuda_graph_observer=_observe_graph,
    )

    # --- Phase 16: attach a real capture/replay runner -------------------
    #
    # THE BUG THE FIRST T4 SMOKE EXPOSED. `use_cuda_graphs=True` only gates a
    # branch in `_decode_forward`; the branch also requires
    # `scheduler._graph_runner is not None`, and NOTHING in this repository ever
    # assigned that attribute. So the graphs=True arm ran eager on every one of
    # its 2350/627 decode steps, and its apparent throughput advantage was
    # ordinary run-to-run variation between two eager runs.
    #
    # Capture happens HERE -- after the scheduler builds its pool, before t0 and
    # before any request is admitted. Both orderings are load-bearing: the
    # graphs bake in this pool's addresses, and the capture pass writes scratch
    # K/V into block 0, which is only safe while the pool is empty. Capture time
    # is outside the measured window by construction.
    graph_report = None
    graph_selftest = None
    if use_cuda_graphs:
        if not torch.cuda.is_available():
            raise SystemExit("the CUDA-graph arm requires CUDA")
        require_graph_repair()
        from src.engine.live_graph import LiveDecodeGraphRunner

        # The largest context any request in THIS workload will reach. Sizing
        # the buckets from the workload (rather than from a constant) keeps the
        # padding a bucket introduces bounded and makes it reportable.
        max_ctx = max(int(s.prompt_ids.shape[1]) + int(s.max_new) for s in specs)
        runner = LiveDecodeGraphRunner(
            model, sched.pool,
            max_batch_size=max_batch,
            max_context_tokens=max_ctx,
            num_seq_buckets=graph_seq_buckets,
            max_graph_batch=graph_max_batch,
        )
        # capture_all() raises on ANY capture failure. It used to record the
        # failure and continue, which meant a fully-failed grid degraded quietly
        # into an eager run wearing a graph label.
        graph_report = runner.capture_all()
        sched._graph_runner = runner
        print(f"    captured {graph_report['graphs_captured']} decode graphs "
              f"(batch {runner.batch_sizes[0]}..{runner.batch_sizes[-1]} x "
              f"blocks {list(runner.buckets)}) in "
              f"{graph_report['capture_seconds']:.1f}s", flush=True)

        # PROOF OF LIFE, before t0. Replays a captured graph against real
        # PagedRequestCache objects and checks it against eager. Without this,
        # the first evidence that replay works is the hit count at the END of
        # the measured run -- which is how two smokes came back at zero with no
        # indication of why.
        graph_selftest = runner.self_test()
        print(f"    graph self-test OK: replay == eager on batch sizes "
              f"{graph_selftest['checked_batch_sizes']} "
              f"(max|delta| = {graph_selftest['max_abs_logit_delta']:.2e})",
              flush=True)

    submit_t: dict[str, float] = {}
    first_t: dict[str, float] = {}
    last_t: dict[str, float] = {}
    ntok: dict[str, int] = {}

    pending = sorted(specs, key=lambda s: s.arrival_offset_s)
    idx = 0

    def _submit(spec) -> None:
        submit_t[spec.rid] = time.perf_counter()
        ntok[spec.rid] = 0
        sched.add_request(spec.rid, spec.prompt_ids, max_new_tokens=spec.max_new,
                          eos_token_id=None)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while idx < len(pending) and pending[idx].arrival_offset_s <= 0.0:
        _submit(pending[idx]); idx += 1

    batch_samples: list[int] = []
    step_times: list[float] = []
    util_samples: list[float] = []
    total_tokens = 0
    last = time.perf_counter()

    # NVML handle for real GPU utilisation, if available. Absent -> None, never
    # a fabricated 0.0 presented as a measurement.
    nvml = _nvml_handle()

    while sched.has_work() or idx < len(pending):
        elapsed = time.perf_counter() - t0
        while idx < len(pending) and pending[idx].arrival_offset_s <= elapsed:
            _submit(pending[idx]); idx += 1
        if not sched.has_work():
            if idx >= len(pending):
                break
            time.sleep(max(0.0, pending[idx].arrival_offset_s
                           - (time.perf_counter() - t0)))
            continue

        emitted = sched.step()
        now = time.perf_counter()
        step_times.append(now - last)
        batch_samples.append(len(sched.active))
        # NVML is queried AFTER the step-time sample is taken and `last` is
        # re-stamped afterwards, so the ~0.1-1ms NVML round trip cannot leak
        # into the next step's measured duration. step_time_mean_ms is the
        # input to fit_hardware_profile(), so contaminating it would corrupt
        # the calibration this experiment exists to produce.
        if nvml is not None:
            util_samples.append(_nvml_util(nvml))
        last = time.perf_counter()
        for rid, _tok in emitted:
            first_t.setdefault(rid, now)
            last_t[rid] = now
            ntok[rid] += 1
            total_tokens += 1
        for r in sched.get_finished():
            pass

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    ttfts = [(first_t[r] - submit_t[r]) * 1000.0 for r in first_t]
    tpots = [((last_t[r] - first_t[r]) / (ntok[r] - 1) * 1000.0)
             for r in first_t if ntok.get(r, 0) > 1]
    mean_batch = statistics.fmean(batch_samples) if batch_samples else 0.0

    row = {
        "max_batch_size_forced": max_batch,
        "chunk_size": chunk,
        "arrival": arrival, "rate_rps": rate_rps, "seed": seed,
        "n_requests": n_requests,
        # --- link 1: did we actually GET the batch we asked for? -------------
        "realised_mean_batch": mean_batch,
        "realised_max_batch": max(batch_samples) if batch_samples else 0,
        "occupancy": mean_batch / max_batch if max_batch else 0.0,
        "cap_was_binding": bool(batch_samples) and max(batch_samples) >= max_batch,
        # --- link 2: occupancy -> utilisation --------------------------------
        "gpu_utilisation_mean": statistics.fmean(util_samples) if util_samples else None,
        "gpu_utilisation_available": nvml is not None,
        # --- link 3: -> throughput / latency ---------------------------------
        "throughput_tps": total_tokens / wall if wall > 0 else 0.0,
        "ttft_p50_ms": _pct(ttfts, 50), "ttft_p99_ms": _pct(ttfts, 99),
        "tpot_p50_ms": _pct(tpots, 50), "tpot_p99_ms": _pct(tpots, 99),
        "wall_s": wall, "steps": len(step_times),
        "total_tokens": total_tokens,
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0 if step_times else 0.0,
        "step_time_p99_ms": _pct([t * 1000.0 for t in step_times], 99),
    }
    # Self-verification for Phase 16. Keys are merged (not nested) so the
    # existing consumers of `use_cuda_graphs` / `cuda_graph_hits` keep working;
    # `use_cuda_graphs` now reports EXECUTION, not the request. See
    # graph_accounting() for the contract and why it is a separate function.
    row.update(graph_accounting(bool(use_cuda_graphs),
                                graph_hits["hit"], graph_hits["eager"]))
    row["cuda_graph_capture"] = graph_report
    row["cuda_graph_selftest"] = graph_selftest
    # The decisive diagnostic. `reasons` holds one entry per batched-decode
    # forward, so a zero-hit arm names its own cause without a rerun:
    #   graph_runner_missing -> the runner was never attached (stale code)
    #   graphs_disabled      -> use_cuda_graphs was False
    #   runner_rejected      -> attached but refusing; runner_fallback_reasons
    #                           says whether it was batch size, KV pool
    #                           identity, ragged contexts or the bucket
    row["cuda_graph_diagnostics"] = sched.graph_diagnostics()
    row["cuda_graph_fallback_reasons"] = (
        dict(sched._graph_runner.fallback_reasons)
        if getattr(sched, "_graph_runner", None) is not None else {})
    row["code_provenance"] = graph_repair_provenance()
    return row


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _nvml_handle():
    try:
        import pynvml
        pynvml.nvmlInit()
        return (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception:
        return None


def _nvml_util(handle) -> float:
    mod, h = handle
    try:
        return float(mod.nvmlDeviceGetUtilizationRates(h).gpu) / 100.0
    except Exception:
        return 0.0


def fit_hardware_profile(rows: list[dict]) -> dict:
    """Least-squares fit of step_time(b) = fixed + b * c_dec from the sweep.

    This is the calibration that replaces the anchored guesses in
    `src/eval/engine_model.HardwareProfile`. Reported with the residual so a
    poor fit is visible rather than silently adopted -- if step time is not
    affine in batch, the engine model's core equation is wrong and should be
    revised rather than re-fitted.
    """
    pts = [(r["realised_mean_batch"], r["step_time_mean_ms"] / 1000.0)
           for r in rows if r.get("steps") and r["realised_mean_batch"] > 0]
    if len(pts) < 2:
        return {"fitted": False, "reason": "fewer than 2 usable points"}
    n = len(pts)
    sx = sum(b for b, _ in pts); sy = sum(t for _, t in pts)
    sxx = sum(b * b for b, _ in pts); sxy = sum(b * t for b, t in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return {"fitted": False, "reason": "degenerate design (all batches equal)"}
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    ss_res = sum((t - (intercept + slope * b)) ** 2 for b, t in pts)
    ss_tot = sum((t - sy / n) ** 2 for _, t in pts)
    return {
        "fitted": True,
        "decode_per_row_s": slope,
        "fixed_per_step_s": intercept,
        "r_squared": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n_points": n,
        "note": ("fixed_per_step_s bundles host_overhead_s and "
                 "decode_kernel_floor_s; the CUDA-graph arm separates them."),
    }


def _binding_summary(rows: list[dict], key) -> dict:
    """Aggregate ``cap_was_binding`` over every row sharing a key.

    The older dict-comprehension summary emitted one key per row, so repeated
    (arrival, rate) combinations overwrote each other and only the final row
    survived. That can collapse a large sweep into a few arbitrary booleans.
    This summary preserves the full evidence by counting all rows in each group.
    """
    out: dict = {}
    for r in rows:
        group = key(r)
        entry = out.setdefault(group, {"n_binding": 0, "n_rows": 0})
        entry["n_rows"] += 1
        if r["cap_was_binding"]:
            entry["n_binding"] += 1

    for entry in out.values():
        n = entry["n_rows"]
        entry["fraction"] = entry["n_binding"] / n if n else 0.0
        entry["all_binding"] = entry["n_binding"] == n
        entry["any_binding"] = entry["n_binding"] > 0
    return out


def build_analysis(rows: list[dict]) -> dict:
    """Post-sweep analysis, as a pure function of the rows.

    Extracted from `main` so it can be exercised WITHOUT a GPU. The failure this
    guards against is specific and has already cost this repository two result
    files: a long GPU sweep that completes, then dies in the summary block on a
    renamed key, losing the artifact. It is also where the mislabelled-arm rule
    is enforced, which is worth having under test in its own right.

    Fits are keyed on what EXECUTED. A requested-graphs row with too few replays
    is not CUDA-graph evidence; folding it into either fit would calibrate the
    engine model against a mixture. Such rows are excluded from BOTH fits and
    listed in `invalid_graph_rows`, so their absence is visible rather than
    silent.
    """
    eager_rows = [r for r in rows if not r["use_cuda_graphs_requested"]]
    graph_rows = [r for r in rows
                  if r["use_cuda_graphs_requested"] and r["cuda_graph_arm_clean"]]
    invalid = [
        {"max_batch_size_forced": r["max_batch_size_forced"], "seed": r["seed"],
         "arrival": r["arrival"], "rate_rps": r["rate_rps"],
         "cuda_graph_hits": r["cuda_graph_hits"],
         "cuda_graph_hit_rate": r["cuda_graph_hit_rate"],
         "cuda_graph_arm_valid": r["cuda_graph_arm_valid"],
         "cuda_graph_fallback_reasons": r.get("cuda_graph_fallback_reasons", {})}
        for r in rows
        if r["use_cuda_graphs_requested"] and not r["cuda_graph_arm_clean"]
    ]
    return {
        "hardware_profile_fit_eager": fit_hardware_profile(eager_rows),
        "hardware_profile_fit_cuda_graph": fit_hardware_profile(graph_rows),
        "n_eager_rows": len(eager_rows),
        "n_clean_graph_rows": len(graph_rows),
        "invalid_graph_rows": invalid,
        # The gate on everything downstream: only a sweep whose every
        # requested-graphs row genuinely replayed graphs may be read as
        # eager-vs-graph evidence, or used to fit the mechanistic
        # HardwareProfile.
        "graph_arm_usable": bool(graph_rows) and not invalid,
        "cap_binding_by_rate": _binding_summary(
            rows, lambda r: f"{r['arrival']}@{r['rate_rps']}"
        ),
        "cap_binding_by_rate_and_batch": _binding_summary(
            rows,
            lambda r: (
                f"{r['arrival']}@{r['rate_rps']}@"
                f"b{r['max_batch_size_forced']}"
            ),
        ),
        "phase16_question": (
            "Compare throughput-vs-realised-batch slope with CUDA graphs off vs "
            "on. If the slope survives, the batch effect is not merely Python "
            "dispatch amortisation. If it flattens, that is a boundary condition "
            "on external validity and must be reported as one."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--rates", default="burst,2.0,8.0",
                    help="'burst' = legacy bulk dump; numbers = Poisson lambda")
    ap.add_argument("--no-cuda-graph-arm", action="store_true")
    # Graph-capture grid size. The runner captures
    # (1..max_batch_size) x (--graph-seq-buckets context buckets) graphs, all
    # sharing one CUDA graph memory pool. At --batches 32 that is 32 x 4 = 128
    # graphs; if capture OOMs on a 16 GB T4, LOWER these rather than letting
    # captures fail silently -- a failed capture becomes an eager fallback,
    # which drops the hit rate and (correctly) invalidates the arm.
    ap.add_argument("--graph-seq-buckets", type=int, default=4,
                    help="context-length buckets per batch size (fewer buckets "
                         "= fewer graphs but more masked padding per step)")
    ap.add_argument("--graph-max-batch", type=int, default=32,
                    help="largest decode batch size to capture a graph for; "
                         "larger batches fall back to eager and are counted")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    _require_cuda()

    import torch
    from transformers import AutoTokenizer

    from src.carl.live import ARRIVAL_BURST, ARRIVAL_POISSON
    from src.engine.device import DEVICE
    from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
    from src.eval.provenance import write_result

    batches = [int(b) for b in args.batches.replace(",", " ").split()]
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    rates = []
    for tok in args.rates.replace(",", " ").split():
        rates.append((ARRIVAL_BURST, 0.0) if tok == "burst"
                     else (ARRIVAL_POISSON, float(tok)))

    graph_arms = [False] if args.no_cuda_graph_arm else [False, True]

    # MODEL + TOKENIZER LOADING.
    #
    # `load_tinyllama_from_hf` returns `(LlamaModel, LlamaConfig)` -- NOT
    # `(model, tokenizer)`. It deliberately avoids `transformers.AutoModelForCausalLM`
    # and uses only huggingface_hub + safetensors, so it never constructs a
    # tokenizer at all. Unpacking its second value as `tokenizer` bound a
    # LlamaConfig to that name, and the first call into `_make_prompt` -- which
    # does `tokenizer("The quick brown fox ...")` -- died with
    # `TypeError: 'LlamaConfig' object is not callable`.
    #
    # The tokenizer is a SEPARATE object, built from transformers. This mirrors
    # exactly what the working callers do (src/carl/live.py:410-412 and
    # scripts/eval/ablation_live.py:582-585): discard the config, build the
    # tokenizer explicitly.
    #
    # dtype matters for calibration, not just memory: every other GPU
    # measurement in this repository is fp16 on a T4. Loading fp32 here would
    # produce step-time constants that are not comparable to any of them, and
    # `fit_hardware_profile()` would calibrate the engine model against the
    # wrong arithmetic. The loader casts on CPU and then moves to DEVICE itself.
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Loading {MODEL_NAME} on {DEVICE} (dtype={dtype}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _config = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    rows: list[dict] = []
    payload = {
        "description": ("PHASE 15+16: forced max_batch_size intervention with "
                        "realised-occupancy and host-overhead instrumentation."),
        "sweep": {"batches": batches, "seeds": seeds, "rates": args.rates,
                  "chunk_size": args.chunk, "requests": args.requests,
                  "cuda_graph_arms": graph_arms},
        # Which CUDA-graph implementation produced this artifact. Absent
        # live_graph.py here means the run CANNOT contain graph execution --
        # the diagnosis of both zero-hit smokes.
        "code_provenance": graph_repair_provenance(),
        "rows": rows,
    }

    for graphs in graph_arms:
        for arrival, rate in rates:
            for b in batches:
                for seed in seeds:
                    r = run_one(model, tokenizer, b, args.chunk, seed,
                                args.requests, arrival, rate, graphs,
                                graph_seq_buckets=args.graph_seq_buckets,
                                graph_max_batch=args.graph_max_batch)
                    rows.append(r)
                    hr = r["cuda_graph_hit_rate"]
                    print(f"  graphs={int(graphs)} {arrival}@{rate} mb={b:<3} "
                          f"seed={seed} -> realised_batch={r['realised_mean_batch']:5.2f} "
                          f"binding={str(r['cap_was_binding']):<5} "
                          f"tps={r['throughput_tps']:7.2f} "
                          f"ttft99={r['ttft_p99_ms']:8.0f} "
                          f"step={r['step_time_mean_ms']:6.2f}ms "
                          f"graph_hits={r['cuda_graph_hits']}/"
                          f"{r['cuda_graph_decode_steps']} "
                          f"({'n/a' if hr is None else f'{hr:.3f}'}) "
                          f"valid={r['cuda_graph_arm_valid']}", flush=True)
                    if graphs and not r["cuda_graph_arm_valid"]:
                        diag = r["cuda_graph_diagnostics"]
                        print(
                            "    !! GRAPH ARM INVALID: zero replays -- this row "
                            "ran EAGER and is not CUDA-graph evidence.\n"
                            f"       decode-forward reasons : {diag['reasons']}\n"
                            f"       runner attached        : "
                            f"{diag['graph_runner_attached']} "
                            f"({diag['graph_runner_class']})\n"
                            f"       runner refusal reasons : "
                            f"{diag['runner_fallback_reasons']}",
                            flush=True)
                    # Write incrementally: the two quarantined result files exist
                    # because a Colab VM died before the artifact was saved.
                    write_result(args.out, payload, "batch_intervention",
                                 script="scripts/eval/repair/batch_intervention.py",
                                 extra={"batches": batches, "seeds": seeds,
                                        "model": MODEL_NAME,
                                        "dtype": str(dtype),
                                        "device": str(DEVICE),
                                        "gpu": torch.cuda.get_device_name(0)})

    # --- analysis ---------------------------------------------------------
    payload["analysis"] = build_analysis(rows)
    write_result(args.out, payload, "batch_intervention",
                 script="scripts/eval/repair/batch_intervention.py",
                 extra={"batches": batches, "seeds": seeds, "model": MODEL_NAME,
                        "dtype": str(dtype), "device": str(DEVICE),
                        "gpu": torch.cuda.get_device_name(0)})
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
