"""
CARL ablation on REAL TinyLlama inference (GPU) -- the measured counterpart to
the simulation in scripts/eval/ablation.py.

It follows src/carl/live.py's inference pattern exactly: real TinyLlama weights,
the real ContinuousBatchScheduler, real prefill/decode forward passes, fp16 on
GPU, speculation pinned off. CARL drives the scheduler's knobs LIVE while requests
are served. Ten configurations are run over a NON-STATIONARY scenario and reported
as mean +/- std over N seeds, with full raw data, an adaptation trace, an
environment capture, a validation-selected Static-Best, and a retrospective
offline DynOracle upper bound.

!!! HONEST SCOPE -- READ BEFORE INTERPRETING THE TABLE !!!
----------------------------------------------------------
This is a SINGLE-MODEL live harness (like live.py). CARL is wired to the
SCHEDULER ONLY, speculation is pinned off (TinyLlama self-spec is below
break-even), there is no router (one model is served), and KV eviction never
triggers at these request sizes. Therefore only the SCHEDULING ablations have a
measurable effect:

  * CARL-NoSched (max_batch_size frozen) and CARL-NoChunk (chunk_size frozen)
    change what the GPU does  -> real, measurable deltas vs CARL-Full.
  * CARL-NoSpec / CARL-NoCache / CARL-NoRouter freeze knobs the live engine does
    not act on here -> they measure ~= CARL-Full (flagged live_effective=false).

So a near-zero contribution for those three is EXPECTED and honest: this table
measures which subsystems move real single-GPU TinyLlama metrics (the scheduler),
and complements the simulation ablation, which can vary all five subsystems. See
docs/eval/README.md.

Run:
  python scripts/eval/ablation_live.py            # N=10, seeds 42..51, 50 requests
  python scripts/eval/ablation_live.py --seeds 42,43,44 --limit 30   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime

# --- path bootstrap so `python scripts/eval/ablation_live.py` finds src/ -----
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from src.carl.bandit import (  # noqa: E402
    LinUCBBandit, PerRegimeBandit, ThompsonSamplingBandit,
)
from src.carl.config import CARLConfig, DEFAULT_CONFIGS, all_arm_sets  # noqa: E402
from src.carl.controller import SLO, CARLController  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker, WorkloadRegime  # noqa: E402
from src.engine.auto_tuner import AutoTuner, TuningConfig  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
from src.engine.profiler import StepProfiler  # noqa: E402
from src.engine.scheduler import ContinuousBatchScheduler  # noqa: E402

# Reuse live.py's prompt/workload/percentile helpers + pool sizing so this is the
# exact same serving setup as the real-inference cell 6c.
from src.carl.live import (  # noqa: E402
    BLOCK_SIZE, NUM_BLOCKS, _build_workload, _percentile,
)

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "ablation")
RESULTS_PATH = os.path.join(DOCS_EVAL, "ablation_live_results.json")
SELECTION_PATH = os.path.join(DOCS_EVAL, "static_best_selection.json")
ENV_PATH = os.path.join(DOCS_EVAL, "environment.json")
TRACE_PATH = os.path.join(RAW_DIR, "carl_full_adaptation_trace.json")

DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
VALIDATION_SEED = 999            # held out: never used for an eval run
SLO_TTFT_MS = 200.0
_SLO = SLO(ttft_ms=SLO_TTFT_MS, tpot_ms=50.0, throughput_ref=50.0)
OBSERVE_INTERVAL = 10            # CARL control cycle cadence (steps)

# The validation search space (all 5 dimensions CARL adapts).
SEARCH_SPACE = {
    "max_batch_size": [4, 8, 16],
    "spec_k": [0, 2, 4],
    "routing_threshold": [0.3, 0.5, 0.7],
    "eviction_threshold": [0.7, 0.8, 0.9],
    "chunk_size": [64, 128, 256, 512],
}
N_LHS_CANDIDATES = 16

# Per-config knob freeze (None = fully adaptive CARL-Full). Spec-exact values.
_FREEZE = {
    "CARL-Full": None,
    "CARL-NoSched": dict(max_batch_size=4),
    "CARL-NoSpec": dict(spec_k=0),
    "CARL-NoCache": dict(eviction_threshold=0.8),
    "CARL-NoRouter": dict(routing_threshold=0.5),
    "CARL-NoChunk": dict(chunk_size=256),
}
# Configs whose frozen knob actually changes live inference in this harness.
_LIVE_EFFECTIVE = {"CARL-Full", "CARL-NoSched", "CARL-NoChunk",
                   "Static-Best", "AutoTuner", "CARL-Thompson", "DynOracle"}

# Run order: DynOracle is LAST (it needs CARL-Full's recorded rewards).
CONFIGS = ["CARL-Full", "CARL-NoSched", "CARL-NoSpec", "CARL-NoCache",
           "CARL-NoRouter", "CARL-NoChunk", "Static-Best", "AutoTuner",
           "CARL-Thompson", "DynOracle"]

_REGIMES = (WorkloadRegime.INTERACTIVE, WorkloadRegime.BATCH)


# ===========================================================================
# Environment capture.
# ===========================================================================


def capture_environment() -> dict:
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
    print(f"Environment: {env['gpu']} | CUDA {env['cuda']} | torch {env['torch']}",
          flush=True)
    return env


# ===========================================================================
# Latin Hypercube Sampling over the discrete search space.
# ===========================================================================


def latin_hypercube(n: int, space: dict, seed: int) -> list:
    """`n` CARLConfigs sampled by LHS over the discrete `space`.

    Stratified one-sample-per-bin in [0,1) per dimension, independently shuffled
    (the LHS property: every 1-D projection is evenly covered), then each [0,1)
    value is quantised onto that dimension's discrete level list. Covers the full
    5-D space with 16 points instead of the 3x3x3x3x4 = 324 exhaustive grid.
    """
    import random
    rng = random.Random(seed)
    strata = {}
    for dim, levels in space.items():
        s = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(s)
        strata[dim] = s

    configs = []
    for i in range(n):
        kw = {}
        for dim, levels in space.items():
            idx = min(len(levels) - 1, int(strata[dim][i] * len(levels)))
            kw[dim] = levels[idx]
        configs.append(CARLConfig(**kw).clamp())
    return configs


# ===========================================================================
# The serving loop (mirrors live.py's _run_config, with richer instrumentation).
# ===========================================================================


def _apply_sched(sched, cfg: CARLConfig) -> None:
    """Push a config's SCHEDULING knobs into the live scheduler (spec stays off)."""
    sched.max_batch_size = int(cfg.max_batch_size)
    sched.chunk_size = int(cfg.chunk_size)
    sched.enable_spec_decode = False


def _new_scheduler(model) -> ContinuousBatchScheduler:
    d = CARLConfig()
    return ContinuousBatchScheduler(
        model, max_batch_size=d.max_batch_size, num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE, chunk_size=d.chunk_size, enable_spec_decode=False,
    )


def _serve(sched, specs, *, controller=None, tracker=None, tuner=None,
           oracle_phase1=None) -> dict:
    """Serve `specs` once; return per-request records, percentiles, overhead.

    controller : a CARLController wired to `sched` (CARL configs). We time each
        maybe_step and re-pin speculation off afterwards.
    tuner      : an AutoTuner (the AutoTuner baseline); observed every step.
    oracle_phase1 : config applied the instant the BATCH phase is injected
        (DynOracle / Oracle perfect-knowledge switch).
    """
    # rid -> static metadata, plus integer id in submission order.
    meta = {}
    for i, spec in enumerate(specs):
        meta[spec.rid] = {
            "request_id": i,
            "prompt_len": int(spec.prompt_ids.shape[-1]),
            "regime": "INTERACTIVE" if spec.phase == 0 else "BATCH",
        }

    phase0 = [s for s in specs if s.phase == 0]
    phase1 = [s for s in specs if s.phase == 1]
    submit_time, first_tok, last_tok, tok_count = {}, {}, {}, {}

    def _submit(spec) -> None:
        submit_time[spec.rid] = time.perf_counter()
        tok_count[spec.rid] = 0
        sched.add_request(spec.rid, spec.prompt_ids, max_new_tokens=spec.max_new,
                          eos_token_id=None)

    t0 = time.perf_counter()
    for spec in phase0:
        _submit(spec)

    records: list[dict] = []
    decision_us: list[float] = []
    # --- per-step throughput log (ADDITIVE: observation only) -----------------
    # One entry per scheduler step so the recovery analysis can build a
    # throughput(t) curve directly, instead of inferring it from the run-level
    # aggregate. step_log/shift_* are written but never READ inside this loop, so
    # they cannot change scheduler or controller behaviour. shift_t / shift_step
    # mark the instant the BATCH phase is injected (the regime shift).
    step_log: list[dict] = []
    shift_t: float | None = None
    shift_step: int | None = None
    total_tokens = 0
    finished_count = 0
    phase1_done = (len(phase1) == 0)
    last_step_t = time.perf_counter()

    while sched.has_work():
        emitted = sched.step()
        now = time.perf_counter()
        dt = now - last_step_t
        last_step_t = now

        # Record this step's wall time (since t0), tokens emitted, active batch
        # size and step index. Pure append -- no control flow depends on it.
        step_log.append({
            "t": now - t0,
            "dt": dt,
            "tokens": len(emitted),
            "active": len(sched.active),
            "step": sched._step_idx,
        })

        for rid, _tok in emitted:
            if rid not in first_tok:
                first_tok[rid] = now
            last_tok[rid] = now
            tok_count[rid] += 1
            total_tokens += 1

        for r in sched.get_finished():
            rid = r.request_id
            finished_count += 1
            n = tok_count[rid]
            ttft_ms = (first_tok[rid] - submit_time[rid]) * 1000.0
            tpot_ms = ((last_tok[rid] - first_tok[rid]) / (n - 1) * 1000.0) if n > 1 else 0.0
            rec = dict(meta[rid], tokens_generated=n, ttft_ms=ttft_ms, tpot_ms=tpot_ms)
            records.append(rec)
            if tracker is not None:
                tracker.record_request(ttft_ms, tpot_ms)

        if tracker is not None:
            tracker.record_batch(len(sched.active))
            if dt > 0 and emitted:
                tracker.record_throughput(len(emitted) / dt)

        if not phase1_done and finished_count >= max(1, len(phase0) // 2):
            for spec in phase1:
                _submit(spec)
            phase1_done = True
            shift_t = now - t0           # wall-clock of the regime shift.
            shift_step = sched._step_idx  # step index of the regime shift.
            if oracle_phase1 is not None:
                _apply_sched(sched, oracle_phase1)

        if controller is not None:
            d0 = time.perf_counter()
            entry = controller.maybe_step(sched._step_idx)
            d1 = time.perf_counter()
            sched.enable_spec_decode = False
            if entry is not None:                 # an actual control decision
                decision_us.append((d1 - d0) * 1e6)
        if tuner is not None:
            tuner.observe(sched, step=sched._step_idx)

    wall = time.perf_counter() - t0
    records.sort(key=lambda r: r["request_id"])
    ttfts = [r["ttft_ms"] for r in records]
    tpots = [r["tpot_ms"] for r in records if r["tokens_generated"] > 1]
    slo_rate = (100.0 * sum(1 for t in ttfts if t < SLO_TTFT_MS) / len(ttfts)
                if ttfts else 0.0)
    overhead_pct = (sum(decision_us) / 1e6 / wall * 100.0) if wall > 0 else 0.0

    return {
        "requests": records,
        "throughput_tps": total_tokens / wall if wall > 0 else 0.0,
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p99": _percentile(ttfts, 99),
        "tpot_p50": _percentile(tpots, 50),
        "tpot_p99": _percentile(tpots, 99),
        "slo_rate": slo_rate,
        "wall_s": wall,
        "decision_us": decision_us,
        "overhead_pct": overhead_pct,
        # Additive per-step throughput trace + regime-shift markers (see above).
        "step_log": step_log,
        "shift_t": shift_t,
        "shift_step": shift_step,
        "n_phase0": len(phase0),
    }


# ===========================================================================
# Building / running one configuration.
# ===========================================================================


def _frozen_arms(freeze: dict | None) -> dict:
    base = all_arm_sets()
    if not freeze:
        return base
    return {r: [replace(a, **freeze).clamp() for a in arms] for r, arms in base.items()}


def _arm_index(arms: list, cfg: CARLConfig) -> int:
    """Index of `cfg` within an arm list (best-effort; -1 if not found)."""
    for i, a in enumerate(arms):
        if a == cfg:
            return i
    return -1


def run_config(name, model, tokenizer, n, seed, *, static_cfg=None,
               oracle_arms=None) -> dict:
    """Serve one configuration once over a fresh NON-STATIONARY workload.

    Returns the _serve dict, augmented for CARL configs with `decisions`
    (list of (regime, arm, reward, config)) extracted from the controller log.
    """
    import random
    specs = _build_workload(tokenizer, "NON-STATIONARY", n, random.Random(seed))
    sched = _new_scheduler(model)
    controller = tracker = tuner = oracle_phase1 = None

    if name == "Static-Best":
        _apply_sched(sched, static_cfg or CARLConfig())
    elif name == "AutoTuner":
        # Real AutoTuner: attach a StepProfiler so the live scheduler feeds it
        # genuine per-phase timings, then tune every OBSERVE_INTERVAL steps.
        sched.profiler = StepProfiler(window=100)
        tuner = AutoTuner(sched.profiler,
                          config=TuningConfig(tune_interval=OBSERVE_INTERVAL, cooldown=20))
    elif name == "DynOracle":
        # Apply the retrospectively-best arm per regime (computed from CARL-Full).
        _apply_sched(sched, oracle_arms[WorkloadRegime.INTERACTIVE])
        oracle_phase1 = oracle_arms[WorkloadRegime.BATCH]
    else:  # CARL-Full or a CARL-NoX ablation, optionally Thompson.
        tracker = MetricsTracker(window=max(50, n))
        if name == "CARL-Thompson":
            bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                                     bandit_cls=ThompsonSamplingBandit, v=0.5, seed=seed)
        else:
            bandit = PerRegimeBandit(_frozen_arms(_FREEZE[name]), d=FEATURE_DIM,
                                     bandit_cls=LinUCBBandit, alpha=0.5)
        controller = CARLController(scheduler=sched, bandit=bandit,
                                    observe_interval=OBSERVE_INTERVAL,
                                    slo=_SLO, metrics=tracker)

    out = _serve(sched, specs, controller=controller, tracker=tracker,
                 tuner=tuner, oracle_phase1=oracle_phase1)

    # Extract per-decision (regime, arm, reward, config) from the controller log
    # for the CARL configs -- feeds the adaptation trace and the DynOracle.
    if controller is not None:
        decisions = []
        for e in controller.controller_log:
            arms = controller.bandit.arms(e.regime)
            decisions.append({
                "regime": e.regime, "arm": _arm_index(arms, e.config),
                "reward": e.reward, "config": e.config,
            })
        out["decisions"] = decisions
    return out


# ===========================================================================
# Static-Best via held-out validation (LHS search).
# ===========================================================================


def select_static_best(model, tokenizer, val_n: int) -> tuple:
    print(f"\n[validation] LHS {N_LHS_CANDIDATES} candidates x {val_n} requests "
          f"(seed {VALIDATION_SEED})", flush=True)
    candidates = latin_hypercube(N_LHS_CANDIDATES, SEARCH_SPACE, VALIDATION_SEED)
    throughputs = []
    for j, cfg in enumerate(candidates):
        m = run_config("Static-Best", model, tokenizer, val_n, VALIDATION_SEED,
                       static_cfg=cfg)
        throughputs.append(m["throughput_tps"])
        print(f"  cand {j+1:2d}/{N_LHS_CANDIDATES}: mb={cfg.max_batch_size:2d} "
              f"cs={cfg.chunk_size:3d} k={cfg.spec_k} -> {m['throughput_tps']:6.1f} tok/s",
              flush=True)
    win = max(range(len(candidates)), key=lambda i: throughputs[i])
    winner = candidates[win]
    selection = {
        "method": f"latin_hypercube_{N_LHS_CANDIDATES}_candidates",
        "search_space": SEARCH_SPACE,
        "candidates": [c.as_dict() for c in candidates],
        "validation_throughputs": throughputs,
        "winner": winner.as_dict(),
        "validation_seed": VALIDATION_SEED,
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"[validation] winner: mb={winner.max_batch_size} cs={winner.chunk_size} "
          f"({throughputs[win]:.1f} tok/s) -> {SELECTION_PATH}", flush=True)
    return winner, selection


# ===========================================================================
# DynOracle: retrospective best arm per regime, from CARL-Full's recorded rewards.
# ===========================================================================


def compute_dynoracle_arms(carl_full_decisions: list) -> tuple:
    """Best arm per regime by mean recorded reward across all CARL-Full runs.

    carl_full_decisions: flattened list of decision dicts from every CARL-Full
    run. Returns ({regime: CARLConfig}, selection_metadata). This uses hindsight
    from completed runs -- it is an offline upper bound, NOT deployable.
    """
    arms = all_arm_sets()
    sums: dict = {r: {} for r in _REGIMES}
    counts: dict = {r: {} for r in _REGIMES}
    for d in carl_full_decisions:
        r, a = d["regime"], d["arm"]
        if r not in sums or a < 0:
            continue
        sums[r][a] = sums[r].get(a, 0.0) + d["reward"]
        counts[r][a] = counts[r].get(a, 0) + 1

    chosen, meta = {}, {}
    for r in _REGIMES:
        means = {a: sums[r][a] / counts[r][a] for a in sums[r]}
        if means:
            best_arm = max(means, key=means.get)
        else:
            best_arm = 0   # fallback: the regime's hand-tuned default (arm 0)
        chosen[r] = arms[r][best_arm]
        meta[r.value] = {
            "best_arm": best_arm,
            "mean_reward": means.get(best_arm, 0.0),
            "config": arms[r][best_arm].as_dict(),
            "per_arm_mean_reward": {str(a): means[a] for a in means},
        }
    return chosen, meta


# ===========================================================================
# Aggregation + reporting.
# ===========================================================================

_METRICS = [
    ("throughput_tps", "tput"),
    ("ttft_p50", "ttftP50"),
    ("ttft_p99", "ttftP99"),
    ("tpot_p50", "tpotP50"),
    ("tpot_p99", "tpotP99"),
    ("slo_rate", "SLO%"),
]


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _save_raw(config: str, seed: int, run: dict) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = {
        "config": config, "seed": seed,
        "requests": run["requests"],
        "throughput_tps": run["throughput_tps"],
        "ttft_p50": run["ttft_p50"], "ttft_p99": run["ttft_p99"],
        "tpot_p50": run["tpot_p50"], "tpot_p99": run["tpot_p99"],
        "slo_rate": run["slo_rate"],
    }
    path = os.path.join(RAW_DIR, f"{config}_run_{seed:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_adaptation_trace(decisions: list) -> None:
    """Compact CARL-Full trace: log only first decision + each arm change."""
    os.makedirs(RAW_DIR, exist_ok=True)
    trace, prev_arm = [], None
    for i, d in enumerate(decisions):
        changed = (prev_arm is None) or (d["arm"] != prev_arm)
        if changed:
            c = d["config"]
            trace.append({
                "request_id": i, "regime": d["regime"].value,
                "selected_arm": d["arm"], "reward": d["reward"],
                "arm_changed": prev_arm is not None,
                "config": {"batch_size": c.max_batch_size, "spec_k": c.spec_k,
                           "routing_threshold": c.routing_threshold,
                           "eviction_threshold": c.eviction_threshold,
                           "chunk_size": c.chunk_size},
            })
        prev_arm = d["arm"]
    with open(TRACE_PATH, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)


def run_all(seeds: list, n: int) -> dict:
    env = capture_environment()
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

    # Validation phase -> Static-Best. Validation uses ~half the eval size.
    static_cfg, selection = select_static_best(model, tokenizer, max(10, n // 2))

    results: dict = {
        "environment": env,
        "scenario": "NON-STATIONARY (1-25 INTERACTIVE prompt16-64/max32, "
                    "26-50 BATCH prompt128-256/max64)",
        "seeds": seeds, "runs": len(seeds), "requests": n,
        "slo_ttft_ms": SLO_TTFT_MS, "validation_seed": VALIDATION_SEED,
        "static_best_selection": selection,
        "live_effective_configs": sorted(_LIVE_EFFECTIVE),
        "scope_note": ("Single-model live harness: CARL wired to the scheduler "
                       "only, speculation pinned off, no router, KV eviction "
                       "inactive. Only NoSched/NoChunk differ from CARL-Full; "
                       "NoSpec/NoCache/NoRouter measure ~= CARL-Full by design. "
                       "See docs/eval/README.md."),
        "configs": {},
    }

    carl_full_decisions: list = []
    carl_full_decision_us: list = []
    carl_full_decision_us_per_run: list = []   # one list per CARL-Full seed
    carl_full_overhead_pct: list = []
    oracle_arms = oracle_meta = None

    for name in CONFIGS:
        if name == "DynOracle":
            # Computed only now, from all CARL-Full decisions gathered above.
            oracle_arms, oracle_meta = compute_dynoracle_arms(carl_full_decisions)

        per_run = []
        for r_idx, seed in enumerate(seeds):
            try:
                run = run_config(
                    name, model, tokenizer, n, seed,
                    static_cfg=static_cfg if name == "Static-Best" else None,
                    oracle_arms=oracle_arms if name == "DynOracle" else None)
                per_run.append(run)
                _save_raw(name, seed, run)
                if name == "CARL-Full":
                    carl_full_decisions.extend(run.get("decisions", []))
                    carl_full_decision_us.extend(run.get("decision_us", []))
                    carl_full_decision_us_per_run.append(list(run.get("decision_us", [])))
                    carl_full_overhead_pct.append(run.get("overhead_pct", 0.0))
                    if r_idx == 0:   # adaptation trace from the first CARL-Full run
                        _save_adaptation_trace(run.get("decisions", []))
                print(f"  {name:<14} {r_idx+1}/{len(seeds)} (seed {seed}): "
                      f"{run['throughput_tps']:6.1f} tok/s, "
                      f"ttftP99={run['ttft_p99']:6.1f}ms", flush=True)
            except Exception:
                print(f"  {name:<14} {r_idx+1}/{len(seeds)} (seed {seed}): FAILED",
                      flush=True)
                traceback.print_exc()
        if not per_run:
            continue

        agg = {"live_effective": name in _LIVE_EFFECTIVE}
        for key, _label in _METRICS:
            mean, std = _mean_std([m[key] for m in per_run])
            agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
        results["configs"][name] = agg

    # CARL overhead (CARL-Full): P99 decision latency (us) + % of inference time.
    #
    # IMPORTANT (see docs/eval/overhead_reconciliation.md): n_decisions here is
    # SMALL -- it is one timing per actual control cycle (every observe_interval
    # scheduler steps), summed across the CARL-Full seeds, so for a 30-request run
    # it is only ~tens of samples. P99 over so few samples is just the single
    # slowest decision, which is dominated by FIRST-CALL WARMUP (the cold numpy
    # linalg path on the first cycle of the process). That makes the raw P99 swing
    # wildly run-to-run and it should NOT be reported as a steady-state tail.
    # We therefore record BOTH the raw P99 and a warmup-excluded P99 (dropping the
    # first decision of each run), keep the cold-start figure separately, and
    # persist the full per-run decision_us so the distribution can be re-analysed.
    if carl_full_decision_us:
        # Per-run lists (first element of each is that run's cold-start decision).
        cold_starts = [d[0] for d in carl_full_decision_us_per_run if d]
        steady = [v for d in carl_full_decision_us_per_run for v in d[1:]]
        results["carl_overhead"] = {
            "p99_decision_latency_us": _percentile(carl_full_decision_us, 99),
            "p99_decision_latency_us_warmup_excluded": _percentile(steady, 99),
            "p50_decision_latency_us_warmup_excluded": _percentile(steady, 50),
            "cold_start_us_per_run": cold_starts,
            "cold_start_p99_us": _percentile(cold_starts, 99),
            "mean_pct_of_inference": _mean_std(carl_full_overhead_pct)[0],
            "n_decisions": len(carl_full_decision_us),
            "n_decisions_per_run": [len(d) for d in carl_full_decision_us_per_run],
            "n_decisions_warmup_excluded": len(steady),
            "decision_us_per_run": carl_full_decision_us_per_run,
            "note": ("p99_decision_latency_us is small-N (==max sample) and "
                     "warmup-dominated; report the warmup_excluded steady-state "
                     "figure as the headline and cold_start separately. See "
                     "docs/eval/overhead_reconciliation.md."),
        }

    if oracle_meta is not None:
        results["dynoracle"] = {
            "label": "DynOracle (retrospective offline UB -- not deployable)",
            "definition": ("Best CARL arm per regime by mean recorded reward "
                           "across all CARL-Full runs, applied statically with a "
                           "boundary switch. Uses hindsight from completed runs."),
            "arms_per_regime": oracle_meta,
            "source": "CARL-Full recorded rewards", "seeds": seeds,
        }

    _finalize(results)
    return results


def _finalize(results: dict) -> None:
    cfgs = results["configs"]

    # Subsystem contributions: delta_X = CARL-Full - CARL-NoX (throughput), ranked.
    full = cfgs.get("CARL-Full", {}).get("throughput_tps_mean", 0.0)
    contrib = {}
    for name in ("CARL-NoSched", "CARL-NoSpec", "CARL-NoCache",
                 "CARL-NoRouter", "CARL-NoChunk"):
        if name in cfgs:
            contrib[name.replace("CARL-No", "")] = full - cfgs[name]["throughput_tps_mean"]
    results["subsystem_contributions"] = dict(
        sorted(contrib.items(), key=lambda kv: kv[1], reverse=True))

    # Oracle gap: (DynOracle - CARL-Full) / DynOracle * 100.
    dyn = cfgs.get("DynOracle", {}).get("throughput_tps_mean", 0.0)
    results["oracle_gap_pct"] = ((dyn - full) / dyn * 100.0) if dyn else None

    # LinUCB vs Thompson.
    th = cfgs.get("CARL-Thompson", {}).get("throughput_tps_mean", 0.0)
    results["linucb_vs_thompson"] = {
        "carl_full_linucb_tput": full, "carl_thompson_tput": th,
        "linucb_minus_thompson": full - th,
    }

    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved live ablation results to {RESULTS_PATH}", flush=True)


def _print(results: dict) -> None:
    cfgs = results["configs"]
    print("\n=== LIVE ABLATION: NON-STATIONARY on real TinyLlama (mean +/- std) ===")
    headers = ["config", "live?", "tput", "ttftP99", "SLO%"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for name in CONFIGS:
        a = cfgs.get(name)
        if a is None:
            continue
        live = "yes" if a.get("live_effective") else "no*"
        print("| " + " | ".join([
            name, live,
            f"{a['throughput_tps_mean']:.1f} +/- {a['throughput_tps_std']:.1f}",
            f"{a['ttft_p99_mean']:.1f} +/- {a['ttft_p99_std']:.1f}",
            f"{a['slo_rate_mean']:.0f}",
        ]) + " |")
    print("\n* 'no' = frozen knob has no live effect in this single-model harness "
          "(spec off / no router / eviction inactive); expected ~= CARL-Full.")

    print("\n=== Subsystem contributions (delta tput = CARL-Full - CARL-NoX, ranked) ===")
    for sub, d in results.get("subsystem_contributions", {}).items():
        print(f"  {sub:<8} {d:+.2f} tok/s")
    if results.get("oracle_gap_pct") is not None:
        print(f"\nOracle gap (DynOracle vs CARL-Full): {results['oracle_gap_pct']:+.1f}%")
    lt = results.get("linucb_vs_thompson", {})
    if lt:
        print(f"LinUCB vs Thompson: {lt['carl_full_linucb_tput']:.1f} vs "
              f"{lt['carl_thompson_tput']:.1f} tok/s "
              f"({lt['linucb_minus_thompson']:+.1f})")
    ov = results.get("carl_overhead")
    if ov:
        print(f"CARL overhead: raw P99 {ov['p99_decision_latency_us']:.1f} us "
              f"(small-N==max, warmup-dominated) over {ov['n_decisions']} decisions "
              f"{ov.get('n_decisions_per_run')}")
        if "p99_decision_latency_us_warmup_excluded" in ov:
            print(f"  steady-state (warmup-excluded): P50 "
                  f"{ov['p50_decision_latency_us_warmup_excluded']:.1f} us / P99 "
                  f"{ov['p99_decision_latency_us_warmup_excluded']:.1f} us "
                  f"over {ov['n_decisions_warmup_excluded']} decisions; "
                  f"cold-start/run: {[round(c,1) for c in ov.get('cold_start_us_per_run',[])]} us")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL ablation on real TinyLlama inference (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42..51)")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else 50
    run_all(seeds, n)


if __name__ == "__main__":
    main()
