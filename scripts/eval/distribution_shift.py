"""
Workload Distribution Shift Evaluation for CARL (GPU) -- a NEW, first-class
experiment, independent of every existing eval script.

SCIENTIFIC QUESTION
-------------------
Can CARL adapt to workload distributions it was NOT tuned for, WITHOUT retraining
or manual reconfiguration?

CARL's per-regime bandits start from a hand-tuned warm start (DEFAULT_CONFIGS, arm
0 per regime) and learn online. The ablation suite already shows it tracks a
single planted INTERACTIVE->BATCH shift. This experiment stresses the harder
claim: take a controller that has been learning on ONE workload, then yank the
distribution out from under it -- a different source/target pair, no reset, no
re-tuning -- and ask whether the SAME online controller recovers the throughput a
statically-best config (tuned offline for the TARGET) would get.

PROTOCOL (per transition, per seed)
-----------------------------------
One continuous run with no intervention at the boundary:

  Phase 1 (requests   0..49): warm up on the SOURCE workload.
  -- SHIFT at request 50: the distribution is swapped, NOTHING is reset --
  Phase 2 (requests 50..99): the TARGET workload, served by the SAME controller.

The controller (and the AutoTuner baseline) keep learning straight through the
shift; the bandit's A/b statistics, the controller log, and the applied config
all carry across. The only thing that changes at request 50 is which workload is
feeding the scheduler. See `_serve` for the exact mechanism and the
CONFIRMATION block at the bottom of this docstring.

TRANSITIONS
-----------
  1. INTERACTIVE  -> BATCH
  2. BATCH        -> INTERACTIVE
  3. INTERACTIVE  -> MIXED
  4. LONG_CONTEXT -> INTERACTIVE

BASELINES (mirror the existing evals' roles)
--------------------------------------------
  * Static-Best : the best STATIC config for the TARGET workload, found offline by
                  Latin-Hypercube search on a held-out validation seed (the same
                  methodology ablation_live.py uses), then frozen for the whole
                  run. It cannot adapt -- it is the "tuned for the target" oracle
                  the recovery percentage is measured against.
  * AutoTuner   : the existing hill-climber (src/engine/auto_tuner.py), adapting
                  online across the shift just like CARL.
  * CARL-Full   : the online per-regime LinUCB controller, NO reset at the shift.

METRICS (per seed; aggregated mean +/- std over seeds)
------------------------------------------------------
Per window (whole run AND the post-shift window = the Phase-2 requests):
  throughput_tps, ttft_p99_ms, ttft_p50_ms, tpot_p50_ms, slo_rate.
CARL-only adaptation metrics, measured from the controller's decision log:
  cumulative_regret (vs the static best-arm-per-regime oracle),
  time_to_adaptation (control cycles from the shift to the last arm switch),
  arm_switches (post-shift count), final_arm_selected, stabilized.
Cross-baseline:
  oracle_recovery_pct = 100 * post_shift_throughput / Static-Best post_shift
                        throughput  (how much of the tuned-static target
                        throughput an ADAPTIVE method recovers after the shift).

Per control CYCLE we also log throughput/ttft_p99/arm/cumulative_regret so the
convergence figure can be plotted (per-seed CSVs, see OUTPUTS).

OUTPUTS
-------
  docs/eval/distribution_shift_results.json
      aggregate schema mirroring the other docs/eval/*_results.json files.
  docs/eval/raw/distribution_shift/<SOURCE>_to_<TARGET>_seed<NNN>.csv
      per-CARL-cycle rows (convergence-figure + root-cause source):
        cycle, phase, throughput_tps, ttft_p99_ms, arm_selected,
        cumulative_regret, regime, reward, is_arm_switch, is_exploitative,
        queue_depth, avg_prompt_len, gpu_utilization, cache_hit_rate
      ttft_p99_ms is the rolling-window per-request TTFT p99 at that cycle; the
      four raw features are denormalized from the logged context; is_exploitative
      is reconstructed offline (greedy exploit-argmax == selected arm). CARL-Full
      only -- Static-Best/AutoTuner have no per-cycle controller log.

SCOPE / HONESTY (same single-model live caveats as ablation_live.py)
--------------------------------------------------------------------
This is a single-TinyLlama live harness: CARL is wired to the SCHEDULER only and
speculation is pinned OFF (TinyLlama self-spec is below break-even -- see
src/carl/live.py). So the live levers that actually move metrics are
max_batch_size and chunk_size. The contribution under test here is therefore
purely "can the SCHEDULING policy re-adapt across a distribution shift", which is
exactly the honest, measurable slice on one GPU. The Static-Best search varies the
full 5-knob space for methodological parity with ablation_live, but only the two
scheduling knobs change live inference.

REUSE (and what is deliberately NOT imported)
---------------------------------------------
Imports only from `src/` (the engine + CARL library), never from any sibling eval
script. In particular it does NOT import from or modify ablation_live.py; the
small pieces it needs in common (an LHS sampler, the best-arm-per-regime oracle)
are re-implemented locally so this file stands alone. It reuses the pure,
torch-free analysis in src/carl/adaptation.py for regret/convergence.

Run:
  python scripts/eval/distribution_shift.py                       # seeds 42,43,44, 50+50
  python scripts/eval/distribution_shift.py --limit 30            # shorter phases (quick)
  python scripts/eval/distribution_shift.py --transitions "INTERACTIVE->BATCH"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime

# --- path bootstrap so `python scripts/eval/distribution_shift.py` finds src/ ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.carl.adaptation import decision_rows, summarize  # noqa: E402
from src.carl.bandit import LinUCBBandit, PerRegimeBandit  # noqa: E402
from src.carl.config import CARLConfig, all_arm_sets  # noqa: E402
from src.carl.controller import SLO, CARLController  # noqa: E402
# _FEATURE_SCALES: the (name -> characteristic scale) map state.to_feature_vector
# divides by; we multiply back to log RAW engine units (queue depth, prompt len,
# GPU util, cache hit rate) for the per-cycle CSV. Read-only use -- nothing here
# changes how features are computed.
from src.carl.state import FEATURE_DIM, MetricsTracker, _FEATURE_SCALES  # noqa: E402
from src.engine.auto_tuner import AutoTuner, TuningConfig  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
from src.engine.profiler import StepProfiler  # noqa: E402
from src.engine.scheduler import ContinuousBatchScheduler  # noqa: E402

# Reuse live.py's prompt builder / percentile / request spec / KV-pool sizing, so
# the serving setup is byte-identical to the real-inference path. (src import is
# allowed; ablation_live is NOT imported.)
from src.carl.live import (  # noqa: E402
    BLOCK_SIZE, NUM_BLOCKS, _ReqSpec, _make_prompt, _percentile,
)

# ===========================================================================
# Constants -- SLO + cadence pinned to ablation_live.py for cross-experiment
# comparability (the task requires the same SLO config).
# ===========================================================================

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "distribution_shift")
RESULTS_PATH = os.path.join(DOCS_EVAL, "distribution_shift_results.json")
ENV_PATH = os.path.join(DOCS_EVAL, "environment.json")

DEFAULT_SEEDS = [42, 43, 44]      # same seeds as the existing evals
VALIDATION_SEED = 999             # held out for Static-Best selection (never an eval seed)

SLO_TTFT_MS = 200.0
_SLO = SLO(ttft_ms=SLO_TTFT_MS, tpot_ms=50.0, throughput_ref=50.0)  # == ablation_live
OBSERVE_INTERVAL = 10             # CARL control cycle cadence (scheduler steps)

PHASE_REQUESTS = 50               # Phase 1 length == Phase 2 length (task spec)

# The four transitions, as (source, target) workload names.
TRANSITIONS = [
    ("INTERACTIVE", "BATCH"),
    ("BATCH", "INTERACTIVE"),
    ("INTERACTIVE", "MIXED"),
    ("LONG_CONTEXT", "INTERACTIVE"),
]

BASELINES = ["Static-Best", "AutoTuner", "CARL-Full"]

# ---------------------------------------------------------------------------
# Workload definitions.
#
# Each workload is BOTH a prompt-length window AND an arrival policy (`outstanding`
# = the max in-flight+queued requests we keep submitted). The arrival policy is
# what actually makes classify_regime (src/carl/state.py) read the INTENDED
# regime off the live scheduler:
#
#   INTERACTIVE  -- short prompts + a SHALLOW queue (outstanding=4): queue_depth
#                   stays < 8 and avg_prompt_len is tiny, so classify_regime falls
#                   through to INTERACTIVE.
#   BATCH        -- prompts >= 256 tokens with a steady queue (outstanding=16):
#                   avg_prompt_len >= 256 trips the BATCH rule directly, and the
#                   queue is kept below the BURST threshold (24) so it reads as a
#                   steady throughput regime, not a surge.
#   LONG_CONTEXT -- prompts > 512 tokens: classify_regime's first rule
#                   (avg_prompt_len > 512) wins regardless of the queue, so this is
#                   robustly LONG_CONTEXT. Shallow arrival keeps KV memory bounded.
#   MIXED        -- each request is drawn from one of the three windows above, so
#                   the classified regime genuinely oscillates request-to-request:
#                   the "distribution it was never tuned for" as a single coherent
#                   regime.
#
# Memory check (KV pool = NUM_BLOCKS * BLOCK_SIZE = 1024*16 = 16384 tokens):
# worst case is LONG_CONTEXT, ~768 prompt + 48 gen ~= 816 tokens ~= 51 blocks;
# outstanding=4 -> ~204 blocks. BATCH ~448 tokens * 16 -> ~448 blocks. Both well
# under capacity, so this fits a Colab T4.
# ---------------------------------------------------------------------------

WORKLOADS = {
    "INTERACTIVE":  {"prompt_lo": 16,  "prompt_hi": 32,  "max_new": 32, "outstanding": 4},
    "BATCH":        {"prompt_lo": 256, "prompt_hi": 384, "max_new": 64, "outstanding": 16},
    "LONG_CONTEXT": {"prompt_lo": 576, "prompt_hi": 768, "max_new": 48, "outstanding": 4},
    # MIXED draws per-request from the members below; it has no single window.
    "MIXED":        {"max_new": 48, "outstanding": 8},
}
MIXED_MEMBERS = ["INTERACTIVE", "BATCH", "LONG_CONTEXT"]

# The Static-Best validation search space (full 5 knobs, == ablation_live.py's
# SEARCH_SPACE for methodological parity; only batch/chunk move live inference).
SEARCH_SPACE = {
    "max_batch_size": [4, 8, 16],
    "spec_k": [0, 2, 4],
    "routing_threshold": [0.3, 0.5, 0.7],
    "eviction_threshold": [0.7, 0.8, 0.9],
    "chunk_size": [64, 128, 256, 512],
}
N_LHS_CANDIDATES = 8              # lighter than ablation_live's 16 to keep T4 runtime sane
VALIDATION_REQUESTS = 24         # single-phase target requests per candidate


# ===========================================================================
# Environment capture (same shape as ablation_live's environment.json).
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
# Latin Hypercube Sampling (re-implemented locally; NOT imported from any eval).
# ===========================================================================


def latin_hypercube(n: int, space: dict, seed: int) -> list:
    """`n` CARLConfigs sampled by LHS over the discrete `space`.

    Stratified one-sample-per-bin per dimension, independently shuffled, then each
    [0,1) draw is quantised onto that dimension's level list. Same construction as
    the existing evals so Static-Best is selected the same way.
    """
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
# Workload construction.
# ===========================================================================


def _build_phase(tokenizer, name: str, n: int, rng, phase: int) -> list:
    """Build `n` _ReqSpec for workload `name`, all tagged with submission `phase`.

    phase 0 == source (Phase 1), phase 1 == target (Phase 2). For MIXED each
    request independently draws one of the member windows so the regime varies.
    """
    specs = []
    if name == "MIXED":
        for i in range(n):
            member = WORKLOADS[rng.choice(MIXED_MEMBERS)]
            L = rng.randint(member["prompt_lo"], member["prompt_hi"])
            specs.append(_ReqSpec(f"mx{phase}_{i}", _make_prompt(tokenizer, L),
                                  member["max_new"], phase))
        return specs

    w = WORKLOADS[name]
    tag = name[:3].lower()
    for i in range(n):
        L = rng.randint(w["prompt_lo"], w["prompt_hi"])
        specs.append(_ReqSpec(f"{tag}{phase}_{i}", _make_prompt(tokenizer, L),
                              w["max_new"], phase))
    return specs


def _cap(name: str) -> int:
    """The arrival cap (max outstanding requests) for a workload."""
    return WORKLOADS[name]["outstanding"]


# ===========================================================================
# Live scheduler helpers (mirror live.py / ablation_live; spec stays OFF).
# ===========================================================================


def _new_scheduler(model) -> ContinuousBatchScheduler:
    d = CARLConfig()
    return ContinuousBatchScheduler(
        model, max_batch_size=d.max_batch_size, num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE, chunk_size=d.chunk_size, enable_spec_decode=False,
    )


def _apply_sched(sched, cfg: CARLConfig) -> None:
    """Push a config's SCHEDULING knobs into the live scheduler (spec stays off)."""
    sched.max_batch_size = int(cfg.max_batch_size)
    sched.chunk_size = int(cfg.chunk_size)
    sched.enable_spec_decode = False


# ===========================================================================
# THE SERVING LOOP -- this is where the distribution shift happens.
# ===========================================================================
#
# Submission is arrival-rate controlled: we keep at most `cap` requests
# outstanding for the CURRENT phase, drawing from that phase's spec list as
# earlier requests finish. Phase 0 (source) is served to COMPLETE DRAIN; only
# then -- with the bandit/controller untouched -- do we begin submitting Phase 1
# (target). That drain point is THE SHIFT: a single, unambiguous boundary at
# "request 50" with no reset of any controller state.
# ===========================================================================


def _serve(sched, source_specs, target_specs, *, src_cap, tgt_cap,
           controller=None, tracker=None, tuner=None) -> dict:
    """Serve source then target through one scheduler, shifting at the drain point.

    Returns raw run data: per-request records (each tagged with its phase), total
    tokens + wall time, the shift bookkeeping (step index, wall clock, token count
    at the shift), and per-control-cycle metrics for CARL.
    """
    # rid -> phase, so each finished request is attributed to the right window.
    meta = {}
    for ph, specs in ((0, source_specs), (1, target_specs)):
        for s in specs:
            meta[s.rid] = ph

    submit_time, first_tok, last_tok, tok_count = {}, {}, {}, {}

    def _submit(spec) -> None:
        submit_time[spec.rid] = time.perf_counter()
        tok_count[spec.rid] = 0
        sched.add_request(spec.rid, spec.prompt_ids, max_new_tokens=spec.max_new,
                          eos_token_id=None)   # exactly max_new tokens (fixed budget)

    # phases[i] = (cap, remaining-to-submit list) for phase i.
    phases = [(src_cap, list(source_specs)), (tgt_cap, list(target_specs))]
    submitted = [0, 0]
    counts = [len(source_specs), len(target_specs)]
    phase_idx = 0

    shift_step = None       # scheduler step at which the target was injected
    shift_wall = None       # wall clock at the shift (for post-shift throughput)
    tokens_at_shift = None  # total_tokens at the shift (post tokens = end - this)

    records = []            # per-request: phase, ttft_ms, tpot_ms, tokens_generated
    cycle_metrics = []      # per CARL control cycle: step, throughput_tps, ttft_p99_ms
    recent_ttft = deque(maxlen=64)   # rolling TTFT window for the per-cycle p99
    total_tokens = 0

    def outstanding() -> int:
        return len(sched.active) + len(sched.waiting)

    t0 = time.perf_counter()
    last_step_t = t0

    while True:
        # 1. Top up the CURRENT phase to its arrival cap.
        cap, q = phases[phase_idx]
        while q and outstanding() < cap:
            _submit(q.pop(0))
            submitted[phase_idx] += 1

        # 2. Nothing in flight -> the current phase has fully drained.
        if not sched.has_work():
            if phase_idx == 0:
                # *** THE SHIFT ***  source fully served; inject target WITHOUT
                # resetting the controller / bandit / applied config.
                phase_idx = 1
                if target_specs:
                    shift_step = sched._step_idx
                    shift_wall = time.perf_counter()
                    tokens_at_shift = total_tokens
                continue          # loop back to submit the target phase
            break                 # target drained (or none) -> done

        # 3. One real engine step.
        emitted = sched.step()
        now = time.perf_counter()
        dt = now - last_step_t
        last_step_t = now

        for rid, _tok in emitted:
            if rid not in first_tok:
                first_tok[rid] = now
            last_tok[rid] = now
            tok_count[rid] += 1
            total_tokens += 1

        for r in sched.get_finished():
            rid = r.request_id
            n = tok_count[rid]
            ttft_ms = (first_tok[rid] - submit_time[rid]) * 1000.0
            tpot_ms = ((last_tok[rid] - first_tok[rid]) / (n - 1) * 1000.0) if n > 1 else 0.0
            records.append({"phase": meta[rid], "ttft_ms": ttft_ms,
                            "tpot_ms": tpot_ms, "tokens_generated": n})
            recent_ttft.append(ttft_ms)
            if tracker is not None:
                tracker.record_request(ttft_ms, tpot_ms)

        if tracker is not None:
            tracker.record_batch(len(sched.active))
            if dt > 0 and emitted:
                tracker.record_throughput(len(emitted) / dt)

        # 4. Controllers run EVERY step and persist across the shift (no reset).
        if controller is not None:
            entry = controller.maybe_step(sched._step_idx)
            sched.enable_spec_decode = False           # re-pin spec off after a cycle
            if entry is not None:                      # an actual control decision
                cycle_metrics.append({
                    "step": entry.step,
                    "throughput_tps": tracker.throughput_tps() if tracker else 0.0,
                    "ttft_p99_ms": _percentile(list(recent_ttft), 99),
                })
        if tuner is not None:
            tuner.observe(sched, step=sched._step_idx)

    end_time = time.perf_counter()
    return {
        "records": records,
        "total_tokens": total_tokens,
        "wall": end_time - t0,
        "end_time": end_time,
        "shift_step": shift_step,
        "shift_wall": shift_wall,
        "tokens_at_shift": tokens_at_shift,
        "cycle_metrics": cycle_metrics,
    }


# ===========================================================================
# Per-window metric aggregation over per-request records.
# ===========================================================================

_WINDOW_KEYS = ["throughput_tps", "ttft_p99_ms", "ttft_p50_ms", "tpot_p50_ms", "slo_rate"]


def _latency_metrics(recs: list) -> dict:
    """TTFT/TPOT percentiles + SLO rate over a set of per-request records."""
    ttfts = [r["ttft_ms"] for r in recs]
    tpots = [r["tpot_ms"] for r in recs if r["tokens_generated"] > 1]
    slo_rate = (100.0 * sum(1 for t in ttfts if t < SLO_TTFT_MS) / len(ttfts)
                if ttfts else 0.0)
    return {
        "ttft_p50_ms": _percentile(ttfts, 50),
        "ttft_p99_ms": _percentile(ttfts, 99),
        "tpot_p50_ms": _percentile(tpots, 50),
        "slo_rate": slo_rate,
    }


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


# ===========================================================================
# Running ONE (baseline, seed) configuration.
# ===========================================================================


def run_one(name, model, tokenizer, source, target, seed, n_phase, *,
            static_cfg=None) -> dict:
    """Serve one (baseline, seed) over source->target; return windowed metrics.

    `target` may be None (single-phase, used for Static-Best validation).
    """
    rng = random.Random(seed)               # one rng -> both phases deterministic from seed
    src = _build_phase(tokenizer, source, n_phase, rng, 0)
    tgt = _build_phase(tokenizer, target, n_phase, rng, 1) if target else []

    sched = _new_scheduler(model)
    controller = tracker = tuner = None

    if name == "Static-Best":
        _apply_sched(sched, static_cfg or CARLConfig())   # frozen; never adapts
    elif name == "AutoTuner":
        sched.profiler = StepProfiler(window=100)
        tuner = AutoTuner(sched.profiler,
                          config=TuningConfig(tune_interval=OBSERVE_INTERVAL, cooldown=20))
    else:  # CARL-Full -- the online per-regime LinUCB controller.
        tracker = MetricsTracker(window=max(50, 2 * n_phase))
        bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
        controller = CARLController(scheduler=sched, bandit=bandit,
                                    observe_interval=OBSERVE_INTERVAL,
                                    slo=_SLO, metrics=tracker)

    out = _serve(sched, src, tgt, src_cap=_cap(source),
                 tgt_cap=_cap(target) if target else 0,
                 controller=controller, tracker=tracker, tuner=tuner)

    recs = out["records"]
    overall = _latency_metrics(recs)
    overall["throughput_tps"] = out["total_tokens"] / out["wall"] if out["wall"] > 0 else 0.0

    res = {
        "overall": overall,
        "post_shift": None,
        "shift_step": out["shift_step"],
        "controller_log": controller.controller_log if controller is not None else None,
        "cycle_metrics": out["cycle_metrics"] if controller is not None else None,
    }

    # Post-shift window: the Phase-2 (target) requests + their wall-clock slice.
    if target and out["shift_wall"] is not None:
        post_recs = [r for r in recs if r["phase"] == 1]
        post = _latency_metrics(post_recs)
        post_tokens = out["total_tokens"] - out["tokens_at_shift"]
        post_wall = out["end_time"] - out["shift_wall"]
        post["throughput_tps"] = post_tokens / post_wall if post_wall > 0 else 0.0
        res["post_shift"] = post
    return res


# ===========================================================================
# Static-Best selection: LHS over the TARGET workload (held-out seed).
# ===========================================================================


def select_static_best(model, tokenizer, target: str) -> tuple:
    """Pick the highest-throughput static config for `target` by LHS validation."""
    print(f"  [static-best:{target}] LHS {N_LHS_CANDIDATES} candidates x "
          f"{VALIDATION_REQUESTS} requests (seed {VALIDATION_SEED})", flush=True)
    candidates = latin_hypercube(N_LHS_CANDIDATES, SEARCH_SPACE, VALIDATION_SEED)
    throughputs = []
    for j, cfg in enumerate(candidates):
        # Single-phase serve of just the target workload (target=None below).
        m = run_one("Static-Best", model, tokenizer, target, None, VALIDATION_SEED,
                    VALIDATION_REQUESTS, static_cfg=cfg)
        throughputs.append(m["overall"]["throughput_tps"])
        print(f"    cand {j + 1:2d}/{N_LHS_CANDIDATES}: mb={cfg.max_batch_size:2d} "
              f"cs={cfg.chunk_size:3d} -> {m['overall']['throughput_tps']:6.1f} tok/s",
              flush=True)
    win = max(range(len(candidates)), key=lambda i: throughputs[i])
    winner = candidates[win]
    selection = {
        "target": target,
        "method": f"latin_hypercube_{N_LHS_CANDIDATES}_candidates",
        "validation_seed": VALIDATION_SEED,
        "validation_requests": VALIDATION_REQUESTS,
        "search_space": SEARCH_SPACE,
        "winner": winner.as_dict(),
        "winner_throughput_tps": throughputs[win],
    }
    print(f"  [static-best:{target}] winner mb={winner.max_batch_size} "
          f"cs={winner.chunk_size} ({throughputs[win]:.1f} tok/s)", flush=True)
    return winner, selection


# ===========================================================================
# CARL adaptation analysis (regret / convergence) -- reuses src/carl/adaptation.
# ===========================================================================


class _ArmsView:
    """Minimal `.arms(regime)` adapter over the static per-regime arm sets.

    decision_rows only needs to map a logged config back to its arm index, which
    depends solely on the (static) arm lists CARL-Full chose among.
    """

    def __init__(self) -> None:
        self._arms = all_arm_sets()

    def arms(self, regime):
        return self._arms[regime]


def _arm_index(arms: list, cfg: CARLConfig) -> int:
    for i, a in enumerate(arms):
        if a == cfg:
            return i
    return -1


def compute_oracle(pooled: list) -> dict:
    """Static best-arm-per-regime oracle: {regime_value -> best arm's mean reward}.

    `pooled` is a list of (regime_enum, arm_index, reward) across ALL CARL-Full
    seeds for one transition. This is the same offline upper bound the ablation's
    DynOracle uses, re-implemented here so nothing is imported from ablation_live.
    """
    sums, counts = {}, {}
    for regime, arm, reward in pooled:
        if arm < 0:
            continue
        rv = regime.value
        sums.setdefault(rv, {})
        counts.setdefault(rv, {})
        sums[rv][arm] = sums[rv].get(arm, 0.0) + reward
        counts[rv][arm] = counts[rv].get(arm, 0) + 1
    oracle = {}
    for rv in sums:
        means = {a: sums[rv][a] / counts[rv][a] for a in sums[rv]}
        best = max(means, key=means.get)
        oracle[rv] = means[best]
    return oracle


def _csv_path(source: str, target: str, seed: int) -> str:
    return os.path.join(RAW_DIR, f"{source}_to_{target}_seed{seed:03d}.csv")


# Feature order (the index of each name in a logged state_features vector).
_FEAT_IDX = {name: i for i, name in enumerate(_FEATURE_SCALES.keys())}


def _raw_feature(state_features, name: str):
    """Denormalize one logged context feature back to RAW engine units.

    state.to_feature_vector() stores feature/scale; we multiply by the scale to
    recover the raw value (queue_depth in requests, avg_prompt_len in tokens,
    gpu_utilization/cache_hit_rate as fractions). Returns "" if the feature is
    absent (e.g. an override cycle logged no context).
    """
    i = _FEAT_IDX.get(name)
    if i is None or not state_features or i >= len(state_features):
        return ""
    return state_features[i] * _FEATURE_SCALES[name]


def _reconstruct_exploitative(log: list, arms_view) -> list:
    """Per-cycle is_exploitative for a CARL run, reconstructed offline.

    LOGGING-ONLY post-hoc analysis -- it does NOT touch the controller, the
    reward, or the live bandit. It replays the controller's recorded decisions
    into a SHADOW PerRegimeBandit (identical class + arm sets; alpha is irrelevant
    because only the exploit term theta^T x is read). Since the live bandit's ONLY
    updates are the controller's delayed (prev_arm, reward, prev_context) folds
    (see controller.py step 3), replaying them in order reproduces the EXACT A/b
    the live bandit held at each selection -- so this is a faithful reconstruction,
    not an approximation.

      is_exploitative[t] = (argmax_a theta_a^T x_t == the arm actually selected)

    True  -> the greedy (exploit-only) pick matched the live selection.
    False -> the UCB exploration bonus changed the pick: an EXPLORATORY decision.
    "" (blank) when the cycle can't be labelled (override arm or missing context).
    """
    shadow = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                             bandit_cls=LinUCBBandit, alpha=0.5)
    flags = []
    prev = None   # (regime, arm, context) of the previous cycle
    for e in log:
        regime = e.regime
        arm = _arm_index(arms_view.arms(regime), e.config)
        x = np.asarray(e.state_features, dtype=np.float64).reshape(-1)
        bandit = shadow.bandits[regime]
        # Label THIS cycle against the shadow's current statistics, which reflect
        # exactly the updates the live bandit had applied before this selection.
        if arm < 0 or x.size != bandit.d:
            flags.append("")
        else:
            exploit = [float(bandit.theta(a) @ x) for a in range(bandit.n_arms)]
            greedy = max(range(bandit.n_arms), key=lambda a: exploit[a])
            flags.append(greedy == arm)
        # Then mirror the controller's delayed update: credit the PREVIOUS arm
        # with THIS cycle's reward and the previous context (affects cycle t+1).
        if prev is not None:
            pr, pa, pctx = prev
            if pa >= 0 and pctx.size == shadow.bandits[pr].d:
                shadow.update(pr, pa, e.reward, pctx)
        prev = (regime, arm, x)
    return flags


def _fmt(v) -> str:
    """4-decimal float, pass-through for blanks/bools/ints already stringified."""
    return f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


# Per-CARL-cycle CSV schema (the convergence-figure + root-cause source).
_CYCLE_CSV_COLUMNS = [
    "cycle", "phase", "throughput_tps", "ttft_p99_ms", "arm_selected",
    "cumulative_regret", "regime", "reward", "is_arm_switch", "is_exploitative",
    "queue_depth", "avg_prompt_len", "gpu_utilization", "cache_hit_rate",
]


def _write_cycle_csv(path: str, rows: list, cycle_metrics: list, log: list,
                     exploit_flags: list, shift_step) -> None:
    """Write the per-CARL-cycle CSV (schema = _CYCLE_CSV_COLUMNS).

    `rows` (decision_rows), `cycle_metrics`, `log` (the controller log) and
    `exploit_flags` are all one-per-control-cycle in the same order, so they align
    by index. `ttft_p99_ms` is the rolling-window per-request TTFT p99 captured at
    each cycle (true per-request series is not persisted -- the "else keep rolling
    window" case). regime/reward/is_arm_switch come straight from `rows`; the four
    raw features are denormalized from the logged context; is_exploitative is the
    offline reconstruction above.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CYCLE_CSV_COLUMNS)
        for i, (row, cm, entry, expl) in enumerate(
                zip(rows, cycle_metrics, log, exploit_flags)):
            phase = 1 if (shift_step is not None and row["step"] >= shift_step) else 0
            sf = entry.state_features or []
            w.writerow([
                i, phase, f"{cm['throughput_tps']:.4f}", f"{cm['ttft_p99_ms']:.4f}",
                row["arm"], f"{row['cumulative_regret']:.6f}",
                row["regime"], f"{row['reward']:.6f}", row["is_arm_switch"], expl,
                _fmt(_raw_feature(sf, "queue_depth")),
                _fmt(_raw_feature(sf, "avg_prompt_len")),
                _fmt(_raw_feature(sf, "gpu_utilization")),
                _fmt(_raw_feature(sf, "cache_hit_rate")),
            ])


def analyze_carl(source, target, carl_runs, seeds) -> dict:
    """Per-seed regret/convergence + per-seed convergence CSVs; aggregate digest.

    Returns the aggregate CARL adaptation metrics to merge into the JSON, having
    written one convergence CSV per seed as a side effect.
    """
    arms_view = _ArmsView()

    # Pool every seed's (regime, arm, reward) to build ONE oracle for the transition.
    pooled = []
    for r in carl_runs:
        for e in r["controller_log"]:
            a = _arm_index(arms_view.arms(e.regime), e.config)
            pooled.append((e.regime, a, e.reward))
    oracle = compute_oracle(pooled)

    total_regret, post_regret = [], []
    t2a, post_switches, finals, stabilized = [], [], [], []

    for r, seed in zip(carl_runs, seeds):
        log = r["controller_log"]
        rows = decision_rows(log, arms_view, oracle)   # per-cycle regret + markers
        _ = summarize(rows)                             # (validates row structure)
        shift_step = r["shift_step"]

        # Index of the first post-shift control cycle.
        sc = next((i for i, row in enumerate(rows)
                   if shift_step is not None and row["step"] >= shift_step), len(rows))
        post = rows[sc:]

        # Arm switches strictly AFTER the shift, and the cycle of the LAST one.
        sw_idx = [i for i, row in enumerate(post) if row["is_arm_switch"]]
        post_switches.append(len(sw_idx))
        # time_to_adaptation: control cycles from the shift to the last arm switch
        # (0 == CARL was already on the arm it keeps). stabilized == it stopped
        # switching before the run ended.
        tta = sw_idx[-1] if sw_idx else 0
        t2a.append(tta)
        stabilized.append((not sw_idx) or (sw_idx[-1] < len(post) - 1))

        if post:
            finals.append({"regime": post[-1]["regime"], "arm": post[-1]["arm"]})
            post_regret.append(sum(row["instant_regret"] for row in post))
        else:
            finals.append(None)
            post_regret.append(0.0)
        total_regret.append(rows[-1]["cumulative_regret"] if rows else 0.0)

        exploit_flags = _reconstruct_exploitative(log, arms_view)
        _write_cycle_csv(_csv_path(source, target, seed), rows, r["cycle_metrics"],
                         log, exploit_flags, shift_step)

    # Final-arm mode across seeds (the policy CARL converges to post-shift).
    final_labels = [f"{f['regime']}#{f['arm']}" for f in finals if f]
    final_mode = Counter(final_labels).most_common(1)[0][0] if final_labels else None

    tr_m, tr_s = _mean_std(total_regret)
    pr_m, pr_s = _mean_std(post_regret)
    tta_m, tta_s = _mean_std([float(x) for x in t2a])
    sw_m, sw_s = _mean_std([float(x) for x in post_switches])

    return {
        "oracle_reward_by_regime": oracle,
        "cumulative_regret_mean": tr_m, "cumulative_regret_std": tr_s,
        "post_shift_cumulative_regret_mean": pr_m, "post_shift_cumulative_regret_std": pr_s,
        "time_to_adaptation_cycles_mean": tta_m, "time_to_adaptation_cycles_std": tta_s,
        "arm_switches_post_shift_mean": sw_m, "arm_switches_post_shift_std": sw_s,
        "final_arm_selected_mode": final_mode,
        "final_arm_selected_per_seed": finals,
        "time_to_adaptation_cycles_per_seed": t2a,
        "arm_switches_post_shift_per_seed": post_switches,
        "stabilized_per_seed": stabilized,
        "stabilized_all_seeds": all(stabilized) if stabilized else False,
    }


# ===========================================================================
# Aggregation across seeds for one baseline.
# ===========================================================================


def aggregate(runs: list) -> dict:
    """Mean +/- std across seeds for the overall and post-shift windows."""
    out = {"overall": {}, "post_shift": {}}
    for window in ("overall", "post_shift"):
        for k in _WINDOW_KEYS:
            vals = [r[window][k] for r in runs if r.get(window)]
            m, s = _mean_std(vals)
            out[window][f"{k}_mean"] = m
            out[window][f"{k}_std"] = s
    return out


# ===========================================================================
# Driver.
# ===========================================================================


def run_all(seeds: list, n_phase: int, transitions: list) -> dict:
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {len(transitions)} transitions x "
          f"{len(BASELINES)} baselines x {len(seeds)} seeds | "
          f"{n_phase}+{n_phase} requests | seeds {seeds}", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA device -- CPU smoke test only; run on a Colab T4.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    os.makedirs(RAW_DIR, exist_ok=True)
    results = {
        "experiment": "workload_distribution_shift",
        "scientific_question": ("Can CARL adapt to workload distributions it was "
                                "not tuned for, without retraining or manual "
                                "reconfiguration?"),
        "protocol": ("Continuous run; Phase 1 warms up on the source workload, the "
                     "distribution is swapped at the drain point (request "
                     f"{n_phase}) with NO reset and NO retuning, Phase 2 serves the "
                     "target while the same controller keeps learning online."),
        "environment": env,
        "seeds": seeds,
        "phase1_requests": n_phase,
        "phase2_requests": n_phase,
        "shift_at_request": n_phase,
        "slo": {"ttft_ms": SLO_TTFT_MS, "tpot_ms": 50.0, "throughput_ref": 50.0},
        "observe_interval": OBSERVE_INTERVAL,
        "baselines": BASELINES,
        "scope_note": ("Single-model live harness (TinyLlama): CARL wired to the "
                       "scheduler only, speculation pinned off. Live levers are "
                       "max_batch_size/chunk_size. See ablation_live.py and "
                       "docs/eval/README.md."),
        "regret_model": ("static best-arm-per-regime oracle (mean recorded reward, "
                         "pooled over CARL-Full seeds per transition); "
                         "instant_regret = max(0, oracle - reward)."),
        "transitions": {},
    }

    static_best_cache: dict = {}   # target name -> (CARLConfig, selection-dict)

    for source, target in transitions:
        key = f"{source}->{target}"
        print(f"\n=== Transition {key} ===", flush=True)

        # Static-Best for the TARGET (cached: INTERACTIVE recurs as a target).
        if target not in static_best_cache:
            static_best_cache[target] = select_static_best(model, tokenizer, target)
        static_cfg, selection = static_best_cache[target]

        baselines_out = {}
        carl_runs = []
        for name in BASELINES:
            runs = []
            for seed in seeds:
                try:
                    r = run_one(name, model, tokenizer, source, target, seed, n_phase,
                                static_cfg=static_cfg if name == "Static-Best" else None)
                    runs.append(r)
                    post = r.get("post_shift") or {}
                    print(f"  {name:<12} seed {seed}: overall "
                          f"{r['overall']['throughput_tps']:6.1f} tok/s | post-shift "
                          f"{post.get('throughput_tps', 0.0):6.1f} tok/s, "
                          f"ttftP99={post.get('ttft_p99_ms', 0.0):6.1f}ms", flush=True)
                except Exception:
                    print(f"  {name:<12} seed {seed}: FAILED", flush=True)
                    traceback.print_exc()
            if not runs:
                continue
            baselines_out[name] = aggregate(runs)
            if name == "CARL-Full":
                carl_runs = runs

        # CARL adaptation analysis (+ per-seed convergence CSVs).
        if carl_runs:
            baselines_out["CARL-Full"].update(analyze_carl(source, target, carl_runs, seeds))

        # oracle_recovery_pct: how much of Static-Best's tuned post-shift throughput
        # each ADAPTIVE method recovers after the shift.
        sb_post = baselines_out.get("Static-Best", {}).get("post_shift", {}).get(
            "throughput_tps_mean", 0.0)
        for nm in ("CARL-Full", "AutoTuner"):
            if nm in baselines_out:
                cp = baselines_out[nm]["post_shift"]["throughput_tps_mean"]
                baselines_out[nm]["oracle_recovery_pct"] = (
                    100.0 * cp / sb_post if sb_post > 0 else None)

        results["transitions"][key] = {
            "source": source,
            "target": target,
            "static_best_config": static_cfg.as_dict(),
            "static_best_selection": selection,
            "baselines": baselines_out,
        }

    _save(results)
    _print(results)
    return results


def _save(results: dict) -> None:
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved distribution-shift results to {RESULTS_PATH}", flush=True)
    print(f"Per-seed convergence CSVs in {RAW_DIR}", flush=True)


def _print(results: dict) -> None:
    print("\n=== WORKLOAD DISTRIBUTION SHIFT (post-shift window, mean +/- std) ===")
    headers = ["transition", "baseline", "tput", "ttftP99", "SLO%", "recover%", "t2a", "regret"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for key, tr in results["transitions"].items():
        for name in BASELINES:
            a = tr["baselines"].get(name)
            if a is None:
                continue
            ps = a["post_shift"]
            rec = a.get("oracle_recovery_pct")
            rec_s = f"{rec:.0f}" if rec is not None else "-"
            t2a = a.get("time_to_adaptation_cycles_mean")
            t2a_s = f"{t2a:.1f}" if t2a is not None else "-"
            reg = a.get("cumulative_regret_mean")
            reg_s = f"{reg:.2f}" if reg is not None else "-"
            print("| " + " | ".join([
                key, name,
                f"{ps['throughput_tps_mean']:.1f}+/-{ps['throughput_tps_std']:.1f}",
                f"{ps['ttft_p99_ms_mean']:.0f}",
                f"{ps['slo_rate_mean']:.0f}",
                rec_s, t2a_s, reg_s,
            ]) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL workload distribution shift evaluation (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=PHASE_REQUESTS,
                        help="requests per phase (default 50; total run = 2x this)")
    # nargs="*" so transitions can be passed space-separated
    # (--transitions "INTERACTIVE->BATCH" "BATCH->INTERACTIVE"); each token is
    # also split on commas, so comma-separated (or a mix of both) still works.
    parser.add_argument("--transitions", nargs="*", default=None,
                        help='optional subset, space- and/or comma-separated, e.g. '
                             '--transitions "INTERACTIVE->BATCH" "BATCH->INTERACTIVE" '
                             'or --transitions "INTERACTIVE->BATCH,BATCH->INTERACTIVE"')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_phase = args.limit if 0 < args.limit <= 200 else PHASE_REQUESTS
    if args.transitions:
        chosen = []
        # Flatten the space-separated values, then split each on commas, so any
        # combination of the two delimiters parses to the same list of pairs.
        for raw in args.transitions:
            for tok in raw.split(","):
                tok = tok.strip()
                if "->" in tok:
                    s, t = tok.split("->", 1)
                    chosen.append((s.strip(), t.strip()))
        transitions = chosen or TRANSITIONS
    else:
        transitions = TRANSITIONS
    run_all(seeds, n_phase, transitions)


if __name__ == "__main__":
    main()
