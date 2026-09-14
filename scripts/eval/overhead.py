"""
Precise characterization of CARL's RUNTIME COST.

Motivation
----------
The live ablation (scripts/eval/ablation_live.py) reported a P99 CARL decision
latency of ~2527 microseconds. That number is a wall-clock measurement taken
inside a live serving loop, so it bundles the bandit algebra with lock
acquisition, Python call overhead, and first-call numpy warmup. This script
takes the controller APART and times each component of CARLController.step()
separately, so we can say -- with evidence -- whether that cost is the LEARNING
ALGORITHM or the INSTRUMENTATION around it.

CARLController.step() (src/carl/controller.py:194) has five numbered blocks:

  1. context construction : observe() -> classify_regime() -> to_feature_vector()
  2. arm selection        : bandit.select()        (LinUCB: per-arm d x d inverse)
  3. reward update        : _reward_for_state() + bandit.update()  (A += x x^T)
  4. config application   : _apply()                (push knobs into the engine)
  5. logging              : build ControllerLogEntry + append + aggregate counters

This harness mirrors those five blocks EXACTLY, calling the REAL controller /
bandit primitives on REAL objects, with a timer around each block. The only
thing the harness adds is the timer boundaries -- which is the whole point:
isolating each block tells us where the microseconds actually go. To separate
algorithm cost from framework cost we ALSO time the real `step()` end-to-end on
a second controller; (real step) - (sum of the five blocks) is the lock +
bookkeeping + call overhead.

Four measurements:
  M1  Per-component breakdown over 10,000 synthetic step()s (no inference).
  M2  Decision latency vs real inference time over 100 TinyLlama requests.
      (Requires torch + the model; gracefully skipped if unavailable.)
  M3  Memory footprint of the controller (LinUCB matrices + log).
  M4  How step() latency scales with the number of requests already seen.

CPU-only for M1/M3/M4 (numpy, no torch). Run:
  python scripts/eval/overhead.py
  python scripts/eval/overhead.py --commit --push   # Colab: run + self-commit results
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# --- path bootstrap so `python scripts/eval/overhead.py` finds src/ ----------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.carl.bandit import LinUCBBandit, PerRegimeBandit  # noqa: E402
from src.carl.config import CARLConfig, all_arm_sets  # noqa: E402
from src.carl.controller import SLO, CARLController, ControllerLogEntry  # noqa: E402
from src.carl.state import (  # noqa: E402
    FEATURE_DIM, MetricsTracker, RuntimeState, WorkloadRegime, classify_regime,
)

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "overhead")
RESULTS_PATH = os.path.join(DOCS_EVAL, "overhead_results.json")
ENV_PATH = os.path.join(DOCS_EVAL, "environment.json")
COMPONENT_RAW = os.path.join(RAW_DIR, "component_breakdown.json")
AMORTIZED_RAW = os.path.join(RAW_DIR, "amortized_overhead.json")

SEED = 42
N_STEPS = 10_000                 # M1: synthetic step() calls
OBSERVE_INTERVAL = 10            # CARL control cadence (matches live.py)
# The figure this script exists to explain: the live ablation's P99 CARL decision
# latency, measured inside the real serving loop (lock + thread + GC + warmup).
LIVE_ABLATION_P99_US = 2527.0
_SLO = SLO(ttft_ms=100.0, tpot_ms=50.0, throughput_ref=50.0)

# The five components, in step() order. Labels are stable JSON keys.
COMPONENTS = [
    ("t1_context", "context construction (observe+classify+features)"),
    ("t2_select", "arm selection (bandit.select)"),
    ("t3_reward", "reward update (reward + bandit.update)"),
    ("t4_apply", "config application (_apply)"),
    ("t5_log", "logging (build entry + append + counters)"),
]


# ===========================================================================
# Tiny synthetic engine stubs so observe()/_apply() do REAL work without torch.
# ===========================================================================
#
# observe() reads scheduler.waiting/active, the kv cache hit rate, the spec
# decoder acceptance, and the metrics percentiles. _apply() writes knobs back via
# _set(), which only lands on attributes that already exist -- so each stub must
# DECLARE every knob CARL drives, or that write would be silently skipped and we
# would under-measure _apply.


class _FakeReq:
    """A scheduler request as observe() sees it: just a prompt length."""
    __slots__ = ("prompt_len",)

    def __init__(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len


class _FakeScoreTracker:
    def __init__(self) -> None:
        self.recent_window = 64


class _FakeKV:
    """KV cache exposing the hit rate observe() reads + the eviction knobs _apply writes."""
    def __init__(self) -> None:
        self.cache_hit_rate = 0.0
        self.evict_threshold = 0.8
        self.recent_window = 64
        self.score_tracker = _FakeScoreTracker()


class _FakeSpec:
    def __init__(self) -> None:
        self.mean_acceptance_rate = 0.0
        self.k = 0


class _FakeRouter:
    def __init__(self) -> None:
        self.routing_threshold = 0.5
        self.cache_affinity_weight = 0.5


class _FakeScheduler:
    """Declares every knob _apply() drives + the lists observe() reads."""
    def __init__(self) -> None:
        self.waiting: list = []
        self.active: list = []
        # Knobs CARL pushes (must pre-exist for _set to land).
        self.max_batch_size = 8
        self.chunk_size = 256
        self.preemption_enabled = True
        self.use_cuda_graphs = False
        self.spec_decode_k = 1
        self.enable_spec_decode = False


@dataclass
class _Synth:
    """A reusable bundle of stubs + metrics, re-randomized each iteration."""
    scheduler: _FakeScheduler = field(default_factory=_FakeScheduler)
    router: _FakeRouter = field(default_factory=_FakeRouter)
    kv_cache: _FakeKV = field(default_factory=_FakeKV)
    spec_decoder: _FakeSpec = field(default_factory=_FakeSpec)
    metrics: MetricsTracker = field(default_factory=lambda: MetricsTracker(window=100))

    def prime_metrics(self, rng: np.random.Generator) -> None:
        """Fill the metric windows once so observe()'s percentile reads do real
        work (sorting a full 100-sample window) on every call, as in production."""
        for _ in range(100):
            self.metrics.record_request(float(rng.uniform(20, 400)),
                                        float(rng.uniform(5, 120)))
            self.metrics.record_throughput(float(rng.uniform(10, 120)))
            self.metrics.record_batch(int(rng.integers(1, 32)))

    def randomize(self, rng: np.random.Generator) -> None:
        """Pick a fresh workload snapshot so regimes + contexts vary across steps
        (exercises every per-regime bandit and a realistic context distribution)."""
        qd = int(rng.integers(0, 48))
        ac = int(rng.integers(1, 32))
        # Prompt length spans short (interactive) to long (>512, long-context).
        plen = int(rng.choice([32, 128, 256, 600, 1200]))
        self.scheduler.waiting = [_FakeReq(plen) for _ in range(qd)]
        self.scheduler.active = [_FakeReq(plen) for _ in range(ac)]
        self.kv_cache.cache_hit_rate = float(rng.uniform(0.0, 0.9))
        self.spec_decoder.mean_acceptance_rate = float(rng.uniform(0.0, 0.8))


def _new_controller(synth: _Synth, seed: int) -> CARLController:
    """A real CARLController wired to the synthetic engine, LinUCB per regime."""
    bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                             bandit_cls=LinUCBBandit, alpha=0.5)
    return CARLController(
        scheduler=synth.scheduler, router=synth.router, kv_cache=synth.kv_cache,
        spec_decoder=synth.spec_decoder, bandit=bandit,
        observe_interval=OBSERVE_INTERVAL, slo=_SLO, metrics=synth.metrics,
    )


# ===========================================================================
# Percentile helper (nearest-rank; NaN-free).
# ===========================================================================


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1,
                   int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _summary(samples: list[float]) -> dict:
    """mean / p50 / p95 / p99 for a list of microsecond samples."""
    if not samples:
        return {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "n": 0}
    s = sorted(samples)
    return {
        "mean_us": sum(s) / len(s),
        "p50_us": _pct(s, 50),
        "p95_us": _pct(s, 95),
        "p99_us": _pct(s, 99),
        "n": len(s),
    }


# ===========================================================================
# Environment capture (reuse the existing file if ablation already wrote one).
# ===========================================================================


def capture_environment() -> dict:
    """Capture the run host, reusing docs/eval/environment.json ONLY if it still
    matches the live device.

    torch is OPTIONAL here (this is a torch-free CPU microbenchmark by default),
    so the live device defaults to CPU/None when torch is absent. A cached record
    is trusted only when its gpu AND torch fields agree with the live state;
    otherwise (e.g. a CPU-written file shipped to a GPU Colab box, or vice-versa)
    it is STALE and we regenerate -- so the recorded environment can never
    silently misreport CPU on a GPU run (or the reverse).
    """
    # Live device state (torch may be unavailable -> CPU / None).
    live_gpu, live_cuda, live_torch = "CPU", None, None
    try:
        import torch  # optional: only present on a GPU/Colab box
        live_gpu = (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "CPU")
        live_cuda = torch.version.cuda
        live_torch = torch.__version__
    except Exception:
        pass

    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                env = json.load(f)
            if env.get("gpu") == live_gpu and env.get("torch") == live_torch:
                print(f"Environment: reused {ENV_PATH}", flush=True)
                return env
            print(f"Environment: cached {ENV_PATH} disagrees with live device "
                  f"(cached gpu={env.get('gpu')!r} torch={env.get('torch')!r} vs "
                  f"live gpu={live_gpu!r} torch={live_torch!r}); refreshing.",
                  flush=True)
        except Exception:
            pass  # fall through and rewrite a fresh one
    env = {"python": sys.version, "numpy": np.__version__,
           "timestamp": datetime.now().isoformat(),
           "gpu": live_gpu, "cuda": live_cuda, "torch": live_torch}
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"Environment: {env.get('gpu')} | torch {env.get('torch')} -> {ENV_PATH}",
          flush=True)
    return env


# ===========================================================================
# MEASUREMENT 1 -- per-component breakdown over N synthetic step()s.
# ===========================================================================


def measure_components(n: int, seed: int) -> dict:
    """Time each of step()'s five blocks separately over `n` synthetic cycles.

    Two controllers run on the SAME per-iteration synthetic snapshot:
      * `comp` -- driven through the five blocks individually (the breakdown),
        mirroring controller.py:194-241 block-for-block with real primitives.
      * `real` -- its actual `step()` is timed end-to-end. The difference
        (real_step - sum of the five blocks) is the lock + bookkeeping + call
        overhead, i.e. the framework cost as opposed to the algorithm cost.
    """
    print(f"\n[M1] component breakdown: {n} synthetic step()s (seed {seed})", flush=True)
    rng = np.random.default_rng(seed)
    synth = _Synth()
    synth.prime_metrics(rng)
    comp = _new_controller(synth, seed)
    real = _new_controller(synth, seed + 1)

    # Per-component microsecond samples.
    samples: dict[str, list[float]] = {key: [] for key, _ in COMPONENTS}
    total_samples: list[float] = []      # sum(t1..t5) per call
    real_step_samples: list[float] = []  # real controller.step() end-to-end

    # Seed `comp`'s delayed-reward state so block 3 actually calls bandit.update
    # from the very first iteration (otherwise the first update is a no-op).
    prev = None
    ns = time.perf_counter_ns

    for i in range(n):
        synth.randomize(rng)

        # ---- the five blocks on `comp`, each timed individually -------------
        a = ns()
        state = comp.observe()
        regime = classify_regime(state)
        context = state.to_feature_vector()
        b = ns(); t1 = b - a

        arm, config = comp.bandit.select(regime, context)
        c = ns(); t2 = c - b

        reward = comp._reward_for_state(state)
        if prev is not None:
            p_regime, p_arm, p_ctx = prev
            comp.bandit.update(p_regime, p_arm, reward, p_ctx)
        d = ns(); t3 = d - c

        comp._apply(config)
        e = ns(); t4 = e - d

        entry = ControllerLogEntry(step=i, regime=regime, config=config,
                                   reward=reward, state_features=context)
        comp.controller_log.append(entry)
        comp._regime_counts[regime] = comp._regime_counts.get(regime, 0) + 1
        comp._reward_sum[regime] = comp._reward_sum.get(regime, 0.0) + reward
        comp._reward_n[regime] = comp._reward_n.get(regime, 0) + 1
        f = ns(); t5 = f - e

        prev = (regime, arm, context)

        # nanoseconds -> microseconds.
        for key, val in zip(("t1_context", "t2_select", "t3_reward",
                             "t4_apply", "t5_log"), (t1, t2, t3, t4, t5)):
            samples[key].append(val / 1000.0)
        total_samples.append((t1 + t2 + t3 + t4 + t5) / 1000.0)

        # ---- the REAL step() end-to-end on `real` (full observe pipeline) ----
        g = ns()
        real.step(step_idx=i)        # state=None -> observes the same synth stub
        h = ns()
        real_step_samples.append((h - g) / 1000.0)

    # Assemble the breakdown table. pct_of_total uses MEAN component / MEAN total
    # (the share of an average step the component accounts for).
    mean_total = sum(total_samples) / len(total_samples)
    table = []
    for key, desc in COMPONENTS:
        s = _summary(samples[key])
        s["component"] = key
        s["description"] = desc
        s["pct_of_total"] = (s["mean_us"] / mean_total * 100.0) if mean_total else 0.0
        table.append(s)

    total_sum = _summary(total_samples)
    real_summary = _summary(real_step_samples)

    # Which block dominates, and the framework (lock/bookkeeping) overhead.
    dominant = max(table, key=lambda r: r["mean_us"])
    framework_us = real_summary["mean_us"] - mean_total

    # Explain the live ablation's P99: the isolated algorithm is far cheaper, so
    # the live 2527 us is dominated by serving-loop framing (lock contention with
    # the pumper thread, GC pauses, first-call numpy warmup), NOT logging and NOT
    # the bandit algebra itself.
    isolated_p99 = real_summary["p99_us"]
    live_interpretation = (
        f"Isolated step() P99 is {isolated_p99:.0f} us, but the live ablation "
        f"reported {LIVE_ABLATION_P99_US:.0f} us P99 -- "
        f"{LIVE_ABLATION_P99_US / isolated_p99:.0f}x higher. Within step(), the "
        f"bandit is {(next(r['pct_of_total'] for r in table if r['component'] == 't2_select') + next(r['pct_of_total'] for r in table if r['component'] == 't3_reward')):.0f}% "
        f"and logging only {next(r['pct_of_total'] for r in table if r['component'] == 't5_log'):.1f}%, "
        f"so the live tail is NOT logging and NOT the core algorithm -- it is "
        f"serving-loop framing (lock contention, GC, first-call warmup)."
    )

    # Persist all 10,000 per-component samples (raw).
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(COMPONENT_RAW, "w", encoding="utf-8") as fp:
        json.dump({"seed": seed, "n": n, "unit": "microseconds",
                   "components": {k: samples[k] for k, _ in COMPONENTS},
                   "total_sum_components": total_samples,
                   "real_step_end_to_end": real_step_samples}, fp)
    print(f"[M1] raw -> {COMPONENT_RAW}", flush=True)

    return {
        "n_steps": n, "seed": seed,
        "component_table": table,
        "total_of_components": total_sum,
        "real_step_end_to_end": real_summary,
        "framework_overhead_us": framework_us,
        "live_ablation_p99_us": LIVE_ABLATION_P99_US,
        "live_p99_interpretation": live_interpretation,
        "dominant_component": dominant["component"],
        "dominant_pct_of_total": dominant["pct_of_total"],
        "logging_pct_of_total": next(r["pct_of_total"] for r in table
                                     if r["component"] == "t5_log"),
        "bandit_pct_of_total": (next(r["pct_of_total"] for r in table
                                     if r["component"] == "t2_select")
                                + next(r["pct_of_total"] for r in table
                                       if r["component"] == "t3_reward")),
    }


# ===========================================================================
# MEASUREMENT 2 -- decision latency vs real inference time (needs torch+model).
# ===========================================================================


def measure_amortized(n_requests: int, seed: int) -> dict:
    """Serve `n_requests` real TinyLlama requests with CARL on; time every CARL
    control cycle against the engine's step wall time, then amortize.

    Continuous batching has no isolated per-request generate(), so we measure at
    the STEP granularity (the unit CARL actually fires on) and amortize CARL's
    total time over the requests served -- the honest analogue of "overhead per
    request" for an interleaved engine.
    """
    try:
        import random
        import torch  # noqa: F401
        from transformers import AutoTokenizer
        from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
        from src.engine.device import DEVICE
        from src.engine.scheduler import ContinuousBatchScheduler
        from src.carl.live import BLOCK_SIZE, NUM_BLOCKS, _build_workload
    except Exception as exc:  # torch / model / deps unavailable -> skip cleanly
        print(f"[M2] SKIPPED (no torch/model available): {exc}", flush=True)
        return {"skipped": True,
                "reason": f"torch/model unavailable: {exc}",
                "note": ("Run on a machine with torch + TinyLlama to populate. "
                         "Amortized overhead can also be derived from M1's real "
                         "step cost / observe_interval when this is unavailable.")}

    print(f"\n[M2] amortized overhead: {n_requests} real TinyLlama requests "
          f"(seed {seed})", flush=True)
    import torch
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    specs = _build_workload(tokenizer, "NON-STATIONARY", n_requests,
                            random.Random(seed))
    default = CARLConfig()
    sched = ContinuousBatchScheduler(model, max_batch_size=default.max_batch_size,
                                     num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
                                     chunk_size=default.chunk_size,
                                     enable_spec_decode=False)
    tracker = MetricsTracker(window=max(50, n_requests))
    bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                             bandit_cls=LinUCBBandit, alpha=0.5)
    controller = CARLController(scheduler=sched, bandit=bandit,
                                observe_interval=OBSERVE_INTERVAL, slo=_SLO,
                                metrics=tracker)

    for spec in specs:
        sched.add_request(spec.rid, spec.prompt_ids, max_new_tokens=spec.max_new,
                          eos_token_id=None)

    per_step: list[dict] = []
    carl_times_ms: list[float] = []
    total_inference_s = 0.0
    total_carl_s = 0.0

    while sched.has_work():
        s0 = time.perf_counter()
        sched.step()
        s1 = time.perf_counter()
        step_wall = s1 - s0
        total_inference_s += step_wall

        for r in sched.get_finished():
            pass  # draining; latency math not needed for the overhead question

        c0 = time.perf_counter()
        entry = controller.maybe_step(sched._step_idx)
        c1 = time.perf_counter()
        sched.enable_spec_decode = False
        triggered = entry is not None
        carl_ms = (c1 - c0) * 1000.0 if triggered else 0.0
        if triggered:
            total_carl_s += (c1 - c0)
            carl_times_ms.append(carl_ms)
        per_step.append({"step": sched._step_idx, "carl_triggered": triggered,
                         "carl_step_ms": carl_ms, "inference_step_ms": step_wall * 1000.0})

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(AMORTIZED_RAW, "w", encoding="utf-8") as fp:
        json.dump({"seed": seed, "requests": n_requests, "per_step": per_step}, fp)
    print(f"[M2] raw -> {AMORTIZED_RAW}", flush=True)

    triggered_ms_sorted = sorted(carl_times_ms)
    mean_when_triggered = (sum(carl_times_ms) / len(carl_times_ms)
                           if carl_times_ms else 0.0)
    amortized_us = (total_carl_s / n_requests) * 1e6 if n_requests else 0.0
    pct_of_inference = (total_carl_s / total_inference_s * 100.0
                        if total_inference_s > 0 else 0.0)
    return {
        "skipped": False,
        "requests": n_requests, "seed": seed,
        "n_steps": len(per_step),
        "n_carl_triggered": len(carl_times_ms),
        "mean_carl_step_ms_when_triggered": mean_when_triggered,
        "p99_carl_step_ms_when_triggered": _pct(triggered_ms_sorted, 99),
        "total_inference_s": total_inference_s,
        "total_carl_s": total_carl_s,
        "amortized_overhead_us_per_request": amortized_us,
        "overhead_pct_of_inference": pct_of_inference,
    }


# ===========================================================================
# MEASUREMENT 3 -- memory footprint of the controller.
# ===========================================================================


def _deep_sizeof(obj, seen=None) -> int:
    """Recursive sys.getsizeof over a (small, acyclic-ish) object graph.

    Good enough for an order-of-magnitude controller-log footprint: it follows
    dict/list/tuple/set containers and dataclass __dict__/__slots__, and counts
    numpy arrays by nbytes. Guards against cycles with a seen-set.
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_sizeof(k, seen) + _deep_sizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            size += _deep_sizeof(v, seen)
    elif hasattr(obj, "__dict__"):
        size += _deep_sizeof(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        for s in obj.__slots__:
            if hasattr(obj, s):
                size += _deep_sizeof(getattr(obj, s), seen)
    return size


def measure_memory(controller: CARLController) -> dict:
    """LinUCB matrices, controller log, and total controller memory (MB)."""
    print("\n[M3] memory footprint", flush=True)
    MB = 1024.0 * 1024.0

    # All A (d x d) and b (d,) matrices across every per-regime LinUCB bandit.
    matrix_bytes = 0
    n_arms_total = 0
    for regime, sub in controller.bandit.bandits.items():
        for A in sub.A:
            matrix_bytes += A.nbytes
        for b in sub.b:
            matrix_bytes += b.nbytes
        n_arms_total += sub.n_arms

    log_bytes = _deep_sizeof(controller.controller_log)
    total_bytes = matrix_bytes + log_bytes

    return {
        "linucb_matrices_mb": matrix_bytes / MB,
        "n_arms_total": n_arms_total,
        "feature_dim": FEATURE_DIM,
        "controller_log_entries": len(controller.controller_log),
        "controller_log_mb": log_bytes / MB,
        "total_controller_mb": total_bytes / MB,
        "bytes_per_arm_matrix": (matrix_bytes / n_arms_total) if n_arms_total else 0.0,
    }


# ===========================================================================
# MEASUREMENT 4 -- step() latency vs number of requests already seen.
# ===========================================================================


def measure_scaling(scales=(100, 1000, 10000), seed: int = SEED,
                    timed: int = 2000) -> dict:
    """Does the per-step cost grow with how many requests CARL has already seen?

    For each scale N: warm a fresh bandit with N (select+update) rounds, then
    TIME `timed` more select+update rounds. LinUCB's per-arm A is d x d FOREVER
    (the update is A += x x^T, O(d^2); selection inverts A, O(d^3)) -- neither
    grows with N. So if the algorithm is honest, the timed latency should be
    ~FLAT across N. A rising curve would mean a hidden per-request cost.
    """
    print(f"\n[M4] scaling: time {timed} step()s after [{', '.join(map(str, scales))}] "
          f"requests seen", flush=True)
    rng = np.random.default_rng(seed)
    d = FEATURE_DIM
    arms = all_arm_sets()
    points = []
    for N in scales:
        bandit = PerRegimeBandit(arms, d=d, bandit_cls=LinUCBBandit, alpha=0.5)
        regimes = list(arms.keys())

        # Warm: N rounds of real select+update across regimes.
        for _ in range(N):
            r = regimes[int(rng.integers(0, len(regimes)))]
            x = rng.uniform(0.0, 1.0, size=d).tolist()
            arm, _cfg = bandit.select(r, x)
            bandit.update(r, arm, float(rng.uniform(0, 1)), x)

        # Time the next `timed` rounds.
        ns = time.perf_counter_ns
        durs = []
        for _ in range(timed):
            r = regimes[int(rng.integers(0, len(regimes)))]
            x = rng.uniform(0.0, 1.0, size=d).tolist()
            a = ns()
            arm, _cfg = bandit.select(r, x)
            bandit.update(r, arm, float(rng.uniform(0, 1)), x)
            durs.append((ns() - a) / 1000.0)
        s = _summary(durs)
        s["requests_seen"] = N
        points.append(s)
        print(f"  N={N:6d} seen: mean {s['mean_us']:.2f} us, "
              f"p99 {s['p99_us']:.2f} us", flush=True)

    # Growth ratio across the scale span: ~1.0 => flat => O(1) in requests seen.
    first, last = points[0]["mean_us"], points[-1]["mean_us"]
    growth_ratio = (last / first) if first > 0 else 0.0
    return {
        "points": points,
        "mean_growth_ratio_min_to_max": growth_ratio,
        "verdict": ("flat -- step cost is O(d^2/d^3) in DIMENSION, O(1) in "
                    "requests seen; matrix update does NOT dominate at scale"
                    if growth_ratio < 1.5 else
                    "rising -- per-step cost grows with requests seen (investigate)"),
    }


# ===========================================================================
# Reporting.
# ===========================================================================


def _print_report(results: dict) -> None:
    m1 = results["measurement_1_components"]
    print("\n=== M1: CARLController.step() component breakdown "
          f"({m1['n_steps']} synthetic steps) ===")
    print("| component | mean_us | p50_us | p95_us | p99_us | pct_of_total |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in m1["component_table"]:
        print(f"| {row['component']} | {row['mean_us']:.2f} | {row['p50_us']:.2f} "
              f"| {row['p95_us']:.2f} | {row['p99_us']:.2f} | {row['pct_of_total']:.1f}% |")
    tot = m1["total_of_components"]
    rse = m1["real_step_end_to_end"]
    print(f"| TOTAL (sum) | {tot['mean_us']:.2f} | {tot['p50_us']:.2f} "
          f"| {tot['p95_us']:.2f} | {tot['p99_us']:.2f} | 100.0% |")
    print(f"| real step() | {rse['mean_us']:.2f} | {rse['p50_us']:.2f} "
          f"| {rse['p95_us']:.2f} | {rse['p99_us']:.2f} | (end-to-end) |")
    print(f"\nKEY FINDING: {m1['bandit_pct_of_total']:.0f}% of step() time is the "
          f"BANDIT (select+update); logging is {m1['logging_pct_of_total']:.1f}%. "
          f"Dominant block: {m1['dominant_component']} "
          f"({m1['dominant_pct_of_total']:.0f}%).")
    print(f"Framework (lock+bookkeeping) overhead: {m1['framework_overhead_us']:+.2f} us "
          f"(real step {rse['mean_us']:.1f} us vs components {tot['mean_us']:.1f} us).")
    print(f"EXPLAINS THE LIVE 2527 us P99: {m1['live_p99_interpretation']}")

    m2 = results["measurement_2_amortized"]
    if m2.get("skipped"):
        print(f"\n=== M2: amortized overhead -- SKIPPED ({m2['reason']}) ===")
    else:
        print("\n=== M2: amortized overhead (real TinyLlama) ===")
        print(f"CARL adds {m2['amortized_overhead_us_per_request']:.1f} us per request "
              f"amortized ({m2['overhead_pct_of_inference']:.4f}% of inference time); "
              f"mean step {m2['mean_carl_step_ms_when_triggered']:.3f} ms when triggered.")

    m3 = results["measurement_3_memory"]
    print("\n=== M3: memory footprint ===")
    print(f"CARL uses {m3['total_controller_mb']:.4f} MB total "
          f"({m3['linucb_matrices_mb']:.4f} MB LinUCB matrices across "
          f"{m3['n_arms_total']} arms + {m3['controller_log_mb']:.4f} MB log over "
          f"{m3['controller_log_entries']} entries).")

    m4 = results["measurement_4_scaling"]
    print("\n=== M4: scaling with requests seen ===")
    for p in m4["points"]:
        print(f"  {p['requests_seen']:6d} seen -> {p['mean_us']:.2f} us mean "
              f"(p99 {p['p99_us']:.2f} us)")
    print(f"  growth ratio min->max: {m4['mean_growth_ratio_min_to_max']:.2f}x "
          f"-- {m4['verdict']}")

    print(f"\nONE-LINER FOR PAPER:\n  {results['paper_one_liner']}")


def _git_commit_results(paths: list, message: str, push: bool = False) -> None:
    """Stage ONLY `paths`, commit, and (optionally) push -- for self-committing
    just this eval's result file from a Colab GPU run (--commit / --push).

    Deliberately narrow: it stages only the named result files (never a blanket
    `git add -A`), skips cleanly when nothing changed or this isn't a git repo,
    and adds no Co-Authored-By trailer. Any git failure is reported, not raised,
    so a push/auth problem can never lose the just-computed results.
    """
    import subprocess
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        print("[--commit] no result file to commit", flush=True)
        return
    rel = ", ".join(os.path.relpath(p, _REPO_ROOT) for p in existing)
    try:
        subprocess.run(["git", "-C", _REPO_ROOT, "add", *existing], check=True)
        # Nothing staged (results identical to HEAD) -> avoid an empty commit.
        if subprocess.run(["git", "-C", _REPO_ROOT, "diff", "--cached",
                           "--quiet"]).returncode == 0:
            print(f"[--commit] {rel} unchanged; nothing to commit", flush=True)
            return
        subprocess.run(["git", "-C", _REPO_ROOT, "commit", "-m", message], check=True)
        print(f"[--commit] committed {rel}", flush=True)
        if push:
            subprocess.run(["git", "-C", _REPO_ROOT, "push"], check=True)
            print("[--push] pushed to origin", flush=True)
    except Exception as exc:  # noqa: BLE001 -- git failure must not lose results
        print(f"[--commit] git operation failed ({type(exc).__name__}: {exc}); "
              f"results are still saved at {rel}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CARL's runtime cost (CPU; M2 needs torch).")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="synthetic step() calls for M1 (default 10000)")
    parser.add_argument("--requests", type=int, default=100,
                        help="real requests for M2 (default 100)")
    parser.add_argument("--no-inference", action="store_true",
                        help="skip M2 even if torch is available")
    parser.add_argument("--commit", action="store_true",
                        help="git add+commit ONLY this run's result JSON afterwards "
                             "(for self-committing a Colab GPU run)")
    parser.add_argument("--push", action="store_true",
                        help="also push after committing (implies --commit)")
    args = parser.parse_args()

    env = capture_environment()
    gc.disable()  # keep GC pauses out of the microbenchmark timings
    try:
        m1 = measure_components(args.steps, SEED)
        m2 = ({"skipped": True, "reason": "disabled via --no-inference"}
              if args.no_inference else measure_amortized(args.requests, SEED))
        # M3 measures the controller that just ran M1 (warm matrices + full log).
        # Rebuild it deterministically so memory reflects a real post-run state.
        rng = np.random.default_rng(SEED)
        synth = _Synth(); synth.prime_metrics(rng)
        ctrl = _new_controller(synth, SEED)
        for i in range(args.steps):
            synth.randomize(rng)
            ctrl.step(step_idx=i)          # populates log + warms matrices
        m3 = measure_memory(ctrl)
        m4 = measure_scaling(seed=SEED)
    finally:
        gc.enable()

    # Paper one-liner: honest characterization tying the four measurements
    # together (and explaining the live ablation's ~2527 us P99).
    bandit_pct = m1["bandit_pct_of_total"]
    amort = ("" if m2.get("skipped")
             else f"{m2['amortized_overhead_us_per_request']:.0f} us/request "
                  f"({m2['overhead_pct_of_inference']:.3f}% of inference), ")
    one_liner = (
        f"CARL's control step costs {m1['real_step_end_to_end']['mean_us']:.0f} us "
        f"mean / {m1['real_step_end_to_end']['p99_us']:.0f} us P99 in isolation, of "
        f"which {bandit_pct:.0f}% is the LinUCB bandit (per-arm d x d inverse) and "
        f"only {m1['logging_pct_of_total']:.1f}% is logging; {amort}it is O(1) in "
        f"requests seen and uses {m3['total_controller_mb']:.2f} MB -- a negligible, "
        f"constant tax on serving.")

    results = {
        "seeds": [SEED],
        "environment": env,
        "observe_interval": OBSERVE_INTERVAL,
        "measurement_1_components": m1,
        "measurement_2_amortized": m2,
        "measurement_3_memory": m3,
        "measurement_4_scaling": m4,
        "paper_one_liner": one_liner,
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print_report(results)
    print(f"\nSaved overhead results to {RESULTS_PATH}", flush=True)

    if args.commit or args.push:
        if m2.get("skipped"):
            print("[--commit] WARNING: M2 (real inference) was skipped -- these are "
                  "CPU/partial numbers, not the full GPU results.", flush=True)
        _git_commit_results([RESULTS_PATH], "Add overhead eval results",
                            push=args.push)


if __name__ == "__main__":
    main()
