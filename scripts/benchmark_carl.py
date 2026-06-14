"""
CARL evaluation harness -- the core paper experiment.

WHAT THIS IS (read this before trusting any number it prints)
------------------------------------------------------------
This is a CONTROL-LOOP SIMULATION, not a GPU benchmark. It drives the REAL CARL
machinery -- the real RuntimeState/classifier, the real LinUCB/Thompson bandits,
the real CARLController with its real reward -- over four workload scenarios, but
the serving metrics (throughput, TTFT, TPOT, cache/spec rates) come from an
explicit analytical cost model (WorkloadModel below), not from running TinyLlama
on hardware. This mirrors scripts/benchmark_router.py, which likewise measures
the decision layer and not model execution.

Why a simulation: the contribution under test is the CONTROLLER -- does a unified
online bandit that jointly adapts all knobs converge to the per-regime optimum
and beat independent tuning under non-stationary load? That question is about
the control policy, which is hardware-independent. A faithful answer needs many
hundreds of regime-varying control cycles; a CPU simulation with a transparent,
documented reward surface isolates the policy cleanly and runs anywhere. The
absolute metric values are therefore illustrative; the COMPARISONS between
baselines (and CARL's convergence to the oracle) are the result.

HONEST LIMITATION: because the cost model encodes the same domain knowledge as
the hand-tuned DEFAULT_CONFIGS, the regime-oracle is near-optimal BY
CONSTRUCTION, and CARL's job is to LEARN to approach it online without being told
the regime. We do NOT claim a measured wall-clock speedup on real hardware; that
would require the GPU end-to-end harness described in the paper's future work.

Scenarios (per the spec):
  1. INTERACTIVE      200 short-prompt requests, latency SLO.
  2. BATCH            200 long-prompt requests, throughput SLO.
  3. NON-STATIONARY   600 requests: INTERACTIVE -> BATCH -> BURST (the key test).
  4. REAL LMSYS       500 LMSYS-Chat-1M prompts (--real) or a synthetic mix.

Baselines:
  static_default          one fixed config, never adapts.
  independent_autotuner   the real AutoTuner (per-component, bottleneck-reactive).
  regime_oracle           DEFAULT_CONFIGS[true_regime] -- perfect regime knowledge.
  carl_linucb             CARL with LinUCB (proposed).
  carl_thompson           CARL with Thompson Sampling (ablation; --thompson).

Run:
  python scripts/benchmark_carl.py                 # all scenarios, synthetic
  python scripts/benchmark_carl.py --scenario 3    # just the non-stationary test
  python scripts/benchmark_carl.py --real          # scenario 4 from real LMSYS
  python scripts/benchmark_carl.py --thompson      # include the TS ablation
Outputs a results table per scenario, an ablation table, and docs/carl_results.json.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from src.carl.bandit import LinUCBBandit, PerRegimeBandit, ThompsonSamplingBandit
from src.carl.config import CARLConfig, DEFAULT_CONFIGS, all_arm_sets
from src.carl.controller import SLO, CARLController
from src.carl.state import (
    FEATURE_DIM,
    MetricsTracker,
    RuntimeState,
    WorkloadRegime,
    classify_regime,
)
from src.engine.auto_tuner import AutoTuner
from src.engine.profiler import StepProfiler

DOCS = Path(__file__).resolve().parent.parent / "docs"
RESULTS_PATH = DOCS / "carl_results.json"
LMSYS_CACHE = DOCS / "lmsys_carl_cache.json"
LMSYS_DATASET = "lmsys/lmsys-chat-1m"

ROUND_SIZE = 10   # requests served per control cycle (one controller.step per round)


# ===========================================================================
# WorkloadModel: the analytical cost model.
# ===========================================================================
#
# Maps (config, true_regime) -> realised serving metrics. The single lever is a
# "match score" m in [0,1]: how close `config` is to that regime's hand-tuned
# optimum (DEFAULT_CONFIGS[regime]) across the salient knobs. A well-matched
# config yields higher throughput and lower latency; a mismatched one is
# penalised. Per-request latencies get multiplicative log-normal-ish noise so the
# percentiles and SLO-violation rates are non-degenerate.
# ===========================================================================

# Per-regime base operating point (the numbers a near-perfect config would hit).
# Chosen so the latency SLOs (TTFT 100ms / TPOT 50ms) and the throughput SLO
# (20 tok/s) are MEETABLE with a good config and MISSED with a bad one -- that
# gap is what separates the baselines.
_REGIME_BASE = {
    #                    tps   ttft   tpot  cache  spec_gain
    WorkloadRegime.INTERACTIVE:  (40.0,  55.0, 28.0, 0.10, 0.50),
    WorkloadRegime.BATCH:        (85.0, 150.0, 42.0, 0.15, 0.40),
    WorkloadRegime.BURST:        (55.0, 170.0, 50.0, 0.10, 0.15),
    WorkloadRegime.CACHE_HEAVY:  (60.0,  70.0, 33.0, 0.70, 0.45),
    WorkloadRegime.LONG_CONTEXT: (28.0, 240.0, 66.0, 0.20, 0.10),
}

# Knob ranges used to normalise the config-to-oracle distance (the match score).
_KNOB_SPAN = {
    "max_batch_size": (1, 32),
    "chunk_size": (64, 512),
    "spec_k": (0, 8),
    "eviction_threshold": (0.5, 0.95),
    "cache_affinity_weight": (0.0, 1.0),
    "eviction_window": (16, 64),
}


def _match_score(config: CARLConfig, regime: WorkloadRegime) -> float:
    """1.0 when `config` equals the regime's oracle, decaying with distance.

    Normalised Euclidean distance over the salient knobs, mapped to [0,1] via
    1 - d/sqrt(K). The oracle config scores 1.0; the further a config strays on
    the knobs that matter, the lower its score and the worse its metrics.
    """
    oracle = DEFAULT_CONFIGS[regime]
    sq = 0.0
    for knob, (lo, hi) in _KNOB_SPAN.items():
        span = hi - lo
        a = (getattr(config, knob) - lo) / span
        b = (getattr(oracle, knob) - lo) / span
        sq += (a - b) ** 2
    dist = (sq ** 0.5) / (len(_KNOB_SPAN) ** 0.5)   # in [0,1]
    return max(0.0, 1.0 - dist)


class WorkloadModel:
    """Turns a config + true regime into realised per-round serving metrics."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def simulate(self, config: CARLConfig, regime: WorkloadRegime, n_requests: int) -> dict:
        base_tps, base_ttft, base_tpot, base_cache, spec_gain = _REGIME_BASE[regime]
        m = _match_score(config, regime)

        # Aggregate throughput for the round: scales from 50% (poor match) to
        # 100% (oracle) of the regime's base rate.
        throughput = base_tps * (0.5 + 0.5 * m) * self._noise(0.05)

        # Speculation: a positive spec_k earns acceptance proportional to the
        # regime's spec_gain, but in BURST/LONG_CONTEXT (many in-flight / deep
        # context) drafting mostly misses and adds per-token overhead.
        if config.spec_k > 0:
            spec_acc = spec_gain * min(1.0, config.spec_k / 4.0)
            tpot_spec_penalty = 1.12 if regime in (
                WorkloadRegime.BURST, WorkloadRegime.LONG_CONTEXT) else 1.0
        else:
            spec_acc, tpot_spec_penalty = 0.0, 1.0

        # Cache: CACHE_HEAVY's hit rate is amplified by the routing affinity
        # weight (exploiting shared prefixes); other regimes get their small base.
        if regime is WorkloadRegime.CACHE_HEAVY:
            cache_hit = base_cache * (0.5 + 0.5 * config.cache_affinity_weight)
        else:
            cache_hit = base_cache
        cache_hit = min(1.0, cache_hit)

        # Per-request latencies. A good match pulls TTFT/TPOT down (1.4x at m=0
        # to 1.0x at m=1); speculation acceptance further trims TPOT.
        latency_factor = 1.4 - 0.4 * m
        ttft_samples, tpot_samples = [], []
        for _ in range(n_requests):
            ttft = base_ttft * latency_factor * self._noise(0.25)
            tpot = (base_tpot * latency_factor * tpot_spec_penalty
                    * (1.0 - 0.3 * spec_acc) * self._noise(0.25))
            ttft_samples.append(ttft)
            tpot_samples.append(tpot)

        return {
            "throughput": throughput,
            "ttft_samples": ttft_samples,
            "tpot_samples": tpot_samples,
            "cache_hit": cache_hit,
            "spec_acc": spec_acc,
            "match": m,
        }

    def _noise(self, sigma: float) -> float:
        """Multiplicative noise ~ 1 + N(0, sigma), floored to stay positive."""
        return max(0.3, 1.0 + self.rng.gauss(0.0, sigma))


# ===========================================================================
# State synthesis: turn a (regime, metrics) pair into a RuntimeState.
# ===========================================================================
#
# The controller observes a RuntimeState; in the simulation we synthesise one
# whose QUEUE/PROMPT features make classify_regime read the intended regime
# (with noise, so detection isn't trivially perfect -- that's what makes "regime
# detection accuracy" a real number), and whose LATENCY/THROUGHPUT features carry
# the previous config's realised metrics (so the reward credits that config).
# ===========================================================================

# Canonical feature seeds per regime -- pre-noise values that classify_regime
# maps to the intended label.
_REGIME_FEATURES = {
    WorkloadRegime.INTERACTIVE:  dict(queue_depth=2,  active=2,  prompt=40,  cache=0.10),
    WorkloadRegime.BATCH:        dict(queue_depth=12, active=10, prompt=300, cache=0.15),
    WorkloadRegime.BURST:        dict(queue_depth=40, active=3,  prompt=60,  cache=0.10),
    WorkloadRegime.CACHE_HEAVY:  dict(queue_depth=6,  active=5,  prompt=50,  cache=0.70),
    WorkloadRegime.LONG_CONTEXT: dict(queue_depth=4,  active=3,  prompt=700, cache=0.20),
}


def _synth_state(regime: WorkloadRegime, metrics: dict | None, rng: random.Random) -> RuntimeState:
    """Build a noisy RuntimeState for a regime, carrying `metrics` if given."""
    f = _REGIME_FEATURES[regime]
    jitter = lambda v, s: max(0.0, v * (1.0 + rng.gauss(0.0, s)))
    if metrics is not None:
        p50_ttft = _percentile(metrics["ttft_samples"], 50)
        p99_tpot = _percentile(metrics["tpot_samples"], 99)
        throughput = metrics["throughput"]
        cache = metrics["cache_hit"]
        spec = metrics["spec_acc"]
        batch_mean = float(f["active"])
    else:
        p50_ttft = p99_tpot = throughput = spec = 0.0
        cache = f["cache"]
        batch_mean = float(f["active"])
    return RuntimeState(
        queue_depth=int(jitter(f["queue_depth"], 0.2)),
        avg_prompt_len=jitter(f["prompt"], 0.15),
        gpu_utilization=0.0,
        cache_hit_rate=min(1.0, jitter(cache, 0.1)),
        spec_acceptance_rate=spec,
        p50_ttft_ms=p50_ttft,
        p99_tpot_ms=p99_tpot,
        throughput_tps=throughput,
        active_requests=int(f["active"]),
        batch_size_mean=batch_mean,
    )


# ===========================================================================
# Agents: each maps a per-round observed state -> the config to run this round.
# ===========================================================================


class StaticAgent:
    """Never adapts: returns one fixed config every round."""

    def __init__(self, config: CARLConfig, name: str) -> None:
        self.config = config
        self.name = name
        self.adaptations = 0   # static -> always 0 by definition

    def choose(self, true_regime, state) -> CARLConfig:
        return self.config


class OracleAgent:
    """Perfect regime knowledge: returns DEFAULT_CONFIGS[true_regime]."""

    name = "regime_oracle"

    def __init__(self) -> None:
        self._last = None
        self.adaptations = 0

    def choose(self, true_regime, state) -> CARLConfig:
        cfg = DEFAULT_CONFIGS[true_regime]
        if self._last is not None and cfg != self._last:
            self.adaptations += 1
        self._last = cfg
        return cfg


class AutoTunerAgent:
    """The real per-component AutoTuner, reacting to a synthetic bottleneck.

    This is the 'independent tuning' baseline. The AutoTuner only knows four
    scheduler knobs (chunk_size, max_batch_size, use_cuda_graphs, evict_threshold)
    and only moves the one matching the current bottleneck, on a cooldown. The
    remaining CARL knobs (spec_k, routing_threshold, cache_affinity_weight,
    eviction_window) are FIXED at the global defaults -- it cannot coordinate
    them. That gap is precisely what CARL is meant to close.
    """

    name = "independent_autotuner"

    # Regime -> which phase dominates, so the tuner reacts plausibly per regime.
    _BOTTLENECK = {
        WorkloadRegime.INTERACTIVE:  dict(decode=0.10, overhead=0.02),
        WorkloadRegime.BATCH:        dict(prefill=0.10, decode=0.02),
        WorkloadRegime.BURST:        dict(kv_alloc=0.10, decode=0.02),
        WorkloadRegime.CACHE_HEAVY:  dict(overhead=0.10, decode=0.02),
        WorkloadRegime.LONG_CONTEXT: dict(prefill=0.10, kv_alloc=0.03),
    }

    def __init__(self) -> None:
        from types import SimpleNamespace
        self.sched = SimpleNamespace(
            chunk_size=256, max_batch_size=8, use_cuda_graphs=True, evict_threshold=0.8,
        )
        self.profiler = StepProfiler(window=20)
        # Act every round with a short cooldown so it can track the workload.
        self.tuner = AutoTuner(self.profiler, tune_interval=1, cooldown=3)
        self._round = 0
        self.adaptations = 0

    def choose(self, true_regime, state) -> CARLConfig:
        self._round += 1
        # Feed the profiler this regime's synthetic phase profile, then tune.
        self.profiler.window.clear()
        phases = self._BOTTLENECK[true_regime]
        for _ in range(5):
            self.profiler.record_step(
                prefill=phases.get("prefill", 0.0), decode=phases.get("decode", 0.0),
                kv_alloc=phases.get("kv_alloc", 0.0), overhead=phases.get("overhead", 0.0),
            )
        entry = self.tuner.observe(self.sched, step=self._round)
        if entry is not None:
            self.adaptations += 1
        # Build a CARLConfig from the tuned scheduler knobs; the rest stay at the
        # global defaults (the knobs the AutoTuner cannot reach).
        return CARLConfig(
            max_batch_size=int(self.sched.max_batch_size),
            chunk_size=int(self.sched.chunk_size),
            use_cuda_graphs=bool(self.sched.use_cuda_graphs),
            eviction_threshold=float(self.sched.evict_threshold),
        ).clamp()


class CarlAgent:
    """The proposed controller. Wraps a CARLController + its delayed-reward loop."""

    def __init__(self, bandit_cls=LinUCBBandit, name: str = "carl_linucb",
                 slo: SLO | None = None, **bandit_kwargs) -> None:
        self.name = name
        self.metrics = MetricsTracker(window=ROUND_SIZE * 5)
        bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM, bandit_cls=bandit_cls,
                                 **bandit_kwargs)
        self.controller = CARLController(
            bandit=bandit, observe_interval=1, slo=slo or SLO(), metrics=self.metrics,
        )
        self._prev_metrics: dict | None = None

    @property
    def adaptations(self) -> int:
        return self.controller._total_adaptations

    def choose(self, true_regime, state) -> CARLConfig:
        # Feed the controller's metric windows with the PREVIOUS round's realised
        # samples so its reward evaluates the previous config; the state we pass
        # already carries those metrics (synthesised by the caller).
        if self._prev_metrics is not None:
            for t, p in zip(self._prev_metrics["ttft_samples"],
                            self._prev_metrics["tpot_samples"]):
                self.metrics.record_request(t, p)
            self.metrics.record_throughput(self._prev_metrics["throughput"])
        entry = self.controller.step(state=state)
        return entry.config

    def note_realised(self, metrics: dict) -> None:
        """Record the metrics actually realised by the chosen config this round."""
        self._prev_metrics = metrics


# ===========================================================================
# Scenario runner.
# ===========================================================================


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def run_scenario(rounds: list[WorkloadRegime], agent, slo: SLO, seed: int) -> dict:
    """Drive `agent` over a list of per-round TRUE regimes; collect metrics.

    Each round: synthesise the observed state (carrying the previous round's
    realised metrics), ask the agent for a config, realise that config's metrics
    via the WorkloadModel, and accumulate. Returns the metric summary.
    """
    rng = random.Random(seed)
    model = WorkloadModel(rng)

    ttft_all, tpot_all = [], []
    tps_all, cache_all, spec_all, cost_all = [], [], [], []
    regime_correct = 0
    prev_metrics: dict | None = None
    detected_regimes: list[WorkloadRegime] = []

    for true_regime in rounds:
        state = _synth_state(true_regime, prev_metrics, rng)
        detected = classify_regime(state)
        detected_regimes.append(detected)
        if detected is true_regime:
            regime_correct += 1

        config = agent.choose(true_regime, state)
        metrics = model.simulate(config, true_regime, ROUND_SIZE)
        if isinstance(agent, CarlAgent):
            agent.note_realised(metrics)

        ttft_all.extend(metrics["ttft_samples"])
        tpot_all.extend(metrics["tpot_samples"])
        tps_all.append(metrics["throughput"])
        cache_all.append(metrics["cache_hit"])
        spec_all.append(metrics["spec_acc"])
        # Cost/token proxy: 1.0 baseline, reduced by cache reuse + speculation.
        cost_all.append(max(0.3, 1.0 - 0.3 * metrics["cache_hit"] - 0.2 * metrics["spec_acc"]))
        prev_metrics = metrics

    # SLO satisfaction: a request is satisfied when it meets BOTH latency SLOs.
    n = len(ttft_all)
    satisfied = sum(1 for t, p in zip(ttft_all, tpot_all)
                    if t <= slo.ttft_ms and p <= slo.tpot_ms)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "agent": agent.name,
        "throughput_tps": mean(tps_all),
        "ttft_p50": _percentile(ttft_all, 50),
        "ttft_p95": _percentile(ttft_all, 95),
        "ttft_p99": _percentile(ttft_all, 99),
        "tpot_p50": _percentile(tpot_all, 50),
        "tpot_p95": _percentile(tpot_all, 95),
        "tpot_p99": _percentile(tpot_all, 99),
        "slo_satisfaction_pct": 100.0 * satisfied / n if n else 0.0,
        "cache_hit_rate": mean(cache_all),
        "spec_acceptance": mean(spec_all),
        "cost_per_token": mean(cost_all),
        "adaptations": getattr(agent, "adaptations", 0),
        "regime_detect_acc_pct": 100.0 * regime_correct / len(rounds),
        "_detected": detected_regimes,
    }


# ===========================================================================
# Scenario definitions.
# ===========================================================================


def _rounds(regime: WorkloadRegime, n_requests: int) -> list[WorkloadRegime]:
    return [regime] * (n_requests // ROUND_SIZE)


def scenario_interactive() -> list[WorkloadRegime]:
    return _rounds(WorkloadRegime.INTERACTIVE, 200)


def scenario_batch() -> list[WorkloadRegime]:
    return _rounds(WorkloadRegime.BATCH, 200)


def scenario_nonstationary() -> list[WorkloadRegime]:
    # 200 INTERACTIVE -> 200 BATCH -> 200 BURST, shifting without notice.
    return (_rounds(WorkloadRegime.INTERACTIVE, 200)
            + _rounds(WorkloadRegime.BATCH, 200)
            + _rounds(WorkloadRegime.BURST, 200))


def scenario_lmsys(real: bool, limit: int = 500) -> list[WorkloadRegime]:
    """Per-prompt regimes for the LMSYS scenario.

    Each real prompt's length sets avg_prompt_len; classify_regime then assigns
    the round's regime (so the stream is a natural mix of INTERACTIVE / BATCH /
    LONG_CONTEXT). With --real off (or LMSYS unavailable) we synthesise a mix
    with the same shape.
    """
    if real:
        prompts = _load_lmsys_prompts(limit)
        regimes = []
        for p in prompts:
            approx_tokens = max(1, len(p) // 4)   # ~4 chars/token heuristic
            s = RuntimeState(avg_prompt_len=approx_tokens, queue_depth=4, active_requests=3,
                             cache_hit_rate=0.1)
            regimes.append(classify_regime(s))
        # Collapse to ROUND_SIZE-sized rounds by majority regime per round.
        rounds = []
        for i in range(0, len(regimes), ROUND_SIZE):
            chunk = regimes[i:i + ROUND_SIZE]
            rounds.append(max(set(chunk), key=chunk.count))
        return rounds
    # Synthetic LMSYS-like mix: mostly interactive, some batch, a little long.
    rng = random.Random(0)
    pool = ([WorkloadRegime.INTERACTIVE] * 6 + [WorkloadRegime.BATCH] * 3
            + [WorkloadRegime.LONG_CONTEXT] * 1)
    return [rng.choice(pool) for _ in range(limit // ROUND_SIZE)]


def _load_lmsys_prompts(limit: int) -> list[str]:
    """Stream `limit` LMSYS-Chat-1M user prompts (cached to docs/ on first run)."""
    if LMSYS_CACHE.exists():
        cached = json.loads(LMSYS_CACHE.read_text(encoding="utf-8"))
        if len(cached) >= limit:
            return cached[:limit]
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the `datasets` package is required for --real. pip install datasets."
        ) from exc
    try:
        stream = load_dataset(LMSYS_DATASET, split="train", streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"could not open {LMSYS_DATASET}: it is gated. Accept its terms and "
            f"authenticate (hf auth login) first.\nUnderlying error: {exc}"
        ) from exc
    prompts: list[str] = []
    for row in stream:
        if len(prompts) >= limit:
            break
        conv = row.get("conversation")
        if isinstance(conv, list):
            for turn in conv:
                if isinstance(turn, dict) and turn.get("role") in ("user", "human"):
                    c = turn.get("content")
                    if isinstance(c, str) and c.strip():
                        prompts.append(c)
                        break
    DOCS.mkdir(exist_ok=True)
    LMSYS_CACHE.write_text(json.dumps(prompts), encoding="utf-8")
    return prompts[:limit]


# ===========================================================================
# Printing + orchestration.
# ===========================================================================

_COLUMNS = [
    ("agent", "agent", "{:>22}"),
    ("throughput_tps", "tok/s", "{:>7.1f}"),
    ("ttft_p50", "ttftP50", "{:>7.1f}"),
    ("ttft_p99", "ttftP99", "{:>7.1f}"),
    ("tpot_p99", "tpotP99", "{:>7.1f}"),
    ("slo_satisfaction_pct", "SLO%", "{:>6.1f}"),
    ("cache_hit_rate", "cache", "{:>6.2f}"),
    ("spec_acceptance", "spec", "{:>5.2f}"),
    ("cost_per_token", "cost/t", "{:>6.2f}"),
    ("adaptations", "adapt", "{:>5d}"),
    ("regime_detect_acc_pct", "regAcc%", "{:>7.1f}"),
]


def _print_table(title: str, results: list[dict]) -> None:
    print(f"\n=== {title} ===")
    header = " | ".join(f"{label:>{max(5, _w(fmt))}}" for _, label, fmt in _COLUMNS)
    print(header)
    print("-" * len(header))
    for r in results:
        cells = []
        for key, _label, fmt in _COLUMNS:
            try:
                cells.append(fmt.format(r[key]))
            except (KeyError, ValueError):
                cells.append(f"{r.get(key, ''):>7}")
        print(" | ".join(cells))


def _w(fmt: str) -> int:
    """Extract the field width from a format like '{:>7.1f}' (best-effort)."""
    digits = "".join(ch for ch in fmt if ch.isdigit())
    # The first run of digits after '>' is the width; fall back to 6.
    import re
    m = re.search(r">(\d+)", fmt)
    return int(m.group(1)) if m else 6


def _adaptation_lag(detected: list, true_rounds: list, boundaries: list[int]) -> list[int]:
    """Rounds from each regime boundary until `detected` matches the new regime.

    Returns one lag per boundary (0 == detected the change on the very first
    round of the new regime). Capped at the segment length if never detected.
    """
    lags: list[int] = []
    for b in boundaries:
        if b >= len(true_rounds):
            continue
        new_regime = true_rounds[b]
        lag = 0
        while b + lag < len(detected) and detected[b + lag] is not new_regime:
            lag += 1
        lags.append(lag)
    return lags


def run_all(args) -> dict:
    slo_latency = SLO(ttft_ms=100.0, tpot_ms=50.0, throughput_ref=50.0)
    out: dict = {}

    def carl_agents():
        agents = [CarlAgent(LinUCBBandit, "carl_linucb", slo_latency, alpha=0.5)]
        if args.thompson:
            agents.append(CarlAgent(ThompsonSamplingBandit, "carl_thompson",
                                    slo_latency, v=0.5, seed=args.seed))
        return agents

    # ---- Scenario 1: INTERACTIVE -------------------------------------
    if args.scenario in ("1", "all"):
        rounds = scenario_interactive()
        results = [
            run_scenario(rounds, StaticAgent(CARLConfig(), "static_default"), slo_latency, args.seed),
            *[run_scenario(rounds, a, slo_latency, args.seed) for a in carl_agents()],
        ]
        _print_table("SCENARIO 1: INTERACTIVE (200 short-prompt reqs, TTFT<100 TPOT<50)", results)
        out["scenario_1_interactive"] = _strip(results)

    # ---- Scenario 2: BATCH -------------------------------------------
    if args.scenario in ("2", "all"):
        rounds = scenario_batch()
        results = [
            run_scenario(rounds, StaticAgent(CARLConfig(), "static_default"), slo_latency, args.seed),
            *[run_scenario(rounds, a, slo_latency, args.seed) for a in carl_agents()],
        ]
        _print_table("SCENARIO 2: BATCH (200 long-prompt reqs, throughput regime)", results)
        out["scenario_2_batch"] = _strip(results)

    # ---- Scenario 3: NON-STATIONARY (key experiment) -----------------
    if args.scenario in ("3", "all"):
        rounds = scenario_nonstationary()
        results = [
            run_scenario(rounds, StaticAgent(
                DEFAULT_CONFIGS[WorkloadRegime.INTERACTIVE], "static_best_interactive"),
                slo_latency, args.seed),
            run_scenario(rounds, StaticAgent(
                DEFAULT_CONFIGS[WorkloadRegime.BATCH], "static_best_batch"),
                slo_latency, args.seed),
            run_scenario(rounds, AutoTunerAgent(), slo_latency, args.seed),
            run_scenario(rounds, OracleAgent(), slo_latency, args.seed),
            *[run_scenario(rounds, a, slo_latency, args.seed) for a in carl_agents()],
        ]
        _print_table("SCENARIO 3: NON-STATIONARY  INTERACTIVE->BATCH->BURST (KEY)", results)
        out["scenario_3_nonstationary"] = _strip(results)
        # Adaptation lag: at each regime boundary, how many rounds until CARL's
        # detected regime first matches the new true regime. The per-regime
        # bandit swaps policy the instant detection flips, so this lag is
        # detection-bound (and is what the paper reports as adaptation latency).
        boundaries = [200 // ROUND_SIZE, 400 // ROUND_SIZE]   # INTERACTIVE->BATCH->BURST
        carl_res = next((r for r in results if r["agent"] == "carl_linucb"), None)
        if carl_res is not None:
            lag = _adaptation_lag(carl_res["_detected"], rounds, boundaries)
            print(f"\nCARL adaptation lag (rounds to detect new regime): {lag}")
            out["scenario_3_adaptation_lag_rounds"] = lag

    # ---- Scenario 4: REAL LMSYS --------------------------------------
    if args.scenario in ("4", "all"):
        rounds = scenario_lmsys(args.real, limit=args.limit)
        src = "real LMSYS-Chat-1M" if args.real else "synthetic LMSYS-like mix"
        results = [
            run_scenario(rounds, StaticAgent(CARLConfig(), "static_default"), slo_latency, args.seed),
            run_scenario(rounds, AutoTunerAgent(), slo_latency, args.seed),
            run_scenario(rounds, OracleAgent(), slo_latency, args.seed),
            *[run_scenario(rounds, a, slo_latency, args.seed) for a in carl_agents()],
        ]
        _print_table(f"SCENARIO 4: {src} ({len(rounds)} rounds, mixed)", results)
        out["scenario_4_lmsys"] = _strip(results)
        out["scenario_4_source"] = src

    # ---- Ablation: CARL with each component frozen -------------------
    if args.scenario in ("3", "all"):
        _print_ablation(scenario_nonstationary(), slo_latency, args.seed, out)

    return out


def _strip(results: list[dict]) -> list[dict]:
    """Drop the bulky internal _detected field before JSON serialisation."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]


# Knob groups CARL coordinates; the ablation freezes each in turn at the global
# default to measure that component's marginal contribution.
_ABLATION_FREEZE = {
    "no_spec_decode": dict(spec_k=CARLConfig().spec_k),
    "no_batch_adapt": dict(max_batch_size=CARLConfig().max_batch_size),
    "no_eviction_adapt": dict(eviction_threshold=CARLConfig().eviction_threshold,
                              eviction_window=CARLConfig().eviction_window),
    "no_cache_affinity": dict(cache_affinity_weight=CARLConfig().cache_affinity_weight),
}


class _FrozenCarlAgent(CarlAgent):
    """CARL, but with one knob group pinned to the default (component disabled)."""

    def __init__(self, freeze: dict, name: str, slo: SLO) -> None:
        super().__init__(LinUCBBandit, name, slo, alpha=0.5)
        self._freeze = freeze

    def choose(self, true_regime, state) -> CARLConfig:
        cfg = super().choose(true_regime, state)
        return replace(cfg, **self._freeze)


def _print_ablation(rounds, slo, seed, out: dict) -> None:
    results = [run_scenario(rounds, CarlAgent(LinUCBBandit, "carl_full", slo, alpha=0.5), slo, seed)]
    for name, freeze in _ABLATION_FREEZE.items():
        results.append(run_scenario(rounds, _FrozenCarlAgent(freeze, f"carl_{name}", slo), slo, seed))
    _print_table("ABLATION (non-stationary): CARL vs each component disabled", results)
    out["ablation_nonstationary"] = _strip(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL evaluation (control-loop simulation).")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--real", action="store_true",
                        help="scenario 4 uses real LMSYS-Chat-1M prompts (gated; needs HF auth)")
    parser.add_argument("--limit", type=int, default=500, help="LMSYS prompts for scenario 4")
    parser.add_argument("--thompson", action="store_true",
                        help="include the Thompson-Sampling CARL ablation")
    parser.add_argument("--live", action="store_true",
                        help="run the REAL TinyLlama inference harness (src/carl/live.py) "
                             "instead of the simulation: serves 3 scenarios through the "
                             "engine, CARL adaptive vs a fixed baseline. Needs torch + the "
                             "model cache; --limit sets the per-scenario request count "
                             "(capped at 50 for Colab).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --live takes a completely different path: real model execution, not the
    # analytical simulation below. We import it lazily so the default simulation
    # run stays torch-free (importable on a box without torch / a GPU).
    if args.live:
        from src.carl.live import main_live
        main_live(args)
        return

    print("NOTE: this is a CONTROL-LOOP SIMULATION (see module docstring). It drives")
    print("the real CARL controller/bandits over an analytical serving cost model;")
    print("it does NOT run a model on a GPU. Comparisons, not absolute values, are")
    print("the result.\n")

    out = run_all(args)
    DOCS.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH.relative_to(Path.cwd())}"
          if RESULTS_PATH.is_relative_to(Path.cwd()) else f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
