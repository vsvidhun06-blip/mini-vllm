"""
Runtime State Observer + Workload Regime classifier.

WHY THIS EXISTS
---------------
A controller can only adapt to what it can see. Every component in the engine
already exposes *some* live signal -- the scheduler knows its queue depth and
batch, the radix cache knows its hit rate, the spec decoder knows its acceptance
rate -- but those signals live in five different objects with five different
shapes. RuntimeState is the single, flat, normalized feature vector that fuses
them, so the bandit downstream sees one consistent context per decision.

DESIGN POINT: DEFENSIVE OBSERVATION
-----------------------------------
observe() reads the live components through getattr with fallbacks, never hard
attribute access. Two reasons:
  * The engine is assembled differently in different settings (a benchmark may
    have no radix cache; a unit test passes SimpleNamespace stubs). The observer
    must degrade to a sensible default (0.0 / 0) instead of throwing.
  * It keeps CARL a *additive* layer: wiring it in cannot break an engine that
    doesn't expose a particular knob yet.

The latency / throughput features can't be read off a single object -- they are
rolling statistics over recent requests and steps. MetricsTracker owns those
windows; the engine (or the benchmark) feeds it samples, and observe() reads the
percentiles back. Keeping the windows here (not in the controller) means the
same tracker can also compute the SLO-violation rates the reward needs.

This module is torch-free. GPU utilization is read via pynvml when present and
falls back to 0.0 otherwise, so nothing here requires a GPU.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# WorkloadRegime
# ---------------------------------------------------------------------------
#
# Five coarse regimes, each with a DIFFERENT optimal configuration shape. The
# whole point of naming them is that the per-regime bandit gets a warm start
# (DEFAULT_CONFIGS) and an isolated learning context: what's optimal for a BURST
# is actively wrong for LONG_CONTEXT, so sharing one bandit across them would
# average out the very differences we want to exploit.
# ---------------------------------------------------------------------------


class WorkloadRegime(Enum):
    INTERACTIVE = "interactive"   # low queue, short prompts, latency-sensitive
    BATCH = "batch"               # high queue, long prompts, throughput-sensitive
    BURST = "burst"               # a backlog the in-flight batch can't absorb yet
    CACHE_HEAVY = "cache_heavy"   # high cache hit rate, prefix-dominated traffic
    LONG_CONTEXT = "long_context"  # avg_prompt_len > 512: memory/compute heavy


# ---------------------------------------------------------------------------
# Feature normalization.
# ---------------------------------------------------------------------------
#
# The bandit is a LINEAR model, so the scale of each feature directly weights its
# influence on the UCB score. We divide each raw feature by a characteristic
# scale so every component lands roughly in [0, 1] and no single feature (e.g.
# throughput in the hundreds) drowns out the rest (e.g. a hit rate in [0, 1]).
# Values can exceed 1.0 for an extreme reading -- that's fine, the model just
# sees a larger-than-usual signal; we deliberately do NOT clamp so genuine
# outliers (a 4000-token prompt) stay distinguishable.
#
# The order of this dict IS the feature-vector order. to_feature_vector() and
# feature_names() both iterate it, so they can never drift apart.
# ---------------------------------------------------------------------------

_FEATURE_SCALES: dict[str, float] = {
    "queue_depth": 64.0,            # a deep queue is ~64 waiting requests
    "avg_prompt_len": 1024.0,       # 1k tokens is a "long" prompt
    "gpu_utilization": 1.0,         # already a fraction
    "cache_hit_rate": 1.0,          # already a fraction
    "spec_acceptance_rate": 1.0,    # already a fraction
    "p50_ttft_ms": 500.0,           # 500ms is a slow median TTFT
    "p99_tpot_ms": 200.0,           # 200ms is a slow tail per-output-token time
    "throughput_tps": 100.0,        # 100 tok/s is a healthy single-GPU rate
    "active_requests": 32.0,        # max sensible decode batch
    "batch_size_mean": 32.0,        # same scale as active_requests
}

# The number of context dimensions the bandit must be built with. Exported so
# the controller / benchmark size their bandits consistently with the observer.
FEATURE_DIM = len(_FEATURE_SCALES)


# ---------------------------------------------------------------------------
# RuntimeState
# ---------------------------------------------------------------------------


@dataclass
class RuntimeState:
    """A single observation of the whole serving system.

    Ten features, deliberately flat and JSON-serialisable, spanning the queue,
    the prompts, the GPU, the caches, the speculative decoder, and the realised
    latency/throughput. `to_feature_vector()` turns it into the normalized
    context the bandit consumes; `as_dict()` is what the /carl/state endpoint
    returns verbatim.
    """

    queue_depth: int = 0
    avg_prompt_len: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    spec_acceptance_rate: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    throughput_tps: float = 0.0
    active_requests: int = 0
    batch_size_mean: float = 0.0

    # ---- vector views ----------------------------------------------------

    @staticmethod
    def feature_names() -> list[str]:
        """Feature order for the context vector (matches to_feature_vector)."""
        return list(_FEATURE_SCALES.keys())

    def to_feature_vector(self) -> list[float]:
        """Normalized context vector, one entry per feature in _FEATURE_SCALES.

        Each raw feature divided by its characteristic scale (see the rationale
        on _FEATURE_SCALES). Returned as a plain list of floats so the module
        stays numpy-agnostic at its boundary; the bandit wraps it in an array.
        """
        return [getattr(self, name) / scale for name, scale in _FEATURE_SCALES.items()]

    def as_dict(self) -> dict:
        """JSON-friendly snapshot for the /carl/state endpoint."""
        return {name: getattr(self, name) for name in _FEATURE_SCALES}

    # ---- the observer ----------------------------------------------------

    @classmethod
    def observe(
        cls,
        scheduler=None,
        spec_decoder=None,
        router=None,
        kv_cache=None,
        metrics: "MetricsTracker | None" = None,
        gpu_utilization: float | None = None,
    ) -> "RuntimeState":
        """Build a RuntimeState from the live engine components.

        Every read is defensive (getattr with a default), so any component may
        be None or missing an attribute without breaking observation.

        Args:
            scheduler: a ContinuousBatchScheduler-like object. We read
                `waiting` (queue depth), `active` (in-flight requests), and the
                prompt lengths of both to estimate avg_prompt_len.
            spec_decoder: a SpeculativeDecoder-like object exposing
                `mean_acceptance_rate` (preferred) or `acceptance_rate`.
            router: an LLMRouter-like object. Reserved for routing-derived
                features; cache_hit_rate is read from kv_cache, so router is
                currently unused but kept in the signature for symmetry and
                forward compatibility (e.g. a future "cost_per_token" feature).
            kv_cache: a cache exposing a hit rate. We try, in order:
                `cache_hit_rate` (float/attr), `hit_rate()` (callable), or a
                (hits, lookups) pair via `cache_hits`/`cache_lookups`.
            metrics: the MetricsTracker holding rolling TTFT/TPOT/throughput/
                batch windows. When None those four features read 0.0.
            gpu_utilization: explicit override (0..1). When None we try pynvml
                and fall back to 0.0.

        Returns:
            A populated RuntimeState.
        """
        waiting = _seq(getattr(scheduler, "waiting", None))
        active = _seq(getattr(scheduler, "active", None))
        queue_depth = len(waiting)
        active_requests = len(active)

        avg_prompt_len = _avg_prompt_len(waiting, active)

        if gpu_utilization is None:
            gpu_utilization = _read_gpu_utilization()

        cache_hit_rate = _read_cache_hit_rate(kv_cache)
        spec_acceptance_rate = _read_spec_acceptance(spec_decoder)

        if metrics is not None:
            p50_ttft = metrics.p50_ttft_ms()
            p99_tpot = metrics.p99_tpot_ms()
            throughput = metrics.throughput_tps()
            batch_mean = metrics.batch_size_mean()
        else:
            p50_ttft = p99_tpot = throughput = batch_mean = 0.0
        # If no batch samples were recorded, fall back to the instantaneous
        # in-flight count so the feature is never silently zero on a live batch.
        if batch_mean == 0.0:
            batch_mean = float(active_requests)

        return cls(
            queue_depth=queue_depth,
            avg_prompt_len=avg_prompt_len,
            gpu_utilization=float(gpu_utilization),
            cache_hit_rate=cache_hit_rate,
            spec_acceptance_rate=spec_acceptance_rate,
            p50_ttft_ms=p50_ttft,
            p99_tpot_ms=p99_tpot,
            throughput_tps=throughput,
            active_requests=active_requests,
            batch_size_mean=batch_mean,
        )


# ---------------------------------------------------------------------------
# Defensive read helpers (kept module-level so observe() stays readable).
# ---------------------------------------------------------------------------


def _seq(obj) -> list:
    """Coerce a maybe-None maybe-deque into a concrete list for len()/iteration."""
    if obj is None:
        return []
    try:
        return list(obj)
    except TypeError:
        return []


def _prompt_len(req) -> int:
    """Best-effort prompt length of a scheduler Request-like object.

    Real Requests carry a (1, S) `prompt_ids` tensor; stubs in tests may carry a
    plain `prompt_len` int or a list. We try the cheap paths first and never
    touch torch (shape access works on a tensor without importing torch).
    """
    pl = getattr(req, "prompt_len", None)
    if isinstance(pl, (int, float)):
        return int(pl)
    ids = getattr(req, "prompt_ids", None)
    if ids is not None:
        shape = getattr(ids, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[-1])
        try:
            return len(ids)
        except TypeError:
            return 0
    return 0


def _avg_prompt_len(waiting: list, active: list) -> float:
    """Mean prompt length across all known (waiting + active) requests."""
    reqs = waiting + active
    if not reqs:
        return 0.0
    return sum(_prompt_len(r) for r in reqs) / len(reqs)


def _read_cache_hit_rate(kv_cache) -> float:
    """Pull a hit rate in [0, 1] from a cache object, trying several shapes."""
    if kv_cache is None:
        return 0.0
    # 1. A plain attribute or property.
    hr = getattr(kv_cache, "cache_hit_rate", None)
    if isinstance(hr, (int, float)):
        return float(hr)
    # 2. A method.
    fn = getattr(kv_cache, "hit_rate", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            pass
    # 3. Raw counters.
    hits = getattr(kv_cache, "cache_hits", None)
    lookups = getattr(kv_cache, "cache_lookups", None)
    if isinstance(hits, (int, float)) and isinstance(lookups, (int, float)) and lookups > 0:
        return float(hits) / float(lookups)
    return 0.0


def _read_spec_acceptance(spec_decoder) -> float:
    """Acceptance rate in [0, 1] from a SpeculativeDecoder-like object."""
    if spec_decoder is None:
        return 0.0
    mean = getattr(spec_decoder, "mean_acceptance_rate", None)
    if isinstance(mean, (int, float)):
        return float(mean)
    last = getattr(spec_decoder, "acceptance_rate", None)
    if isinstance(last, (int, float)):
        return float(last)
    return 0.0


# pynvml handle is cached across calls: nvmlInit is not free, and observe() runs
# on every controller tick. A single failed import disables GPU reads for the
# whole process (the flag below), so we don't pay the import cost repeatedly.
_NVML_STATE = {"tried": False, "ok": False, "handle": None, "mod": None}


def _read_gpu_utilization() -> float:
    """GPU utilization fraction (0..1) via pynvml, or 0.0 if unavailable.

    Lazily initialises NVML on first call and caches the device handle. Any
    failure (no pynvml, no NVIDIA driver, CPU box) latches the disabled state so
    subsequent calls are a cheap no-op returning 0.0.
    """
    st = _NVML_STATE
    if not st["tried"]:
        st["tried"] = True
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            st["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            st["mod"] = pynvml
            st["ok"] = True
        except Exception:
            st["ok"] = False
    if not st["ok"]:
        return 0.0
    try:
        util = st["mod"].nvmlDeviceGetUtilizationRates(st["handle"])
        return float(util.gpu) / 100.0   # NVML reports percent
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------
#
# Rolling windows over the two things you can't read off a single object: per-
# request latencies (TTFT, TPOT) and engine throughput. The engine/benchmark
# pushes samples in; observe() and the reward read summaries out. Default window
# 100, matching the StepProfiler and the spec's "rolling 100-step window".
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; NaN-free (returns 0.0 on an empty window)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


@dataclass
class MetricsTracker:
    """Rolling TTFT / TPOT / throughput / batch-size statistics."""

    window: int = 100
    _ttft: deque = field(default_factory=lambda: deque(maxlen=100))
    _tpot: deque = field(default_factory=lambda: deque(maxlen=100))
    _tps: deque = field(default_factory=lambda: deque(maxlen=100))
    _batch: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self) -> None:
        # Re-create the deques at the requested window if it isn't the default.
        if self.window != 100:
            self._ttft = deque(self._ttft, maxlen=self.window)
            self._tpot = deque(self._tpot, maxlen=self.window)
            self._tps = deque(self._tps, maxlen=self.window)
            self._batch = deque(self._batch, maxlen=self.window)

    # ---- sample ingestion ------------------------------------------------

    def record_request(self, ttft_ms: float, tpot_ms: float) -> None:
        """One completed request's time-to-first-token and per-output-token time."""
        self._ttft.append(float(ttft_ms))
        self._tpot.append(float(tpot_ms))

    def record_throughput(self, tokens_per_sec: float) -> None:
        self._tps.append(float(tokens_per_sec))

    def record_batch(self, batch_size: int) -> None:
        self._batch.append(float(batch_size))

    # ---- summaries -------------------------------------------------------

    def p50_ttft_ms(self) -> float:
        return _percentile(list(self._ttft), 50)

    def p99_tpot_ms(self) -> float:
        return _percentile(list(self._tpot), 99)

    def throughput_tps(self) -> float:
        vals = list(self._tps)
        return sum(vals) / len(vals) if vals else 0.0

    def batch_size_mean(self) -> float:
        vals = list(self._batch)
        return sum(vals) / len(vals) if vals else 0.0

    # ---- SLO violation rates (used by the reward) ------------------------

    def ttft_violation_rate(self, slo_ms: float) -> float:
        """Fraction of windowed requests whose TTFT exceeded `slo_ms`."""
        vals = list(self._ttft)
        if not vals:
            return 0.0
        return sum(1 for v in vals if v > slo_ms) / len(vals)

    def tpot_violation_rate(self, slo_ms: float) -> float:
        vals = list(self._tpot)
        if not vals:
            return 0.0
        return sum(1 for v in vals if v > slo_ms) / len(vals)


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------
#
# Rule-based, snapshot-only classifier. It maps one RuntimeState to one regime
# using thresholds, checked IN ORDER (the first match wins). Order encodes
# priority: a property that most dictates the right config is checked first.
#
# Why rule-based and not learned: the regime label only has to be a useful
# *context bucket* for the per-regime bandit; it does not have to be perfect.
# A cheap, legible, deterministic classifier is debuggable and never needs
# training data -- and any misclassification is recoverable because the bandit
# under the (wrong) regime still optimises against the real reward.
# ---------------------------------------------------------------------------

# Threshold constants, each justified inline at the use site.
_LONG_CONTEXT_PROMPT_TOKENS = 512.0   # spec-defined: > 512 avg prompt == long ctx
_CACHE_HEAVY_HIT_RATE = 0.5           # majority of lookups hitting the prefix cache
_BURST_QUEUE_DEPTH = 24.0             # a sizeable backlog...
_BURST_BACKLOG_RATIO = 2.0            # ...that is > 2x what's currently in flight
_BATCH_QUEUE_DEPTH = 8.0              # a steady queue the batch is absorbing
_BATCH_PROMPT_TOKENS = 256.0          # or moderately long prompts (throughput regime)


def classify_regime(state: RuntimeState) -> WorkloadRegime:
    """Classify a RuntimeState into a WorkloadRegime (first matching rule wins).

    Rule order and rationale:

      1. LONG_CONTEXT  -- avg_prompt_len > 512. Long prompts dominate both KV
         footprint and prefill cost; the right config (small batch, big chunk,
         gentle eviction, no speculation) is driven by prompt length regardless
         of what the queue is doing, so this is checked first.

      2. CACHE_HEAVY   -- cache_hit_rate >= 0.5. When most lookups hit the radix
         prefix cache the workload is prefix-dominated (shared system prompts,
         few-shot templates); the lever that matters is routing cache-affinity,
         not batch sizing, so it gets its own regime before the queue-based
         rules.

      3. BURST         -- queue_depth >= 24 AND queue_depth > 2 * active. A deep
         backlog that the in-flight batch is NOT yet absorbing is the snapshot
         signature of a sudden arrival surge. We separate it from steady BATCH
         because the transient response differs: drain aggressively (large
         batch, aggressive eviction) and DON'T spend cycles speculating.

      4. BATCH         -- queue_depth >= 8 OR avg_prompt_len >= 256. A sustained
         queue or moderately long prompts: a throughput-sensitive steady state.

      5. INTERACTIVE   -- the default. Shallow queue, short prompts: optimise for
         latency (small batch, light speculation, CUDA graphs).
    """
    # 1. Long context.
    if state.avg_prompt_len > _LONG_CONTEXT_PROMPT_TOKENS:
        return WorkloadRegime.LONG_CONTEXT

    # 2. Cache-heavy / prefix-dominated.
    if state.cache_hit_rate >= _CACHE_HEAVY_HIT_RATE:
        return WorkloadRegime.CACHE_HEAVY

    # 3. Burst: deep backlog relative to in-flight work.
    if (
        state.queue_depth >= _BURST_QUEUE_DEPTH
        and state.queue_depth > _BURST_BACKLOG_RATIO * max(state.active_requests, 1)
    ):
        return WorkloadRegime.BURST

    # 4. Steady batch / throughput regime.
    if state.queue_depth >= _BATCH_QUEUE_DEPTH or state.avg_prompt_len >= _BATCH_PROMPT_TOKENS:
        return WorkloadRegime.BATCH

    # 5. Default: interactive.
    return WorkloadRegime.INTERACTIVE
