"""
Prometheus-style observability for the inference engine.

Design (see the Day 13 walkthrough):

  The metric instruments live HERE, in the server layer, not in the
  engine. The engine stays a pure inference library with no
  `prometheus_client` dependency. Everything this module needs is
  already carried on the EventBus -- so the collector is just another
  bus subscriber, exactly like the SSE token pusher and the WebSocket
  fan-out.

  `MetricsCollector.on_event` is registered against the engine's
  EventBus at startup. It translates engine events into instrument
  updates:

    request_admitted   -> requests_total{admitted}++, active_requests++,
                          prefix-cache hit/miss counters, record admit_time
    prefill_done       -> TTFT observed (now - admit_time), seed TPOT clock
    decode_step        -> TPOT observed per request in the batch
    request_finished   -> requests_total{finished}++, active_requests--,
                          E2E latency observed, clear per-request state
    pool_state         -> POOL_BLOCKS_{USED,CACHED,FREE} gauges set

  Per-request timing state (admit_time, last_token_time) is keyed by
  request_id. It is touched only from the bus-emit path, which runs on
  the single pumper thread, so no lock is needed. The prometheus_client
  instruments are themselves thread-safe.

Why Prometheus text format and not a bespoke JSON endpoint:

  The text exposition format is the industry-standard scrape format.
  Exposing /metrics in it means a Prometheus server, Grafana, or a
  Datadog agent can scrape this engine with zero code changes. A JSON
  endpoint would force every consumer to learn a one-off schema -- a
  worse, non-portable reinvention of a solved problem. The visualiser
  is then "just another scraper": it fetches /metrics and parses the
  handful of series it charts.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from src.engine.events import Event


# ---------------------------------------------------------------------------
# Instruments.
# ---------------------------------------------------------------------------
#
# prometheus_client appends `_total` to Counter names automatically, so
# `Counter("requests", ...)` is exposed as `requests_total`. Histograms
# expose `<name>_bucket`, `<name>_sum`, `<name>_count`. Gauges expose
# `<name>` verbatim.
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "requests",
    "Lifecycle count of requests, by status.",
    ["status"],  # admitted | rejected | finished
)

PREFIX_CACHE_HITS_TOTAL = Counter(
    "prefix_cache_hits",
    "Prefill blocks satisfied by a prefix-cache hit (shared an existing block).",
)
PREFIX_CACHE_MISSES_TOTAL = Counter(
    "prefix_cache_misses",
    "Prefill blocks that had to be computed fresh (no cache hit).",
)

# TTFT: admission -> first token (prefill_done). Buckets span "instant"
# to "this request waited a while behind a long prefill".
TTFT_SECONDS = Histogram(
    "ttft_seconds",
    "Time to first token: request_admitted -> prefill_done.",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)
# TPOT: gap between consecutive tokens for one request. Buckets are
# tighter -- a healthy decode step is single-digit to tens of ms.
TPOT_SECONDS = Histogram(
    "tpot_seconds",
    "Time per output token: gap between consecutive tokens for a request.",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
)
# End-to-end: admission -> request_finished. Wide buckets -- a long
# generation legitimately runs for tens of seconds.
E2E_LATENCY_SECONDS = Histogram(
    "e2e_latency_seconds",
    "End-to-end latency: request_admitted -> request_finished.",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Requests currently admitted and not yet finished.",
)

# Pool gauges. The three are mutually exclusive and sum to the pool size:
#   used   = blocks held by exactly one request (ref_count == 1)
#   cached = blocks SHARED across requests (ref_count >= 2) -- prefix
#            caching actively doing work
#   free   = unallocated blocks
POOL_BLOCKS_USED = Gauge(
    "pool_blocks_used",
    "KV-cache blocks uniquely owned by a single request (ref_count == 1).",
)
POOL_BLOCKS_CACHED = Gauge(
    "pool_blocks_cached",
    "KV-cache blocks shared across requests by the prefix cache (ref_count >= 2).",
)
POOL_BLOCKS_FREE = Gauge(
    "pool_blocks_free",
    "Unallocated KV-cache blocks.",
)

# Day 15: speculative decoding acceptance rate. One observation per
# spec-decode round, recording the fraction of K drafted tokens that
# matched the base model's argmax. Buckets are uniform 0.0..1.0; the
# coarse-grained ones are enough for "drafts are mostly accepted /
# mostly rejected" at a glance, fine-grained ones for trend analysis.
#
# Aliases:
#   * Empty histogram (count == 0) means spec decode never ran this
#     process -- either disabled at the scheduler or the engine had
#     more than one DECODE request every step (v0.3 falls back to
#     batched decode in that case).
#   * Histogram with non-zero count but very low mean (~0.01) is the
#     honest TinyLlama early-exit result: algorithm correct, draft
#     fidelity poor because the model was not trained for early-exit.
#     See docs/design.md for the v0.4 plan (trained draft head).
SPEC_DECODE_ACCEPTANCE_RATE = Histogram(
    "spec_decode_acceptance_rate",
    "Per-round speculative decoding acceptance rate (accepted / K).",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ---------------------------------------------------------------------------
# Observability-stack instruments (Grafana dashboard + Prometheus scrape).
# ---------------------------------------------------------------------------
#
# These carry an explicit `mini_vllm_` prefix so they namespace cleanly
# alongside any other exporter on the same Prometheus, and so the Grafana
# dashboard queries are unambiguous. (The older series above predate the
# convention and stay unprefixed to avoid breaking existing scrapers/tests.)
# ---------------------------------------------------------------------------

# Backlog: requests queued for admission but not yet in the active batch.
# Driven off the per-step pool_state event's `waiting` count. A persistently
# high queue depth means the engine is saturated -- offered load exceeds the
# batch/pool capacity, and TTFT will climb because requests wait to be admitted.
QUEUE_DEPTH = Gauge(
    "mini_vllm_queue_depth",
    "Requests waiting for admission (not yet in the active batch).",
)

# KV-cache occupancy as a percentage of the block pool. The single most
# important saturation signal for a paged-attention engine: as this approaches
# 100% the scheduler can no longer admit new requests (queue depth rises) and
# may have to preempt. Driven off pool_state ((used+cached)/total * 100).
CACHE_UTILISATION = Gauge(
    "mini_vllm_cache_utilisation",
    "Percentage of KV-cache blocks in use (0-100).",
)

# Decode batch size per step. Throughput on a memory-bandwidth-bound decode is
# roughly proportional to how many requests share each forward, so this is the
# batching-efficiency signal. Buckets cover 1..64 concurrent decoders.
BATCH_SIZE = Histogram(
    "mini_vllm_batch_size",
    "Number of requests in each batched decode step.",
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

# Cumulative KV evictions (H2O long-context feature, Phase 2). Stays 0 unless an
# EvictingPagedKVCache is in use; when it moves, the engine is trading old
# context for budget to serve a sequence longer than the cache.
EVICTIONS_TOTAL = Counter(
    "mini_vllm_evictions",
    "Total H2O KV-cache evictions (tokens dropped to stay within budget).",
)

# CUDA-graph decode acceleration (Phase 1). Labelled hit/miss: `hit` = a
# captured graph was replayed (one launch), `miss` = eager fallback. The ratio
# hit/(hit+miss) is the graph hit rate the dashboard charts -- on CPU or before
# any capture it is all misses, which is the honest eager-path picture.
CUDA_GRAPH_HITS_TOTAL = Counter(
    "mini_vllm_cuda_graph_hits",
    "Batched-decode forwards served by a CUDA graph vs eager, by outcome.",
    ["outcome"],  # hit | miss
)
for _outcome in ("hit", "miss"):
    CUDA_GRAPH_HITS_TOTAL.labels(outcome=_outcome)


def observe_eviction(num_tokens: int = 1) -> None:
    """Record an H2O eviction dropping `num_tokens` tokens.

    Wired into EvictingPagedKVCache via its `eviction_observer` hook so the
    engine itself stays free of any prometheus dependency.
    """
    if num_tokens > 0:
        EVICTIONS_TOTAL.inc(num_tokens)


def observe_cuda_graph(hit: bool) -> None:
    """Record one batched-decode forward as a graph hit or an eager miss.

    Wired into ContinuousBatchScheduler via its `cuda_graph_observer` hook.
    """
    CUDA_GRAPH_HITS_TOTAL.labels(outcome="hit" if hit else "miss").inc()


# SLA scheduling: TTFT deadline misses. The counter is the headline SLA
# violation signal; the histogram shows HOW LATE misses were (tail behaviour).
# Both stay at 0 unless the SLA scheduler is running with TTFT deadlines set.
DEADLINE_MISSES_TOTAL = Counter(
    "mini_vllm_deadline_misses",
    "Requests that missed their TTFT deadline under the SLA scheduler.",
)
DEADLINE_MISS_MS = Histogram(
    "mini_vllm_deadline_miss_ms",
    "How far past the TTFT deadline a missed request was, in milliseconds.",
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500],
)


# Continuous profiling: which phase is currently the engine's bottleneck. A
# label gauge -- the active phase reads 1, the others 0 -- so a Grafana pie can
# render the bottleneck distribution and a state-timeline shows shifts over time.
BOTTLENECK_TYPE = Gauge(
    "mini_vllm_bottleneck_type",
    "Current dominant scheduler bottleneck (1 for the active phase, else 0).",
    ["type"],  # prefill | decode | memory | overhead
)
_BOTTLENECK_TYPES = ("prefill", "decode", "memory", "overhead")
for _bt in _BOTTLENECK_TYPES:
    BOTTLENECK_TYPE.labels(type=_bt).set(0)


def set_bottleneck(bottleneck: str | None) -> None:
    """Set the active-bottleneck gauge (1 for `bottleneck`, 0 for the rest).

    Driven from the engine pump loop off StepProfiler.bottleneck(). None (no
    steps profiled yet) clears every series to 0.
    """
    for t in _BOTTLENECK_TYPES:
        BOTTLENECK_TYPE.labels(type=t).set(1.0 if t == bottleneck else 0.0)


def observe_deadline_miss(missed_by_ms: float) -> None:
    """Record one TTFT deadline miss (count + how late it was).

    Wired into SLAScheduler via its `deadline_miss_callback` hook so the engine
    stays free of any prometheus dependency.
    """
    DEADLINE_MISSES_TOTAL.inc()
    if missed_by_ms > 0:
        DEADLINE_MISS_MS.observe(missed_by_ms)


def observe_spec_decode_round(accepted: int, k: int) -> None:
    """Record one spec-decode round's acceptance ratio.

    Wired to ContinuousBatchScheduler via the `spec_decode_observer`
    constructor argument, so each round of draft+verify produces one
    histogram observation. The scheduler emits this from the same
    thread as the rest of its work (no asyncio bridging needed -- the
    prometheus_client instrument is thread-safe).
    """
    if k <= 0:
        return
    SPEC_DECODE_ACCEPTANCE_RATE.observe(accepted / k)


# Touch every label value once so all three series appear in /metrics
# output from the first scrape, even before anything has happened.
# "rejected" stays at 0 for v0.2: the scheduler queues over-capacity
# requests (request_waiting) rather than hard-rejecting them, so there
# is no rejection path yet -- the label exists for schema completeness.
for _status in ("admitted", "rejected", "finished"):
    REQUESTS_TOTAL.labels(status=_status)


# ---------------------------------------------------------------------------
# The collector: an EventBus subscriber.
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Translates engine events into Prometheus instrument updates.

    One instance is created at module load (`collector` below) and its
    `on_event` method is subscribed to the engine's EventBus. All event
    timestamps come from `Event.timestamp` (wall clock at event
    construction) so every latency delta is measured on one consistent
    clock.
    """

    def __init__(self) -> None:
        # request_id -> admit timestamp. Set on request_admitted, read
        # on prefill_done (TTFT) and request_finished (E2E), cleared on
        # finish.
        self._admit_time: dict[str, float] = {}
        # request_id -> timestamp of that request's most recent token.
        # Seeded with the first-token time on prefill_done, advanced on
        # every decode_step the request appears in. The delta between
        # the stored value and the current event is one TPOT sample.
        self._last_token_time: dict[str, float] = {}

    def on_event(self, event: Event) -> None:
        et = event.event_type
        ts = event.timestamp
        p = event.payload

        if et == "request_admitted":
            rid = p["request_id"]
            REQUESTS_TOTAL.labels(status="admitted").inc()
            ACTIVE_REQUESTS.inc()
            self._admit_time[rid] = ts
            # Day 12 put the prefix-cache hit data right on this event.
            cached = p.get("cached_blocks", 0)
            total = p.get("total_prefill_blocks", 0)
            if cached:
                PREFIX_CACHE_HITS_TOTAL.inc(cached)
            misses = total - cached
            if misses:
                PREFIX_CACHE_MISSES_TOTAL.inc(misses)

        elif et == "prefill_done":
            rid = p["request_id"]
            admit = self._admit_time.get(rid)
            if admit is not None:
                TTFT_SECONDS.observe(ts - admit)
            # The prefill token IS this request's first token. Seed the
            # TPOT clock with it so the first decode_step yields a real
            # inter-token gap rather than being skipped.
            self._last_token_time[rid] = ts

        elif et == "decode_step":
            # One batched event covers every request that decoded this
            # step. Each gets its own TPOT sample against its own
            # previous-token time -- requests share scheduler steps but
            # keep independent token timelines.
            batch = p.get("batch", [])
            # Batching efficiency: how many requests shared this forward.
            if batch:
                BATCH_SIZE.observe(len(batch))
            for row in batch:
                rid = row["request_id"]
                prev = self._last_token_time.get(rid)
                if prev is not None:
                    TPOT_SECONDS.observe(ts - prev)
                self._last_token_time[rid] = ts

        elif et == "request_finished":
            rid = p["request_id"]
            REQUESTS_TOTAL.labels(status="finished").inc()
            ACTIVE_REQUESTS.dec()
            admit = self._admit_time.pop(rid, None)
            if admit is not None:
                E2E_LATENCY_SECONDS.observe(ts - admit)
            self._last_token_time.pop(rid, None)

        elif et == "pool_state":
            # cached_blocks was added to pool_state in Day 13. Older
            # emitters (or a hand-built event) may omit it; default 0.
            free = p["free_blocks"]
            used_total = p["used_blocks"]
            cached = p.get("cached_blocks", 0)
            total = p.get("total_blocks", 0)
            POOL_BLOCKS_FREE.set(free)
            POOL_BLOCKS_CACHED.set(cached)
            # used_blocks in the event is "everything not free"; the
            # uniquely-owned count is that minus the shared ones.
            POOL_BLOCKS_USED.set(used_total - cached)
            # Observability-stack gauges: backlog and cache pressure.
            QUEUE_DEPTH.set(p.get("waiting", 0))
            CACHE_UTILISATION.set(100.0 * used_total / total if total else 0.0)


# Module-level singleton. api.py and the test fixture both subscribe
# `collector.on_event` to the bus (inside their idempotent engine-init
# guards, so it is subscribed exactly once per process).
collector = MetricsCollector()
