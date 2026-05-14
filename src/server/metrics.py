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
            for row in p.get("batch", []):
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
            POOL_BLOCKS_FREE.set(free)
            POOL_BLOCKS_CACHED.set(cached)
            # used_blocks in the event is "everything not free"; the
            # uniquely-owned count is that minus the shared ones.
            POOL_BLOCKS_USED.set(used_total - cached)


# Module-level singleton. api.py and the test fixture both subscribe
# `collector.on_event` to the bus (inside their idempotent engine-init
# guards, so it is subscribed exactly once per process).
collector = MetricsCollector()
