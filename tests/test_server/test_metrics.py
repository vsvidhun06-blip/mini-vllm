"""
GET /metrics: the Prometheus scrape endpoint.

What we're asserting:
  * /metrics returns 200 with the Prometheus text content type.
  * The response carries every metric series the dashboard depends on.
  * The lifecycle counters move: firing N /generate requests bumps
    requests_total{status="admitted"} by N (and finished by N).
  * Prefix-cache counters record hits when concurrent requests share a
    prompt prefix.

Counters are process-global and cumulative -- other tests in the same
pytest run also fire requests through the shared engine. So every
assertion here works on a DELTA: scrape, act, scrape again, diff. We
never assert an absolute counter value.

The prefix-cache test fires its requests CONCURRENTLY (threads), the
same way test_events.py does. Prefix caching only shares a block while
the first request still holds it (Day 12 has no LRU retention); if we
fired sequentially, request 1 would finish and free its blocks before
request 2 was admitted, and there would be nothing to share.
"""
from __future__ import annotations

import re
import time
import uuid

import pytest


@pytest.fixture(scope="module")
def client(server_engine):
    """TestClient against the shared session engine."""
    from fastapi.testclient import TestClient
    from src.server.api import app
    with TestClient(app) as c:
        yield c


def _scrape(client) -> str:
    resp = client.get("/metrics")
    assert resp.status_code == 200, resp.text
    return resp.text


def _counter(body: str, name: str, labels: str = "") -> float:
    """Pull a single counter/gauge value out of a Prometheus scrape body.

    `name` is the fully-qualified series name (e.g. "requests_total").
    `labels` is the literal label block including braces, or "" for an
    unlabelled series. Returns 0.0 if the series isn't present yet --
    a counter that has never been incremented with a given label may
    legitimately be absent.
    """
    # Match `name{labels} value` at the start of a line, value last.
    pattern = re.compile(
        rf"^{re.escape(name)}{re.escape(labels)}\s+([0-9eE.+-]+)$",
        re.MULTILINE,
    )
    m = pattern.search(body)
    return float(m.group(1)) if m else 0.0


def test_metrics_endpoint_returns_prometheus(client) -> None:
    """200, and the canonical Prometheus text content type."""
    resp = client.get("/metrics")
    assert resp.status_code == 200, resp.text
    ct = resp.headers.get("content-type", "")
    # prometheus_client's CONTENT_TYPE_LATEST. We don't pin the exact
    # version digits (0.0.4 vs 1.0.0 depending on the client release) --
    # just that it's the text exposition format.
    assert ct.startswith("text/plain"), f"wrong content-type: {ct!r}"
    assert "version=" in ct, f"content-type missing format version: {ct!r}"


def test_metrics_contains_expected_series(client) -> None:
    """Every metric the engine and dashboard rely on is present."""
    body = _scrape(client)
    # Histograms surface as <name>_bucket/_sum/_count; checking the bare
    # stem is enough to know the instrument is registered and exported.
    expected = [
        "requests_total",
        "prefix_cache_hits_total",
        "prefix_cache_misses_total",
        "ttft_seconds",
        "tpot_seconds",
        "e2e_latency_seconds",
        "active_requests",
        "pool_blocks_used",
        "pool_blocks_cached",
        "pool_blocks_free",
    ]
    missing = [name for name in expected if name not in body]
    assert not missing, f"/metrics is missing series: {missing}"


def test_requests_counter_increments(client) -> None:
    """Firing 2 /generate requests bumps the admitted+finished counters by 2."""
    before = _scrape(client)
    admitted_before = _counter(before, "requests_total", '{status="admitted"}')
    finished_before = _counter(before, "requests_total", '{status="finished"}')

    for _ in range(2):
        resp = client.post(
            "/generate",
            json={"prompt": "The capital of France is", "max_tokens": 4},
        )
        assert resp.status_code == 200, resp.text

    # /generate blocks until the request finishes, so the request_finished
    # event has already fired by the time we get here. A short sleep
    # absorbs any last bus-callback latency.
    time.sleep(0.5)

    after = _scrape(client)
    admitted_after = _counter(after, "requests_total", '{status="admitted"}')
    finished_after = _counter(after, "requests_total", '{status="finished"}')

    assert admitted_after - admitted_before == 2, (
        f"expected +2 admitted, got "
        f"{admitted_after - admitted_before} ({admitted_before} -> {admitted_after})"
    )
    assert finished_after - finished_before == 2, (
        f"expected +2 finished, got "
        f"{finished_after - finished_before} "
        f"({finished_before} -> {finished_after})"
    )


def test_prefix_cache_hits_recorded(client) -> None:
    """Co-resident requests sharing a prompt prefix register cache hits.

    Day 12 has no LRU retention -- a block is only shareable while some
    request still references it. So request 2 gets a prefix-cache hit
    only if it is admitted while request 1's blocks are still live.

    That overlap is hard to force through the HTTP layer: the single
    pumper thread drains each request to completion before TestClient
    can get the next one submitted, so requests run sequentially and
    never co-reside. (test_requests_counter_increments already covers
    the HTTP submission path end-to-end.)

    So this test submits the shared-prefix requests straight to the
    scheduler -- but ALL of them under one acquisition of `_sched_lock`,
    so they are all sitting in the WAITING queue before the pumper's
    next admission step. That step admits them together: the first
    registers its block hashes, the rest hit. The metrics pipeline
    under test is the real one end to end -- request_admitted events,
    the EventBus, the subscribed collector, and the /metrics scrape.
    """
    from src.server import api

    shared_prefix = (
        "You are a careful and concise assistant. You answer questions "
        "using only the information you are given, you never invent "
        "facts, and you keep your answers short and direct. Given that, "
    )
    queries = [
        "what is the capital of France?",
        "what is the largest ocean on Earth?",
        "name a programming language designed for readability.",
        "in what year did humans first land on the Moon?",
    ]

    before = _scrape(client)
    hits_before = _counter(before, "prefix_cache_hits_total")

    tok = api._tokenizer
    eos = tok.eos_token_id
    rids = [f"pc-test-{uuid.uuid4().hex[:8]}" for _ in queries]

    # Queue every request under a single lock acquisition. The pumper
    # cannot step in between, so all four are WAITING together when it
    # next runs admission -> they are admitted in one step -> shares fire.
    with api._sched_lock:
        for rid, q in zip(rids, queries):
            ids = tok(shared_prefix + q, return_tensors="pt")["input_ids"]
            api._scheduler.add_request(
                request_id=rid,
                prompt_ids=ids,
                max_new_tokens=6,
                eos_token_id=eos,
            )
    api._pump_wakeup.set()

    # Poll /metrics until the hit counter moves (admission happens on the
    # pumper's very next step) or we give up. The requests themselves
    # finish in the background; small max_tokens keeps that quick.
    deadline = time.time() + 60
    hits_after = hits_before
    while time.time() < deadline:
        time.sleep(0.3)
        hits_after = _counter(_scrape(client), "prefix_cache_hits_total")
        if hits_after > hits_before:
            break

    assert hits_after > hits_before, (
        f"expected prefix_cache_hits_total to increase when four "
        f"requests sharing a prompt prefix are admitted together, but "
        f"it stayed at {hits_before} -> {hits_after}. Hit accounting "
        f"or the metrics wiring is broken."
    )

    # Let the requests drain so the pool is clean for later tests.
    time.sleep(1.0)
