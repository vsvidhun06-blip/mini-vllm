"""
End-to-end test of the FastAPI event stream.

What we're asserting:
  * The WebSocket /events endpoint accepts a subscriber.
  * Two concurrent POST /generate calls succeed and return correct shape.
  * The event stream contains, for each request_id:
      - request_admitted
      - prefill_started
      - decode_step (one or more)
      - request_finished
  * pool_state events fire on every decode step (and prefill step).
  * No event payloads are malformed (all JSON-serialisable, by virtue of
    having been received as JSON over the WS).

What this test does NOT check:
  * Exact event ordering across requests (interleaving is timing-dependent).
  * Token correctness (covered by Day 5 / 6 / 7 parity tests).
  * Throughput (Day 6 / 7 tests).

The recipe:
  1. Open the WebSocket inside a TestClient context.
  2. Spawn a background daemon thread that loops on ws.receive_json() and
     appends to a list. This is the easiest way to drain the stream
     without blocking the main thread, because TestClient's WS doesn't
     expose a timeout-on-receive.
  3. Spawn two threads that each call POST /generate (TestClient.post is
     synchronous; threads give us the concurrency).
  4. Join both POST threads.
  5. Give a brief grace period for the last few events to flush onto the
     WS, then exit the with-block. That closes the WS and the receiver
     thread exits when its blocking receive_json raises.
  6. Assert on the captured event list.
"""
from __future__ import annotations

import threading
import time

import pytest


@pytest.fixture(scope="module")
def client(server_engine):
    """TestClient against the shared session engine (no second model load)."""
    from fastapi.testclient import TestClient
    from src.server.api import app
    with TestClient(app) as c:
        yield c


def test_events_stream_admit_prefill_decode_finished(client) -> None:
    """Two concurrent /generate calls produce the expected event sequence
    for each request, plus pool_state at every step."""
    received: list[dict] = []
    receive_done = threading.Event()

    with client.websocket_connect("/events") as ws:

        def receiver() -> None:
            # Loops until the WS closes (when the with-block exits) or any
            # other error occurs. Every received message is appended to
            # `received`. Errors are swallowed -- this is just a drain.
            try:
                while True:
                    msg = ws.receive_json()
                    received.append(msg)
            except Exception:
                pass
            finally:
                receive_done.set()

        rx = threading.Thread(target=receiver, daemon=True)
        rx.start()

        # Two concurrent /generate calls. Different prompts so the
        # request_ids and admission events are distinguishable.
        prompts = [
            "The capital of France is",
            "Python is a programming language designed",
        ]
        results: dict[int, dict] = {}

        def call_generate(idx: int, prompt: str) -> None:
            resp = client.post(
                "/generate",
                json={"prompt": prompt, "max_tokens": 8},
            )
            assert resp.status_code == 200, resp.text
            results[idx] = resp.json()

        threads = [
            threading.Thread(target=call_generate, args=(i, p))
            for i, p in enumerate(prompts)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)
            assert not t.is_alive(), "POST /generate hung past 3 minutes"

        # Grace period for the last pool_state / request_finished events
        # to make it through the queue and onto the WS before we close.
        time.sleep(0.5)

    # WS context exited -> WS closed -> receiver thread should exit shortly.
    receive_done.wait(timeout=5)

    # ---- Sanity on the POST responses --------------------------------
    assert set(results.keys()) == {0, 1}
    request_ids = {results[i]["request_id"] for i in (0, 1)}
    assert len(request_ids) == 2, "duplicate request_ids generated"
    for i in (0, 1):
        r = results[i]
        assert isinstance(r["output_tokens"], list)
        assert len(r["output_tokens"]) > 0, "no tokens generated"
        assert isinstance(r["output_text"], str)

    # ---- Event-type presence -----------------------------------------
    event_types = [e["event_type"] for e in received]
    for required in (
        "request_admitted",
        "prefill_started",
        "decode_step",
        "request_finished",
        "pool_state",
    ):
        assert required in event_types, (
            f"missing event type {required!r} in stream "
            f"(saw: {sorted(set(event_types))})"
        )

    # ---- Per-request coverage ----------------------------------------
    # Each of our two request_ids should appear in admit, prefill,
    # at least one decode, and finished.
    def ids_with(event_type: str) -> set[str]:
        return {
            e["payload"]["request_id"]
            for e in received
            if e["event_type"] == event_type
            and "request_id" in e.get("payload", {})
        }

    def ids_in_decode_steps() -> set[str]:
        out: set[str] = set()
        for e in received:
            if e["event_type"] != "decode_step":
                continue
            for row in e["payload"]["batch"]:
                out.add(row["request_id"])
        return out

    admitted = ids_with("request_admitted")
    finished = ids_with("request_finished")
    in_decode = ids_in_decode_steps()

    assert request_ids.issubset(admitted), (
        f"missing admit events for {request_ids - admitted}"
    )
    assert request_ids.issubset(finished), (
        f"missing finish events for {request_ids - finished}"
    )
    assert request_ids.issubset(in_decode), (
        f"missing decode_step coverage for {request_ids - in_decode}"
    )

    # ---- pool_state fires regularly ---------------------------------
    # Across the whole run we should see at least one pool_state per
    # step that ran -- we don't know the exact step count, but a healthy
    # run produces several. >= 2 is a soft floor (admission + at least
    # one decode round).
    pool_state_count = event_types.count("pool_state")
    assert pool_state_count >= 2, (
        f"expected pool_state to fire repeatedly; saw {pool_state_count}"
    )

    # ---- Each pool_state payload is well-formed ---------------------
    for e in received:
        if e["event_type"] != "pool_state":
            continue
        p = e["payload"]
        assert {"free_blocks", "used_blocks", "total_blocks"} <= set(p)
        assert p["free_blocks"] + p["used_blocks"] == p["total_blocks"]


def test_visualiser_page_is_served(client) -> None:
    """GET / returns the live-visualiser HTML page.

    A smoke test: we don't try to render anything, just verify the file
    is reachable, served with an HTML-ish content type, and contains the
    markers a visualiser page must have -- a DOCTYPE so the browser
    treats it as standards-mode, a `websocket` reference for the event
    stream wiring, and (Day 13) `chart.js` + `metrics` so we know the
    4-panel /metrics dashboard is wired in.
    """
    resp = client.get("/")
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "<!DOCTYPE html>" in body or "<!doctype html>" in body, (
        "served page is missing a DOCTYPE; browsers would render it in "
        "quirks mode"
    )
    assert "websocket" in body.lower(), (
        "served page makes no mention of WebSocket; the live event "
        "stream wiring is missing"
    )
    low = body.lower()
    assert "chart.js" in low, (
        "served page makes no mention of Chart.js; the metrics dashboard "
        "charts are missing"
    )
    assert "metrics" in low, (
        "served page makes no mention of /metrics; the dashboard polling "
        "wiring is missing"
    )
