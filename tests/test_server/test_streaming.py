"""
End-to-end test of POST /generate/stream (Server-Sent Events).

What we're asserting:
  * The SSE endpoint streams `data: {...}\n\n` frames over text/event-stream.
  * One frame per generated token, each with `token_id`, `token_str`, `step`.
  * A final `{"done": true, "total_tokens": ..., "total_steps": ...}` frame.
  * Tokens arrive in order.
  * The assembled output_text (decoded from the streamed token_ids) is
    byte-identical to what /generate (buffered) produces for the same
    prompt. Greedy is deterministic, so this MUST hold.

If this test fails, either:
  - the SSE plumbing is dropping/reordering tokens, or
  - /generate and /generate/stream are wired to different scheduling
    paths and one of them is breaking parity with the engine.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="module")
def client(server_engine):
    """TestClient against the shared session engine."""
    from fastapi.testclient import TestClient
    from src.server.api import app
    with TestClient(app) as c:
        yield c


PROMPT = "The capital of France is"
MAX_TOKENS = 8


def _parse_sse_frames(body: str) -> list[dict]:
    """Split an SSE response body into a list of decoded JSON payloads.

    SSE format: events separated by a blank line; each event has one or
    more `data: ...` lines. We're only emitting `data:` lines, so
    extracting them is enough.
    """
    out: list[dict] = []
    for frame in body.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


def test_streaming_tokens_match_buffered(client) -> None:
    """SSE tokens reassemble to the same output as the buffered endpoint."""
    # Buffered request first, to get the reference token sequence.
    buf_resp = client.post(
        "/generate",
        json={"prompt": PROMPT, "max_tokens": MAX_TOKENS},
    )
    assert buf_resp.status_code == 200, buf_resp.text
    buf = buf_resp.json()
    assert buf["output_tokens"], "buffered run produced no tokens"

    # Now the streaming endpoint. TestClient.stream() gives us the raw
    # streaming response so we can iterate the body as it arrives.
    streamed_tokens: list[int] = []
    streamed_strs: list[str] = []
    done_event: dict | None = None
    steps_seen: list[int] = []

    with client.stream(
        "POST",
        "/generate/stream",
        json={"prompt": PROMPT, "max_tokens": MAX_TOKENS},
    ) as resp:
        assert resp.status_code == 200, resp.read()
        ct = resp.headers.get("content-type", "")
        assert ct.startswith("text/event-stream"), f"wrong content-type: {ct!r}"
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    frames = _parse_sse_frames(body)
    for frame in frames:
        if frame.get("done"):
            done_event = frame
            continue
        assert "token_id" in frame, f"non-done frame missing token_id: {frame}"
        assert "token_str" in frame, f"non-done frame missing token_str: {frame}"
        assert "step" in frame, f"non-done frame missing step: {frame}"
        streamed_tokens.append(frame["token_id"])
        streamed_strs.append(frame["token_str"])
        steps_seen.append(frame["step"])

    # ---- Done event well-formed ---------------------------------------
    assert done_event is not None, "stream ended without a done frame"
    assert done_event.get("total_tokens") == len(streamed_tokens), (
        f"done.total_tokens={done_event.get('total_tokens')} disagrees "
        f"with number of streamed frames={len(streamed_tokens)}"
    )
    assert done_event.get("total_steps", 0) >= 1, "no steps recorded"

    # ---- Token sequence matches buffered ------------------------------
    assert streamed_tokens == buf["output_tokens"], (
        f"streamed token ids differ from buffered.\n"
        f"  buffered: {buf['output_tokens']}\n"
        f"  streamed: {streamed_tokens}"
    )

    # ---- Step indices non-decreasing ---------------------------------
    # The pumper drives step() monotonically; per-token step values must
    # be non-decreasing within a single request.
    for a, b in zip(steps_seen, steps_seen[1:]):
        assert b >= a, f"step indices went backward: {a} -> {b}"
