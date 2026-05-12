"""
FastAPI server: buffered /generate, SSE /generate/stream, WS /events.

Shape of the system:

    POST /generate -------+              +---- WS  /events (consumer A)
                          v              ^
    POST /generate/stream-+              +---- WS  /events (consumer B)
                          v              ^
                       [ContinuousBatchScheduler]
                       [_pumper thread drains step() under _sched_lock]
                          ^
                          |
                       [EventBus]                    +-- per-request
                          |                          |   asyncio.Queue
                       [token_emitter callback] -----+   (_token_streams)
                                                     |
                       [request_finished subscriber] +-- pushes done sentinel

  All /generate* requests funnel into one shared scheduler. The scheduler
  emits structured events into a shared EventBus and, for each generated
  token, calls a `token_emitter` callback. The server registers a
  callback that bridges those tokens onto per-request asyncio.Queues for
  SSE delivery.

Threading model:

  ONE dedicated daemon "pumper" thread drains the scheduler. It waits
  on a `threading.Event`; `add_request` sets the event. The pumper
  loops `while scheduler.has_work(): step()` under `_sched_lock`, then
  sleeps until the next wakeup. This replaces the v0.1 pattern where
  every /generate caller pumped the scheduler -- that was a workaround
  for /generate being sync, and it doesn't work for async streaming.
  One pumper, two endpoints, same submission path.

  The SSE endpoint runs in the FastAPI event loop. To get a token from
  the pumper thread onto an asyncio.Queue we use the documented
  thread-safe primitive `loop.call_soon_threadsafe(queue.put_nowait, ...)`.
  The scheduler stays asyncio-unaware; bridging happens at the callback
  layer the same way /events handles WebSocket fan-out.

Endpoint contract:

  POST /generate         JSON in, JSON out, blocks until done.
                         Internally consumes _stream_tokens_for_request
                         and joins.
  POST /generate/stream  JSON in, text/event-stream out.
                         One SSE event per token; final event has
                         {"done": true, "total_tokens": ..., "total_steps": ...}.
  WS /events             Scheduler-state events (admissions, blocks,
                         pool snapshots). Separate channel from per-request
                         token streaming -- two independent uses of the
                         engine state.
  GET /                  The visualiser SPA.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.engine.device import DEVICE
from src.engine.events import Event, EventBus
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler

# The visualiser is a single static HTML file. FileResponse re-reads on
# each request so iteration on the page doesn't need a server restart.
VISUALISER_PATH = Path(__file__).resolve().parent.parent / "visualiser" / "index.html"


# ---------------------------------------------------------------------------
# Module-level shared state.
# ---------------------------------------------------------------------------
#
# Loaded once at first request:
#   _event_bus, _scheduler, _tokenizer  -- the engine.
#   _sched_lock                         -- serialises scheduler mutations.
#   _token_streams                      -- per-request SSE queues, keyed
#                                          by request_id. (loop, queue) pairs.
#   _streams_lock                       -- guards _token_streams.
#   _pump_thread, _pump_wakeup          -- the dedicated pumper.
#
# Two queues per running request:
#   1. The shared EventBus delivers scheduler-level events to /events
#      subscribers (admissions, pool snapshots, decode_step batches).
#   2. _token_streams[request_id] delivers the request's own token stream
#      to its /generate or /generate/stream handler.
# ---------------------------------------------------------------------------


_event_bus: EventBus | None = None
_scheduler: ContinuousBatchScheduler | None = None
_tokenizer: Any = None
_sched_lock = threading.Lock()
_token_streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
_streams_lock = threading.Lock()
_pump_thread: threading.Thread | None = None
_pump_wakeup = threading.Event()
_pump_shutdown = threading.Event()


# ---------------------------------------------------------------------------
# Engine init.
# ---------------------------------------------------------------------------


def _push_to_stream(payload: dict, request_id: str) -> None:
    """Push `payload` onto the request's queue, if it has one."""
    with _streams_lock:
        stream = _token_streams.get(request_id)
    if stream is None:
        return
    loop, queue = stream
    # call_soon_threadsafe is the documented bridge from any thread into
    # an asyncio event loop. O(1), returns immediately, never blocks.
    loop.call_soon_threadsafe(queue.put_nowait, payload)


def _on_token(request_id: str, token_id: int, token_str: str, step_idx: int) -> None:
    """token_emitter callback: scheduler fires this for every token."""
    _push_to_stream(
        {
            "request_id": request_id,
            "token_id": token_id,
            "token_str": token_str,
            "step": step_idx,
        },
        request_id,
    )


def _on_bus_event(event: Event) -> None:
    """EventBus subscriber: surface request_finished as a done sentinel."""
    if event.event_type != "request_finished":
        return
    p = event.payload
    request_id = p["request_id"]
    _push_to_stream(
        {
            "request_id": request_id,
            "done": True,
            "total_tokens": p["total_tokens"],
            "total_steps": p["total_steps"],
            "reason": p.get("reason", ""),
        },
        request_id,
    )


def _pumper_loop() -> None:
    """Dedicated thread: drain scheduler.step() while there's outstanding work.

    Waits on `_pump_wakeup`. add_request sets the wakeup; we drain under
    `_sched_lock` until the scheduler has no more work, then sleep again.
    `time.sleep(0)` between steps lets other threads (the SSE handler
    sending bytes, /events forwarding state) make progress while we're
    iterating; on Windows this is a meaningful yield.
    """
    assert _scheduler is not None
    while not _pump_shutdown.is_set():
        # Wait briefly so shutdown stays responsive even if no one ever
        # wakes us. The cost of an extra spurious wake-and-check is
        # microseconds.
        _pump_wakeup.wait(timeout=1.0)
        _pump_wakeup.clear()
        while not _pump_shutdown.is_set():
            with _sched_lock:
                if not _scheduler.has_work():
                    break
                _scheduler.step()
                # Drain finished requests; we don't read them here (the
                # done sentinel is delivered via the event bus) but
                # leaving them in the scheduler's `finished` list would
                # grow unbounded.
                _scheduler.get_finished()
            time.sleep(0)


def _init_engine() -> None:
    """Load the model exactly once. Idempotent for tests + threads."""
    global _event_bus, _scheduler, _tokenizer, _pump_thread
    if _scheduler is not None:
        return

    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # load_tinyllama_from_hf now moves the model to DEVICE (CUDA if
    # available). The .eval() call is harmless after the move.
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()

    _event_bus = EventBus()
    _event_bus.subscribe(_on_bus_event)

    _scheduler = ContinuousBatchScheduler(
        model,
        max_batch_size=8,
        num_blocks=64,
        block_size=16,
        event_bus=_event_bus,
        token_decoder=lambda tid: _tokenizer.decode([tid], skip_special_tokens=False),
        token_emitter=_on_token,
    )

    # Single pumper thread for the lifetime of the process. Daemon so it
    # doesn't block interpreter shutdown if the user Ctrl-Cs uvicorn.
    _pump_thread = threading.Thread(target=_pumper_loop, daemon=True, name="engine-pump")
    _pump_thread.start()


# ---------------------------------------------------------------------------
# Request / response schemas.
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20


class GenerateResponse(BaseModel):
    request_id: str
    output_tokens: list[int]
    output_text: str


# ---------------------------------------------------------------------------
# The shared streaming generator.
# ---------------------------------------------------------------------------
#
# Both /generate and /generate/stream submit through this. It:
#   1. Generates a request_id.
#   2. Registers an asyncio.Queue in _token_streams BEFORE submitting --
#      registering after submit risks losing the first token if the
#      pumper sprints (unlikely on CPU, possible on GPU).
#   3. Submits to the scheduler under _sched_lock.
#   4. Wakes the pumper.
#   5. yield's per-token events as they arrive.
#   6. Stops after the done sentinel.
#   7. Cleans up the queue in `finally:` so a disconnect leaves no leak.
# ---------------------------------------------------------------------------


async def _stream_tokens_for_request(prompt: str, max_tokens: int) -> AsyncIterator[dict]:
    """Submit one prompt; yield per-token events until done."""
    assert _scheduler is not None and _tokenizer is not None
    request_id = str(uuid.uuid4())

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    # Register BEFORE submission so we cannot miss the first token. The
    # pumper, even if it sprints, will look up `request_id` in the
    # registry and find our queue.
    with _streams_lock:
        _token_streams[request_id] = (loop, queue)

    try:
        # Tokenise on CPU; the scheduler moves prompt_ids to DEVICE
        # internally in add_request.
        input_ids = _tokenizer(prompt, return_tensors="pt")["input_ids"]
        with _sched_lock:
            _scheduler.add_request(
                request_id=request_id,
                prompt_ids=input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=_tokenizer.eos_token_id,
                prompt_text=prompt,
            )
        _pump_wakeup.set()

        while True:
            event = await queue.get()
            yield event
            if event.get("done"):
                break
    finally:
        # Idempotent unregister. Runs on normal completion AND on client
        # disconnect (StreamingResponse cancellation propagates here).
        with _streams_lock:
            _token_streams.pop(request_id, None)


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def _startup() -> None:
        _init_engine()

    # -- POST /generate (buffered) ----------------------------------------
    #
    # Async now. Internally consumes the same per-token stream that
    # /generate/stream serves, collects tokens, joins via tokenizer,
    # returns the v0.1 response shape unchanged.
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        _init_engine()
        assert _tokenizer is not None

        request_id: str | None = None
        tokens: list[int] = []
        async for ev in _stream_tokens_for_request(req.prompt, req.max_tokens):
            if request_id is None and "request_id" in ev:
                request_id = ev["request_id"]
            if "token_id" in ev:
                tokens.append(ev["token_id"])
            if ev.get("done"):
                break

        text = _tokenizer.decode(tokens, skip_special_tokens=True)
        return GenerateResponse(
            request_id=request_id or "",
            output_tokens=tokens,
            output_text=text,
        )

    # -- POST /generate/stream (SSE) ---------------------------------------
    #
    # Server-Sent Events: one `data: {...}\n\n` frame per token, then a
    # final `data: {"done": true, ...}\n\n` frame. text/event-stream is
    # the canonical SSE content type and is what browsers' EventSource
    # parses; we use raw fetch() in the visualiser instead because we
    # POST a body and EventSource only does GET.
    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest) -> StreamingResponse:
        _init_engine()

        async def sse() -> AsyncIterator[str]:
            async for ev in _stream_tokens_for_request(req.prompt, req.max_tokens):
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # -- WS /events --------------------------------------------------------
    @app.websocket("/events")
    async def events_ws(ws: WebSocket) -> None:
        _init_engine()
        assert _event_bus is not None
        await ws.accept()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue()

        def _push(event: Event) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        _event_bus.subscribe(_push)
        try:
            while True:
                event = await queue.get()
                await ws.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            _event_bus.unsubscribe(_push)

    # -- GET / -------------------------------------------------------------
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(VISUALISER_PATH, media_type="text/html")

    return app


app = create_app()
