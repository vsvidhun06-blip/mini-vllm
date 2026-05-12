"""
FastAPI server: synchronous /generate + WebSocket /events stream.

Shape of the system:

    POST /generate -------+              +---- WS  /events (consumer A)
                          v              ^
                       [ContinuousBatchScheduler]
                          ^              v
    POST /generate -------+              +---- WS  /events (consumer B)
                            ^
                            |
                       [EventBus]

  All /generate requests funnel into one shared scheduler instance, so
  the scheduler can batch them. The scheduler emits events into a shared
  EventBus; every WebSocket subscriber gets every event.

Threading model (the bit that matters):

  /generate is a sync FastAPI endpoint, so FastAPI runs it in a
  threadpool worker thread. Several concurrent /generate calls -> several
  threads. We serialise their access to the scheduler with a single
  Lock; each iteration of the request's "pump loop" briefly holds the
  lock to call scheduler.step(). step() pumps EVERY active request at
  once -- that's the whole point of continuous batching -- so a request
  whose lock-holder just stepped doesn't have to wait for its own turn.

  /events is an async WebSocket endpoint. Inside, we:
    1. Capture the asyncio loop with asyncio.get_running_loop().
    2. Create a per-subscriber asyncio.Queue.
    3. Register a sync callback with the bus:
         lambda evt: loop.call_soon_threadsafe(queue.put_nowait, evt)
       call_soon_threadsafe is the documented thread-safe primitive for
       getting work onto an event loop from a non-loop thread. It runs
       in O(1) and returns immediately -- emit() never blocks.
    4. Loop: `evt = await queue.get(); await ws.send_json(...)`.

  The bus never knows about asyncio. The bridge is one line of subscriber
  glue, owned by the WebSocket endpoint.

Caveats kept tiny on purpose:

  * Backpressure: the asyncio.Queue is unbounded. A dead WS will grow
    its queue silently until we discover it on the next send_json and
    unsubscribe. Bounded queue + drop-on-full is the production move.
  * The /generate pump holds the scheduler lock during the full
    forward pass. step() is the only serialised section; the bus is
    not held under the scheduler lock.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.engine.events import Event, EventBus
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler

# The visualiser is a single static HTML file sitting next to the server
# package. We resolve it relative to this file so the path doesn't depend
# on the current working directory. FileResponse re-reads on each request,
# so edits to the file take effect without a server restart.
VISUALISER_PATH = Path(__file__).resolve().parent.parent / "visualiser" / "index.html"


# ---------------------------------------------------------------------------
# Module-level shared state.
# ---------------------------------------------------------------------------
#
# A single LlamaModel, scheduler, event bus, and tokenizer live at module
# scope. Loading TinyLlama costs a few seconds, so we do it once at
# import. The TestClient picks up exactly the same state -- which is what
# we want, so the WebSocket test sees events from the test's own
# /generate POSTs.
#
# `_sched_lock` serialises scheduler mutations (add_request, step,
# get_finished). Threads can be blocked on this for the duration of a
# forward pass (~50ms decode, more for prefill on a long prompt). That's
# acceptable: continuous batching only ever runs one engine iteration at
# a time anyway.
#
# `_results` collects finished requests across pump loops. We can't rely
# on whichever thread happens to pump the step that finishes request X
# being the same thread as the one waiting for X. So whoever sees a
# finished request stashes it; whoever was waiting picks it up.
# ---------------------------------------------------------------------------


_event_bus: EventBus | None = None
_scheduler: ContinuousBatchScheduler | None = None
_tokenizer: Any = None
_sched_lock = threading.Lock()
_results: dict[str, list[int]] = {}


def _init_engine() -> None:
    """Load the model exactly once, build the shared scheduler + bus."""
    global _event_bus, _scheduler, _tokenizer
    if _scheduler is not None:
        return

    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()

    _event_bus = EventBus()
    _scheduler = ContinuousBatchScheduler(
        model,
        max_batch_size=8,
        # 64 blocks at block_size=16 -> 1024 cached tokens, comfortable
        # for the demo workloads we expect.
        num_blocks=64,
        block_size=16,
        event_bus=_event_bus,
        # The decode_step event carries human-readable token text.
        # tokenizer.decode handles BPE merges so partial utf-8 sequences
        # render reasonably; for one token it returns a (possibly empty)
        # string fragment.
        token_decoder=lambda tid: _tokenizer.decode([tid], skip_special_tokens=False),
    )


# ---------------------------------------------------------------------------
# Request schema.
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20


class GenerateResponse(BaseModel):
    request_id: str
    output_tokens: list[int]
    output_text: str


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build the FastAPI app. Called once at module load and by tests."""
    app = FastAPI()

    @app.on_event("startup")
    def _startup() -> None:
        _init_engine()

    # -- POST /generate ----------------------------------------------------
    #
    # Synchronous request semantics: block until the request finishes,
    # then return tokens + text. The pump loop:
    #   - acquire lock
    #   - drain finished -> _results
    #   - if our id is in _results, return
    #   - else if scheduler has work, step()
    #   - else (no work for us, our id not finished) we're stuck; bail
    #   - release lock briefly so other threads can interleave
    #
    # The brief lock release between iterations lets concurrent /generate
    # callers add their requests and (more importantly) lets a thread
    # whose request finished early in someone else's step() actually
    # return.
    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        _init_engine()  # idempotent
        assert _scheduler is not None and _tokenizer is not None
        request_id = str(uuid.uuid4())
        input_ids = _tokenizer(req.prompt, return_tensors="pt")["input_ids"]

        with _sched_lock:
            _scheduler.add_request(
                request_id=request_id,
                prompt_ids=input_ids,
                max_new_tokens=req.max_tokens,
                eos_token_id=_tokenizer.eos_token_id,
                prompt_text=req.prompt,
            )

        # Pump until our request is done. Other threads may grab the lock
        # between iterations to do the same; that's fine -- whichever
        # thread runs step() drains finished requests into _results, and
        # everyone reads from _results.
        while True:
            with _sched_lock:
                # Move any newly-finished requests into the shared map.
                for r in _scheduler.get_finished():
                    _results[r.request_id] = list(r.generated_token_ids)
                if request_id in _results:
                    tokens = _results.pop(request_id)
                    break
                # Step the engine. If there's no work AND our request
                # isn't done, something is very wrong (our request
                # vanished); bail out empty rather than spin.
                if not _scheduler.has_work():
                    tokens = []
                    break
                _scheduler.step()
            # Tiny yield so other threads can take the lock between
            # forward passes. time.sleep(0) is enough to let the OS
            # reschedule on CPython under the GIL.
            time.sleep(0)

        text = _tokenizer.decode(tokens, skip_special_tokens=True)
        return GenerateResponse(
            request_id=request_id,
            output_tokens=tokens,
            output_text=text,
        )

    # -- WS /events --------------------------------------------------------
    #
    # The sync->async bridge. See module docstring for the threading
    # rationale. The flow:
    #   1. accept the WS.
    #   2. capture the loop, build a queue, subscribe a thread-safe
    #      pusher.
    #   3. loop: await queue.get(); await ws.send_json(...).
    #   4. on disconnect or any error, unsubscribe in finally.
    @app.websocket("/events")
    async def events_ws(ws: WebSocket) -> None:
        _init_engine()
        assert _event_bus is not None
        await ws.accept()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue()

        def _push(event: Event) -> None:
            # Runs in the worker thread that called emit(). Schedule a
            # put_nowait on the event loop and return immediately.
            loop.call_soon_threadsafe(queue.put_nowait, event)

        _event_bus.subscribe(_push)
        try:
            while True:
                event = await queue.get()
                await ws.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        except Exception:
            # Any other error (client gone, send failure) -- bail out and
            # let the finally clause unsubscribe.
            pass
        finally:
            _event_bus.unsubscribe(_push)

    # -- GET / -------------------------------------------------------------
    #
    # Serve the live visualiser. The HTML/CSS/JS lives in one file under
    # src/visualiser/index.html; FileResponse re-reads it on each request
    # so the page can be iterated on without restarting the server.
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(VISUALISER_PATH, media_type="text/html")

    return app


# Create the app at module load so `uvicorn src.server.api:app` works.
app = create_app()
