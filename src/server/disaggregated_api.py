"""
FastAPI server over the DisaggregatedEngine (Mooncake-style prefill/decode split).

WHY A SEPARATE SERVER MODULE
----------------------------
src/server/api.py serves the UNIFIED engine (one ContinuousBatchScheduler that
interleaves prefill and decode in the same step loop). This module serves the
DISAGGREGATED engine, where prefill and decode are distinct workers connected by
a transfer queue. The two engines have different internals and different
observability, so they get different servers rather than one overloaded factory.

What this module keeps identical to api.py (so a client can talk to either):
  * POST /generate  -- JSON in, JSON out, blocks until the completion is done.
  * WS   /events    -- the same structured EventBus stream (prefill_started,
                       prefill_done, decode_step, request_finished, ...). The
                       DisaggregatedEngine emits into the same EventBus shape,
                       so an existing /events consumer works unchanged.

What this module ADDS -- the whole point of disaggregation observability:
  * GET  /stats     -- the three queue depths that make the architecture legible:
                         prefill_queue_depth   prompts submitted, not yet prefilled
                         decode_queue_depth    requests actively decoding
                         transfer_buffer_size  KV bundles in flight on the link
                       In a real deployment these are the numbers you watch to
                       see whether prefill or decode is the bottleneck and whether
                       the interconnect is backing up.

ENGINE LIFECYCLE / TESTABILITY
------------------------------
create_app(engine=None):
  * Production: pass nothing. The app lazily loads TinyLlama on first request
    (same lazy-init contract as api.py -- import is cheap, model load is not).
  * Tests: pass a pre-built DisaggregatedEngine over a tiny random-weight model.
    The lazy loader short-circuits, no HF download happens, and the test can
    populate engine.prefill_queue / transfer_queue / decode_worker.active by
    hand to assert /stats deterministically.

THREADING NOTE
--------------
DisaggregatedEngine.generate() is synchronous and internally calls
asyncio.run(), which needs its OWN event loop with no loop already running on
the thread. FastAPI's async handlers run inside the server's event loop, so we
must NOT call generate() directly from an `async def` handler (asyncio.run would
raise "cannot be called from a running event loop"). We therefore run /generate
as a SYNC handler -- FastAPI dispatches sync handlers to a threadpool worker,
which has no running loop, so asyncio.run is free to spin its own. /events stays
async because it only reads the EventBus and the queue depths.
"""
from __future__ import annotations

from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.engine.device import DEVICE
from src.engine.disaggregated import DisaggregatedEngine
from src.engine.events import Event, EventBus
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf


# ---------------------------------------------------------------------------
# Request / response schemas.
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20


class GenerateResponse(BaseModel):
    output_tokens: list[int]
    output_text: str


class StatsResponse(BaseModel):
    prefill_queue_depth: int
    decode_queue_depth: int
    transfer_buffer_size: int


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app(engine: DisaggregatedEngine | None = None) -> FastAPI:
    """Build the disaggregated server.

    If `engine` is supplied it is used as-is (the test path -- a tiny model, no
    download). Otherwise the engine, tokenizer, and EventBus are built lazily on
    first request via `_ensure_engine`, so importing this module never loads a
    4 GB checkpoint.
    """
    app = FastAPI()

    # Per-app state. We attach to `app.state` rather than module globals so two
    # apps in one process (e.g. unified + disaggregated under the same test run)
    # never clobber each other.
    app.state.engine = engine
    app.state.tokenizer = None
    app.state.event_bus = engine.event_bus if engine is not None else None

    def _ensure_engine() -> DisaggregatedEngine:
        """Lazily construct the engine + tokenizer if not injected. Idempotent."""
        if app.state.engine is not None:
            return app.state.engine

        from transformers import AutoTokenizer

        app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
        model.eval()
        bus = EventBus()
        app.state.event_bus = bus
        app.state.engine = DisaggregatedEngine(model, event_bus=bus)
        return app.state.engine

    # -- POST /generate (buffered, SYNC handler -- see module docstring) ------
    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        eng = _ensure_engine()
        tok = app.state.tokenizer

        # The injected-engine (test) path may have no tokenizer wired. Accept a
        # space-separated list of integer token ids as a fallback so the server
        # is exercisable without a real tokenizer.
        if tok is not None:
            input_ids = tok(req.prompt, return_tensors="pt")["input_ids"]
            eos = tok.eos_token_id
        else:
            ids = [int(x) for x in req.prompt.split()]
            input_ids = torch.tensor([ids], dtype=torch.long)
            eos = None

        out = eng.generate(input_ids, max_new_tokens=req.max_tokens, eos_token_id=eos)
        new_tokens = out[0, input_ids.shape[1]:].tolist()
        text = tok.decode(new_tokens, skip_special_tokens=True) if tok is not None else ""
        return GenerateResponse(output_tokens=new_tokens, output_text=text)

    # -- GET /stats -- the disaggregation observability surface ---------------
    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        eng = _ensure_engine()
        s = eng.stats()
        return StatsResponse(
            prefill_queue_depth=s["prefill_queue_depth"],
            decode_queue_depth=s["decode_queue_depth"],
            transfer_buffer_size=s["transfer_buffer_size"],
        )

    # -- WS /events -- identical contract to api.py's /events -----------------
    #
    # Same bridge pattern: capture the loop, subscribe a thread-safe push, and
    # forward every Event to the socket as JSON. The DisaggregatedEngine emits
    # prefill_started / prefill_done / decode_step / request_finished into this
    # bus, so an existing visualiser consumer renders it unchanged.
    @app.websocket("/events")
    async def events_ws(ws: WebSocket) -> None:
        import asyncio

        eng = _ensure_engine()
        bus = app.state.event_bus
        await ws.accept()

        # If the engine was built without a bus there is nothing to stream;
        # hold the socket open until the client disconnects.
        if bus is None:
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue()

        def _push(event: Event) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        bus.subscribe(_push)
        try:
            while True:
                event = await queue.get()
                await ws.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            bus.unsubscribe(_push)

    return app


# Module-level app for `uvicorn src.server.disaggregated_api:app`. Lazy engine
# init means constructing this does not load the model.
app = create_app()
