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
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.engine.device import DEVICE
from src.engine.events import Event, EventBus
from src.engine.lora import LoRAManager
from src.engine.lora_model import LoRALlamaModel, random_adapter_weights
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler
from src.engine.sla_scheduler import (
    SLAScheduler,
    parse_priority,
    sla_scheduler_enabled,
)
from src.engine.auto_tuner import AutoTuner
from src.engine.profiler import StepProfiler
from src.server import metrics

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
# The raw (un-LoRA-wrapped) LlamaModel, kept so the draft/target speculative
# decoder can drive the model directly (it needs model.layers / model.embed for
# the early-exit draft, which the LoRA wrapper does not expose). Set in
# _init_engine; None until then.
_base_model: Any = None
# LoRA adapter registry, shared with the LoRALlamaModel the scheduler drives.
# POST /adapters registers into this; GenerateRequest.adapter_id selects from it.
_lora_manager: LoRAManager | None = None
_lora_model: LoRALlamaModel | None = None
# Continuous profiling + auto-tuning. The profiler is attached to the scheduler
# (it marks each step's phase boundaries); the tuner reads its rolling bottleneck
# and nudges live scheduler params. Both are inspectable via /profiler and
# /tuning-log. None until _init_engine runs.
_profiler: Any = None
_auto_tuner: Any = None
# CARL coordinated controller. None unless ENABLE_CARL is set (backward compat:
# a server started without it is byte-identical to before). When active, the
# pumper calls _carl_controller.maybe_step() once per scheduler step and the
# /carl/* routes expose its state. See src/carl/.
_carl_controller: Any = None
_sched_lock = threading.Lock()
_token_streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
_streams_lock = threading.Lock()
_pump_thread: threading.Thread | None = None
_pump_wakeup = threading.Event()
_pump_shutdown = threading.Event()


# ---------------------------------------------------------------------------
# Engine init.
# ---------------------------------------------------------------------------


def _carl_enabled() -> bool:
    """Whether the CARL coordinated controller should be constructed.

    Opt-in via the ENABLE_CARL env var (truthy: "1"/"true"/"yes"), mirroring the
    ENABLE_SLA_SCHEDULER gate. Off by default so the engine is unchanged.
    """
    return os.environ.get("ENABLE_CARL", "").strip().lower() in ("1", "true", "yes")


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


def _on_deadline_miss(request_id: str, missed_by_ms: float) -> None:
    """SLAScheduler hook: a request blew its TTFT deadline. Surface it on the
    request's own stream (if any) so a client can observe the SLA violation,
    and record it for the metrics layer."""
    metrics.observe_deadline_miss(missed_by_ms)
    _push_to_stream(
        {"request_id": request_id, "deadline_miss_ms": missed_by_ms},
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
                # Continuous profiling + auto-tuning: let the tuner read the
                # rolling bottleneck and adjust live params (it only acts on its
                # own interval), and publish the bottleneck gauge for Grafana.
                # Both run under _sched_lock so param mutation is race-free.
                if _auto_tuner is not None:
                    _auto_tuner.observe(_scheduler)
                if _profiler is not None:
                    metrics.set_bottleneck(_profiler.bottleneck())
                # CARL: one coordinated control cycle, self-gated to its observe
                # interval. Runs under _sched_lock so its live param mutations are
                # race-free with the scheduler step above. Disabled (None) by
                # default, so this is a no-op unless ENABLE_CARL was set.
                if _carl_controller is not None:
                    _carl_controller.maybe_step(_scheduler._step_idx)
            time.sleep(0)


def _init_engine() -> None:
    """Load the model exactly once. Idempotent for tests + threads."""
    global _event_bus, _scheduler, _tokenizer, _pump_thread, _lora_manager, _lora_model
    global _base_model, _profiler, _auto_tuner, _carl_controller
    if _scheduler is not None:
        return

    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # load_tinyllama_from_hf now moves the model to DEVICE (CUDA if
    # available). The .eval() call is harmless after the move.
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    # Keep the raw model for the speculative-decoding endpoint (it drives the
    # model directly, bypassing the LoRA wrapper and the scheduler).
    _base_model = model

    # Wrap the base model for multi-LoRA serving. With no adapter selected this
    # is numerically identical to the base model (LoRALinear's zero-overhead
    # path), so wrapping costs nothing until a request names an adapter_id.
    _lora_manager = LoRAManager(max_adapters=8)
    _lora_model = LoRALlamaModel(model, _lora_manager)
    model = _lora_model

    _event_bus = EventBus()
    _event_bus.subscribe(_on_bus_event)
    # The metrics collector is just another bus subscriber. Subscribing
    # here, inside the idempotent _init_engine guard, means it is wired
    # exactly once per process. (The test fixture in conftest.py builds
    # the engine by hand and subscribes the collector there for the
    # same reason.)
    _event_bus.subscribe(metrics.collector.on_event)

    # Speculative decoding is OFF in the public server by default because
    # on TinyLlama (no early-exit training) the measured acceptance rate
    # is ~1% -- below the breakeven threshold, so enabling it would slow
    # every /generate response down. The infrastructure is wired up
    # regardless: the metrics observer is always passed in, so /metrics
    # surfaces the spec_decode_acceptance_rate histogram with count==0
    # by default. Flipping `enable_spec_decode=True` here (or in a fork
    # of this file) is all that's needed to demo the algorithm on the
    # visualiser dashboard.
    # Chunked-prefill token budget per iteration. Configurable via the
    # CHUNK_SIZE env var (default 256). Lower it to make long prompts yield to
    # decode more aggressively; raise it toward "effectively unbounded" to get
    # the old single-shot full-prefill behaviour.
    chunk_size = int(os.environ.get("CHUNK_SIZE", "256"))

    # Shared scheduler kwargs. The SLA scheduler is a drop-in subclass, so when
    # ENABLE_SLA_SCHEDULER is set we construct it with the same arguments plus a
    # deadline-miss hook; otherwise the default FIFO engine is built unchanged.
    sched_kwargs = dict(
        max_batch_size=8,
        num_blocks=64,
        block_size=16,
        chunk_size=chunk_size,
        event_bus=_event_bus,
        token_decoder=lambda tid: _tokenizer.decode([tid], skip_special_tokens=False),
        token_emitter=_on_token,
        enable_spec_decode=False,
        spec_decode_k=4,
        spec_decode_observer=metrics.observe_spec_decode_round,
        cuda_graph_observer=metrics.observe_cuda_graph,
    )
    if sla_scheduler_enabled():
        _scheduler = SLAScheduler(
            model,
            deadline_miss_callback=_on_deadline_miss,
            **sched_kwargs,
        )
    else:
        _scheduler = ContinuousBatchScheduler(model, **sched_kwargs)

    # Continuous profiling + auto-tuning. The profiler marks each step's phase
    # boundaries (wall-clock by default -- a CUDAEventProfiler clock would sync
    # every step, which is too costly for the live serving loop). The tuner
    # reads its rolling bottleneck on its own interval and nudges live params.
    _profiler = StepProfiler(window=100)
    _scheduler.profiler = _profiler
    _auto_tuner = AutoTuner(_profiler)

    # CARL coordinated controller (opt-in via ENABLE_CARL). It JOINTLY adapts the
    # scheduler knobs the AutoTuner tunes independently, plus speculation depth,
    # routing, and eviction -- one bandit decision per observe interval. We wire
    # it to the scheduler (the only component the public server runs live); the
    # spec decoder / router / evicting cache slots are None here, so CARL drives
    # the scheduler subset and leaves the rest untouched. With ENABLE_CARL unset
    # this stays None and the engine behaves exactly as before. The independent
    # AutoTuner above is left running too; in practice you would enable one or
    # the other, but keeping both wired lets /tuning-log and /carl/* be compared
    # side by side on the same engine.
    if _carl_enabled():
        from src.carl.bandit import PerRegimeBandit
        from src.carl.config import all_arm_sets
        from src.carl.controller import CARLController
        from src.carl.state import FEATURE_DIM

        _carl_controller = CARLController(
            scheduler=_scheduler,
            bandit=PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM, alpha=0.5),
            observe_interval=50,
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
    # Optional LoRA adapter to serve this request under. None == base model.
    # Must already be registered via POST /adapters.
    adapter_id: str | None = None
    # Draft/target speculative decoding. 0 == disabled (the default; the
    # request goes through the normal continuous-batch scheduler). When > 0,
    # this request is served by a SpeculativeDecoder that drafts `speculative_k`
    # tokens per round and verifies them against the full model. See
    # _speculative_generate for the (buffered, demo) path and its caveats.
    speculative_k: int = 0
    # SLA scheduling hints. Only honoured when the engine was started with the
    # SLA scheduler (ENABLE_SLA_SCHEDULER); ignored by the default FIFO engine.
    # priority is one of "interactive" | "batch" | "background".
    priority: str = "interactive"
    ttft_deadline_ms: float | None = None


class AdapterRequest(BaseModel):
    """Register a LoRA adapter for serving.

    For this educational server the weights are synthesised (random, seeded by
    `seed`) at the model's projection dims -- enough to demonstrate hot-swap and
    per-request routing without shipping a multi-MB PEFT checkpoint over HTTP. A
    production endpoint would instead stream/point at real adapter weights.
    """
    adapter_id: str
    rank: int = 16
    alpha: float = 32.0
    seed: int = 0


class AdapterResponse(BaseModel):
    adapter_id: str
    rank: int
    alpha: float
    resident_adapters: list[str]


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


async def _stream_tokens_for_request(
    prompt: str, max_tokens: int, adapter_id: str | None = None,
    priority: str = "interactive", ttft_deadline_ms: float | None = None,
) -> AsyncIterator[dict]:
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
            # The SLA scheduler takes priority/deadline kwargs; the FIFO engine
            # does not. Pass them only when the SLA scheduler is in use so the
            # default path is byte-identical to before.
            if isinstance(_scheduler, SLAScheduler):
                _scheduler.add_request(
                    request_id=request_id,
                    prompt_ids=input_ids,
                    max_new_tokens=max_tokens,
                    eos_token_id=_tokenizer.eos_token_id,
                    prompt_text=prompt,
                    adapter_id=adapter_id,
                    priority=parse_priority(priority),
                    ttft_deadline_ms=ttft_deadline_ms,
                )
            else:
                _scheduler.add_request(
                    request_id=request_id,
                    prompt_ids=input_ids,
                    max_new_tokens=max_tokens,
                    eos_token_id=_tokenizer.eos_token_id,
                    prompt_text=prompt,
                    adapter_id=adapter_id,
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
# Draft/target speculative decoding (buffered demo path).
# ---------------------------------------------------------------------------
#
# When a request sets speculative_k > 0 we serve it OUTSIDE the continuous-batch
# scheduler, through a SpeculativeDecoder. The draft here is the target run
# shallow (SelfSpecDraftModel) -- the only "smaller model" we have on hand
# without a second checkpoint, so this demonstrates the exact acceptance-
# rejection algorithm on real weights without shipping a distilled draft.
#
# Honest scope: this path is buffered (no token streaming), runs the no-KV-cache
# verify forward (O(N^2) over the generation, fine for short demos), and does
# not flow through the scheduler's metrics/event bus. It exists to make the
# algorithm callable from the server; production batched spec-decode would wire
# it into the scheduler's decode loop, which is a larger change.
# ---------------------------------------------------------------------------


def _speculative_generate(prompt: str, max_tokens: int, k: int) -> tuple[list[int], float]:
    """Greedy-context speculative decode loop. Returns (token_ids, mean_accept)."""
    assert _base_model is not None and _tokenizer is not None
    from src.engine.spec_decode import (
        FullModelTarget,
        SelfSpecDraftModel,
        SpeculativeDecoder,
    )

    device = _base_model.embed.weight.device
    input_ids = _tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    eos = _tokenizer.eos_token_id

    decoder = SpeculativeDecoder(
        SelfSpecDraftModel(_base_model),
        FullModelTarget(_base_model),
        k=k,
    )
    ctx = input_ids
    out: list[int] = []
    while len(out) < max_tokens:
        emitted = decoder.decode_step(ctx)
        for tok in emitted:
            out.append(tok)
            ctx = torch.cat(
                [ctx, torch.tensor([[tok]], dtype=torch.long, device=device)], dim=1
            )
            if len(out) >= max_tokens or (eos is not None and tok == eos):
                break
        if eos is not None and out and out[-1] == eos:
            break
    return out[:max_tokens], decoder.mean_acceptance_rate


# ---------------------------------------------------------------------------
# Adapter registration / validation.
# ---------------------------------------------------------------------------


def _validate_adapter(adapter_id: str | None) -> None:
    """Raise 404 if a request names an adapter that isn't registered."""
    if adapter_id is None:
        return
    if _lora_manager is None or adapter_id not in _lora_manager:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail=f"adapter {adapter_id!r} not registered; POST it to /adapters first",
        )


def _register_adapter(req: "AdapterRequest") -> AdapterResponse:
    """Synthesise + register a LoRA adapter sized to the model's projections."""
    _init_engine()
    assert _lora_manager is not None and _lora_model is not None
    weights = random_adapter_weights(_lora_model, rank=req.rank, seed=req.seed)
    _lora_manager.load_adapter(req.adapter_id, req.rank, req.alpha, weights)
    return AdapterResponse(
        adapter_id=req.adapter_id,
        rank=req.rank,
        alpha=req.alpha,
        resident_adapters=_lora_manager.adapter_ids(),
    )


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def _startup() -> None:
        _init_engine()

    # -- POST /adapters (register a LoRA adapter) --------------------------
    @app.post("/adapters", response_model=AdapterResponse)
    def register_adapter(req: AdapterRequest) -> AdapterResponse:
        return _register_adapter(req)

    # -- POST /generate (buffered) ----------------------------------------
    #
    # Async now. Internally consumes the same per-token stream that
    # /generate/stream serves, collects tokens, joins via tokenizer,
    # returns the v0.1 response shape unchanged.
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        _init_engine()
        assert _tokenizer is not None

        _validate_adapter(req.adapter_id)

        # Draft/target speculative decoding bypasses the scheduler (buffered,
        # single-request demo path). Run it off the event loop so the blocking
        # forward passes don't stall other connections.
        if req.speculative_k > 0:
            tokens, _accept = await asyncio.to_thread(
                _speculative_generate, req.prompt, req.max_tokens, req.speculative_k
            )
            text = _tokenizer.decode(tokens, skip_special_tokens=True)
            return GenerateResponse(
                request_id=str(uuid.uuid4()),
                output_tokens=tokens,
                output_text=text,
            )

        request_id: str | None = None
        tokens: list[int] = []
        async for ev in _stream_tokens_for_request(
            req.prompt, req.max_tokens, req.adapter_id,
            priority=req.priority, ttft_deadline_ms=req.ttft_deadline_ms,
        ):
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

        _validate_adapter(req.adapter_id)

        async def sse() -> AsyncIterator[str]:
            async for ev in _stream_tokens_for_request(
                req.prompt, req.max_tokens, req.adapter_id,
                priority=req.priority, ttft_deadline_ms=req.ttft_deadline_ms,
            ):
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

    # -- GET /metrics ------------------------------------------------------
    #
    # Prometheus text exposition format. generate_latest() walks the
    # default registry (every instrument in src/server/metrics.py is
    # registered there on creation) and renders the standard scrape
    # body. CONTENT_TYPE_LATEST is the format's canonical content type
    # -- a real Prometheus server or Grafana scrapes this unchanged.
    @app.get("/metrics")
    def get_metrics() -> Response:
        _init_engine()
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # -- GET /profiler -----------------------------------------------------
    #
    # Current bottleneck + rolling per-phase stats from the StepProfiler. The
    # auto-tuner consumes the same data; this endpoint exposes it for dashboards
    # and ad-hoc inspection.
    @app.get("/profiler")
    def get_profiler() -> dict:
        _init_engine()
        if _profiler is None:
            return {"n_steps": 0, "bottleneck": None}
        return _profiler.to_dict()

    # -- GET /tuning-log ---------------------------------------------------
    #
    # The AutoTuner's history: every parameter change it has made, with the
    # step, the bottleneck that triggered it, and the old/new values.
    @app.get("/tuning-log")
    def get_tuning_log() -> dict:
        _init_engine()
        if _auto_tuner is None:
            return {"tuning_log": []}
        return {
            "tuning_log": _auto_tuner.log_as_dicts(),
            "current_bottleneck": _profiler.bottleneck() if _profiler else None,
        }

    # -- /carl/* (coordinated adaptive runtime) ----------------------------
    #
    # Mounted unconditionally; the routes return {"enabled": false} until CARL is
    # activated (ENABLE_CARL), so mounting is always safe and never changes the
    # behaviour of the existing endpoints. The router reads the live controller
    # through a callback so it always sees the current module global.
    from src.carl.api import build_carl_router

    app.include_router(build_carl_router(lambda: _carl_controller))

    # -- GET / -------------------------------------------------------------
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(VISUALISER_PATH, media_type="text/html")

    return app


app = create_app()
