"""
FastAPI surface for the LLM router.

Four endpoints, mirroring what a real routed-inference service exposes:

  POST /route/generate         Route a prompt automatically and return the
                               completion plus the routing decision metadata.
  GET  /route/stats            The router's running statistics (traffic split,
                               per-model latency, cost savings, complexity mix).
  GET  /route/models           The configured fleet (name, cost weight, the most
                               complex request each model handles, live latency).
  POST /route/generate/stream  SSE token streaming version of /route/generate.

This is a SELF-CONTAINED app (its own FastAPI instance), deliberately separate
from src/server/api.py -- that server hosts the continuous-batch engine; this
one hosts the routing layer. Keeping them apart means the router can be demoed,
benchmarked, or deployed on its own without standing up the whole scheduler.

The MultiModelServer + LLMRouter are process singletons, built once on first
request. Model weights are lazy-loaded inside the server (see serving.py), so
constructing the app is cheap and import-time has no torch cost.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.router.classifier import RuleBasedClassifier
from src.router.router import LLMRouter, default_model_configs
from src.router.serving import MultiModelServer


# ---------------------------------------------------------------------------
# Process-singleton state, built once by _get_components().
# ---------------------------------------------------------------------------
#
# We default to the RuleBasedClassifier: it's deterministic, dependency-free,
# and microsecond-fast, which is what you want sitting in front of every
# request. Swap in EmbeddingClassifier() here to get the semantic path (it falls
# back to rules automatically if sentence-transformers isn't installed).
# ---------------------------------------------------------------------------


_router: LLMRouter | None = None
_server: MultiModelServer | None = None


def _get_components() -> tuple[LLMRouter, MultiModelServer]:
    """Build the router + server once; return the shared instances."""
    global _router, _server
    if _router is None or _server is None:
        _router = LLMRouter(default_model_configs(), RuleBasedClassifier())
        _server = MultiModelServer()
    return _router, _server


# ---------------------------------------------------------------------------
# Request / response schemas.
# ---------------------------------------------------------------------------


class RouteGenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 32


class RouteGenerateResponse(BaseModel):
    response: str
    model_used: str
    complexity: str
    latency_ms: float
    cost_weight: float


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_router_app() -> FastAPI:
    app = FastAPI(title="mini-vLLM router")

    # -- POST /route/generate ---------------------------------------------
    #
    # Classify -> route -> generate, all in one call. We classify a second time
    # only to surface the complexity LABEL in the response; route() has already
    # made (and counted) the actual decision. The classify call is pure-Python
    # and effectively free, so this is cheaper than threading the label out of
    # route().
    @app.post("/route/generate", response_model=RouteGenerateResponse)
    def route_generate(req: RouteGenerateRequest) -> RouteGenerateResponse:
        router, server = _get_components()
        complexity = router._classify(req.prompt)
        response, model_used, latency_ms = server.generate(
            req.prompt, req.max_tokens, router
        )
        cost_weight = router._by_name[model_used].cost_per_token
        return RouteGenerateResponse(
            response=response,
            model_used=model_used,
            complexity=complexity.name,
            latency_ms=latency_ms,
            cost_weight=cost_weight,
        )

    # -- GET /route/stats --------------------------------------------------
    @app.get("/route/stats")
    def route_stats() -> dict:
        router, _server = _get_components()
        return router.routing_stats()

    # -- GET /route/models -------------------------------------------------
    #
    # The configured fleet, with live avg_latency_ms folded in. complexity is
    # rendered by .name so the JSON is human-readable ("MODERATE", not 1).
    @app.get("/route/models")
    def route_models() -> dict:
        router, server = _get_components()
        models = [
            {
                "name": c.name,
                "model_path": c.model_path,
                "cost_per_token": c.cost_per_token,
                "max_complexity": c.max_complexity.name,
                "avg_latency_ms": c.avg_latency_ms,
                "loaded": c.name in server.loaded_models,
            }
            for c in router.configs
        ]
        return {"models": models}

    # -- POST /route/generate/stream (SSE) --------------------------------
    #
    # Same routing decision as /route/generate, but the completion streams token
    # by token as `data: {...}\n\n` frames, with a final frame carrying the
    # routing metadata + assembled response. text/event-stream is the canonical
    # SSE content type. The blocking decode loop runs in a worker thread so it
    # doesn't stall the event loop; tokens are handed across via an asyncio queue.
    @app.post("/route/generate/stream")
    def route_generate_stream(req: RouteGenerateRequest) -> StreamingResponse:
        router, server = _get_components()

        async def sse() -> AsyncIterator[str]:
            import asyncio

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()

            def produce() -> None:
                # Runs in a worker thread: drive the (blocking) token generator
                # and bridge each event onto the asyncio queue thread-safely.
                try:
                    for event in server.generate_stream(
                        req.prompt, req.max_tokens, router
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            # Kick off the producer without blocking the event loop.
            asyncio.get_running_loop().run_in_executor(None, produce)

            while True:
                event = await queue.get()
                if event is _SENTINEL:
                    break
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


app = create_router_app()
