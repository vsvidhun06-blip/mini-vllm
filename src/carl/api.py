"""
FastAPI surface for the CARL controller.

Five endpoints, exposed as an APIRouter so the main server can mount them with a
single include_router() and so they can be unit-tested against a stub controller
without standing up the whole engine:

  GET  /carl/state    current RuntimeState (observed live) + detected regime.
  GET  /carl/stats    controller.stats() (regime/config distributions, rewards).
  GET  /carl/log      the last 100 control decisions.
  POST /carl/config   manually override the CARLConfig (sticky, for ablation).
  POST /carl/reset    wipe bandit learning + controller history.

The router is built from a `get_controller` callback rather than a direct
reference, so the server can wire it to its module-global `_carl_controller`
(which is None until CARL is enabled). Every handler degrades gracefully to a
{"enabled": false} response when the controller is absent -- mounting the routes
is therefore always safe, even on a server started with CARL disabled.
"""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter
from pydantic import BaseModel

from src.carl.config import CARLConfig
from src.carl.state import classify_regime


class CARLConfigOverride(BaseModel):
    """Partial CARLConfig for POST /carl/config; any omitted knob keeps its default.

    All fields optional so an ablation can pin a single knob (e.g. spec_k=0) and
    leave the rest at the dataclass defaults. The controller clamps on apply.
    """
    max_batch_size: int | None = None
    chunk_size: int | None = None
    preemption_enabled: bool | None = None
    spec_k: int | None = None
    routing_threshold: float | None = None
    cache_affinity_weight: float | None = None
    eviction_threshold: float | None = None
    eviction_window: int | None = None
    use_cuda_graphs: bool | None = None


def build_carl_router(get_controller: Callable[[], object]) -> APIRouter:
    """Build the /carl APIRouter bound to a controller-provider callback.

    Args:
        get_controller: zero-arg callable returning the live CARLController, or
            None when CARL is disabled.
    """
    router = APIRouter(prefix="/carl", tags=["carl"])

    def _disabled() -> dict:
        return {"enabled": False, "detail": "CARL controller is not active"}

    # -- GET /carl/state ---------------------------------------------------
    @router.get("/state")
    def carl_state() -> dict:
        controller = get_controller()
        if controller is None:
            return _disabled()
        state = controller.observe()
        return {
            "enabled": True,
            "regime": classify_regime(state).value,
            "state": state.as_dict(),
            "feature_vector": state.to_feature_vector(),
        }

    # -- GET /carl/stats ---------------------------------------------------
    @router.get("/stats")
    def carl_stats() -> dict:
        controller = get_controller()
        if controller is None:
            return _disabled()
        return {"enabled": True, **controller.stats()}

    # -- GET /carl/log -----------------------------------------------------
    @router.get("/log")
    def carl_log() -> dict:
        controller = get_controller()
        if controller is None:
            return _disabled()
        # Last 100 decisions, oldest-first, as JSON-friendly dicts.
        tail = controller.controller_log[-100:]
        return {"enabled": True, "log": [e.as_dict() for e in tail]}

    # -- POST /carl/config -------------------------------------------------
    @router.post("/config")
    def carl_config(override: CARLConfigOverride) -> dict:
        controller = get_controller()
        if controller is None:
            return _disabled()
        # Merge the partial override onto the dataclass defaults, clamp, apply.
        provided = {k: v for k, v in override.model_dump().items() if v is not None}
        config = CARLConfig.from_dict(provided)
        controller.apply_override(config)
        return {"enabled": True, "applied": config.as_dict()}

    # -- POST /carl/reset --------------------------------------------------
    @router.post("/reset")
    def carl_reset() -> dict:
        controller = get_controller()
        if controller is None:
            return _disabled()
        controller.reset()
        return {"enabled": True, "reset": True}

    return router
