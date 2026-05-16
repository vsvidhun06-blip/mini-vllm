"""
Shared session-scoped fixtures for the whole test suite.

Why these live at the tests/ root:
    TinyLlama-1.1B in fp32 is ~4.4 GB. The engine parity tests each
    need a copy; the server tests need a *running scheduler* over the
    same model. With three engine modules + one server module, naive
    per-module fixtures load 4-5 copies. On an 8 GB GPU that OOMs after
    the second.

    Session scope means ONE model load for the whole pytest invocation.
    The server fixture wires its engine globals to the same model
    instance, so the server's `_init_engine` short-circuits and no
    second copy gets allocated.

What's exposed:
    cached_or_skip   -- skips if TinyLlama isn't already in the HF cache.
    hf_model         -- transformers.AutoModelForCausalLM, CPU, fp32.
                        Used as the parity reference.
    my_model         -- our LlamaModel on DEVICE (CUDA if available), fp32.
    tokenizer        -- HF AutoTokenizer for TinyLlama.
    model_and_tokenizer -- (my_model, tokenizer) convenience pair for
                           scheduler/server tests that don't need HF.
    server_engine    -- pre-populates src.server.api's module globals
                        (tokenizer, scheduler, event_bus, pump thread)
                        from my_model + tokenizer, so the TestClient
                        startup is a no-op.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.model import MODEL_NAME, load_tinyllama_from_hf


def _checkpoint_is_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return bool(try_to_load_from_cache(model_name, "config.json")) and \
           bool(try_to_load_from_cache(model_name, "model.safetensors"))


@pytest.fixture(scope="session")
def cached_or_skip() -> None:
    if not _checkpoint_is_cached(MODEL_NAME):
        pytest.skip(
            f"{MODEL_NAME} not in HF cache. "
            "Run `python -m src.engine.model` once to download it, "
            "then re-run pytest."
        )


@pytest.fixture(scope="session")
def hf_model(cached_or_skip):
    """Reference HF model: CPU, fp32. Used by engine parity tests."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def my_model(cached_or_skip):
    """Our LlamaModel on DEVICE, fp32. The single engine instance."""
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tokenizer(cached_or_skip):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def model_and_tokenizer(my_model, tokenizer):
    return my_model, tokenizer


@pytest.fixture(scope="session")
def server_engine(my_model, tokenizer):
    """Pre-populate src.server.api's globals so TestClient doesn't reload.

    `_init_engine` returns early if `_scheduler` is already set, which is
    the contract we exploit: we set tokenizer, event_bus, scheduler, and
    start the pumper thread ourselves using the SHARED my_model. The
    TestClient's startup hook becomes a no-op, and no second copy of
    TinyLlama gets allocated on the GPU.
    """
    import threading

    from src.engine.events import EventBus
    from src.engine.scheduler import ContinuousBatchScheduler
    from src.server import api
    from src.server import metrics

    if api._scheduler is not None:
        yield
        return

    api._tokenizer = tokenizer
    api._event_bus = EventBus()
    api._event_bus.subscribe(api._on_bus_event)
    # Mirror _init_engine: the metrics collector is a bus subscriber.
    # Wiring it here keeps test_metrics.py exercising the real path.
    api._event_bus.subscribe(metrics.collector.on_event)
    api._scheduler = ContinuousBatchScheduler(
        my_model,
        max_batch_size=8,
        num_blocks=64,
        block_size=16,
        event_bus=api._event_bus,
        token_decoder=lambda tid: tokenizer.decode([tid], skip_special_tokens=False),
        token_emitter=api._on_token,
        enable_spec_decode=False,
        spec_decode_k=4,
        spec_decode_observer=metrics.observe_spec_decode_round,
    )
    api._pump_thread = threading.Thread(
        target=api._pumper_loop, daemon=True, name="engine-pump"
    )
    api._pump_thread.start()
    yield
