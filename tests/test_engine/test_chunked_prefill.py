"""
Chunked-prefill scheduling tests.

Chunked prefill splits a long prompt across several scheduler iterations so it
can't monopolise the GPU and stall decode requests (vLLM v2 / SGLang). These
tests pin the four properties that matter:

  1. A 512-token prompt at chunk_size=256 prefills in exactly 2 iterations.
  2. Decode requests run BETWEEN a long prompt's prefill chunks (the whole
     point -- no head-of-line stall).
  3. Chunked prefill yields the SAME KV cache (and the same generated tokens)
     as one-shot full prefill. Splitting must be a pure scheduling change.
  4. PREFILL_CHUNK_START / _DONE events carry the right chunk_index /
     tokens_in_chunk / progress fields.

These run on a tiny randomly-initialised LlamaModel on CPU -- no HF download,
no GPU. Weights are random but FIXED per test (one model instance, reused), so
"chunked vs full" comparisons are apples-to-apples; absolute values are
irrelevant, only that the two paths agree.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.events import EventBus
from src.engine.kv_cache import PagedKVCache, PagedRequestCache
from src.engine.model import LlamaConfig, LlamaModel
from src.engine.scheduler import ContinuousBatchScheduler

BLOCK_SIZE = 16


def _tiny_model() -> LlamaModel:
    """A small, random-weight LlamaModel. head_dim=16 is a power of 2 (so the
    same model would also work on the CUDA flash path); 3 layers keeps forward
    passes cheap even for a 512-token prompt."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,      # head_dim = 128/8 = 16
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaModel(config)
    model.eval()
    return model


def _random_prompt(n: int) -> torch.Tensor:
    """A (1, n) prompt of random token ids."""
    g = torch.Generator().manual_seed(99)
    return torch.randint(0, 256, (1, n), generator=g)


def _collect_events() -> tuple[EventBus, list]:
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    return bus, captured


def _events_of(captured, event_type, request_id=None):
    out = []
    for e in captured:
        if e.event_type != event_type:
            continue
        if request_id is not None and e.payload.get("request_id") != request_id:
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 1. A 512-token prompt at chunk_size=256 takes exactly two prefill iterations.
# ---------------------------------------------------------------------------


def test_512_prompt_takes_exactly_two_chunks():
    model = _tiny_model()
    bus, captured = _collect_events()
    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=128, chunk_size=256, event_bus=bus,
    )
    sched.add_request("long", _random_prompt(512), max_new_tokens=3, eos_token_id=None)

    # Drive only until the prefill of "long" finishes; we don't need full decode.
    while sched.has_work():
        sched.step()
        if _events_of(captured, "prefill_done", "long"):
            break

    chunk_starts = _events_of(captured, "prefill_chunk_start", "long")
    assert len(chunk_starts) == 2, (
        f"expected exactly 2 prefill chunks for a 512-token prompt at "
        f"chunk_size=256, got {len(chunk_starts)}"
    )
    assert [e.payload["tokens_in_chunk"] for e in chunk_starts] == [256, 256]
    assert [e.payload["chunk_index"] for e in chunk_starts] == [0, 1]


# ---------------------------------------------------------------------------
# 2. Decode requests are scheduled between a long prompt's prefill chunks.
# ---------------------------------------------------------------------------


def test_decode_runs_between_prefill_chunks():
    model = _tiny_model()
    bus, captured = _collect_events()
    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=128, chunk_size=256, event_bus=bus,
    )
    # A: short prompt -> prefills in one chunk, then decodes for many steps.
    # B: long prompt  -> prefills across several chunks (budget shared with A's
    #    decode token each step), so B is still chunking while A decodes.
    sched.add_request("short", _random_prompt(4), max_new_tokens=8, eos_token_id=None)
    sched.add_request("long", _random_prompt(512), max_new_tokens=8, eos_token_id=None)

    while sched.has_work():
        sched.step()
        # Stop once both have moved past prefill to keep the test short.
        if _events_of(captured, "prefill_done", "long") and \
           len(_events_of(captured, "decode_step")) > 0:
            # keep going a touch so long's prefill_done is captured
            if _events_of(captured, "prefill_done", "long"):
                break

    # Find emission-order index of long's prefill completion.
    order = {id(e): i for i, e in enumerate(captured)}
    long_done = _events_of(captured, "prefill_done", "long")
    assert long_done, "long prompt never finished prefill"
    long_done_idx = order[id(long_done[0])]

    # The long prompt must have taken MORE than one chunk (i.e. it was actually
    # split) -- otherwise there's no "between chunks" to speak of.
    assert len(_events_of(captured, "prefill_chunk_start", "long")) >= 2

    # "short" must have emitted at least one decode token BEFORE "long"
    # finished prefilling -- proof that decode interleaved with the chunks.
    short_decode_before = False
    for e in captured[:long_done_idx]:
        if e.event_type == "decode_step":
            if any(row["request_id"] == "short" for row in e.payload["batch"]):
                short_decode_before = True
                break
    assert short_decode_before, (
        "no decode token from 'short' was scheduled before 'long' finished "
        "prefill -- decode was stalled behind the long prefill"
    )


# ---------------------------------------------------------------------------
# 3. Output parity: chunked prefill == full prefill (KV cache AND tokens).
# ---------------------------------------------------------------------------


def _run_prefill_capture_kv(model, prompt_ids, chunk):
    """Prefill `prompt_ids` through a fresh paged cache, splitting into
    `chunk`-token forwards. Return per-layer (K, V) gathered from the cache
    plus the next-token logits from the final forward."""
    cfg = model.config
    prompt_len = prompt_ids.shape[1]
    n_blocks = (prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=n_blocks + 4,
        block_size=BLOCK_SIZE,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    pool.admit_request("r", prefill_blocks_needed=n_blocks, total_blocks_needed=n_blocks)
    cache = PagedRequestCache(pool, "r", num_layers=cfg.num_hidden_layers)
    last_logits = None
    with torch.no_grad():
        for start in range(0, prompt_len, chunk):
            last_logits = model(prompt_ids[:, start:start + chunk], kv_cache=cache)
    kv = [cache.get(layer) for layer in range(cfg.num_hidden_layers)]
    return kv, last_logits


def test_kv_cache_parity_chunked_vs_full():
    model = _tiny_model()
    prompt = _random_prompt(512)

    full_kv, full_logits = _run_prefill_capture_kv(model, prompt, chunk=512)
    chunk_kv, chunk_logits = _run_prefill_capture_kv(model, prompt, chunk=256)

    # Every layer's cached K and V must match (chunking is a pure
    # re-association of the same fp32 math, so ~1e-4 is generous).
    for layer, ((kf, vf), (kc, vc)) in enumerate(zip(full_kv, chunk_kv)):
        assert kf.shape == kc.shape == (1, 512, model.config.num_key_value_heads,
                                        model.config.hidden_size // model.config.num_attention_heads)
        torch.testing.assert_close(kc, kf, atol=1e-4, rtol=1e-4,
                                   msg=f"K mismatch at layer {layer}")
        torch.testing.assert_close(vc, vf, atol=1e-4, rtol=1e-4,
                                   msg=f"V mismatch at layer {layer}")

    # And the first generated token (argmax of the last position) must agree.
    assert int(full_logits[0, -1].argmax()) == int(chunk_logits[0, -1].argmax())


def test_output_parity_chunked_vs_full_scheduler():
    """End-to-end: the generated token sequence is identical whether the
    prompt is chunk-prefilled or full-prefilled."""
    model = _tiny_model()
    prompt = _random_prompt(300)  # > 256 so chunked actually splits

    def run(chunk_size):
        sched = ContinuousBatchScheduler(
            model, max_batch_size=2, num_blocks=128, chunk_size=chunk_size,
        )
        sched.add_request("r", prompt, max_new_tokens=12, eos_token_id=None)
        out: list[int] = []
        while sched.has_work():
            for rid, tok in sched.step():
                out.append(tok)
        return out

    chunked = run(256)            # splits into 256 + 44
    full = run(100_000)           # single-shot full prefill
    assert chunked == full, (
        f"chunked prefill diverged from full prefill:\n"
        f"  chunked: {chunked}\n  full:    {full}"
    )


# ---------------------------------------------------------------------------
# 4. PREFILL_CHUNK events are emitted with correct fields.
# ---------------------------------------------------------------------------


def test_prefill_chunk_events_fields():
    model = _tiny_model()
    bus, captured = _collect_events()
    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=128, chunk_size=256, event_bus=bus,
    )
    sched.add_request("long", _random_prompt(512), max_new_tokens=2, eos_token_id=None)

    while sched.has_work():
        sched.step()
        if _events_of(captured, "prefill_done", "long"):
            break

    starts = _events_of(captured, "prefill_chunk_start", "long")
    dones = _events_of(captured, "prefill_chunk_done", "long")

    # Two chunks, indexed 0 and 1.
    assert len(starts) == 2 and len(dones) == 2

    # chunk_start carries the PRE-chunk progress; chunk_done the POST-chunk.
    assert [e.payload["chunk_index"] for e in starts] == [0, 1]
    assert [e.payload["prefilled_so_far"] for e in starts] == [0, 256]
    assert [e.payload["tokens_in_chunk"] for e in starts] == [256, 256]

    assert [e.payload["chunk_index"] for e in dones] == [0, 1]
    assert [e.payload["prefilled_so_far"] for e in dones] == [256, 512]
    assert all(e.payload["prompt_len"] == 512 for e in starts + dones)

    # Legacy bracketing events still fire exactly once each (metrics + the
    # events test depend on them).
    assert len(_events_of(captured, "prefill_started", "long")) == 1
    assert len(_events_of(captured, "prefill_done", "long")) == 1
