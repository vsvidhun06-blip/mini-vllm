"""
Disaggregated prefill/decode tests (Mooncake-style single-process simulation).

The architecture splits generation into a PrefillWorker (computes prompt KV) and
a DecodeWorker (consumes that KV and emits tokens), connected by a transfer
queue. These tests pin the properties that make the split correct and legible:

  1. Worker independence -- the prefill worker emits a self-contained KV bundle;
     the decode worker ingests it and decodes with no further reference to the
     prefill worker. Either can be driven in isolation.
  2. KV-transfer parity -- because both workers share one model instance and the
     transfer ships post-RoPE K/V, end-to-end output is BYTE-IDENTICAL to the
     unified model.generate(). Transferring the cache must not change the math.
  3. /stats observability -- the endpoint reports the three queue depths
     (prefill_queue_depth, decode_queue_depth, transfer_buffer_size).

Everything runs on a tiny random-weight LlamaModel on CPU -- no GPU, no HF
download. (The /stats test additionally needs FastAPI's TestClient; it skips
cleanly if FastAPI isn't installed.)
"""
from __future__ import annotations

import pytest
import torch

from src.engine.disaggregated import (
    DecodeWorker,
    DisaggregatedEngine,
    PrefillWorker,
    _Request,
)
from src.engine.model import LlamaConfig, LlamaModel


def _tiny_model() -> LlamaModel:
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,       # head_dim = 16
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


def _prompt(length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, length), generator=g)


# ---------------------------------------------------------------------------
# 1. Worker independence: prefill emits a bundle, decode ingests it solo.
# ---------------------------------------------------------------------------


def test_workers_are_independent():
    model = _tiny_model()
    prefill = PrefillWorker(model)
    decode = DecodeWorker(model, num_blocks=64)

    prompt = _prompt(20)
    req = _Request("r0", prompt, max_new_tokens=5)

    # Prefill alone produces a complete, transferable bundle.
    transfer = prefill.process_prefill(req)
    assert transfer.seq_len == 20
    assert len(transfer.layers_kv) == model.config.num_hidden_layers
    for k, v in transfer.layers_kv:
        assert k.shape[1] == 20 and v.shape[1] == 20      # (1, P, NKV, D)
    assert isinstance(transfer.first_token, int)

    # The decode worker had no involvement in prefill; it ingests the bundle
    # cold and decodes from it. After receive_kv the request is active with the
    # prefill's first token already counted.
    assert not decode.has_work()
    decode.receive_kv(transfer)
    assert decode.has_work()
    st = decode.active["r0"]
    assert st.generated == [transfer.first_token]
    assert st.cache.seq_len() == 20                       # KV rebuilt to prompt len

    # Driving the decode worker emits the remaining tokens, then finishes.
    while decode.has_work():
        decode.prune_finished()
        if decode.has_work():
            decode.step()
    decode.prune_finished()
    assert len(decode.finished["r0"]) == 5


# ---------------------------------------------------------------------------
# 2. KV-transfer parity: disaggregated output == unified model.generate output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_len,max_new", [(12, 8), (20, 1), (33, 16)])
def test_parity_with_unified_generate(prompt_len, max_new):
    model = _tiny_model()
    engine = DisaggregatedEngine(model)
    prompt = _prompt(prompt_len, seed=prompt_len)

    reference = model.generate(prompt, max_new_tokens=max_new)
    disagg = engine.generate(prompt, max_new_tokens=max_new)

    assert disagg.shape == reference.shape
    assert torch.equal(disagg, reference), (
        "disaggregated transfer changed the generated tokens"
    )


def test_parity_batch_of_mixed_prompts():
    """run_batch over several requests must match per-request unified generate."""
    import asyncio

    model = _tiny_model()
    engine = DisaggregatedEngine(model)

    specs = [("a", _prompt(10, 1), 6), ("b", _prompt(25, 2), 4), ("c", _prompt(8, 3), 7)]
    requests = [_Request(rid, ids, n) for rid, ids, n in specs]

    results = asyncio.run(engine.run_batch(requests))

    for rid, ids, n in specs:
        ref = model.generate(ids, max_new_tokens=n)
        ref_new = ref[0, ids.shape[1]:].tolist()
        assert results[rid] == ref_new, f"request {rid} diverged from unified"


# ---------------------------------------------------------------------------
# 3. /stats endpoint reports the three queue depths.
# ---------------------------------------------------------------------------


def test_stats_endpoint_reports_queue_depths():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from src.server.disaggregated_api import create_app

    model = _tiny_model()
    engine = DisaggregatedEngine(model)

    # Hand-populate the three coordination structures so the depths are
    # deterministic and decoupled from any running pipeline:
    #   prefill_queue        -> 2 submitted-not-prefilled requests
    #   transfer_queue       -> 1 KV bundle in flight
    #   decode_worker.active -> 1 request actively decoding
    engine.prefill_queue.append(_Request("p1", _prompt(4), 3))
    engine.prefill_queue.append(_Request("p2", _prompt(4), 3))

    transfer = engine.prefill_worker.process_prefill(_Request("t1", _prompt(6), 3))
    engine.transfer_queue.put_nowait(transfer)

    active = engine.prefill_worker.process_prefill(_Request("d1", _prompt(6), 3))
    engine.decode_worker.receive_kv(active)

    client = fastapi_testclient.TestClient(create_app(engine=engine))
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "prefill_queue_depth": 2,
        "decode_queue_depth": 1,
        "transfer_buffer_size": 1,
    }


def test_generate_endpoint_matches_engine():
    """POST /generate (token-id prompt fallback, no tokenizer) returns the same
    tokens the engine would, end to end through the HTTP layer."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from src.server.disaggregated_api import create_app

    model = _tiny_model()
    engine = DisaggregatedEngine(model)
    prompt = _prompt(10, seed=99)
    expected = engine.generate(prompt, max_new_tokens=5)[0, 10:].tolist()

    # A fresh engine over the SAME model gives the app a clean pipeline (the one
    # above already ran). Greedy decode is deterministic, so tokens match.
    app_engine = DisaggregatedEngine(model)
    client = fastapi_testclient.TestClient(create_app(engine=app_engine))
    prompt_str = " ".join(str(int(t)) for t in prompt[0])
    resp = client.post("/generate", json={"prompt": prompt_str, "max_tokens": 5})
    assert resp.status_code == 200
    assert resp.json()["output_tokens"] == expected
