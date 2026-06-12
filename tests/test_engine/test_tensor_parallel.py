"""
Tensor-parallel (Megatron-LM style) tests, CPU-compatible via the gloo backend.

What we pin:

  1. world_size=1 is a true no-op -- TensorParallelLlamaModel is byte-identical
     to the base LlamaModel.
  2. The column-parallel -> row-parallel pair reconstructs a dense two-layer
     linear: sum_r RowParallel_r(ColumnParallel_r(x)) == fc2(fc1(x)). This is the
     Megatron identity W2(W1 x) = sum_r W2[:,r-cols] (W1[r-rows] x).
  3. TensorParallelAttention sharded across world_size=2 (summed across ranks)
     equals dense multi-head attention.
  4. The whole model sharded across world_size=2 matches the dense forward.

The "world_size=2" runs are SIMULATED in one process by instantiating the rank
shards and summing their row-parallel partials -- which is exactly what the
all-reduce does across real devices. A gloo process group is initialised for the
attention test so `all_reduce_sum` is exercised through torch.distributed (it is
the identity at group size 1; the cross-rank combine is the explicit sum). Real
multi-GPU runs one rank per process where the all-reduce does the summation; see
src/engine/tensor_parallel.py.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from src.engine.model import LlamaConfig, LlamaModel
from src.engine.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelAttention,
    TensorParallelConfig,
    all_reduce_sum,
)
from src.engine.tp_model import TensorParallelLlamaModel


def _tiny_model() -> LlamaModel:
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,       # head_dim = 16
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


# ---------------------------------------------------------------------------
# 1. world_size=1 is a no-op.
# ---------------------------------------------------------------------------


def test_world_size_1_matches_base_exactly():
    model = _tiny_model()
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (1, 7), generator=g)

    base_logits = model(ids)
    tp = TensorParallelLlamaModel(model).replace_with_tp_layers(1)
    tp_logits = tp.forward(ids)

    assert torch.equal(base_logits, tp_logits), "world_size=1 must be identical to base"


# ---------------------------------------------------------------------------
# 2. Column-parallel + row-parallel round-trip == dense two-layer linear.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4])
def test_column_then_row_matches_dense(world_size):
    torch.manual_seed(0)
    in_f, mid, out_f = 16, 24, 12       # mid divisible by 2 and 4
    fc1 = nn.Linear(in_f, mid, bias=False)     # column-parallel (shard output)
    fc2 = nn.Linear(mid, out_f, bias=False)    # row-parallel (shard input)
    x = torch.randn(3, 5, in_f)

    dense = fc2(fc1(x))

    cols = [ColumnParallelLinear.from_linear(fc1, world_size, r) for r in range(world_size)]
    rows = [RowParallelLinear.from_linear(fc2, world_size, r) for r in range(world_size)]
    # Each rank: its output-shard of fc1 feeds its input-shard of fc2; the
    # row-parallel partials sum (here explicitly; via all-reduce on real devices)
    # to the dense result.
    tp = sum(rows[r](cols[r](x)) for r in range(world_size))

    assert torch.allclose(tp, dense, atol=1e-5), "TP column->row did not reconstruct the dense linear"


def test_column_parallel_shard_is_a_slice_of_output():
    """ColumnParallelLinear rank r really holds the r-th output slice."""
    torch.manual_seed(1)
    fc = nn.Linear(8, 12, bias=False)
    x = torch.randn(2, 8)
    full = fc(x)                                   # (2, 12)
    ws = 3
    shards = [ColumnParallelLinear.from_linear(fc, ws, r)(x) for r in range(ws)]
    assert torch.allclose(torch.cat(shards, dim=-1), full, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. TensorParallelAttention (world_size=2, gloo group) == dense MHA.
# ---------------------------------------------------------------------------


@pytest.fixture
def gloo_group():
    """A world_size=1 gloo process group, so all_reduce_sum exercises the real
    torch.distributed path (identity at group size 1). Skips if unavailable."""
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed not available")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    try:
        yield
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


def test_all_reduce_sum_identity_without_group():
    # No group initialised -> identity (the in-process simulation sums itself).
    t = torch.randn(3, 4)
    out = all_reduce_sum(t.clone())
    assert torch.equal(out, t)


def test_tp_attention_matches_dense_mha(gloo_group):
    model = _tiny_model()
    attn = model.layers[0].attn                    # NQ=4, NKV=2
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 5, model.config.hidden_size, generator=g)

    # Dense reference: the base MHA's no-cache (prefill) forward.
    ref = attn(x)                                  # (B, S, H)

    # world_size=2 simulated: each rank owns 2 query heads + 1 KV head; the
    # row-parallel output partials sum to the dense attention output.
    ws = 2
    ranks = [TensorParallelAttention.from_attention(attn, ws, r) for r in range(ws)]
    tp_out = ranks[0](x)
    for r in range(1, ws):
        tp_out = tp_out + ranks[r](x)

    assert tp_out.shape == ref.shape
    assert torch.allclose(tp_out, ref, atol=1e-5), "TP attention did not match dense MHA"


# ---------------------------------------------------------------------------
# 4. Whole model sharded across world_size=2 matches the dense forward.
# ---------------------------------------------------------------------------


def test_tp_model_world_size_2_matches_base():
    model = _tiny_model()
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, 256, (1, 6), generator=g)

    base = model(ids)                              # dense no-cache forward
    tp = TensorParallelLlamaModel(model).replace_with_tp_layers(2)
    out = tp.forward(ids)

    assert out.shape == base.shape
    # TP is mathematically exact; tolerance is float reduction-order only.
    assert torch.allclose(out, base, atol=1e-4), "world_size=2 TP model diverged from dense"


def test_tp_config_validates_rank():
    TensorParallelConfig(world_size=4, rank=3)     # ok
    with pytest.raises(ValueError):
        TensorParallelConfig(world_size=2, rank=2)  # rank out of range
