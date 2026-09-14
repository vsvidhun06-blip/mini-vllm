"""
Tests for INT8 (W8A8) quantization: the primitives, the int8 GEMM, the
quantized attention block, and the memory win.

Tolerance philosophy:
  * quantize/dequantize roundtrip is bounded EXACTLY by the scale (one LSB).
  * the int8 GEMM must equal the dequantized-operands fp matmul (that's the
    math it implements) to fp16 precision.
  * the quantized attention block must track the fp block to within atol=0.1 --
    the "quantization tolerance" the phase asks for. Attention output is a
    softmax-weighted average of value vectors, so it's O(1) and 0.1 is a
    meaningful bound.

All GPU-only (the kernels are CUDA Triton); skips cleanly on CPU-only hosts.
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="quantization kernels are CUDA-only (Triton)"
)


@cuda_only
def test_quantize_dequantize_roundtrip():
    """Roundtrip error is bounded by one quantization step (scale/2 in theory,
    <= scale with rounding) and zero maps to zero (symmetric)."""
    from src.engine.kernels.quantization import dequantize_tensor, quantize_tensor

    torch.manual_seed(0)
    x = torch.randn(512, 512, device="cuda") * 3.0
    q, scale = quantize_tensor(x)

    assert q.dtype == torch.int8
    assert q.abs().max().item() <= 127
    x_hat = dequantize_tensor(q, scale).to(x.dtype)
    # Worst-case rounding error is half a step; allow a full step for safety.
    assert (x - x_hat).abs().max().item() <= scale.item() + 1e-6


@cuda_only
@pytest.mark.parametrize("M,K,N", [(64, 256, 128), (130, 2048, 512)])
def test_int8_gemm_matches_dequant_matmul(M, K, N):
    """The int8 GEMM must equal (dequant A) @ (dequant B)^T -- the exact math it
    approximates -- to fp16 precision. This is kernel correctness."""
    from src.engine.kernels.quantization import quantize_tensor, quantized_linear

    torch.manual_seed(1)
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")  # (out, in), like nn.Linear.weight

    w_int8, scale_w = quantize_tensor(w)
    out = quantized_linear(x, w_int8, scale_w).to(torch.float32)

    # Reference: quantize x the SAME way the wrapper does internally, then do
    # the matmul in float on the dequantized operands.
    x_int8, scale_x = quantize_tensor(x)
    ref = (x_int8.float() * scale_x) @ (w_int8.float() * scale_w).t()

    torch.testing.assert_close(out, ref, atol=0.1, rtol=0)


def _small_mha(**overrides):
    """Build a small attention block with realistically-scaled (std=0.02)
    projection weights so its output is O(1) and the 0.1 tolerance is meaningful.
    head_dim = 64 (power of 2) is required by the flash-attention kernel."""
    from src.engine.attention import MultiHeadAttention

    kw = dict(hidden_size=256, num_heads=4, num_kv_heads=2, max_seq_len=64)
    kw.update(overrides)
    mha = MultiHeadAttention(**kw)
    with torch.no_grad():
        for proj in (mha.q_proj, mha.k_proj, mha.v_proj, mha.o_proj):
            proj.weight.normal_(mean=0.0, std=0.02)
    return mha.cuda().eval()


@cuda_only
def test_quantized_attention_parity():
    """QuantizedMultiHeadAttention output tracks the fp block within atol=0.1."""
    from src.engine.kernels.quant_attention import QuantizedMultiHeadAttention

    torch.manual_seed(2)
    mha = _small_mha()
    qmha = QuantizedMultiHeadAttention.from_float(mha).cuda().eval()

    x = torch.randn(2, 16, 256, device="cuda")
    with torch.no_grad():
        ref = mha(x)        # fp32 reference (CUDA flash path)
        out = qmha(x)       # int8 projections

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)


@cuda_only
def test_quantized_weights_are_4x_smaller():
    """The q/k/v/o projection weights shrink ~4x (fp32 -> int8)."""
    from src.engine.kernels.quant_attention import QuantizedMultiHeadAttention

    mha = _small_mha()
    qmha = QuantizedMultiHeadAttention.from_float(mha)

    def fp_bytes(m):
        return sum(
            p.weight.numel() * p.weight.element_size()
            for p in (m.q_proj, m.k_proj, m.v_proj, m.o_proj)
        )

    def int8_bytes(m):
        total = 0
        for p in (m.q_proj, m.k_proj, m.v_proj, m.o_proj):
            total += p.weight_int8.numel() * p.weight_int8.element_size()
            total += p.weight_scale.numel() * p.weight_scale.element_size()
        return total

    ratio = fp_bytes(mha) / int8_bytes(qmha)
    # fp32 is 4 bytes, int8 is 1 byte; the only overhead is one scalar scale
    # per projection, which is negligible against a (256x256) weight.
    assert ratio >= 3.9, f"expected ~4x weight shrink, got {ratio:.2f}x"


@cuda_only
def test_model_quantize_swaps_blocks(cached_or_skip):
    """End-to-end: LlamaModel.quantize() replaces every attention block and the
    quantized model still produces sane next-token predictions vs the fp model.

    Loads its OWN model copy so it never mutates the shared session fixture.
    """
    from src.engine.kernels.quant_attention import QuantizedMultiHeadAttention
    from src.engine.model import MODEL_NAME, load_tinyllama_from_hf

    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()

    ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], device="cuda")  # arbitrary prompt
    with torch.no_grad():
        fp_logits = model(ids)[:, -1, :]
    fp_top1 = fp_logits.argmax(dim=-1)

    model.quantize()
    assert all(
        isinstance(b.attn, QuantizedMultiHeadAttention) for b in model.layers
    ), "not all attention blocks were quantized"

    with torch.no_grad():
        q_logits = model(ids)[:, -1, :]
    q_top1 = q_logits.argmax(dim=-1)

    # The quantized model should agree with the fp model on the greedy token,
    # and its top-5 should overlap heavily -- "within acceptable range".
    assert torch.equal(fp_top1, q_top1), "quantized model changed the greedy token"
    fp_top5 = set(fp_logits.topk(5, dim=-1).indices[0].tolist())
    q_top5 = set(q_logits.topk(5, dim=-1).indices[0].tolist())
    assert len(fp_top5 & q_top5) >= 4, f"top-5 drifted: {fp_top5} vs {q_top5}"
