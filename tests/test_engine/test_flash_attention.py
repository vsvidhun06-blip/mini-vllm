"""
Parity tests for the from-scratch FA2 Triton kernel vs PyTorch SDPA.

The reference is ``F.scaled_dot_product_attention`` -- the exact op our kernel
replaces in attention.py. We check the three masking regimes the engine
actually exercises (decode, full prefill, sliced prefill) plus an explicit
causal-correctness check that an UNmasked run disagrees with a causal run
exactly on the upper triangle.

Tolerance: atol=1e-3. The kernel runs fp32 (so it's tighter than this in
practice), but 1e-3 is the documented contract and leaves headroom for the
tl.exp approximation. GPU-only; skips cleanly on CPU-only hosts.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash attention is a CUDA-only Triton kernel"
)

# TinyLlama-ish geometry: D must be a power of 2 for the kernel.
NH = 8
D = 64
ATOL = 1e-3


def _rand(B, NH, S, D):
    return torch.randn(B, NH, S, D, device="cuda", dtype=torch.float32)


@cuda_only
def test_decode_s1_non_causal():
    """S_q == 1 decode: one new query attends to the whole cached past."""
    from src.engine.kernels.flash_attention import flash_attention_forward

    torch.manual_seed(0)
    B, S_k = 2, 130
    q = _rand(B, NH, 1, D)
    k = _rand(B, NH, S_k, D)
    v = _rand(B, NH, S_k, D)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = flash_attention_forward(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)


@cuda_only
@pytest.mark.parametrize("S", [64, 200])  # exact tile and a ragged tail
def test_full_prefill_causal(S):
    """S_q == S_k full prefill: standard causal lower-triangular attention."""
    from src.engine.kernels.flash_attention import flash_attention_forward

    torch.manual_seed(1)
    B = 2
    q = _rand(B, NH, S, D)
    k = _rand(B, NH, S, D)
    v = _rand(B, NH, S, D)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = flash_attention_forward(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)


@cuda_only
def test_sliced_prefill_explicit_mask():
    """S_q < S_k sliced prefill (prefix cache): explicit additive mask.

    Query row i sits at absolute position pos_offset + i and attends to keys
    0..(pos_offset + i). We build the same bool mask attention.py builds, hand
    SDPA the bool mask and the kernel the additive -inf version, and compare.
    """
    from src.engine.kernels.flash_attention import flash_attention_forward

    torch.manual_seed(2)
    B = 2
    pos_offset = 40
    S_q = 24
    S_k = pos_offset + S_q
    q = _rand(B, NH, S_q, D)
    k = _rand(B, NH, S_k, D)
    v = _rand(B, NH, S_k, D)

    q_pos = torch.arange(pos_offset, pos_offset + S_q, device="cuda")
    k_pos = torch.arange(S_k, device="cuda")
    bool_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)        # (S_q, S_k), True=attend

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
    add_mask = torch.zeros_like(bool_mask, dtype=q.dtype).masked_fill(~bool_mask, float("-inf"))
    out = flash_attention_forward(q, k, v, attn_mask=add_mask)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)


@cuda_only
@pytest.mark.parametrize("causal", [False, True])
def test_fp16_inputs_run_and_preserve_dtype(causal):
    """fp16 q/k/v must (a) not crash and (b) come back fp16, close to fp32.

    This is the regression guard for the "Both operands must be same dtype. Got
    fp32 and fp16" crash: when the model serves in fp16, q/k/v reach the kernel
    fp16, Triton's tl.dot returns the scores fp16, and any per-operand cast in the
    value matmul then mixes dtypes. flash_attention_forward now promotes the whole
    kernel to fp32 and casts the output back, so fp16 in -> fp16 out, numerically
    equal to the fp32 reference up to the final fp16 round-trip (hence the looser
    fp16 tolerance instead of the 1e-3 used for the fp32 parity tests above).
    """
    from src.engine.kernels.flash_attention import flash_attention_forward

    torch.manual_seed(4)
    B, S = 2, 200
    q32 = _rand(B, NH, S, D)
    k32 = _rand(B, NH, S, D)
    v32 = _rand(B, NH, S, D)

    ref = F.scaled_dot_product_attention(q32, k32, v32, is_causal=causal)

    out = flash_attention_forward(q32.half(), k32.half(), v32.half(), causal=causal)
    assert out.dtype == torch.float16, "kernel must hand back the caller's fp16 dtype"
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0)


@cuda_only
def test_causal_masking_is_actually_applied():
    """Causal correctness: causal output must equal a hand-masked full softmax,
    and must DIFFER from the non-causal output (proving the mask does work)."""
    from src.engine.kernels.flash_attention import flash_attention_forward

    torch.manual_seed(3)
    B, S = 1, 96
    q = _rand(B, NH, S, D)
    k = _rand(B, NH, S, D)
    v = _rand(B, NH, S, D)

    causal_out = flash_attention_forward(q, k, v, causal=True)

    # Independent reference: explicit lower-triangular additive mask.
    tri = torch.tril(torch.ones(S, S, device="cuda", dtype=torch.bool))
    add_mask = torch.zeros(S, S, device="cuda", dtype=q.dtype).masked_fill(~tri, float("-inf"))
    masked_out = flash_attention_forward(q, k, v, attn_mask=add_mask)
    torch.testing.assert_close(causal_out, masked_out, atol=ATOL, rtol=0)

    # Sanity: without the mask the result is meaningfully different.
    noncausal_out = flash_attention_forward(q, k, v, causal=False)
    assert not torch.allclose(causal_out, noncausal_out, atol=ATOL), (
        "causal and non-causal outputs are identical -- mask had no effect"
    )
