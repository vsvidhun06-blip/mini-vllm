"""
INT8 (W8A8) quantization primitives + a native-PyTorch int8 GEMM.

WHY QUANTIZE
------------
The TinyLlama weights are 1.1B fp32 numbers (~4.4 GB) / fp16 (~2.2 GB). Most
of inference is memory-bound: the GPU spends its time hauling weights from HBM,
not doing arithmetic. If we store weights as int8 we move 1/4 (vs fp32) or 1/2
(vs fp16) the bytes, and int8 tensor cores do the matmul at higher throughput.

W8A8 = 8-bit Weights, 8-bit Activations. Both operands of the projection
matmuls are int8; the accumulation is int32 (no overflow for K up to ~2^17 with
int8*int8), and we rescale back to floating point at the end.

PER-TENSOR SYMMETRIC QUANTIZATION
---------------------------------
The simplest scheme, and the one asked for here:

    scale = max(|x|) / 127
    q     = round(x / scale)        clamped to [-127, 127], stored as int8
    x ~=  q * scale                 (dequantization)

"Symmetric" means zero maps to zero (no zero-point offset), so a single scalar
``scale`` fully describes the mapping. "Per-tensor" means ONE scale for the
whole tensor (as opposed to per-row / per-channel, which is more accurate but
needs a vector of scales and complicates the GEMM rescale). 127 (not 128) keeps
the range symmetric and avoids the -128 corner that has no positive twin.

THE GEMM RESCALE
----------------
For a linear layer  y = x @ W^T  with x quantized at scale_a and W at scale_b:

    x ~= q_a * scale_a,   W ~= q_b * scale_b
    y ~= (q_a @ q_b^T) * scale_a * scale_b

So we compute the int8*int8 product into int32, multiply by the two scalar
scales, and cast to fp16.

WHY torch._int_mm AND NOT A TRITON KERNEL
-----------------------------------------
The original implementation here was a hand-written Triton int8 GEMM using
``tl.dot`` with int8 inputs. That fails to compile on Turing GPUs (T4 / sm_75):
Triton's ``TritonGPUAccelerateMatmul`` pass has no int8 MMA lowering for sm_75,
so the int8 ``tl.dot`` errors out. (It works on Ampere+/Ada, but T4 is the most
common free-tier GPU, so "works only on Ampere+" is a poor default.)

``torch._int_mm`` is PyTorch's native int8 matmul. It dispatches to cuBLAS /
cuBLASLt int8 IMMA, which IS supported on sm_75, so it runs on T4. We use it as
the compute backend and keep everything around it (quantize/dequantize,
``QuantizedLinear``, ``QuantizedMultiHeadAttention``) unchanged.

``torch._int_mm`` has hard shape constraints on CUDA, though:
    * the M dimension must be > 16,
    * K and N must both be multiples of 8.
TinyLlama's projection K/N (2048, 256, ...) are all multiples of 8, and prefill
has a large M (the prompt length), so prefill takes the fast int8 path. DECODE
has M == 1, which violates M > 16 -- there we fall back to dequantizing the
(still int8-stored) weight and doing an fp matmul. Either way the weights stay
4x smaller in memory; only the GEMM compute path differs. M == 1 is memory-bound
anyway, so dequant-on-read costs little and the int8 storage win is preserved.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# torch._int_mm constraints on CUDA: M must exceed this, and K/N must be
# multiples of INT_MM_ALIGN. Below these we use the dequant fallback.
_INT_MM_MIN_M = 16
_INT_MM_ALIGN = 8


def quantize_tensor(
    x: torch.Tensor,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric quantization.

    Args:
        x: the float tensor to quantize.
        dtype: integer storage dtype (int8 -> range [-127, 127]).

    Returns:
        (q, scale) where ``q`` is ``x`` quantized to ``dtype`` and ``scale`` is
        a 0-dim float tensor such that ``q * scale`` approximates ``x``.
    """
    # 127 is the largest magnitude an int8 should take (we never emit -128 so
    # the mapping stays symmetric).
    qmax = 127.0
    amax = x.abs().max()
    scale = amax / qmax
    # Guard the all-zeros tensor: scale 0 would divide by zero. A scale of 1.0
    # is harmless because every value is 0 anyway.
    if scale == 0:
        scale = torch.ones_like(scale)
    q = torch.round(x / scale).clamp(-qmax, qmax).to(dtype)
    # Keep scale as a 0-dim tensor on x's device for clean storage/broadcast.
    return q, scale.reshape(())


def dequantize_tensor(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_tensor`: ``x ~= x_int8 * scale`` in fp32."""
    return x_int8.to(torch.float32) * scale


def quantized_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale_w: torch.Tensor,
) -> torch.Tensor:
    """W8A8 linear:  y = x @ W^T, with W pre-quantized and x quantized at runtime.

    Args:
        x: (..., K) float activations. Quantized per-tensor on the fly.
        weight_int8: (N, K) int8 -- the quantized nn.Linear weight (out, in),
            same layout as ``nn.Linear.weight``.
        scale_w: 0-dim float tensor, the weight's per-tensor scale.

    Returns:
        (..., N) fp16 tensor. Callers that want a different dtype should cast.
    """
    assert x.is_cuda, "quantized_linear is CUDA-only; use a dequant fallback on CPU"
    *batch, K = x.shape
    N, Kw = weight_int8.shape
    assert Kw == K, f"weight in_features {Kw} != activation features {K}"

    x2d = x.reshape(-1, K)
    M = x2d.shape[0]

    # torch._int_mm only accepts M > 16 and K, N multiples of 8 (see module
    # docstring). When that holds -- prefill, where M is the prompt length --
    # take the true int8 path.
    if M > _INT_MM_MIN_M and K % _INT_MM_ALIGN == 0 and N % _INT_MM_ALIGN == 0:
        # Runtime activation quantization (per-tensor symmetric, same scheme as W).
        x_int8, scale_x = quantize_tensor(x2d)
        # _int_mm computes A @ B with A:(M,K) int8, B:(K,N) int8 -> (M,N) int32.
        # The weight is stored (N,K); .t() gives the (K,N) column-major view
        # cuBLAS wants for B, so no copy is needed. x_int8 is freshly produced
        # by quantize_tensor and is therefore already contiguous (row-major).
        acc_int32 = torch._int_mm(x_int8, weight_int8.t())
        out = acc_int32.to(torch.float32) * (scale_x * scale_w)
        out = out.to(torch.float16)
    else:
        # Small-M / odd-shape fallback (notably decode, M == 1): dequantize the
        # int8 weight and do an fp matmul on the full-precision activation.
        # Weights stay int8 in memory; only this GEMM runs in fp.
        w = dequantize_tensor(weight_int8, scale_w).to(x2d.dtype)
        out = F.linear(x2d, w).to(torch.float16)

    return out.reshape(*batch, N)
