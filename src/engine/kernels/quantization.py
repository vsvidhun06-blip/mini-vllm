"""
INT8 (W8A8) quantization primitives + a Triton int8 GEMM.

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

So the kernel accumulates the int8*int8 products into int32, then multiplies
the int32 result by the two scalar scales and casts to fp16 on the way out.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# GEMM tile sizes (per the phase spec). 64x64 output tile, 32-deep K steps.
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


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


@triton.jit
def int8_matmul_kernel(
    A, B, C,                       # A:(M,K) int8, B:(K,N) int8, C:(M,N) fp16
    M, N, K,
    scale_a, scale_b,              # fp32 scalars: activation + weight scales
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program computes a BLOCK_M x BLOCK_N output tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # int32 accumulator -- int8*int8 sums can't overflow int32 for K up to ~2^17.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] + k0 < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=((offs_k[:, None] + k0) < K) & (offs_n[None, :] < N),
            other=0,
        )
        # int8 @ int8 -> int32 accumulation on the tensor cores.
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Rescale int32 -> fp32 with the two per-tensor scales, store as fp16.
    c = acc.to(tl.float32) * scale_a * scale_b
    c = c.to(tl.float16)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
        (..., N) fp16 tensor (the kernel rescales to fp16, per spec). Callers
        that want a different dtype should cast.
    """
    assert x.is_cuda, "quantized_linear is CUDA-only; use a dequant fallback on CPU"
    *batch, K = x.shape
    N, Kw = weight_int8.shape
    assert Kw == K, f"weight in_features {Kw} != activation features {K}"

    x2d = x.reshape(-1, K)
    M = x2d.shape[0]

    # Runtime activation quantization (per-tensor symmetric, same scheme as W).
    x_int8, scale_x = quantize_tensor(x2d)

    # The kernel wants B as (K, N); the stored weight is (N, K), so transpose.
    # This is a strided view -- the kernel reads it through explicit strides.
    B = weight_int8.t()

    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    int8_matmul_kernel[grid](
        x_int8, B, out,
        M, N, K,
        float(scale_x), float(scale_w),
        x_int8.stride(0), x_int8.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out.reshape(*batch, N)
