"""
Parity tests for the fused Triton RoPE kernel.

The contract is simple and strict: ``fused_rope`` (single fused CUDA launch)
must produce byte-for-byte-close output to the reference ``apply_rope`` (the
6-launch PyTorch path) for the shapes the engine actually uses. If these drift
the kernel is wrong, full stop.

These tests REQUIRE a GPU -- ``fused_rope`` is a CUDA-only kernel. On a CPU-only
host (CI without a GPU, the correctness box) they skip cleanly rather than fail.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.attention import apply_rope, build_rope_cache

# Triton/CUDA gate. Skips the whole module on CPU-only hosts.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused_rope is a CUDA-only Triton kernel"
)


@cuda_only
@pytest.mark.parametrize(
    "shape",
    [
        (2, 32, 16, 64),   # 32 query heads (TinyLlama Q layout)
        (4, 4, 16, 64),    # 4 KV heads (TinyLlama K layout)
    ],
)
def test_fused_rope_matches_apply_rope(shape):
    """fused_rope == apply_rope within atol=1e-5 for the engine's Q/K shapes."""
    from src.engine.kernels.rope import fused_rope

    B, H, S, D = shape
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    # cos/sin sliced to the sequence length, on the same device.
    cos, sin = build_rope_cache(D, S)
    cos = cos.to("cuda")
    sin = sin.to("cuda")

    # Reference: apply_rope rotates a (q, k) pair; feed x as both and take the
    # q output. Identical math to rotating x alone.
    ref, _ = apply_rope(x, x, cos, sin)
    out = fused_rope(x, cos, sin)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
