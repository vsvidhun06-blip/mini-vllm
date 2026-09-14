"""
Fused Rotary Position Embedding (RoPE) Triton kernel.

WHY THIS EXISTS
---------------
The reference PyTorch implementation in ``src/engine/attention.py``
(``apply_rope`` / ``_rotate_half``) computes the rotation as:

    q_rot = q * cos + rotate_half(q) * sin

where ``rotate_half([a, b]) = [-b, a]``. Written in PyTorch primitives that
expands into roughly SIX separate CUDA kernel launches *per tensor*:

    1. q * cos                          (elementwise mul)
    2. slice q into halves a, b         (two views -> a cat)
    3. cat([-b, a])                     (negate + concatenate = extra kernel)
    4. rotate_half(q) * sin             (elementwise mul)
    5. (q*cos) + (...)                  (elementwise add)

and we do that twice (once for Q, once for K). Every launch reads the whole
tensor from HBM, does a trivial amount of arithmetic, and writes it back.
RoPE is utterly memory-bound, so the launch overhead and the redundant
round-trips to global memory dominate.

This kernel collapses all of that into ONE launch per tensor. Each program
loads a single token's head vector once, does the rotation in registers, and
writes it back once. No intermediate ``rotate_half`` tensor is ever
materialized in HBM.

LAYOUT (must match HF / the reference exactly)
----------------------------------------------
We use HF "GPT-J style" split-halves RoPE (NOT interleaved). For a head
vector of dimension D we split it into two contiguous halves:

    a = x[..., :D/2]    (the "first half")
    b = x[..., D/2:]    (the "second half")

and treat ``(a[i], b[i])`` as the 2D vector rotated by angle theta_i:

    out_first  = a * cos - b * sin
    out_second = a * sin + b * cos

The cos/sin tables built by ``build_rope_cache`` are *duplicated* across the
two halves (``cat([freqs, freqs])``), so cos[:D/2] == cos[D/2:]. This kernel
only ever reads the first-half segment of cos/sin (length D/2) and applies it
to both output halves -- which is mathematically identical to the reference
that multiplies the full duplicated table, but loads half as much trig data.

cos/sin SHAPES SUPPORTED
------------------------
The same kernel serves two call sites in attention.py, which differ only in
whether the rotation angle varies across the batch axis:

  * (S, D)    -- the prefill / single-cache forward. Every batch row shares
                 the same per-position angles. We broadcast across batch by
                 passing a batch stride of 0.
  * (B, S, D) -- batched continuous decode, where each request sits at its
                 own absolute position, so the angle differs per batch row.

Both are handled by the *same* kernel: the wrapper just sets the cos/sin
batch stride to 0 in the (S, D) case. The grid is always
(batch, head, seq_pos) and each program owns exactly one token's head vector.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rope_forward_kernel(
    x_ptr,            # *input*  : (B, H, S, D)
    cos_ptr,          # *cos*    : (B_or_1, S, D) -- see batch-stride note
    sin_ptr,          # *sin*    : same layout as cos
    out_ptr,          # *output* : (B, H, S, D), same strides as x's logical shape
    # --- x strides (in elements) ---
    x_sb, x_sh, x_ss, x_sd,
    # --- out strides ---
    o_sb, o_sh, o_ss, o_sd,
    # --- cos strides (cos_sb == 0 means "broadcast over batch") ---
    cos_sb, cos_ss, cos_sd,
    # --- sin strides ---
    sin_sb, sin_ss, sin_sd,
    HALF: tl.constexpr,   # D // 2
    BLOCK: tl.constexpr,  # next_power_of_2(HALF) -- vector width per program
):
    # One program == one (batch, head, seq_pos) token vector.
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Lane index into the half-width vector, masked off above HALF so a
    # non-power-of-2 head dim (e.g. HALF=32 is fine, but be safe) never reads
    # or writes out of bounds.
    lane = tl.arange(0, BLOCK)
    mask = lane < HALF

    # Base pointers for THIS token's row in x / out.
    x_row = x_ptr + b * x_sb + h * x_sh + s * x_ss
    o_row = out_ptr + b * o_sb + h * o_sh + s * o_ss

    # Load the two halves. a = first half, b_ = second half.
    a = tl.load(x_row + lane * x_sd, mask=mask, other=0.0)
    b_ = tl.load(x_row + (HALF + lane) * x_sd, mask=mask, other=0.0)

    # cos/sin for this (batch, seq) position. cos_sb is 0 in the broadcast
    # (S, D) case, so all batch rows read the same angles. We only read the
    # first-half segment because the table is duplicated across halves.
    trig_row_cos = cos_ptr + b * cos_sb + s * cos_ss
    trig_row_sin = sin_ptr + b * sin_sb + s * sin_ss
    cos = tl.load(trig_row_cos + lane * cos_sd, mask=mask, other=0.0)
    sin = tl.load(trig_row_sin + lane * sin_sd, mask=mask, other=0.0)

    # The 2D rotation, computed entirely in registers:
    #   [cos -sin] [a]   [a*cos - b*sin]   <- first half of output
    #   [sin  cos] [b] = [a*sin + b*cos]   <- second half of output
    out_first = a * cos - b_ * sin
    out_second = a * sin + b_ * cos

    tl.store(o_row + lane * o_sd, out_first, mask=mask)
    tl.store(o_row + (HALF + lane) * o_sd, out_second, mask=mask)


def fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply HF split-halves RoPE to one tensor with a single fused kernel.

    Args:
        x: (B, H, S, D) -- Q or K already in (batch, head, seq, dim) layout.
            May be non-contiguous (it usually is, coming straight off a
            ``.transpose(1, 2)``); we pass explicit strides so that's fine.
        cos: rotation cosines. Either:
                * (S, D)    -- shared across the batch (prefill / single decode)
                * (B, S, D) -- per-batch-row angles (batched continuous decode)
        sin: rotation sines, same shape as ``cos``.

    Returns:
        A new (B, H, S, D) tensor with the same dtype/device as ``x``.
    """
    # --- shape validation ------------------------------------------------
    assert x.dim() == 4, f"x must be (B, H, S, D), got {tuple(x.shape)}"
    B, H, S, D = x.shape
    assert D % 2 == 0, f"head_dim must be even for RoPE, got {D}"
    assert cos.shape == sin.shape, (
        f"cos {tuple(cos.shape)} and sin {tuple(sin.shape)} must match"
    )
    assert x.is_cuda, "fused_rope is a CUDA kernel; use apply_rope on CPU"
    assert cos.is_cuda and sin.is_cuda, "cos/sin must be on the same CUDA device"

    # Resolve the cos/sin batch stride. A stride of 0 makes every batch row
    # read the SAME angles -- exactly the (S, D) broadcast semantics of the
    # reference apply_rope. For the (B, S, D) layout we use the real stride.
    if cos.dim() == 2:
        assert cos.shape == (S, D), (
            f"2D cos must be (S={S}, D={D}), got {tuple(cos.shape)}"
        )
        cos_sb = 0
        cos_ss, cos_sd = cos.stride()
        sin_sb = 0
        sin_ss, sin_sd = sin.stride()
    elif cos.dim() == 3:
        assert cos.shape == (B, S, D), (
            f"3D cos must be (B={B}, S={S}, D={D}), got {tuple(cos.shape)}"
        )
        cos_sb, cos_ss, cos_sd = cos.stride()
        sin_sb, sin_ss, sin_sd = sin.stride()
    else:
        raise AssertionError(
            f"cos/sin must be (S, D) or (B, S, D), got {tuple(cos.shape)}"
        )

    # Output: empty tensor with x's logical shape. empty_like preserves x's
    # (possibly transposed) memory layout, but since we hand the kernel
    # out's real strides that's irrelevant to correctness.
    out = torch.empty_like(x)

    half = D // 2
    # Each program processes HALF lanes (one half-vector); pad to a power of
    # two for Triton's vectorized loads, masking the tail.
    block = triton.next_power_of_2(half)

    # One program per token's head vector.
    grid = (B, H, S)
    rope_forward_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos_sb, cos_ss, cos_sd,
        sin_sb, sin_ss, sin_sd,
        HALF=half,
        BLOCK=block,
    )
    return out
