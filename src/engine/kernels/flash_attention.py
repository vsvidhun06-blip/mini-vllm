"""
From-scratch Flash Attention 2 forward pass in Triton (inference only).

WHY THIS EXISTS
---------------
The reference path in ``attention.py`` calls
``F.scaled_dot_product_attention`` (SDPA). SDPA is already fused and fast, but
it's a black box: for an educational, "I-can-defend-every-line" engine we want
the actual FA2 algorithm written out. This kernel reproduces SDPA's math with
the FlashAttention-2 tiling + online-softmax scheme and is the CUDA path; SDPA
remains the CPU fallback.

THE ALGORITHM (FA2 forward, no backward -- we never train)
----------------------------------------------------------
Naive attention materializes the full (S, S) score matrix:
    S = QK^T / sqrt(d);  P = softmax(S);  O = P V
That's O(S^2) memory and S^2 reads/writes to HBM. Flash Attention never
forms the full matrix. Instead, for each block of queries it streams over
blocks of keys/values, keeping a running softmax:

  For a fixed block of BLOCK_M query rows we keep three accumulators in SRAM:
      m_i  -- running row-max of the scores seen so far      (BLOCK_M,)
      l_i  -- running sum of exp(scores - m_i) (the denom)   (BLOCK_M,)
      acc  -- running unnormalized output  sum(p * V)        (BLOCK_M, D)

  For each new key/value block we:
      1. s   = scale * Q_block @ K_block^T          (BLOCK_M, BLOCK_N)
      2. apply masking (causal / padding / explicit bias)
      3. m_new = max(m_i, rowmax(s))                -- new running max
      4. p   = exp(s - m_new)                       -- rescale THIS block
      5. alpha = exp(m_i - m_new)                   -- correction for the
                                                       PAST, because the max
                                                       just moved
      6. l_i = l_i * alpha + rowsum(p)              -- fix old denom + add new
      7. acc = acc * alpha + p @ V_block            -- fix old output + add new
      8. m_i = m_new
  After the last block: O = acc / l_i.

  Steps 5-7 are the "online" trick: when the running max increases we
  retro-actively down-scale everything accumulated so far by ``alpha`` so the
  final result is identical to a single global softmax -- but we never had to
  hold the whole row of scores at once. This is what makes attention O(S)
  memory instead of O(S^2), and it's the entire point.

NUMERICAL PARITY NOTE
---------------------
The engine runs fp32 with TF32 explicitly disabled (see device.py) so the HF
parity tests hold at atol=1e-4. ``tl.dot`` would happily use TF32 tensor cores
by default, which blows that tolerance, so every ``tl.dot`` here passes
``allow_tf32=False``. Online softmax is just a re-association of the same
fp32 arithmetic SDPA does, so the outputs agree to ~1e-5.

GRID / TILING
-------------
grid = (batch, heads, ceil(S_q / BLOCK_M)). Each program owns one BLOCK_M
slice of query rows for one (batch, head) and streams the whole K/V sequence.
GQA is handled by the caller (K/V are already head-expanded to match Q), so
this kernel only ever sees standard multi-head shapes.

PERFORMANCE: WHY THIS IS SLOWER THAN F.scaled_dot_product_attention
-------------------------------------------------------------------
Measured on the benchmark, this kernel is SLOWER than PyTorch's
``F.scaled_dot_product_attention`` (SDPA), and that is expected. The point of
writing FA2 from scratch here is to understand the algorithm end to end, not to
beat a vendor library. The gap comes from several layers of optimisation that
SDPA has and this kernel does not:

  1. cuDNN / FlashAttention-2 CUDA fusion. SDPA dispatches to hand-written
     CUDA C++ backends (cuDNN's fused attention, or the official FA2 kernels)
     that fuse the QK^T matmul, the scale, the softmax, and the PV matmul into
     one tightly-scheduled kernel with optimal HBM<->SRAM data movement. Ours
     expresses the same math in Triton at a higher level and leaves a lot of
     that fusion/scheduling to the compiler.

  2. No persistent kernels. SDPA's backends use persistent / warp-specialised
     designs where a fixed set of CTAs stays resident and streams many tiles,
     amortising launch cost and keeping every SM busy. We launch one program
     per (batch, head, M-block) and exit; for small and medium problem sizes
     the per-launch overhead and the "tail" of partly-filled waves dominate the
     runtime.

  3. No register-level tiling / pipelining tuning. The fast backends carefully
     stage operands through registers and shared memory with multi-stage
     software pipelining (double/triple buffering) and tuned warp-level MMA
     schedules. We use Triton's defaults and never autotuned BLOCK_M/BLOCK_N,
     ``num_warps``, or ``num_stages`` per head-dim/arch, so memory loads and the
     two matmuls are not overlapped as aggressively.

  4. We deliberately run true fp32 (``allow_tf32=False``) to hold the engine's
     atol=1e-4 parity contract, whereas SDPA is free to pick faster reduced- or
     mixed-precision tensor-core paths.

So the takeaway is not "Triton is slow" -- it's that matching a cuDNN/FA2
production kernel needs persistent scheduling + autotuned register tiling that a
readable from-scratch kernel intentionally omits. See docs/design.md for the
fuller write-up. The correctness win (we reproduce SDPA's output to ~1e-5) is
the deliverable; the speed gap is the cost of clarity.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Tile sizes. 64x64 is a solid default for head_dim<=128 on Ampere/Ada and
# keeps the per-program SRAM footprint modest.
BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    Mask,                         # additive float bias (S_q, S_k) or dummy
    scale,                        # 1 / sqrt(head_dim)
    # Q strides (B, H, S_q, D)
    stride_qb, stride_qh, stride_qm, stride_qd,
    # K strides (B, H, S_k, D)
    stride_kb, stride_kh, stride_kn, stride_kd,
    # V strides (B, H, S_k, D)
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Out strides (B, H, S_q, D)
    stride_ob, stride_oh, stride_om, stride_od,
    # Mask strides (S_q, S_k)
    stride_mm, stride_mn,
    S_q, S_k,                     # runtime sequence lengths
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,        # = head_dim (power of 2), loaded whole
    CAUSAL: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Row indices for this query block, and the full head-dim column range.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_d = tl.arange(0, BLOCK_D)                      # (BLOCK_D,)

    # Per-(batch, head) base pointers.
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh

    # Load this query block once into registers/SRAM. Rows past S_q are
    # padding (masked off on store), loaded as 0.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S_q, other=0.0)

    # Online-softmax accumulators.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # The alignment offset between query and key absolute positions. For full
    # prefill S_q == S_k so this is 0; if a caller ever passes S_q < S_k with
    # causal=True, query row i still maps to absolute position q_offset + i.
    q_offset = S_k - S_q

    # Causal early-stop: a query block never needs key columns beyond its own
    # last row's diagonal. Capping the loop here is the FA2 efficiency win --
    # we skip whole future K/V blocks instead of masking them to -inf.
    if CAUSAL:
        n_end = tl.minimum((pid_m + 1) * BLOCK_M + q_offset, S_k)
    else:
        n_end = S_k

    for start_n in range(0, n_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)        # (BLOCK_N,)

        # Load K, V blocks. (BLOCK_N, BLOCK_D). Padding rows -> 0.
        kv_row_valid = offs_n[:, None] < S_k
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=kv_row_valid, other=0.0)
        v = tl.load(v_ptrs, mask=kv_row_valid, other=0.0)

        # Scores: (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N).
        # allow_tf32=False keeps this true fp32 (see module docstring).
        s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale

        # --- masking, all expressed as additive -inf on the scores ---
        # Key padding: columns past S_k can never be attended.
        s = tl.where(offs_n[None, :] < S_k, s, float("-inf"))
        if CAUSAL:
            # query abs pos (q_offset + offs_m) must be >= key abs pos (offs_n)
            causal_ok = (q_offset + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_ok, s, float("-inf"))
        if USE_MASK:
            m_ptrs = Mask + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
            bias = tl.load(
                m_ptrs,
                mask=(offs_m[:, None] < S_q) & (offs_n[None, :] < S_k),
                other=0.0,
            )
            # The wrapper casts every input (incl. this additive mask) up to
            # fp32 before launch, so bias is already fp32 and the add keeps the
            # scores fp32 -- no operand-dtype mixing.
            s = s + bias

        # --- online softmax update ---
        m_new = tl.maximum(m_i, tl.max(s, axis=1))      # new running max
        p = tl.exp(s - m_new[:, None])                  # this block's weights
        alpha = tl.exp(m_i - m_new)                     # correction for the past
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Both operands are fp32: the wrapper forces q/k/v to fp32, so the
        # scores dot above returns fp32 -> p is fp32, and v was loaded fp32.
        # That alignment is the whole point of the fp32-everywhere policy --
        # tl.dot rejects mixed-dtype operands ("Both operands must be same
        # dtype"), which is exactly what bit us when v was loaded fp16.
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new

    # Normalize. No real query row is ever fully masked (every row attends to
    # at least its own key), so l_i > 0 for all stored rows.
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < S_q)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA2 forward pass. Drop-in for SDPA on the engine's attention shapes.

    Args:
        q: (B, NH, S_q, D) -- queries.
        k: (B, NH, S_k, D) -- keys.  S_k may exceed S_q (decode / sliced prefill).
        v: (B, NH, S_k, D) -- values.
        causal: if True, apply a causal mask (query abs-pos >= key abs-pos,
            with query positions aligned to the END of the key sequence so it
            stays correct when S_q < S_k).
        attn_mask: optional ADDITIVE float bias of shape (S_q, S_k) -- 0.0 to
            attend, -inf to forbid. This is the "explicit mask" path used for
            sliced prefill (prefix cache). Mutually exclusive with ``causal``.

    Returns:
        (B, NH, S_q, D) attention output, same dtype/device as ``q``.
    """
    # --- fp32 everywhere (do this FIRST, before any shape/stride reads) -----
    # The model serves in fp16 on GPU, so q/k/v arrive fp16. Triton's tl.dot
    # returns its result in the INPUT operands' dtype: with fp16 q/k the scores
    # dot comes back fp16, so the softmax weights p are fp16 -- but the value
    # matmul then mixes that fp16 p with v, and any per-operand cast leaves the
    # operands mismatched. That is exactly the "Both operands must be same dtype.
    # Got fp32 and fp16" crash. Rather than thread casts through every op, run
    # the WHOLE kernel in fp32: promote every input (incl. the additive mask)
    # here, compute, then cast the output back to the caller's dtype at the end.
    # The guard keeps the genuine-fp32 path allocation-free. Strides are read
    # AFTER this so the kernel always sees the fp32 tensors' layout. (This also
    # matches the engine's true-fp32 parity contract: the kernel disables TF32,
    # so promoting fp16->fp32 leaves the numerics identical to the fp32 path.)
    orig_dtype = q.dtype
    if q.dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.float32)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "expected (B, NH, S, D)"
    B, NH, S_q, D = q.shape
    Bk, NHk, S_k, Dk = k.shape
    assert (Bk, NHk, Dk) == (B, NH, D), "q/k batch, head, dim must match"
    assert v.shape == (B, NH, S_k, D), "v must match k's (B, NH, S_k, D)"
    assert (D & (D - 1)) == 0 and D > 0, f"head_dim must be a power of 2, got {D}"
    assert q.is_cuda, "flash_attention_forward is CUDA-only; use SDPA on CPU"
    assert not (causal and attn_mask is not None), (
        "pass either causal=True or an explicit attn_mask, not both"
    )

    out = torch.empty_like(q)          # fp32 (q is now fp32)
    scale = 1.0 / (D ** 0.5)

    use_mask = attn_mask is not None
    if use_mask:
        assert attn_mask.shape == (S_q, S_k), (
            f"attn_mask must be (S_q={S_q}, S_k={S_k}), got {tuple(attn_mask.shape)}"
        )
        mask_arg = attn_mask
        stride_mm, stride_mn = attn_mask.stride()
    else:
        # Pass q as a harmless non-null placeholder; USE_MASK=False means the
        # kernel never dereferences it.
        mask_arg = q
        stride_mm, stride_mn = 0, 0

    grid = (B, NH, triton.cdiv(S_q, BLOCK_M))
    flash_attention_kernel[grid](
        q, k, v, out,
        mask_arg,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stride_mm, stride_mn,
        S_q, S_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        CAUSAL=causal,
        USE_MASK=use_mask,
    )
    # Hand back the caller's original dtype (e.g. fp16) so this stays a drop-in
    # for SDPA; the fp32 promotion above is an internal compute detail.
    return out.to(orig_dtype)
