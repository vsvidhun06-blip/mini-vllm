"""
Attention machinery for the LLaMA-family forward pass.

This module owns two distinct concerns that both live "inside" the attention
block of a transformer layer:

  1. Rotary Position Embeddings (RoPE) -- how the model learns *where* a
     token is in the sequence. RoPE is applied to Q and K just before the
     attention dot product.

  2. Grouped-Query Attention (GQA) -- TinyLlama has 32 query heads but only
     4 key/value heads, with each KV head shared across 8 Q heads. This
     shrinks the KV cache 8x at inference with minimal quality loss.

Everything here is plain PyTorch primitives -- no `transformers` dependency.
The actual softmax(QK^T / sqrt(d)) V dot product is delegated to
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which fuses the
math and avoids materializing the (S, S) attention matrix in memory.

Shape conventions used throughout this file:
    B   = batch size
    S   = sequence length
    H   = total hidden size (e.g. 2048 for TinyLlama-1.1B)
    NQ  = number of query heads     (32)
    NKV = number of key/value heads ( 4)
    D   = per-head dimension        (64 = 2048 / 32)
    G   = NQ // NKV = group size    ( 8) -- how many Q heads share one KV head
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE: Rotary Position Embeddings
# ---------------------------------------------------------------------------
#
# Why RoPE (and not sinusoidal-add or learned absolute positions)?
#
#   The original Transformer added a positional vector to the token embedding
#   ONCE at the input. Every layer afterwards has to "remember" position
#   through the residual stream. RoPE takes a different approach: it injects
#   position info directly into Q and K *every time* attention runs, by
#   ROTATING the vectors. Because rotation preserves length, it doesn't
#   distort the magnitudes the model learned; it only shifts angles.
#
#   The math trick: if you rotate q at position m by angle (m * theta) and
#   k at position n by angle (n * theta), then q . k after rotation depends
#   only on (m - n). So you get *relative* position info via *absolute*
#   rotations -- no extra parameters, no added embedding vector, and the
#   attention dot product naturally becomes translation-invariant.
#
# How HF implements it (we must match this byte-for-byte for parity):
#
#   For a head of dimension D, we treat dims as D/2 pairs. For pair index i,
#   the frequency is:
#       theta_i = base^(-2i / D)        with base = 10000 by default
#   For a token at position m, the rotation angle for pair i is m * theta_i.
#
#   HF's layout splits the head dim into two HALVES (not interleaved pairs):
#       dims [0      .. D/2)   -- the "first half"
#       dims [D/2    .. D)     -- the "second half"
#   and treats (first[i], second[i]) as the 2D vector to rotate. This is
#   sometimes called "GPT-J style" RoPE and is equivalent under a permutation
#   to interleaved style, but the WEIGHTS are trained for this layout, so
#   we have to use it exactly.
#
#   cos/sin tables are pre-built once with shape (max_seq, D), where the
#   second half is a duplicate of the first. Then for q, k of shape
#   (..., S, D), the rotation is:
#       q_rot = q * cos + rotate_half(q) * sin
#   where rotate_half([a, b]) = [-b, a] (treating the head as two halves).
# ---------------------------------------------------------------------------


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables for RoPE.

    We build this ONCE at model construction and slice it per forward call.
    Building per-call would waste time recomputing the same trig values.

    Args:
        head_dim: per-head dimension D (must be even).
        max_seq_len: longest sequence we'll ever serve. We pre-build the
            table up to this length so forward never has to recompute.
        base: the "theta" base, 10000 in vanilla RoPE. Larger base ==
            slower rotation == better length extrapolation.
        dtype: precision of the cache. fp32 keeps the trig values exact.

    Returns:
        cos, sin: each of shape (max_seq_len, head_dim). The dim axis is
            "duplicated" -- second half is a copy of the first half -- so
            that the dot product `q * cos + rotate_half(q) * sin` works
            with full-width tensors and not just pairs.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # inv_freq[i] = 1 / base^(2i / D)  for i in [0, D/2)
    # Shape: (D/2,)
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))

    # Outer product: positions x frequencies.
    # freqs[m, i] = m * inv_freq[i]  -- the angle for position m, pair i.
    # Shape: (max_seq_len, D/2)
    t = torch.arange(max_seq_len, dtype=dtype)
    freqs = torch.outer(t, inv_freq)

    # HF layout: duplicate along the last dim so both halves carry the same
    # angles. Then applying rotate_half (defined below) gives the correct
    # 2D rotation per (first_half[i], second_half[i]) pair.
    # Shape after cat: (max_seq_len, D)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    assert cos.shape == (max_seq_len, head_dim)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: [a, b] -> [-b, a].

    This is the second half of the 2D rotation formula:
        [cos -sin] [a]   [a*cos - b*sin]
        [sin  cos] [b] = [a*sin + b*cos]

    With cos/sin laid out as (first_half | second_half) duplicated, the
    full-width identity is:
        q_rot = q * cos + rotate_half(q) * sin
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]       # "first half"  (the a's)
    x2 = x[..., half:]       # "second half" (the b's)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Q and K.

    Args:
        q: (B, NQ,  S, D)
        k: (B, NKV, S, D)
        cos, sin: (S, D) -- already sliced to the current seq length.

    Returns:
        q_rot, k_rot with the same shapes as inputs.
    """
    # Broadcast cos/sin from (S, D) to (1, 1, S, D) so they apply across
    # the batch and head axes uniformly.
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Grouped-Query Attention block
# ---------------------------------------------------------------------------
#
# Why GQA (and not standard multi-head)?
#
#   In vanilla MHA, every head has its own K and V projection. At inference
#   we cache K and V for every token already seen, so the KV cache size is:
#       per-token bytes = 2 * num_layers * num_heads * head_dim * dtype_size
#   For a 22-layer, 32-head, 64-dim, fp16 model that's 2*22*32*64*2 = 180 KB
#   per token. A 2048-token context burns ~360 MB. GQA cuts num_heads on
#   the K/V side from 32 down to 4 -- a literal 8x KV-cache shrink -- while
#   keeping 32 Q heads so representation power doesn't collapse.
#
#   At training time GQA is a small quality hit vs MHA. At inference time
#   it's a massive memory and bandwidth win. This is why every modern serving
#   LLM uses it (LLaMA 2 70B, Mistral, etc).
#
# How we make GQA "fit" into standard attention:
#
#   Q stays at (B, NQ=32, S, D). K and V come out of their projections at
#   (B, NKV=4, S, D). To run SDPA, the K/V head count must match Q's. We
#   simply REPEAT each KV head G=8 times along the head axis. This is cheap
#   (memory view trick via repeat_interleave; no large matmul) and recovers
#   the (B, 32, S, D) shape SDPA expects.
#
#   In production serving you'd skip this repeat and instead pass a grouped
#   layout into a fused kernel. For the parity-checking forward pass this
#   simpler approach is fine.
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Self-attention block: pre-norm is applied OUTSIDE this module."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden must divide evenly into heads"
        assert num_heads % num_kv_heads == 0, "Q heads must be a multiple of KV heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads          # NQ
        self.num_kv_heads = num_kv_heads    # NKV
        self.head_dim = hidden_size // num_heads  # D
        self.group_size = num_heads // num_kv_heads  # G

        # Linear projections. LLaMA uses NO bias on these, unlike GPT-2.
        # Q maps the full hidden to all 32 heads worth of dim:  H -> NQ * D
        # K and V map to the smaller KV-head count:             H -> NKV * D
        # The output projection mixes the heads back to H:      NQ * D -> H
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # RoPE cache. Registered as a non-persistent buffer so it moves with
        # .to(device) but doesn't show up in state_dict (we don't want to
        # save/load a deterministic lookup table from a checkpoint).
        cos, sin = build_rope_cache(self.head_dim, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, H) -- already RMSNorm'd by the caller.

        Returns:
            (B, S, H) -- to be added to the residual stream by the caller.
        """
        B, S, _ = x.shape

        # 1) Project to Q, K, V.
        #    Q: (B, S, NQ*D)
        #    K: (B, S, NKV*D)
        #    V: (B, S, NKV*D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) Reshape to per-head form, then transpose seq and head axes so
        #    the layout is (B, num_heads, S, D) -- the layout SDPA expects.
        q = q.view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3) Apply RoPE to Q and K. V is NOT rotated -- only the things that
        #    participate in the dot product need position info.
        cos = self.rope_cos[:S].to(dtype=q.dtype)
        sin = self.rope_sin[:S].to(dtype=q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # 4) GQA expansion: replicate each KV head G times so K/V have 32
        #    head slots instead of 4. repeat_interleave keeps the order:
        #    [h0, h0, ..., h0(x8), h1, h1, ..., h1(x8), ...]
        #    which matches how Q heads were laid out (heads 0..7 share KV
        #    head 0, heads 8..15 share KV head 1, etc).
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 5) Scaled dot-product attention with causal mask. SDPA handles
        #    the scaling by 1/sqrt(D), the softmax, and the mask in one fused
        #    op -- and crucially, it never materializes the (S, S) matrix
        #    when using FlashAttention-style backends.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # attn_out: (B, NQ, S, D)

        # 6) Merge heads back: (B, NQ, S, D) -> (B, S, NQ*D) -> (B, S, H).
        #    .contiguous() because transpose makes the tensor non-contiguous
        #    and view requires contiguous memory.
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        # 7) Output projection mixes information across heads.
        return self.o_proj(attn_out)
