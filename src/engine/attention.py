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

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.engine.kv_cache import PagedRequestCache


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
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden must divide evenly into heads"
        assert num_heads % num_kv_heads == 0, "Q heads must be a multiple of KV heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads          # NQ
        self.num_kv_heads = num_kv_heads    # NKV
        self.head_dim = hidden_size // num_heads  # D
        self.group_size = num_heads // num_kv_heads  # G

        # Linear projections. LLaMA uses NO bias on these, unlike GPT-2 -- which
        # is why qkv_bias DEFAULTS to False and TinyLlama / every existing test
        # is byte-identical. Qwen2, however, keeps a learned bias on Q/K/V (but
        # NOT on the output projection); set qkv_bias=True for that family so the
        # checkpoint's bias vectors have a home to load into.
        # Q maps the full hidden to all 32 heads worth of dim:  H -> NQ * D
        # K and V map to the smaller KV-head count:             H -> NKV * D
        # The output projection mixes the heads back to H:      NQ * D -> H
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # RoPE cache. Registered as a non-persistent buffer so it moves with
        # .to(device) but doesn't show up in state_dict (we don't want to
        # save/load a deterministic lookup table from a checkpoint).
        cos, sin = build_rope_cache(self.head_dim, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _sdpa_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit softmax attention that ALSO returns the attention weights.

        ``F.scaled_dot_product_attention`` and our FA2 kernel both fuse the
        softmax and never materialise the (S_q, S_k) weight matrix -- which is
        the whole speed win, but means there is nothing for H2O to score. When
        a score tracker is attached (eviction is active) we fall back to this
        un-fused path so we can read the per-key attention mass. It is
        mathematically identical to SDPA; the cost is materialising the weights
        and an extra matmul, paid only while eviction is on.

        Returns (out, weights) with out (B, NQ, S_q, D) and weights
        (B, NQ, S_q, S_k).
        """
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, NQ, S_q, S_k)
        if attn_mask is not None:
            # Boolean mask, True == attend (the sliced-prefill case).
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        elif is_causal:
            S_q, S_k = scores.shape[-2], scores.shape[-1]
            causal = torch.tril(
                torch.ones(S_q, S_k, dtype=torch.bool, device=q.device)
            )
            scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return out, weights

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: "PagedRequestCache | list[PagedRequestCache] | None" = None,
        layer_idx: int | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, H) -- already RMSNorm'd by the caller.
            kv_cache: one of:
                * None: no cache, recompute everything (the Day 2-4 path).
                * PagedRequestCache: single request, prefill or decode.
                * list[PagedRequestCache]: BATCHED DECODE across B requests, each
                    with its own cache. S must be 1. Each cache is at a
                    different point in its own sequence, so RoPE is per-row.
            layer_idx: which layer slot in the cache to use. Required when
                kv_cache is not None.
            return_attn_weights: when True, return (output, attn_weights)
                instead of just output, computed via the un-fused softmax path.
                Independently, if the kv_cache carries a ``score_tracker``
                attribute (H2O eviction is active) the weights are captured and
                fed to that tracker as a side effect even when this is False --
                that is how eviction observes attention mass without changing
                model.forward's signature. With neither in play, the fast
                SDPA/FA2 path runs unchanged (so existing tests are unaffected).

        Returns:
            (B, S, H) -- to be added to the residual stream by the caller; OR
            ((B, S, H), weights) when return_attn_weights is True.
        """
        # Dispatch to the batched-decode path when caller passes a list of
        # per-request caches. This is the Day 6 continuous-batching entry.
        if isinstance(kv_cache, list):
            return self._forward_decode_batched(x, kv_cache, layer_idx)

        B, S, _ = x.shape

        if kv_cache is not None and layer_idx is None:
            raise ValueError("layer_idx is required when kv_cache is provided")

        # 1) Project to Q, K, V on the input tokens (just the new ones in decode).
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

        # 3) RoPE -- THE position-offset gotcha.
        #
        #    The absolute position of the FIRST token in `x` is the current
        #    seq_len of the cache. In prefill (empty cache) that's 0. In a
        #    decode step that just added 5 prompt tokens earlier, the next
        #    input's position is 5. RoPE indexes the cos/sin table by
        #    ABSOLUTE position, not position-within-this-input.
        #
        #    If we forgot this offset, we'd always rotate the new token's Q
        #    and K by angle 0, breaking every dot product against cached K
        #    that was rotated at its real position.
        # Read THIS layer's seq_len, not a global one. Earlier layers in
        # this same forward pass have already appended to their slots; if
        # we asked for layer 0's count we'd see N+1 instead of N for every
        # layer past the first.
        pos_offset = kv_cache.seq_len(layer_idx) if kv_cache is not None else 0
        cos = self.rope_cos[pos_offset:pos_offset + S].to(dtype=q.dtype)
        sin = self.rope_sin[pos_offset:pos_offset + S].to(dtype=q.dtype)
        # On CUDA, fuse the rotation into a single kernel launch per tensor
        # (see src/engine/kernels/rope.py). On CPU we keep the reference
        # PyTorch path -- it stays correct and avoids importing Triton, which
        # isn't installed (or supported) on CPU-only hosts. The import is lazy
        # for exactly that reason: a top-level `import triton` would break the
        # CPU correctness tests.
        if q.is_cuda:
            from src.engine.kernels.rope import fused_rope
            q = fused_rope(q, cos, sin)  # cos/sin are (S, D) -> broadcast batch
            k = fused_rope(k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # 4) Cache update.
        #    We cache K *after* RoPE, V as-is (V isn't rotated). Caching
        #    post-rotation means future decode steps don't have to re-rotate
        #    the historical K -- which would defeat the cache's purpose.
        #
        #    The cache layout stores (B, seq, NKV, D), so we transpose
        #    BACK to seq-major to append. Then we read the full cached K, V
        #    back out and transpose to (B, NKV, seq_total, D) for SDPA.
        if kv_cache is not None:
            k_to_cache = k.transpose(1, 2)  # (B, S, NKV, D)
            v_to_cache = v.transpose(1, 2)  # (B, S, NKV, D)
            kv_cache.append(layer_idx, k_to_cache, v_to_cache)
            k_full, v_full = kv_cache.get(layer_idx)        # (B, S_total, NKV, D)
            k = k_full.transpose(1, 2)                       # (B, NKV, S_total, D)
            v = v_full.transpose(1, 2)                       # (B, NKV, S_total, D)
        # If no cache: k, v stay at (B, NKV, S, D) -- already correct shape.

        # 5) GQA expansion: replicate each KV head G times to match the
        #    32 query heads. Cheap memory-replication, no real compute.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 6) SDPA. Causal mask cases:
        #
        #    The previously-cached K has length pos_offset; the new k
        #    (S rows) was just appended; total K length is pos_offset + S.
        #    Q has S rows whose absolute positions are
        #    [pos_offset, pos_offset + S).
        #
        #    Three regimes:
        #      * Decode (S == 1): the one new query at the END of K
        #        attends to every past key. No mask needed.
        #      * Full prefill (S > 1 and pos_offset == 0): Q and K are
        #        the same length, standard upper-left causal triangle
        #        applies. is_causal=True is correct.
        #      * SLICED prefill (S > 1 and pos_offset > 0): this is the
        #        prefix-cache path. Q has fewer rows than K. SDPA's
        #        built-in is_causal mask is upper-LEFT aligned -- it
        #        would force row 0 of Q to attend only to column 0 of K,
        #        which is catastrophically wrong (we want row 0 attending
        #        to columns 0..pos_offset). We build the mask explicitly:
        #        row i (absolute position pos_offset + i) attends to
        #        columns 0..(pos_offset + i).
        if S == 1:
            attn_mask = None
            is_causal = False
        elif pos_offset == 0:
            attn_mask = None
            is_causal = True
        else:
            S_total = k.shape[2]
            q_pos = torch.arange(pos_offset, pos_offset + S, device=q.device)
            k_pos = torch.arange(S_total, device=q.device)
            # Broadcast to (S, S_total). True == attend.
            attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            is_causal = False
        # H2O hook: if the cache carries a score tracker, eviction is active and
        # we MUST see the attention weights. Otherwise stay on the fast path.
        tracker = getattr(kv_cache, "score_tracker", None) if kv_cache is not None else None
        need_weights = return_attn_weights or tracker is not None

        attn_weights = None
        if need_weights:
            # Un-fused softmax so the weights are available (CPU and CUDA alike).
            attn_out, attn_weights = self._sdpa_with_weights(q, k, v, attn_mask, is_causal)
            if tracker is not None:
                # Accumulate per-key attention mass for this layer. Detached --
                # we never backprop through the eviction policy.
                tracker.update(layer_idx, attn_weights.detach())
        elif q.is_cuda:
            # On CUDA, run our from-scratch FA2 kernel; SDPA stays the CPU
            # fallback (and avoids importing Triton on CPU-only hosts -- hence
            # the lazy import). The three masking cases map cleanly onto the
            # kernel:
            #   * decode  (attn_mask None, is_causal False) -> causal=False
            #   * prefill (attn_mask None, is_causal True ) -> causal=True
            #   * sliced  (bool attn_mask)                  -> additive -inf bias
            from src.engine.kernels.flash_attention import flash_attention_forward
            if attn_mask is not None:
                # Convert the boolean "True == attend" mask to the additive
                # float bias the kernel wants: 0.0 to attend, -inf to forbid.
                add_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                add_mask = add_mask.masked_fill(~attn_mask, float("-inf"))
                attn_out = flash_attention_forward(q, k, v, attn_mask=add_mask)
            else:
                attn_out = flash_attention_forward(q, k, v, causal=is_causal)
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            )
        # attn_out: (B, NQ, S, D) -- still only S queries on the output side.

        # 7) Merge heads back: (B, NQ, S, D) -> (B, S, NQ*D) -> (B, S, H).
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        # 8) Output projection mixes information across heads.
        out = self.o_proj(attn_out)
        if return_attn_weights:
            return out, attn_weights
        return out

    # -----------------------------------------------------------------------
    # Batched decode across multiple per-request KV caches (Day 6).
    # -----------------------------------------------------------------------
    #
    # Why this needs its own path:
    #
    #   In single-cache decode, every batch row (there's just one) is at the
    #   same absolute position and shares one cache tensor. In a continuous
    #   batch, B requests are each at their OWN point in their OWN sequence:
    #     - request 0 might be on its 4th decode token
    #     - request 1 might be on its 17th decode token
    #     - their K/V caches have different lengths
    #
    #   Two consequences:
    #     1. RoPE per row: each row's new token sits at a different absolute
    #        position. We gather per-row cos/sin from a positions vector and
    #        broadcast over the head axis.
    #     2. SDPA per row: each row's cached K/V has a different seq length,
    #        so we can't stack them into one rectangular tensor without
    #        padding+masking. We loop over rows for SDPA only.
    #
    #   What STAYS batched: the big GEMMs (Q/K/V/output projections) and
    #   the MLP block (handled by TransformerBlock). Those are the compute
    #   that matters; batching them is the whole reason continuous batching
    #   is a throughput win. The per-row SDPA loop is the next bottleneck,
    #   and Day 7 (paged attention) is what eliminates it.
    # -----------------------------------------------------------------------

    def _forward_decode_batched(
        self,
        x: torch.Tensor,
        caches: list["PagedRequestCache"],
        layer_idx: int,
    ) -> torch.Tensor:
        """Batched-decode forward across B per-request caches.

        Args:
            x: (B, 1, H) -- one new token per request, RMSNorm'd by caller.
            caches: list of length B, each a PagedRequestCache for one request.
                Each cache may have a DIFFERENT seq_len at entry.
            layer_idx: required.

        Returns:
            (B, 1, H).
        """
        B, S, _ = x.shape
        assert S == 1, "batched decode is exactly one new token per request"
        assert len(caches) == B, (
            f"caches list length {len(caches)} != batch size {B}"
        )
        if layer_idx is None:
            raise ValueError("layer_idx is required for batched decode")

        # 1) Batched projections. The GEMMs run once over the whole batch.
        q = self.q_proj(x)  # (B, 1, NQ*D)
        k = self.k_proj(x)  # (B, 1, NKV*D)
        v = self.v_proj(x)  # (B, 1, NKV*D)

        # 2) Per-head reshape. (B, 1, NH, D) -> (B, NH, 1, D)
        q = q.view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3) Per-row RoPE.
        #    Each request's new token sits at a different absolute position --
        #    that position is the request's CURRENT cache length (before this
        #    layer appends). We gather a (B,) position vector and use it to
        #    index into the cos/sin table.
        # Build the per-row positions WITHOUT a host->device copy: each cache
        # hands back its seq_len as a cached device scalar (kv_cache.py), and we
        # stack them on-device. The old `torch.tensor([...], device=...)` copied
        # from pageable host memory every layer every step -- pure decode-hot-
        # path overhead, and illegal inside a CUDA-graph capture. Same values.
        positions = torch.stack(
            [c.seq_len_tensor(layer_idx) for c in caches]
        )                                                     # (B,) long, device
        cos = self.rope_cos[positions].to(dtype=q.dtype)      # (B, D)
        sin = self.rope_sin[positions].to(dtype=q.dtype)      # (B, D)
        # Same CUDA/CPU split as the single-cache path. Here the angle varies
        # PER BATCH ROW (each request is at its own position), so we pass cos/
        # sin as (B, S=1, D) and the fused kernel indexes them per-batch via a
        # real batch stride -- no broadcast.
        if q.is_cuda:
            from src.engine.kernels.rope import fused_rope
            cos_k = cos.unsqueeze(1)                           # (B, 1, D)
            sin_k = sin.unsqueeze(1)                           # (B, 1, D)
            q = fused_rope(q, cos_k, sin_k)
            k = fused_rope(k, cos_k, sin_k)
        else:
            # Broadcast over heads (dim 1) and seq (dim 2): (B, 1, 1, D).
            cos_b = cos.unsqueeze(1).unsqueeze(1)
            sin_b = sin.unsqueeze(1).unsqueeze(1)
            q = q * cos_b + _rotate_half(q) * sin_b
            k = k * cos_b + _rotate_half(k) * sin_b

        # 4) Per-request cache update + SDPA. This is the loop we cannot
        #    avoid without padding (or paged attention -- Day 7). The body
        #    is the same per-request work the single-cache path does.
        attn_outs: list[torch.Tensor] = []
        for i in range(B):
            # Append this row's new K/V to its own cache.
            k_to_cache = k[i:i+1].transpose(1, 2)  # (1, 1, NKV, D)
            v_to_cache = v[i:i+1].transpose(1, 2)
            caches[i].append(layer_idx, k_to_cache, v_to_cache)

            # Fetch this request's full cached K, V.
            k_full, v_full = caches[i].get(layer_idx)         # (1, S_i+1, NKV, D)
            k_i = k_full.transpose(1, 2)                       # (1, NKV, S_i+1, D)
            v_i = v_full.transpose(1, 2)

            # GQA: replicate KV heads to match Q heads.
            k_i = k_i.repeat_interleave(self.group_size, dim=1)
            v_i = v_i.repeat_interleave(self.group_size, dim=1)

            q_i = q[i:i+1]                                    # (1, NQ, 1, D)
            # Decode mode: no causal mask. The new query sees the full past.
            # Same CUDA(FA2)/CPU(SDPA) split as the single-cache path.
            if q_i.is_cuda:
                from src.engine.kernels.flash_attention import flash_attention_forward
                out_i = flash_attention_forward(q_i, k_i, v_i, causal=False)
            else:
                out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)
            attn_outs.append(out_i)                            # (1, NQ, 1, D)

        # 5) Re-batch and project out. Back to a single GEMM for o_proj.
        attn_out = torch.cat(attn_outs, dim=0)                # (B, NQ, 1, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)
