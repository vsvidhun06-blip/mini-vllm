"""
INT8-quantized attention block (W8A8).

This wires the quantization primitives in ``quantization.py`` into the engine's
attention. The design goal is a true DROP-IN: ``QuantizedMultiHeadAttention``
has the same ``__init__`` and ``forward`` signatures as the fp
``MultiHeadAttention``, so ``LlamaModel.quantize()`` can swap one for the other
inside every TransformerBlock without touching any calling code.

WHAT CHANGES vs MultiHeadAttention
----------------------------------
Only the four projections (q/k/v/o) become int8. Everything else -- RoPE, the
flash-attention call, the KV-cache plumbing, GQA expansion, the causal-mask
logic -- is INHERITED unchanged from ``MultiHeadAttention``. We achieve that by
replacing the four ``nn.Linear`` submodules with ``QuantizedLinear`` modules
that expose the exact same ``module(x) -> tensor`` call interface. The parent's
``forward`` calls ``self.q_proj(x)`` etc. and neither knows nor cares that the
projection is now an int8 GEMM.

WEIGHTS OFFLINE, ACTIVATIONS ONLINE
-----------------------------------
Weights are quantized ONCE at load time (``from_float``) -- they're constant,
so we pay the rounding cost once and store int8. Activations change every
forward, so they're quantized at runtime inside ``quantized_linear``.

DTYPE CONTRACT
--------------
``quantized_linear`` returns fp16 (the kernel rescales to fp16). To keep the
quantized block a seamless drop-in inside the otherwise-fp32 engine -- so the
residual stream, RoPE tables, and KV cache all stay one dtype -- ``QuantizedLinear``
casts the GEMM output back to the activation's dtype. The quantization error is
already baked into the values; the cast only changes the container.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.attention import MultiHeadAttention
from src.engine.kernels.quantization import (
    dequantize_tensor,
    quantize_tensor,
    quantized_linear,
)


class QuantizedLinear(nn.Module):
    """A bias-less linear layer whose weight is stored int8 (per-tensor scale).

    Call interface matches ``nn.Linear``: ``y = layer(x)``. On CUDA it runs the
    int8 Triton GEMM; on CPU it falls back to dequantize-then-fp-matmul so the
    module still works (and so CPU-only test hosts don't need a GPU).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # int8 weight (out, in) -- same layout as nn.Linear.weight -- plus its
        # scalar scale. Registered as buffers so they move with .to(device) and
        # are saved/restored by state_dict.
        self.register_buffer(
            "weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Build a QuantizedLinear by quantizing an existing fp ``nn.Linear``."""
        assert linear.bias is None, "LLaMA projections are bias-less"
        ql = cls(linear.in_features, linear.out_features)
        q, scale = quantize_tensor(linear.weight.data)
        # Overwrite the registered buffers (keeps them as buffers, moves them to
        # the source weight's device/dtype).
        ql.weight_int8 = q
        ql.weight_scale = scale.to(torch.float32)
        return ql

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            # int8 GEMM -> fp16 -> cast back to the activation dtype so the rest
            # of the engine stays in its native precision.
            out = quantized_linear(x, self.weight_int8, self.weight_scale)
            return out.to(x.dtype)
        # CPU fallback: dequantize the weight and do a normal fp matmul. Same
        # math the int8 GEMM approximates, just without Triton.
        w = dequantize_tensor(self.weight_int8, self.weight_scale).to(x.dtype)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8=True"


class QuantizedMultiHeadAttention(MultiHeadAttention):
    """Drop-in MultiHeadAttention with int8 q/k/v/o projections.

    Same constructor signature as the parent. A freshly-constructed instance
    quantizes its (random) projection weights immediately; the realistic entry
    point is :meth:`from_float`, which quantizes a TRAINED ``MultiHeadAttention``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
    ) -> None:
        # Build the full fp attention (creates nn.Linear projections + RoPE
        # buffers), then replace the projections with int8 versions.
        super().__init__(hidden_size, num_heads, num_kv_heads, max_seq_len, rope_base)
        self.q_proj = QuantizedLinear.from_float(self.q_proj)
        self.k_proj = QuantizedLinear.from_float(self.k_proj)
        self.v_proj = QuantizedLinear.from_float(self.v_proj)
        self.o_proj = QuantizedLinear.from_float(self.o_proj)

    @classmethod
    def from_float(
        cls,
        mha: MultiHeadAttention,
        max_seq_len: int | None = None,
        rope_base: float = 10000.0,
    ) -> "QuantizedMultiHeadAttention":
        """Quantize a TRAINED ``MultiHeadAttention`` into a W8A8 block.

        Args:
            mha: the source fp attention with loaded weights.
            max_seq_len: RoPE table length. Defaults to the source's table
                length (``rope_cos.shape[0]``) so the rebuilt buffers match.
            rope_base: only affects the throwaway buffers built in ``__init__``;
                we overwrite RoPE with the source's exact tables below.
        """
        if max_seq_len is None:
            max_seq_len = mha.rope_cos.shape[0]
        self = cls(
            hidden_size=mha.hidden_size,
            num_heads=mha.num_heads,
            num_kv_heads=mha.num_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        # Re-quantize from the TRAINED projections (the __init__ above quantized
        # the fresh random weights; replace them with the real ones).
        self.q_proj = QuantizedLinear.from_float(mha.q_proj)
        self.k_proj = QuantizedLinear.from_float(mha.k_proj)
        self.v_proj = QuantizedLinear.from_float(mha.v_proj)
        self.o_proj = QuantizedLinear.from_float(mha.o_proj)
        # Copy the source's RoPE tables verbatim (preserves exact values,
        # device, and dtype -- no dependence on rope_base).
        self.rope_cos = mha.rope_cos.clone()
        self.rope_sin = mha.rope_sin.clone()
        return self

    # forward() is inherited from MultiHeadAttention unchanged: self.q_proj(x)
    # and friends now dispatch to the int8 GEMM, and everything downstream
    # (RoPE, flash attention, cache) is identical to the fp path.
