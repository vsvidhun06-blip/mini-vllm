"""
TinyLlama-1.1B forward pass, implemented from scratch in PyTorch.

What this file owns:

  * RMSNorm                   -- the LLaMA-family normalization layer.
  * SwiGLUMLP                 -- the gated feed-forward block.
  * TransformerBlock          -- pre-norm + attention + pre-norm + MLP,
                                 with two residual adds.
  * LlamaModel                -- token embedding, 22 blocks, final norm,
                                 LM head. The whole forward pass top to
                                 bottom.
  * load_tinyllama_from_hf    -- download the HF safetensors checkpoint,
                                 map its parameter names to ours, and load.
  * main()                    -- Day 2 entry point: load the model, run a
                                 prompt, print the greedy next token.

Heavily commented because I am going to read this code and defend the
architecture decisions in interviews. See `attention.py` for RoPE and GQA.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.attention import MultiHeadAttention
from src.engine.device import DEVICE
from src.engine.kv_cache import PagedKVCache, PagedRequestCache


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
#
# Why RMSNorm instead of LayerNorm?
#
#   LayerNorm does two things to each token vector x:
#       1. Recenter:  x' = x - mean(x)
#       2. Rescale:   x'' = x' / std(x')
#       3. Affine:    out = gamma * x'' + beta
#
#   RMSNorm drops steps 1 and the bias in step 3:
#       1. Rescale:   x' = x / sqrt(mean(x^2) + eps)
#       2. Affine:    out = gamma * x'        (no beta)
#
#   The RMSNorm paper's claim: the recentering step doesn't actually
#   contribute to model quality -- only the rescaling does. Empirically
#   they're right. RMSNorm is faster (one less reduction over the dim),
#   has fewer parameters (no beta), and matches or beats LayerNorm.
#
#   Every LLaMA-family model uses RMSNorm.
#
# Why compute in fp32 even when the model runs in lower precision?
#
#   `mean(x^2)` sums D squared activations. In fp16 the sum can overflow
#   for big activations, or underflow for small ones, both of which corrupt
#   the normalization scale. HF computes RMSNorm in fp32 unconditionally
#   and casts back to the input dtype at the end. We match that exactly --
#   subtle parity bug otherwise.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        # The "gamma" rescaling factor, initialized to ones. One scalar per
        # hidden dim. After training these encode "how much should each
        # feature dim contribute downstream".
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Promote to fp32 for the variance + rsqrt to avoid precision loss.
        x_fp32 = x.to(torch.float32)
        # Mean of squares along the last (feature) dim, keep that dim so
        # broadcasting against x works.
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt(v + eps) is the reciprocal of the RMS. Multiplying by it
        # rescales each row so its RMS becomes 1.
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        # Cast back to whatever dtype the caller is using, THEN apply the
        # learned gamma. The cast-then-multiply order matches HF.
        return (self.weight * x_normed.to(input_dtype))


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
#
# Why SwiGLU instead of GELU-MLP?
#
#   A vanilla transformer MLP is:
#       h = activation(x @ W_in) @ W_out   -- two matrices.
#
#   SwiGLU is:
#       h = (silu(x @ W_gate) * (x @ W_up)) @ W_down
#       silu(x) = x * sigmoid(x)
#
#   Three matrices instead of two. The `silu(gate) * up` part is a "gated
#   linear unit": one projection's output is used to elementwise-gate
#   another projection. Intuition: the gate decides feature-by-feature how
#   much of `up` to let through, giving the MLP a kind of dynamic, input-
#   dependent on/off switch.
#
#   GLU variants beat plain MLPs at the same compute budget in practice
#   (see Shazeer 2020, "GLU Variants Improve Transformer"). SwiGLU costs
#   1.5x more parameters at the same intermediate_size, so LLaMA models
#   compensate by shrinking intermediate_size. TinyLlama uses 5632 (about
#   2.75x hidden) instead of the GPT-classic 4x.
# ---------------------------------------------------------------------------


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # All three projections are bias-less, matching LLaMA.
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate and up share the same input x. silu(gate) is the "switch",
        # up is the "value", elementwise product gates the value.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block (one decoder layer)
# ---------------------------------------------------------------------------
#
# Pre-norm vs post-norm:
#
#   Original "Attention Is All You Need" used POST-norm: sublayer first,
#   then add residual, then norm. That training is unstable at depth --
#   gradients have to backprop through every LayerNorm to reach earlier
#   layers, so signals fade out.
#
#   PRE-norm puts the norm INSIDE each sublayer call:
#       x = x + sublayer(norm(x))
#   Now the residual stream `x` flows uninterrupted from input to output
#   with nothing scaling or shifting it. Training is stable at hundreds
#   of layers. Every modern LLM uses pre-norm.
#
# Two sublayers, two residual adds:
#   1. Attention sublayer:   x += Attn(RMSNorm(x))
#   2. MLP sublayer:         x += MLP(RMSNorm(x))
#
# Note: the two RMSNorms are separate parameters. HF calls them
# `input_layernorm` and `post_attention_layernorm` -- the names are
# legacy from when LLaMA was first written using LayerNorm naming.
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_base: float,
        rms_eps: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            qkv_bias=qkv_bias,
        )
        self.mlp_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: "PagedRequestCache | list[PagedRequestCache] | None" = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # Sublayer 1: attention. The cache lives inside attention; we just
        # pass it through. Type may be a single cache OR a list (batched
        # decode); attention dispatches.
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, layer_idx=layer_idx)
        # Sublayer 2: MLP. Position-agnostic, no cache involvement, batches
        # cleanly across rows.
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------
#
# The forward pass top to bottom:
#
#   input_ids (B, S)
#       -> embed lookup           -> (B, S, H)
#       -> 22 transformer blocks  -> (B, S, H)   each block reads/writes the
#                                                residual stream
#       -> final RMSNorm          -> (B, S, H)
#       -> LM head Linear         -> (B, S, vocab) -- next-token logits
#
# Why a final RMSNorm before the LM head?
#
#   With pre-norm, the residual stream `x` exits the last block UN-normed.
#   The LM head expects a normalized input distribution; without the final
#   norm the model's outputs would be at the mercy of however the stream
#   happened to scale by layer 22. One more RMSNorm fixes that.
#
# Why is the LM head a separate Linear (not tied to the embedding)?
#
#   Some models tie the embedding and LM head to save params (GPT-2 does).
#   TinyLlama does NOT tie -- `tie_word_embeddings=False` in its config --
#   so `lm_head.weight` is a distinct learned matrix. We have to match.
# ---------------------------------------------------------------------------


class LlamaConfig:
    """Plain config object. Matches the fields we need from HF's config.json."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        rms_norm_eps: float,
        rope_theta: float,
        tie_word_embeddings: bool,
        attention_bias: bool = False,
        model_type: str = "llama",
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        # Qwen2 keeps a learned Q/K/V bias; LLaMA does not. Threaded into every
        # attention block so the right family loads without retuning anything.
        self.attention_bias = attention_bias
        self.model_type = model_type

    @classmethod
    def from_hf_json(cls, path: str | Path) -> "LlamaConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        model_type = d.get("model_type", "llama")
        # Architecture gate: the from-scratch engine faithfully represents the
        # LLaMA family and Qwen2 (same RMSNorm + SwiGLU + GQA + RoPE skeleton,
        # Qwen2 just adds a Q/K/V bias). Anything else (e.g. Gemma's GeGLU +
        # embedding scaling + (1+w) norms) would load WRONG, so we refuse rather
        # than silently produce garbage -- the caller turns this into a skip.
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedArchitectureError(
                f"model_type '{model_type}' is not supported by the from-scratch "
                f"engine (supported: {sorted(SUPPORTED_MODEL_TYPES)})")
        # Qwen2 uses a Q/K/V bias whether or not the JSON spells it out; LLaMA
        # honours an explicit attention_bias flag (default False).
        attention_bias = bool(d.get("attention_bias", model_type == "qwen2"))
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            # Some configs default this to num_attention_heads (i.e. MHA);
            # TinyLlama explicitly sets it to 4.
            num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            max_position_embeddings=d["max_position_embeddings"],
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 10000.0),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            attention_bias=attention_bias,
            model_type=model_type,
        )


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding: vocab_size x hidden. Looks up a learned vector
        # per token id. This is the input to the residual stream.
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Stack of decoder blocks. Each writes a delta into the residual stream.
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_seq_len=config.max_position_embeddings,
                rope_base=config.rope_theta,
                rms_eps=config.rms_norm_eps,
                qkv_bias=config.attention_bias,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final norm before LM head (see comment block above).
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head: hidden -> vocab. No bias, matching LLaMA.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # If the config asks for weight tying, we point lm_head at embed.
        # TinyLlama doesn't, but we honor the flag for completeness.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: PagedRequestCache | list[PagedRequestCache] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) int64. In prefill S = prompt length. In a
                cached decode step S = 1.
            kv_cache: one of:
                * None: no cache, recompute everything.
                * PagedRequestCache: single-request prefill or decode.
                * list[PagedRequestCache]: batched decode across B requests,
                    each with its own cache. S must be 1; each cache is at
                    its own current seq_len. Attention dispatches.

        Returns:
            logits: (B, S, vocab_size).
        """
        # Be device-transparent: callers (tests, scheduler, server) may pass
        # a CPU tensor even when the model is on CUDA. The .to() is a
        # no-op when devices already match, so this costs nothing in the
        # steady-state hot path.
        input_ids = input_ids.to(self.embed.weight.device)
        # Embed: (B, S) -> (B, S, H). Initial state of the residual stream.
        x = self.embed(input_ids)

        # Walk the stack, passing each block its own index so attention can
        # find its slot in the cache.
        for i, block in enumerate(self.layers):
            x = block(x, kv_cache=kv_cache, layer_idx=i)

        # Final norm + projection to vocab.
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    # -----------------------------------------------------------------------
    # Autoregressive greedy generation (Day 4: no KV cache yet)
    # -----------------------------------------------------------------------
    #
    # The naive approach: at each step, re-run the *entire* prompt + all
    # previously generated tokens through the model. This is O(N^2) total
    # work for N generated tokens, because position t recomputes attention
    # over positions 0..t-1 every time.
    #
    # Why do it the naive way first?
    #   1. It is identical to the parity-tested forward pass, so any drift
    #      between us and HF can only come from generation logic, not from
    #      the model itself. That's a clean baseline before adding caching.
    #   2. The KV-cache version (Day 5) will be a strict optimization with
    #      the same outputs. Having the naive reference makes that diff
    #      testable.
    #
    # Greedy decoding: at each step pick argmax over the vocab. This is
    # deterministic -- no temperature, no sampling. Equivalent to HF's
    # `generate(do_sample=False, num_beams=1)`.
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Greedy autoregressive generation.

        Args:
            input_ids: (B, S_prompt) int64.
            max_new_tokens: cap on tokens generated.
            eos_token_id: if provided, stop early when all batch rows emit it.
            use_cache: True = O(N) cached decode. False = O(N^2) naive
                recompute (kept as a parity reference and a clean fallback).

        Returns:
            (B, S_prompt + n_generated) int64. Prompt + completion.
        """
        # Move once at the entry point. _generate_naive and
        # _generate_cached both `torch.cat` model outputs (on DEVICE)
        # onto the caller-supplied tensor; if the latter is on CPU the
        # cat raises. Moving here keeps the inner methods clean.
        input_ids = input_ids.to(self.embed.weight.device)
        if use_cache:
            return self._generate_cached(input_ids, max_new_tokens, eos_token_id)
        return self._generate_naive(input_ids, max_new_tokens, eos_token_id)

    def _make_single_request_cache(
        self,
        prompt_len: int,
        max_new_tokens: int,
        block_size: int = 16,
    ) -> PagedRequestCache:
        """Build a paged pool sized for ONE request + return its view.

        Used by `_generate_cached` so that solo generation goes through
        the same paged-cache code as the scheduler. The pool is sized to
        the request's worst-case footprint (prompt + max_new_tokens).
        """
        config = self.config
        head_dim = config.hidden_size // config.num_attention_heads
        n_blocks = (prompt_len + max_new_tokens + block_size - 1) // block_size
        n_prefill = (prompt_len + block_size - 1) // block_size
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        pool = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_blocks=n_blocks,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        pool.admit_request(
            request_id="solo",
            prefill_blocks_needed=n_prefill,
            total_blocks_needed=n_blocks,
        )
        return PagedRequestCache(pool, "solo", num_layers=config.num_hidden_layers)

    def _generate_naive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        """O(N^2) reference: recompute the full forward each step."""
        generated = input_ids
        for _ in range(max_new_tokens):
            # Full forward over the WHOLE current sequence.
            logits = self(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    def _generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        """O(N) cached decode using a paged KV cache.

        Two distinct phases:
          1. PREFILL: forward the entire prompt once. The cache absorbs
             the prompt's K/V at every layer. We then pick the next token
             off the LAST position's logits.
          2. DECODE: feed only the just-predicted single token to the model.
             Attention will RoPE-rotate it at position cache.seq_len(),
             append its K/V, and let Q attend to the full cached past.
             We loop this max_new_tokens - 1 more times.

        Both phases call the same forward(). The only difference is the
        input length and the cache's current seq_len, which together
        determine RoPE's position offset and the causal-mask flip.

        For solo generation we build a tiny one-request paged pool sized
        to the worst-case footprint of THIS request. The pool/request-view
        split is overkill for a single request, but using it here means we
        exercise exactly the same code path the scheduler does, so any
        paged-cache bug surfaces in both places.
        """
        cache = self._make_single_request_cache(
            prompt_len=input_ids.shape[1],
            max_new_tokens=max_new_tokens,
        )

        # ---- Phase 1: prefill ----
        # Feed the prompt. After this, the cache holds K/V for positions
        # [0 .. S_prompt - 1] in every layer.
        logits = self(input_ids, kv_cache=cache)
        # Greedy pick from the prompt's last position. This is the FIRST
        # generated token.
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            return generated

        # ---- Phase 2: decode, one token at a time ----
        # We've already emitted one new token above, so loop max_new_tokens - 1
        # more times.
        for _ in range(max_new_tokens - 1):
            # Crucially, the input is just the single newest token. The
            # cache provides everything else.
            logits = self(next_token, kv_cache=cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # -----------------------------------------------------------------------
    # INT8 (W8A8) quantization
    # -----------------------------------------------------------------------
    #
    # Swap every block's fp MultiHeadAttention for a QuantizedMultiHeadAttention,
    # quantizing the q/k/v/o projection weights to int8 in place. This is an
    # OFFLINE, in-place transform: call it once after the fp weights are loaded.
    #
    # The import is lazy on purpose. QuantizedMultiHeadAttention pulls in Triton
    # (via the quantization kernels); a top-level import here would force every
    # consumer of model.py -- including the CPU-only correctness tests -- to
    # have Triton installed. Keeping it inside the method means only callers who
    # actually quantize pay that dependency.
    # -----------------------------------------------------------------------

    def quantize(self) -> "LlamaModel":
        """Replace all attention blocks with int8 W8A8 versions, in place.

        Returns ``self`` so callers can write ``model = load(...).quantize()``.
        """
        from src.engine.kernels.quant_attention import QuantizedMultiHeadAttention

        cfg = self.config
        device = next(self.parameters()).device
        for block in self.layers:
            quantized = QuantizedMultiHeadAttention.from_float(
                block.attn,
                max_seq_len=cfg.max_position_embeddings,
                rope_base=cfg.rope_theta,
            )
            # from_float copies tensors off the source (already on `device`), but
            # be explicit so a mismatched source can't smuggle a stray CPU tensor
            # into the hot path.
            block.attn = quantized.to(device)

        # Sanity check: every projection is now int8 and on the right device.
        # This is the "verify weight loading works after quantization" guard --
        # it fails loudly here rather than as a cryptic dtype error mid-forward.
        for block in self.layers:
            for proj in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
            ):
                assert proj.weight_int8.dtype == torch.int8, "projection not int8"
                assert proj.weight_int8.device == device, "projection on wrong device"
        return self


# ---------------------------------------------------------------------------
# Weight loading from a Hugging Face checkpoint
# ---------------------------------------------------------------------------
#
# Strategy:
#   1. Download `config.json` and `model.safetensors` from the HF hub via
#      huggingface_hub (no `transformers` import needed).
#   2. Build a LlamaConfig from the JSON.
#   3. Construct our LlamaModel.
#   4. Read the state_dict from the safetensors file.
#   5. Rename HF parameter keys to OUR parameter names.
#   6. Cast to fp32 (TinyLlama ships in bf16; we want deterministic parity).
#   7. Call load_state_dict(strict=True) -- this is the load-time assertion
#      that we got every param accounted for, with the right shapes.
#
# HF -> our names:
#   model.embed_tokens.weight                            -> embed.weight
#   model.layers.{i}.input_layernorm.weight              -> layers.{i}.attn_norm.weight
#   model.layers.{i}.self_attn.q_proj.weight             -> layers.{i}.attn.q_proj.weight
#   model.layers.{i}.self_attn.k_proj.weight             -> layers.{i}.attn.k_proj.weight
#   model.layers.{i}.self_attn.v_proj.weight             -> layers.{i}.attn.v_proj.weight
#   model.layers.{i}.self_attn.o_proj.weight             -> layers.{i}.attn.o_proj.weight
#   model.layers.{i}.post_attention_layernorm.weight     -> layers.{i}.mlp_norm.weight
#   model.layers.{i}.mlp.gate_proj.weight                -> layers.{i}.mlp.gate_proj.weight
#   model.layers.{i}.mlp.up_proj.weight                  -> layers.{i}.mlp.up_proj.weight
#   model.layers.{i}.mlp.down_proj.weight                -> layers.{i}.mlp.down_proj.weight
#   model.norm.weight                                    -> final_norm.weight
#   lm_head.weight                                       -> lm_head.weight
# ---------------------------------------------------------------------------


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Architectures the from-scratch engine can load FAITHFULLY (not just without
# erroring). LLaMA family + Qwen2 share the RMSNorm/SwiGLU/GQA/RoPE skeleton;
# Qwen2 only differs by a learned Q/K/V bias (handled via config.attention_bias).
SUPPORTED_MODEL_TYPES = {"llama", "qwen2"}


class UnsupportedArchitectureError(RuntimeError):
    """Raised when a checkpoint's architecture the engine cannot represent
    exactly (e.g. Gemma's GeGLU + embedding scaling). Callers catch this to
    skip the model rather than load wrong weights."""


def _remap_hf_key(hf_key: str) -> str | None:
    """Translate one HF parameter name to our module's name.

    Returns None for keys we deliberately ignore (none currently).
    """
    # Top-level rewrites.
    if hf_key == "model.embed_tokens.weight":
        return "embed.weight"
    if hf_key == "model.norm.weight":
        return "final_norm.weight"
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # Per-layer rewrites. HF emits `model.layers.{i}.<rest>`.
    prefix = "model.layers."
    if hf_key.startswith(prefix):
        rest = hf_key[len(prefix):]
        # rest looks like "{i}.input_layernorm.weight" etc.
        idx_str, sublayer = rest.split(".", 1)
        sublayer_map = {
            "input_layernorm.weight":            "attn_norm.weight",
            "self_attn.q_proj.weight":           "attn.q_proj.weight",
            "self_attn.k_proj.weight":           "attn.k_proj.weight",
            "self_attn.v_proj.weight":           "attn.v_proj.weight",
            "self_attn.o_proj.weight":           "attn.o_proj.weight",
            # Qwen2's Q/K/V bias vectors (absent in LLaMA checkpoints).
            "self_attn.q_proj.bias":             "attn.q_proj.bias",
            "self_attn.k_proj.bias":             "attn.k_proj.bias",
            "self_attn.v_proj.bias":             "attn.v_proj.bias",
            "post_attention_layernorm.weight":   "mlp_norm.weight",
            "mlp.gate_proj.weight":              "mlp.gate_proj.weight",
            "mlp.up_proj.weight":                "mlp.up_proj.weight",
            "mlp.down_proj.weight":              "mlp.down_proj.weight",
        }
        if sublayer in sublayer_map:
            return f"layers.{idx_str}.{sublayer_map[sublayer]}"
    return None


def load_tinyllama_from_hf(
    model_name: str = MODEL_NAME,
    dtype: torch.dtype = torch.float32,
) -> tuple[LlamaModel, LlamaConfig]:
    """Build a LlamaModel and populate it from the HF safetensors checkpoint.

    Despite the historical name, this loads any LLaMA-FAMILY checkpoint by id:
    the default is TinyLlama (so every existing caller and test is unchanged),
    but it also loads Qwen2 (which adds a Q/K/V bias and ties its embeddings).
    Architectures the from-scratch engine cannot represent exactly raise
    UnsupportedArchitectureError (via LlamaConfig.from_hf_json) so the caller can
    skip rather than load wrong weights. See `load_model_from_hf` for the
    architecture-neutral alias.

    We deliberately avoid `transformers.AutoModelForCausalLM` -- only
    `huggingface_hub` (which is just a download client) and `safetensors`
    (which is a tensor file format).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # 1) Download config + weights into the HF cache. These calls are
    #    cache-aware: if the files are already on disk they just return
    #    the path without re-downloading.
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

    # 2) Build our config from HF's config.json.
    config = LlamaConfig.from_hf_json(config_path)

    # 3) Construct an EMPTY model with the right architecture.
    model = LlamaModel(config)

    # 4) Read the raw HF state_dict off disk.
    hf_state = load_file(weights_path)

    # 5) Rename HF keys to ours and cast to the requested dtype.
    new_state: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for hf_key, tensor in hf_state.items():
        our_key = _remap_hf_key(hf_key)
        if our_key is None:
            unmapped.append(hf_key)
            continue
        new_state[our_key] = tensor.to(dtype)
    if unmapped:
        raise RuntimeError(
            f"Unrecognized HF parameters (refusing to silently drop): {unmapped}"
        )

    # 6) Strict load. This will raise loudly if any of our params are
    #    missing from the HF dict, or vice versa, or if shapes don't match.
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    # We use strict=False so we can inspect what's missing/unexpected and
    # produce a clearer error. RoPE cos/sin buffers aren't in the state
    # dict (they're not persistent), so they show up in `missing` and that's
    # fine.
    rope_buffers = {f"layers.{i}.attn.rope_cos" for i in range(config.num_hidden_layers)} | \
                   {f"layers.{i}.attn.rope_sin" for i in range(config.num_hidden_layers)}
    ignorable = set(rope_buffers)
    # Tied-embedding models (e.g. Qwen2-0.5B) ship NO lm_head.weight: the LM head
    # IS the embedding matrix. LlamaModel.__init__ already pointed lm_head.weight
    # at embed.weight, so loading embed.weight also populates the head -- the
    # "missing" lm_head.weight here is expected, not an error.
    if config.tie_word_embeddings:
        ignorable.add("lm_head.weight")
    real_missing = [m for m in missing if m not in ignorable]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Weight load mismatch. Missing: {real_missing}. Unexpected: {unexpected}"
        )

    # 7) Move to eval mode and cast the whole module (covers buffers and
    #    any non-parameter state) to the requested dtype, THEN move to
    #    the engine's target device (DEVICE = cuda if available else cpu).
    #
    #    Why dtype-cast on CPU first, then .to(DEVICE):
    #      The HF safetensors load gives us a CPU state_dict. Casting on
    #      CPU avoids materialising a second copy in VRAM during the
    #      cast itself; the single transfer to DEVICE happens after the
    #      dtype is already settled. For TinyLlama-1.1B in fp32 (~4.4 GB)
    #      this matters on 8 GB cards.
    model.eval()
    model.to(dtype)
    model.to(DEVICE)
    return model, config


# Architecture-neutral alias. New code (e.g. the cross-model harness) should call
# this; load_tinyllama_from_hf stays as the back-compat name every existing
# caller already imports.
load_model_from_hf = load_tinyllama_from_hf


# ---------------------------------------------------------------------------
# Day 2 entry point: load, predict one greedy next token, print it.
# ---------------------------------------------------------------------------


def main() -> None:
    # The tokenizer is just a deterministic string<->id mapping. Using HF's
    # tokenizer here is fine -- the "from scratch" requirement is for the
    # model's forward pass, not for tokenization (which is a huge BPE table).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _config = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)

    print(f"Device: {DEVICE}")

    prompt = "The capital of France is"
    # The .to(DEVICE) is technically redundant -- forward() does the same
    # move -- but doing it here makes the print-out below honest about
    # what device the input is on for the timed greedy step.
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    # Greedy generation: 20 new tokens, stop early on EOS.
    output_ids = model.generate(
        input_ids,
        max_new_tokens=20,
        eos_token_id=tokenizer.eos_token_id,
    )

    # output_ids includes the prompt; decode the whole thing so the user
    # can read prompt + completion as one continuous string.
    # output_ids may live on DEVICE; decoder runs on CPU ints, so move first.
    output_text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
