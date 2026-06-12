"""
TensorParallelLlamaModel -- a LlamaModel with attention + MLP weights sharded
across N tensor-parallel ranks, simulated in a single process.

world_size == 1 is a NO-OP: the wrapper delegates straight to the base model, so
it is byte-identical to LlamaModel. For world_size > 1 the wrapper holds N
per-rank shards of every attention and MLP block (built from the loaded
checkpoint by ColumnParallel/RowParallel.from_linear) and runs a SIMULATED
forward that, for each transformer layer, computes every rank's partial output
and sums them -- which is exactly what the row-parallel all-reduce does across
real devices, so the result equals the dense model (to float precision).

See tensor_parallel.py for the sharding math and the real-multi-GPU extension
path. The simulated forward here is the no-KV-cache (prefill) path; that is all
the benchmark and the correctness tests need, and the docstring there explains
what wiring the paged cache + decode through TP would add.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.engine.tensor_parallel import (
    TensorParallelAttention,
    TensorParallelConfig,
    TensorParallelMLP,
)

if TYPE_CHECKING:
    from src.engine.model import LlamaModel


class TensorParallelLlamaModel(nn.Module):
    """Tensor-parallel wrapper over a LlamaModel (single-process simulation)."""

    def __init__(self, base_model: "LlamaModel", config: TensorParallelConfig | None = None) -> None:
        super().__init__()
        self.model = base_model
        self.tp_config = config or TensorParallelConfig(world_size=1, rank=0)
        self.world_size = self.tp_config.world_size
        # Per-layer rank shards, populated by replace_with_tp_layers when
        # world_size > 1. Empty (and unused) at world_size == 1.
        self.tp_attn: nn.ModuleList = nn.ModuleList()
        self.tp_mlp: nn.ModuleList = nn.ModuleList()

    # ---- drop-in attributes -------------------------------------------------

    @property
    def config(self):
        return self.model.config

    @property
    def layers(self):
        return self.model.layers

    # ---- sharding -----------------------------------------------------------

    def replace_with_tp_layers(self, world_size: int) -> "TensorParallelLlamaModel":
        """Shard every attention + MLP block across `world_size` ranks.

        world_size == 1 is a no-op (the base model is already a single rank).
        Otherwise we build N per-rank shards of each block from the dense
        checkpoint weights; the dense layers stay resident (a real deployment
        would keep only this rank's shard on each GPU -- here we keep all ranks
        in one process to simulate the group).
        """
        self.world_size = world_size
        self.tp_config = TensorParallelConfig(world_size=world_size, rank=self.tp_config.rank)
        self.tp_attn = nn.ModuleList()
        self.tp_mlp = nn.ModuleList()
        if world_size == 1:
            return self  # no-op: identical to the base model

        for block in self.model.layers:
            attn_ranks = nn.ModuleList([
                TensorParallelAttention.from_attention(block.attn, world_size, r)
                for r in range(world_size)
            ])
            mlp_ranks = nn.ModuleList([
                TensorParallelMLP.from_mlp(block.mlp, world_size, r)
                for r in range(world_size)
            ])
            self.tp_attn.append(attn_ranks)
            self.tp_mlp.append(mlp_ranks)
        return self

    # ---- forward ------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, kv_cache=None) -> torch.Tensor:
        """Logits for `input_ids`. world_size==1 delegates to the base model.

        For world_size>1 this runs the simulated TP prefill: per layer, sum each
        rank's partial attention/MLP output (the in-process stand-in for the
        row-parallel all-reduce). No KV cache in the simulated path.
        """
        if self.world_size == 1:
            return self.model(input_ids, kv_cache=kv_cache)
        if kv_cache is not None:
            raise NotImplementedError(
                "the single-process TP simulation runs the no-cache prefill path; "
                "KV-cache/decode under TP is the documented next step"
            )

        m = self.model
        input_ids = input_ids.to(m.embed.weight.device)
        x = m.embed(input_ids)
        for li, block in enumerate(m.layers):
            # Attention sublayer: sum the per-rank partial outputs == dense attn.
            h = block.attn_norm(x)
            attn_out = self.tp_attn[li][0](h)
            for r in range(1, self.world_size):
                attn_out = attn_out + self.tp_attn[li][r](h)
            x = x + attn_out
            # MLP sublayer: same per-rank partial sum.
            h2 = block.mlp_norm(x)
            mlp_out = self.tp_mlp[li][0](h2)
            for r in range(1, self.world_size):
                mlp_out = mlp_out + self.tp_mlp[li][r](h2)
            x = x + mlp_out
        x = m.final_norm(x)
        return m.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 eos_token_id: int | None = None) -> torch.Tensor:
        """Greedy generation through the (no-cache) TP forward.

        O(N^2) recompute -- the simulated TP path has no KV cache -- but correct
        and enough to measure TTFT/TPOT in the benchmark. world_size==1 routes
        through the base model's cached generate.
        """
        if self.world_size == 1:
            return self.model.generate(input_ids, max_new_tokens, eos_token_id=eos_token_id)
        generated = input_ids.to(self.model.embed.weight.device)
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None and (nxt == eos_token_id).all():
                break
        return generated

    # ---- factory ------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, world_size: int) -> "TensorParallelLlamaModel":
        """Load TinyLlama from `model_path` and shard it across `world_size`."""
        from src.engine.model import load_tinyllama_from_hf

        base, _ = load_tinyllama_from_hf(model_path)
        base.eval()
        tp = cls(base, TensorParallelConfig(world_size=world_size, rank=0))
        tp.replace_with_tp_layers(world_size)
        return tp
