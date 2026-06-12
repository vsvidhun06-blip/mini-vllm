"""
LoRALlamaModel -- a LlamaModel with LoRA-wrapped attention projections and
per-request adapter routing.

WHAT IT DOES
------------
Wraps an existing LlamaModel and replaces every attention block's q/k/v/o
projection (the four `nn.Linear`s in MultiHeadAttention) with a LoRALinear that
shares the original weights. With no adapter active the model is numerically
IDENTICAL to the base LlamaModel (LoRALinear's zero-overhead path), so wrapping
is free until an adapter is selected.

Two activation modes, both routed down to every wrapped LoRALinear:
  * set_adapter(id)            -- one adapter for the whole next forward.
  * set_batch_adapters([...])  -- per-row routing: row i of the batch uses
                                  adapter ids[i] (None = base only). This is the
                                  mixed-adapter batching that lets one batched
                                  forward serve several fine-tunes at once.

DROP-IN FOR THE SCHEDULER
-------------------------
forward(input_ids, kv_cache=...) matches LlamaModel.forward, and `.config` /
`.parameters()` / `.generate()` are forwarded, so the ContinuousBatchScheduler
can drive a LoRALlamaModel exactly like a LlamaModel. The scheduler sets the
batch adapters (in active-batch order) right before each forward via the
`set_batch_adapters` hook -- a plain LlamaModel simply lacks that method, so the
scheduler's routing is a no-op there and existing behaviour is unchanged.

We only LoRA-wrap the attention projections (q/k/v/o), matching the most common
LoRA target set; MLP and embedding stay base. Extending to gate/up/down would be
the same pattern.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.engine.lora import ActiveSpec, LoRALinear, LoRAManager

if TYPE_CHECKING:
    from src.engine.model import LlamaModel

# The four attention projections we wrap, by attribute name on MultiHeadAttention.
_TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def layer_name(layer_idx: int, proj: str) -> str:
    """Canonical adapter-weight key for one wrapped projection."""
    return f"layers.{layer_idx}.attn.{proj}"


class LoRALlamaModel(nn.Module):
    """LlamaModel + LoRA-wrapped attention projections + adapter routing."""

    def __init__(self, base_model: "LlamaModel", manager: LoRAManager) -> None:
        super().__init__()
        self.model = base_model
        self.manager = manager
        # Keep direct references to every wrapped projection so set_adapter /
        # set_batch_adapters is an O(#projections) attribute write, not a tree
        # walk every step.
        self._lora_layers: list[LoRALinear] = []

        for i, block in enumerate(base_model.layers):
            attn = block.attn
            for proj in _TARGET_PROJECTIONS:
                base_linear = getattr(attn, proj)
                if isinstance(base_linear, LoRALinear):
                    wrapped = base_linear            # idempotent re-wrap guard
                else:
                    wrapped = LoRALinear(base_linear, manager, layer_name(i, proj))
                    setattr(attn, proj, wrapped)
                self._lora_layers.append(wrapped)

    # ---- forwarded attributes so this is a drop-in for LlamaModel -----------

    @property
    def config(self):
        return self.model.config

    @property
    def layers(self):
        return self.model.layers

    # ---- adapter routing ----------------------------------------------------

    def set_adapter(self, adapter_id: str | None) -> None:
        """Activate one adapter (or None = base) for the next forward pass."""
        for layer in self._lora_layers:
            layer.set_active(adapter_id)

    def set_batch_adapters(self, adapter_ids: list[str | None] | None) -> None:
        """Per-row routing for a mixed-adapter batch.

        adapter_ids[i] is the adapter for batch row i (None = base only). Passing
        None clears routing back to the base path. The list is shared by
        reference across all wrapped layers -- they all see the same per-row plan.
        """
        active: ActiveSpec = adapter_ids
        for layer in self._lora_layers:
            layer.set_active(active)

    def clear_adapters(self) -> None:
        self.set_batch_adapters(None)

    # ---- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache=None,
        adapter_ids: "list[str | None] | str | None" = None,
    ):
        """Drop-in for LlamaModel.forward, plus optional adapter routing.

        adapter_ids:
            * None  -- use whatever routing was last set (set_adapter /
                       set_batch_adapters), or base if nothing set.
            * str   -- apply this one adapter to every row for this call.
            * list  -- per-row routing for this call (len == batch size).
        When given, it is applied for THIS call (it sets the routing state).
        """
        if adapter_ids is not None:
            if isinstance(adapter_ids, str):
                self.set_adapter(adapter_ids)
            else:
                self.set_batch_adapters(adapter_ids)
        return self.model(input_ids, kv_cache=kv_cache)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None,
                 use_cache=True, adapter_id: str | None = None):
        """Greedy generation under a single adapter (or base if None).

        Sets the adapter once and delegates to the base model's generate, which
        calls self.model(...) internally -- the routing state persists across the
        prefill + decode forwards of one generation.
        """
        self.set_adapter(adapter_id)
        return self.model.generate(
            input_ids, max_new_tokens, eos_token_id=eos_token_id, use_cache=use_cache
        )


# ---------------------------------------------------------------------------
# Synthetic-adapter helper (tests, benchmark, and the demo /adapters endpoint).
# ---------------------------------------------------------------------------


def random_adapter_weights(
    model: "LlamaModel | LoRALlamaModel",
    rank: int,
    seed: int = 0,
    scale_init: float = 0.02,
    projections: tuple[str, ...] = _TARGET_PROJECTIONS,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build a random {layer_name: (A, B)} weights_dict sized to `model`.

    A is (rank, in), B is (out, rank), both small-magnitude Gaussian so the delta
    is a gentle perturbation (B is NOT zero-initialised here -- a real adapter
    would train from B=0, but tests need a non-trivial, deterministic delta).
    Useful wherever a real PEFT checkpoint isn't available.
    """
    base = model.model if isinstance(model, LoRALlamaModel) else model
    g = torch.Generator().manual_seed(seed)
    device = next(base.parameters()).device
    dtype = next(base.parameters()).dtype
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, block in enumerate(base.layers):
        attn = block.attn
        for proj in projections:
            linear = getattr(attn, proj)
            base_linear = linear.base if isinstance(linear, LoRALinear) else linear
            in_f = base_linear.in_features
            out_f = base_linear.out_features
            A = torch.randn(rank, in_f, generator=g, dtype=dtype) * scale_init
            B = torch.randn(out_f, rank, generator=g, dtype=dtype) * scale_init
            weights[layer_name(i, proj)] = (A.to(device), B.to(device))
    return weights
