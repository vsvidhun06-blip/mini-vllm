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

ROUTING STATE IS STICKY, AND `None` MEANS BASE
----------------------------------------------
The active routing plan persists across forwards until it is changed -- the
scheduler relies on that (it sets the plan once per step, and generate() sets it
once for a whole prefill+decode sequence). So `forward` has to distinguish two
different things a caller can mean:

    forward(ids)                    -- argument OMITTED: keep the current plan.
    forward(ids, adapter_ids=None)  -- argument EXPLICITLY None: route to base.

A plain `None` default cannot express both, so the omitted case is spelled with
the `UNSET` sentinel below. `None` therefore means "base" everywhere in this
module -- as a whole-batch plan, as a per-row entry, and as generate()'s
adapter_id -- with no stale-state exception.

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

from enum import Enum, auto
from typing import TYPE_CHECKING, Final, List, Literal, Optional, Union

import torch
import torch.nn as nn

from src.engine.lora import ActiveSpec, LoRALinear, LoRAManager, normalize_routing_plan

if TYPE_CHECKING:
    from src.engine.model import LlamaModel

# The four attention projections we wrap, by attribute name on MultiHeadAttention.
_TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

# A routing plan a caller can hand to forward(): one adapter for every row, a
# per-row list (None entry = that row is base-only), or None = base for all rows.
RoutingSpec = Union[str, List[Optional[str]], None]


class _Sentinel(Enum):
    """Single-member enum used as a typed sentinel.

    The enum form (rather than `object()`) is what static checkers can narrow:
    `Literal[_Sentinel.UNSET]` in the union lets `is not UNSET` refine the
    parameter down to RoutingSpec inside the branch.
    """
    UNSET = auto()

    def __repr__(self) -> str:                      # nicer signature rendering
        return "UNSET"


#: Default for forward()'s `adapter_ids`: the argument was omitted, so the
#: existing routing state is preserved. Distinct from an explicit None (= base).
UNSET: Final = _Sentinel.UNSET




def model_routing_plan(model) -> tuple[str | None, ...] | str | None:
    """Return the model's current routing plan in canonical form.

    Plain LlamaModel instances have no LoRA routing state and return None.
    A single adapter is returned as its string id; mixed routing is returned
    as a tuple so graph-capture comparisons are stable and hashable.
    """
    layers = getattr(model, "_lora_layers", None)
    if not layers:
        return None
    return normalize_routing_plan(layers[0]._active)

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
        None (rather than a list) clears routing back to the base path for the
        whole batch. The list is shared by reference across all wrapped layers --
        they all see the same per-row plan.

        Unlike forward(), there is no omitted case here: calling this method is
        itself the request to change routing, so None is unambiguously "base".
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
        adapter_ids: RoutingSpec | Literal[_Sentinel.UNSET] = UNSET,
    ):
        """Drop-in for LlamaModel.forward, plus optional adapter routing.

        adapter_ids:
            * omitted -- preserve the routing state last set (set_adapter /
                         set_batch_adapters / a previous forward), or base if
                         nothing has ever been set. This is the scheduler's and
                         generate()'s path: they set the plan, then forward.
            * None    -- explicitly route every row to the base path, discarding
                         any previously-set plan. NOT "leave state alone".
            * str     -- apply this one adapter to every row.
            * list    -- per-row routing (len == batch size), None entry = base.

        Anything other than the omitted case sets the routing state, so it also
        applies to subsequent forwards until changed -- decode steps of one
        generation inherit the prefill's plan, which is what the KV-cache loop
        wants.
        """
        if adapter_ids is not UNSET:
            # str and None are both whole-batch plans; a list is per-row.
            if adapter_ids is None or isinstance(adapter_ids, str):
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
