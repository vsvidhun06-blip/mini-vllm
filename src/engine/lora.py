"""
Multi-LoRA adapter serving -- hot-swap low-rank adapters per request without
touching base weights.

WHAT LoRA IS
------------
A fine-tuned linear layer learns a weight delta dW on top of a frozen base W:
    y = (W + dW) x
LoRA (Hu et al. 2021) constrains dW to be LOW RANK -- dW = (alpha/r) * B @ A,
with A: (r, d_in) and B: (d_out, r), r << d. So instead of storing a full
d_out x d_in delta you store two skinny matrices. At inference:

    y = W x  +  (alpha / r) * B (A x)
        \___/    \_________________/
        base            LoRA delta

The base forward is unchanged; the delta is two small matmuls. That is the whole
trick that makes adapters cheap to store and -- crucially for serving -- cheap to
SWAP: the base weights stay resident on the GPU, and switching "personalities"
is just pointing at a different (A, B) pair.

WHY THIS MATTERS FOR SERVING
----------------------------
One base model + N small adapters lets a single deployment serve N fine-tunes
(per-customer, per-task, per-language) from ONE copy of the 1.1B base weights.
The adapters are megabytes, not gigabytes. And because the base GEMM is shared,
a single batched forward can serve a MIX of adapters at once (different rows of
the batch using different adapters) -- "mixed-adapter batching", the property
that makes multi-tenant LoRA serving actually efficient. See LoRALinear's
per-row routing and lora_model.LoRALlamaModel.

WHAT'S HERE
-----------
  * LoRAAdapter -- a registered adapter: id, rank, alpha, and per-layer (A, B).
  * LoRAManager -- an LRU registry of adapters (cap: 8 resident). Load, fetch.
  * LoRALinear  -- wraps an nn.Linear; base forward + optional LoRA delta, with
                   a genuine zero-overhead path when no adapter is active, and
                   per-row routing for mixed-adapter batches.

The base weights live in the wrapped nn.Linear and are never mutated. Adapters
carry only their A/B matrices. Nothing here depends on the server or scheduler.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# A per-row adapter routing entry is either an adapter id or None (= base only).
# The "active" state of a LoRALinear is one of:
#   * None            -> base path, zero overhead
#   * str             -> one adapter applied to every row
#   * list[str|None]  -> per-row routing (mixed-adapter batch), len == batch
ActiveSpec = "str | list[str | None] | None"


@dataclass
class LoRAAdapter:
    """One registered adapter.

    weight_A[layer] : (rank, in_features)  -- the down-projection A.
    weight_B[layer] : (out_features, rank) -- the up-projection B.
    Keyed by the layer name a LoRALinear was wrapped with (e.g.
    "layers.0.attn.q_proj"). A layer absent from both dicts gets no delta.

    `scale` = alpha / rank is the standard LoRA scaling, applied to the delta.
    """
    adapter_id: str
    rank: int
    alpha: float
    weight_A: dict[str, torch.Tensor] = field(default_factory=dict)
    weight_B: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


class LoRAManager:
    """LRU registry of resident adapters.

    Holds at most `max_adapters` adapters in memory at once -- mirroring a real
    multi-tenant server that can't keep unbounded adapters resident. When the
    cap is exceeded the LEAST-recently-used adapter is evicted (load and fetch
    both count as a use), so hot adapters stay and cold ones fall out.
    """

    def __init__(self, max_adapters: int = 8) -> None:
        self.max_adapters = max_adapters
        # OrderedDict as an LRU: most-recently-used at the end.
        self._registry: "OrderedDict[str, LoRAAdapter]" = OrderedDict()
        self.num_evicted = 0

    def load_adapter(
        self,
        adapter_id: str,
        rank: int,
        alpha: float,
        weights_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> LoRAAdapter:
        """Register (or replace) an adapter.

        weights_dict maps layer_name -> (A, B) with A (rank, in) and B (out, rank).
        Returns the registered LoRAAdapter. If the registry is full and this is a
        new id, the least-recently-used adapter is evicted first.
        """
        weight_A: dict[str, torch.Tensor] = {}
        weight_B: dict[str, torch.Tensor] = {}
        for layer_name, (A, B) in weights_dict.items():
            if A.shape[0] != rank or B.shape[1] != rank:
                raise ValueError(
                    f"adapter {adapter_id!r} layer {layer_name!r}: expected "
                    f"A (rank={rank}, in) and B (out, rank={rank}), got "
                    f"A{tuple(A.shape)} B{tuple(B.shape)}"
                )
            weight_A[layer_name] = A
            weight_B[layer_name] = B

        adapter = LoRAAdapter(
            adapter_id=adapter_id, rank=rank, alpha=alpha,
            weight_A=weight_A, weight_B=weight_B,
        )
        # Replace-in-place keeps the id but refreshes recency.
        self._registry.pop(adapter_id, None)
        self._registry[adapter_id] = adapter
        self._registry.move_to_end(adapter_id)
        # Evict LRU entries until we're within the cap.
        while len(self._registry) > self.max_adapters:
            self._registry.popitem(last=False)  # drop least-recently-used
            self.num_evicted += 1
        return adapter

    def get_adapter(self, adapter_id: str) -> LoRAAdapter:
        """Fetch an adapter for serving, marking it most-recently-used."""
        adapter = self._registry[adapter_id]      # KeyError if unknown -- explicit
        self._registry.move_to_end(adapter_id)
        return adapter

    def __contains__(self, adapter_id: str) -> bool:
        return adapter_id in self._registry

    def adapter_ids(self) -> list[str]:
        """Currently-resident adapter ids, LRU order (oldest first)."""
        return list(self._registry.keys())

    def __len__(self) -> int:
        return len(self._registry)


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with optional, per-row LoRA deltas.

    The wrapped `base` Linear is frozen and shared; this module adds nothing to
    the base forward unless an adapter is active. Three ways an adapter becomes
    active for a forward call:

      * Pass `adapter_id="x"` to forward() directly (used in unit tests).
      * Set a single active adapter via set_active("x") (model.set_adapter).
      * Set per-row routing via set_active([...]) for a mixed-adapter batch
        (model.set_batch_adapters), one entry per batch row (None = base only).

    When nothing is active the forward is EXACTLY `base(x)` -- one GEMM, no
    branches taken, no tensors allocated. That is the zero-overhead guarantee
    the base model relies on.
    """

    def __init__(self, base: nn.Linear, manager: LoRAManager, layer_name: str) -> None:
        super().__init__()
        self.base = base
        self.manager = manager
        self.layer_name = layer_name
        # Active routing spec; None => pure base path.
        self._active: ActiveSpec = None

    # ---- activation control (called by LoRALlamaModel) ----------------------

    def set_active(self, active: ActiveSpec) -> None:
        self._active = active

    # ---- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, adapter_id: str | None = None) -> torch.Tensor:
        base_out = self.base(x)
        # Explicit arg wins; otherwise use the externally-set routing state.
        active: ActiveSpec = adapter_id if adapter_id is not None else self._active
        if active is None:
            # Zero-overhead path: identical to the wrapped nn.Linear.
            return base_out
        if isinstance(active, str):
            return self._apply_single(x, base_out, active)
        return self._apply_per_row(x, base_out, active)

    # ---- delta computation --------------------------------------------------

    def _delta(self, x: torch.Tensor, adapter_id: str) -> torch.Tensor | None:
        """LoRA delta for `x` under one adapter, or None if this adapter has no
        weights for this layer. delta = scale * (x @ A^T) @ B^T."""
        adapter = self.manager.get_adapter(adapter_id)
        A = adapter.weight_A.get(self.layer_name)
        B = adapter.weight_B.get(self.layer_name)
        if A is None or B is None:
            return None
        A = A.to(device=x.device, dtype=x.dtype)
        B = B.to(device=x.device, dtype=x.dtype)
        # x (..., in) @ A^T (in, r) -> (..., r) @ B^T (r, out) -> (..., out)
        return adapter.scale * (x @ A.transpose(0, 1)) @ B.transpose(0, 1)

    def _apply_single(self, x: torch.Tensor, base_out: torch.Tensor, adapter_id: str) -> torch.Tensor:
        delta = self._delta(x, adapter_id)
        return base_out if delta is None else base_out + delta

    def _apply_per_row(
        self, x: torch.Tensor, base_out: torch.Tensor, active: list[str | None]
    ) -> torch.Tensor:
        """Mixed-adapter batch: row i uses adapter active[i] (None = base only).

        Rows sharing an adapter are computed together (one grouped matmul per
        distinct adapter), then scattered back into the output by row index.
        """
        B = base_out.shape[0]
        if len(active) != B:
            raise ValueError(
                f"per-row adapter list length {len(active)} != batch size {B}"
            )
        out = base_out
        # Group row indices by adapter id (skip None -> base-only rows).
        groups: dict[str, list[int]] = {}
        for i, aid in enumerate(active):
            if aid is not None:
                groups.setdefault(aid, []).append(i)
        for aid, rows in groups.items():
            idx = torch.tensor(rows, dtype=torch.long, device=out.device)
            delta = self._delta(x.index_select(0, idx), aid)
            if delta is not None:
                # Out-of-place index_add so we never alias base_out's storage.
                out = out.index_add(0, idx, delta)
        return out
