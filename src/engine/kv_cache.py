"""
Per-request KV cache.

The point of a KV cache: during autoregressive decode, the only NEW work
at each step is computing K and V for the one new token. K and V for all
previous tokens were computed during prefill (or earlier decode steps)
and never change -- so we should keep them around instead of recomputing.

Without a cache: step t recomputes Q, K, V for positions 0..t. Total work
across N decode steps is O(N^2).
With a cache:    step t computes only Q, K, V for position t. Total work
                 across N decode steps is O(N).

Memory cost: 2 * num_layers * num_kv_heads * head_dim * dtype_bytes per
cached token. For TinyLlama (22 layers, 4 KV heads, 64 head_dim, fp32 = 4B)
that's 2 * 22 * 4 * 64 * 4 = ~45 KB per token. A 2048-context burns ~92 MB.

This file owns the storage container only. The attention block in
attention.py owns the "what gets stored, when, and with what rotation"
logic -- that's where the RoPE position math lives.

Layout choice (per layer):
    K, V each have shape (B, seq_so_far, num_kv_heads, head_dim).

Note: SDPA wants the head dim ahead of seq, i.e. (B, num_kv_heads, seq, head_dim).
We pay one transpose at attention time to convert. The chosen storage layout
makes the "append along seq" step a `torch.cat(..., dim=1)` which is cheap
and obvious. The transpose is a metadata op on a contiguous tensor most of
the time and not a real cost. Either layout works -- we picked this one
because the user's spec asked for it and the seq dim is the natural growth
axis.
"""
from __future__ import annotations

import torch


class SimpleKVCache:
    """A single-request KV cache. One instance per generation call.

    Lazy allocation: layers are populated on first append. Before any append
    a layer's slot is None. We could pre-allocate `(B, max_seq, NKV, D)` and
    track a write index instead, which avoids the per-step concat -- that's
    a perf optimization for Day 6+. For Day 5 we keep it correct and obvious.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        # One slot per layer. None means "this layer has never been written".
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append k_new, v_new to the running cache for `layer_idx`.

        Args:
            layer_idx: 0-based layer index.
            k_new, v_new: each shape (B, S_new, num_kv_heads, head_dim).
                In prefill S_new == prompt length. In decode S_new == 1.
                These should already be POST-RoPE for K (RoPE doesn't apply
                to V).

        After this call, the cache for this layer is grown by S_new along
        the seq dim.
        """
        if self._k[layer_idx] is None:
            # First write: just store. Saves a no-op cat.
            self._k[layer_idx] = k_new
            self._v[layer_idx] = v_new
        else:
            # Grow along seq dim.
            self._k[layer_idx] = torch.cat([self._k[layer_idx], k_new], dim=1)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], v_new], dim=1)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (K, V) for `layer_idx`. Caller must have appended first.

        Shapes: each (B, seq_so_far, num_kv_heads, head_dim).
        """
        k = self._k[layer_idx]
        v = self._v[layer_idx]
        if k is None or v is None:
            raise RuntimeError(
                f"KV cache for layer {layer_idx} is empty -- "
                f"call append() before get()."
            )
        return k, v

    def seq_len(self, layer_idx: int = 0) -> int:
        """Number of tokens currently cached for the given layer.

        Why per-layer instead of a single global count: within a single
        forward pass, the layers append to the cache one at a time, in
        order. If attention asks for `seq_len()` to compute its RoPE
        position offset, it MUST get the count BEFORE this forward pass
        started -- not the count after some earlier layer in the same
        forward already appended.

        The simplest way to guarantee that: each layer reads its OWN
        slot's length, which it hasn't written to yet. All layers had
        the same length entering this forward pass (lockstep invariant),
        so they all derive the same absolute position for the input tokens.

        External callers (`generate()`, tests) usually want the global
        seq length and can call `seq_len()` with the default layer_idx=0.
        """
        if self._k[layer_idx] is None:
            return 0
        return int(self._k[layer_idx].shape[1])
