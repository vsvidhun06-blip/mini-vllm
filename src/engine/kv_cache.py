"""
Paged KV cache -- the vLLM PagedAttention data structure.

The problem with the Day 5 SimpleKVCache:

  Each request held its own contiguous K/V tensor that grew by torch.cat
  every decode step. Two failure modes for a real serving system:

    1. Memory fragmentation. A naive allocator gives each request a
       max_seq_len-sized buffer up front. Most requests are far shorter,
       so 60-90% of the cache memory sits unused. With dozens of
       concurrent requests you OOM despite plenty of headroom.

    2. Reallocation churn. torch.cat allocates a fresh tensor and copies
       the old contents every step. For long sequences this becomes the
       bottleneck on the decode hot path.

The fix (Kwon et al. 2023, "Efficient Memory Management for Large Language
Model Serving with PagedAttention"):

  Carve physical KV memory into fixed-size BLOCKS (16 tokens each here).
  Maintain a free list. When a request needs cache space, allocate ONE
  block from the free list. A request's blocks need not be physically
  contiguous; a per-request BLOCK TABLE maps logical sequence position
  to physical block index. This is OS virtual memory, applied to KV cache.

  Two-block table indirection:
      logical_pos -> block_table[logical_pos // block_size]
                          -> physical_block_index
                          -> K_pool[layer, physical_block_index, logical_pos % block_size]

Layout decision (load-bearing for performance):

    K_pool: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    V_pool: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)

  Why split K and V into two pools instead of packing under a (2,) dim:
    SDPA takes K and V as separate tensor args. Packing forces a strided
    slice on every read which materializes as a copy. Splitting is free.

  Why layer-major (num_layers is outermost):
    K_pool[layer_idx] is a contiguous slice -- no copy, just a view. The
    layer is the natural iteration unit in attention.

  Why block_size before num_kv_heads:
    After gathering a request's blocks the shape is
    (n_blocks, block_size, NKV, D). `.view(-1, NKV, D)[:seq_len]` flattens
    block+slot into one seq dim without copying. Reversing block_size and
    NKV would force a copy on flatten.

Memory footprint per block (TinyLlama-1.1B, fp32):
    22 layers * 16 tokens * 4 KV heads * 64 head_dim * 4 bytes
    = 360 KB per block per K_pool (and the same for V_pool).
    Per block total: ~720 KB. 256 blocks = ~180 MB pool, ~4096 cached tokens.

Two classes in this module:

  * PagedKVCache       -- the global pool. Owns the K/V tensors, the
                          free-block set, and per-request bookkeeping.
                          Multiple requests share one pool.
  * PagedRequestCache  -- per-request view. Same .append / .get /
                          .seq_len(layer_idx) interface as the old
                          SimpleKVCache, so attention code stays
                          type-agnostic. Under the hood it routes
                          through the pool.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.engine import events

if TYPE_CHECKING:
    from src.engine.events import EventBus


class PagedKVCache:
    """The global physical block pool.

    Carries the K/V tensors and tracks which physical block indices are
    free, which are allocated to which request, and how many remain
    reserved for in-flight requests' future decode steps.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        event_bus: "EventBus | None" = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Optional event sink. When None, all the emit() calls below
        # short-circuit. Engine tests run without a bus and see no behavior
        # change.
        self.event_bus = event_bus

        # The big pool tensors. See module header for layout rationale.
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.K_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.V_pool = torch.zeros(shape, dtype=dtype, device=device)

        # Free-block set. Initially every physical block is free.
        # Using a set (instead of a list/deque) for O(1) pop-arbitrary and
        # O(1) re-add on free. We don't care about FIFO order; reuse just
        # has to be O(1) and correct.
        self._free_blocks: set[int] = set(range(num_blocks))

        # Per-request bookkeeping.
        #   _blocks: maps request_id -> list of physical block indices
        #            (the block table for that request, in logical order)
        #   _reserved: maps request_id -> int (blocks that are accounted
        #              for in the budget but not yet physically allocated)
        self._blocks: dict[str, list[int]] = {}
        self._reserved: dict[str, int] = {}

    # ---- Accounting / admission --------------------------------------

    def num_free_blocks(self) -> int:
        """Blocks available to a NEW admission.

        Free physical blocks minus the sum of outstanding reservations.
        Reservations represent "this request will need this many more
        blocks during its remaining decode steps" -- they're not yet
        allocated to a physical index, but they ARE off the table for
        future admissions.
        """
        return len(self._free_blocks) - sum(self._reserved.values())

    def can_admit(self, total_blocks_needed: int) -> bool:
        """Can the pool accommodate a request that needs this many blocks total?"""
        return self.num_free_blocks() >= total_blocks_needed

    def admit_request(
        self,
        request_id: str,
        prefill_blocks_needed: int,
        total_blocks_needed: int,
    ) -> None:
        """Reserve capacity for a new request and allocate its prefill blocks.

        Caller must have already verified can_admit. We allocate the prefill
        portion immediately (we'll write to those blocks during prefill);
        the remainder is RESERVED but not yet bound to physical indices.
        JIT allocation happens during decode via allocate_block().
        """
        if request_id in self._blocks:
            raise RuntimeError(f"Request {request_id!r} already admitted.")
        if not self.can_admit(total_blocks_needed):
            raise RuntimeError(
                f"Cannot admit request {request_id!r}: needs "
                f"{total_blocks_needed} blocks, {self.num_free_blocks()} free."
            )

        # Physically allocate the prefill blocks. Emit one block_allocated
        # event per block so the visualiser can draw the prefill footprint
        # block-by-block, matching how decode-time allocations arrive.
        allocated: list[int] = []
        for logical_idx in range(prefill_blocks_needed):
            phys = self._free_blocks.pop()
            allocated.append(phys)
            if self.event_bus is not None:
                self.event_bus.emit(events.block_allocated(
                    request_id=request_id,
                    physical_block_idx=phys,
                    logical_idx=logical_idx,
                ))
        self._blocks[request_id] = allocated
        # Reserve the remainder.
        self._reserved[request_id] = total_blocks_needed - prefill_blocks_needed

    def allocate_block(self, request_id: str) -> int:
        """Physically allocate one more block to an already-admitted request.

        Called by the per-request cache when a decode step fills the last
        block and needs a fresh one. This consumes from the request's
        reservation so global accounting stays consistent.
        """
        if request_id not in self._blocks:
            raise RuntimeError(f"Request {request_id!r} not admitted.")
        if self._reserved[request_id] <= 0:
            raise RuntimeError(
                f"Request {request_id!r} exhausted its block reservation. "
                f"The scheduler's admission control under-counted."
            )
        b = self._free_blocks.pop()
        self._blocks[request_id].append(b)
        self._reserved[request_id] -= 1
        if self.event_bus is not None:
            self.event_bus.emit(events.block_allocated(
                request_id=request_id,
                physical_block_idx=b,
                # logical_idx is len-1 because we just appended.
                logical_idx=len(self._blocks[request_id]) - 1,
            ))
        return b

    def free_request(self, request_id: str) -> None:
        """Return all of a request's blocks (allocated + reserved) to the pool."""
        for b in self._blocks.pop(request_id, []):
            self._free_blocks.add(b)
            if self.event_bus is not None:
                self.event_bus.emit(events.block_freed(
                    request_id=request_id,
                    physical_block_idx=b,
                ))
        # Clearing the reservation gives the unused budget back to other
        # admissions.
        self._reserved.pop(request_id, None)

    def get_block_table(self, request_id: str) -> list[int]:
        """Return the (currently allocated) block table for a request.

        Order matters: index `i` in this list is the physical block holding
        logical tokens [i*block_size, (i+1)*block_size).
        """
        return self._blocks[request_id]


class PagedRequestCache:
    """Per-request view over the pool.

    Exposes the same (.seq_len / .append / .get) trio as the Day-5
    SimpleKVCache so that attention code in attention.py stays type-
    agnostic and the dispatch for single-request vs batched-decode is
    unchanged. Under the hood, every operation routes through the pool.
    """

    def __init__(self, pool: PagedKVCache, request_id: str, num_layers: int) -> None:
        self.pool = pool
        self.request_id = request_id
        # Per-layer seq lens. Same lockstep pattern as Day 5: each layer
        # reads its own seq_len BEFORE appending, so that within a single
        # forward pass each layer sees the correct pre-step position
        # offset for RoPE. (Layer N's append would otherwise be visible
        # to layer N+1's seq_len read, which is the Day 5 bug.)
        self._seq_lens: list[int] = [0] * num_layers

    def seq_len(self, layer_idx: int = 0) -> int:
        return self._seq_lens[layer_idx]

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Write k_new, v_new into the pool at the right block/slot positions.

        Args:
            layer_idx: which layer slot in the pool we're writing.
            k_new, v_new: each shape (1, S_new, num_kv_heads, head_dim).
                S_new == prompt length in prefill, S_new == 1 in decode.
                K must already be POST-RoPE (caller's responsibility).
        """
        S_new = k_new.shape[1]
        cur = self._seq_lens[layer_idx]
        bs = self.pool.block_size

        # JIT-allocate enough physical blocks for the writes we're about to
        # do. The pool returns physical indices from the free list; the
        # request's reservation guarantees we have enough budget for these.
        new_total = cur + S_new
        blocks_needed = (new_total + bs - 1) // bs
        block_table = self.pool.get_block_table(self.request_id)
        while len(block_table) < blocks_needed:
            self.pool.allocate_block(self.request_id)
            block_table = self.pool.get_block_table(self.request_id)

        # Write block by block. A single block holds `bs` tokens; we may
        # write into the tail of the current block then start a new block.
        # For prefill of length L this loops ceil(L/bs) times (~7 for a
        # 100-token prompt, block_size=16). For decode S_new=1, one pass.
        written = 0
        while written < S_new:
            pos = cur + written
            block_idx_in_table = pos // bs
            slot_start = pos % bs
            space_in_block = bs - slot_start
            n_to_write = min(space_in_block, S_new - written)
            phys = block_table[block_idx_in_table]

            # Slice writes -- no copies, just writing into a contiguous
            # region of the pool tensor.
            self.pool.K_pool[layer_idx, phys, slot_start:slot_start + n_to_write] = \
                k_new[0, written:written + n_to_write]
            self.pool.V_pool[layer_idx, phys, slot_start:slot_start + n_to_write] = \
                v_new[0, written:written + n_to_write]

            written += n_to_write

        self._seq_lens[layer_idx] += S_new

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this request's K, V for one layer.

        Returns: (K, V), each (1, seq_len, num_kv_heads, head_dim).

        Implementation: advanced-index the pool with the block table to
        gather all blocks at once, flatten block+slot into one seq dim,
        and trim trailing padding from the (possibly partial) last block.
        """
        S = self._seq_lens[layer_idx]
        block_table = self.pool.get_block_table(self.request_id)
        bt = torch.tensor(block_table, dtype=torch.long, device=self.pool.K_pool.device)

        # blocks_k: (n_blocks, block_size, NKV, D)
        blocks_k = self.pool.K_pool[layer_idx, bt]
        blocks_v = self.pool.V_pool[layer_idx, bt]

        # Flatten block+slot -> seq, trim to actual seq_len (last block
        # may be partially filled).
        NKV = self.pool.num_kv_heads
        D = self.pool.head_dim
        k_flat = blocks_k.reshape(-1, NKV, D)[:S]   # (S, NKV, D)
        v_flat = blocks_v.reshape(-1, NKV, D)[:S]

        # Add batch dim. Same shape contract as SimpleKVCache.get returned.
        return k_flat.unsqueeze(0), v_flat.unsqueeze(0)
