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
from src.engine.radix_cache import RadixCache

if TYPE_CHECKING:
    from src.engine.events import EventBus
    from src.engine.radix_cache import RadixNode


class PagedKVCache:
    """The global physical block pool.

    Carries the K/V tensors and tracks which physical block indices are
    free, which are allocated to which request, and how many remain
    reserved for in-flight requests' future decode steps.

    Day 12 addition: prefix caching via reference counting.

      Multiple requests can point at the SAME physical block when their
      prompts share an aligned prefix. Sharing is content-addressable:
      a block's hash is a chain of (prev_block_hash, tokens, start_pos)
      so two blocks collide only if they sit at the same logical index
      with the same full history. The shared block is K/V-immutable
      after prefill -- decode appends only ever touch each request's
      own freshly-allocated tail block. Refcount is incremented when
      a hit binds an existing block to a new request, and decremented
      on free; when it hits zero the block goes back to the free pool
      AND its hash entries are evicted (no LRU retention in v0.2).
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
        enable_prefix_cache: bool = True,
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
        # Master switch for prefix caching. When False, admit_request
        # ignores any hashes the caller passes and allocates every
        # prefill block fresh -- restoring the pre-Day-12 behavior
        # byte-for-byte. The parity tests construct two schedulers
        # (one on, one off) and compare outputs across runs.
        self.enable_prefix_cache = enable_prefix_cache

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

        # Prefix-cache bookkeeping. Keys/values are physical block indices
        # and content hashes; refcount covers ALL allocated blocks (not
        # just shared ones), so the free path is uniform.
        #   block_hashes: physical_idx -> content hash (only present for
        #                 blocks that were registered as shareable)
        #   hash_to_block: content hash -> physical_idx (the inverse;
        #                  authoritative source for "is this hash cached")
        #   ref_count:    physical_idx -> int. Always >= 1 while the
        #                 block is allocated; hitting 0 triggers eviction.
        self.block_hashes: dict[int, int] = {}
        self.hash_to_block: dict[int, int] = {}
        self.ref_count: dict[int, int] = {}

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
        prefill_block_hashes: list[int | None] | None = None,
    ) -> int:
        """Reserve capacity for a new request and allocate its prefill blocks.

        Caller must have already verified can_admit. We allocate the prefill
        portion immediately (we'll write to those blocks during prefill);
        the remainder is RESERVED but not yet bound to physical indices.
        JIT allocation happens during decode via allocate_block().

        prefill_block_hashes (when not None) drives prefix-cache sharing:
            * must have length == prefill_blocks_needed
            * entry None: allocate FRESH from the free list. Used for the
              partial-tail block (always per-request) and for the last
              full block when prompt_len is an exact multiple of
              block_size (so prefill forward has somewhere to write).
            * entry int h: try hash_to_block[h]. If hit, the existing
              physical block is shared (refcount incremented, no free-
              list draw). If miss, allocate fresh AND register the hash
              for future requests to find.

        When prefix caching is disabled at the pool level
        (enable_prefix_cache=False), the hashes argument is ignored and
        every block is allocated fresh -- pre-Day-12 byte parity.

        Returns the number of blocks that were satisfied as cache hits
        (0 when caching is off or no hashes were passed). The scheduler
        uses this both to decide hit_boundary for prefill slicing and
        to emit the request_admitted event's hit-rate fields.
        """
        if request_id in self._blocks:
            raise RuntimeError(f"Request {request_id!r} already admitted.")
        if not self.can_admit(total_blocks_needed):
            raise RuntimeError(
                f"Cannot admit request {request_id!r}: needs "
                f"{total_blocks_needed} blocks, {self.num_free_blocks()} free."
            )

        # Normalise hash input. When caching is disabled or the caller
        # didn't bother computing hashes, this collapses to "every entry
        # None" which is exactly the old all-fresh behaviour.
        if not self.enable_prefix_cache or prefill_block_hashes is None:
            hashes_iter: list[int | None] = [None] * prefill_blocks_needed
        else:
            if len(prefill_block_hashes) != prefill_blocks_needed:
                raise ValueError(
                    f"prefill_block_hashes has length {len(prefill_block_hashes)}; "
                    f"expected {prefill_blocks_needed}"
                )
            hashes_iter = list(prefill_block_hashes)

        allocated: list[int] = []
        hits = 0
        for logical_idx, h in enumerate(hashes_iter):
            shared = False
            if h is not None and h in self.hash_to_block:
                # Cache hit. Bind to the existing physical block by
                # bumping its refcount. NO draw from the free list, so
                # other admissions get the saved capacity for free.
                phys = self.hash_to_block[h]
                self.ref_count[phys] += 1
                shared = True
                hits += 1
            else:
                # Miss. Pull a fresh physical block from the free pool.
                # If the caller provided a hash for this slot we record
                # it so the NEXT request with an identical chunk hits.
                phys = self._free_blocks.pop()
                self.ref_count[phys] = 1
                if h is not None:
                    self.block_hashes[phys] = h
                    self.hash_to_block[h] = phys
            allocated.append(phys)
            if self.event_bus is not None:
                self.event_bus.emit(events.block_allocated(
                    request_id=request_id,
                    physical_block_idx=phys,
                    logical_idx=logical_idx,
                    shared=shared,
                ))
        self._blocks[request_id] = allocated
        # Reserve the remainder. The reservation count is unchanged by
        # sharing: admission control is conservative (assumes all-miss)
        # so future decode JIT allocations still have budget.
        self._reserved[request_id] = total_blocks_needed - prefill_blocks_needed
        return hits

    def allocate_block(self, request_id: str) -> int:
        """Physically allocate one more block to an already-admitted request.

        Called by the per-request cache when a decode step fills the last
        block and needs a fresh one. This consumes from the request's
        reservation so global accounting stays consistent.

        Decode-time growth ALWAYS allocates a fresh block (refcount=1,
        no hash recorded). Decode appends mutate the tail block; sharing
        a mutable block would race. Only prefill-time blocks can be
        shared, and they are by construction immutable after admit.
        """
        if request_id not in self._blocks:
            raise RuntimeError(f"Request {request_id!r} not admitted.")
        if self._reserved[request_id] <= 0:
            raise RuntimeError(
                f"Request {request_id!r} exhausted its block reservation. "
                f"The scheduler's admission control under-counted."
            )
        b = self._free_blocks.pop()
        self.ref_count[b] = 1
        self._blocks[request_id].append(b)
        self._reserved[request_id] -= 1
        if self.event_bus is not None:
            self.event_bus.emit(events.block_allocated(
                request_id=request_id,
                physical_block_idx=b,
                # logical_idx is len-1 because we just appended.
                logical_idx=len(self._blocks[request_id]) - 1,
                shared=False,
            ))
        return b

    def free_request(self, request_id: str) -> None:
        """Decrement refcounts for the request's blocks; return any whose
        refcount drops to zero to the free pool.

        Edge cases:
          * Double-free: _blocks.pop returns [] and the loop is a no-op.
            Safe.
          * Underflow: hard assert. Should never happen with correct
            accounting; if it does, the bug is in admit_request /
            allocate_block balancing, not here.
          * Eviction: when refcount drops to zero we also evict the
            block's hash entry from hash_to_block and block_hashes.
            This is the "no LRU retention" policy: a cached prefix
            stays alive only as long as some live request references
            it. Simpler than vLLM's real eviction; correct.
        """
        for b in self._blocks.pop(request_id, []):
            self.ref_count[b] -= 1
            if self.ref_count[b] < 0:
                raise RuntimeError(
                    f"refcount underflow on block {b} freeing {request_id!r}; "
                    f"admit/allocate accounting is broken"
                )
            if self.ref_count[b] == 0:
                del self.ref_count[b]
                self._free_blocks.add(b)
                h = self.block_hashes.pop(b, None)
                if h is not None:
                    # Defensive: only drop the inverse mapping if it
                    # still points at THIS block. A future LRU policy
                    # might reassign a hash to a different physical
                    # block; today they always agree.
                    if self.hash_to_block.get(h) == b:
                        del self.hash_to_block[h]
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

    def trim_request_blocks(self, request_id: str, keep_blocks: int) -> int:
        """Free a request's blocks beyond the first `keep_blocks`, returning
        them to the free pool. Used by KV-cache eviction (H2O) after it has
        compacted the surviving tokens into the leading blocks.

        Unlike `free_request` (which releases everything when a request
        finishes), this is a PARTIAL free: the request stays alive with a
        shorter block table. The freed blocks are added back to this request's
        reservation so it can grow again later (eviction makes room, the
        request keeps generating, eventually re-allocates -- the whole point of
        serving sequences longer than the block budget).

        Returns the number of physical blocks freed.
        """
        bt = self._blocks.get(request_id)
        if bt is None:
            raise RuntimeError(f"Request {request_id!r} not admitted.")
        if keep_blocks >= len(bt):
            return 0
        freed = bt[keep_blocks:]
        self._blocks[request_id] = bt[:keep_blocks]
        for b in freed:
            self.ref_count[b] -= 1
            if self.ref_count[b] < 0:
                raise RuntimeError(
                    f"refcount underflow trimming block {b} of {request_id!r}"
                )
            if self.ref_count[b] == 0:
                del self.ref_count[b]
                self._free_blocks.add(b)
                # Evicted tail blocks are decode-time blocks (refcount 1, no
                # shared hash), but drop any hash entry defensively to match
                # free_request's bookkeeping.
                h = self.block_hashes.pop(b, None)
                if h is not None and self.hash_to_block.get(h) == b:
                    del self.hash_to_block[h]
                if self.event_bus is not None:
                    self.event_bus.emit(events.block_freed(
                        request_id=request_id,
                        physical_block_idx=b,
                    ))
        # Hand the capacity back as reservation so future appends can re-draw
        # these slots (admission already counted them against this request).
        self._reserved[request_id] = self._reserved.get(request_id, 0) + len(freed)
        return len(freed)

    def num_shared_blocks(self) -> int:
        """Count physical blocks currently shared by 2+ requests.

        A block with ref_count >= 2 is one the prefix cache is actively
        deduplicating -- a single physical block standing in for the
        same K/V across multiple requests. This is the "prefix caching
        is doing work" signal the metrics layer reports as
        POOL_BLOCKS_CACHED. Blocks with ref_count == 1 are uniquely
        owned; blocks with no ref_count entry are free.
        """
        return sum(1 for c in self.ref_count.values() if c >= 2)


class RadixPagedKVCache(PagedKVCache):
    """A PagedKVCache whose prefix cache is a radix tree (SGLang-style).

    Drop-in for PagedKVCache: same constructor and the same
    can_admit / admit_request / allocate_block / free_request / get_block_table
    surface, so PagedRequestCache and the attention code work over it unchanged.
    The only difference is the prefix-sharing mechanism.

    What changes vs the flat-hash base class:

      * Sharing unit. The base class shares fixed 16-token blocks keyed by a
        chained block hash. Here a RadixCache stores the actual token paths, so
        prefixes are shared at TOKEN granularity and divergent prompts split at
        their first differing token automatically.

      * RETENTION. The base class evicts a cached block the instant its last
        live request frees it (no-LRU-retention). A completed request's prefix
        therefore can't be reused by a LATER request. This cache INSERTS a
        completed request's block-aligned prefix into the tree and KEEPS those
        blocks resident (out of the free list) until LRU eviction reclaims
        them. That is what lets a stream of requests sharing one system prompt
        reuse it across completions -- the whole point of the radix cache.

    Block lifecycle / accounting:

      pool.ref_count[phys] counts LIVE-REQUEST references only. A block held by
      the tree but no live request has NO ref_count entry and is NOT in the free
      list (it lives in self._tree_blocks). A block can be both: a live request
      that reused a cached prefix bumps ref_count while the block stays
      tree-owned. A block returns to the free list only when it is neither live
      (ref_count hits 0) nor tree-owned (evicted, or never inserted -- e.g. the
      partial tail and decode-grown blocks).

    Interface note: admit_request takes an extra optional `token_ids` (the
    prompt token ids). When provided (and prefix caching is on) it drives radix
    matching; when omitted it falls back to all-fresh allocation, so calling
    this exactly like the base class still works.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.radix = RadixCache()
        # Blocks currently referenced by the tree (evictable when not also
        # locked by a live match). Kept out of the free list while resident.
        self._tree_blocks: set[int] = set()
        # Per-request state needed at completion time.
        self._request_tokens: dict[str, list[int]] = {}
        self._request_match_node: dict[str, "RadixNode | None"] = {}

    # ---- admission accounting (now eviction-aware) ----------------------

    def _live_blocks(self) -> set[int]:
        """Every physical block currently held by some live request."""
        live: set[int] = set()
        for bt in self._blocks.values():
            live.update(bt)
        return live

    def num_evictable_blocks(self) -> int:
        """Tree-owned blocks not pinned by a live request -- reclaimable now."""
        return len(self._tree_blocks - self._live_blocks())

    def can_admit(self, total_blocks_needed: int) -> bool:
        """Like the base class, but evictable tree blocks count as available --
        admit_request will reclaim them on demand."""
        return self.num_free_blocks() + self.num_evictable_blocks() >= total_blocks_needed

    def _reclaim_from_tree(self, n_needed: int) -> None:
        """Evict LRU tree leaves until at least `n_needed` blocks are free."""
        deficit = n_needed - len(self._free_blocks)
        if deficit <= 0:
            return
        freed = self.radix.evict_lru(deficit)
        for phys in freed:
            self._tree_blocks.discard(phys)
            # Evicted tree blocks carry no live ref (locked nodes never evict),
            # so they go straight back to the free list.
            if self.ref_count.get(phys, 0) == 0:
                self.ref_count.pop(phys, None)
                self._free_blocks.add(phys)

    # ---- admit / free (radix-backed) ------------------------------------

    def admit_request(
        self,
        request_id: str,
        prefill_blocks_needed: int,
        total_blocks_needed: int,
        prefill_block_hashes=None,           # accepted for signature parity; unused
        token_ids=None,
    ) -> int:
        """Admit a request, reusing a matched radix prefix where possible.

        Returns the number of prefill blocks satisfied by the cache (hits).
        """
        if request_id in self._blocks:
            raise RuntimeError(f"Request {request_id!r} already admitted.")

        # Fall back to all-fresh when caching is off or we weren't given tokens.
        use_radix = self.enable_prefix_cache and token_ids is not None
        bs = self.block_size

        reused_phys: list[int] = []
        match_node = None
        if use_radix:
            token_ids = list(token_ids)
            matched_len, matched_blocks, match_node = self.radix.match_node(token_ids)
            reusable = matched_len // bs                       # whole blocks only
            reusable = min(reusable, prefill_blocks_needed)
            # If the prompt is an exact multiple of block_size and the WHOLE
            # prompt matched, leave one block un-reused so prefill has a token to
            # run on (mirrors the scheduler's forced-fresh last block). Harmless
            # in pure-simulation use; required when wired to a real forward.
            prompt_len = len(token_ids)
            if reusable == prefill_blocks_needed and prompt_len % bs == 0:
                reusable -= 1
            # matched_blocks is per-token; the physical block for whole block i
            # is the entry at that block's first token.
            reused_phys = [matched_blocks[i * bs] for i in range(reusable)]

        fresh_needed = prefill_blocks_needed - len(reused_phys)
        # Reclaim from the tree if the free list can't cover the fresh blocks
        # plus the reservation we're about to make.
        self._reclaim_from_tree(fresh_needed)
        if self.num_free_blocks() < total_blocks_needed - len(reused_phys):
            # Try harder: reclaim enough for the full reservation too.
            self._reclaim_from_tree(total_blocks_needed - len(reused_phys))

        allocated: list[int] = []
        hits = 0
        # 1) Reused (cache-hit) blocks: bump the live ref; they stay tree-owned.
        for logical_idx, phys in enumerate(reused_phys):
            self.ref_count[phys] = self.ref_count.get(phys, 0) + 1
            allocated.append(phys)
            hits += 1
            if self.event_bus is not None:
                self.event_bus.emit(events.block_allocated(
                    request_id=request_id, physical_block_idx=phys,
                    logical_idx=logical_idx, shared=True,
                ))
        # 2) Fresh blocks for the non-cached remainder of the prefill.
        for j in range(fresh_needed):
            phys = self._free_blocks.pop()
            self.ref_count[phys] = 1
            allocated.append(phys)
            if self.event_bus is not None:
                self.event_bus.emit(events.block_allocated(
                    request_id=request_id, physical_block_idx=phys,
                    logical_idx=len(reused_phys) + j, shared=False,
                ))

        self._blocks[request_id] = allocated
        self._reserved[request_id] = total_blocks_needed - prefill_blocks_needed
        self._request_tokens[request_id] = list(token_ids) if use_radix else []
        self._request_match_node[request_id] = match_node    # locked by match_node
        return hits

    def free_request(self, request_id: str) -> None:
        """Complete a request: INSERT its block-aligned prefix into the tree
        (so later requests can reuse it), then release its live block refs.

        Blocks the tree now owns stay resident; blocks nobody references
        (partial tail, decode-grown blocks, or evicted prefixes) return to the
        free pool. This replaces the base class's evict-on-last-free policy.
        """
        # Unlock the prefix this request matched at admit (split-safe: we stored
        # the node object, so an intervening edge split doesn't desync the lock).
        node = self._request_match_node.pop(request_id, None)
        if node is not None:
            self.radix.dec_ref_path(node)

        token_ids = self._request_tokens.pop(request_id, None)
        bt = self._blocks.pop(request_id, [])

        # Insert the block-aligned prompt prefix into the tree, mapping each
        # token to the physical block covering it. Only full blocks we actually
        # hold are insertable.
        if self.enable_prefix_cache and token_ids:
            n_full = min(len(token_ids) // self.block_size, len(bt))
            aligned_len = n_full * self.block_size
            if aligned_len > 0:
                per_token = [bt[t // self.block_size] for t in range(aligned_len)]
                self.radix.insert(tuple(token_ids[:aligned_len]), per_token)
                self._tree_blocks.update(per_token)

        # Release live refs. A block hitting ref 0 returns to free ONLY if the
        # tree doesn't own it.
        for b in bt:
            self.ref_count[b] -= 1
            if self.ref_count[b] < 0:
                raise RuntimeError(
                    f"refcount underflow on block {b} freeing {request_id!r}"
                )
            if self.ref_count[b] == 0:
                del self.ref_count[b]
                if b not in self._tree_blocks:
                    self._free_blocks.add(b)
                    if self.event_bus is not None:
                        self.event_bus.emit(events.block_freed(
                            request_id=request_id, physical_block_idx=b,
                        ))
        self._reserved.pop(request_id, None)

    def num_cached_prefix_blocks(self) -> int:
        """Distinct blocks the radix tree currently retains (metrics)."""
        return len(self._tree_blocks)


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

        # ---- Cached device tensors for the decode hot path (Day 17) --------
        # Two values used inside attention came from Python ints/lists and were
        # turned into CUDA tensors EVERY layer EVERY step:
        #   * the block table (here, in .get())
        #   * the per-row RoPE positions (attention._forward_decode_batched)
        # Each `torch.tensor(py_list, device="cuda")` is a host->device copy
        # from pageable memory. On the decode hot path that's pure overhead;
        # worse, it makes the forward un-capturable as a CUDA graph (pageable
        # H2D copies force a sync, which is illegal mid-capture). We cache both
        # as persistent device tensors, rebuilt/updated only when their value
        # actually changes -- which never happens inside a single captured
        # step. The cached values are identical to what we copied before, so
        # every existing parity test is unaffected.
        self._bt_tensor: torch.Tensor | None = None
        self._seqlen_tensors: dict[int, torch.Tensor] = {}
        self._seqlen_vals: dict[int, int] = {}

    def seq_len(self, layer_idx: int = 0) -> int:
        return self._seq_lens[layer_idx]

    def seq_len_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """`seq_len(layer_idx)` as a cached 0-dim long tensor on the pool device.

        Used by batched decode to build the per-row RoPE positions vector
        WITHOUT a per-call ``torch.tensor([...], device="cuda")`` host->device
        copy (see the __init__ note). The backing tensor is allocated once and
        updated in place with ``fill_`` only when the value changes, so its
        address is stable -- which is what CUDA-graph capture requires. The
        numeric value is identical to the old Python-int path.
        """
        val = self._seq_lens[layer_idx]
        t = self._seqlen_tensors.get(layer_idx)
        if t is None:
            t = torch.empty((), dtype=torch.long, device=self.pool.K_pool.device)
            self._seqlen_tensors[layer_idx] = t
            self._seqlen_vals[layer_idx] = -1  # force the first fill_ below
        if self._seqlen_vals[layer_idx] != val:
            t.fill_(val)
            self._seqlen_vals[layer_idx] = val
        return t

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
        # Cache the block-table-as-device-tensor. Rebuild only on a length
        # change (decode appends a fresh block when the tail fills); the table
        # only ever grows and existing entries never move, so a length check is
        # sufficient. Inside a captured CUDA-graph step we pre-allocate so the
        # length never changes -> no host->device copy here during replay.
        bt = self._bt_tensor
        if bt is None or bt.shape[0] != len(block_table):
            bt = torch.tensor(
                block_table, dtype=torch.long, device=self.pool.K_pool.device
            )
            self._bt_tensor = bt

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
