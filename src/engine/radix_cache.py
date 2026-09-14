"""
Radix-tree prefix cache (SGLang-style), replacing the flat hash map.

Why a radix tree beats a flat (token_ids tuple -> block) hash map:

  The hash map keys on a WHOLE sequence (or a whole block's token chunk).
  Two requests share cache only if their keys collide exactly. The chained
  block hash in PagedKVCache gets you block-aligned prefix sharing, but the
  unit of sharing is a fixed 16-token block and the structure is flat: there
  is no notion of "these two prompts diverge after token 137".

  A radix tree (a.k.a. compressed/PATRICIA trie) stores token sequences as
  PATHS. Each edge is labelled with a run of tokens; a node's children branch
  on the first divergent token. So:

      "System: ... User: What is"   -> path A
      "System: ... User: Where is"  -> path B

  share the entire "System: ... User: Wh" prefix as ONE path, and split at the
  first token that differs ('a' vs 'e'). The shared prefix's KV blocks are
  stored exactly once, and a new request that walks the same prefix reuses them
  for free -- automatic, content-addressed prefix sharing at TOKEN granularity,
  not just block granularity. This is the structure SGLang's "RadixAttention"
  uses for its prefix cache.

This module is deliberately torch-free: it is a pure data structure over token
ids and opaque KV-block ids, so it can be unit-tested on any machine without a
GPU or even PyTorch. The paging integration that maps tree blocks onto the
physical K/V pool lives in kv_cache.py (RadixPagedKVCache).

Token / block alignment contract
--------------------------------
A node carries `kv_block_ids` that is index-aligned 1:1 with its `token_ids`
edge label: kv_block_ids[i] is the KV storage id for the i-th token on the
edge. (RadixPagedKVCache fills this with the physical block id covering each
token, so consecutive tokens in the same 16-token block repeat a block id; the
tree itself does not care -- it just stores and slices the list alongside the
tokens.) insert() therefore requires len(token_ids) == len(kv_block_ids), and
match_prefix() returns the per-token slice for the matched prefix.

Reference counting (eviction safety)
------------------------------------
`match_prefix` LOCKS the matched path: it increments ref_count on the matched
endpoint node and all its ancestors (locking a node implicitly locks its
prefix). `evict_lru` only ever removes leaves whose ref_count is 0, so a node a
live request is using -- or whose descendant it is using -- is never evicted.
`release` (or dec_ref_path on the stored node) unlocks. This is the SGLang
lock/unlock discipline.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _common_prefix_len(a: tuple, b: tuple) -> int:
    """Length of the longest common prefix of two token tuples."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class RadixNode:
    """One node of the radix tree.

    token_ids:    the edge label -- the run of tokens on the edge leading INTO
                  this node from its parent. The root has the empty tuple.
    children:     keyed by the FIRST token of each child's edge, so a lookup is
                  O(1) on the branching token.
    kv_block_ids: KV storage ids, 1:1 with token_ids (None only for the root).
    ref_count:    lock count for eviction. > 0 means a live request is using
                  this node or one of its descendants; such nodes never evict.
    last_access:  logical timestamp of the most recent insert/match touching
                  this node, used to pick the LRU victim. We use a monotonically
                  increasing logical clock (not wall-clock) so eviction order is
                  deterministic and test-stable.
    parent:       back-pointer, needed to detach a leaf on eviction and to walk
                  up the ancestor chain when locking/unlocking. Internal -- not
                  part of the conceptual node state.
    """

    token_ids: tuple = ()
    children: dict = field(default_factory=dict)
    kv_block_ids: list | None = None
    ref_count: int = 0
    last_access: float = 0.0
    parent: "RadixNode | None" = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class RadixCache:
    """A radix tree over token sequences with per-node KV blocks + LRU eviction."""

    def __init__(self) -> None:
        # The root is an empty-edge sentinel: it owns no tokens and no blocks
        # and is never evicted. Every real sequence hangs beneath it.
        self.root = RadixNode(token_ids=(), kv_block_ids=[], ref_count=0)
        # Logical clock for LRU. Incremented on every access; a node's
        # last_access is the tick at its most recent touch.
        self._tick: float = 0.0

    # ---- internal helpers ---------------------------------------------------

    def _now(self) -> float:
        self._tick += 1.0
        return self._tick

    def _match(self, token_ids: tuple) -> tuple[int, list, "RadixNode"]:
        """Walk the tree matching `token_ids`. No locking; updates last_access.

        Returns (matched_len, matched_blocks, last_node) where last_node is the
        deepest node touched (the root if nothing matched). matched_blocks is
        the per-token KV-block slice for the matched prefix.
        """
        node = self.root
        matched_len = 0
        blocks: list = []
        n = len(token_ids)
        while matched_len < n:
            first = token_ids[matched_len]
            child = node.children.get(first)
            if child is None:
                break
            rem = token_ids[matched_len:]
            cp = _common_prefix_len(child.token_ids, rem)
            # cp >= 1 always: the child is keyed on `first`, so its edge and the
            # remaining query agree on at least the first token.
            matched_len += cp
            if child.kv_block_ids is not None:
                blocks.extend(child.kv_block_ids[:cp])
            child.last_access = self._now()
            node = child
            if cp < len(child.token_ids):
                # Partial edge match -- the query diverges in the middle of this
                # edge. We stop here; the partially-used node is the endpoint.
                break
        return matched_len, blocks, node

    def _inc_ref_path(self, node: "RadixNode") -> None:
        """Lock `node` and all its ancestors (excluding the root sentinel)."""
        while node is not None and node is not self.root:
            node.ref_count += 1
            node = node.parent

    def _dec_ref_path(self, node: "RadixNode") -> None:
        """Unlock `node` and ancestors. Clamps at 0 (never goes negative even if
        an intervening split reshaped the path between lock and unlock)."""
        while node is not None and node is not self.root:
            if node.ref_count > 0:
                node.ref_count -= 1
            node = node.parent

    # ---- edge splitting -----------------------------------------------------

    def _split_edge(self, node: "RadixNode", split_pos: int) -> "RadixNode":
        """Split `node`'s edge at `split_pos`, returning the (unchanged-identity)
        prefix node.

        Before:  parent --[T]--> node(children C, blocks B)
        After:   parent --[T[:split_pos]]--> node --[T[split_pos:]]--> suffix(C, B[split_pos:])

        `node` KEEPS its object identity as the PREFIX (so any lock held on it,
        and any external reference to it, stay valid). A fresh `suffix` node
        takes over the tail tokens, the old children, and the tail blocks. The
        lock count is copied onto the suffix too, because a request that had
        locked the deeper content still needs it protected.
        """
        assert 0 < split_pos < len(node.token_ids), "split_pos must be interior"

        suffix = RadixNode(
            token_ids=node.token_ids[split_pos:],
            children=node.children,
            kv_block_ids=(None if node.kv_block_ids is None
                          else node.kv_block_ids[split_pos:]),
            ref_count=node.ref_count,
            last_access=node.last_access,
            parent=node,
        )
        # Re-parent the moved children onto the suffix.
        for ch in suffix.children.values():
            ch.parent = suffix

        # Shrink `node` down to the prefix and point it at the new suffix.
        node.token_ids = node.token_ids[:split_pos]
        if node.kv_block_ids is not None:
            node.kv_block_ids = node.kv_block_ids[:split_pos]
        node.children = {suffix.token_ids[0]: suffix}
        return node

    # ---- public API ---------------------------------------------------------

    def match_prefix(self, token_ids) -> tuple[int, list]:
        """Return (matched_len, kv_block_ids) for the longest cached prefix.

        Side effect (by design, per the eviction contract): LOCKS the matched
        path -- the endpoint node and its ancestors get ref_count += 1 so they
        cannot be evicted while this caller holds the match. Pair every
        match_prefix with a later `release(token_ids)` (or dec_ref_path on the
        node) to unlock.
        """
        token_ids = tuple(token_ids)
        matched_len, blocks, node = self._match(token_ids)
        self._inc_ref_path(node)
        return matched_len, blocks

    def match_node(self, token_ids):
        """Like match_prefix but also returns the endpoint node and DOES lock it.

        RadixPagedKVCache uses this so it can stash the exact node object and
        unlock via dec_ref_path on completion, immune to any edge splits that
        happen on the path in between. Returns (matched_len, kv_block_ids, node).
        """
        token_ids = tuple(token_ids)
        matched_len, blocks, node = self._match(token_ids)
        self._inc_ref_path(node)
        return matched_len, blocks, node

    def release(self, token_ids) -> None:
        """Unlock the path previously locked by `match_prefix(token_ids)`."""
        _, _, node = self._match(tuple(token_ids))
        self._dec_ref_path(node)

    def dec_ref_path(self, node: "RadixNode") -> None:
        """Unlock a node (and ancestors) by direct reference. Split-safe."""
        self._dec_ref_path(node)

    def insert(self, token_ids, kv_block_ids) -> "RadixNode":
        """Insert `token_ids` with its 1:1 `kv_block_ids`, splitting edges as
        needed. Returns the endpoint node for the full sequence.

        Standard radix insert: descend matching edges; on a partial edge match
        split that edge so the divergence point becomes a node; append a fresh
        leaf for the remaining (novel) tokens. Re-inserting an existing prefix
        is a no-op for the shared part -- the FIRST writer's blocks win, so
        shared prefixes keep one canonical set of KV blocks.
        """
        token_ids = tuple(token_ids)
        kv_block_ids = list(kv_block_ids)
        if len(token_ids) != len(kv_block_ids):
            raise ValueError(
                f"token_ids ({len(token_ids)}) and kv_block_ids "
                f"({len(kv_block_ids)}) must be the same length"
            )

        node = self.root
        i = 0
        n = len(token_ids)
        while i < n:
            first = token_ids[i]
            child = node.children.get(first)
            if child is None:
                # No edge starts with this token -> hang the whole remainder as
                # a new leaf and we're done.
                leaf = RadixNode(
                    token_ids=token_ids[i:],
                    children={},
                    kv_block_ids=kv_block_ids[i:],
                    ref_count=0,
                    last_access=self._now(),
                    parent=node,
                )
                node.children[first] = leaf
                return leaf

            rem = token_ids[i:]
            cp = _common_prefix_len(child.token_ids, rem)
            if cp == len(child.token_ids):
                # Whole edge matches -> descend and keep going.
                i += cp
                child.last_access = self._now()
                node = child
                continue
            # Partial match: the new sequence diverges inside this edge. Split
            # the edge at cp so child becomes the shared prefix, then loop --
            # the next iteration sees the divergent token has no child and adds
            # a fresh leaf for the novel suffix.
            self._split_edge(child, cp)
            i += cp
            child.last_access = self._now()
            node = child
        # token_ids was fully consumed walking existing edges -- it is a prefix
        # of (or equal to) an existing path. `node` is its endpoint.
        return node

    def evict_lru(self, n_blocks: int) -> list:
        """Evict least-recently-used UNLOCKED leaves until >= n_blocks freed.

        Returns the list of KV-block ids that were freed. Only ref_count==0
        leaves are eligible (a locked node, or any ancestor of a locked node,
        is protected). After evicting a leaf its parent may become a leaf and
        thus a candidate on the next pass.
        """
        freed: list = []
        if n_blocks <= 0:
            return freed
        while len(freed) < n_blocks:
            victim = self._lru_evictable_leaf()
            if victim is None:
                break  # nothing left we're allowed to evict
            if victim.kv_block_ids:
                freed.extend(victim.kv_block_ids)
            parent = victim.parent
            if parent is not None and victim.token_ids:
                parent.children.pop(victim.token_ids[0], None)
        return freed

    def _lru_evictable_leaf(self) -> "RadixNode | None":
        """The unlocked leaf with the smallest last_access, or None."""
        best: "RadixNode | None" = None
        # Iterative DFS; the tree is shallow (one level per divergent prefix).
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf and node is not self.root:
                if node.ref_count == 0:
                    if best is None or node.last_access < best.last_access:
                        best = node
            else:
                stack.extend(node.children.values())
        return best

    # ---- introspection (metrics / debugging) --------------------------------

    def cached_block_ids(self) -> set:
        """Every distinct KV-block id currently held anywhere in the tree."""
        out: set = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.kv_block_ids:
                out.update(node.kv_block_ids)
            stack.extend(node.children.values())
        return out

    def num_cached_blocks(self) -> int:
        return len(self.cached_block_ids())
