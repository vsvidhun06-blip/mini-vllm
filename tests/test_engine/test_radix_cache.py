"""
Radix-tree prefix cache tests.

Pure data-structure tests over token ids and opaque KV-block ids -- no torch,
no GPU. They pin the SGLang-style behaviours the cache exists to provide:

  1. exact insert + match round-trips the blocks,
  2. two prompts that share a prefix get the SAME blocks for the shared part
     (stored once),
  3. inserting a divergent sequence SPLITS the shared edge,
  4. LRU eviction frees the least-recently-used leaves,
  5. ref_count (the match lock) protects an active path from eviction.
"""
from __future__ import annotations

import pytest

from src.engine.radix_cache import RadixCache, RadixNode


def _seq(*toks: int) -> tuple:
    return tuple(toks)


# ---------------------------------------------------------------------------
# 1. Exact insert + match.
# ---------------------------------------------------------------------------


def test_insert_then_exact_match():
    cache = RadixCache()
    tokens = _seq(10, 11, 12, 13)
    blocks = [100, 101, 102, 103]
    cache.insert(tokens, blocks)

    matched_len, matched_blocks = cache.match_prefix(tokens)
    assert matched_len == 4
    assert matched_blocks == blocks


def test_match_of_proper_prefix_returns_partial():
    cache = RadixCache()
    cache.insert(_seq(1, 2, 3, 4, 5), [10, 20, 30, 40, 50])

    matched_len, blocks = cache.match_prefix(_seq(1, 2, 3))
    assert matched_len == 3
    assert blocks == [10, 20, 30]


def test_miss_returns_zero():
    cache = RadixCache()
    cache.insert(_seq(1, 2, 3), [10, 20, 30])
    matched_len, blocks = cache.match_prefix(_seq(9, 9, 9))
    assert matched_len == 0
    assert blocks == []


# ---------------------------------------------------------------------------
# 2. Shared prefix is stored once.
# ---------------------------------------------------------------------------


def test_shared_prefix_returns_same_blocks_for_shared_part():
    """Two sequences sharing the first N tokens must return identical
    kv_block_ids for that shared part -- the prefix is stored exactly once and
    the first writer's blocks are canonical."""
    cache = RadixCache()
    shared = _seq(1, 2, 3, 4)           # the "system prompt"
    seq_a = shared + _seq(5, 6)         # ... User A
    seq_b = shared + _seq(7, 8)         # ... User B

    blocks_a = [10, 11, 12, 13, 14, 15]
    # Deliberately DIFFERENT block ids for B's shared part: the tree must keep
    # A's (the first writer's), proving the prefix isn't duplicated.
    blocks_b = [90, 91, 92, 93, 94, 95]

    cache.insert(seq_a, blocks_a)
    cache.insert(seq_b, blocks_b)

    _, a_blocks = cache.match_prefix(seq_a)
    _, b_blocks = cache.match_prefix(seq_b)

    # Shared first 4 tokens -> same blocks (A's), even though B was inserted
    # with different ones.
    assert a_blocks[:4] == [10, 11, 12, 13]
    assert b_blocks[:4] == [10, 11, 12, 13]
    assert a_blocks[:4] == b_blocks[:4]
    # Divergent tails stay distinct.
    assert a_blocks[4:] == [14, 15]
    assert b_blocks[4:] == [94, 95]


# ---------------------------------------------------------------------------
# 3. Edge splitting.
# ---------------------------------------------------------------------------


def test_edge_split_on_divergence():
    """insert "hello world" then "hello there": the edge splits at "hello"."""
    # Encode words as token ids: hello=1, world=2, there=3 (single-token words
    # for clarity; the tree works the same on long edges).
    cache = RadixCache()
    cache.insert(_seq(1, 2), [10, 20])     # "hello world"

    # Before the second insert: root has one child edge (1,2).
    assert set(cache.root.children.keys()) == {1}
    assert cache.root.children[1].token_ids == (1, 2)

    cache.insert(_seq(1, 3), [10, 30])     # "hello there"

    # After: root -> [hello] -> {world, there}. The (1,2) edge split at "hello".
    hello = cache.root.children[1]
    assert hello.token_ids == (1,)                 # shared prefix node
    assert set(hello.children.keys()) == {2, 3}    # branches on world/there
    assert hello.children[2].token_ids == (2,)
    assert hello.children[3].token_ids == (3,)

    # Both still match end-to-end with the right blocks.
    assert cache.match_prefix(_seq(1, 2)) == (2, [10, 20])
    assert cache.match_prefix(_seq(1, 3)) == (2, [10, 30])


def test_split_when_new_is_prefix_of_existing():
    """Inserting a sequence that is a strict prefix of an existing edge splits
    that edge at the prefix boundary."""
    cache = RadixCache()
    cache.insert(_seq(1, 2, 3, 4), [10, 20, 30, 40])
    cache.insert(_seq(1, 2), [10, 20])              # prefix of the above

    node12 = cache.root.children[1]
    assert node12.token_ids == (1, 2)
    # Its single child carries the remaining (3, 4).
    assert set(node12.children.keys()) == {3}
    assert node12.children[3].token_ids == (3, 4)
    assert cache.match_prefix(_seq(1, 2)) == (2, [10, 20])
    assert cache.match_prefix(_seq(1, 2, 3, 4)) == (4, [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# 4. LRU eviction.
# ---------------------------------------------------------------------------


def test_lru_eviction_frees_least_recently_used():
    """Insert 10 distinct one-block sequences, evict 3, and verify exactly the
    3 least-recently-inserted are gone and the other 7 survive."""
    cache = RadixCache()
    # Distinct first tokens -> 10 separate leaves off the root, each one block.
    # Insertion order sets last_access (logical clock), so 0 is the LRU.
    for i in range(10):
        cache.insert(_seq(i), [1000 + i])

    freed = cache.evict_lru(3)

    # 3 leaves, 1 block each -> 3 blocks freed, and they are the 3 oldest.
    assert sorted(freed) == [1000, 1001, 1002]
    for i in range(3):
        assert cache.match_prefix(_seq(i)) == (0, [])     # evicted -> miss
    for i in range(3, 10):
        assert cache.match_prefix(_seq(i)) == (1, [1000 + i])  # survived


def test_lru_respects_recency_after_match():
    """A match bumps recency, so a re-touched old entry is no longer the LRU."""
    cache = RadixCache()
    for i in range(3):
        cache.insert(_seq(i), [i])
    # Touch entry 0 so it is now the most-recently-used. But match_prefix LOCKS
    # it (ref_count > 0), which would also protect it -- so release immediately
    # to test recency, not the lock.
    cache.match_prefix(_seq(0))
    cache.release(_seq(0))

    freed = cache.evict_lru(1)
    # Entry 1 is now the oldest untouched leaf.
    assert freed == [1]
    assert cache.match_prefix(_seq(1)) == (0, [])
    assert cache.match_prefix(_seq(0))[0] == 1   # still present
    cache.release(_seq(0))


# ---------------------------------------------------------------------------
# 5. ref_count protects active nodes.
# ---------------------------------------------------------------------------


def test_ref_count_prevents_eviction_of_active_node():
    cache = RadixCache()
    for i in range(3):
        cache.insert(_seq(i), [i])

    # Lock the OLDEST entry (entry 0) via match_prefix. It is both LRU and now
    # locked; the lock must win.
    matched_len, _ = cache.match_prefix(_seq(0))
    assert matched_len == 1
    assert cache.root.children[0].ref_count == 1

    freed = cache.evict_lru(1)
    # Entry 0 is protected, so the evictor skips to the next-oldest unlocked
    # leaf (entry 1).
    assert freed == [1]
    assert cache.match_prefix(_seq(0))[0] == 1   # still present (re-locks -> ref 2)
    cache.release(_seq(0))
    cache.release(_seq(0))                        # balance both matches -> ref 0

    # Now unlocked: it can be evicted. (match bumped entry 0's recency above 2,
    # so the evictor reaches entry 0 only after entry 2 -- ask for both blocks.)
    assert cache.root.children[0].ref_count == 0
    freed2 = cache.evict_lru(2)
    assert 0 in freed2


def test_locking_protects_whole_prefix_chain():
    """Locking a deep node also protects its ancestors (you can't evict a
    prefix whose descendant is in use)."""
    cache = RadixCache()
    cache.insert(_seq(1, 2, 3), [10, 20, 30])     # one path, becomes 1 leaf
    # Lock the full path; the leaf and its ancestors get ref_count > 0.
    cache.match_prefix(_seq(1, 2, 3))

    freed = cache.evict_lru(5)
    assert freed == []                            # nothing evictable
    cache.release(_seq(1, 2, 3))
    freed2 = cache.evict_lru(5)
    assert sorted(freed2) == [10, 20, 30]


def test_split_preserves_node_identity_and_lock():
    """A split keeps the prefix node's object identity, so a lock taken before
    the split still protects it (RadixPagedKVCache stashes the node and unlocks
    it on completion -- this is the invariant that makes that safe)."""
    cache = RadixCache()
    node = cache.insert(_seq(1, 2, 3, 4), [10, 20, 30, 40])
    assert node.token_ids == (1, 2, 3, 4)
    cache._inc_ref_path(node)                      # lock the full path
    assert node.ref_count == 1

    # Insert a divergent sibling -> the (1,2,3,4) edge splits at (1,2). `node`
    # KEEPS its identity but is now the (1,2) PREFIX node.
    cache.insert(_seq(1, 2, 9), [10, 20, 90])
    assert node is cache.root.children[1]
    assert node.token_ids == (1, 2)                # shrunk to the prefix
    assert node.ref_count == 1                     # lock survived the split
    assert set(node.children.keys()) == {3, 9}     # old tail + new sibling
    assert node.children[3].token_ids == (3, 4)

    # The locked prefix and its locked (3,4) tail are protected; only the
    # unlocked new sibling (9 -> block 90) is evictable.
    freed = cache.evict_lru(10)
    assert freed == [90]
    assert cache.match_prefix(_seq(1, 2, 3, 4))[0] == 4   # survived
