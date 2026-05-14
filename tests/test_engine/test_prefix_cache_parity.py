"""
Day 12: Prefix caching parity tests.

What we're asserting:

  Prefix caching is the most parity-critical change in the project --
  the whole point is that requests get identical TOKENS whether or not
  their prompt blocks were satisfied as cache hits. If a hit ever
  changes a single output token, the design is broken. So:

    test_shared_prefix_correctness   -- 4 requests with an identical
                                        long prefix. cache=ON outputs
                                        must equal cache=OFF outputs,
                                        token-for-token, per request.
    test_diverging_prefix             -- 2 requests share an aligned
                                        prefix then diverge. The block
                                        holding the divergence must be
                                        allocated FRESH for both (i.e.
                                        their block tables differ at
                                        that logical index).
    test_no_shared_prefix             -- 2 disjoint prompts. 0% hit
                                        rate. Outputs identical to
                                        a cache=OFF run.
    test_partial_overlap              -- 48-token prompts that share
                                        the first 32 tokens (2 blocks)
                                        and diverge in block 2. With
                                        block_size=16 and prompt_len%bs
                                        == 0, max_shareable=2 (last
                                        full block forced fresh). So:
                                        blocks 0,1 shared (refcount 2),
                                        block 2 distinct per request.
    test_refcount_decrement           -- two requests share a block;
                                        on finish the refcount goes
                                        from 2 down to 0; the block
                                        returns to the free pool and
                                        the hash entry is evicted.

Synthetic prompts:

  The parity tests don't care about token semantics, only that the
  same prompt gives the same output deterministically. We build the
  prompt tensors directly from small integer IDs, which lets us pin
  prompt length to the exact block boundary the test wants. The
  resulting "text" is gibberish; greedy argmax is still deterministic,
  which is all parity assertions need.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.device import DEVICE
from src.engine.scheduler import ContinuousBatchScheduler


BLOCK_SIZE = 16


def _drain(scheduler: ContinuousBatchScheduler, n_requests: int) -> dict[str, list[int]]:
    """Run the scheduler to completion, collect per-request token lists."""
    outs: dict[str, list[int]] = {f"req-{i}": [] for i in range(n_requests)}
    while scheduler.has_work():
        for rid, tok in scheduler.step():
            outs[rid].append(tok)
    return outs


def _ids(tokens: list[int]) -> torch.Tensor:
    """Wrap a list of token ids as a (1, S) int64 tensor on the model device.
    The scheduler also moves prompt_ids to device internally, but doing it
    here keeps the test setup explicit.
    """
    return torch.tensor([tokens], dtype=torch.long, device=DEVICE)


def _run_scheduler(
    my_model,
    prompts: list[list[int]],
    *,
    enable_prefix_cache: bool,
    max_new_tokens: int = 6,
    num_blocks: int = 64,
) -> tuple[dict[str, list[int]], ContinuousBatchScheduler]:
    """Submit prompts to a fresh scheduler, drain, return (outputs, scheduler).

    The scheduler is returned so block-table / refcount assertions can
    inspect its pool after the run.
    """
    scheduler = ContinuousBatchScheduler(
        my_model,
        max_batch_size=max(8, len(prompts)),
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        enable_prefix_cache=enable_prefix_cache,
    )
    for i, p in enumerate(prompts):
        scheduler.add_request(
            request_id=f"req-{i}",
            prompt_ids=_ids(p),
            max_new_tokens=max_new_tokens,
            eos_token_id=None,  # always run to max_new_tokens for determinism
        )
    outs = _drain(scheduler, len(prompts))
    return outs, scheduler


def test_shared_prefix_correctness(my_model) -> None:
    """4 requests share a 50-token prefix; cache=ON tokens == cache=OFF tokens.

    Why this is the most important test:
      Prefix caching SKIPS computing K/V for cached blocks during
      prefill. If the cached K/V isn't bit-equivalent to "compute it
      fresh", the model's attention input differs and the argmax can
      flip. This test catches that immediately by comparing the two
      regimes token-for-token.

    Prompt layout:
      shared[0..49] + unique[0..9] = 60 tokens
      With block_size=16: 4 prefill blocks (3 full + 1 partial of 14).
      max_shareable for caching = 3 (the 3 full blocks).
      Request 0 misses all 3 (first to compute the prefix); requests
      1-3 each hit all 3. Per-request hit rate goes 0%, 100%, 100%,
      100% on the shareable portion.
    """
    # Make the shared prefix and divergent suffixes deterministic.
    # Small ints keep us well inside any vocab; values are arbitrary.
    shared_prefix = list(range(100, 150))  # 50 tokens, ids 100..149
    suffixes = [
        list(range(200, 210)),
        list(range(300, 310)),
        list(range(400, 410)),
        list(range(500, 510)),
    ]
    prompts = [shared_prefix + s for s in suffixes]  # 4 prompts of length 60

    off_outs, _ = _run_scheduler(my_model, prompts, enable_prefix_cache=False)
    on_outs, sched_on = _run_scheduler(my_model, prompts, enable_prefix_cache=True)

    for i in range(4):
        rid = f"req-{i}"
        assert off_outs[rid] == on_outs[rid], (
            f"prefix-cache changed output for {rid}:\n"
            f"  off: {off_outs[rid]}\n"
            f"  on:  {on_outs[rid]}\n"
            "Prefix caching MUST be a pure speedup -- any token "
            "difference here is a correctness regression."
        )

    # Sanity: the first 3 logical blocks of every request's block
    # table should be the SAME physical indices (they shared during
    # admit). The 4th block (partial tail) should differ.
    # ...but the requests have already finished and their block tables
    # have been freed by the eviction phase. Instead, check that with
    # the cache off we'd see distinct physical indices everywhere -- which
    # we don't have a handle on either (pool state is post-eviction).
    # The output-equality assertion above is the load-bearing one.


def test_diverging_prefix(my_model) -> None:
    """2 requests with a 50-token shared prefix that diverges at token 51.

    Specifically:
      prompt_A = shared[0..49] + [777, 778, ...]   (length 70)
      prompt_B = shared[0..49] + [999, 998, ...]   (length 70)
    Block layout (block_size=16):
      block 0: tokens 0..15   (all shared)
      block 1: tokens 16..31  (all shared)
      block 2: tokens 32..47  (all shared)
      block 3: tokens 48..63  (mixed: 48-49 shared, 50 = shared marker,
                              51+ = divergent)
      block 4: tokens 64..69  (partial tail, divergent)
    Expectation: blocks 0,1,2 are shared (refcount 2). Block 3 is the
    divergence-point and has a different hash per request -> distinct
    physical block for each.

    We inspect the block tables BEFORE the requests finish (otherwise
    free_request returns the blocks to the pool and we can't observe
    the shared state). Do this by halting the scheduler right after
    admission/prefill, before the requests run to completion.
    """
    shared_prefix = list(range(100, 150))
    prompt_a = shared_prefix + [777 + i for i in range(20)]
    prompt_b = shared_prefix + [999 - i for i in range(20)]

    scheduler = ContinuousBatchScheduler(
        my_model,
        max_batch_size=8,
        num_blocks=64,
        block_size=BLOCK_SIZE,
        enable_prefix_cache=True,
    )
    scheduler.add_request("req-0", _ids(prompt_a), max_new_tokens=3, eos_token_id=None)
    scheduler.add_request("req-1", _ids(prompt_b), max_new_tokens=3, eos_token_id=None)

    # One step gets both admitted (admit phase), runs prefill, and
    # produces the first decode token each. The requests are still
    # alive (max_new_tokens=2 means one more decode step after the
    # first generated token), so the block tables are still in the pool.
    scheduler.step()

    table_a = scheduler.pool.get_block_table("req-0")
    table_b = scheduler.pool.get_block_table("req-1")

    # First three full blocks must be shared (same physical index).
    for i in range(3):
        assert table_a[i] == table_b[i], (
            f"Block {i} should be shared (identical tokens at same "
            f"positions) but A={table_a[i]} != B={table_b[i]}"
        )
        assert scheduler.pool.ref_count[table_a[i]] == 2, (
            f"Shared block {table_a[i]} should have refcount 2; got "
            f"{scheduler.pool.ref_count[table_a[i]]}"
        )

    # Block 3 holds the divergence point -- must NOT share.
    assert table_a[3] != table_b[3], (
        f"Block 3 holds the divergence point; should be distinct, "
        f"but A={table_a[3]} == B={table_b[3]}"
    )
    assert scheduler.pool.ref_count[table_a[3]] == 1
    assert scheduler.pool.ref_count[table_b[3]] == 1

    # Drain so the fixture doesn't leak refcounts into other tests.
    while scheduler.has_work():
        scheduler.step()


def test_no_shared_prefix(my_model) -> None:
    """2 disjoint prompts: cache=ON behaves identically to cache=OFF.

    No two blocks have matching token sequences, so every admit is
    all-miss. Hit rate = 0%. Outputs must still match the no-cache
    baseline (otherwise we'd be testing a cache that mutates output
    even on misses, which would be a bug in admit_request or the
    seq_len pre-seeding logic).
    """
    prompts = [
        list(range(100, 132)),  # 32 tokens; ids 100..131
        list(range(200, 232)),  # 32 tokens; ids 200..231 (no overlap)
    ]
    off_outs, _ = _run_scheduler(my_model, prompts, enable_prefix_cache=False)
    on_outs, sched_on = _run_scheduler(my_model, prompts, enable_prefix_cache=True)

    for i in range(2):
        rid = f"req-{i}"
        assert off_outs[rid] == on_outs[rid], (
            f"{rid} diverged between cache=OFF and cache=ON despite "
            f"having no shared tokens:\n"
            f"  off: {off_outs[rid]}\n  on:  {on_outs[rid]}"
        )


def test_partial_overlap(my_model) -> None:
    """48-tok prompts sharing first 32 tokens.

    prompt_a = shared[0..31] + suffix_a[0..15]    (48 tokens)
    prompt_b = shared[0..31] + suffix_b[0..15]    (48 tokens, suffix differs)

    Block layout (bs=16):
      block 0: tokens 0..15   (all shared, hashable)
      block 1: tokens 16..31  (all shared, hashable)
      block 2: tokens 32..47  (suffix; differs)

    Important: 48 % 16 == 0 so max_shareable = n_full - 1 = 2. Block
    2 is the LAST FULL BLOCK and is forced fresh (so prefill has a
    writable slot for the next-token logits). Expected state after
    admission:
      A's block 0, 1 == B's block 0, 1; refcount 2 each.
      A's block 2 != B's block 2; refcount 1 each.
    """
    shared = list(range(100, 132))  # 32 tokens
    prompt_a = shared + list(range(300, 316))  # 16-token unique suffix
    prompt_b = shared + list(range(400, 416))

    scheduler = ContinuousBatchScheduler(
        my_model,
        max_batch_size=8,
        num_blocks=64,
        block_size=BLOCK_SIZE,
        enable_prefix_cache=True,
    )
    scheduler.add_request("req-0", _ids(prompt_a), max_new_tokens=3, eos_token_id=None)
    scheduler.add_request("req-1", _ids(prompt_b), max_new_tokens=3, eos_token_id=None)

    scheduler.step()

    table_a = scheduler.pool.get_block_table("req-0")
    table_b = scheduler.pool.get_block_table("req-1")

    # After step 1 the tables have grown past their 3 prefill blocks
    # because the same step also ran one batched decode step (the
    # request entered DECODE state after prefill and decode ran in the
    # same iteration). We only care about the PREFILL portion:
    #   indices 0, 1 = shared (matching tokens at matching positions)
    #   index 2     = forced fresh (48 % 16 == 0 rule)
    #   index 3+    = decode-time growth, always fresh, distinct.
    assert len(table_a) >= 3 and len(table_b) >= 3
    assert table_a[0] == table_b[0], "block 0 must be shared"
    assert table_a[1] == table_b[1], "block 1 must be shared"
    assert table_a[2] != table_b[2], (
        "block 2 is the last full block of an exact-multiple prompt; "
        "the rule says force-fresh, so phys indices must differ"
    )
    assert scheduler.pool.ref_count[table_a[0]] == 2
    assert scheduler.pool.ref_count[table_a[1]] == 2
    assert scheduler.pool.ref_count[table_a[2]] == 1
    assert scheduler.pool.ref_count[table_b[2]] == 1

    while scheduler.has_work():
        scheduler.step()


def test_refcount_decrement(my_model) -> None:
    """Two requests share a block; on finish the refcount drops to 0 and
    the block returns to the free pool with its hash entry evicted.

    Prompt layout (32-token aligned prompts, like test_partial_overlap):
      Both prompts are identical for the first 16 tokens (block 0).
      Block 1 (last full) is forced fresh (32 % 16 == 0 rule), so only
      block 0 is shared.

    Walk:
      * Admit A, admit B (same step). After step 1:
          ref_count[shared_block] == 2
          ref_count[A.block_1] == 1, ref_count[B.block_1] == 1
          hash_to_block has the entry for block 0.
      * Drain to completion. A and B both run a few decode steps and
        eventually exit via max_new_tokens. As each finishes,
        free_request decrements all its blocks.
      * Final state:
          hash_to_block is empty (last refcount hit zero -> evict).
          ref_count is empty (no live blocks).
          free_blocks contains all 64 physical blocks.
    """
    shared = list(range(100, 116))  # 16 tokens -> exactly block 0
    prompt_a = shared + list(range(300, 316))
    prompt_b = shared + list(range(400, 416))

    scheduler = ContinuousBatchScheduler(
        my_model,
        max_batch_size=8,
        num_blocks=64,
        block_size=BLOCK_SIZE,
        enable_prefix_cache=True,
    )
    scheduler.add_request("req-0", _ids(prompt_a), max_new_tokens=3, eos_token_id=None)
    scheduler.add_request("req-1", _ids(prompt_b), max_new_tokens=3, eos_token_id=None)

    # After first step both are admitted; the shared block must have
    # refcount 2.
    scheduler.step()
    table_a = scheduler.pool.get_block_table("req-0")
    table_b = scheduler.pool.get_block_table("req-1")
    assert table_a[0] == table_b[0], "block 0 should be shared"
    shared_phys = table_a[0]
    assert scheduler.pool.ref_count[shared_phys] == 2
    assert shared_phys in scheduler.pool.block_hashes
    h = scheduler.pool.block_hashes[shared_phys]
    assert scheduler.pool.hash_to_block[h] == shared_phys

    # Drain to completion.
    while scheduler.has_work():
        scheduler.step()

    # Final state: pool fully reclaimed, no stale hash entries.
    assert scheduler.pool.hash_to_block == {}, (
        "hash_to_block must be empty after all sharing requests free "
        "their blocks (no LRU retention in v0.2)"
    )
    assert scheduler.pool.block_hashes == {}, "block_hashes must be empty"
    assert scheduler.pool.ref_count == {}, "ref_count must be empty"
    assert len(scheduler.pool._free_blocks) == 64, (
        "all 64 blocks should be back in the free pool"
    )
