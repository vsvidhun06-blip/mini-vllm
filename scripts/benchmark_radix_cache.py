"""
Benchmark: flat hash-based prefix cache vs radix-tree prefix cache.

Run:
    python scripts/benchmark_radix_cache.py

Workload (the canonical shared-prefix serving pattern):
    100 requests, each = a SHARED 128-token system prompt + a UNIQUE 32-token
    user message. With block_size 16 that's 8 shared blocks + 2 unique blocks
    = 10 blocks per request. Requests arrive and COMPLETE sequentially (the
    realistic case: you can't hold 100 sequences resident at once).

Why the two caches differ on this workload
-------------------------------------------
Both can share the 8 system-prompt blocks. The difference is RETENTION:

  * The hash cache (PagedKVCache) frees a block as soon as its last live
    request finishes -- "no LRU retention". On a sequential stream only one
    request is resident at a time, so the shared prefix is gone before the next
    request arrives. Cross-request hit rate ~ 0.

  * The radix cache (RadixPagedKVCache) inserts a completed request's prefix
    into the tree and keeps it resident until LRU eviction. Every later request
    re-walks and reuses the 8 system-prompt blocks. Hit rate ~ 80%.

Metrics: cache hit rate (reused blocks / blocks needed), blocks saved vs
no-cache, and mean admit() time.
"""
from __future__ import annotations

import time

from src.engine.kv_cache import PagedKVCache, RadixPagedKVCache

# --- workload shape -------------------------------------------------------
BLOCK_SIZE = 16
SYS_PROMPT_LEN = 128                 # shared across all requests -> 8 blocks
USER_MSG_LEN = 32                    # unique per request        -> 2 blocks
PROMPT_LEN = SYS_PROMPT_LEN + USER_MSG_LEN          # 160 tokens -> 10 blocks
BLOCKS_PER_REQ = PROMPT_LEN // BLOCK_SIZE
N_REQUESTS = 100

# Pool tensors are tiny here (we never run a forward); keep them small.
POOL_KW = dict(
    num_layers=1, block_size=BLOCK_SIZE, num_kv_heads=1, head_dim=8,
    dtype=None, device="cpu",
)


def _make_pool(cls, num_blocks):
    kw = dict(POOL_KW)
    import torch
    kw["dtype"] = torch.float32
    return cls(num_blocks=num_blocks, **kw)


def _build_prompts() -> list[list[int]]:
    """100 prompts: identical system prefix, unique user suffix."""
    system = list(range(1, SYS_PROMPT_LEN + 1))     # tokens 1..128, shared
    prompts = []
    for r in range(N_REQUESTS):
        # Unique user tokens drawn from a per-request disjoint range so no two
        # requests accidentally share a suffix block.
        base = 100_000 + r * 1000
        user = list(range(base, base + USER_MSG_LEN))
        prompts.append(system + user)
    return prompts


def _chained_block_hashes(token_ids: list[int]) -> list[int | None]:
    """Replicate the scheduler's chained per-block hash for the hash cache."""
    bs = BLOCK_SIZE
    n_full = len(token_ids) // bs
    has_partial = (len(token_ids) % bs) != 0
    max_shareable = max(0, n_full - (0 if has_partial else 1))
    n_prefill = (len(token_ids) + bs - 1) // bs
    hashes: list[int | None] = []
    prev = 0
    for i in range(n_prefill):
        if i < max_shareable:
            chunk = tuple(token_ids[i * bs:(i + 1) * bs])
            prev = hash((prev, chunk, i * bs))
            hashes.append(prev)
        else:
            hashes.append(None)
    return hashes


def _run_hash_cache(prompts) -> dict:
    pool = _make_pool(PagedKVCache, num_blocks=64)
    total_hits = 0
    admit_times: list[float] = []
    allocated_fresh = 0
    for r, prompt in enumerate(prompts):
        rid = f"r{r}"
        hashes = _chained_block_hashes(prompt)
        t0 = time.perf_counter()
        hits = pool.admit_request(
            request_id=rid,
            prefill_blocks_needed=BLOCKS_PER_REQ,
            total_blocks_needed=BLOCKS_PER_REQ,
            prefill_block_hashes=hashes,
        )
        admit_times.append(time.perf_counter() - t0)
        total_hits += hits
        allocated_fresh += BLOCKS_PER_REQ - hits
        # Sequential serving: the request finishes before the next arrives.
        pool.free_request(rid)
    return _summarise("hash", total_hits, allocated_fresh, admit_times)


def _run_radix_cache(prompts) -> dict:
    # Pool big enough to retain shared prefix + every request's unique blocks
    # so we measure hit rate without eviction noise (8 + 100*2 = 208 < 512).
    pool = _make_pool(RadixPagedKVCache, num_blocks=512)
    total_hits = 0
    admit_times: list[float] = []
    allocated_fresh = 0
    for r, prompt in enumerate(prompts):
        rid = f"r{r}"
        t0 = time.perf_counter()
        hits = pool.admit_request(
            request_id=rid,
            prefill_blocks_needed=BLOCKS_PER_REQ,
            total_blocks_needed=BLOCKS_PER_REQ,
            token_ids=prompt,
        )
        admit_times.append(time.perf_counter() - t0)
        total_hits += hits
        allocated_fresh += BLOCKS_PER_REQ - hits
        pool.free_request(rid)
    return _summarise("radix", total_hits, allocated_fresh, admit_times,
                      retained=pool.num_cached_prefix_blocks())


def _summarise(name, total_hits, allocated_fresh, admit_times, retained=None) -> dict:
    blocks_needed = N_REQUESTS * BLOCKS_PER_REQ
    return {
        "name": name,
        "hit_rate": total_hits / blocks_needed,
        "blocks_saved": blocks_needed - allocated_fresh,
        "mean_admit_us": 1e6 * sum(admit_times) / len(admit_times),
        "retained": retained,
    }


def main() -> None:
    prompts = _build_prompts()
    print(f"{N_REQUESTS} requests | {SYS_PROMPT_LEN}-token shared system prompt "
          f"+ {USER_MSG_LEN}-token unique user msg")
    print(f"block_size={BLOCK_SIZE} -> {BLOCKS_PER_REQ} blocks/req "
          f"({SYS_PROMPT_LEN // BLOCK_SIZE} shared + {USER_MSG_LEN // BLOCK_SIZE} unique)\n")

    hash_res = _run_hash_cache(prompts)
    radix_res = _run_radix_cache(prompts)

    header = f"{'cache':>6} | {'hit rate':>8} | {'blocks saved':>12} | {'mean admit (us)':>15}"
    print(header)
    print("-" * len(header))
    for res in (hash_res, radix_res):
        print(f"{res['name']:>6} | {res['hit_rate']:>7.1%} | "
              f"{res['blocks_saved']:>12} | {res['mean_admit_us']:>15.2f}")

    print(f"\nRadix tree retained {radix_res['retained']} prefix blocks across "
          f"completions.")
    print("The hash cache shares only among CO-RESIDENT requests (no retention "
          "after\ncompletion), so on a sequential stream its cross-request hit "
          "rate is ~0.\nThe radix tree keeps the shared system prompt resident, "
          "so every request\nafter the first reuses its 8 blocks -- higher hit "
          "rate, fewer allocations.")


if __name__ == "__main__":
    main()
