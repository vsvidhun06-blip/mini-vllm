"""
H2O KV-cache eviction tests.

H2O evicts the lowest-attention ("cold") cache entries while always keeping a
recency window, so a fixed memory budget can serve a much longer sequence. These
tests pin the properties that make it correct:

  1. Eviction fires once occupancy crosses the 80% threshold.
  2. Recency bias: the most-recent `recent_window` tokens are never evicted.
  3. Position remapping: after eviction the RoPE position (true absolute) is
     decoupled from the resident entry count, and the per-slot position map
     stays consistent (monotonic, aligned with the survivors).
  4. An evicting cache processes >= 2x more tokens than a non-evicting cache of
     the SAME block budget (which simply runs out of blocks).

Plus a pure-logic unit test of the scorer (no model). Everything runs on a tiny
random-weight LlamaModel on CPU -- no GPU, no HF download.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.kv_cache import PagedKVCache, PagedRequestCache
from src.engine.kv_eviction import (
    AttentionScoreTracker,
    EvictingPagedKVCache,
    make_evicting_cache,
)
from src.engine.model import LlamaConfig, LlamaModel

BLOCK_SIZE = 16


def _tiny_model() -> LlamaModel:
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,       # head_dim = 16
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


def _drive(model, cache, n_tokens, seed=0):
    """Feed `n_tokens` single tokens through the model with `cache`, evicting
    between steps when `cache` supports it."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_tokens):
        tok = torch.randint(0, 256, (1, 1), generator=g)
        with torch.no_grad():
            model(tok, kv_cache=cache)
        if isinstance(cache, EvictingPagedKVCache):
            cache.maybe_evict()


# ---------------------------------------------------------------------------
# Unit: the scorer's keep-top-k + recency logic (no model).
# ---------------------------------------------------------------------------


def test_tracker_keeps_topk_plus_recency():
    tr = AttentionScoreTracker(num_layers=1, recent_window=4)
    # 10 positions. Scores chosen so the heavy hitters among the first six are
    # positions 1 (9) and 2 (8); position 5 (7) is third and must be dropped.
    scores = torch.tensor([0., 9, 8, 1, 1, 7, 0.5, 0.5, 0.5, 0.5]).reshape(1, 1, 1, 10)
    tr.update(0, scores)

    evict = tr.get_eviction_candidates(keep_budget=6)
    # Survivors: recency {6,7,8,9} + top-2 heavy {1,2}. Evict the rest.
    assert set(evict) == {0, 3, 4, 5}

    # At/under budget -> nothing to evict.
    assert tr.get_eviction_candidates(keep_budget=10) == []
    assert tr.get_eviction_candidates(keep_budget=20) == []


def test_tracker_evict_compacts_aligned():
    tr = AttentionScoreTracker(num_layers=1, recent_window=2)
    tr.update(0, torch.tensor([3., 1., 4., 1., 5.]).reshape(1, 1, 1, 5))
    tr.evict([1, 3])  # drop positions 1 and 3
    assert torch.allclose(tr.scores, torch.tensor([3., 4., 5.], dtype=torch.float64))


# ---------------------------------------------------------------------------
# 1. Eviction fires at 80% capacity.
# ---------------------------------------------------------------------------


def test_eviction_fires_at_80_percent():
    model = _tiny_model()
    cache = make_evicting_cache(
        model, capacity_tokens=40, keep_budget=20, recent_window=8,
    )
    # 31 tokens -> 31 < 0.8*40 = 32, so no eviction yet.
    _drive(model, cache, 31)
    assert cache.num_evictions == 0
    assert cache.entry_count() == 31

    # The 32nd token crosses the threshold and triggers exactly one eviction
    # down to keep_budget.
    _drive(model, cache, 1, seed=1)
    assert cache.num_evictions == 1
    assert cache.entry_count() == 20


# ---------------------------------------------------------------------------
# 2. Recency bias: last `recent_window` tokens are never evicted.
# ---------------------------------------------------------------------------


def test_recency_window_never_evicted():
    model = _tiny_model()
    rw = 8
    cache = make_evicting_cache(
        model, capacity_tokens=40, keep_budget=20, recent_window=rw,
    )
    _drive(model, cache, 60)          # enough to trigger several evictions
    assert cache.num_evictions >= 1

    resident = set(cache._abs_positions)
    last_window = set(range(cache._tokens_seen - rw, cache._tokens_seen))
    assert last_window.issubset(resident), "a recent token was evicted"


# ---------------------------------------------------------------------------
# 3. Position remapping is consistent after eviction.
# ---------------------------------------------------------------------------


def test_position_remapping_consistent():
    model = _tiny_model()
    cache = make_evicting_cache(
        model, capacity_tokens=40, keep_budget=20, recent_window=8,
    )
    _drive(model, cache, 60)
    assert cache.num_evictions >= 1

    # RoPE position == total tokens seen (monotonic), and exceeds the resident
    # entry count once eviction has happened -- the two are decoupled.
    assert cache.seq_len() == cache._tokens_seen == 60
    assert cache.entry_count() < cache.seq_len()

    # The per-slot absolute-position map matches the resident count, is strictly
    # increasing (survivors stay in temporal order), and stays in range.
    ap = cache._abs_positions
    assert len(ap) == cache.entry_count()
    assert all(ap[i] < ap[i + 1] for i in range(len(ap) - 1)), "positions not monotonic"
    assert 0 <= ap[0] and ap[-1] < cache._tokens_seen


# ---------------------------------------------------------------------------
# 4. Evicting cache fits >= 2x the tokens of a non-evicting cache, same budget.
# ---------------------------------------------------------------------------


def test_evicting_cache_fits_2x_more_tokens():
    model = _tiny_model()
    cfg = model.config
    cap = 32                          # 2 blocks of 16
    cap_blocks = cap // BLOCK_SIZE
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # Non-evicting baseline with the same block budget must run out of blocks
    # well before 2x the budget -- it has no way to make room.
    base_pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_blocks=cap_blocks, block_size=BLOCK_SIZE,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        dtype=next(model.parameters()).dtype, device=next(model.parameters()).device,
    )
    base_pool.admit_request("b", prefill_blocks_needed=0, total_blocks_needed=cap_blocks)
    base = PagedRequestCache(base_pool, "b", num_layers=cfg.num_hidden_layers)
    with pytest.raises(RuntimeError):
        _drive(model, base, 2 * cap)

    # Evicting cache with the SAME budget processes 2x the tokens, no error.
    cache = make_evicting_cache(
        model, capacity_tokens=cap, keep_budget=cap // 2, recent_window=8,
    )
    _drive(model, cache, 2 * cap)
    assert cache._tokens_seen == 2 * cap
    assert cache.entry_count() <= cap
    assert cache.num_evictions >= 1


# ---------------------------------------------------------------------------
# Guard: a plain cache (no tracker) is on the fast attention path -- attaching
# eviction must be opt-in and not perturb the standard forward.
# ---------------------------------------------------------------------------


def test_plain_cache_has_no_tracker():
    model = _tiny_model()
    cfg = model.config
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_blocks=8, block_size=BLOCK_SIZE,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=next(model.parameters()).dtype, device=next(model.parameters()).device,
    )
    pool.admit_request("p", prefill_blocks_needed=0, total_blocks_needed=8)
    plain = PagedRequestCache(pool, "p", num_layers=cfg.num_hidden_layers)
    assert getattr(plain, "score_tracker", None) is None
