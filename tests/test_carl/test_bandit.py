"""
Tests for the contextual-bandit controllers and the utility reward.

CPU, torch-free, model-free. The only dependency is numpy (the bandit's per-arm
linear algebra). We verify the LinUCB math directly (A/b updates, exploration),
that selection follows reward, that per-regime bandits are isolated, that
Thompson Sampling concentrates on the better arm, and that utility() computes the
documented weighted sum.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.carl.bandit import (
    DEFAULT_UTILITY_WEIGHTS,
    LinUCBBandit,
    PerRegimeBandit,
    ThompsonSamplingBandit,
    utility,
)
from src.carl.config import all_arm_sets
from src.carl.state import FEATURE_DIM, WorkloadRegime


# ---------------------------------------------------------------------------
# utility()
# ---------------------------------------------------------------------------


def test_utility_known_inputs():
    # All-perfect metrics -> reward = sum of weights = 1.0.
    m = {
        "throughput_norm": 1.0,
        "ttft_violation_rate": 0.0,
        "tpot_violation_rate": 0.0,
        "cache_hit_rate": 1.0,
    }
    assert utility(m) == pytest.approx(1.0)


def test_utility_all_bad_is_zero():
    m = {
        "throughput_norm": 0.0,
        "ttft_violation_rate": 1.0,
        "tpot_violation_rate": 1.0,
        "cache_hit_rate": 0.0,
    }
    assert utility(m) == pytest.approx(0.0)


def test_utility_weighted_sum():
    # Half throughput, no violations, no cache.
    m = {
        "throughput_norm": 0.5,
        "ttft_violation_rate": 0.0,
        "tpot_violation_rate": 0.0,
        "cache_hit_rate": 0.0,
    }
    w = DEFAULT_UTILITY_WEIGHTS
    expected = w["throughput"] * 0.5 + w["ttft"] * 1.0 + w["tpot"] * 1.0 + w["cache"] * 0.0
    assert utility(m) == pytest.approx(expected)


def test_utility_missing_keys_default_neutral():
    # Empty dict: throughput 0, no violations assumed, cache 0.
    # -> w_ttft + w_tpot.
    expected = DEFAULT_UTILITY_WEIGHTS["ttft"] + DEFAULT_UTILITY_WEIGHTS["tpot"]
    assert utility({}) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LinUCB update math.
# ---------------------------------------------------------------------------


def test_linucb_init_shapes():
    b = LinUCBBandit(n_arms=3, d=4)
    assert len(b.A) == 3 and len(b.b) == 3
    for a in range(3):
        assert np.allclose(b.A[a], np.identity(4))
        assert np.allclose(b.b[a], np.zeros(4))


def test_linucb_update_modifies_A_and_b():
    b = LinUCBBandit(n_arms=2, d=3)
    x = [1.0, 2.0, 3.0]
    b.update(arm=1, reward=2.0, context=x)
    xv = np.array(x)
    # A_1 = I + x x^T ; b_1 = r x.
    assert np.allclose(b.A[1], np.identity(3) + np.outer(xv, xv))
    assert np.allclose(b.b[1], 2.0 * xv)
    # Arm 0 untouched.
    assert np.allclose(b.A[0], np.identity(3))
    assert np.allclose(b.b[0], np.zeros(3))


def test_linucb_update_validates_arm():
    b = LinUCBBandit(n_arms=2, d=2)
    with pytest.raises(IndexError):
        b.update(arm=5, reward=1.0, context=[1.0, 1.0])


def test_linucb_context_dim_checked():
    b = LinUCBBandit(n_arms=2, d=3)
    with pytest.raises(ValueError):
        b.select([1.0, 2.0])          # wrong dim


def test_linucb_selection_follows_reward():
    # Two arms, fixed context. Reward arm 1 repeatedly; it must become preferred.
    b = LinUCBBandit(n_arms=2, d=2, alpha=0.1)
    x = [1.0, 0.0]
    for _ in range(20):
        b.update(arm=1, reward=1.0, context=x)
        b.update(arm=0, reward=0.0, context=x)
    assert b.select(x) == 1


def test_linucb_exploration_prefers_untried_arm():
    # With a big alpha and one arm already pulled (tighter confidence), the
    # untried arm's exploration bonus should win at the same context.
    b = LinUCBBandit(n_arms=2, d=2, alpha=2.0)
    x = [1.0, 1.0]
    # Give arm 0 a modest positive reward and several pulls (shrinks its bonus).
    for _ in range(5):
        b.update(arm=0, reward=0.3, context=x)
    # Arm 1 is untried -> maximal exploration bonus -> selected.
    assert b.select(x) == 1


def test_linucb_cold_start_breaks_to_arm_zero():
    # All-identity cold start: every arm scores identically -> argmax picks 0.
    b = LinUCBBandit(n_arms=4, d=FEATURE_DIM)
    assert b.select([0.1] * FEATURE_DIM) == 0


# ---------------------------------------------------------------------------
# Thompson Sampling.
# ---------------------------------------------------------------------------


def test_thompson_concentrates_on_better_arm():
    # Seeded for determinism. Arm 1 yields higher reward; over many draws TS
    # should select it the majority of the time once learned.
    b = ThompsonSamplingBandit(n_arms=2, d=2, v=0.2, seed=0)
    x = [1.0, 0.0]
    for _ in range(50):
        b.update(arm=1, reward=1.0, context=x)
        b.update(arm=0, reward=0.0, context=x)
    picks = [b.select(x) for _ in range(200)]
    assert picks.count(1) > picks.count(0)


def test_thompson_update_matches_linucb_stats():
    # TS uses the same sufficient statistics as LinUCB.
    b = ThompsonSamplingBandit(n_arms=2, d=2, seed=1)
    x = [1.0, 2.0]
    b.update(arm=0, reward=1.5, context=x)
    xv = np.array(x)
    assert np.allclose(b.A[0], np.identity(2) + np.outer(xv, xv))
    assert np.allclose(b.b[0], 1.5 * xv)


# ---------------------------------------------------------------------------
# PerRegimeBandit isolation.
# ---------------------------------------------------------------------------


def test_per_regime_bandit_independent():
    prb = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM, alpha=0.5)
    ctx = [0.2] * FEATURE_DIM

    # Update only the BURST bandit, arm 1.
    prb.update(WorkloadRegime.BURST, arm=1, reward=1.0, context=ctx)

    burst = prb.bandits[WorkloadRegime.BURST]
    interactive = prb.bandits[WorkloadRegime.INTERACTIVE]
    # BURST arm 1 changed.
    assert not np.allclose(burst.A[1], np.identity(FEATURE_DIM))
    # INTERACTIVE untouched -- all arms still at the identity prior.
    for a in range(interactive.n_arms):
        assert np.allclose(interactive.A[a], np.identity(FEATURE_DIM))


def test_per_regime_select_returns_arm_and_config():
    prb = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM)
    arm, config = prb.select(WorkloadRegime.INTERACTIVE, [0.1] * FEATURE_DIM)
    assert isinstance(arm, int)
    # Cold start -> arm 0 -> the regime's hand-tuned default config.
    assert arm == 0
    assert config == prb.arms(WorkloadRegime.INTERACTIVE)[0]


def test_per_regime_reset_wipes_learning():
    prb = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM)
    ctx = [0.3] * FEATURE_DIM
    prb.update(WorkloadRegime.BATCH, arm=2, reward=1.0, context=ctx)
    assert not np.allclose(prb.bandits[WorkloadRegime.BATCH].A[2], np.identity(FEATURE_DIM))
    prb.reset()
    assert np.allclose(prb.bandits[WorkloadRegime.BATCH].A[2], np.identity(FEATURE_DIM))


def test_per_regime_select_uses_thompson_when_asked():
    prb = PerRegimeBandit(
        all_arm_sets(), d=FEATURE_DIM, bandit_cls=ThompsonSamplingBandit, seed=0,
    )
    assert isinstance(prb.bandits[WorkloadRegime.BATCH], ThompsonSamplingBandit)
    arm, config = prb.select(WorkloadRegime.BATCH, [0.1] * FEATURE_DIM)
    assert 0 <= arm < len(prb.arms(WorkloadRegime.BATCH))
