"""Controller-level invariants: bandit repair, closed-loop feedback, rule-only.

Each test corresponds to a finding in docs/eval/CARL_REPAIR_STATUS.md.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPAIR = os.path.join(_ROOT, "scripts", "eval", "repair")
for _p in (_ROOT, _REPAIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harness2 as H  # noqa: E402

from src.carl.bandit import LinUCBBandit, RepairedLinUCBBandit  # noqa: E402
from src.carl.config import CARLConfig, DEFAULT_CONFIGS, all_arm_sets  # noqa: E402
from src.carl.reward import RewardScales  # noqa: E402
from src.carl.state import WorkloadRegime, classify_regime  # noqa: E402
from src.eval.engine_model import HardwareProfile, simulate  # noqa: E402

HW = HardwareProfile()
SCALES = RewardScales(t_half=60.0, ttft_target=250.0, tpot_target=77.0)

# A context with the SAME magnitude the real observer produces (||x|| ~ 0.5).
CTX = [0.03, 0.04, 0.0, 0.07, 0.25, 0.13, 0.09, 0.40, 0.06, 0.06]
TRUE = [0.70, 0.72, 0.82, 0.68]     # arm 2 is best


def _drive(cls, n=200, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    b = cls(4, len(CTX), alpha=alpha)
    for _ in range(n):
        a = b.select(CTX)
        b.update(a, TRUE[a] + rng.normal(0, 0.01), CTX)
    return b


# --- R1/R2: the as-published bandit cannot explore -------------------------

def test_as_published_linucb_locks_on_arm_zero():
    """Reproduces the measured defect on a problem where arm 2 is provably best."""
    b = _drive(LinUCBBandit)
    assert b.counts[0] == 200, f"expected total lock-in, got {b.counts}"
    assert b.counts[2] == 0, "never even tries the best arm"


def test_repaired_linucb_finds_the_best_arm():
    b = _drive(RepairedLinUCBBandit)
    assert b.counts[2] == max(b.counts), f"did not converge to arm 2: {b.counts}"
    assert sum(1 for c in b.counts if c > 0) >= 3, "must actually explore"


def test_repair_is_the_intercept_and_the_centring():
    """Isolates each half of the fix so neither is cargo-culted."""
    neither = RepairedLinUCBBandit(4, len(CTX), alpha=0.5,
                                   use_intercept=False, center_rewards=False)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = neither.select(CTX)
        neither.update(a, TRUE[a] + rng.normal(0, 0.01), CTX)
    # With both repairs disabled it must degenerate exactly like the original.
    assert neither.counts[0] == 200


def test_intercept_raises_the_context_norm_above_one():
    b = RepairedLinUCBBandit(2, len(CTX), alpha=0.5)
    x = b._context(CTX)
    assert len(x) == len(CTX) + 1
    assert np.linalg.norm(x) >= 1.0


def test_reward_baseline_tracks_the_running_mean():
    b = RepairedLinUCBBandit(2, len(CTX))
    for r in (0.2, 0.4, 0.6):
        b.update(0, r, CTX)
    assert b.reward_baseline == pytest.approx(0.4)


# --- Phase 5: RuleOnly is genuinely stateless ------------------------------

def test_rule_only_has_no_learned_state():
    c = H.RuleOnlyController()
    assert not hasattr(c, "bandit")
    from src.carl.state import RuntimeState
    s = RuntimeState(queue_depth=2, avg_prompt_len=40.0)
    first = c.choose(s)
    for _ in range(50):
        c.observe(0.9)          # feed it rewards; it must ignore them entirely
    assert c.choose(s) == first, "RuleOnly must be a pure function of the state"


def test_rule_only_matches_the_default_config_table():
    from src.carl.state import RuntimeState
    c = H.RuleOnlyController()
    s = RuntimeState(queue_depth=2, avg_prompt_len=40.0)
    assert c.choose(s) == DEFAULT_CONFIGS[classify_regime(s)]


# --- Phase 9: the AutoTuner loop is closed ---------------------------------

def test_autotuner_observes_consequences_of_its_own_configuration():
    """The defect: benchmark_carl fed a fixed per-regime profile that ignored the
    chosen config. Two different configs must now yield two different profiles."""
    w = H.WORKLOADS["batch"]
    a, b = H.ClosedLoopAutoTunerController(), H.ClosedLoopAutoTunerController()
    a.feed(simulate(CARLConfig(max_batch_size=2, chunk_size=64),
                    w.generate(random.Random(0)), HW, random.Random(1)))
    b.feed(simulate(CARLConfig(max_batch_size=32, chunk_size=512),
                    w.generate(random.Random(0)), HW, random.Random(1)))
    assert a.observed_profiles[0] != b.observed_profiles[0], (
        "the AutoTuner is open-loop: its observation does not depend on the "
        "configuration it selected")


def test_legacy_autotuner_agent_is_open_loop():
    """Documents the original defect so a 'fix' cannot silently regress it."""
    import inspect

    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import benchmark_carl as bc
    src = inspect.getsource(bc.AutoTunerAgent.choose)
    assert "_BOTTLENECK[true_regime]" in src, (
        "AutoTunerAgent.choose no longer keys its observation on the true "
        "regime -- if it was repaired, update this test and the claim ledger")


# --- R6: seeds must matter -------------------------------------------------

def test_seeds_produce_distinct_episodes():
    vals = {H.run_episode(H.RuleOnlyController(), ["interactive", "batch"],
                          HW, SCALES, s, n_per_phase=20)["mean_reward"]
            for s in range(5)}
    assert len(vals) == 5, "identical results across seeds means the seed is dead"


# --- Phase 8/11: the two static baselines are genuinely different ----------

def test_static_grid_is_wider_than_the_bandit_arm_set():
    grid = {(c.max_batch_size, c.chunk_size) for c in H.static_candidates()}
    arms = {(a.max_batch_size, a.chunk_size)
            for s in all_arm_sets().values() for a in s}
    assert not grid.issubset(arms), (
        "the static baseline must be able to reach configs CARL cannot, "
        "otherwise it is handicapped by construction")


def test_arm_set_coverage_is_reported_per_regime():
    """Pooling across regimes hides that INTERACTIVE caps at max_batch_size=8."""
    cov = H.arm_set_coverage(CARLConfig(max_batch_size=16), all_arm_sets())
    assert cov["per_regime"]["interactive"]["reachable_max_batch_sizes"] == [2, 4, 8]
    assert cov["per_regime"]["interactive"]["target_max_batch_reachable"] is False
    assert cov["reachable_in_every_regime"] is False
