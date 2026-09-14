"""RuleOnly must be the null baseline, and must run the controller's own path.

The whole interpretation of `CARL-Repaired - RuleOnly` as "what learning bought"
rests on two properties that are cheap to assert and expensive to discover
broken after a GPU run:

  1. RuleOnly plays `DEFAULT_CONFIGS[regime]` and nothing else.
  2. It is driven through the IDENTICAL `CARLController` code path as CARL, so
     any behavioural difference comes from the policy and not from a divergent
     branch.
"""
from __future__ import annotations

import pytest

from src.carl.config import DEFAULT_CONFIGS, all_arm_sets
from src.carl.controller import CARLController
from src.carl.rule_only import RuleOnlyBandit
from src.carl.state import FEATURE_DIM, RuntimeState, classify_regime

_STATES = [
    RuntimeState(queue_depth=2, avg_prompt_len=40.0, throughput_tps=60.0),
    RuntimeState(queue_depth=60, avg_prompt_len=300.0, throughput_tps=90.0),
    RuntimeState(queue_depth=5, avg_prompt_len=1200.0, throughput_tps=30.0),
]


def _bandit() -> RuleOnlyBandit:
    return RuleOnlyBandit(all_arm_sets(), d=FEATURE_DIM)


def test_always_plays_arm_zero_which_is_the_regime_default():
    b = _bandit()
    for regime in b.arms_by_regime:
        arm, cfg = b.select(regime, [0.0] * FEATURE_DIM)
        assert arm == 0
        assert cfg == DEFAULT_CONFIGS[regime]


def test_selection_is_independent_of_context():
    """RuleOnly is stateless: no context may change what it plays."""
    b = _bandit()
    regime = next(iter(b.arms_by_regime))
    seen = {b.select(regime, [float(i)] * FEATURE_DIM)[0] for i in range(20)}
    assert seen == {0}


def test_update_is_a_noop_but_accepts_the_reward():
    """It must be SCORED by the same reward it ignores, so the controller's
    reward computation, gating and logging all run unchanged."""
    b = _bandit()
    regime = next(iter(b.arms_by_regime))
    ctx = [0.1] * FEATURE_DIM
    before = b.selection_counts()
    b.update(regime, 0, 0.9, ctx)
    b.update(regime, 0, 0.1, ctx)
    assert b.selection_counts() == before, "update() must not change state"
    assert b.select(regime, ctx)[0] == 0


def test_rejects_an_arm_set_whose_arm_zero_is_not_the_default():
    """If arm 0 drifted from DEFAULT_CONFIGS, RuleOnly would silently become
    some other policy and the comparison would stop isolating learning."""
    arms = {r: list(v) for r, v in all_arm_sets().items()}
    regime = next(iter(arms))
    arms[regime] = list(reversed(arms[regime]))       # arm 0 is no longer default
    with pytest.raises(ValueError, match="not DEFAULT_CONFIGS"):
        RuleOnlyBandit(arms, d=FEATURE_DIM)


def test_is_interface_compatible_with_the_controller():
    """Same observe -> classify -> select -> apply -> reward loop as CARL."""
    b = _bandit()
    c = CARLController(bandit=b, observe_interval=1)
    for st in _STATES:
        c.step(state=st)

    assert len(c.controller_log) == len(_STATES)
    for entry, st in zip(c.controller_log, _STATES):
        regime = classify_regime(st)
        assert entry.regime is regime
        assert entry.config == DEFAULT_CONFIGS[regime]
        assert entry.arm == 0
        assert len(entry.state_features) == FEATURE_DIM
        assert entry.observed["queue_depth"] == st.queue_depth
    # It still adapts -- by switching regimes, which is the classifier's doing.
    assert c.stats()["total_adaptations"] >= 1


def test_reset_clears_counts():
    b = _bandit()
    regime = next(iter(b.arms_by_regime))
    b.select(regime, [0.0] * FEATURE_DIM)
    assert sum(b.selection_counts()[regime.value]) == 1
    b.reset()
    assert sum(b.selection_counts()[regime.value]) == 0


def test_arms_still_describes_the_full_action_space():
    """Arm indices in a RuleOnly trace must mean what they mean for CARL."""
    b = _bandit()
    shipped = all_arm_sets()
    for regime, arms in shipped.items():
        assert list(b.arms(regime)) == list(arms)
