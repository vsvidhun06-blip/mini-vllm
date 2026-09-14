"""The live harness must actually evaluate the REPAIRED bandit.

WHY THESE TESTS EXIST. Every GPU number this project ever produced for "CARL"
came from `LinUCBBandit` -- the as-published learner that
`tests/test_eval/test_controllers.py::test_as_published_linucb_locks_on_arm_zero`
proves never leaves arm 0. `RepairedLinUCBBandit` existed only in the simulation
harness. A hardware experiment that silently constructed the inert learner would
measure a stateless `DEFAULT_CONFIGS[classify_regime(state)]` lookup while
claiming to measure online learning, and nothing in the artifact would reveal it.

So the wiring is asserted here rather than trusted: which learner each treatment
constructs, which arm set it draws from, and that the configs the OTHER GPU
experiments call (`run_config("CARL-Full", ...)` in adaptation_analysis.py and
classifier_robustness.py) still get exactly what they got before.
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL = os.path.join(_ROOT, "scripts", "eval")
for _p in (_ROOT, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch", reason="ablation_live imports torch at module level")

import ablation_live as abl  # noqa: E402

from src.carl.bandit import (  # noqa: E402
    LinUCBBandit, PerRegimeBandit, RepairedLinUCBBandit,
)
from src.carl.config import DEFAULT_CONFIGS, all_arm_sets  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.state import FEATURE_DIM, RuntimeState, classify_regime  # noqa: E402

BATCH_AXIS = [2, 4, 8, 12, 16, 24, 32]


def _bandit_for(name: str) -> PerRegimeBandit:
    """Rebuild exactly what run_config builds, without touching a GPU.

    Mirrors the two registry lookups in run_config's CARL branch. If that branch
    is ever changed without updating this helper, the identity assertions below
    stop describing the real harness -- which is why the wiring smoke in
    test_run_config_branch_uses_the_registries also asserts on the source.
    """
    arm_set = abl._ARM_SET_FOR.get(name, "restricted")
    bandit_cls = abl._BANDIT_CLS_FOR.get(name, LinUCBBandit)
    return PerRegimeBandit(abl._frozen_arms(abl._FREEZE.get(name), arm_set),
                           d=FEATURE_DIM, bandit_cls=bandit_cls, alpha=0.5)


# --- the treatments construct the repaired learner -------------------------

@pytest.mark.parametrize("name", ["CARL-Repaired", "CARL-Expanded"])
def test_treatment_constructs_the_repaired_bandit(name):
    """The whole point: these two must NOT be the as-published learner."""
    b = _bandit_for(name)
    for regime, per_regime in b.bandits.items():
        assert isinstance(per_regime, RepairedLinUCBBandit), (
            f"{name}/{regime.value} built {type(per_regime).__name__}, "
            "not RepairedLinUCBBandit")
        # The repair is the intercept feature AND reward centring; a
        # RepairedLinUCBBandit with both disabled degenerates to the original.
        assert per_regime.use_intercept is True
        assert per_regime.center_rewards is True
        assert per_regime.alpha == 0.5, "alpha must stay at the published 0.5"


def test_treatment_arm_sets_are_the_intended_ones():
    assert abl._ARM_SET_FOR.get("CARL-Repaired", "restricted") == "restricted"
    assert abl._ARM_SET_FOR["CARL-Expanded"] == "expanded"


# --- restricted vs expanded remain distinct --------------------------------

def test_restricted_and_expanded_arm_sets_stay_distinct():
    restricted = abl._frozen_arms(None, "restricted")
    expanded = abl._frozen_arms(None, "expanded")
    for regime in restricted:
        r, e = restricted[regime], expanded[regime]
        assert len(e) > len(r), f"{regime.value}: expanded is not larger"
        assert all(a in e for a in r), f"{regime.value}: not a superset"
        assert sorted({a.max_batch_size for a in e}) == BATCH_AXIS
        # Every added arm moves max_batch_size and nothing else.
        base = DEFAULT_CONFIGS[regime].as_dict()
        for a in (x for x in e if x not in r):
            diffs = [k for k, v in a.as_dict().items() if v != base[k]]
            assert diffs == ["max_batch_size"], (regime.value, diffs)


def test_expanded_widens_only_the_batch_axis():
    restricted = abl._frozen_arms(None, "restricted")
    expanded = abl._frozen_arms(None, "expanded")

    def values(sets, dim):
        return sorted({repr(getattr(a, dim))
                       for arms in sets.values() for a in arms})

    for dim in ("chunk_size", "spec_k", "eviction_threshold", "eviction_window",
                "routing_threshold", "cache_affinity_weight", "use_cuda_graphs",
                "preemption_enabled"):
        assert values(restricted, dim) == values(expanded, dim), dim


# --- nothing that already runs has changed ---------------------------------

@pytest.mark.parametrize("name", ["CARL-Full", "CARL-NoSched", "CARL-NoSpec",
                                  "CARL-NoCache", "CARL-NoRouter", "CARL-NoChunk"])
def test_existing_configs_still_build_the_as_published_bandit(name):
    """adaptation_analysis.py and classifier_robustness.py call
    run_config("CARL-Full", ...) directly. Repointing its learner would change
    two other GPU experiments silently."""
    b = _bandit_for(name)
    for per_regime in b.bandits.values():
        assert type(per_regime) is LinUCBBandit, (
            f"{name} must keep the as-published learner, got "
            f"{type(per_regime).__name__}")


def test_existing_configs_still_use_the_shipped_arm_set():
    shipped = all_arm_sets()
    base = abl._frozen_arms(abl._FREEZE.get("CARL-Full"), "restricted")
    assert {r: list(v) for r, v in base.items()} == {r: list(v) for r, v in shipped.items()}


def test_new_treatments_are_opt_in():
    """CONFIGS drives the default run; the treatments must not appear there."""
    for name in abl._TREATMENTS:
        assert name not in abl.CONFIGS
    import inspect
    params = inspect.signature(abl.run_all).parameters
    assert params["include_repaired"].default is False
    assert params["include_expanded"].default is False
    assert abl._frozen_arms.__defaults__[-1] == "restricted"


def test_provenance_fields_survive():
    """arm_set / bandit_cls / search space / validation seed stay recorded."""
    assert abl.VALIDATION_SEED == 999
    assert set(abl.SEARCH_SPACES) == {"restricted", "wide"}
    assert abl.SEARCH_SPACE["max_batch_size"] == [4, 8, 16]
    assert abl.SEARCH_SPACE_WIDE["max_batch_size"] == BATCH_AXIS
    assert abl._bandit_cls_name("CARL-Full") == "LinUCBBandit"
    assert abl._bandit_cls_name("CARL-Repaired") == "RepairedLinUCBBandit"
    assert abl._bandit_cls_name("CARL-Expanded") == "RepairedLinUCBBandit"


def test_run_config_branch_uses_the_registries():
    """Guards the helper above against drifting from the real code path."""
    import inspect
    src = inspect.getsource(abl.run_config)
    assert "_BANDIT_CLS_FOR.get(name, LinUCBBandit)" in src
    assert '_ARM_SET_FOR.get(name, "restricted")' in src
    assert "alpha=0.5" in src


# --- the controller's decision path is otherwise unchanged -----------------

@pytest.mark.parametrize("name", ["CARL-Full", "CARL-Repaired", "CARL-Expanded"])
def test_controller_decision_path_is_unchanged(name):
    """Same observe -> classify -> select -> apply loop for every treatment.

    The repaired learner is a drop-in: it must accept FEATURE_DIM contexts,
    return arms from its own set, and leave the controller's logging, regime
    accounting and adaptation counting untouched.
    """
    bandit = _bandit_for(name)
    c = CARLController(bandit=bandit, observe_interval=1)
    states = [RuntimeState(queue_depth=2, avg_prompt_len=40.0),
              RuntimeState(queue_depth=60, avg_prompt_len=300.0),
              RuntimeState(queue_depth=5, avg_prompt_len=1200.0)]
    for st in states:
        c.step(state=st)

    assert len(c.controller_log) == len(states)
    for entry, st in zip(c.controller_log, states):
        regime = classify_regime(st)
        assert entry.regime is regime
        # The applied config came from THAT regime's arm set.
        assert entry.config in bandit.arms(regime)
        assert len(entry.state_features) == FEATURE_DIM
        # Timestamp instrumentation still populated.
        assert isinstance(entry.t_monotonic_s, float) and entry.t_monotonic_s > 0.0
    assert [e.t_monotonic_s for e in c.controller_log] == sorted(
        e.t_monotonic_s for e in c.controller_log)
    assert c.stats()["total_adaptations"] >= 1


def test_repaired_learner_actually_learns_in_the_live_wiring():
    """A behavioural check, not just an isinstance check.

    Driven on a fixed regime with a reward that favours a non-zero arm, the
    repaired learner must leave arm 0 -- the exact failure that makes the
    as-published bandit inert on hardware.
    """
    bandit = _bandit_for("CARL-Repaired")
    state = RuntimeState(queue_depth=2, avg_prompt_len=40.0)
    regime = classify_regime(state)
    ctx = state.to_feature_vector()
    n_arms = len(bandit.arms(regime))
    best = n_arms - 1
    for _ in range(200):
        arm, _cfg = bandit.select(regime, ctx)
        bandit.update(regime, arm, 0.9 if arm == best else 0.3, ctx)
    counts = bandit.selection_counts()[regime.value]
    assert counts[0] < 200, "repaired learner still locked on arm 0"
    assert counts[best] == max(counts), f"did not converge to the best arm: {counts}"
