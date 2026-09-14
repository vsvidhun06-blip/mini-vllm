"""RuleOnly -- the null hypothesis, as a drop-in for the LIVE controller.

WHY THIS EXISTS
---------------
The paper's central question is what the *learning* buys, not what the *regime
classifier* buys. Those two are entangled in every CARL configuration: CARL
observes a state, classifies a regime, and then a bandit picks an arm within
that regime. Arm 0 of every regime IS `DEFAULT_CONFIGS[regime]` (see
`config.config_arms`), so a bandit that never leaves arm 0 is behaviourally a
stateless lookup table -- which is exactly what the as-published LinUCB was
measured to be (`docs/eval/raw/repair/bandit_null_check.json`, 5/5 seeds).

RuleOnly makes that lookup table an explicit, first-class baseline:

    classify the observed state -> play DEFAULT_CONFIGS[regime] -> done.

No arm indices beyond 0, no learned statistics, no exploration, no context.

WHAT THE COMPARISON ISOLATES
----------------------------
RuleOnly runs the SAME observe -> classify -> apply -> reward loop, at the SAME
cadence, over the SAME arm-0 configs, and is subject to the SAME classifier
error as CARL. The ONLY thing it lacks is learning. Therefore

    CARL-Repaired - RuleOnly  ==  what online learning actually bought

with every other factor held fixed. A tie means the contribution is the
classifier, not the bandit.

WHY A BANDIT-SHAPED SHIM RATHER THAN A CONTROLLER FLAG
------------------------------------------------------
`CARLController` consumes its policy through a small duck-typed interface
(`select`, `update`, `arms`, `selection_counts`, `reset`, `bandits`). Supplying
RuleOnly as an object satisfying that interface means the controller needs NO
branch for it: identical locking, identical delayed-reward timing, identical
logging, identical adaptation accounting. Any behavioural difference between
RuleOnly and CARL is then guaranteed to come from the policy and not from a
divergent code path -- which is the entire point of a null ablation.

`update()` is deliberately a no-op that still ACCEPTS the reward, so the
controller's reward computation, degeneracy checking and logging all run
unchanged. RuleOnly is scored by exactly the same reward it ignores.

This mirrors `scripts/eval/repair/harness2.RuleOnlyController` (the simulation's
null baseline) so the hardware and simulation results answer the same question.
"""
from __future__ import annotations

from src.carl.config import DEFAULT_CONFIGS
from src.carl.state import WorkloadRegime


class _NullArm:
    """Per-regime stand-in for a bandit, so `.bandits[r]` type-name reporting
    and `.counts` diagnostics keep working without a special case upstream."""

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.counts = [0] * n_arms

    def select(self, context) -> int:      # noqa: ARG002 -- context is ignored
        return 0

    def update(self, arm: int, reward: float, context) -> None:
        """Accept and discard. RuleOnly does not learn -- that is the treatment."""


class RuleOnlyBandit:
    """Stateless `DEFAULT_CONFIGS[regime]` lookup, shaped like `PerRegimeBandit`.

    Always plays arm 0, which for every regime is that regime's hand-tuned
    default. Interface-compatible with `PerRegimeBandit` so `CARLController`
    drives it through the identical code path.
    """

    name = "RuleOnly"

    def __init__(self, arms_by_regime: dict, d: int | None = None, **_kw) -> None:
        """
        Args:
            arms_by_regime: {regime: [CARLConfig, ...]}, normally
                `all_arm_sets()`. Only arm 0 is ever played; the rest are kept
                so `arms()` still describes the action space the comparison was
                run against, and so an arm-index recorded in a trace means the
                same thing it means for CARL.
            d: accepted and ignored (RuleOnly consumes no context). Present so
                the constructor is substitutable for PerRegimeBandit's.
            **_kw: accepted and ignored (alpha, seed, bandit_cls, ...).
        """
        self.d = d
        self.arms_by_regime = {r: list(arms) for r, arms in arms_by_regime.items()}
        self.bandits = {r: _NullArm(len(arms))
                        for r, arms in self.arms_by_regime.items()}
        self._assert_arm0_is_the_default()

    def _assert_arm0_is_the_default(self) -> None:
        """Arm 0 must BE the regime default, or this is not the null baseline.

        Checked at construction rather than documented, because the whole
        interpretation of `CARL - RuleOnly` depends on it: if arm 0 drifted away
        from `DEFAULT_CONFIGS[regime]`, RuleOnly would silently become some
        other policy and the comparison would stop isolating learning.
        """
        for regime, arms in self.arms_by_regime.items():
            if not arms:
                raise ValueError(f"RuleOnlyBandit: {regime} has no arms")
            expected = DEFAULT_CONFIGS.get(regime)
            if expected is not None and arms[0] != expected:
                raise ValueError(
                    f"RuleOnlyBandit: arm 0 for {regime.value} is not "
                    f"DEFAULT_CONFIGS[{regime.value}]. RuleOnly is only the "
                    "null baseline if arm 0 is the hand-tuned default.")

    def reset(self) -> None:
        self.bandits = {r: _NullArm(len(arms))
                        for r, arms in self.arms_by_regime.items()}

    def arms(self, regime: WorkloadRegime) -> list:
        return self.arms_by_regime[regime]

    def select(self, regime: WorkloadRegime, context):
        """Always (0, DEFAULT_CONFIGS[regime]). Context is deliberately unused."""
        self.bandits[regime].counts[0] += 1
        return 0, self.arms_by_regime[regime][0]

    def update(self, regime: WorkloadRegime, arm: int, reward: float, context) -> None:
        """No-op by design: RuleOnly is scored by a reward it never consumes."""

    def selection_counts(self) -> dict:
        return {r.value: list(self.bandits[r].counts) for r in self.bandits}
