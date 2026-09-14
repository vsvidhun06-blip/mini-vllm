"""COLD START: missing latency must not earn reward and must not update a learner.

WHAT THESE TESTS PIN DOWN
=========================
The reward-v2 T4 artifact shows, in every seed, that the first three control
cycles logged `ttft_p99_ms = 0` and `tpot_p99_ms = 0` -- not measured zeros, but
the value `state._percentile` returns for an EMPTY window. Reward v2's
`latency_term` maps 0 ms to 1.0, so those cycles collected the full TTFT + TPOT
weight (0.5 of the maximum) for latencies nobody had observed, and LinUCB
consumed the result. The credit lands on the PREVIOUS cycle's arm, and the first
arm played in every regime is arm 0, so the bias has a direction.

The tests below are written against the two claims that repair has to make:

    1. missing TTFT/TPOT CANNOT receive maximum reward
    2. missing TTFT/TPOT CANNOT update the learner

and against the failure modes a lazy fix would introduce -- a zero fill, a
penalty fill, a silently-carried-forward reward, or a warm-up that never
actually engages. Every test states the specific wrong behaviour it forbids,
because a test that only asserts the current behaviour will happily keep passing
after that behaviour regresses into a different-but-equally-wrong one.
"""
from __future__ import annotations

import pytest

from src.carl.config import CARLConfig, all_arm_sets
from src.carl.controller import CARLController
from src.carl.live_reward import LiveRewardV2
from src.carl.reward import DEFAULT_WEIGHTS, RewardScales, utility_v2
from src.carl.state import FEATURE_DIM, MetricsTracker, RuntimeState

# Scales in the range the T4 actually operates in, so "maximum reward" below is
# the real ceiling and not an artefact of an unreachable normalisation.
SCALES = RewardScales(t_half=96.0, ttft_target=60000.0, tpot_target=300.0,
                      sharpness=2.0)

# The maximum reward_v2 can ever pay: infinite throughput, zero latency, a
# perfect cache. With DEFAULT_WEIGHTS that is 1.0, but it is computed rather
# than typed so a weight change cannot silently invalidate the comparison.
MAX_POSSIBLE = sum(DEFAULT_WEIGHTS.values())


def _cold_state() -> RuntimeState:
    """A state observed BEFORE any request completed: the artifact's cycle 0.

    Throughput is non-zero because `record_throughput` is fed every step with an
    emission, while TTFT/TPOT are fed only on COMPLETION -- which is exactly the
    asymmetry that produced the bug.
    """
    return RuntimeState(queue_depth=40, avg_prompt_len=48.0, throughput_tps=90.0,
                        p50_ttft_ms=0.0, p99_tpot_ms=0.0, active_requests=8)


def _warm_state() -> RuntimeState:
    return RuntimeState(queue_depth=40, avg_prompt_len=48.0, throughput_tps=90.0,
                        p50_ttft_ms=21000.0, p99_tpot_ms=275.0, active_requests=8)


def _cold_tracker() -> MetricsTracker:
    """Throughput observed, no completed request. The artifact's opening state."""
    t = MetricsTracker(window=200)
    t.record_throughput(90.0)
    t.record_batch(8)
    return t


def _warm_tracker() -> MetricsTracker:
    t = _cold_tracker()
    t.record_request(ttft_ms=21000.0, tpot_ms=275.0)
    return t


class _SpyBandit:
    """PerRegimeBandit-shaped, but records every update it is handed.

    Not a mock library double on purpose: the controller drives its policy
    through a duck-typed interface, and a hand-written stub that satisfies that
    interface exercises the SAME code path the real bandit takes.
    """

    def __init__(self) -> None:
        self.arms_by_regime = all_arm_sets()
        self.updates: list[tuple] = []
        self.selections: list = []
        self._next_arm = 0

    def arms(self, regime):
        return self.arms_by_regime[regime]

    def select(self, regime, context):
        arm = self._next_arm % len(self.arms_by_regime[regime])
        self.selections.append((regime, arm))
        return arm, self.arms_by_regime[regime][arm]

    def update(self, regime, arm, reward, context):
        self.updates.append((regime, arm, reward))

    def selection_counts(self):
        return {}


# ---------------------------------------------------------------------------
# CLAIM 1: missing TTFT/TPOT cannot receive maximum reward.
# ---------------------------------------------------------------------------


def test_the_bug_is_real_the_unfixed_reward_pays_almost_the_maximum():
    """The pre-repair behaviour, asserted so the repair has something to be a
    repair OF. Without this, a later reader cannot tell whether the fix below
    addresses a real defect or a hypothetical one."""
    unfixed = LiveRewardV2(SCALES, require_valid_metrics=False)
    reward, terms = unfixed(_cold_state(), _cold_tracker())

    assert terms["ttft_term"] == 1.0, "0 ms scored as anything but perfect?"
    assert terms["tpot_term"] == 1.0
    # Both latency terms fully paid: 0.3 + 0.2 of a 1.0 maximum, on no data.
    assert reward >= 0.5 * MAX_POSSIBLE
    assert reward > 0.9 * MAX_POSSIBLE * (
        (DEFAULT_WEIGHTS["ttft"] + DEFAULT_WEIGHTS["tpot"]) / MAX_POSSIBLE)


def test_missing_latency_cannot_receive_maximum_reward():
    """THE CENTRAL CLAIM. With the repair, an unmeasured metric yields NO reward.

    Asserted as `is None`, not as "a small number": any number at all would be a
    fabricated measurement, and a small one would still enter the learner's
    running mean.
    """
    reward, terms = LiveRewardV2(SCALES)(_cold_state(), _cold_tracker())

    assert reward is None
    assert terms["reward_valid"] is False
    assert set(terms["reward_validity"]["missing"]) == {"ttft_p99_ms", "tpot_p99_ms"}


def test_no_fill_value_is_substituted_in_either_direction():
    """Forbids BOTH lazy fixes: a zero/penalty fill and a neutral mid-point.

    A penalty fill is not the conservative choice it looks like -- it is the same
    error with the opposite sign, and it would teach the learner that whichever
    arm happened to be first is bad.
    """
    reward, _ = LiveRewardV2(SCALES)(_cold_state(), _cold_tracker())
    assert reward is None
    assert reward is not False        # `0.0`/`False` would both be fill values
    assert not isinstance(reward, (int, float))


def test_the_withheld_reward_is_still_auditable():
    """The refusal must be inspectable in the trace, not merely absent from it.

    `would_be_reward` is what the pre-repair code WOULD have paid. It is
    diagnostic output, is labelled as such, and is never the returned reward --
    which is what the first assertion pins.
    """
    reward, terms = LiveRewardV2(SCALES)(_cold_state(), _cold_tracker())
    assert reward is None
    assert terms["would_be_reward"] > 0.5 * MAX_POSSIBLE
    assert "NOT zero latency" in terms["invalid_reason"]
    assert terms["reward_validity"]["n_ttft_samples"] == 0
    assert terms["reward_validity"]["n_tpot_samples"] == 0


def test_reward_resumes_once_a_genuine_observation_exists():
    """The gate must OPEN. A permanently-withheld reward is not a repair, it is
    a disabled learner, and it would pass every test above."""
    fn = LiveRewardV2(SCALES)
    assert fn(_cold_state(), _cold_tracker())[0] is None

    reward, terms = fn(_warm_state(), _warm_tracker())
    assert reward is not None
    assert terms["reward_valid"] is True
    assert terms["reward_validity"]["all_valid"] is True
    # And the resumed reward is NOT the optimistic one: real latency scores
    # strictly below the perfect-latency ceiling the cold cycle was paying.
    assert terms["ttft_term"] < 1.0 and terms["tpot_term"] < 1.0


def test_a_zero_latency_sample_is_a_sentinel_not_a_measurement():
    """`_serve` records `tpot_ms = 0.0` for single-token requests, because
    inter-token time is undefined with fewer than two tokens. Counting that as an
    observation would reopen the bug through the back door: the window would be
    non-empty, the p99 would still be 0, and the reward would go back to paying
    for perfect latency."""
    t = _cold_tracker()
    t.record_request(ttft_ms=0.0, tpot_ms=0.0)

    v = t.metric_validity()
    assert v["n_ttft_observations_raw"] == 1, "the sample WAS recorded"
    assert v["n_ttft_samples"] == 0, "but it is not a measurement"
    assert v["n_tpot_samples"] == 0
    assert v["all_valid"] is False
    assert LiveRewardV2(SCALES)(_cold_state(), t)[0] is None


def test_min_latency_samples_must_be_at_least_one():
    with pytest.raises(ValueError, match="min_latency_samples"):
        LiveRewardV2(SCALES, min_latency_samples=0)


def test_a_higher_sample_threshold_lengthens_the_warm_up():
    """The threshold is a knob with an honest meaning, so it is checked. One
    observation satisfies the default and not a stricter setting."""
    t = _warm_tracker()
    assert LiveRewardV2(SCALES, min_latency_samples=1)(_warm_state(), t)[0] is not None
    assert LiveRewardV2(SCALES, min_latency_samples=5)(_warm_state(), t)[0] is None


# ---------------------------------------------------------------------------
# CLAIM 2: missing TTFT/TPOT cannot update the learner.
# ---------------------------------------------------------------------------


def _controller(bandit, tracker, reward_fn):
    return CARLController(scheduler=None, bandit=bandit, observe_interval=1,
                          metrics=tracker, reward_fn=reward_fn)


def test_the_learner_receives_nothing_while_the_metrics_are_unmeasured():
    """THE SECOND CENTRAL CLAIM, at the controller boundary.

    Three cold cycles -- the artifact's own warm-up length -- must produce ZERO
    calls to `bandit.update`. Not a call carrying 0.0; no call.
    """
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))

    for _ in range(3):
        c.step(state=_cold_state())

    assert spy.updates == []
    assert [e.reward for e in c.controller_log] == [None, None, None]
    assert [e.reward_valid for e in c.controller_log] == [False, False, False]
    assert all(e.rewarded_arm == -1 for e in c.controller_log)


def test_learning_begins_only_after_a_measured_window_exists():
    """Warm-up, then a real observation, then updates -- and the first update
    carries the MEASURED reward, not a replayed cold one."""
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))

    c.step(state=_cold_state())
    c.step(state=_cold_state())
    assert spy.updates == []

    tracker.record_request(ttft_ms=21000.0, tpot_ms=275.0)
    c.step(state=_warm_state())
    c.step(state=_warm_state())

    assert len(spy.updates) == 2
    expected = utility_v2({"throughput_tps": 90.0, "ttft_p99_ms": 21000.0,
                           "tpot_p99_ms": 275.0, "cache_hit_rate": 0.0},
                          DEFAULT_WEIGHTS, SCALES)
    # The measured reward must be STRICTLY BELOW what the cold cycle would have
    # paid. Comparing against the optimistic value rather than an absolute
    # threshold is the point: the claim is that the optimism is gone, and how
    # much reward a real operating point earns depends on the scales.
    optimistic = utility_v2({"throughput_tps": 90.0, "ttft_p99_ms": 0.0,
                             "tpot_p99_ms": 0.0, "cache_hit_rate": 0.0},
                            DEFAULT_WEIGHTS, SCALES)
    for _regime, _arm, reward in spy.updates:
        assert reward == pytest.approx(expected)
        assert reward < optimistic, "the cold-start optimism is gone"


def test_a_withheld_cycle_is_dropped_not_carried_forward():
    """The uncredited interval must NOT be paid to a later arm.

    A reward observed at cycle t scores the config that ran over [t-1, t].
    Deferring it to t+1 would attribute it to a DIFFERENT config, which is worse
    than dropping it. So after one warm-up cycle and two warm cycles there are
    exactly two updates -- not three with the first one back-filled.
    """
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))

    c.step(state=_cold_state())                       # cycle 0: withheld
    tracker.record_request(ttft_ms=21000.0, tpot_ms=275.0)
    c.step(state=_warm_state())                       # cycle 1: credits cycle 0's arm
    c.step(state=_warm_state())                       # cycle 2: credits cycle 1's arm

    assert len(spy.updates) == 2
    assert c.controller_log[0].reward is None


def test_mean_reward_is_over_scored_cycles_only():
    """A withheld cycle must not enter the mean. Averaging it in would need a
    fill value -- the thing being repaired -- and would drag the reported mean
    toward whatever that value was."""
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))

    c.step(state=_cold_state())
    tracker.record_request(ttft_ms=21000.0, tpot_ms=275.0)
    c.step(state=_warm_state())

    means = c.stats()["mean_reward_per_regime"]
    assert means, "the scored cycle should still be averaged"
    scored = [e.reward for e in c.controller_log if e.reward_valid]
    assert all(v == pytest.approx(sum(scored) / len(scored)) for v in means.values())


# ---------------------------------------------------------------------------
# The warm-up record: it has to be reportable per seed.
# ---------------------------------------------------------------------------


def test_the_warm_up_is_recorded_not_left_to_be_re_derived():
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))

    for _ in range(3):
        c.step(state=_cold_state())
    tracker.record_request(ttft_ms=21000.0, tpot_ms=275.0)
    for _ in range(4):
        c.step(state=_warm_state())

    r = c.reward_validity_report()
    assert r["cycles"] == 7
    assert r["warmup_cycles"] == 3
    assert r["valid_reward_cycles"] == 4
    assert r["withheld_reward_cycles"] == 3
    assert r["first_valid_reward_cycle_index"] == 3
    assert r["valid_reward_updates"] == 4
    assert r["withheld_after_first_valid"] == 0, "validity must be monotone"
    assert r["warmup_duration_s"] is not None and r["warmup_duration_s"] >= 0.0


def test_valid_reward_updates_is_smaller_than_the_cycle_count():
    """The honest denominator for a learning claim. It is strictly smaller than
    the number of control cycles -- by the warm-up plus the one first cycle that
    has no previous arm to credit -- and reporting the cycle count in its place
    would overstate what the learner saw."""
    spy, tracker = _SpyBandit(), _warm_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))
    for _ in range(5):
        c.step(state=_warm_state())

    r = c.reward_validity_report()
    assert r["cycles"] == 5
    assert r["warmup_cycles"] == 0
    assert r["valid_reward_updates"] == 4 == len(spy.updates)


def test_the_report_survives_a_reset():
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = _controller(spy, tracker, LiveRewardV2(SCALES))
    c.step(state=_cold_state())
    c.reset()

    r = c.reward_validity_report()
    assert r["cycles"] == 0 and r["warmup_cycles"] == 0
    assert r["valid_reward_updates"] == 0
    assert r["first_valid_reward_cycle_index"] is None


# ---------------------------------------------------------------------------
# Nothing that already worked may change.
# ---------------------------------------------------------------------------


def test_reward_v1_is_untouched_and_never_withholds():
    """Reward v1 has no notion of validity and must keep producing a number on
    every cycle: the v1 hardware run has to stay reproducible."""
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = CARLController(scheduler=None, bandit=spy, observe_interval=1,
                       metrics=tracker)          # reward_fn=None => v1

    for _ in range(3):
        c.step(state=_cold_state())

    assert all(isinstance(e.reward, float) for e in c.controller_log)
    assert all(e.reward_valid for e in c.controller_log)
    assert len(spy.updates) == 2


def test_a_tracker_without_validity_support_is_treated_as_valid():
    """Unit tests and stubs drive the controller with synthetic states and no
    real tracker. There is no cold start to protect them from, so they must keep
    scoring -- and the record has to SAY that is why, rather than reporting an
    unexplained `all_valid`."""
    class _OldTracker:
        def p99_ttft_ms(self):
            return 1500.0

    reward, terms = LiveRewardV2(SCALES)(_warm_state(), _OldTracker())
    assert reward is not None
    assert terms["reward_valid"] is True
    assert "no tracker" in terms["reward_validity"]["source"]


def test_metric_validity_does_not_disturb_the_context_vector():
    """FEATURE_DIM pins the bandit's input dimension. Changing it would break
    every controlled pair against an earlier run, so the additive validity state
    must not have leaked into the feature set."""
    assert FEATURE_DIM == 10
    assert len(_warm_state().to_feature_vector()) == FEATURE_DIM
    assert "metric_validity" not in RuntimeState.feature_names()


def test_percentile_still_returns_zero_on_an_empty_window():
    """The 0.0 was never the bug -- consuming it as a MEASUREMENT was. It stays,
    because the context vector depends on being NaN-free, and this test says so
    explicitly so a later reader does not 'fix' it there instead."""
    t = MetricsTracker(window=10)
    assert t.p99_ttft_ms() == 0.0
    assert t.p99_tpot_ms() == 0.0
    assert t.metric_validity()["all_valid"] is False


def test_config_default_is_unchanged():
    """A sanity pin: none of this touched the configuration surface."""
    assert CARLConfig().max_batch_size == 8
    assert CARLConfig().chunk_size == 256


def test_the_gate_covers_the_optimistically_scored_metrics_and_says_which():
    """Scope, stated as a test rather than left implicit.

    A missing latency percentile scores 1.0 -- perfect -- and inflates the
    credited arm. A missing throughput scores 0.0 and understates. Only the
    first class can manufacture a preference, so only it gates; the throughput
    sample count is still REPORTED so the cold start is fully visible.
    """
    v = _cold_tracker().metric_validity()
    assert v["gated_metrics"] == ["tpot_p99_ms", "ttft_p99_ms"]
    assert "throughput_tps" not in v["valid"]
    assert v["n_throughput_samples"] == 1, "reported even though it does not gate"
    assert "cannot" in v["gate_rationale"]


def test_an_empty_throughput_window_alone_does_not_withhold_the_reward():
    """Gating on a channel the reward does not read from would withhold on every
    caller that drives the controller with a synthetic state, for no gain."""
    t = MetricsTracker(window=200)
    t.record_request(ttft_ms=21000.0, tpot_ms=275.0)     # latency only, no tps
    assert t.n_throughput_samples() == 0
    reward, terms = LiveRewardV2(SCALES)(_warm_state(), t)
    assert reward is not None
    assert terms["reward_validity"]["n_throughput_samples"] == 0
