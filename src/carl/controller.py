"""
CARLController -- the coordinated adaptive control loop.

This is where the pieces compose into the contribution. Every `observe_interval`
scheduler steps the controller runs one cycle:

    observe -> classify regime -> select a joint config -> apply it live
            -> reward the PREVIOUS config -> log.

The two subtleties that make this correct rather than just plausible:

  1. DELAYED REWARD. A config's effect on latency/throughput only shows up AFTER
     it has been running for an interval. So the reward computed at tick t (from
     the freshly observed state) measures the config that was applied at tick
     t-1, and is credited to THAT arm -- not to the arm we're about to select.
     This is the standard "act, then observe the consequence" bandit timing.

  2. THREAD-SAFE LIVE RECONFIGURATION. The engine runs on a pumper thread that
     reads scheduler.max_batch_size / chunk_size / etc. every step. The
     controller mutates those same attributes from whatever thread drives it.
     The whole step() -- selection, application, and bandit update -- runs under
     one lock so a half-applied config is never observed and the bandit's numpy
     state never races.

INTEGRATION
-----------
The controller holds REFERENCES to the live components and mutates their
attributes in place (exactly as the AutoTuner does). It never reaches into a
component's internals beyond the documented knobs, and every write is defensive
(skipped if the component or attribute is absent), so wiring CARL into an engine
that lacks a given knob is a no-op for that knob rather than a crash.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.carl.bandit import DEFAULT_UTILITY_WEIGHTS, utility
from src.carl.config import CARLConfig
from src.carl.state import MetricsTracker, RuntimeState, WorkloadRegime, classify_regime


# ---------------------------------------------------------------------------
# SLO targets feed the reward's violation terms.
# ---------------------------------------------------------------------------


@dataclass
class SLO:
    """Service-level objectives the reward scores against.

    ttft_ms / tpot_ms are the deadlines a request must meet; throughput_ref is
    the rate at which throughput_norm saturates to 1.0 (so the reward doesn't
    reward unbounded batching past the point of usefulness).
    """
    ttft_ms: float = 100.0
    tpot_ms: float = 50.0
    throughput_ref: float = 50.0


# ---------------------------------------------------------------------------
# Controller log entry.
# ---------------------------------------------------------------------------


@dataclass
class ControllerLogEntry:
    """One row of the controller's decision history."""
    step: int
    regime: WorkloadRegime
    config: CARLConfig
    reward: float
    state_features: list

    def as_dict(self) -> dict:
        """JSON-friendly view for /carl/log."""
        return {
            "step": self.step,
            "regime": self.regime.value,
            "config": self.config.as_dict(),
            "reward": self.reward,
            "state_features": self.state_features,
        }


# ---------------------------------------------------------------------------
# CARLController
# ---------------------------------------------------------------------------


class CARLController:
    def __init__(
        self,
        scheduler=None,
        spec_decoder=None,
        router=None,
        kv_cache=None,
        bandit=None,
        observe_interval: int = 50,
        slo: SLO | None = None,
        weights: dict | None = None,
        metrics: MetricsTracker | None = None,
    ) -> None:
        """
        Args:
            scheduler, spec_decoder, router, kv_cache: the live engine
                components whose knobs CARL drives. Any may be None (the
                corresponding knobs are then no-ops).
            bandit: a PerRegimeBandit (LinUCB by default; Thompson for the
                ablation). Required for adaptive operation; without it the
                controller can still observe and apply manual overrides.
            observe_interval: how many scheduler steps between control cycles.
                50 matches the AutoTuner's cadence -- enough steps for the
                metric windows to reflect the current config.
            slo: the SLO targets the reward scores against.
            weights: utility-term weights (defaults to DEFAULT_UTILITY_WEIGHTS).
            metrics: the MetricsTracker holding rolling latency/throughput
                windows. Created fresh if not supplied; the engine/benchmark
                feeds it samples.
        """
        self.scheduler = scheduler
        self.spec_decoder = spec_decoder
        self.router = router
        self.kv_cache = kv_cache
        self.bandit = bandit
        self.observe_interval = observe_interval
        self.slo = slo or SLO()
        self.weights = weights or DEFAULT_UTILITY_WEIGHTS
        self.metrics = metrics if metrics is not None else MetricsTracker()

        self._lock = threading.Lock()

        # Decision history + aggregates for stats().
        self.controller_log: list[ControllerLogEntry] = []
        self._regime_counts: dict[WorkloadRegime, int] = {}
        self._config_counts: dict[str, int] = {}
        self._reward_sum: dict[WorkloadRegime, float] = {}
        self._reward_n: dict[WorkloadRegime, int] = {}
        self._total_adaptations = 0

        # Previous cycle's (regime, arm, config, context) for delayed reward.
        self._prev: tuple | None = None
        # The config most recently pushed into the engine (for adaptation count).
        self._applied_config: CARLConfig | None = None
        # Sticky manual override (POST /carl/config). When set, step() applies it
        # instead of consulting the bandit -- used for ablations.
        self._override: CARLConfig | None = None

    # ---- observation -----------------------------------------------------

    def observe(self) -> RuntimeState:
        """Snapshot the live engine into a RuntimeState (see state.observe)."""
        return RuntimeState.observe(
            scheduler=self.scheduler,
            spec_decoder=self.spec_decoder,
            router=self.router,
            kv_cache=self.kv_cache,
            metrics=self.metrics,
        )

    def _reward_for_state(self, state: RuntimeState) -> float:
        """Map an observed state + SLOs into the scalar utility reward."""
        throughput_norm = min(1.0, state.throughput_tps / self.slo.throughput_ref) \
            if self.slo.throughput_ref > 0 else 0.0
        metrics = {
            "throughput_norm": throughput_norm,
            "ttft_violation_rate": self.metrics.ttft_violation_rate(self.slo.ttft_ms),
            "tpot_violation_rate": self.metrics.tpot_violation_rate(self.slo.tpot_ms),
            "cache_hit_rate": state.cache_hit_rate,
        }
        return utility(metrics, self.weights)

    # ---- the control cycle ----------------------------------------------

    def maybe_step(self, scheduler_step_idx: int) -> ControllerLogEntry | None:
        """Run a control cycle only on observe_interval boundaries.

        This is the hook the server's pumper loop calls every scheduler step;
        it self-gates so the pumper doesn't have to track cadence.
        """
        if self.observe_interval <= 0 or scheduler_step_idx % self.observe_interval != 0:
            return None
        return self.step(step_idx=scheduler_step_idx)

    def step(self, step_idx: int | None = None, state: RuntimeState | None = None):
        """One full control cycle. Returns the ControllerLogEntry recorded.

        Args:
            step_idx: the scheduler step this cycle corresponds to (for the log).
            state: an explicit RuntimeState to act on instead of observing the
                live engine. Used by tests/benchmarks to drive deterministic
                workloads; in production this is None and we observe().
        """
        with self._lock:
            if state is None:
                state = self.observe()
            regime = classify_regime(state)
            context = state.to_feature_vector()

            # 1. Choose the config for THIS cycle (bandit, or sticky override).
            if self._override is not None:
                config = self._override
                arm = -1  # sentinel: override configs aren't bandit arms
            elif self.bandit is not None:
                arm, config = self.bandit.select(regime, context)
            else:
                # No bandit and no override: hold the last applied config (or a
                # default). Keeps the controller usable as a pure observer.
                config = self._applied_config or CARLConfig()
                arm = -1

            # 2. Apply it live.
            self._apply(config)

            # 3. Reward the PREVIOUS config from the state we just observed
            #    (delayed-reward timing), and credit it to the previous arm.
            reward = self._reward_for_state(state)
            if self._prev is not None and self.bandit is not None:
                prev_regime, prev_arm, _prev_config, prev_context = self._prev
                if prev_arm >= 0:   # don't update on override-applied steps
                    self.bandit.update(prev_regime, prev_arm, reward, prev_context)

            # 4. Log + aggregate.
            entry = ControllerLogEntry(
                step=step_idx if step_idx is not None else len(self.controller_log),
                regime=regime,
                config=config,
                reward=reward,
                state_features=context,
            )
            self.controller_log.append(entry)
            self._regime_counts[regime] = self._regime_counts.get(regime, 0) + 1
            ckey = self._config_key(regime, config)
            self._config_counts[ckey] = self._config_counts.get(ckey, 0) + 1
            self._reward_sum[regime] = self._reward_sum.get(regime, 0.0) + reward
            self._reward_n[regime] = self._reward_n.get(regime, 0) + 1

            # 5. Remember this cycle for the next reward, and stash the applied
            #    config (only bandit/override-selected arms count as adaptations).
            self._prev = (regime, arm, config, context)
            return entry

    # ---- live reconfiguration -------------------------------------------

    def _apply(self, config: CARLConfig) -> None:
        """Push a CARLConfig into the live components. Assumes the lock is held.

        Every write is via _set (skipped if the component/attribute is absent),
        so an engine missing a knob simply doesn't receive that one. spec_k maps
        to BOTH the spec decoder's k and the scheduler's enable/k pair, because
        the scheduler gates speculation on `enable_spec_decode` and reads
        `spec_decode_k` for the draft length.
        """
        config = config.clamp()

        # Scheduler.
        _set(self.scheduler, "max_batch_size", config.max_batch_size)
        _set(self.scheduler, "chunk_size", config.chunk_size)
        _set(self.scheduler, "preemption_enabled", config.preemption_enabled)
        _set(self.scheduler, "use_cuda_graphs", config.use_cuda_graphs)
        # Speculation: scheduler gate + draft length, and the standalone decoder.
        _set(self.scheduler, "spec_decode_k", max(1, config.spec_k))
        _set(self.scheduler, "enable_spec_decode", config.spec_k > 0)
        _set(self.spec_decoder, "k", config.spec_k)

        # Router.
        _set(self.router, "routing_threshold", config.routing_threshold)
        _set(self.router, "cache_affinity_weight", config.cache_affinity_weight)

        # KV cache (H2O eviction knobs). recent_window also lives on the score
        # tracker, so push it there too when present.
        _set(self.kv_cache, "evict_threshold", config.eviction_threshold)
        _set(self.kv_cache, "recent_window", config.eviction_window)
        tracker = getattr(self.kv_cache, "score_tracker", None)
        _set(tracker, "recent_window", config.eviction_window)

        # Adaptation accounting: a change in the applied config is one adaptation.
        if self._applied_config is None or config != self._applied_config:
            self._total_adaptations += 1
        self._applied_config = config

    def apply_override(self, config: CARLConfig) -> None:
        """Set a sticky manual config (POST /carl/config). Applied immediately
        and on every subsequent step until reset()."""
        with self._lock:
            self._override = config.clamp()
            self._apply(self._override)

    def clear_override(self) -> None:
        with self._lock:
            self._override = None

    def reset(self) -> None:
        """Reset bandit statistics and controller history (POST /carl/reset)."""
        with self._lock:
            if self.bandit is not None and hasattr(self.bandit, "reset"):
                self.bandit.reset()
            self.controller_log.clear()
            self._regime_counts.clear()
            self._config_counts.clear()
            self._reward_sum.clear()
            self._reward_n.clear()
            self._total_adaptations = 0
            self._prev = None
            self._override = None
            self._applied_config = None

    # ---- stats -----------------------------------------------------------

    @staticmethod
    def _config_key(regime: WorkloadRegime, config: CARLConfig) -> str:
        """A compact, hashable signature for the config-distribution histogram."""
        c = config
        return (
            f"{regime.value}|mb{c.max_batch_size}|cs{c.chunk_size}|k{c.spec_k}|"
            f"ev{c.eviction_threshold}|caw{c.cache_affinity_weight}"
        )

    def stats(self) -> dict:
        """Aggregate view of the controller's behaviour.

        Returns:
            regime_distribution    {regime_value: cycles spent in it}
            config_distribution    {config_signature: times applied}
            mean_reward_per_regime {regime_value: mean reward observed}
            total_adaptations      number of times the applied config changed
            best_config_per_regime {regime_value: the bandit's most-selected
                                    arm's config} -- the policy CARL converged on
        """
        regime_distribution = {r.value: n for r, n in self._regime_counts.items()}
        mean_reward = {
            r.value: (self._reward_sum[r] / self._reward_n[r])
            for r in self._reward_sum if self._reward_n.get(r)
        }
        return {
            "regime_distribution": regime_distribution,
            "config_distribution": dict(self._config_counts),
            "mean_reward_per_regime": mean_reward,
            "total_adaptations": self._total_adaptations,
            "best_config_per_regime": self._best_config_per_regime(),
        }

    def _best_config_per_regime(self) -> dict:
        """Per regime, the config of the arm the bandit has selected most.

        This is the bandit's revealed preference -- the operating point it has
        converged on for each regime after learning. Falls back to the regime's
        default (arm 0) when a regime was never visited.
        """
        out: dict = {}
        bandit = self.bandit
        if bandit is None or not hasattr(bandit, "selection_counts"):
            return out
        counts = bandit.selection_counts()           # {regime_value: [per-arm]}
        for regime, arm_counts in counts.items():
            if not arm_counts or sum(arm_counts) == 0:
                continue
            best_arm = max(range(len(arm_counts)), key=lambda a: arm_counts[a])
            # Map regime_value back to the enum to index the arm set.
            regime_enum = next((r for r in WorkloadRegime if r.value == regime), None)
            if regime_enum is None:
                continue
            arms = bandit.arms(regime_enum)
            out[regime] = arms[best_arm].as_dict()
        return out


# ---------------------------------------------------------------------------
# Defensive setter (module-level so _apply stays readable).
# ---------------------------------------------------------------------------


def _set(obj, attr: str, value) -> None:
    """setattr(obj, attr, value) iff obj is not None and already has that attr.

    Requiring the attribute to pre-exist means CARL only drives knobs a
    component actually declares -- it never silently grows a phantom attribute on
    an object that wouldn't read it. (The router gains routing_threshold /
    cache_affinity_weight as explicit live knobs precisely so this write lands.)
    """
    if obj is None:
        return
    if hasattr(obj, attr):
        setattr(obj, attr, value)
