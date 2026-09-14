"""Evaluation harness v2 -- drives controllers over the mechanistic engine model.

Replaces scripts/eval/_harness.py for the repair pass. The differences that
matter, all of them forced by findings in docs/eval/CARL_REPAIR_STATUS.md:

  * SUBSTRATE. src/eval/engine_model, whose throughput/latency emerge from a
    step-time equation, instead of benchmark_carl.WorkloadModel, whose optimum
    is DEFAULT_CONFIGS by construction (R4).
  * ORACLE. Computed by brute-force search over the arm set, evaluated BY the
    model, under the SAME objective the controller optimises. It is not a table
    lookup and it moves if the hardware profile changes (R4).
  * STATIC BASELINES. Two of them, independently searched over a wide LHS grid
    that is NOT the arm set and NOT DEFAULT_CONFIGS: one selected on throughput
    (as the paper did) and one on the full multi-objective utility (Phase 8/11),
    so a latency comparison is not made against a throughput-tuned baseline.
  * REWARD. utility_v2 on raw metrics, checked for degeneracy every run (R5).
  * ARRIVALS. Poisson / deterministic / burst, per WorkloadSpec (Phase 10).

A "cycle" serves one slice of the request stream under a fixed configuration.
The controller observes the realised state of cycle t-1 and chooses the config
for cycle t -- the same act-then-observe timing the live controller uses.
"""
from __future__ import annotations

import os
import random
import statistics
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.carl.bandit import (  # noqa: E402
    LinUCBBandit, PerRegimeBandit, RepairedLinUCBBandit,
)
from src.carl.config import CARLConfig, DEFAULT_CONFIGS, all_arm_sets  # noqa: E402
from src.carl.reward import (  # noqa: E402
    DEFAULT_WEIGHTS, RewardScales, check_non_degenerate, term_breakdown, utility_v2,
)
from src.carl.state import (  # noqa: E402
    FEATURE_DIM, RuntimeState, WorkloadRegime, classify_regime,
)
from src.eval.engine_model import (  # noqa: E402
    HardwareProfile, Request, RunResult, WorkloadSpec, simulate,
)

R = WorkloadRegime

# ---------------------------------------------------------------------------
# Workload library.
# ---------------------------------------------------------------------------
#
# Rates are chosen so the engine is loaded but not permanently saturated, using
# the mechanistic model's own capacity (~76-97 tok/s depending on batch). A
# workload pinned in saturation makes every configuration look the same, which
# is a different way of producing the degeneracy this pass exists to remove.
# ---------------------------------------------------------------------------

WORKLOADS: dict[str, WorkloadSpec] = {
    # -- Stationary single-regime workloads, LIGHT load (rho well below 1) ----
    # Measured property of this substrate: below saturation the mean batch is
    # ~1.8 regardless of max_batch_size, so the knob does not bind and NO
    # controller can differentiate itself. Included precisely so that fact is
    # visible in the results rather than assumed away.
    "interactive": WorkloadSpec("interactive", 60, 32, 8, 32, 8,
                                arrival="poisson", rate_rps=2.0),
    "batch": WorkloadSpec("batch", 60, 220, 40, 64, 16,
                          arrival="poisson", rate_rps=1.2),
    "long_context": WorkloadSpec("long_context", 40, 900, 150, 48, 12,
                                 arrival="poisson", rate_rps=0.7),

    # -- The same regimes under HEAVY load ------------------------------------
    # These exist because a load sweep showed the reward-optimal max_batch_size
    # is driven by LOAD, not by prompt length: light load wants ~16, heavy load
    # wants ~2 (batching raises TPOT faster than it raises throughput once the
    # queue is deep). A controller that can track that is doing real work, so
    # these give adaptation its best available case.
    "interactive_heavy": WorkloadSpec("interactive_heavy", 60, 32, 8, 32, 8,
                                      arrival="poisson", rate_rps=12.0),
    "batch_heavy": WorkloadSpec("batch_heavy", 60, 220, 40, 64, 16,
                                arrival="poisson", rate_rps=4.0),
    "long_context_heavy": WorkloadSpec("long_context_heavy", 40, 900, 150, 48, 12,
                                       arrival="poisson", rate_rps=2.5),

    # -- Stress workloads (retained deliberately, clearly labelled) -----------
    # burst_dump reproduces the LEGACY live.py behaviour: every request arrives
    # at t=0. It is kept as a STRESS case, not as a serving workload, so the
    # difference between the two is measurable rather than rhetorical.
    "burst_dump": WorkloadSpec("burst_dump", 50, 120, 80, 48, 16,
                               arrival="burst"),
    "overload": WorkloadSpec("overload", 60, 220, 40, 64, 16,
                             arrival="poisson", rate_rps=6.0),
}


def nonstationary(phases: list[str], n_per_phase: int = 40) -> list[WorkloadSpec]:
    """A sequence of workload phases spliced end to end, with real arrival times.

    Each phase inherits its shape from WORKLOADS but is re-timed so arrivals are
    continuous across the boundary -- the regime flips with no notice, which is
    the condition CARL is supposed to exploit.
    """
    out: list[WorkloadSpec] = []
    for name in phases:
        base = WORKLOADS[name]
        out.append(WorkloadSpec(
            name=name, n_requests=n_per_phase,
            prompt_mean=base.prompt_mean, prompt_std=base.prompt_std,
            output_mean=base.output_mean, output_std=base.output_std,
            arrival=base.arrival, rate_rps=base.rate_rps,
        ))
    return out


# ---------------------------------------------------------------------------
# Config search space for the STATIC baselines.
# ---------------------------------------------------------------------------
#
# Deliberately NOT the bandit's arm set and NOT DEFAULT_CONFIGS. A static
# baseline that can only choose from the adaptive method's own arms is
# handicapped by construction; this grid is strictly larger and includes points
# CARL cannot reach.
# ---------------------------------------------------------------------------

STATIC_GRID_AXES = {
    "max_batch_size": [2, 4, 8, 12, 16, 24, 32],
    "chunk_size": [64, 128, 256, 384, 512],
    "eviction_threshold": [0.6, 0.75, 0.9],
    "preemption_enabled": [True, False],
}


def static_candidates(limit: int | None = None, seed: int = 0) -> list[CARLConfig]:
    """Latin-hypercube sample of STATIC_GRID_AXES (or the full grid if small)."""
    import itertools
    full = [
        CARLConfig(max_batch_size=mb, chunk_size=cs,
                   eviction_threshold=ev, preemption_enabled=pe).clamp()
        for mb, cs, ev, pe in itertools.product(
            STATIC_GRID_AXES["max_batch_size"], STATIC_GRID_AXES["chunk_size"],
            STATIC_GRID_AXES["eviction_threshold"],
            STATIC_GRID_AXES["preemption_enabled"])
    ]
    if limit is None or limit >= len(full):
        return full
    rng = random.Random(seed)
    return rng.sample(full, limit)


# ---------------------------------------------------------------------------
# Objectives. Passing these explicitly is what makes "Static-Best-throughput"
# and "Static-Best-SLO" genuinely different baselines (Phase 8/11).
# ---------------------------------------------------------------------------


def objective_throughput(r: RunResult) -> float:
    return r.throughput_tps


def make_objective_utility(scales: RewardScales, weights: dict | None = None):
    """The SAME multi-objective utility the controller maximises."""
    def _obj(r: RunResult) -> float:
        return utility_v2(run_to_metrics(r), weights, scales)
    return _obj


def run_to_metrics(r: RunResult) -> dict:
    """RunResult -> the raw-metric dict reward v2 consumes."""
    return {
        "throughput_tps": r.throughput_tps,
        "ttft_p99_ms": r.ttft_p99_ms,
        "tpot_p99_ms": r.tpot_p99_ms,
        "cache_hit_rate": 0.0,   # no prefix reuse in these synthetic streams
    }


def run_to_state(r: RunResult, queue_depth: int, avg_prompt_len: float) -> RuntimeState:
    """RunResult -> the RuntimeState the controller classifies and conditions on.

    Only OBSERVABLE quantities are passed: the controller never sees the phase
    label. queue_depth and avg_prompt_len come from the scheduler's own view, as
    they do in live.py.
    """
    return RuntimeState(
        queue_depth=int(queue_depth),
        avg_prompt_len=float(avg_prompt_len),
        gpu_utilization=min(1.0, r.mean_occupancy),
        cache_hit_rate=0.0,
        spec_acceptance_rate=0.0,
        p50_ttft_ms=r.ttft_p50_ms,
        p99_tpot_ms=r.tpot_p99_ms,
        throughput_tps=r.throughput_tps,
        active_requests=int(round(r.mean_batch)),
        batch_size_mean=r.mean_batch,
    )


# ---------------------------------------------------------------------------
# Agents.
# ---------------------------------------------------------------------------


class RuleOnlyController:
    """PHASE 5 null hypothesis: classify, look up, done. No learning, no state.

    This is the baseline the entire paper hinges on. If CARL cannot beat it, the
    contribution is the classifier, not the bandit.
    """

    name = "RuleOnly"

    def __init__(self) -> None:
        self.adaptations = 0
        self.arm_changes = 0
        self._last = None
        self.trace: list[dict] = []

    def choose(self, state: RuntimeState) -> CARLConfig:
        regime = classify_regime(state)
        cfg = DEFAULT_CONFIGS[regime]
        if self._last is not None and cfg != self._last:
            self.adaptations += 1
            self.arm_changes += 1
        self._last = cfg
        self.trace.append({"regime": regime.value, "arm": 0})
        return cfg

    def observe(self, reward: float) -> None:
        pass


def wide_arm_sets(per_regime: int = 8, seed: int = 0) -> dict:
    """A wider arm set covering the same space the STATIC baselines search.

    WHY THIS EXISTS. CARL's shipped arm sets are built by perturbing
    DEFAULT_CONFIGS along one or two knobs (`config.config_arms`). For
    INTERACTIVE that yields max_batch_size in {2, 4, 8} -- and on the mechanistic
    substrate the optimum for a light interactive workload is 16. **CARL cannot
    reach the answer.** A loss under those conditions says nothing about whether
    learning works; it says the action space excludes the target.

    This variant gives every regime the same broad arm set, drawn from the static
    search grid, so that "CARL loses" and "CARL's arm set is too narrow" become
    separately testable. Reported alongside the shipped arm set, never instead
    of it.
    """
    grid = static_candidates(limit=None)
    rng = random.Random(seed)
    # Deterministic spread over max_batch_size so the set is not accidentally
    # concentrated: take a stratified sample across the batch-size axis.
    by_mb: dict[int, list] = {}
    for c in grid:
        by_mb.setdefault(c.max_batch_size, []).append(c)
    arms: list[CARLConfig] = []
    for mb in sorted(by_mb):
        arms.append(rng.choice(by_mb[mb]))
    while len(arms) < per_regime:
        arms.append(rng.choice(grid))
    arms = arms[:per_regime]
    return {regime: list(arms) for regime in WorkloadRegime}


def arm_set_coverage(target: CARLConfig, arms_by_regime: dict) -> dict:
    """Can the bandit reach `target`, PER REGIME?

    Per-regime matters and pooling hides the defect. The bandit may only choose
    among the arms of the regime the classifier currently reports, so an arm
    available under BATCH is unreachable while the classifier says INTERACTIVE.
    Concretely, `config.config_arms` gives INTERACTIVE max_batch_size in
    {2, 4, 8} -- so if a light interactive workload's optimum is 16, CARL cannot
    express it no matter how well it learns, and a loss says nothing about
    learning.
    """
    per_regime = {}
    for regime, arms in arms_by_regime.items():
        mbs = sorted({a.max_batch_size for a in arms})
        per_regime[regime.value] = {
            "reachable_max_batch_sizes": mbs,
            "exact_match_available": any(a == target for a in arms),
            "target_max_batch_reachable": target.max_batch_size in mbs,
        }
    pooled_mbs = sorted({a.max_batch_size for arms in arms_by_regime.values()
                         for a in arms})
    return {
        "target": target.as_dict(),
        "target_max_batch_size": target.max_batch_size,
        "per_regime": per_regime,
        "pooled_reachable_max_batch_sizes": pooled_mbs,
        "reachable_in_every_regime": all(
            v["target_max_batch_reachable"] for v in per_regime.values()),
        "reachable_in_any_regime": any(
            v["target_max_batch_reachable"] for v in per_regime.values()),
    }


class BanditController:
    """CARL: per-regime contextual bandit over the arm sets.

    REWARD TIMING. This harness serves a whole slice under the chosen config and
    then measures it, so the reward observed after `choose()` is *caused by the
    config just chosen*. Attribution is therefore immediate, and crediting the
    PREVIOUS arm (as the live controller must, because it reads a rolling metric
    window that lags the configuration) would be a genuine mis-attribution here.

    That difference is deliberate and worth stating: the live path's delayed
    credit is a concession to its measurement apparatus, not a property of the
    problem. Where the substrate permits clean attribution, we use it.
    """

    def __init__(self, bandit_cls=LinUCBBandit, alpha: float = 0.5,
                 name: str = "CARL", arms: dict | None = None) -> None:
        self.name = name
        self.bandit = PerRegimeBandit(arms if arms is not None else all_arm_sets(),
                                      d=FEATURE_DIM,
                                      bandit_cls=bandit_cls, alpha=alpha)
        self.adaptations = 0
        self.arm_changes = 0
        self._last_cfg = None
        self._last_arm = None
        self._pending: tuple | None = None
        self.trace: list[dict] = []

    def choose(self, state: RuntimeState) -> CARLConfig:
        regime = classify_regime(state)
        context = state.to_feature_vector()
        arm, cfg = self.bandit.select(regime, context)
        if self._last_cfg is not None and cfg != self._last_cfg:
            self.adaptations += 1
        if self._last_arm is not None and self._last_arm != (regime, arm):
            self.arm_changes += 1
        self._last_cfg, self._last_arm = cfg, (regime, arm)
        self._pending = (regime, arm, context)
        self.trace.append({"regime": regime.value, "arm": arm})
        return cfg

    def observe(self, reward: float) -> None:
        """Credit the reward to the arm that produced it."""
        if self._pending is not None:
            regime, arm, context = self._pending
            self.bandit.update(regime, arm, reward, context)
            self._pending = None


class ClosedLoopAutoTunerController:
    """PHASE 9: the bottleneck-reactive baseline, with its feedback wire reconnected.

    THE DEFECT BEING FIXED. `benchmark_carl.AutoTunerAgent.choose` does:

        self.profiler.window.clear()
        phases = self._BOTTLENECK[true_regime]     # <- keyed ONLY on the regime
        for _ in range(5):
            self.profiler.record_step(**phases)    # <- synthetic, constant
        self.tuner.observe(self.sched, ...)

    The profiler is refilled every round with a fixed profile that depends only
    on the TRUE regime label and never on the configuration the tuner selected.
    The tuner therefore cannot perceive the consequences of its own actions: it
    is an open loop. Naturally it thrashes, and "a reactive autotuner does worse
    than static" -- a headline claim of the paper's introduction -- is a
    description of that bug, not a finding about reactive tuning.

    THE REPAIR. Feed the profiler the REAL phase times measured from the run the
    tuner's own configuration just produced (`RunResult.prefill_s / decode_s /
    host_s`). Now a config that shifts the bottleneck changes what the tuner
    sees next round, which is what closing the loop means.

    `tests/test_eval/test_autotuner_feedback.py` asserts the loop is closed: two
    different configurations must produce two different observed bottleneck
    profiles.
    """

    name = "AutoTuner-ClosedLoop"

    def __init__(self, tune_interval: int = 1, cooldown: int = 3) -> None:
        from types import SimpleNamespace

        from src.engine.auto_tuner import AutoTuner
        from src.engine.profiler import StepProfiler

        self.sched = SimpleNamespace(
            chunk_size=256, max_batch_size=8, use_cuda_graphs=True,
            evict_threshold=0.8,
        )
        self.profiler = StepProfiler(window=20)
        self.tuner = AutoTuner(self.profiler, tune_interval=tune_interval,
                               cooldown=cooldown)
        self._round = 0
        self.adaptations = 0
        self.arm_changes = 0
        self._last = None
        self.trace: list[dict] = []
        self.observed_profiles: list[dict] = []

    def feed(self, result: RunResult) -> None:
        """Record the REAL phase breakdown produced by the last chosen config."""
        steps = max(1, result.steps)
        profile = {
            "prefill": result.prefill_s / steps,
            "decode": result.decode_s / steps,
            "kv_alloc": 0.0,
            "overhead": result.host_s / steps,
        }
        self.observed_profiles.append(profile)
        for _ in range(5):
            self.profiler.record_step(**profile)

    def choose(self, state: RuntimeState) -> CARLConfig:
        self._round += 1
        entry = self.tuner.observe(self.sched, step=self._round)
        if entry is not None:
            self.adaptations += 1
        cfg = CARLConfig(
            max_batch_size=int(self.sched.max_batch_size),
            chunk_size=int(self.sched.chunk_size),
            use_cuda_graphs=bool(self.sched.use_cuda_graphs),
            eviction_threshold=float(self.sched.evict_threshold),
        ).clamp()
        if self._last is not None and cfg != self._last:
            self.arm_changes += 1
        self._last = cfg
        self.trace.append({"regime": classify_regime(state).value, "arm": -1})
        return cfg

    def observe(self, reward: float) -> None:
        pass


class StaticController:
    """Never adapts."""

    def __init__(self, config: CARLConfig, name: str) -> None:
        self.config = config
        self.name = name
        self.adaptations = 0
        self.arm_changes = 0
        self.trace: list[dict] = []

    def choose(self, state: RuntimeState) -> CARLConfig:
        return self.config

    def observe(self, reward: float) -> None:
        pass


class OracleController:
    """Per-phase brute-force optimum, precomputed BY THE MODEL.

    Takes a {phase_name: config} map produced by `solve_oracle`, and is given the
    TRUE phase label -- that is what makes it an oracle. Note it is an upper
    bound only over the candidate set it was solved on; that set is recorded in
    the result so the bound is interpretable.
    """

    name = "Oracle"

    def __init__(self, by_phase: dict[str, CARLConfig]) -> None:
        self.by_phase = by_phase
        self.adaptations = 0
        self.arm_changes = 0
        self._phase = None
        self.trace: list[dict] = []

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def choose(self, state: RuntimeState) -> CARLConfig:
        return self.by_phase[self._phase]

    def observe(self, reward: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Oracle / static solving.
# ---------------------------------------------------------------------------


def solve_oracle(phases: list[str], candidates: list[CARLConfig],
                 hw: HardwareProfile, objective, seeds=(0, 1, 2),
                 n_per_phase: int = 40, cycle_requests: int = 10,
                 scales=None, weights=None, rounds: int = 2) -> dict:
    """Best per-phase config, optimised IN EPISODE CONTEXT by coordinate ascent.

    WHY NOT PER-PHASE-IN-ISOLATION. The obvious implementation -- optimise each
    phase as a standalone workload -- is wrong, and measurably so: it produced an
    "oracle" that LOST to Static-Best on 4 of 10 scenarios. The reason is
    carry-over. A phase does not run in isolation inside an episode; it inherits
    the queue the previous phase left behind, and it hands one on. A config that
    is optimal for a sustained heavy phase (small batch, protecting TPOT) builds
    a backlog that the following light phase must drain, and the isolated
    solver cannot see that cost.

    An oracle that loses to a baseline is not an oracle, so this solves in
    context: start from the best uniform config, then repeatedly re-optimise one
    phase at a time holding the others fixed, scoring the WHOLE episode each
    time. Coordinate ascent is monotone in the episode objective, so the result
    is never worse than the best uniform config -- which is exactly the property
    the isolated solver violated.

    HONEST SCOPE: coordinate ascent gives a LOWER BOUND on the true per-phase
    optimum, not the global argmax over |candidates|^|phases|. It is reported as
    `oracle_method: "coordinate_ascent"` so nobody reads it as exhaustive.
    """
    from statistics import fmean

    uniq = list(dict.fromkeys(phases))

    def episode_value(assign: dict[str, CARLConfig]) -> float:
        vals = []
        for s in seeds:
            ctrl = OracleController(assign)
            r = run_episode(ctrl, phases, hw, scales, s, n_per_phase=n_per_phase,
                            cycle_requests=cycle_requests, weights=weights)
            vals.append(r["mean_reward"] if objective is None else r["mean_reward"])
        return fmean(vals)

    # Seed from the best UNIFORM assignment (same config everywhere).
    best_uniform, best_v = candidates[0], float("-inf")
    for cfg in candidates:
        v = episode_value({p: cfg for p in uniq})
        if v > best_v:
            best_uniform, best_v = cfg, v
    assign = {p: best_uniform for p in uniq}
    history = [{"round": -1, "assignment": {p: c.as_dict() for p, c in assign.items()},
                "value": best_v, "note": "best uniform (coordinate-ascent seed)"}]

    for rnd in range(rounds):
        improved = False
        for phase in uniq:
            cur = assign[phase]
            for cfg in candidates:
                if cfg == cur:
                    continue
                trial = dict(assign)
                trial[phase] = cfg
                v = episode_value(trial)
                if v > best_v + 1e-12:
                    best_v, assign, improved = v, trial, True
        history.append({"round": rnd,
                        "assignment": {p: c.as_dict() for p, c in assign.items()},
                        "value": best_v})
        if not improved:
            break

    detail = {
        p: {
            "best_config": assign[p].as_dict(),
            "is_a_default_config": assign[p] in list(DEFAULT_CONFIGS.values()),
        } for p in uniq
    }
    return {
        "by_phase": assign,
        "detail": detail,
        "episode_value": best_v,
        "oracle_method": "coordinate_ascent",
        "oracle_scope_note": (
            "Per-phase configs optimised against the WHOLE-EPISODE objective by "
            "coordinate ascent from the best uniform config. Monotone, so never "
            "worse than best-uniform, but a LOWER BOUND on the exhaustive "
            "|candidates|^|phases| optimum."
        ),
        "n_candidates": len(candidates),
        "history": history,
    }


def solve_static_best(phases: list[str], candidates: list[CARLConfig],
                      hw: HardwareProfile, objective, seeds=(0, 1, 2),
                      n_per_phase: int = 40, cycle_requests: int = 10,
                      scales=None, weights=None) -> tuple[CARLConfig, dict]:
    """The single best FIXED config across the whole (multi-phase) workload.

    Searched on held-out seeds over the wide static grid, under whichever
    objective is passed. This is what makes Static-Best-SLO possible.
    """
    # Scored IN EPISODE CONTEXT, exactly like the controllers it is compared
    # against. Scoring each phase in isolation and averaging (the first
    # implementation) ignores queue carry-over across phase boundaries and
    # produced a "Static-Best-SLO" that scored 0.3417 while "Static-Best-Tput"
    # scored 0.4851 on the SAME utility -- i.e. the selector was picking a config
    # that its own objective rated worse. A baseline must be selected under the
    # measurement it will be judged by.
    scored = []
    for cfg in candidates:
        vals = []
        for s in seeds:
            ctrl = StaticController(cfg, "probe")
            ep = run_episode(ctrl, phases, hw, scales, s, n_per_phase=n_per_phase,
                             cycle_requests=cycle_requests, weights=weights)
            vals.append(ep["throughput_tps"] if objective is objective_throughput
                        else ep["mean_reward"])
        scored.append((sum(vals) / len(vals), cfg))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return scored[0][1], {
        "best_value": scored[0][0],
        "best_config": scored[0][1].as_dict(),
        "n_candidates": len(candidates),
        "search_space": {k: list(v) for k, v in STATIC_GRID_AXES.items()},
        "is_a_default_config": scored[0][1] in list(DEFAULT_CONFIGS.values()),
        "top5": [{"value": v, "config": c.as_dict()} for v, c in scored[:5]],
    }


# ---------------------------------------------------------------------------
# The run loop.
# ---------------------------------------------------------------------------


def run_episode(controller, phases: list[str], hw: HardwareProfile,
                scales: RewardScales, seed: int, n_per_phase: int = 40,
                cycle_requests: int = 10, weights: dict | None = None) -> dict:
    """Drive `controller` across a multi-phase workload; return metrics + trace.

    Requests are generated per phase with continuous arrival times, then served
    in `cycle_requests`-sized slices. One controller decision per slice.
    """
    rng = random.Random(seed)
    all_slices: list[tuple[str, list[Request]]] = []
    rid = 0
    t = 0.0
    for i, phase in enumerate(phases):
        spec = WORKLOADS[phase]
        sub = WorkloadSpec(spec.name, n_per_phase, spec.prompt_mean, spec.prompt_std,
                           spec.output_mean, spec.output_std, spec.arrival,
                           spec.rate_rps, t_offset=t)
        reqs = sub.generate(random.Random(seed + 1000 * i), start_rid=rid)
        rid += len(reqs)
        t = max(r.arrival_s for r in reqs) if reqs else t
        for j in range(0, len(reqs), cycle_requests):
            all_slices.append((phase, reqs[j:j + cycle_requests]))

    # Cold start: a neutral observed state (nothing has run yet).
    state = RuntimeState(queue_depth=0, avg_prompt_len=0.0)
    rewards: list[float] = []
    per_cycle: list[dict] = []
    tps_all, ttft99, tpot99 = [], [], []
    occ_all, host_frac = [], []

    for idx, (phase, reqs) in enumerate(all_slices):
        if isinstance(controller, OracleController):
            controller.set_phase(phase)
        cfg = controller.choose(state)
        # Re-base arrival times so each slice starts at t=0 (slices are served
        # back to back; absolute time across the episode is not meaningful here).
        base = min(r.arrival_s for r in reqs)
        for r in reqs:
            r.arrival_s -= base
            r.admit_s = r.first_token_s = r.finish_s = None
            r.prompt_done = r.generated = 0
        res = simulate(cfg, reqs, hw, random.Random(seed * 7919 + idx))

        # Closed-loop baselines receive the REAL phase breakdown their own
        # configuration just produced (Phase 9).
        if hasattr(controller, "feed"):
            controller.feed(res)

        metrics = run_to_metrics(res)
        reward = utility_v2(metrics, weights, scales)
        controller.observe(reward)
        rewards.append(reward)

        avg_prompt = statistics.fmean(r.prompt_len for r in reqs)
        state = run_to_state(res, queue_depth=len(reqs), avg_prompt_len=avg_prompt)

        tps_all.append(res.throughput_tps)
        ttft99.append(res.ttft_p99_ms)
        tpot99.append(res.tpot_p99_ms)
        occ_all.append(res.mean_occupancy)
        host_frac.append(res.host_overhead_fraction)
        per_cycle.append({
            "cycle": idx, "phase": phase,
            "config": cfg.as_dict(),
            "reward": reward,
            "terms": term_breakdown(metrics, weights, scales),
            "throughput_tps": res.throughput_tps,
            "ttft_p99_ms": res.ttft_p99_ms,
            "tpot_p99_ms": res.tpot_p99_ms,
            "mean_batch": res.mean_batch,
            "mean_occupancy": res.mean_occupancy,
            "livelocked": res.livelocked,
            "regime_detected": (controller.trace[-1]["regime"]
                                if controller.trace else None),
            "arm": controller.trace[-1]["arm"] if controller.trace else None,
        })

    return {
        "controller": getattr(controller, "name", type(controller).__name__),
        "seed": seed,
        "mean_reward": statistics.fmean(rewards),
        "throughput_tps": statistics.fmean(tps_all),
        "ttft_p99_ms": statistics.fmean(ttft99),
        "tpot_p99_ms": statistics.fmean(tpot99),
        "mean_occupancy": statistics.fmean(occ_all),
        "host_overhead_fraction": statistics.fmean(host_frac),
        "adaptations": controller.adaptations,
        "arm_changes": controller.arm_changes,
        "n_cycles": len(all_slices),
        "rewards": rewards,
        "per_cycle": per_cycle,
    }


def mean_std(values) -> dict:
    vals = [float(v) for v in values]
    n = len(vals)
    m = statistics.fmean(vals) if vals else 0.0
    s = statistics.stdev(vals) if n > 1 else 0.0
    return {"mean": m, "std": s, "n": n,
            "ci95": [m - 1.96 * s / (n ** 0.5), m + 1.96 * s / (n ** 0.5)] if n > 1 else [m, m]}


__all__ = [
    "WORKLOADS", "nonstationary", "static_candidates", "STATIC_GRID_AXES",
    "objective_throughput", "make_objective_utility", "run_to_metrics",
    "run_to_state", "RuleOnlyController", "BanditController", "StaticController",
    "OracleController", "ClosedLoopAutoTunerController", "solve_oracle", "solve_static_best", "run_episode",
    "mean_std", "check_non_degenerate", "wide_arm_sets", "arm_set_coverage",
    "term_breakdown",
    "LinUCBBandit", "RepairedLinUCBBandit", "HardwareProfile", "RewardScales",
]
