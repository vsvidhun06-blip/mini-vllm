"""A mechanistic serving model whose optimum is NOT written down anywhere.

WHY THIS REPLACES scripts/benchmark_carl.WorkloadModel
------------------------------------------------------
The previous cost model was:

    throughput = base_tps[regime] * (0.5 + 0.5 * m)
    m          = 1 - ||config - DEFAULT_CONFIGS[regime]|| / sqrt(6)

That is an answer key. Its global maximum IS `DEFAULT_CONFIGS[regime]`, which is
also arm 0 of every bandit, which is also what `OracleAgent` returns, which is
also five of the six candidates `best_static_config` searches. On such a
substrate no controller can demonstrate learning: the best achievable policy is
"never move", and any exploration is pure loss. Verified in
`docs/eval/raw/repair/bandit_null_check.json`.

This module models the *engine* instead of the *answer*. Nothing here knows what
a good configuration is. Throughput and latency fall out of a step-time equation;
the best configuration for a workload is whatever maximises the objective, and it
must be found by search (`optimal_arm`, `oracle_config`) rather than looked up.

THE MODEL
---------
The engine advances in discrete steps. Each step either prefills chunks of
admitted requests or decodes one token for every request in the active batch.

    step_time(b, prefill_tokens) =
        decode_fixed + b * c_dec                          (if decoding)
        host_overhead + prefill_kernel_floor
                      + prefill_tokens * c_pre            (if prefilling)

The single most important structural fact is that `decode_fixed` is paid ONCE PER
STEP regardless of batch size, while `b * c_dec` scales with the batch.
Therefore:

    per-step throughput = b / step_time(b)
                        = b / (fixed + b * c_dec)

which is INCREASING in b and saturates at 1/c_dec. Larger batches amortise fixed
cost -- that is the whole throughput story, and it is a *mechanism*, not a table.

Simultaneously, each in-flight request waits `step_time(b)` per token, so:

    TPOT(b) = step_time(b) = fixed + b * c_dec          -- INCREASING in b

Throughput and per-token latency pull in opposite directions through the same
scalar. That tension is what makes a configuration choice non-trivial, and it is
exactly what the old model lacked.

Two further constraints make the space genuinely multi-dimensional:

  * KV CAPACITY. A request holds ceil((prompt + generated)/block_size) blocks out
    of `num_blocks`. Admission is capped by free blocks as well as by
    `max_batch_size`, so with long prompts the effective batch is memory-bound
    and raising `max_batch_size` does nothing (or, with preemption enabled,
    causes thrash). This is why LONG_CONTEXT does not simply want batch=32.

  * CHUNKED PREFILL. `chunk_size` sets the prefill token budget per step. A large
    chunk finishes a prompt in fewer steps (better TTFT for that request) but
    makes each prefill step long, stalling decode for everyone already in flight
    (worse TPOT). The classic Sarathi trade-off, and it means chunk_size has a
    real interior optimum that depends on the prompt-length mix.

WHERE THE CONSTANTS COME FROM -- READ THIS BEFORE TRUSTING A NUMBER
--------------------------------------------------------------------
The DECODE constants are CALIBRATED against real T4 measurements. The Phase 15+16
forced-`max_batch_size` intervention

    docs/eval/raw/repair/batch_intervention_phase15_16_s42_s43_s44.json
    (scripts/eval/repair/batch_intervention.py; TinyLlama-1.1B-Chat-v1.0,
     Tesla T4, batches 1..32 x seeds 42,43,44 x {burst, poisson 2.0, poisson 8.0})

sweeps the realised batch and least-squares fits `step_time(b) = fixed + b*c_dec`
separately for the eager arm and for the arm whose every row provably replayed
captured CUDA graphs:

    EAGER        fixed = 0.02546177948065939 s   c_dec = 0.009448482794434556 s
                 R^2   = 0.9985308333916275      n     = 54
    CUDA GRAPHS  fixed = 0.010862193551853631 s  c_dec = 0.0051007379055177885 s
                 R^2   = 0.9927282657468219      n     = 54

This REPLACES the previous decode timing model, which was an assumption about
*shape* anchored to two points (`spec_breakeven_results.json` at batch=1 plus a
reconstructed batch~8 point) and which held `c_dec` constant across the two arms.
The measurement contradicts that shape: CUDA graphs cut the MARGINAL per-row cost
by ~46% as well as the fixed cost, so both coefficients are arm-keyed here.

WHAT THE INTERVENTION DOES NOT IDENTIFY. The fit resolves one AGGREGATE fixed
term and one AGGREGATE marginal term per arm. It does NOT decompose `fixed` into
Python dispatch time versus fixed GPU kernel work; the eager-minus-graph
difference bounds how much of it graph capture removes, but does not attribute
it. Consequently `host_overhead_s` / `cuda_graph_host_overhead_s` no longer set
decode wall-clock time. They survive for two jobs only: the per-step fixed cost
of a PREFILL step, and an ACCOUNTING split of the measured fixed term into
`RunResult.host_s` versus `RunResult.decode_s`. That split is a convention
constrained by the measured aggregate -- it is NOT measured Python dispatch time
and must not be reported as such.

PREFILL remains UNCALIBRATED. `prefill_kernel_floor_s` and `prefill_per_token_s`
are still assumptions: the sweep averages step time over prefill and decode steps
alike, so it cannot separate them. Prefill timing is deliberately left unchanged
by this calibration.

The profile is a dataclass so that calibration is a data change, not a code
change.

Torch-free, numpy-free, deterministic given a seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Hardware / engine profile.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Timing constants for one (engine, model, GPU) combination.

    All times in SECONDS. The DECODE coefficients are fitted to the Phase 15+16
    T4 intervention; the prefill constants and host/decode accounting split are
    not directly identified by that experiment.
    """

    # Calibrated decode model: fixed + batch_size * marginal.
    decode_fixed_per_step_s: float = 0.0254618
    decode_per_row_s: float = 0.00944848
    cuda_graph_decode_fixed_per_step_s: float = 0.0108622
    cuda_graph_decode_per_row_s: float = 0.00510074

    # These remain the prefill fixed term and an ACCOUNTING split of the measured
    # decode fixed term. They are not independently measured Python-dispatch time.
    host_overhead_s: float = 0.0140
    cuda_graph_host_overhead_s: float = 0.0020

    # Prefill: fixed cost per prefill step, plus marginal cost per prompt token.
    prefill_kernel_floor_s: float = 0.0040
    prefill_per_token_s: float = 0.00018

    # KV pool geometry (mirrors src/carl/live.py: 1024 blocks x 16 tokens).
    num_blocks: int = 1024
    block_size: int = 16

    # Multiplicative timing jitter, 1 + N(0, sigma). Models real run-to-run
    # variation. Deliberately modest: seeds must produce genuinely different
    # numbers, but the effect size must not be an artefact of this knob.
    # (The old model used 0.05 on throughput directly, which is what produced
    # the d=23.17 headline -- an effect size set by a simulator constant.)
    timing_jitter_sigma: float = 0.04

    def decode_fixed(self, use_cuda_graphs: bool) -> float:
        """Measured per-step decode cost that does not scale with batch."""
        return (
            self.cuda_graph_decode_fixed_per_step_s
            if use_cuda_graphs
            else self.decode_fixed_per_step_s
        )

    def decode_per_row_for(self, use_cuda_graphs: bool) -> float:
        """Measured marginal decode cost per row in the batch."""
        return (
            self.cuda_graph_decode_per_row_s
            if use_cuda_graphs
            else self.decode_per_row_s
        )

    def host_overhead(self, use_cuda_graphs: bool) -> float:
        """Prefill fixed cost and host-share convention used for accounting."""
        return self.cuda_graph_host_overhead_s if use_cuda_graphs else self.host_overhead_s

    def kv_capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size


# ---------------------------------------------------------------------------
# Requests and workloads.
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """One request with an arrival time, a prompt length and an output length."""
    rid: int
    arrival_s: float
    prompt_len: int
    output_len: int

    # Filled in by the simulator.
    admit_s: float | None = None
    first_token_s: float | None = None
    finish_s: float | None = None
    prompt_done: int = 0          # prompt tokens prefilled so far
    generated: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.arrival_s) * 1000.0

    @property
    def tpot_ms(self) -> float | None:
        if self.finish_s is None or self.first_token_s is None or self.generated < 2:
            return None
        return (self.finish_s - self.first_token_s) / (self.generated - 1) * 1000.0

    def kv_tokens(self) -> int:
        return self.prompt_done + self.generated


@dataclass
class WorkloadSpec:
    """A request stream: arrival process + length distributions.

    `arrival` is one of:
      "poisson"       -- exponential inter-arrival at `rate_rps` (open loop)
      "deterministic" -- fixed 1/rate_rps spacing
      "burst"         -- every request arrives at t=0 (the legacy bulk dump,
                         retained deliberately as a STRESS workload)
    """
    name: str
    n_requests: int
    prompt_mean: float
    prompt_std: float
    output_mean: float
    output_std: float
    arrival: str = "poisson"
    rate_rps: float = 2.0
    t_offset: float = 0.0

    def generate(self, rng: random.Random, start_rid: int = 0) -> list[Request]:
        reqs: list[Request] = []
        t = self.t_offset
        for i in range(self.n_requests):
            if self.arrival == "burst":
                arrival = self.t_offset
            elif self.arrival == "deterministic":
                arrival = t
                t += 1.0 / max(self.rate_rps, 1e-9)
            else:  # poisson
                arrival = t
                t += rng.expovariate(max(self.rate_rps, 1e-9))
            p = max(1, int(rng.gauss(self.prompt_mean, self.prompt_std)))
            o = max(1, int(rng.gauss(self.output_mean, self.output_std)))
            reqs.append(Request(rid=start_rid + i, arrival_s=arrival,
                                prompt_len=p, output_len=o))
        return reqs


# ---------------------------------------------------------------------------
# The simulator.
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregate metrics for one simulated serving run."""
    throughput_tps: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    mean_batch: float
    mean_occupancy: float          # mean_batch / max_batch_size, in [0,1]
    kv_utilisation: float
    preemptions: int
    n_completed: int
    wall_s: float
    steps: int
    host_overhead_fraction: float  # share of wall time spent in host dispatch
    # Real phase accounting, in seconds. Exists so a bottleneck-reactive
    # baseline (AutoTuner) can be driven CLOSED-LOOP off the consequences of the
    # configuration it actually chose, instead of off a synthetic per-regime
    # profile that ignores its own actions (the defect documented as R-AT in
    # docs/eval/CARL_REPAIR_STATUS.md).
    prefill_s: float = 0.0
    decode_s: float = 0.0
    host_s: float = 0.0
    # True when the run hit the preemption-livelock guard: KV pressure was high
    # enough that preemption kept destroying progress. This is a REAL failure
    # mode of preemption under memory pressure, not a simulator artefact, so it
    # is surfaced rather than smoothed away.
    livelocked: bool = False
    # True when the step budget was exhausted before the workload drained.
    truncated: bool = False

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def simulate(config, requests: list[Request], hw: HardwareProfile,
             rng: random.Random, max_steps: int = 200_000) -> RunResult:
    """Serve `requests` under `config` and return realised metrics.

    `config` is duck-typed on CARLConfig: it must expose max_batch_size,
    chunk_size, preemption_enabled, eviction_threshold, use_cuda_graphs.

    The loop mirrors ContinuousBatchScheduler.step(): admit while batch and KV
    allow, prefill chunks for anyone still prefilling, otherwise decode one token
    for the whole active batch.

    NOTHING in here consults DEFAULT_CONFIGS or any per-regime table. The quality
    of a configuration is an emergent property of the timing equation.
    """
    max_batch = int(config.max_batch_size)
    chunk = int(config.chunk_size)
    preempt_ok = bool(config.preemption_enabled)
    evict_thresh = float(config.eviction_threshold)
    graphs = bool(config.use_cuda_graphs)

    host = hw.host_overhead(graphs)
    kv_cap = hw.kv_capacity_tokens()
    # Eviction threshold gates how full the pool is allowed to get before the
    # engine stops admitting. A higher threshold packs memory denser (more
    # concurrency) at the cost of leaving less slack for in-flight growth.
    admit_cap_tokens = int(kv_cap * evict_thresh)

    pending = sorted(requests, key=lambda r: r.arrival_s)
    idx = 0
    waiting: list[Request] = []
    active: list[Request] = []
    done: list[Request] = []

    now = 0.0
    steps = 0
    batch_samples: list[int] = []
    kv_samples: list[float] = []
    preemptions = 0
    host_time = 0.0
    prefill_time = 0.0
    decode_time = 0.0
    tokens_emitted = 0
    # Livelock guard. Preemption resets a victim's KV (recompute semantics), so
    # under sustained memory pressure a victim can be preempted forever and the
    # run never drains. Real engines hit this too; we cap it, record it, and stop
    # preempting rather than spinning to the step budget.
    preempt_budget = 10 * max(1, len(requests))
    livelocked = False

    def kv_in_use() -> int:
        return sum(
            math.ceil(max(1, r.kv_tokens()) / hw.block_size) * hw.block_size
            for r in active
        )

    while steps < max_steps:
        # --- release arrivals up to `now` -----------------------------------
        while idx < len(pending) and pending[idx].arrival_s <= now:
            waiting.append(pending[idx])
            idx += 1

        if not active and not waiting:
            if idx >= len(pending):
                break
            # Idle: jump to the next arrival. Idle time counts toward wall
            # clock, which is what makes an open-loop arrival process produce a
            # throughput number bounded by the ARRIVAL RATE rather than by the
            # engine -- the single biggest difference from a bulk dump.
            now = pending[idx].arrival_s
            continue

        # --- admission -------------------------------------------------------
        used = kv_in_use()
        while waiting and len(active) < max_batch:
            r = waiting[0]
            need = math.ceil(max(1, r.prompt_len) / hw.block_size) * hw.block_size
            if used + need > admit_cap_tokens:
                # Preempt the most recently admitted request to make room, but
                # only while the livelock budget lasts and only if evicting it
                # could actually admit the head (otherwise we destroy progress
                # for nothing).
                victim_frees = 0
                if active:
                    v = active[-1]
                    victim_frees = math.ceil(
                        max(1, v.kv_tokens()) / hw.block_size) * hw.block_size
                if (preempt_ok and active and preemptions < preempt_budget
                        and used - victim_frees + need <= admit_cap_tokens):
                    victim = active.pop()
                    victim.prompt_done = 0
                    victim.generated = 0
                    victim.first_token_s = None
                    waiting.insert(1, victim)
                    preemptions += 1
                    used = kv_in_use()
                    continue
                if preempt_ok and preemptions >= preempt_budget:
                    livelocked = True
                break
            waiting.pop(0)
            r.admit_s = now
            active.append(r)
            used += need

        if not active:
            # Queue is non-empty but nothing fits: advance to the next arrival
            # or bail out to avoid an infinite loop on an unservable request.
            if idx < len(pending):
                now = max(now, pending[idx].arrival_s)
                continue
            break

        # --- one engine step --------------------------------------------------
        prefilling = [r for r in active if r.prompt_done < r.prompt_len]
        jitter = max(0.5, 1.0 + rng.gauss(0.0, hw.timing_jitter_sigma))

        if prefilling:
            # Chunked prefill: spend up to `chunk` tokens this step, FIFO.
            budget = chunk
            for r in prefilling:
                if budget <= 0:
                    break
                take = min(budget, r.prompt_len - r.prompt_done)
                r.prompt_done += take
                budget -= take
            spent = chunk - budget
            step_s = (host + hw.prefill_kernel_floor_s
                      + spent * hw.prefill_per_token_s) * jitter
            prefill_time += (hw.prefill_kernel_floor_s
                             + spent * hw.prefill_per_token_s) * jitter
        else:
            b = len(active)
            fixed = hw.decode_fixed(graphs)
            marginal = b * hw.decode_per_row_for(graphs)
            step_s = (fixed + marginal) * jitter

            # host is only an accounting split of the measured aggregate fixed
            # decode term. Never let the accounting share exceed that fixed term.
            host_part = min(host, fixed)
            decode_time += ((fixed - host_part) + marginal) * jitter

            for r in active:
                r.generated += 1
                tokens_emitted += 1
                if r.first_token_s is None:
                    r.first_token_s = now + step_s
            batch_samples.append(b)

        if prefilling:
            host_time += host * jitter
        else:
            host_time += host_part * jitter
        now += step_s
        steps += 1
        kv_samples.append(kv_in_use() / kv_cap)

        # --- retire finished --------------------------------------------------
        still: list[Request] = []
        for r in active:
            if r.generated >= r.output_len:
                r.finish_s = now
                done.append(r)
            else:
                still.append(r)
        active = still

    wall = max(now, 1e-9)
    ttfts = [r.ttft_ms for r in done if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in done if r.tpot_ms is not None]
    mean_batch = (sum(batch_samples) / len(batch_samples)) if batch_samples else 0.0

    return RunResult(
        throughput_tps=tokens_emitted / wall,
        ttft_p50_ms=_pct(ttfts, 50), ttft_p99_ms=_pct(ttfts, 99),
        tpot_p50_ms=_pct(tpots, 50), tpot_p99_ms=_pct(tpots, 99),
        mean_batch=mean_batch,
        mean_occupancy=mean_batch / max(1, max_batch),
        kv_utilisation=(sum(kv_samples) / len(kv_samples)) if kv_samples else 0.0,
        preemptions=preemptions,
        n_completed=len(done),
        wall_s=wall,
        steps=steps,
        host_overhead_fraction=host_time / wall,
        livelocked=livelocked,
        truncated=steps >= max_steps,
        prefill_s=prefill_time,
        decode_s=decode_time,
        host_s=host_time,
    )


# ---------------------------------------------------------------------------
# Independent oracle / best-static search.
# ---------------------------------------------------------------------------
#
# These exist so that NOTHING in the evaluation reads a per-regime answer table.
# The oracle is whatever the model says is best, found by evaluating every
# candidate. If someone changes HardwareProfile, the oracle moves -- which is the
# correctness property the old substrate lacked.
# ---------------------------------------------------------------------------


def evaluate_config(config, workload: WorkloadSpec, hw: HardwareProfile,
                    seed: int) -> RunResult:
    """Deterministically evaluate one config on one workload+seed."""
    rng = random.Random(seed)
    reqs = workload.generate(rng, start_rid=0)
    # Fresh RNG for timing so the request stream is identical across configs
    # (a paired comparison: the only thing that differs is the configuration).
    return simulate(config, reqs, hw, random.Random(seed ^ 0x5EED))


def optimal_arm(candidates: list, workload: WorkloadSpec, hw: HardwareProfile,
                objective, seeds: tuple[int, ...] = (0, 1, 2)) -> tuple[int, float]:
    """Brute-force argmax over `candidates` under `objective`.

    Args:
        candidates: list of configs.
        objective: callable(RunResult) -> float, higher is better. Passing the
            objective in explicitly (rather than hard-coding throughput) is what
            makes Static-Best-SLO and Static-Best-throughput different baselines
            rather than the same one under two names.
        seeds: averaged over, so the winner is not a single-draw fluke.

    Returns:
        (index, mean_objective_value).
    """
    best_i, best_v = 0, float("-inf")
    for i, cfg in enumerate(candidates):
        vals = [objective(evaluate_config(cfg, workload, hw, s)) for s in seeds]
        v = sum(vals) / len(vals)
        if v > best_v:
            best_i, best_v = i, v
    return best_i, best_v


def oracle_config(candidates: list, workload: WorkloadSpec, hw: HardwareProfile,
                  objective, seeds: tuple[int, ...] = (0, 1, 2)):
    """The best candidate for this workload, computed by the model itself."""
    i, _ = optimal_arm(candidates, workload, hw, objective, seeds)
    return candidates[i]
