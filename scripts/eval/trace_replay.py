"""
LMSYS-Chat-1M trace-replay evaluation for the CARL controller.

WHAT THIS IS (read before trusting any number it prints)
--------------------------------------------------------
A CONTROL-LOOP SIMULATION, exactly like the rest of scripts/eval/ (see
_harness.SIM_NOTE). It drives the REAL CARL controller, the REAL LinUCB bandit,
the REAL AutoTuner and the REAL regime classifier, but the serving metrics
(throughput / TTFT / TPOT / cache) come from benchmark_carl.py's analytical
WorkloadModel, NOT from running a model on a GPU. The contribution under test is
the CONTROL POLICY, which is hardware-independent; the comparisons between
methods (and CARL's approach to the oracle) are the result, not the absolute
values. The regime oracle is near-optimal BY CONSTRUCTION.

WHAT THE TRACE ADDS over the synthetic scenarios in benchmark_carl.py
---------------------------------------------------------------------
benchmark_carl's scenarios hand-author the regime sequence (200 INTERACTIVE,
then 200 BATCH, ...). Here the regime sequence is DERIVED from a real request
trace (LMSYS-Chat-1M), so the workload's temporal structure -- burstiness, idle
gaps, multi-turn context growth, and the natural length mix -- decides the regime
stream instead of us. Concretely, per request we extract:

  * input token length      (tokenize the human turn)
  * output token length     (tokenize the assistant turn)
  * inter-arrival time      (original timestamps where available, else sampled
                             from the empirical inter-arrival distribution)
  * conversation length     (number of turns)

and we PRESERVE the temporal characteristics rather than smoothing them:

  * burstiness   -- arrivals feed a virtual queue; a clump of fast arrivals
                    grows the backlog so classify_regime reads BURST.
  * idle periods -- a gap > 10 s drains the queue back to the INTERACTIVE floor.
  * multi-turn   -- the prompt GROWS with the conversation (each turn's effective
                    prompt is the accumulated prior context + its own input), so
                    deep conversations drift into LONG_CONTEXT / CACHE_HEAVY.

The per-request regimes are then grouped into ROUND_SIZE-request control cycles
(the cadence the controller actually fires on, identical to benchmark_carl), and
every baseline is driven over that identical regime stream.

DATASET & FALLBACK
------------------
LMSYS-Chat-1M is gated on HuggingFace. If the download fails (gated / offline /
no auth) we FALL BACK to a synthetic trace whose length distribution matches the
LMSYS shape (log-normal, mean=128, std=256 tokens) and LOG that the fallback was
used. Each seed samples a contiguous 10000-request window from a DIFFERENT
position in the dataset (a different RNG stream in fallback mode).

BASELINES
---------
  CARL-Full       LinUCB, alpha=0.5 (the proposed method).
  Static-Best     the single fixed config grid-searched on the first 100 requests.
  AutoTuner       the real per-component hill-climber (independent tuning).
  EpsilonGreedy   epsilon=0.1, context-free, over CARL's exact arm space.
  UCB1            context-free UCB1, over CARL's exact arm space.

EpsilonGreedy and UCB1 are NEW here (context-free bandits the rest of the suite
doesn't define); they reuse CARL's per-regime arm sets (config.all_arm_sets) so
the ONLY difference from CARL is that they ignore the context vector -- isolating
the value of CARL's contextual model.

OUTPUTS
-------
  --mode replay       -> docs/eval/trace_replay_results.json   (default)
  --mode scalability  -> docs/eval/scalability_results.json

Run:
  python scripts/eval/trace_replay.py                    # 10 seeds, 10k window
  python scripts/eval/trace_replay.py --seeds 3 --requests 2000   # quick
  python scripts/eval/trace_replay.py --mode scalability
  python scripts/eval/trace_replay.py --real             # force the gated LMSYS load

CONSTRAINTS honored: same SLO as the suite (ttft=200, tpot=50, throughput_ref=50);
CPU-only / Colab-T4 friendly (torch-free path); prints a runtime estimate before
starting; never imports from or modifies ablation_live.py (or any existing script).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace

# --- path bootstrap ---------------------------------------------------------
# Mirror the other eval scripts: put repo root + scripts/ on sys.path so this
# runs standalone as `python scripts/eval/trace_replay.py`. _harness performs the
# same bootstrap on import, but we do it here too so our direct `import
# benchmark_carl` / `import src...` resolve regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # scripts/eval
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)                    # scripts
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)                   # repo root
for _p in (_REPO_ROOT, _SCRIPTS_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# Reuse the shared harness + the cost-model source of truth. _harness is the
# suite's shared module (every eval imports it); benchmark_carl owns the
# WorkloadModel/_synth_state cost model. We import NOTHING from ablation_live.
import _harness as h  # noqa: E402
import benchmark_carl as bc  # noqa: E402
from src.carl.bandit import (  # noqa: E402
    DEFAULT_UTILITY_WEIGHTS,
    LinUCBBandit,
    PerRegimeBandit,
    utility,
)
from src.carl.config import (  # noqa: E402
    CARLConfig,
    DEFAULT_CONFIGS,
    all_arm_sets,
    config_arms,
)
from src.carl.controller import SLO, CARLController  # noqa: E402
from src.carl.state import (  # noqa: E402
    FEATURE_DIM,
    MetricsTracker,
    RuntimeState,
    WorkloadRegime,
    classify_regime,
)

tqdm = h.tqdm  # tqdm-or-noop shim from the harness.


# ===========================================================================
# Constants (the spec's experimental parameters live here, single source).
# ===========================================================================

ROUND_SIZE = bc.ROUND_SIZE          # 10 requests per control cycle (the cadence).
DEFAULT_SEEDS = 10                  # different contiguous trace windows.
WINDOW_SIZE = 10_000               # requests per window.
CHECKPOINTS = [100, 500, 1000, 5000, 10_000]   # request-count checkpoints.

# SLO config -- IDENTICAL to the rest of the eval suite (the spec's constraint).
SLO_TTFT_MS = 200.0
SLO_TPOT_MS = 50.0
SLO_THROUGHPUT_REF = 50.0

# LMSYS length distribution for the synthetic fallback (the spec's numbers).
LMSYS_MEAN_TOK = 128.0
LMSYS_STD_TOK = 256.0

# Idle-period threshold: an inter-arrival gap larger than this drains the queue.
IDLE_GAP_S = 10.0
# Virtual-queue cap on concurrently-served requests (so a deep backlog reads as
# BURST -- queue_depth > 2 * active -- rather than being absorbed as steady BATCH).
MAX_ACTIVE = 16

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
REPLAY_OUT = os.path.join(DOCS_EVAL, "trace_replay_results.json")
SCALABILITY_OUT = os.path.join(DOCS_EVAL, "scalability_results.json")
LMSYS_DATASET = "lmsys/lmsys-chat-1m"

# Bootstrap / statistics settings.
N_BOOTSTRAP = 1000                 # resamples for the 95% CIs (the spec's count).
BOOTSTRAP_SEED = 12345             # deterministic resampling.

# DynOracle estimation: rounds simulated per (regime, arm) to estimate its mean
# reward in hindsight (the static best-arm-per-regime oracle).
ORACLE_ROUNDS = 200
ORACLE_SEED = 99_999

BASELINES = ["CARL-Full", "Static-Best", "AutoTuner", "EpsilonGreedy", "UCB1"]


def make_slo() -> SLO:
    """The suite-standard SLO (ttft<200ms, tpot<50ms, throughput_ref=50)."""
    return SLO(ttft_ms=SLO_TTFT_MS, tpot_ms=SLO_TPOT_MS,
               throughput_ref=SLO_THROUGHPUT_REF)


# ===========================================================================
# Trace extraction: a flat request stream with temporal + multi-turn structure.
# ===========================================================================


@dataclass
class TraceRequest:
    """One request (= one conversation turn) as the replay consumes it.

    input_len / output_len are this turn's own token counts; context_len is the
    ACCUMULATED prior context in its conversation (so input_len + context_len is
    the effective prompt the engine would actually run -- this is how multi-turn
    growth drives the LONG_CONTEXT / CACHE_HEAVY regimes). arrival is an absolute
    timestamp in seconds (real or sampled); turn is the 0-based turn index.
    """
    input_len: int
    output_len: int
    context_len: int
    arrival: float
    conv_id: int
    turn: int

    @property
    def effective_prompt_len(self) -> float:
        return float(self.context_len + self.input_len)


def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """(mu, sigma) of the underlying normal so the log-normal has this mean/std.

    For X ~ LogNormal(mu, sigma):  E[X] = exp(mu + sigma^2/2),
    Var[X] = (exp(sigma^2) - 1) * exp(2 mu + sigma^2). Inverting gives the
    closed form below; cv = std/mean is the coefficient of variation.
    """
    cv2 = (std / mean) ** 2
    sigma2 = math.log(1.0 + cv2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, math.sqrt(sigma2)


def _sample_interarrival(rng: random.Random, *, between_conversations: bool) -> float:
    """Sample one inter-arrival gap (seconds), preserving burstiness + idle gaps.

    We DELIBERATELY do not smooth: most gaps are short (rapid turns / a busy
    server -> bursts), but between conversations there is a 15% chance of a true
    idle gap (> IDLE_GAP_S) so the trace contains the long quiet periods the
    spec asks us to preserve. Within a conversation the gap is short "think
    time". Means are illustrative; the SHAPE (heavy-tailed, occasional long
    idles) is what matters for the regime stream.
    """
    if between_conversations and rng.random() < 0.15:
        # A genuine idle period: server goes quiet between sessions.
        return rng.uniform(IDLE_GAP_S, 3.0 * IDLE_GAP_S)
    # Exponential think-time: short within a conversation, slightly longer
    # between conversations. Heavy-tailed -> natural bursts.
    mean = 0.4 if between_conversations else 0.25
    return rng.expovariate(1.0 / mean)


def synthetic_window(seed: int, n_requests: int) -> list[TraceRequest]:
    """A synthetic request stream with the LMSYS length shape + temporal texture.

    Lengths are log-normal(mean=128, std=256) per the spec. Conversations have a
    log-normal number of turns; context accumulates across turns (multi-turn
    growth); arrivals are laid down with _sample_interarrival so burstiness and
    idle gaps are present. Different `seed` == a different contiguous "window".
    """
    rng = random.Random(seed * 1_000_003 + 7)   # distinct stream per seed/window.
    mu_len, sigma_len = _lognormal_params(LMSYS_MEAN_TOK, LMSYS_STD_TOK)
    # Conversation length ~ log-normal with a small mean (most chats are short,
    # a few are long). mean ~ 2.7 turns, std ~ 2.5 -> a realistic skew.
    mu_turns, sigma_turns = _lognormal_params(2.7, 2.5)

    reqs: list[TraceRequest] = []
    t = 0.0
    conv_id = 0
    while len(reqs) < n_requests:
        conv_id += 1
        conv_len = max(1, int(round(rng.lognormvariate(mu_turns, sigma_turns))))
        conv_len = min(conv_len, 40)            # guard the tail.
        t += _sample_interarrival(rng, between_conversations=True)
        context = 0
        for turn in range(conv_len):
            if len(reqs) >= n_requests:
                break
            in_len = max(1, int(round(rng.lognormvariate(mu_len, sigma_len))))
            out_len = max(1, int(round(rng.lognormvariate(mu_len, sigma_len))))
            reqs.append(TraceRequest(input_len=in_len, output_len=out_len,
                                     context_len=context, arrival=t,
                                     conv_id=conv_id, turn=turn))
            context += in_len + out_len
            if turn < conv_len - 1:
                t += _sample_interarrival(rng, between_conversations=False)
    return reqs[:n_requests]


def _make_token_counter(tokenizer_name: str | None):
    """Return a `count_tokens(text)->int`, preferring a real tokenizer.

    Tries to load `tokenizer_name` (default the LMSYS model's vicuna tokenizer);
    on any failure (offline / not cached) falls back to a ~4-chars/token
    heuristic. The chosen mode is reported in the output so a reader knows
    whether lengths are real token counts or an estimate.
    """
    name = tokenizer_name or "lmsys/vicuna-7b-v1.5"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)

        def count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        return count, f"transformers:{name}"
    except Exception as exc:   # offline / uncached / gated tokenizer -> heuristic.
        print(f"  tokenizer '{name}' unavailable ({type(exc).__name__}); "
              f"using ~4-chars/token heuristic.", flush=True)

        def count(text: str) -> int:
            return max(1, len(text) // 4)

        return count, "heuristic:chars/4"


def lmsys_window(seed: int, n_requests: int, token_count, *, offset: int):
    """Extract a contiguous `n_requests`-request window from LMSYS-Chat-1M.

    Streams the gated dataset, skips `offset` conversations (so each seed reads a
    DIFFERENT contiguous window), and flattens each conversation into per-turn
    TraceRequests: human turns -> input_len, the following assistant turn ->
    output_len, with context accumulating across turns. Inter-arrivals use the
    row's `tstamp` when present (consecutive conversation timestamps), otherwise
    a sampled gap. Raises on any failure so the caller can fall back to synthetic.
    """
    from datasets import load_dataset
    stream = load_dataset(LMSYS_DATASET, split="train", streaming=True)

    rng = random.Random(seed * 7919 + 1)
    reqs: list[TraceRequest] = []
    conv_id = 0
    t = 0.0
    prev_tstamp: float | None = None

    for i, row in enumerate(stream):
        if i < offset:
            continue
        if len(reqs) >= n_requests:
            break
        conv = row.get("conversation")
        if not isinstance(conv, list) or not conv:
            continue
        conv_id += 1

        # Inter-conversation gap: prefer real timestamps, else sample.
        tstamp = row.get("tstamp")
        if isinstance(tstamp, (int, float)) and prev_tstamp is not None:
            gap = max(0.0, float(tstamp) - prev_tstamp)
            # Guard absurd gaps from unsorted streams: cap at a long idle.
            t += min(gap, 5.0 * IDLE_GAP_S)
        else:
            t += _sample_interarrival(rng, between_conversations=True)
        if isinstance(tstamp, (int, float)):
            prev_tstamp = float(tstamp)

        # Walk the turns, pairing each human turn with its assistant reply.
        context = 0
        turn = 0
        pending_input: int | None = None
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            n_tok = token_count(content)
            if role in ("user", "human"):
                pending_input = n_tok
            elif role in ("assistant", "gpt", "bot") and pending_input is not None:
                if len(reqs) >= n_requests:
                    break
                reqs.append(TraceRequest(
                    input_len=pending_input, output_len=n_tok,
                    context_len=context, arrival=t, conv_id=conv_id, turn=turn))
                context += pending_input + n_tok
                turn += 1
                pending_input = None
                if turn < 40:   # within-conversation think time.
                    t += _sample_interarrival(rng, between_conversations=False)

    if not reqs:
        raise RuntimeError("LMSYS stream yielded no usable conversations")
    return reqs[:n_requests]


def load_window(seed: int, n_requests: int, *, force_real: bool,
                token_count, offset: int) -> tuple[list[TraceRequest], bool]:
    """Load one trace window; return (requests, fallback_used).

    Tries the real gated LMSYS stream first when `force_real` is set OR the
    `datasets` package is importable; on ANY failure logs and falls back to the
    synthetic LMSYS-shaped window. fallback_used=True means the numbers come from
    the synthetic distribution, not the real dataset.
    """
    try:
        import datasets  # noqa: F401
        have_datasets = True
    except Exception:
        have_datasets = False

    if force_real or have_datasets:
        try:
            reqs = lmsys_window(seed, n_requests, token_count, offset=offset)
            return reqs, False
        except Exception as exc:
            print(f"  [seed {seed}] LMSYS load failed ({type(exc).__name__}: "
                  f"{exc}); FALLING BACK to synthetic LMSYS-shaped trace.",
                  flush=True)
    else:
        print(f"  [seed {seed}] `datasets` unavailable; using synthetic "
              f"LMSYS-shaped trace.", flush=True)
    return synthetic_window(seed, n_requests), True


# ===========================================================================
# Regime derivation: trace -> per-request regimes -> control-cycle regimes.
# ===========================================================================


def _service_rate(reqs: list[TraceRequest]) -> float:
    """Virtual server service rate (req/s) sized just under the mean arrival rate.

    Sized at 0.9x the trace's mean arrival rate so the queue is slightly
    under-provisioned -> bursts of fast arrivals build a visible backlog (read as
    BURST) while idle gaps drain it. A rate >= arrival rate would never queue and
    the trace would collapse to INTERACTIVE everywhere.
    """
    if len(reqs) < 2:
        return 1.0
    span = reqs[-1].arrival - reqs[0].arrival
    if span <= 0:
        return float(len(reqs))      # all simultaneous -> high rate.
    mean_rate = (len(reqs) - 1) / span
    return max(1e-3, 0.9 * mean_rate)


def derive_regimes(reqs: list[TraceRequest]) -> tuple[list[WorkloadRegime], list[WorkloadRegime], dict]:
    """Map a request stream to per-request regimes (via a virtual queue), then to
    per-control-cycle regimes (majority over each ROUND_SIZE block).

    The virtual queue advances by the real inter-arrival times and drains at
    _service_rate, so a clump of fast arrivals grows queue_depth (BURST) and a
    gap > IDLE_GAP_S drains it (back toward INTERACTIVE). The effective prompt
    length (accumulated context + input) and a turn-growing cache-hit estimate
    (deep multi-turn conversations share a prefix) feed the REAL classify_regime,
    so the regime stream is decided by the same rule the live engine uses.

    Returns (round_regimes, per_request_regimes, trace_stats).
    """
    mu = _service_rate(reqs)
    queue = 0.0
    last_t = reqs[0].arrival if reqs else 0.0
    per_req: list[WorkloadRegime] = []
    idle_periods = 0
    burst_reqs = 0

    for r in reqs:
        dt = max(0.0, r.arrival - last_t)
        last_t = r.arrival
        if dt > IDLE_GAP_S:
            idle_periods += 1
        # Drain what the server could serve during the gap, then admit this one.
        queue = max(0.0, queue - mu * dt) + 1.0
        qd = int(queue)
        active = max(1, min(qd, MAX_ACTIVE))
        # Cache-hit estimate grows with conversation depth (shared prefix reuse),
        # capped below 1.0; turn 0 is a cold prompt.
        cache_hit = min(0.9, 0.1 + 0.2 * r.turn)
        st = RuntimeState(
            avg_prompt_len=r.effective_prompt_len,
            queue_depth=qd,
            active_requests=active,
            cache_hit_rate=cache_hit,
        )
        regime = classify_regime(st)
        if regime is WorkloadRegime.BURST:
            burst_reqs += 1
        per_req.append(regime)

    # Group into control cycles by majority regime (matches benchmark_carl's
    # ROUND_SIZE cadence: the controller decides once per ROUND_SIZE requests).
    rounds: list[WorkloadRegime] = []
    for i in range(0, len(per_req), ROUND_SIZE):
        chunk = per_req[i:i + ROUND_SIZE]
        rounds.append(max(set(chunk), key=chunk.count))

    # Trace summary stats (lengths, conversation structure, regime mix).
    in_lens = [r.input_len for r in reqs]
    out_lens = [r.output_len for r in reqs]
    conv_turns: dict[int, int] = {}
    for r in reqs:
        conv_turns[r.conv_id] = max(conv_turns.get(r.conv_id, 0), r.turn + 1)
    mix: dict[str, int] = {}
    for reg in per_req:
        mix[reg.value] = mix.get(reg.value, 0) + 1
    stats = {
        "n_requests": len(reqs),
        "n_conversations": len(conv_turns),
        "input_len_mean": statistics.fmean(in_lens) if in_lens else 0.0,
        "input_len_p99": bc._percentile(in_lens, 99),
        "output_len_mean": statistics.fmean(out_lens) if out_lens else 0.0,
        "conv_len_mean": statistics.fmean(list(conv_turns.values())) if conv_turns else 0.0,
        "service_rate_rps": mu,
        "idle_periods_gt_10s": idle_periods,
        "burst_request_fraction": burst_reqs / len(reqs) if reqs else 0.0,
        "regime_mix": mix,
    }
    return rounds, per_req, stats


# ===========================================================================
# Reward + DynOracle (the static best-arm-per-regime oracle, in this codebase's
# "DynOracle" sense: best arm per regime by mean reward in hindsight).
# ===========================================================================


def round_reward(metrics: dict, slo: SLO) -> float:
    """Scalar utility for one control cycle's realised metrics.

    Mirrors CARLController._reward_for_state but over a single round's samples:
    throughput is normalized to the reference, the TTFT/TPOT violation rates are
    the fraction of this round's requests that missed each deadline, and cache is
    the round's hit rate. Uses the bandit's utility() with the default weights so
    it is the SAME objective CARL optimizes -- making oracle-capture and regret
    comparable across all baselines.
    """
    ttft = metrics["ttft_samples"]
    tpot = metrics["tpot_samples"]
    n = len(ttft) or 1
    tps_norm = min(1.0, metrics["throughput"] / slo.throughput_ref) if slo.throughput_ref > 0 else 0.0
    ttft_viol = sum(1 for t in ttft if t > slo.ttft_ms) / n
    tpot_viol = sum(1 for p in tpot if p > slo.tpot_ms) / n
    return utility({
        "throughput_norm": tps_norm,
        "ttft_violation_rate": ttft_viol,
        "tpot_violation_rate": tpot_viol,
        "cache_hit_rate": metrics["cache_hit"],
    }, DEFAULT_UTILITY_WEIGHTS)


def dynoracle_rewards(slo: SLO, arm_sets: dict[WorkloadRegime, list[CARLConfig]],
                      *, seed: int = ORACLE_SEED, rounds: int = ORACLE_ROUNDS) -> dict:
    """Best achievable mean reward per regime (the DynOracle reward target).

    For each regime, simulate every arm over `rounds` rounds and take the arm
    with the highest mean round_reward -- the best-arm-per-regime an omniscient
    operator could pick in hindsight. No online method can beat this, so it is
    the natural denominator for oracle_capture_pct and the reference for regret.
    """
    rng = random.Random(seed)
    model = bc.WorkloadModel(rng)
    out: dict[WorkloadRegime, float] = {}
    for regime, arms in arm_sets.items():
        best = float("-inf")
        for cfg in arms:
            rewards = [round_reward(model.simulate(cfg, regime, ROUND_SIZE), slo)
                       for _ in range(rounds)]
            best = max(best, statistics.fmean(rewards))
        out[regime] = best
    return out


# ===========================================================================
# Context-free baseline bandits (NEW): EpsilonGreedy + UCB1 over CARL's arms.
# ===========================================================================
#
# Both keep ONE independent context-free bandit per regime over config.config_arms
# -- the exact arm space CARL's per-regime LinUCB uses. The only thing they drop
# vs CARL is the context vector: they learn a scalar mean reward per (regime,
# arm) and pick by that, so any gap to CARL is the value of the CONTEXTUAL model.
# Arm 0 is each regime's hand-tuned default, so (like CARL) they cold-start at the
# regime oracle and only deviate once data warrants.


class PerRegimeEpsilonGreedy:
    """Per-regime epsilon-greedy (epsilon=0.1) over the CARL arm sets."""

    name = "EpsilonGreedy"

    def __init__(self, arm_sets: dict, epsilon: float = 0.1, seed: int = 0) -> None:
        self.arm_sets = arm_sets
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts = {r: [0] * len(a) for r, a in arm_sets.items()}
        self.values = {r: [0.0] * len(a) for r, a in arm_sets.items()}

    def select(self, regime: WorkloadRegime) -> int:
        arms = self.arm_sets[regime]
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(arms))
        vals = self.values[regime]
        # argmax with ties -> lowest index (== the warm-start default arm 0).
        return max(range(len(vals)), key=lambda i: vals[i])

    def update(self, regime: WorkloadRegime, arm: int, reward: float) -> None:
        self.counts[regime][arm] += 1
        n = self.counts[regime][arm]
        self.values[regime][arm] += (reward - self.values[regime][arm]) / n


class PerRegimeUCB1:
    """Per-regime context-free UCB1 over the CARL arm sets.

    Standard UCB1: pull each arm once, then pick argmax of
    mean + sqrt(2 ln(t) / n_a). Rewards are in [0, 1] (the utility range) so the
    confidence term is well-scaled without extra tuning.
    """

    name = "UCB1"

    def __init__(self, arm_sets: dict, c: float = 2.0) -> None:
        self.arm_sets = arm_sets
        self.c = c
        self.counts = {r: [0] * len(a) for r, a in arm_sets.items()}
        self.values = {r: [0.0] * len(a) for r, a in arm_sets.items()}
        self.total = {r: 0 for r in arm_sets}

    def select(self, regime: WorkloadRegime) -> int:
        counts = self.counts[regime]
        for i, c in enumerate(counts):
            if c == 0:                       # cold start: try each arm once.
                return i
        total = self.total[regime]
        vals = self.values[regime]
        ucb = [vals[i] + math.sqrt(self.c * math.log(total) / counts[i])
               for i in range(len(counts))]
        return max(range(len(ucb)), key=lambda i: ucb[i])

    def update(self, regime: WorkloadRegime, arm: int, reward: float) -> None:
        self.counts[regime][arm] += 1
        self.total[regime] += 1
        n = self.counts[regime][arm]
        self.values[regime][arm] += (reward - self.values[regime][arm]) / n


# ===========================================================================
# The per-(agent, seed) run: drive an agent over the trace's round regimes and
# record everything the checkpoint metrics need.
# ===========================================================================


def _config_sig(config: CARLConfig) -> tuple:
    """A hashable signature of a config (for arm-stability / convergence)."""
    d = config.as_dict()
    return tuple(d[k] for k in sorted(d))


def run_agent(name: str, round_regimes: list[WorkloadRegime], slo: SLO, seed: int,
              *, static_best_cfg: CARLConfig | None, oracle_rewards: dict) -> list[dict]:
    """Drive one agent over the full trace; return one record per control cycle.

    Each cycle: synthesise the observed state (carrying the previous cycle's
    realised metrics so the reward credits the previous config -- identical
    timing to benchmark_carl), classify the detected regime, let the agent pick a
    config, realise it through the WorkloadModel, score the external reward, and
    feed the context-free bandits their (immediate) reward. CARL learns through
    its own controller (delayed reward, internal); the external reward we record
    is purely the evaluation objective so all methods are scored identically.
    """
    rng = random.Random(seed)
    model = bc.WorkloadModel(rng)
    arm_sets = all_arm_sets()

    # Build the agent. CARL/Static/AutoTuner come from the shared factory; the
    # two context-free bandits are local.
    if name == "CARL-Full":
        agent = h.make_agent("CARL-Full", slo)
    elif name == "Static-Best":
        agent = h.make_agent("Static-Best", slo, static_best_cfg=static_best_cfg)
    elif name == "AutoTuner":
        agent = h.make_agent("AutoTuner", slo)
    elif name == "EpsilonGreedy":
        agent = PerRegimeEpsilonGreedy(arm_sets, epsilon=0.1, seed=seed)
    elif name == "UCB1":
        agent = PerRegimeUCB1(arm_sets)
    else:
        raise ValueError(f"unknown baseline {name!r}")

    records: list[dict] = []
    prev_metrics: dict | None = None
    for true_regime in round_regimes:
        state = bc._synth_state(true_regime, prev_metrics, rng)
        detected = classify_regime(state)

        if name in ("EpsilonGreedy", "UCB1"):
            arm = agent.select(detected)
            config = arm_sets[detected][arm]
        else:
            config = agent.choose(true_regime, state)

        metrics = model.simulate(config, true_regime, ROUND_SIZE)
        reward = round_reward(metrics, slo)

        if name == "CARL-Full":
            agent.note_realised(metrics)         # feed CARL's delayed-reward loop.
        elif name in ("EpsilonGreedy", "UCB1"):
            agent.update(detected, arm, reward)  # immediate reward (immediate in sim).

        oracle_r = oracle_rewards.get(true_regime, 0.0)
        records.append({
            "true_regime": true_regime.value,
            "detected": detected.value,
            "sig": _config_sig(config),
            "reward": reward,
            "oracle_reward": oracle_r,
            "instant_regret": max(0.0, oracle_r - reward),
            "throughput": metrics["throughput"],
            "ttft": metrics["ttft_samples"],
            "tpot": metrics["tpot_samples"],
        })
        prev_metrics = metrics
    return records


# ===========================================================================
# Convergence + checkpoint metrics over a window of records.
# ===========================================================================


def detect_convergence(records: list[dict]) -> tuple[int | None, bool]:
    """(convergence_cycle, converged_within_window) for an arm-stability settle.

    "Arm stabilizes" = the agent stops switching its chosen arm WITHIN each
    detected regime. We find, per regime, the last cycle at which that regime's
    chosen config changed; the convergence cycle is the latest such cycle across
    regimes (after it, no regime ever switches arm again). converged_within_window
    is False if that last switch is the final cycle (still moving at the end).

    Returns (None, False) for empty input; (first index, True) when nothing ever
    switched (a static policy converges immediately).
    """
    if not records:
        return None, False
    last_change = -1
    last_sig_by_regime: dict[str, tuple] = {}
    for i, r in enumerate(records):
        reg = r["detected"]
        sig = r["sig"]
        if reg in last_sig_by_regime and sig != last_sig_by_regime[reg]:
            last_change = i
        last_sig_by_regime[reg] = sig
    if last_change < 0:
        return 0, True                       # never switched -> converged at 0.
    return last_change, last_change < len(records) - 1


def checkpoint_metrics(records: list[dict], n_rounds: int, slo: SLO) -> dict:
    """All headline metrics over the first `n_rounds` control cycles.

    Throughput is the mean per-round throughput; TTFT/TPOT p99 are over every
    request in the window; slo_rate is the fraction of requests with TTFT < the
    SLO; cumulative_regret and oracle_capture_pct are summed over the window;
    convergence_cycle is the arm-stability settle point WITHIN the window; and
    steady_state_oracle_capture is the oracle capture over the cycles strictly
    AFTER convergence (post-warm-up, i.e. the "post-cycle-9" steady state the
    spec describes -- convergence here is typically an early cycle).
    """
    w = records[:n_rounds]
    if not w:
        return {}
    ttft_all: list[float] = []
    tpot_all: list[float] = []
    tps_series: list[float] = []
    reward_sum = oracle_sum = regret = 0.0
    for r in w:
        ttft_all.extend(r["ttft"])
        tpot_all.extend(r["tpot"])
        tps_series.append(r["throughput"])
        reward_sum += r["reward"]
        oracle_sum += r["oracle_reward"]
        regret += r["instant_regret"]

    n_req = len(ttft_all) or 1
    slo_rate = sum(1 for t in ttft_all if t < slo.ttft_ms) / n_req
    oc = 100.0 * reward_sum / oracle_sum if oracle_sum > 0 else 0.0

    conv_cycle, converged = detect_convergence(w)
    # Steady-state capture: cycles strictly after the convergence point.
    post = w[conv_cycle + 1:] if conv_cycle is not None else []
    post_reward = sum(r["reward"] for r in post)
    post_oracle = sum(r["oracle_reward"] for r in post)
    steady = (100.0 * post_reward / post_oracle) if post_oracle > 0 else None

    return {
        "throughput_tps": statistics.fmean(tps_series),
        "ttft_p99_ms": bc._percentile(ttft_all, 99),
        "tpot_p99_ms": bc._percentile(tpot_all, 99),
        "slo_rate": slo_rate,
        "cumulative_regret": regret,
        "oracle_capture_pct": oc,
        "convergence_cycle": float(conv_cycle) if conv_cycle is not None else None,
        "converged_within_window": converged,
        "steady_state_oracle_capture": steady,
    }


# ===========================================================================
# Cross-seed statistics: mean +/- std, bootstrap CIs, paired t-test, Cohen's d.
# ===========================================================================


def bootstrap_ci(values: list[float], *, n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> list[float]:
    """Percentile bootstrap 95% CI of the MEAN over `values` (across seeds).

    Resamples `values` with replacement `n_boot` times, takes the mean of each
    resample, and returns the [alpha/2, 1-alpha/2] percentiles of those means.
    Returns [mean, mean] for a single value (no spread to resample).
    """
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        m = vals[0] if vals else 0.0
        return [m, m]
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2.0) * n_boot)]
    hi = means[min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))]
    return [lo, hi]


def agg_metric(per_seed: list[float]) -> dict:
    """{mean, std, ci95} for one metric's per-seed values (None-tolerant)."""
    vals = [v for v in per_seed if v is not None]
    mean, std = h.mean_std(vals)
    return {"mean": mean, "std": std, "ci95": bootstrap_ci(vals),
            "n": len(vals)}


# --- Student's t (no scipy): regularised incomplete beta, copied so this file
# --- stays self-contained and imports nothing from another eval SCRIPT. -------


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    res = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        res *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        res *= delta
        if abs(delta - 1.0) < EPS:
            break
    return res


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for Student's t with `df` dof."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def paired_test(a: list[float], b: list[float]) -> dict:
    """Paired t-test + Cohen's d for a (treatment) vs b (control), same seeds.

    a and b are paired per seed (same trace window). We test H0: mean(a-b)=0 with
    a paired t-test and report Cohen's d = mean(diff)/std(diff) (the paired
    effect size). Higher t / |d| and p<0.05 => the difference is more than seed
    noise (in simulation).
    """
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean_diff": (diffs[0] if diffs else 0.0),
                "t_statistic": None, "df": n - 1, "p_value": None,
                "cohens_d": None, "significant_at_0.05": False}
    mean_d = statistics.fmean(diffs)
    std_d = statistics.stdev(diffs)
    se_d = std_d / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else float("inf")
    p = t_two_sided_p(t_stat, n - 1) if se_d > 0 else 0.0
    cohens_d = mean_d / std_d if std_d > 0 else float("inf")
    return {
        "n": n, "mean_diff": mean_d, "std_diff": std_d,
        "t_statistic": t_stat, "df": n - 1, "p_value": p,
        "cohens_d": cohens_d, "significant_at_0.05": bool(p is not None and p < 0.05),
    }


# ===========================================================================
# MODE: replay -- the main trace-replay experiment.
# ===========================================================================


def estimate_replay_runtime(round_regimes: list, slo: SLO, n_agents: int,
                            n_seeds: int) -> float:
    """Time a short CARL run and extrapolate to the full sweep (seconds)."""
    probe_rounds = round_regimes[:min(30, len(round_regimes))]
    oracle = dynoracle_rewards(slo, all_arm_sets(), rounds=20)
    t0 = time.perf_counter()
    run_agent("CARL-Full", probe_rounds, slo, 0,
              static_best_cfg=DEFAULT_CONFIGS[WorkloadRegime.INTERACTIVE],
              oracle_rewards=oracle)
    dt = time.perf_counter() - t0
    per_round = dt / max(1, len(probe_rounds))
    # CARL is the most expensive agent (per-arm matrix inverse); others are
    # cheaper, so this over-estimates a little -- which is the safe direction.
    return per_round * len(round_regimes) * n_agents * n_seeds


def run_replay(args) -> dict:
    slo = make_slo()
    n_seeds = args.seeds
    n_requests = args.requests
    checkpoints = [c for c in CHECKPOINTS if c <= n_requests]
    if not checkpoints:
        checkpoints = [n_requests]
    checkpoint_rounds = {c: c // ROUND_SIZE for c in checkpoints}

    token_count, tokenizer_mode = _make_token_counter(args.tokenizer)
    arm_sets = all_arm_sets()
    oracle_rewards = dynoracle_rewards(slo, arm_sets)

    # ---- Build the trace windows (one per seed) and their regime streams. -----
    print(f"Loading {n_seeds} trace window(s) of {n_requests} requests "
          f"({'real LMSYS' if args.real else 'LMSYS or synthetic fallback'})...",
          flush=True)
    windows: list[dict] = []
    any_fallback = False
    for seed in range(n_seeds):
        reqs, fell_back = load_window(
            seed, n_requests, force_real=args.real,
            token_count=token_count, offset=seed * n_requests)
        any_fallback = any_fallback or fell_back
        round_regimes, _per_req, tstats = derive_regimes(reqs)
        windows.append({"seed": seed, "rounds": round_regimes,
                        "fallback": fell_back, "stats": tstats})

    # ---- Runtime estimate (printed BEFORE the heavy sweep). -------------------
    est = estimate_replay_runtime(windows[0]["rounds"], slo, len(BASELINES), n_seeds)
    print(f"\nEstimated runtime for the full sweep: ~{est:.0f}s "
          f"({len(BASELINES)} methods x {n_seeds} seeds x "
          f"{len(windows[0]['rounds'])} cycles). Starting...\n", flush=True)

    # ---- Run every method over every seed, slice metrics at each checkpoint. --
    # per_seed_metrics[method][checkpoint][metric] -> list over seeds.
    per_seed: dict = {m: {c: {} for c in checkpoints} for m in BASELINES}
    # Keep per-seed throughput at the test checkpoints for the paired tests.
    tput_for_test: dict = {m: {c: [] for c in (1000, 5000) if c in checkpoints}
                           for m in BASELINES}

    for w in windows:
        seed = w["seed"]
        rounds = w["rounds"]
        # Static-Best is grid-searched on the FIRST 100 requests (= 10 cycles).
        first100 = rounds[:max(1, 100 // ROUND_SIZE)]
        static_best = h.best_static_config(first100, slo, seed=seed)

        for method in tqdm(BASELINES, desc=f"seed {seed}"):
            records = run_agent(method, rounds, slo, seed,
                                static_best_cfg=static_best,
                                oracle_rewards=oracle_rewards)
            for c in checkpoints:
                m = checkpoint_metrics(records, checkpoint_rounds[c], slo)
                for key, val in m.items():
                    per_seed[method][c].setdefault(key, []).append(val)
                if c in tput_for_test.get(method, {}):
                    tput_for_test[method][c].append(m["throughput_tps"])

    # ---- Aggregate across seeds (mean +/- std + bootstrap CI per metric). -----
    metric_keys = ["throughput_tps", "ttft_p99_ms", "tpot_p99_ms", "slo_rate",
                   "cumulative_regret", "oracle_capture_pct", "convergence_cycle",
                   "steady_state_oracle_capture"]
    checkpoints_out: dict = {}
    for c in checkpoints:
        checkpoints_out[str(c)] = {}
        for method in BASELINES:
            agg = {k: agg_metric(per_seed[method][c].get(k, [])) for k in metric_keys}
            # converged_within_window: fraction of seeds that settled.
            conv_flags = per_seed[method][c].get("converged_within_window", [])
            agg["converged_fraction"] = (sum(1 for f in conv_flags if f) / len(conv_flags)
                                         if conv_flags else 0.0)
            checkpoints_out[str(c)][method] = agg

    # ---- Statistical tests: CARL vs Static-Best and CARL vs AutoTuner. --------
    stat_tests: dict = {}
    for opponent in ("Static-Best", "AutoTuner"):
        key = f"CARL-Full_vs_{opponent}"
        stat_tests[key] = {}
        for c in (1000, 5000):
            if c not in checkpoints:
                continue
            a = tput_for_test["CARL-Full"][c]
            b = tput_for_test[opponent][c]
            stat_tests[key][str(c)] = paired_test(a, b)

    return {
        "note": ("CONTROL-LOOP SIMULATION over an LMSYS-Chat-1M-derived regime "
                 "trace. Drives the real CARL controller/bandit/AutoTuner over "
                 "benchmark_carl's analytical cost model; comparisons (not "
                 "absolute values) are the result. Oracle is near-optimal by "
                 "construction."),
        "mode": "replay",
        "settings": {
            "seeds": list(range(n_seeds)),
            "window_size_requests": n_requests,
            "checkpoints_requests": checkpoints,
            "round_size_requests": ROUND_SIZE,
            "slo": {"ttft_ms": SLO_TTFT_MS, "tpot_ms": SLO_TPOT_MS,
                    "throughput_ref": SLO_THROUGHPUT_REF},
            "baselines": BASELINES,
            "dataset": LMSYS_DATASET,
            "tokenizer": tokenizer_mode,
            "fallback_to_synthetic_used": any_fallback,
            "n_bootstrap": N_BOOTSTRAP,
            "dynoracle_reward_per_regime": {r.value: v for r, v in oracle_rewards.items()},
        },
        "trace_stats_per_seed": [{"seed": w["seed"], "fallback": w["fallback"],
                                  **w["stats"]} for w in windows],
        "checkpoints": checkpoints_out,
        "statistical_tests": stat_tests,
    }


def _print_replay_summary(results: dict) -> None:
    """A compact human-readable summary (full detail is in the JSON)."""
    cps = results["settings"]["checkpoints_requests"]
    print(h.SIM_NOTE)
    for c in cps:
        rows = []
        for method in results["settings"]["baselines"]:
            a = results["checkpoints"][str(c)][method]
            t = a["throughput_tps"]
            oc = a["oracle_capture_pct"]
            rows.append([
                method,
                h.fmt_pm(t["mean"], t["std"], 1),
                f"{a['ttft_p99_ms']['mean']:.0f}",
                f"{a['slo_rate']['mean']*100:.1f}",
                h.fmt_pm(oc["mean"], oc["std"], 1),
                f"{a['cumulative_regret']['mean']:.2f}",
            ])
        h.print_pipe_table(
            f"CHECKPOINT {c} requests",
            ["method", "tok/s", "ttftP99", "SLO%", "oracleCap%", "cumRegret"],
            rows)

    print("\n=== STATISTICAL TESTS (paired over seeds) ===")
    for key, by_ckpt in results["statistical_tests"].items():
        for c, t in by_ckpt.items():
            if t.get("p_value") is None:
                print(f"  {key} @ {c}: n={t['n']} (need >=2 seeds for a test)")
                continue
            verdict = "SIGNIFICANT" if t["significant_at_0.05"] else "n.s."
            print(f"  {key} @ {c} req: mean_diff={t['mean_diff']:+.2f} tok/s, "
                  f"t={t['t_statistic']:.2f}, p={t['p_value']:.2e}, "
                  f"d={t['cohens_d']:.2f} [{verdict}]")


# ===========================================================================
# MODE: scalability -- controller overhead vs arm-space size.
# ===========================================================================

# Knob value grids used to GROW the arm set beyond config.config_arms' 6 arms.
# We perturb the knobs that the cost model actually rewards, so larger arm sets
# are genuinely denser around each regime's optimum (not padded with junk).
_GRID = {
    "max_batch_size": [2, 4, 8, 16, 24, 32],
    "spec_k": [0, 1, 2, 4, 6, 8],
    "chunk_size": [64, 128, 256, 384, 512],
    "eviction_threshold": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "cache_affinity_weight": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}


def expand_arm_set(regime: WorkloadRegime, n: int) -> list[CARLConfig]:
    """Return `n` distinct arms for `regime`, growing config_arms deterministically.

    Leads with the regime's natural arm set (so arm 0 stays the warm-start
    default) and then walks the knob grid, appending each distinct clamped config
    until `n` arms exist. Deterministic (itertools.product order) so a given
    (regime, n) always yields the same set.
    """
    import itertools
    arms = list(config_arms(regime))
    if n <= len(arms):
        return arms[:n]
    base = DEFAULT_CONFIGS[regime]
    seen = {_config_sig(a) for a in arms}
    for mb, sk, cs, ev, caw in itertools.product(
            _GRID["max_batch_size"], _GRID["spec_k"], _GRID["chunk_size"],
            _GRID["eviction_threshold"], _GRID["cache_affinity_weight"]):
        cfg = replace(base, max_batch_size=mb, spec_k=sk, chunk_size=cs,
                      eviction_threshold=ev, cache_affinity_weight=caw).clamp()
        sig = _config_sig(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        arms.append(cfg)
        if len(arms) >= n:
            break
    return arms[:n]


class _CarlCustomArms(bc.CarlAgent):
    """CARL (LinUCB, alpha=0.5) over an arbitrary per-regime arm set.

    bc.CarlAgent hard-wires all_arm_sets(); the scalability experiment needs to
    vary the arm count, so we rebuild the same controller stack with custom arm
    sets while keeping every other piece (delayed-reward loop, reward, cadence)
    identical to the proposed method.
    """

    def __init__(self, arm_sets: dict, slo: SLO) -> None:
        self.name = "CARL-Full"
        self.metrics = MetricsTracker(window=ROUND_SIZE * 5)
        bandit = PerRegimeBandit(arm_sets, d=FEATURE_DIM,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
        self.controller = CARLController(bandit=bandit, observe_interval=1,
                                         slo=slo, metrics=self.metrics)
        self._prev_metrics = None


def measure_decision_latency(n_arms: int, *, iters: int = 5000,
                             seed: int = 0) -> float:
    """Mean wall time (ms) of ONE LinUCB select() over `n_arms` arms.

    select() inverts each arm's d x d design matrix, so its cost scales with the
    arm count -- this is the controller's per-decision overhead. We warm the
    bandit, then time `iters` select() calls on a fixed context and average.
    """
    rng = np.random.default_rng(seed)
    bandit = LinUCBBandit(n_arms=n_arms, d=FEATURE_DIM, alpha=0.5)
    context = rng.uniform(0.0, 1.0, size=FEATURE_DIM).tolist()
    for _ in range(200):                          # warm the per-arm stats.
        arm = bandit.select(context)
        bandit.update(arm, float(rng.uniform(0, 1)), context)
    ns = time.perf_counter_ns
    t0 = ns()
    for _ in range(iters):
        bandit.select(context)
    return ((ns() - t0) / iters) / 1e6            # ns/iter -> ms.


def measure_bandit_memory(arm_sets: dict) -> float:
    """Total LinUCB parameter storage (MB) across every per-regime bandit.

    Each arm holds a d x d design matrix A and a d-vector b (float64). We build
    the real PerRegimeBandit and sum the actual numpy nbytes, so the number
    reflects exactly what the controller would hold in memory at this arm count.
    """
    bandit = PerRegimeBandit(arm_sets, d=FEATURE_DIM, bandit_cls=LinUCBBandit,
                             alpha=0.5)
    total = 0
    for sub in bandit.bandits.values():
        total += sum(A.nbytes for A in sub.A) + sum(b.nbytes for b in sub.b)
    return total / (1024.0 * 1024.0)


def run_scalability(args) -> dict:
    slo = make_slo()
    arm_counts = [6, 12, 24, 48]
    n_seeds = 3
    n_requests = 500
    n_rounds = n_requests // ROUND_SIZE
    seeds = list(range(n_seeds))

    print(f"Scalability: arm counts {arm_counts} (per regime), "
          f"{n_requests} requests x {n_seeds} seeds.", flush=True)

    # Runtime estimate: time one (arm=6, seed=0) CARL run and extrapolate.
    arm_sets6 = {r: expand_arm_set(r, 6) for r in WorkloadRegime}
    oracle6 = dynoracle_rewards(slo, arm_sets6, rounds=50)
    probe_regimes, _pr, _st = derive_regimes(synthetic_window(0, n_requests))
    t0 = time.perf_counter()
    _run_carl_custom(arm_sets6, probe_regimes, slo, 0, oracle6)
    probe_dt = time.perf_counter() - t0
    est = probe_dt * len(arm_counts) * n_seeds
    print(f"Estimated runtime: ~{est:.0f}s. Starting...\n", flush=True)

    results_by_count: dict = {}
    for n_arms in arm_counts:
        arm_sets = {r: expand_arm_set(r, n_arms) for r in WorkloadRegime}
        oracle_rewards = dynoracle_rewards(slo, arm_sets, rounds=50)

        latency_ms = measure_decision_latency(n_arms)
        memory_mb = measure_bandit_memory(arm_sets)

        tput_seeds: list[float] = []
        oc_seeds: list[float] = []
        for seed in seeds:
            regimes, _pr, _st = derive_regimes(synthetic_window(seed, n_requests))
            records = _run_carl_custom(arm_sets, regimes, slo, seed, oracle_rewards)
            m = checkpoint_metrics(records, n_rounds, slo)
            tput_seeds.append(m["throughput_tps"])
            oc_seeds.append(m["oracle_capture_pct"])

        results_by_count[str(n_arms)] = {
            "arms_per_regime": n_arms,
            "total_arms": n_arms * len(WorkloadRegime),
            "decision_latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "throughput_tps": agg_metric(tput_seeds),
            "oracle_capture_pct": agg_metric(oc_seeds),
        }
        print(f"  arms/regime={n_arms:2d}: select {latency_ms*1000:.1f} us, "
              f"mem {memory_mb:.4f} MB, tok/s {agg_metric(tput_seeds)['mean']:.1f}, "
              f"oracleCap {agg_metric(oc_seeds)['mean']:.1f}%", flush=True)

    return {
        "note": ("CONTROL-LOOP SIMULATION. Measures CARL controller overhead "
                 "(LinUCB select latency + parameter memory) and policy quality "
                 "vs per-regime arm-space size. select() inverts a d x d matrix "
                 "per arm, so latency scales with arm count; memory is exactly "
                 "n_arms * (d^2 + d) float64 per regime."),
        "mode": "scalability",
        "settings": {
            "arm_counts_per_regime": arm_counts,
            "seeds": seeds,
            "requests": n_requests,
            "round_size_requests": ROUND_SIZE,
            "feature_dim": FEATURE_DIM,
            "n_regimes": len(WorkloadRegime),
            "slo": {"ttft_ms": SLO_TTFT_MS, "tpot_ms": SLO_TPOT_MS,
                    "throughput_ref": SLO_THROUGHPUT_REF},
        },
        "by_arm_count": results_by_count,
    }


def _run_carl_custom(arm_sets: dict, round_regimes: list, slo: SLO, seed: int,
                     oracle_rewards: dict) -> list[dict]:
    """run_agent's CARL path but with a custom-arm CARL agent (for scalability)."""
    rng = random.Random(seed)
    model = bc.WorkloadModel(rng)
    agent = _CarlCustomArms(arm_sets, slo)
    records: list[dict] = []
    prev_metrics: dict | None = None
    for true_regime in round_regimes:
        state = bc._synth_state(true_regime, prev_metrics, rng)
        detected = classify_regime(state)
        config = agent.choose(true_regime, state)
        metrics = model.simulate(config, true_regime, ROUND_SIZE)
        reward = round_reward(metrics, slo)
        agent.note_realised(metrics)
        oracle_r = oracle_rewards.get(true_regime, 0.0)
        records.append({
            "true_regime": true_regime.value, "detected": detected.value,
            "sig": _config_sig(config), "reward": reward,
            "oracle_reward": oracle_r,
            "instant_regret": max(0.0, oracle_r - reward),
            "throughput": metrics["throughput"],
            "ttft": metrics["ttft_samples"], "tpot": metrics["tpot_samples"],
        })
        prev_metrics = metrics
    return records


# ===========================================================================
# Entry point.
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LMSYS-Chat-1M trace-replay evaluation for CARL (simulation).")
    parser.add_argument("--mode", choices=["replay", "scalability"],
                        default="replay")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="number of trace windows (replay mode)")
    parser.add_argument("--requests", type=int, default=WINDOW_SIZE,
                        help="requests per window (replay mode)")
    parser.add_argument("--real", action="store_true",
                        help="force the gated LMSYS-Chat-1M load (else: try it, "
                             "fall back to synthetic on failure)")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer for token counts (default vicuna; "
                             "falls back to a chars/4 heuristic if unavailable)")
    parser.add_argument("--out", default=None, help="override output JSON path")
    args = parser.parse_args()

    if args.mode == "scalability":
        results = run_scalability(args)
        out_path = args.out or SCALABILITY_OUT
    else:
        results = run_replay(args)
        _print_replay_summary(results)
        out_path = args.out or REPLAY_OUT

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.mode} results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
