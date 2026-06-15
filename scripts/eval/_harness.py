"""
Shared evaluation harness for the CARL paper's evaluation suite (scripts/eval/).

WHAT THIS IS (read before trusting any number any eval script prints)
--------------------------------------------------------------------
Every script in scripts/eval/ is a CONTROL-LOOP SIMULATION, exactly like
scripts/benchmark_carl.py's default mode -- and this module reuses that file's
machinery so there is a SINGLE source of truth for the cost model. It drives the
REAL CARL controller, the REAL LinUCB/Thompson bandits, the REAL AutoTuner over
the REAL workload-regime classifier; only the serving metrics (throughput, TTFT,
TPOT, cache/spec rates) come from an analytical cost model (benchmark_carl's
WorkloadModel), not from running TinyLlama on a GPU.

Why a simulation and not real inference: the ablations remove individual
adaptive SUBSYSTEMS (scheduler / speculation / cache / router / chunking). The
real-inference harness (src/carl/live.py) only ever wires the controller to the
scheduler -- it cannot turn the router/cache/spec subsystems on and off as
adaptive knobs -- so it physically cannot express these ablations. The honest,
reproducible substrate for "what does each subsystem contribute to the
controller's decisions" is therefore the simulation, which isolates the policy
and runs deterministically on any CPU.

HONEST LIMITATION (inherited verbatim from benchmark_carl.py): the cost model
encodes the same domain knowledge as the hand-tuned DEFAULT_CONFIGS, so the
regime oracle is near-optimal BY CONSTRUCTION. CARL's job is to LEARN online to
APPROACH that known-good target without being told the regime. We do NOT claim a
measured wall-clock speedup on real hardware. "Oracle gap" therefore measures
how closely CARL's online learning tracks a known-good policy IN SIMULATION, not
a hardware result. See docs/eval/README.md.

This module exposes:
  * run_once()        -- drive one agent over a per-request regime sequence and
                         return a flat metrics dict (+ per-request series).
  * make_agent()      -- the agent factory for the 9 ablation configurations.
  * mean_std(), aggregate_runs() -- multi-seed statistics.
  * regime sequence builders + best_static_config() helper.
  * SIM_NOTE          -- the honesty banner every script prints.
Torch-free: it never imports torch, so it runs anywhere benchmark_carl does.
"""
from __future__ import annotations

import os
import statistics
import sys

# --- path bootstrap ---------------------------------------------------------
# Make `python scripts/eval/<x>.py` work standalone: put the repo root on the
# path (so `import src...` resolves) AND the scripts/ dir (so we can reuse
# benchmark_carl.py, which is a sibling module, as the cost-model source of
# truth). Both are idempotent no-ops when the caller already set PYTHONPATH.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # scripts/eval
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)                    # scripts
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)                   # repo root
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse benchmark_carl's proven simulation primitives rather than reimplementing
# (and risking divergence from) the cost model the paper already references.
import benchmark_carl as bc  # noqa: E402
from src.carl.bandit import LinUCBBandit, ThompsonSamplingBandit  # noqa: E402
from src.carl.config import CARLConfig, DEFAULT_CONFIGS  # noqa: E402
from src.carl.controller import SLO  # noqa: E402
from src.carl.state import WorkloadRegime, classify_regime  # noqa: E402

# tqdm if present, else a no-op shim so the scripts run without the dependency.
try:
    from tqdm import tqdm  # noqa: E402
except ImportError:  # pragma: no cover - exercised only when tqdm is absent
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []


# The banner every eval script prints up top, so a reader can never mistake a
# simulation table for a measured-hardware table.
SIM_NOTE = (
    "NOTE: CONTROL-LOOP SIMULATION (no GPU / no model execution). Drives the real\n"
    "CARL controller, bandits, AutoTuner and regime classifier over an analytical\n"
    "serving cost model (scripts/benchmark_carl.py WorkloadModel). COMPARISONS\n"
    "between methods are the result, not absolute values. The regime oracle is\n"
    "near-optimal BY CONSTRUCTION, so 'oracle gap' measures how well CARL LEARNS\n"
    "to approach a known-good target online -- not a measured hardware speedup.\n"
)

# TTFT-only SLO used across the suite: a request is "satisfied" when TTFT < 200ms
# (per the eval spec). We bind only TTFT by pushing the TPOT deadline to +inf, so
# run_once's joint (TTFT and TPOT) satisfaction check collapses to TTFT-only.
SLO_TTFT_MS = 200.0


def slo_ttft_only(ttft_ms: float = SLO_TTFT_MS) -> SLO:
    """An SLO whose only binding deadline is TTFT (TPOT/throughput unbounded)."""
    return SLO(ttft_ms=ttft_ms, tpot_ms=float("inf"), throughput_ref=50.0)


# ---------------------------------------------------------------------------
# Per-request scenario runner.
# ---------------------------------------------------------------------------
#
# benchmark_carl.run_scenario works in fixed 10-request "rounds"; the eval spec
# needs exact request counts (30 / 15 / 20 / 50) and per-REQUEST series (for
# phase slicing and adaptation-lag-in-requests). So we run at request
# granularity here: one control cycle per request, reusing benchmark_carl's
# WorkloadModel and _synth_state so the cost model is identical.
# ---------------------------------------------------------------------------

# Metric keys run_once reports as scalars (everything aggregate-able across seeds).
METRIC_KEYS = [
    "throughput", "ttft_p50", "ttft_p95", "ttft_p99",
    "tpot_p50", "tpot_p95", "tpot_p99",
    "slo_sat", "cache_hit", "spec_acc", "adaptations", "regime_acc",
]


def run_once(agent, regimes: list, slo: SLO, seed: int) -> dict:
    """Drive `agent` over a per-request `regimes` sequence; return a metrics dict.

    Each request: synthesise the observed RuntimeState for that request's TRUE
    regime (carrying the previous request's realised metrics so the controller's
    reward credits the previous config), ask the agent for a config, realise that
    config's metrics via the WorkloadModel (one request), and accumulate.

    Returns the scalar metrics in METRIC_KEYS plus, under underscore keys, the
    per-request series needed by oracle_comparison (throughput, TTFT, detected
    regime), each aligned 1:1 with `regimes`.
    """
    import random

    rng = random.Random(seed)
    model = bc.WorkloadModel(rng)

    ttft_all: list[float] = []
    tpot_all: list[float] = []
    tps_series: list[float] = []
    cache_all: list[float] = []
    spec_all: list[float] = []
    detected: list[WorkloadRegime] = []
    regime_correct = 0
    prev_metrics: dict | None = None

    for true_regime in regimes:
        state = bc._synth_state(true_regime, prev_metrics, rng)
        det = classify_regime(state)
        detected.append(det)
        if det is true_regime:
            regime_correct += 1

        config = agent.choose(true_regime, state)
        metrics = model.simulate(config, true_regime, 1)   # one request
        if isinstance(agent, bc.CarlAgent):
            agent.note_realised(metrics)

        ttft_all.extend(metrics["ttft_samples"])
        tpot_all.extend(metrics["tpot_samples"])
        tps_series.append(metrics["throughput"])
        cache_all.append(metrics["cache_hit"])
        spec_all.append(metrics["spec_acc"])
        prev_metrics = metrics

    n = len(ttft_all)
    # TTFT-only SLO satisfaction (slo.tpot_ms is +inf for slo_ttft_only).
    satisfied = sum(1 for t, p in zip(ttft_all, tpot_all)
                    if t <= slo.ttft_ms and p <= slo.tpot_ms)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    pct = bc._percentile

    return {
        "agent": agent.name,
        "throughput": mean(tps_series),
        "ttft_p50": pct(ttft_all, 50),
        "ttft_p95": pct(ttft_all, 95),
        "ttft_p99": pct(ttft_all, 99),
        "tpot_p50": pct(tpot_all, 50),
        "tpot_p95": pct(tpot_all, 95),
        "tpot_p99": pct(tpot_all, 99),
        "slo_sat": 100.0 * satisfied / n if n else 0.0,
        "cache_hit": mean(cache_all),
        "spec_acc": mean(spec_all),
        "adaptations": float(getattr(agent, "adaptations", 0)),
        "regime_acc": 100.0 * regime_correct / len(regimes) if regimes else 0.0,
        # Per-request series (underscore = not aggregated as a headline metric).
        "_tps_series": tps_series,
        "_ttft_series": ttft_all,
        "_detected": detected,
    }


# ---------------------------------------------------------------------------
# Multi-seed statistics.
# ---------------------------------------------------------------------------


def mean_std(values: list[float]) -> tuple[float, float]:
    """(mean, sample-std). std is 0.0 for a single sample (stdev needs n>=2)."""
    if not values:
        return 0.0, 0.0
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def aggregate_runs(run_dicts: list[dict], keys: list[str] = METRIC_KEYS) -> dict:
    """Collapse N per-seed run dicts into {key_mean, key_std} for each metric."""
    out: dict = {}
    for k in keys:
        m, s = mean_std([r[k] for r in run_dicts])
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s
    return out


# ---------------------------------------------------------------------------
# The agent factory: the 9 ablation configurations + the extras.
# ---------------------------------------------------------------------------
#
# Each name maps a per-request observed state -> the config to run. The five
# "CARL-NoX" ablations FREEZE one knob group at the global default while the rest
# stay bandit-adaptive, isolating that subsystem's marginal contribution. The
# freeze targets map onto CARLConfig fields exactly:
#     NoSched  -> max_batch_size + chunk_size pinned to defaults (8 / 256)
#     NoSpec   -> spec_k = 0          (speculation disabled)
#     NoCache  -> eviction_threshold = 0.8
#     NoRouter -> routing_threshold = 0.5
#     NoChunk  -> chunk_size = 256
# Fresh objects every call: CarlAgent/AutoTunerAgent carry per-run learning state
# that MUST be reset between seeds, so never reuse an agent across run_once calls.
# ---------------------------------------------------------------------------

_DEF = CARLConfig()   # the global defaults the ablations freeze toward.

_ABLATION_FREEZE = {
    "CARL-NoSched": dict(max_batch_size=_DEF.max_batch_size, chunk_size=_DEF.chunk_size),
    "CARL-NoSpec": dict(spec_k=0),
    "CARL-NoCache": dict(eviction_threshold=0.8),
    "CARL-NoRouter": dict(routing_threshold=0.5),
    "CARL-NoChunk": dict(chunk_size=256),
}

# Names this factory understands (used by scripts to drive their config lists).
ABLATION_CONFIGS = [
    "CARL-Full", "CARL-NoSched", "CARL-NoSpec", "CARL-NoCache",
    "CARL-NoRouter", "CARL-NoChunk", "Static-Best", "AutoTuner", "Oracle",
]


def make_agent(name: str, slo: SLO, *,
               static_best_cfg: CARLConfig | None = None,
               thompson_seed: int = 0):
    """Build a FRESH agent for `name`. See _ABLATION_FREEZE for the freeze map.

    static_best_cfg must be supplied for "Static-Best" (resolve it once per
    scenario with best_static_config); thompson_seed seeds "carl_thompson".
    """
    # Accept lowercase aliases so scripts can use the spec's method labels.
    name = {"oracle": "Oracle", "autotuner": "AutoTuner"}.get(name, name)
    if name == "CARL-Full":
        return bc.CarlAgent(LinUCBBandit, "CARL-Full", slo, alpha=0.5)
    if name in _ABLATION_FREEZE:
        return bc._FrozenCarlAgent(_ABLATION_FREEZE[name], name, slo)
    if name == "Static-Best":
        if static_best_cfg is None:
            raise ValueError("Static-Best needs static_best_cfg (use best_static_config)")
        return bc.StaticAgent(static_best_cfg, "Static-Best")
    if name == "static_default":
        return bc.StaticAgent(CARLConfig(), "static_default")
    if name == "AutoTuner":
        return bc.AutoTunerAgent()
    if name == "Oracle":
        return bc.OracleAgent()
    if name == "carl_linucb":
        return bc.CarlAgent(LinUCBBandit, "carl_linucb", slo, alpha=0.5)
    if name == "carl_thompson":
        return bc.CarlAgent(ThompsonSamplingBandit, "carl_thompson", slo,
                            v=0.5, seed=thompson_seed)
    raise ValueError(f"unknown agent name: {name!r}")


def best_static_config(regimes: list, slo: SLO, seed: int = 0) -> CARLConfig:
    """The single best STATIC config for a workload (the 'Static-Best' baseline).

    'Hand-tuned best static config per workload' = the one fixed config (no
    adaptation) that maximises mean throughput on this regime sequence. We pick
    it from the natural candidate set -- the global default plus each regime's
    hand-tuned DEFAULT_CONFIG -- by actually running each as a StaticAgent once.
    Resolved on a fixed seed so the chosen config is stable across the multi-seed
    evaluation that follows.
    """
    candidates = [CARLConfig()] + [DEFAULT_CONFIGS[r] for r in WorkloadRegime]
    best_cfg, best_tps = candidates[0], float("-inf")
    for cfg in candidates:
        m = run_once(bc.StaticAgent(cfg, "probe"), regimes, slo, seed)
        if m["throughput"] > best_tps:
            best_tps, best_cfg = m["throughput"], cfg
    return best_cfg


# ---------------------------------------------------------------------------
# Per-request regime-sequence builders.
# ---------------------------------------------------------------------------
#
# A "scenario" here is a list of per-request TRUE regimes. run_once consumes one
# regime per request. These builders centralise the regime patterns the eval
# spec describes so every script agrees on what (e.g.) "NON-STATIONARY" means.
# ---------------------------------------------------------------------------

R = WorkloadRegime


def interactive(n: int) -> list:
    return [R.INTERACTIVE] * n


def batch(n: int) -> list:
    return [R.BATCH] * n


def nonstationary(a: int, b: int) -> list:
    """`a` INTERACTIVE requests, then `b` BATCH -- a regime flip with no notice."""
    return [R.INTERACTIVE] * a + [R.BATCH] * b


def phased(*segments: tuple) -> list:
    """Concatenate (regime, count) segments into one per-request sequence."""
    out: list = []
    for regime, count in segments:
        out += [regime] * count
    return out


def regimes_from_prompt_lengths(lengths: list) -> list:
    """Classify each prompt length into its regime via the REAL classifier.

    Builds a RuntimeState whose avg_prompt_len is the given length (queue/cache
    held at light interactive levels) and runs classify_regime, so the regime mix
    a workload produces is decided by the same rule-based classifier the live
    engine uses -- not hand-assigned. This is the honest length->regime mapping.
    """
    from src.carl.state import RuntimeState
    out = []
    for L in lengths:
        st = RuntimeState(avg_prompt_len=float(L), queue_depth=4,
                          active_requests=3, cache_hit_rate=0.1)
        out.append(classify_regime(st))
    return out


def boundaries_of(segments: list) -> list:
    """Cumulative start index of each segment after the first (phase boundaries).

    segments is a list of (regime, count); returns the request index at which
    each new segment begins, used to measure adaptation lag at each transition.
    """
    idx, bounds = 0, []
    for _regime, count in segments:
        if idx > 0:
            bounds.append(idx)
        idx += count
    return bounds


def adaptation_lag(detected: list, boundary: int, new_regime) -> int:
    """Requests after `boundary` until `detected` first reads `new_regime`.

    0 == detected the change on the very first request of the new phase. Capped
    at the remaining length if never detected.
    """
    lag = 0
    while boundary + lag < len(detected) and detected[boundary + lag] is not new_regime:
        lag += 1
    return lag


# ---------------------------------------------------------------------------
# Output helpers (shared so every script prints the same pipe-table format that
# docs/run_benchmarks.ipynb's to_md_table() can turn into GitHub markdown).
# ---------------------------------------------------------------------------


def fmt_pm(mean: float, std: float, prec: int = 1) -> str:
    """Format a 'mean +/- std' cell. ASCII '+/-' (not the U+00B1 sign) so the
    tables never hit a UnicodeEncodeError on a cp1252 Windows console."""
    return f"{mean:.{prec}f} +/- {std:.{prec}f}"


def print_pipe_table(title: str, headers: list, rows: list) -> None:
    """Print a pipe-delimited table (header row, separator, then string rows)."""
    if title:
        print(f"\n=== {title} ===")
    print("| " + " | ".join(str(h) for h in headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def eval_docs_dir():
    """docs/eval/ under the repo root, created on demand. Returns a Path."""
    from pathlib import Path
    d = Path(_REPO_ROOT) / "docs" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d
