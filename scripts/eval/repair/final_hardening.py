"""THE FINAL HARDENING EXPERIMENT -- fix the objective mismatch and the cold start.

WHY THIS EXPERIMENT EXISTS
==========================
`scripts/eval/repair/reward_v2_live.py` restored a non-degenerate reward to the
live controller. Per-cycle forensic inspection of its artifact then found two
defects that make its headline comparison uninterpretable. Neither is a coding
slip; both are errors in what was being compared.

DEFECT 1 -- OBJECTIVE MISMATCH
------------------------------
CARL maximises `utility_v2`. Its Static-Best baseline was selected by
`ablation_live.select_static_best`, which ranks candidates on THROUGHPUT alone.
So the reported comparison scored an agent optimising one objective against a
baseline tuned for a different one.

That would be a small worry if the two objectives agreed. They do not, and the
experiment's own calibration sweep is the evidence: on the held-out seed,
`max_batch_size=32` maximises throughput while `max_batch_size=12` maximises the
utility axis. Under an objective mismatch, "CARL ties the static baseline on
throughput" and "CARL beats the static baseline on utility" are both artefacts
of which yardstick each side was tuned to, and neither is an adaptive-control
result.

The fix is TWO static baselines, both selected on the same held-out seed 999,
from the same exhaustive sweep, differing only in the ranking function:

    Static-Tput      argmax throughput_tps
    Static-Utility   argmax utility_v2, under the SAME frozen scales CARL uses

Now every comparison has a baseline tuned for the metric it is judged on.

DEFECT 2 -- COLD-START MISSING-METRIC REWARD
--------------------------------------------
In EVERY seed of the reward-v2 artifact, the first three control cycles logged

    ttft_p99_ms = 0        tpot_p99_ms = 0

not because latency was zero but because no request had completed, so
`MetricsTracker`'s windows were empty and `state._percentile` returned its
NaN-free 0.0. Reward v2's `latency_term` maps 0 ms to 1.0 -- PERFECT latency --
so those cycles collected the full TTFT and TPOT weight (0.5 of the maximum) for
latencies that had never been measured, and handed the result to LinUCB. LinUCB
credits a reward to the PREVIOUS cycle's arm, and the first arm played in every
regime is arm 0 by construction, so the bias has a direction: it inflates the
hand-tuned default before any evidence exists.

The repair is semantic and lives in the library, not in this script:

    src/carl/state.py       MetricsTracker.metric_validity() -- explicit
                            per-metric sample counts. A latency of exactly 0.0 ms
                            is counted as a SENTINEL, not an observation.
    src/carl/live_reward.py LiveRewardV2 returns (None, terms) with
                            reward_valid=False while any consumed metric is
                            unmeasured. No fill value is substituted: zero
                            fabricates a perfect measurement, a penalty
                            fabricates a bad one.
    src/carl/controller.py  a None reward updates NOTHING. The cycle is logged
                            with reward_valid=False and the previous arm goes
                            uncredited; `reward_validity_report()` reports the
                            warm-up length and the first scored cycle per run.

Learning therefore begins only once a genuine measured TTFT/TPOT window exists,
and every run states when that was.

SEARCH ONLY WHAT ACTUALLY EXECUTES
==================================
`reward_v2_live`'s static search sampled 16 Latin-hypercube points from a 5-D
space: max_batch_size, chunk_size, spec_k, routing_threshold, eviction_threshold.
THREE OF THOSE FIVE CANNOT CHANGE ANYTHING IN THIS HARNESS, so the search was
narrower than it appeared while looking wider:

    spec_k              `ablation_live._apply_sched` pins enable_spec_decode
                        False, and `_serve` re-pins it False after every
                        controller step.
    routing_threshold   no router exists; CARLController.router is None and
                        `_set` drops the write.
    eviction_threshold  no KV cache object is wired; same.

Two more config fields are inert for the same class of reason:

    preemption_enabled  ContinuousBatchScheduler has no such attribute, so
                        `_set` skips it.
    use_cuda_graphs     the scheduler HAS the flag, but `_decode_forward` also
                        requires `_graph_runner is not None` and
                        `_new_scheduler` never attaches one -- every decode
                        falls back to eager with reason `graph_runner_missing`.
                        (This is the same trap that produced two "CUDA graph"
                        arms with cuda_graph_hits == 0 on an earlier T4 smoke.)

`probe_live_dimensions()` VERIFIES all of this against the real scheduler object
at run time and writes the evidence into the artifact, so the claim is measured
rather than asserted. Anything it finds live and unsearched ABORTS the run.

What is left is `max_batch_size x chunk_size`, and this experiment sweeps that
plane EXHAUSTIVELY -- 7 x 4 = 28 configurations -- rather than sampling it. At
this size exhaustive is both cheaper to defend and not much dearer to run than
16 LHS points, and it removes "the static baseline got unlucky in the sample" as
an explanation for any result.

ONE HELD-OUT SWEEP, THREE USES
==============================
The same 28-configuration sweep on seed 999, at the SAME 200 requests as
evaluation, produces:

    1. the FROZEN RewardScales (medians of the observed operating range);
    2. Static-Tput   (argmax throughput);
    3. Static-Utility(argmax utility_v2 under those frozen scales).

Deriving scales from evaluation data would let the yardstick move with what it
measures; seed 999 is documented in `ablation_live` as never used for an
evaluation run. Calibrating at the evaluation's own request count matters
because under burst arrivals the TTFT p99 scales with how many requests are
queued.

Ordering note: the scales are frozen from the sweep's MEASUREMENTS before any
utility is computed, so Static-Utility is selected under scales it could not
have influenced.

COMPARABILITY OF THE UTILITY NUMBERS
====================================
Two different estimators of utility appear, and they are NOT interchangeable:

  run_utility_v2   utility_v2 of a run's END-OF-RUN aggregates, under the frozen
                   scales. Defined for EVERY config including the statics, and
                   the basis of comparison B. For a static config -- constant for
                   the whole run -- this simply IS its utility.
  mean_valid_cycle_utility
                   mean over a controller's SCORED control cycles. Reported for
                   the three controller configs because it is what the learner
                   actually consumed, and NOT compared against the statics,
                   which have no control cycles.

Both are reported. Neither is silently substituted for the other.

THE FIVE CONFIGS
================
    Static-Tput      fixed config, argmax throughput on seed 999
    Static-Utility   fixed config, argmax utility_v2 on seed 999
    RuleOnly         classify -> DEFAULT_CONFIGS[regime]; no learning
    CARL-Repaired    RepairedLinUCB + restricted arms
    CARL-Expanded    RepairedLinUCB + expanded arms

COMPARISONS
===========
    A  CARL-Repaired  vs Static-Tput      throughput
    B  CARL-Repaired  vs Static-Utility   utility (run-level)
    C  CARL-Repaired  vs RuleOnly         learning, arm set + classifier fixed
    D  CARL-Expanded  vs CARL-Repaired    action space, learner fixed

HELD FIXED against the reward-v2 run, by importing its code rather than
reimplementing it: Tesla T4, TinyLlama-1.1B fp16, seeds 42..51, 200 requests,
the two-phase NON-STATIONARY workload, OBSERVE_INTERVAL=10, validation seed 999,
the restricted/expanded arm-set definitions, RepairedLinUCBBandit at alpha=0.5,
and FEATURE_DIM=10.

WHAT THIS SCRIPT WILL NOT DO
============================
It will not claim convergence from a mean. `convergence_diagnostic()` requires
the trace to show exploration FOLLOWED BY exploitation, by a rule fixed before
the run, and reports `supported: false` when it does not.

It writes to `docs/eval/final_hardening_results.json` and
`docs/eval/raw/final_hardening/`. It never reads or writes the reward-v2
artifact, which is preserved as the experiment that exposed both defects.

Run:
    python scripts/eval/repair/source_provenance.py --archive <the uploaded zip>
    python scripts/eval/repair/final_hardening.py
    python scripts/eval/repair/final_hardening.py --seeds 42 --limit 20 \\
        --grid-batches 4,8 --grid-chunks 128,256 --allow-cpu-smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_EVAL_DIR = os.path.join(_REPO_ROOT, "scripts", "eval")
for _p in (_REPO_ROOT, _EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# The reward-v2 run's own harness, which is Run A's harness. Imported, never
# reimplemented -- that is what keeps every "held fixed" factor actually fixed.
import ablation_live as abl  # noqa: E402

from src.carl.bandit import PerRegimeBandit, RepairedLinUCBBandit  # noqa: E402
from src.carl.config import CARLConfig  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.live_reward import (  # noqa: E402
    MIN_ARM_REWARD_SPREAD, MIN_LIVE_DISTINCT, MIN_LIVE_REWARD_SPREAD,
    THRESHOLD_PROVENANCE, LiveRewardV2, gate_candidate_arm_spread,
    gate_live_reward_stream,
)
from src.carl.reward import (  # noqa: E402
    DEFAULT_WEIGHTS, DegenerateRewardError, RewardScales, utility_v2,
)
from src.carl.rule_only import RuleOnlyBandit  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
from src.eval import provenance  # noqa: E402

EXPERIMENT = "final_hardening"
DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "final_hardening")
RESULTS_PATH = os.path.join(DOCS_EVAL, "final_hardening_results.json")
PREREG_PATH = os.path.join(DOCS_EVAL, "PREREGISTRATION_final_hardening.md")
SOURCE_PROVENANCE_PATH = os.path.join(DOCS_EVAL, "SOURCE_PROVENANCE.json")

# Held fixed against the reward-v2 run. Read from ablation_live so they cannot
# drift: if that module changes, this experiment changes with it or not at all.
DEFAULT_SEEDS = list(abl.DEFAULT_SEEDS)                 # 42..51
DEFAULT_REQUESTS = 200
VALIDATION_SEED = abl.VALIDATION_SEED                   # 999, held out
OBSERVE_INTERVAL = abl.OBSERVE_INTERVAL                 # 10
ALPHA = 0.5

# THE LIVE PLANE. Exhaustive, not sampled. Both axes come from the harness's own
# definitions rather than being retyped here.
GRID_BATCH_AXIS = list(abl.ARM_SET_BATCH_AXIS)                    # 2..32
GRID_CHUNK_AXIS = list(abl.SEARCH_SPACE_WIDE["chunk_size"])       # 64..512

# CARLConfig fields that provably cannot change execution in this harness. Each
# is verified against the live scheduler by probe_live_dimensions(); the string
# is the reason a reader gets in the artifact.
INERT_DIMENSIONS = {
    "spec_k": ("ablation_live._apply_sched pins enable_spec_decode=False and "
               "_serve re-pins it False after every controller step"),
    "routing_threshold": "no router is wired; CARLController._set drops the write",
    "cache_affinity_weight": "no router is wired; CARLController._set drops the write",
    "eviction_threshold": "no KV cache is wired; CARLController._set drops the write",
    "eviction_window": "no KV cache is wired; CARLController._set drops the write",
    "preemption_enabled": ("ContinuousBatchScheduler declares no such attribute, "
                           "so CARLController._set skips it"),
    "use_cuda_graphs": ("the scheduler has the flag, but _decode_forward also "
                        "requires _graph_runner is not None and _new_scheduler "
                        "never attaches one -- every decode falls back to eager "
                        "with reason 'graph_runner_missing'"),
}
LIVE_DIMENSIONS = ("max_batch_size", "chunk_size")

CONFIGS = ["Static-Tput", "Static-Utility", "RuleOnly", "CARL-Repaired",
           "CARL-Expanded"]
_STATIC_CONFIGS = {"Static-Tput", "Static-Utility"}
_CONTROLLER_CONFIGS = {"RuleOnly", "CARL-Repaired", "CARL-Expanded"}
_ARM_SET_FOR = {"CARL-Repaired": "restricted", "CARL-Expanded": "expanded",
                "RuleOnly": "restricted"}

COMPARISONS = [
    ("A_adaptation_vs_throughput_baseline", "CARL-Repaired", "Static-Tput",
     "throughput_tps",
     "Does adaptation beat the best fixed config FOR THROUGHPUT, on the metric "
     "that baseline was selected for?"),
    ("B_adaptation_vs_utility_baseline", "CARL-Repaired", "Static-Utility",
     "run_utility_v2",
     "Does adaptation beat the best fixed config FOR CARL'S OWN OBJECTIVE, on "
     "that objective? This is the comparison the objective mismatch destroyed."),
    ("C_learning_isolation", "CARL-Repaired", "RuleOnly", "throughput_tps",
     "What did LEARNING buy, holding the classifier and the arm set fixed?"),
    ("D_action_space_isolation", "CARL-Expanded", "CARL-Repaired", "throughput_tps",
     "What did a WIDER ACTION SPACE buy, holding the learner fixed?"),
]

PRE_REGISTRATION = {
    "registered_before_run": True,
    "document": "docs/eval/PREREGISTRATION_final_hardening.md",
    "H1": {
        "statement": (
            "CARL-Repaired will not materially outperform Static-Tput on "
            "THROUGHPUT. The substrate is saturated at this operating point and "
            "the exhaustive held-out sweep is expected to select the largest "
            "batch the axis offers."),
        "comparison": "A", "metric": "throughput_tps",
        "prediction": "paired mean difference <= 0, or a 95% CI containing 0",
        "falsified_if": ("CARL-Repaired exceeds Static-Tput throughput by more "
                         "than 2% with a paired 95% CI excluding 0"),
    },
    "H2": {
        "statement": (
            "CARL-Repaired will not materially outperform Static-Utility on "
            "UTILITY either. This is the hypothesis the previous experiment "
            "could not test, because its only static baseline was tuned for a "
            "different objective. If CARL's apparent utility advantage was an "
            "artefact of that mismatch, it disappears here."),
        "comparison": "B", "metric": "run_utility_v2",
        "prediction": "paired mean difference <= 0, or a 95% CI containing 0",
        "falsified_if": ("CARL-Repaired exceeds Static-Utility run utility with "
                         "a paired 95% CI excluding 0"),
    },
    "H3": {
        "statement": (
            "With cold-start rewards withheld, CARL-Repaired remains "
            "BEHAVIOURALLY distinct from RuleOnly -- it still moves between "
            "arms -- but the distinction does not convert into a throughput "
            "gain. H3 is about whether learning DOES anything, not whether it "
            "HELPS."),
        "comparison": "C", "metric": "throughput_tps",
        "prediction": ("arm-change rate strictly greater than RuleOnly's 0; "
                       "throughput difference of unspecified sign"),
        "falsified_if": ("CARL-Repaired is bit-identical to RuleOnly on every "
                         "seed, which would mean the learner is inert once the "
                         "cold-start rewards it was consuming are withheld"),
    },
    "H4": {
        "statement": ("CARL-Expanded will not recover enough benefit to offset "
                      "the extra exploration a larger arm set costs."),
        "comparison": "D", "metric": "throughput_tps",
        "prediction": "paired mean difference <= 0",
        "falsified_if": ("CARL-Expanded exceeds CARL-Repaired with a paired 95% "
                         "CI excluding 0"),
    },
    "H5_cold_start": {
        "statement": (
            "Every seed will show a non-empty warm-up: at least one leading "
            "control cycle whose reward is withheld for want of a measured "
            "TTFT/TPOT. This is a POSITIVE CONTROL on the repair -- a warm-up "
            "of zero everywhere would mean the validity gate never engaged and "
            "the run says nothing about the defect it was built to fix."),
        "metric": "reward_validity.warmup_cycles",
        "prediction": "warmup_cycles >= 1 for every controller run",
        "falsified_if": "any controller run reports warmup_cycles == 0",
    },
    "note": (
        "H1, H2 and H4 predict null or negative results; H3 predicts a "
        "behavioural difference of unspecified sign; H5 predicts the repair "
        "engages. Registering them before the run is what makes a null result "
        "evidence rather than an absence of evidence."),
}

# The convergence rule, fixed HERE so it cannot be chosen after seeing the
# traces. See convergence_diagnostic().
CONVERGENCE_RULE = {
    "min_decisions_in_regime": 6,
    "min_first_half_distinct_arms": 2,
    "min_second_half_modal_share": 0.75,
    "requires_increase_in_modal_share": True,
    "statement": (
        "A regime's trace shows convergence only if it EXPLORED and then "
        "EXPLOITED: at least 6 decisions, at least 2 distinct arms in the first "
        "half, a second-half modal-arm share of at least 0.75, and a modal share "
        "that strictly increased from the first half to the second. A high modal "
        "share with no exploration is a learner that never moved, which is the "
        "failure the as-published LinUCB already exhibited (200/200 on arm 0) "
        "and must not be reported as convergence."),
}


# ===========================================================================
# Liveness probe -- prove which dimensions can change execution.
# ===========================================================================


def probe_live_dimensions(sched) -> dict:
    """Check, against the REAL scheduler, which config fields can do anything.

    Returns a JSON-serialisable evidence block. Raises RuntimeError if a
    dimension declared inert turns out to be live, because that would mean the
    exhaustive sweep is not exhaustive over the space that matters -- and a
    baseline searched over the wrong space is exactly the defect this experiment
    was built to remove. Failing here costs nothing; discovering it after the
    GPU hours costs the run.
    """
    probe = CARLConfig()
    findings = {}
    for field in probe.as_dict():
        has_attr = hasattr(sched, field)
        findings[field] = {
            "declared_live": field in LIVE_DIMENSIONS,
            "scheduler_has_attribute": bool(has_attr),
            "inert_reason": INERT_DIMENSIONS.get(field),
        }
    # use_cuda_graphs is the one field with an attribute that is nonetheless
    # inert, and the reason is a SECOND condition inside _decode_forward. Record
    # the condition's actual value rather than trusting the comment.
    findings["use_cuda_graphs"]["graph_runner_attached"] = (
        getattr(sched, "_graph_runner", None) is not None)
    findings["spec_k"]["enable_spec_decode"] = bool(
        getattr(sched, "enable_spec_decode", False))

    unexpected = []
    # A dimension is LIVE-BUT-UNSEARCHED only if the scheduler both exposes it
    # and would act on it. The two documented exceptions are checked explicitly
    # rather than being special-cased away.
    for field, f in findings.items():
        if f["declared_live"] or not f["scheduler_has_attribute"]:
            continue
        if field == "use_cuda_graphs" and not f["graph_runner_attached"]:
            continue
        if field == "spec_k" and not f["enable_spec_decode"]:
            continue
        if field in ("enable_spec_decode",):
            continue
        unexpected.append(field)

    report = {
        "live_dimensions_searched": list(LIVE_DIMENSIONS),
        "grid_is_exhaustive_over": {
            "max_batch_size": GRID_BATCH_AXIS,
            "chunk_size": GRID_CHUNK_AXIS,
            "n_configurations": len(GRID_BATCH_AXIS) * len(GRID_CHUNK_AXIS),
        },
        "scheduler_class": type(sched).__name__,
        "per_field": findings,
        "unexpectedly_live": unexpected,
        "method": (
            "read against the live scheduler object this run will serve on, not "
            "asserted from source comments"),
        "passed": not unexpected,
    }
    if unexpected:
        raise RuntimeError(
            "probe_live_dimensions: {} would change execution but is NOT in the "
            "searched grid {}. A static baseline searched over the wrong space "
            "is the defect this experiment exists to remove -- widen the grid or "
            "justify the exclusion; do not proceed.".format(
                unexpected, list(LIVE_DIMENSIONS)))
    return report


# ===========================================================================
# The single held-out sweep: scales + both static baselines.
# ===========================================================================


def run_static(cfg: CARLConfig, model, tokenizer, n: int, seed: int) -> dict:
    """Serve one fixed configuration once, with no controller attached."""
    specs = abl._build_workload(tokenizer, "NON-STATIONARY", n, random.Random(seed))
    sched = abl._new_scheduler(model)
    abl._apply_sched(sched, cfg)
    return abl._serve(sched, specs)


def run_utility(metrics: dict, scales: RewardScales) -> float:
    """utility_v2 of one run's END-OF-RUN aggregates, under the frozen scales.

    The only utility estimator defined for a static config, and therefore the
    one comparison B uses for every config. `cache_hit_rate` is 0.0 because no
    prefix cache is wired in this harness -- the term is a constant here and
    cancels in every paired difference.
    """
    return utility_v2({
        "throughput_tps": metrics["throughput_tps"],
        "ttft_p99_ms": metrics["ttft_p99"],
        "tpot_p99_ms": metrics["tpot_p99"],
        "cache_hit_rate": 0.0,
    }, DEFAULT_WEIGHTS, scales)


def sweep_live_plane(model, tokenizer, n: int, seed: int,
                     batches: list, chunks: list) -> list:
    """EXHAUSTIVE max_batch_size x chunk_size on the held-out seed.

    One sweep, three consumers: the frozen scales, Static-Tput and
    Static-Utility. Every other CARLConfig field is left at its default and is
    inert here (see probe_live_dimensions), so this grid IS the reachable
    configuration space of the static baseline.
    """
    total = len(batches) * len(chunks)
    print(f"\n[held-out sweep] EXHAUSTIVE {len(batches)}x{len(chunks)} = {total} "
          f"configs x {n} requests on seed {seed} (never an evaluation seed)",
          flush=True)
    rows = []
    for mb in batches:
        for cs in chunks:
            cfg = CARLConfig(max_batch_size=mb, chunk_size=cs).clamp()
            m = run_static(cfg, model, tokenizer, n, seed)
            rows.append({
                "max_batch_size": mb,
                "chunk_size": cs,
                "config": cfg.as_dict(),
                "throughput_tps": m["throughput_tps"],
                "ttft_p50_ms": m["ttft_p50"], "ttft_p99_ms": m["ttft_p99"],
                "tpot_p50_ms": m["tpot_p50"], "tpot_p99_ms": m["tpot_p99"],
                "slo_rate": m["slo_rate"], "wall_s": m["wall_s"],
            })
            print(f"  mb={mb:2d} cs={cs:3d} -> {m['throughput_tps']:7.2f} tok/s  "
                  f"ttftP99={m['ttft_p99']:9.1f}ms  tpotP99={m['tpot_p99']:7.1f}ms  "
                  f"({m['wall_s']:.1f}s)", flush=True)
    return rows


def freeze_scales(rows: list, n: int, seed: int) -> tuple:
    """Medians of the held-out sweep, frozen. Nothing here reads evaluation data."""
    scales = RewardScales.from_measurements(
        [r["throughput_tps"] for r in rows],
        [r["ttft_p99_ms"] for r in rows],
        [r["tpot_p99_ms"] for r in rows],
    )
    record = {
        "method": ("medians of the exhaustive held-out max_batch_size x "
                   "chunk_size sweep, frozen before any evaluation run and "
                   "before any utility is computed"),
        "held_out_seed": seed,
        "requests_per_sweep_run": n,
        "n_observations": len(rows),
        "why_held_out": (
            "seed 999 is documented in ablation_live as never used for an "
            "evaluation run; deriving scales from the evaluation seeds would set "
            "the reward's mid-point from the data being measured"),
        "why_same_n_as_evaluation": (
            "under burst arrivals the TTFT p99 scales with how many requests are "
            "queued, so scales fitted at a smaller n would mis-set ttft_target"),
        "why_frozen_before_static_utility_selection": (
            "Static-Utility is chosen by ranking these same rows under these "
            "scales; freezing the scales from the MEASUREMENTS first means the "
            "baseline cannot have influenced the yardstick it is ranked by"),
        "frozen_scales": {
            "t_half": scales.t_half, "ttft_target": scales.ttft_target,
            "tpot_target": scales.tpot_target, "sharpness": scales.sharpness,
        },
        "weights": dict(DEFAULT_WEIGHTS),
        "observations": rows,
    }
    print(f"[held-out sweep] FROZEN scales: t_half={scales.t_half:.4f} "
          f"ttft_target={scales.ttft_target:.4f} "
          f"tpot_target={scales.tpot_target:.4f} sharpness={scales.sharpness}",
          flush=True)
    return scales, record


def select_static_baselines(rows: list, scales: RewardScales) -> tuple:
    """Static-Tput and Static-Utility from the SAME rows, ranked differently.

    Returns (cfg_tput, cfg_utility, selection_record). The record carries the
    full ranked table under BOTH objectives and states explicitly whether they
    agree -- the disagreement is the whole reason this experiment exists, so it
    is measured and reported rather than assumed in either direction.
    """
    scored = []
    for r in rows:
        scored.append(dict(r, utility_v2=run_utility(
            {"throughput_tps": r["throughput_tps"], "ttft_p99": r["ttft_p99_ms"],
             "tpot_p99": r["tpot_p99_ms"]}, scales)))

    best_t = max(scored, key=lambda r: r["throughput_tps"])
    best_u = max(scored, key=lambda r: r["utility_v2"])
    agree = (best_t["max_batch_size"] == best_u["max_batch_size"]
             and best_t["chunk_size"] == best_u["chunk_size"])

    cfg_t = CARLConfig(max_batch_size=best_t["max_batch_size"],
                       chunk_size=best_t["chunk_size"]).clamp()
    cfg_u = CARLConfig(max_batch_size=best_u["max_batch_size"],
                       chunk_size=best_u["chunk_size"]).clamp()

    # Utility spread across the candidate plane, reported next to the winner: a
    # baseline chosen from a flat objective is a coin toss, and the reader is
    # entitled to see which it was.
    utils = [r["utility_v2"] for r in scored]
    tputs = [r["throughput_tps"] for r in scored]
    record = {
        "method": ("exhaustive max_batch_size x chunk_size sweep on the held-out "
                   "seed, ranked twice: once by throughput, once by utility_v2 "
                   "under the frozen scales"),
        "why_two_baselines": (
            "CARL maximises utility_v2. Ranking a static baseline by throughput "
            "and then comparing it to CARL on utility (or the reverse) scores "
            "each side on a different objective, so the sign of the result is a "
            "property of the mismatch rather than of adaptation."),
        "search_space": {"max_batch_size": sorted({r["max_batch_size"] for r in rows}),
                         "chunk_size": sorted({r["chunk_size"] for r in rows})},
        "search_is_exhaustive": True,
        "n_candidates": len(scored),
        "inert_dimensions_excluded": dict(INERT_DIMENSIONS),
        "objectives_agree": agree,
        "objective_disagreement_note": (
            "the two objectives selected the SAME configuration; the mismatch "
            "the previous experiment suffered from is not expressed at this "
            "operating point, and comparisons A and B share a baseline"
            if agree else
            "the two objectives selected DIFFERENT configurations -- this is the "
            "objective mismatch, measured. Comparing CARL's utility against the "
            "throughput-selected config would have been comparing across "
            "objectives."),
        "static_tput": {"winner": cfg_t.as_dict(), "row": best_t,
                        "ranked_by": "throughput_tps"},
        "static_utility": {"winner": cfg_u.as_dict(), "row": best_u,
                           "ranked_by": "utility_v2 under the frozen scales"},
        "objective_spread": {
            "throughput_tps": {"min": min(tputs), "max": max(tputs),
                               "spread": max(tputs) - min(tputs)},
            "utility_v2": {"min": min(utils), "max": max(utils),
                           "spread": max(utils) - min(utils)},
        },
        "ranked_table": sorted(scored, key=lambda r: -r["utility_v2"]),
    }
    print(f"[static] Static-Tput    : mb={cfg_t.max_batch_size} "
          f"cs={cfg_t.chunk_size} ({best_t['throughput_tps']:.2f} tok/s, "
          f"u={best_t['utility_v2']:.4f})", flush=True)
    print(f"[static] Static-Utility : mb={cfg_u.max_batch_size} "
          f"cs={cfg_u.chunk_size} ({best_u['throughput_tps']:.2f} tok/s, "
          f"u={best_u['utility_v2']:.4f})", flush=True)
    print(f"[static] objectives agree: {agree}", flush=True)
    return cfg_t, cfg_u, record


def gate_sweep(rows: list, scales: RewardScales) -> dict:
    """PRE-RUN GATE: the frozen scales must separate the candidate plane."""
    arm_rewards = {
        f"mb{r['max_batch_size']}_cs{r['chunk_size']}": run_utility(
            {"throughput_tps": r["throughput_tps"], "ttft_p99": r["ttft_p99_ms"],
             "tpot_p99": r["tpot_p99_ms"]}, scales)
        for r in rows
    }
    report = gate_candidate_arm_spread(
        arm_rewards, context="final_hardening held-out sweep",
        threshold=MIN_ARM_REWARD_SPREAD)
    print(f"[gate:pre-run] candidate reward spread {report['reward_spread']:.6f} "
          f">= {MIN_ARM_REWARD_SPREAD} -- PASS", flush=True)
    return report


# ===========================================================================
# One evaluation run.
# ===========================================================================


def _build_bandit(name: str):
    arms = abl._frozen_arms(None, _ARM_SET_FOR[name])
    if name == "RuleOnly":
        return RuleOnlyBandit(arms, d=FEATURE_DIM)
    return PerRegimeBandit(arms, d=FEATURE_DIM,
                           bandit_cls=RepairedLinUCBBandit, alpha=ALPHA)


def _cycle_records(controller) -> list:
    """Per-control-cycle rows. `reward` is None on a withheld cycle, never 0.

    `rewarded_arm` is recorded next to `selected_arm` because under
    delayed-reward timing the reward at row t scores row t-1's config, and a
    reader must not have to re-derive that. On a withheld cycle `rewarded_arm`
    is -1 and `reward_valid` is False, which together say the learner received
    nothing for that interval.
    """
    out = []
    for e in controller.controller_log:
        terms = dict(e.reward_terms)
        raw = terms.pop("raw_metrics", None)
        validity = terms.pop("reward_validity", None)
        out.append({
            "step": e.step,
            "t_monotonic_s": e.t_monotonic_s,
            "regime": e.regime.value,
            "selected_arm": e.arm,
            "rewarded_arm": e.rewarded_arm,
            "config": e.config.as_dict(),
            "context": e.state_features,
            "reward": e.reward,
            "reward_valid": e.reward_valid,
            "reward_validity": validity,
            "reward_terms": terms,
            "reward_raw_metrics": raw,
            "observed": e.observed,
        })
    return out


def convergence_diagnostic(cycles: list) -> dict:
    """Did the trace EXPLORE and then EXPLOIT? Per regime, by CONVERGENCE_RULE.

    Written against the rule fixed at module level before the run. A high modal
    share alone is NOT convergence: the as-published LinUCB scored 200/200 on
    arm 0 while learning nothing, and that trace would pass any "it settled"
    test. Exploration in the first half is therefore required, and the modal
    share must have RISEN.
    """
    rule = CONVERGENCE_RULE
    by_regime: dict = {}
    for c in cycles:
        by_regime.setdefault(c["regime"], []).append(c["selected_arm"])

    per_regime = {}
    for regime, arms in by_regime.items():
        n = len(arms)
        half = n // 2
        first, second = arms[:half], arms[half:]

        def modal_share(seq):
            if not seq:
                return 0.0
            return max(seq.count(a) for a in set(seq)) / len(seq)

        rec = {
            "decisions": n,
            "distinct_arms": len(set(arms)),
            "first_half_distinct_arms": len(set(first)),
            "second_half_distinct_arms": len(set(second)),
            "first_half_modal_share": modal_share(first),
            "second_half_modal_share": modal_share(second),
            "terminal_run_length": _terminal_run(arms),
            "arm_sequence_head": arms[:20],
            "arm_sequence_tail": arms[-20:],
        }
        rec["explored"] = (n >= rule["min_decisions_in_regime"]
                           and rec["first_half_distinct_arms"]
                           >= rule["min_first_half_distinct_arms"])
        rec["exploited"] = (
            rec["second_half_modal_share"] >= rule["min_second_half_modal_share"]
            and rec["second_half_modal_share"] > rec["first_half_modal_share"])
        rec["supported"] = bool(rec["explored"] and rec["exploited"])
        rec["why_not"] = None if rec["supported"] else (
            "fewer than {} decisions".format(rule["min_decisions_in_regime"])
            if n < rule["min_decisions_in_regime"] else
            "no exploration: the first half played {} distinct arm(s)".format(
                rec["first_half_distinct_arms"])
            if not rec["explored"] else
            "no exploitation: second-half modal share {:.2f} (needs >= {} and "
            "> the first half's {:.2f})".format(
                rec["second_half_modal_share"],
                rule["min_second_half_modal_share"],
                rec["first_half_modal_share"]))
        per_regime[regime] = rec

    return {
        "rule": rule,
        "per_regime": per_regime,
        "supported_in_any_regime": any(r["supported"] for r in per_regime.values()),
        "supported_in_all_regimes": bool(per_regime) and all(
            r["supported"] for r in per_regime.values()),
    }


def _terminal_run(arms: list) -> int:
    """Length of the final unbroken run of one arm (1 for an empty tail)."""
    if not arms:
        return 0
    k, last = 1, arms[-1]
    for a in reversed(arms[:-1]):
        if a != last:
            break
        k += 1
    return k


def run_one(name: str, model, tokenizer, n: int, seed: int, *,
            static_cfg=None, reward_fn=None, scales=None) -> dict:
    """Serve one configuration once. Same workload construction as the v1/v2 runs."""
    if name in _STATIC_CONFIGS:
        out = run_static(static_cfg, model, tokenizer, n, seed)
        out["applied_config"] = static_cfg.as_dict()
        out["run_utility_v2"] = run_utility(out, scales)
        return out

    specs = abl._build_workload(tokenizer, "NON-STATIONARY", n, random.Random(seed))
    sched = abl._new_scheduler(model)
    tracker = MetricsTracker(window=max(50, n))
    controller = CARLController(
        scheduler=sched, bandit=_build_bandit(name),
        observe_interval=OBSERVE_INTERVAL, slo=abl._SLO, metrics=tracker,
        reward_fn=reward_fn)

    out = abl._serve(sched, specs, controller=controller, tracker=tracker)
    out["run_utility_v2"] = run_utility(out, scales)

    cycles = _cycle_records(controller)
    out["cycles"] = cycles
    out["arm_set"] = _ARM_SET_FOR[name]
    out["bandit_cls"] = type(controller.bandit).__name__
    out["controller_stats"] = controller.stats()
    out["reward_validity"] = controller.reward_validity_report()

    arms_played = [c["selected_arm"] for c in cycles]
    out["decisions"] = len(arms_played)
    out["arm_changes"] = sum(1 for a, b in zip(arms_played, arms_played[1:])
                             if a != b)
    # RATE, not just count: a run with more control cycles gets more chances to
    # change arms, so the count alone is not comparable across configs whose
    # runs differ in length. Denominator is the number of OPPORTUNITIES to
    # change, which is one fewer than the number of decisions.
    out["arm_change_rate"] = (out["arm_changes"] / (len(arms_played) - 1)
                              if len(arms_played) > 1 else 0.0)
    out["arm_histogram"] = {}
    for c in cycles:
        key = f"{c['regime']}|arm{c['selected_arm']}"
        out["arm_histogram"][key] = out["arm_histogram"].get(key, 0) + 1
    out["convergence"] = convergence_diagnostic(cycles)

    valid = [c["reward"] for c in cycles if c["reward_valid"]]
    out["n_valid_rewards"] = len(valid)
    out["n_withheld_rewards"] = len(cycles) - len(valid)
    out["mean_valid_cycle_utility"] = (statistics.fmean(valid) if valid else None)
    # PER-RUN GATE, over the VALID rewards only. A withheld cycle carries no
    # number, so including it would mean inventing one -- and a stream padded
    # with a constant fill would pass or fail this gate for a reason that has
    # nothing to do with whether the reward discriminates.
    out["reward_gate"] = gate_live_reward_stream(
        valid, context=f"final_hardening {name} seed={seed} (valid rewards only)",
        threshold=MIN_LIVE_REWARD_SPREAD, min_distinct=MIN_LIVE_DISTINCT)
    return out


# ===========================================================================
# Aggregation.
# ===========================================================================

_METRICS = ["throughput_tps", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
            "slo_rate", "run_utility_v2"]


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _paired(a_vals: list, b_vals: list) -> dict:
    """Paired difference a - b over matched seeds, with a 95% CI and Cohen's d.

    `all_diffs_zero` is reported explicitly so a bit-identical IDENTITY is never
    read as a statistical tie -- the distinction H3 turns on.
    """
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = (sd / (n ** 0.5)) if n > 1 else 0.0
    return {
        "n": n,
        "mean_difference": mean,
        "std_difference": sd,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se] if se else [mean, mean],
        "cohens_d_paired": (mean / sd) if sd else None,
        "all_diffs_zero": all(d == 0.0 for d in diffs),
        "per_seed_difference": diffs,
    }


def _save_raw(name: str, seed: int, run: dict) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = {
        "config": name, "seed": seed,
        "throughput_tps": run["throughput_tps"],
        "ttft_p50_ms": run["ttft_p50"], "ttft_p99_ms": run["ttft_p99"],
        "tpot_p50_ms": run["tpot_p50"], "tpot_p99_ms": run["tpot_p99"],
        "slo_rate": run["slo_rate"], "wall_s": run["wall_s"],
        "run_utility_v2": run.get("run_utility_v2"),
        "applied_config": run.get("applied_config"),
        "requests": run["requests"],
        "step_log": run.get("step_log"),
        "shift_t": run.get("shift_t"), "shift_step": run.get("shift_step"),
        "cycles": run.get("cycles"),
        "reward_gate": run.get("reward_gate"),
        "reward_validity": run.get("reward_validity"),
        "n_valid_rewards": run.get("n_valid_rewards"),
        "n_withheld_rewards": run.get("n_withheld_rewards"),
        "mean_valid_cycle_utility": run.get("mean_valid_cycle_utility"),
        "decisions": run.get("decisions"),
        "arm_changes": run.get("arm_changes"),
        "arm_change_rate": run.get("arm_change_rate"),
        "arm_histogram": run.get("arm_histogram"),
        "convergence": run.get("convergence"),
        "controller_stats": run.get("controller_stats"),
        "decision_us": run.get("decision_us"),
    }
    with open(os.path.join(RAW_DIR, f"{name}_seed{seed:03d}.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


# ===========================================================================
# Provenance.
# ===========================================================================


def load_source_provenance() -> dict:
    """The sidecar from `source_provenance.py`, or an explicit statement of absence.

    The previous runtime artifact carried git_sha=null and git_dirty=null because
    the Colab archive excludes `.git`, and a null there is indistinguishable from
    "clean". So this NEVER emits a bare null: either the captured block, or a
    record saying the sidecar was missing and what to run.
    """
    if not os.path.isfile(SOURCE_PROVENANCE_PATH):
        return {
            "present": False,
            "path": os.path.relpath(SOURCE_PROVENANCE_PATH, _REPO_ROOT),
            "why_missing": (
                "SOURCE_PROVENANCE.json was not in the uploaded archive. The "
                "runtime has no .git, so git_sha/git_dirty in this artifact's "
                "own _provenance block will be null and cannot be trusted to "
                "mean 'clean'."),
            "remedy": ("run `python scripts/eval/repair/source_provenance.py "
                       "--archive <zip>` on the machine that has the repository, "
                       "BEFORE building the archive, and include the JSON"),
        }
    with open(SOURCE_PROVENANCE_PATH, encoding="utf-8") as f:
        block = json.load(f)
    block["present"] = True
    block["embedded_by"] = "scripts/eval/repair/final_hardening.py"
    block["embedding_note"] = (
        "captured on the machine that HAS the git repository, before the archive "
        "was built; the runtime's own git_sha/git_dirty are expected to be null "
        "because .git is excluded from the archive, and this block is what "
        "replaces them")
    return block


def _provenance_extra(seeds: list, n: int, results: dict) -> dict:
    return {
        "model": MODEL_NAME,
        "dtype": str(torch.float16 if DEVICE.type == "cuda" else torch.float32),
        "device": str(DEVICE),
        "seeds": seeds,
        "requests": n,
        "arrival_mode": "burst",
        "observe_interval": OBSERVE_INTERVAL,
        "validation_seed": VALIDATION_SEED,
        "learner_class": "RepairedLinUCBBandit",
        "alpha": ALPHA,
        "configs_run": list(CONFIGS),
        "arm_set_definition": {
            "restricted": "src.carl.config.all_arm_sets()",
            "expanded": "ablation_live.expanded_arm_sets()",
            "expanded_batch_axis": abl.ARM_SET_BATCH_AXIS,
        },
        "static_search_definition": {
            "method": "exhaustive max_batch_size x chunk_size",
            "max_batch_size": GRID_BATCH_AXIS,
            "chunk_size": GRID_CHUNK_AXIS,
            "validation_seed": VALIDATION_SEED,
            "inert_dimensions_excluded": sorted(INERT_DIMENSIONS),
        },
        "frozen_reward_scales": results.get("reward", {}).get("scales"),
        "reward_weights": results.get("reward", {}).get("weights"),
        "cold_start_policy": results.get("reward", {}).get("cold_start_policy"),
        # THE SIDECAR. See load_source_provenance().
        "source_provenance": load_source_provenance(),
    }


def write_artifact(results: dict, seeds: list, n: int) -> str:
    """Stamp with provenance and write. Called after EVERY config, not once.

    WHY INCREMENTAL. Two artifacts in this repository were lost to a Colab VM
    torn down before the JSON was downloaded and were later reconstructed by
    hand, which is why they are quarantined. A gate firing on the last config, or
    a disconnect, must not destroy the GPU hours already spent.
    """
    return provenance.write_result(
        RESULTS_PATH, results, EXPERIMENT,
        script="scripts/eval/repair/final_hardening.py",
        extra=_provenance_extra(seeds, n, results))


# ===========================================================================
# Driver.
# ===========================================================================


def run_all(seeds: list, n: int, *, allow_cpu: bool = False,
            batches: list | None = None, chunks: list | None = None,
            sink: dict | None = None) -> dict:
    if DEVICE.type != "cuda" and not allow_cpu:
        raise SystemExit(
            "final_hardening REFUSES to run without CUDA. This experiment exists "
            "to answer a hardware question and a CPU number would not answer it. "
            "Pass --allow-cpu-smoke for a wiring smoke test (never a result).")

    batches = batches or GRID_BATCH_AXIS
    chunks = chunks or GRID_CHUNK_AXIS
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {len(seeds)} seeds x {n} requests "
          f"| grid {len(batches)}x{len(chunks)} | seeds {seeds}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    # 0. PROVE which dimensions execute, before spending anything.
    liveness = probe_live_dimensions(abl._new_scheduler(model))
    print(f"[liveness] searched {liveness['live_dimensions_searched']} "
          f"exhaustively; nothing unexpectedly live -- PASS", flush=True)

    # 1. ONE held-out sweep -> frozen scales -> pre-run gate -> both baselines.
    rows = sweep_live_plane(model, tokenizer, n, VALIDATION_SEED, batches, chunks)
    scales, calibration = freeze_scales(rows, n, VALIDATION_SEED)
    calibration["gate"] = gate_sweep(rows, scales)
    cfg_tput, cfg_utility, selection = select_static_baselines(rows, scales)
    static_cfg_for = {"Static-Tput": cfg_tput, "Static-Utility": cfg_utility}

    # 2. The repaired reward. require_valid_metrics defaults True -- stated here
    #    explicitly because it is the whole of defect 2's repair.
    reward_fn = LiveRewardV2(scales, require_valid_metrics=True,
                             min_latency_samples=1)

    results: dict = {
        "objective": (
            "Repair two defects found by per-cycle inspection of the reward-v2 "
            "artifact -- an objective mismatch in the static baseline, and a "
            "cold-start reward computed from unmeasured latency -- and re-run "
            "the decisive comparisons with both fixed."),
        "supersedes": {
            "artifact": "docs/eval/reward_v2_live_results.json",
            "preserved": True,
            "why_preserved": (
                "it is the experiment that EXPOSED both defects: its per-cycle "
                "trace is the evidence for the cold-start optimism, and its "
                "calibration sweep is the evidence that the throughput and "
                "utility objectives disagree. It is not rewritten, and this run "
                "writes to different paths."),
        },
        "defects_repaired": {
            "objective_mismatch": {
                "found": ("Static-Best was selected by throughput while CARL "
                          "optimises utility_v2; the calibration sweep showed "
                          "the two objectives select different batch sizes"),
                "repair": ("two baselines, Static-Tput and Static-Utility, "
                           "selected from ONE exhaustive held-out sweep by the "
                           "two ranking functions; every comparison now has a "
                           "baseline tuned for the metric it is judged on"),
            },
            "cold_start_missing_metric_reward": {
                "found": ("in every seed the first three control cycles logged "
                          "ttft_p99_ms=0 and tpot_p99_ms=0 because no request "
                          "had completed; latency_term scores 0 ms as PERFECT, "
                          "so those cycles earned near-maximum reward and "
                          "updated LinUCB, biasing arm 0"),
                "repair": ("MetricsTracker.metric_validity exposes per-metric "
                           "sample counts; LiveRewardV2 returns reward=None with "
                           "reward_valid=False until every consumed metric is "
                           "measured; CARLController performs no learner update "
                           "on a withheld reward and substitutes no fill value"),
                "positive_control": "pre-registered H5: warmup_cycles >= 1 per run",
            },
        },
        "pre_registration": PRE_REGISTRATION,
        "liveness_probe": liveness,
        "scenario": abl.scenario_description(n),
        "seeds": seeds, "runs": len(seeds), "requests": n,
        "arrival_mode": "burst (bulk dump; ablation_live default)",
        "observe_interval": OBSERVE_INTERVAL,
        "validation_seed": VALIDATION_SEED,
        "held_fixed": [
            "Tesla T4", "TinyLlama-1.1B-Chat-v1.0 fp16", "seeds 42..51",
            f"{n} requests per run",
            "two-phase NON-STATIONARY workload (ablation_live._build_workload)",
            f"controller cadence OBSERVE_INTERVAL={OBSERVE_INTERVAL}",
            "restricted/expanded arm-set definitions",
            "RepairedLinUCBBandit, alpha=0.5",
            f"held-out calibration seed {VALIDATION_SEED}",
            f"context vector FEATURE_DIM={FEATURE_DIM}",
        ],
        "changed_vs_reward_v2_run": [
            "Static-Best (throughput-selected, LHS x16 over 3 inert dimensions) "
            "-> Static-Tput + Static-Utility (exhaustive over the live plane)",
            "reward withheld while TTFT/TPOT are unmeasured, instead of scoring "
            "them as 0 ms (perfect)",
            "learner updated only on valid rewards",
            "source provenance carried in as a sidecar captured off-runtime",
        ],
        "reward": {
            "version": "v2",
            "function": "src.carl.reward.utility_v2",
            "adapter": "src.carl.live_reward.LiveRewardV2",
            **reward_fn.as_dict(),
            "thresholds": {
                "min_arm_reward_spread": MIN_ARM_REWARD_SPREAD,
                "min_live_reward_spread": MIN_LIVE_REWARD_SPREAD,
                "min_live_distinct": MIN_LIVE_DISTINCT,
                "provenance": THRESHOLD_PROVENANCE,
            },
        },
        "reward_calibration": calibration,
        "static_selection": selection,
        "utility_estimators": {
            "run_utility_v2": (
                "utility_v2 of a run's end-of-run aggregates under the frozen "
                "scales. Defined for EVERY config, including the statics, and "
                "the basis of comparison B. For a config that is constant for "
                "the whole run this simply IS its utility."),
            "mean_valid_cycle_utility": (
                "mean over a controller's SCORED control cycles -- what the "
                "learner actually consumed. Reported for controllers only and "
                "NOT compared against the statics, which have no control "
                "cycles. The two estimators are not interchangeable."),
        },
        "convergence_rule": CONVERGENCE_RULE,
        "configs_run": list(CONFIGS),
        "config_arm_sets": dict(_ARM_SET_FOR),
        "config_policy": {
            "Static-Tput": "fixed config, argmax throughput on held-out seed 999",
            "Static-Utility": ("fixed config, argmax utility_v2 (CARL's own "
                               "objective, frozen scales) on held-out seed 999"),
            "RuleOnly": "src.carl.rule_only.RuleOnlyBandit (no learning)",
            "CARL-Repaired": "RepairedLinUCBBandit + restricted arms",
            "CARL-Expanded": "RepairedLinUCBBandit + expanded arms",
        },
        "arm_sets": {
            "restricted": abl.arm_set_summary(abl.all_arm_sets()),
            "expanded": abl.arm_set_summary(abl.expanded_arm_sets()),
            "expanded_batch_axis": abl.ARM_SET_BATCH_AXIS,
        },
        "scope_note": (
            "ONE operating point: burst arrivals at rho >> 1 on a single T4 with "
            "one 1.1B model, over the two live knobs this harness exposes. It "
            "does NOT establish a load envelope, and it says nothing about "
            "speculation, routing or KV eviction, which are inactive here."),
        "configs": {},
    }
    if sink is not None:
        sink.clear()
        sink.update(results)
        results = sink

    per_seed: dict = {}
    for name in CONFIGS:
        per_run = []
        for i, seed in enumerate(seeds):
            run = run_one(
                name, model, tokenizer, n, seed,
                static_cfg=static_cfg_for.get(name),
                reward_fn=reward_fn if name in _CONTROLLER_CONFIGS else None,
                scales=scales)
            per_run.append(run)
            _save_raw(name, seed, run)
            extra = ""
            if name in _CONTROLLER_CONFIGS:
                rv = run["reward_validity"]
                extra = (f" | u={run['mean_valid_cycle_utility']:.4f} "
                         f"valid={run['n_valid_rewards']}/{run['decisions']} "
                         f"warmup={rv['warmup_cycles']} "
                         f"updates={rv['valid_reward_updates']} "
                         f"armChg={run['arm_changes']} "
                         f"rate={run['arm_change_rate']:.3f}")
            print(f"  {name:<15} {i+1}/{len(seeds)} (seed {seed}): "
                  f"{run['throughput_tps']:7.2f} tok/s "
                  f"ttftP99={run['ttft_p99']:9.1f}ms "
                  f"runU={run['run_utility_v2']:.4f}{extra}", flush=True)

        agg = {"arm_set": _ARM_SET_FOR.get(name),
               "policy": results["config_policy"][name]}
        for key in _METRICS:
            mean, std = _mean_std([m[key] for m in per_run])
            agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
        agg["_per_seed"] = {k: [m[k] for m in per_run]
                            for k in ("throughput_tps", "run_utility_v2")}
        if name in _STATIC_CONFIGS:
            agg["applied_config"] = per_run[0]["applied_config"]
        if name in _CONTROLLER_CONFIGS:
            agg["decisions_mean"] = statistics.fmean(m["decisions"] for m in per_run)
            agg["arm_changes_mean"] = statistics.fmean(m["arm_changes"] for m in per_run)
            agg["arm_change_rate_mean"] = statistics.fmean(
                m["arm_change_rate"] for m in per_run)
            agg["mean_valid_cycle_utility"] = statistics.fmean(
                m["mean_valid_cycle_utility"] for m in per_run)
            agg["valid_reward_updates_mean"] = statistics.fmean(
                m["reward_validity"]["valid_reward_updates"] for m in per_run)
            agg["valid_reward_updates_min"] = min(
                m["reward_validity"]["valid_reward_updates"] for m in per_run)
            # THE COLD-START RECORD, per seed. Pre-registered H5 reads this.
            agg["cold_start"] = {
                "warmup_cycles_per_seed": {
                    str(s): m["reward_validity"]["warmup_cycles"]
                    for s, m in zip(seeds, per_run)},
                "first_valid_reward_cycle_per_seed": {
                    str(s): m["reward_validity"]["first_valid_reward_cycle_index"]
                    for s, m in zip(seeds, per_run)},
                "first_valid_reward_step_per_seed": {
                    str(s): m["reward_validity"]["first_valid_reward_scheduler_step"]
                    for s, m in zip(seeds, per_run)},
                "warmup_duration_s_per_seed": {
                    str(s): m["reward_validity"]["warmup_duration_s"]
                    for s, m in zip(seeds, per_run)},
                "withheld_after_first_valid_total": sum(
                    m["reward_validity"]["withheld_after_first_valid"]
                    for m in per_run),
                "h5_positive_control_holds": all(
                    m["reward_validity"]["warmup_cycles"] >= 1 for m in per_run),
            }
            agg["convergence_supported_seeds"] = sum(
                1 for m in per_run if m["convergence"]["supported_in_any_regime"])
            agg["convergence_claim"] = (
                "supported in at least one regime on {}/{} seeds by the "
                "pre-registered rule".format(
                    agg["convergence_supported_seeds"], len(per_run)))
            agg["reward_spread_min"] = min(m["reward_gate"]["reward_spread"]
                                           for m in per_run)
            agg["reward_distinct_min"] = min(m["reward_gate"]["n_distinct"]
                                             for m in per_run)
            agg["seeds_behaviourally_distinct"] = len({
                tuple(c["selected_arm"] for c in m["cycles"]) for m in per_run}) > 1
        results["configs"][name] = agg
        per_seed[name] = agg["_per_seed"]
        results["completed_configs"] = list(results["configs"])
        write_artifact(results, seeds, n)      # checkpoint after every config

    results["comparisons"] = {
        label: {
            "a": a, "b": b, "metric": metric, "question": q,
            metric: _paired(per_seed[a][metric], per_seed[b][metric]),
            # Both metrics are reported for every comparison; `metric` names the
            # PRE-REGISTERED one, so a reader can see the secondary without the
            # primary being chosen after the fact.
            "secondary": {
                k: _paired(per_seed[a][k], per_seed[b][k])
                for k in ("throughput_tps", "run_utility_v2") if k != metric
            },
        }
        for label, a, b, metric, q in COMPARISONS
        if a in per_seed and b in per_seed
    }
    return results


def _print(results: dict) -> None:
    print(f"\n=== FINAL HARDENING (mean +/- std over {results['runs']} seeds, "
          f"{results['requests']} requests) ===")
    hdr = ["config", "tok/s", "ttft p50/p99 ms", "tpot p50/p99 ms", "run util",
           "cycle util", "validUpd", "decisions", "armChg", "armChg rate"]
    print("| " + " | ".join(hdr) + " |")
    print("| " + " | ".join("---" for _ in hdr) + " |")
    for name in CONFIGS:
        a = results["configs"].get(name)
        if not a:
            continue
        nan = float("nan")
        print("| " + " | ".join([
            name,
            f"{a['throughput_tps_mean']:.2f} +/- {a['throughput_tps_std']:.2f}",
            f"{a['ttft_p50_mean']:.0f} / {a['ttft_p99_mean']:.0f}",
            f"{a['tpot_p50_mean']:.1f} / {a['tpot_p99_mean']:.1f}",
            f"{a['run_utility_v2_mean']:.4f} +/- {a['run_utility_v2_std']:.4f}",
            f"{a.get('mean_valid_cycle_utility', nan):.4f}",
            f"{a.get('valid_reward_updates_mean', nan):.1f}",
            f"{a.get('decisions_mean', nan):.1f}",
            f"{a.get('arm_changes_mean', nan):.1f}",
            f"{a.get('arm_change_rate_mean', nan):.3f}",
        ]) + " |")

    print("\n=== Cold start (pre-registered H5: warmup_cycles >= 1) ===")
    for name in CONFIGS:
        cs = (results["configs"].get(name) or {}).get("cold_start")
        if not cs:
            continue
        w = list(cs["warmup_cycles_per_seed"].values())
        print(f"  {name:<15} warmup cycles per seed {w} | H5 holds: "
              f"{cs['h5_positive_control_holds']} | withheld after first valid: "
              f"{cs['withheld_after_first_valid_total']}")

    print("\n=== Convergence (pre-registered rule; NOT inferred from a mean) ===")
    for name in CONFIGS:
        a = results["configs"].get(name) or {}
        if "convergence_claim" in a:
            print(f"  {name:<15} {a['convergence_claim']}")

    print("\n=== Comparisons (paired over seeds) ===")
    for label, c in results.get("comparisons", {}).items():
        t = c[c["metric"]]
        print(f"  {label}")
        print(f"    {c['a']} - {c['b']} on {c['metric']}: "
              f"{t['mean_difference']:+.4f} "
              f"CI95 [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] "
              f"d={t['cohens_d_paired']} all_zero={t['all_diffs_zero']}")
        for k, s in c["secondary"].items():
            print(f"    (secondary) on {k}: {s['mean_difference']:+.4f} "
                  f"CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Final hardening experiment: two objective-matched static "
                    "baselines + cold-start-safe reward.")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--limit", type=int, default=DEFAULT_REQUESTS,
                    help="requests per run (the reward-v2 run used 200)")
    ap.add_argument("--grid-batches", default=",".join(map(str, GRID_BATCH_AXIS)),
                    help="max_batch_size axis of the exhaustive held-out sweep")
    ap.add_argument("--grid-chunks", default=",".join(map(str, GRID_CHUNK_AXIS)),
                    help="chunk_size axis of the exhaustive held-out sweep")
    ap.add_argument("--allow-cpu-smoke", action="store_true",
                    help="permit a CPU run for WIRING VERIFICATION ONLY; the "
                         "artifact is written with a cpu_smoke marker and is "
                         "never a result")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    batches = [int(s) for s in args.grid_batches.split(",") if s.strip()]
    chunks = [int(s) for s in args.grid_chunks.split(",") if s.strip()]

    t_start = time.time()
    # `partial` is filled in by run_all as it goes, so an abort still has
    # something to write, and the FAILURE ITSELF is evidence worth keeping.
    partial: dict = {}
    try:
        results = run_all(seeds, args.limit, allow_cpu=args.allow_cpu_smoke,
                          batches=batches, chunks=chunks, sink=partial)
    except DegenerateRewardError as exc:
        partial["aborted"] = {
            "reason": "DegenerateRewardError",
            "message": str(exc),
            "completed_configs": list(partial.get("configs", {})),
            "interpretation": (
                "A gate fired: the reward could not distinguish configurations. "
                "This is a RESULT, not a crash. Do not re-run with a lowered "
                "threshold; re-derive the scales, or report the degeneracy."),
        }
        partial["elapsed_s"] = time.time() - t_start
        write_artifact(partial, seeds, args.limit)
        print(f"\nABORTED (gate fired). Partial artifact + reason written to "
              f"{RESULTS_PATH}", flush=True)
        raise
    except BaseException as exc:
        # OOM, disconnect, KeyboardInterrupt: the completed configs must survive.
        # Seeds are NOT skipped and the run is NOT continued -- the paired
        # comparisons require complete seed sets, so a partial config is evidence
        # to inspect, never a result to report.
        partial["aborted"] = {
            "reason": type(exc).__name__,
            "message": str(exc)[:2000],
            "completed_configs": list(partial.get("configs", {})),
            "interpretation": (
                "Run did not complete. Any config in completed_configs has all "
                "its seeds; anything else is absent. NOT a reportable result."),
        }
        partial["elapsed_s"] = time.time() - t_start
        try:
            write_artifact(partial, seeds, args.limit)
            print(f"\nABORTED ({type(exc).__name__}). Partial artifact written "
                  f"to {RESULTS_PATH}", flush=True)
        except Exception:
            pass
        raise

    results["elapsed_s"] = time.time() - t_start
    if DEVICE.type != "cuda":
        results["cpu_smoke"] = True
        results["scope_note"] = ("CPU SMOKE -- wiring verification only. "
                                 "NOT A RESULT.")

    write_artifact(results, seeds, args.limit)
    _print(results)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)
    print(f"Per-seed + per-cycle raw in {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
