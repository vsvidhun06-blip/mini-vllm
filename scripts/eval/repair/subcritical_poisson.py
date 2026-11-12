"""THE SUB-CRITICAL POISSON EXPERIMENT -- does adaptation's economics change with load?

WHAT THIS IS, AND WHAT IT IS NOT
================================
This is NOT a repair experiment. Nothing in `final_hardening.py` is alleged to
be broken, and its artifact is never rewritten or superseded here: it is only
read, hash-verified, as the saturated side of H1/H2 (saturated_reference.py,
recovered 2026-10-22). This is a
SCOPE experiment: it asks whether the final hardening's central finding is a
property of ADAPTATION or a property of the OPERATING POINT it was measured at.

The final hardening measured one point: a saturated, non-stationary burst. Every
request is dumped at t=0, the queue never drains, and the engine -- not the
arrival process -- sets the pace. At that point it found, over seeds 42..51 with
200 requests on a Tesla T4:

    CARL-Repaired - Static-Tput     throughput   -3.3678 tok/s  CI [-3.6633, -3.0723]
    CARL-Repaired - Static-Utility  run utility  -0.0245        CI [-0.0263, -0.0227]
    CARL-Repaired - RuleOnly        throughput   +3.7633 tok/s  CI [+3.2932, +4.2335]
    CARL-Expanded - CARL-Repaired   throughput   -1.7956 tok/s  CI [-2.2754, -1.3158]

Adaptive reconfiguration was useful but insufficient: better than a fixed rule,
worse than an objective-matched static configuration.

A saturated burst is exactly the regime in which `max_batch_size` binds hardest,
so it is the regime in which a well-chosen static configuration has the MOST to
offer. The obvious alternative explanation for the result is therefore not
"adaptation does not work" but "there was little left for adaptation to win".
This experiment moves the operating point and re-measures.

THE OPERATING POINT IS DERIVED, NOT CHOSEN
==========================================
A sub-critical condition stated as "we lowered the arrival rate" is not a
condition, because rho is unstated. So lambda is derived mechanically from a
measured service rate:

    lambda_phase = TARGET_RHO * mu_phase

`mu_phase` comes from `docs/eval/raw/repair/load_sweep_b7.json`, whose
`saturation` block records the maximum sustainable completion rate for each
workload shape under each EXECUTION ARM.

WHY THE EAGER NUMBERS AND NOT THE CUDA-GRAPH ONES
-------------------------------------------------
The artifact reports both arms:

    interactive@graph 5.5638 req/s      interactive@eager 2.9614 req/s
    batch@graph       2.5649 req/s      batch@eager       1.4135 req/s

`CARLConfig.use_cuda_graphs` defaults True, so a reader taking the config field
at face value would pick the graph rows and set lambda roughly 1.9x too high --
which at a nominal rho of 0.40 would actually run at rho ~ 0.75, and the
"sub-critical" condition would be a label rather than a fact.

It would be wrong, because `ablation_live._new_scheduler` -- the scheduler this
harness serves on -- never assigns `_graph_runner`, and
`ContinuousBatchScheduler._decode_forward` requires `_graph_runner is not None`
before it will replay a graph. Every decode falls back to eager. The same trap
already produced two "CUDA graph" arms with `cuda_graph_hits == 0` on an earlier
T4 smoke, and `final_hardening.probe_live_dimensions` exists because of it.

So the arm is read from the LIVE SCHEDULER OBJECT (`execution_arm_for`), never
from a config field, and the derivation refuses to proceed if the artifact does
not carry an unambiguous row for the arm that will actually execute.

The eager rows are traceable to hardware, not to a guess. The
`hardware_profile` block that produced them holds

    decode_per_row_s        0.009448482794434556
    decode_fixed_per_step_s 0.025461779480659390

and those are bit-identical to `hardware_profile_fit_eager` in
`docs/eval/raw/repair/batch_intervention_phase15_16_s42_s43_s44.json`, a real
Tesla T4 / TinyLlama-1.1B / fp16 measurement (54 eager rows, R^2 = 0.9985). The
CUDA-graph parameters are a SEPARATE fit from the same run, so the two arms are
distinguishable by construction rather than by assumption. `assert_arm_evidence`
re-checks this equality at run time and aborts if it ever stops holding.

WHAT CARRIES OVER AND WHAT MOVES
================================
MOVED (exactly one thing): arrivals. Burst dump -> Poisson, with a separate
lambda per phase, each at rho = 0.40 of ITS OWN measured capacity. The two
phases have very different service rates (2.96 vs 1.41 req/s), so a single
lambda would run one phase at roughly twice the other's offered load and the
"sub-critical" claim would hold for only one of them.

HELD: TinyLlama-1.1B fp16 on a T4, seeds 42..51, 200 requests, the semantic
phase split (1-100 INTERACTIVE shape, 101-200 BATCH shape), OBSERVE_INTERVAL=10,
held-out seed 999, the repaired LiveRewardV2 with `require_valid_metrics=True`,
the restricted/expanded arm-set definitions, RepairedLinUCBBandit at alpha=0.5,
FEATURE_DIM=10, the reward gates and their thresholds, and
`final_hardening.convergence_diagnostic` -- imported, not reimplemented, so the
convergence rule cannot drift between the two experiments.

RE-DERIVED, NOT REUSED: the reward scales and both static baselines. The
saturated selections (Static-Tput mb=32/cs=512, Static-Utility mb=8/cs=256) and
the saturated scales are properties of the saturated operating point. Carrying
them here would hand this experiment a baseline tuned for a different condition
and guarantee a CARL win for a reason that has nothing to do with adaptation. So
the full 7 x 4 = 28-configuration live plane is swept again on seed 999 under
the Poisson workload, and the same three consumers are fed from it, in the same
order: MEASUREMENTS -> frozen scales -> utility -> Static-Utility. Static-Utility
therefore cannot influence the yardstick it is ranked under.

THE FIVE TREATMENTS
===================
    Static-Tput      fixed config, argmax throughput on seed 999
    Static-Utility   fixed config, argmax utility_v2 on seed 999
    RuleOnly         classify -> DEFAULT_CONFIGS[regime]; no learning
    CARL-Repaired    RepairedLinUCB + restricted arms
    CARL-Expanded    RepairedLinUCB + expanded arms

No DynOracle -- it is not an oracle (it ranks arms by rewards collected in a
particular cycle ORDER) and was withdrawn. No inactive ablations: speculation,
routing and KV eviction cannot execute in this harness, so freezing them
measures nothing.

ARRIVALS ARE PROVED, NOT ASSERTED
=================================
A workload described as Poisson and generated as something else is
indistinguishable in a results table, so every run records the requested lambda,
the mu and rho it came from, the empirical interarrival mean and median, the
empirical rate, the phase duration, and `1/lambda` next to the mean it is
supposed to match. The raw arrival offsets are retained per seed. The generator
is deterministic in the seed, and its arrival stream is drawn from a NAMED
sub-stream so that changing the prompt-length draws cannot silently change the
arrival pattern.

TWO DIFFERENT QUANTITIES, KEPT UNDER TWO DIFFERENT NAMES
========================================================
Only one thing here is entitled to be called rho:

    offered_load.per_phase[p].empirical_rho
        = empirical_lambda_rps / mu_req_per_s

computed per phase from the SCHEDULED arrival offsets. It is a property of the
seeded draw, known before a second of GPU time is spent, and it scatters around
0.40 because that is what a Poisson process does. NO SEED IS EVER REJECTED,
RE-DRAWN OR RE-RUN because its empirical rho came out high or low. The first
seeded draw is the evidence; discarding draws that missed the target would make
the stated rho a selection effect rather than a condition.

    drain_overrun.arrival_window_overrun_ratio = wall_s / arrival_window_s

is a wall-clock observation about the ENGINE, and is NOT a load, a utilisation
or a rho. `mu` is measured on the calibrated mechanistic substrate while this
runs on a real T4, so under a genuinely sub-critical open-loop stream the run
should end shortly after its last arrival and this ratio should sit near 1.0.
Well above 1.0 means the engine, not the arrival process, set the pace, which
bounds what the run can claim. REPORTED, never used to re-tune lambda.

PUBLISHED RUNS AND SMOKE RUNS NEVER SHARE A DESTINATION
=======================================================
A published run must be on CUDA, must use exactly the pre-registered seeds, n,
grid, rho and mu artifact, and must verify the executing source against the
sidecar's archive manifest; it writes to docs/eval. Anything else is a smoke
(`--smoke`, or `--allow-cpu-smoke` on a CPU) and writes under
smoke_runs/subcritical_poisson unless `--output-dir` says otherwise -- never to
the published paths. CUDA alone does not make a run a result.

Run (provenance is captured LOCALLY, against the archive, before upload):
    python scripts/eval/repair/source_provenance.py --archive <the uploaded zip>
    python scripts/eval/repair/subcritical_poisson.py
    python scripts/eval/repair/subcritical_poisson.py --smoke --seeds 42 \\
        --limit 40 --grid-batches 4,8 --grid-chunks 128,256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_EVAL_DIR = os.path.join(_REPO_ROOT, "scripts", "eval")
_REPAIR_DIR = os.path.join(_EVAL_DIR, "repair")
for _p in (_REPO_ROOT, _EVAL_DIR, _REPAIR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# The saturated experiment's harness. IMPORTED, never reimplemented: that is
# what makes "same controller, same reward, same convergence rule" true by
# construction rather than by inspection. The only thing this module writes is
# its own artifact, at its own paths.
import ablation_live as abl  # noqa: E402
import final_hardening as fh  # noqa: E402
import saturated_reference as sref  # noqa: E402
import source_provenance as srcprov  # noqa: E402

from src.carl.config import CARLConfig  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.live import (  # noqa: E402
    ARRIVAL_POISSON, _ReqSpec, _SCENARIO_SPEC, _make_prompt, build_arrivals,
)
from src.carl.live_reward import (  # noqa: E402
    MIN_ARM_REWARD_SPREAD, MIN_LIVE_DISTINCT, MIN_LIVE_REWARD_SPREAD,
    THRESHOLD_PROVENANCE, LiveRewardV2, gate_candidate_arm_spread,
    gate_live_reward_stream,
)
from src.carl.reward import DEFAULT_WEIGHTS, DegenerateRewardError  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
from src.eval import provenance  # noqa: E402

EXPERIMENT = "subcritical_poisson"
# Bumped whenever a change would make an older checkpoint unsafe to resume from.
# /2: per-unit sweep files, the calibration digest, schema'd raw units.
# /3: runtime + model identity in the fingerprint, fresh-evidence validation,
#     terminal markers, the 2026-10-22 statistical amendments (H5, t-CI, H2).
# /4: the declared engine warm-up (ENGINE_WARMUP_VERSION) runs before the first
#     measured unit of every process; units measured without it are a different
#     measurement protocol and are never reused.
EXPERIMENT_VERSION = "subcritical_poisson/4"
# The declared engine warm-up; see engine_warmup_spec.
ENGINE_WARMUP_VERSION = "engine_warmup/1"
RAW_UNIT_SCHEMA = "subcritical_poisson/raw_unit/1"
SWEEP_UNIT_SCHEMA = "subcritical_poisson/sweep_unit/1"
CALIBRATION_SCHEMA = "subcritical_poisson/calibration/1"
FAILED_UNIT_SCHEMA = "subcritical_poisson/failed_unit/1"
TERMINAL_MARKER_SCHEMA = "subcritical_poisson/terminal_marker/1"
INVALID_UNIT_SCHEMA = "subcritical_poisson/invalid_fresh_unit/1"

# The hardware the operating point was derived on. A published run must execute
# on it: mu (and therefore lambda and rho) is a T4 measurement.
REQUIRED_GPU_NAME = "Tesla T4"
DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RESULTS_BASENAME = "subcritical_poisson_results.json"
PUBLISHED_OUTPUT_DIR = DOCS_EVAL
SMOKE_OUTPUT_DIR = os.path.join(_REPO_ROOT, "smoke_runs", "subcritical_poisson")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "subcritical_poisson")
RESULTS_PATH = os.path.join(DOCS_EVAL, RESULTS_BASENAME)
PREREG_PATH = os.path.join(DOCS_EVAL, "PREREGISTRATION_subcritical_poisson.md")
# Where the frozen saturated reference (final_hardening's results JSON and the
# raw seed files H2 reads) is verified. READ ONLY; see saturated_reference.py.
SATURATED_REFERENCE_ROOT = _REPO_ROOT

RUN_MODE_PUBLISHED = "published"
RUN_MODE_SMOKE = "smoke"

# Run outcomes, kept apart because their recovery rules are opposite: an
# operational interruption may be resumed; a scientific gate failure may not,
# because re-measuring until the gate passes would turn the gate into a retry.
STATUS_OPERATIONAL = "operational_interruption"
STATUS_TERMINAL_GATE = "terminal_scientific_gate_failure"
# Freshly measured evidence that fails the same validation a resume applies.
# Not an interruption (re-measuring would silently regenerate it) and not a
# scientific result: a validity defect that stops the run for investigation.
STATUS_INVALID_EVIDENCE = "invalid_fresh_evidence"


class RunPaths:
    """Every file one run may write, all under ONE output directory.

    With `output_dir == docs/eval` these are exactly the published paths
    (RESULTS_PATH, RAW_DIR). A smoke run gets a different directory, so a smoke
    and a published run cannot write to the same file.
    """

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.results = os.path.join(self.output_dir, RESULTS_BASENAME)
        self.raw_dir = os.path.join(self.output_dir, "raw", "subcritical_poisson")
        self.sweep_dir = os.path.join(self.raw_dir, "sweep")
        self.failed_dir = os.path.join(self.raw_dir, "failed")
        self.invalid_dir = os.path.join(self.raw_dir, "invalid")
        self.superseded_dir = os.path.join(self.raw_dir, "superseded")
        self.calibration = os.path.join(self.raw_dir, "calibration.json")
        # OUTSIDE raw_dir and never the results file: a start-up failure is
        # recorded here and touches no evidence.
        self.preflight_dir = os.path.join(
            self.output_dir, "subcritical_poisson_preflight_errors")

    def eval_raw(self, name: str, seed: int) -> str:
        return os.path.join(self.raw_dir, f"{name}_seed{int(seed):03d}.json")

    def sweep_raw(self, mb: int, cs: int) -> str:
        return os.path.join(self.sweep_dir, f"mb{int(mb)}_cs{int(cs)}.json")

    def evidence_files(self) -> list:
        """Every unit / calibration / failure file currently on disk."""
        found = []
        for d in (self.raw_dir, self.sweep_dir, self.failed_dir, self.invalid_dir):
            if os.path.isdir(d):
                found += [os.path.join(d, f) for f in sorted(os.listdir(d))
                          if f.endswith(".json")]
        return found

    def as_dict(self) -> dict:
        return {k: os.path.relpath(v, _REPO_ROOT).replace(os.sep, "/")
                for k, v in vars(self).items()}

# ---------------------------------------------------------------------------
# The operating point.
# ---------------------------------------------------------------------------

TARGET_RHO = 0.40

# The artifact the service rates are read from. A RELATIVE path, resolved
# against the repo root, so the derivation is reproducible from the archive.
MU_SOURCE_ARTIFACT = os.path.join("docs", "eval", "raw", "repair",
                                  "load_sweep_b7.json")

# The T4 run whose least-squares fit produced the eager decode parameters the
# saturation probe used. Cross-checked at run time by assert_arm_evidence().
MU_CALIBRATION_ARTIFACT = os.path.join(
    "docs", "eval", "raw", "repair",
    "batch_intervention_phase15_16_s42_s43_s44.json")

# Which saturation row belongs to which phase of this harness's workload. The
# left side is this experiment's phase name; the right side is the shape key in
# the load-sweep artifact.
PHASE_SHAPE_KEY = {"interactive": "interactive", "batch": "batch"}

# Named arrival sub-stream. Prompt lengths and arrival offsets are drawn from
# SEPARATE generators so that a change to one cannot silently perturb the other
# -- the arrival process is the independent variable here and must be isolated.
ARRIVAL_RNG_NAMESPACE = "subcritical_poisson/arrivals"

# ---------------------------------------------------------------------------
# Everything below is READ from the saturated experiment so it cannot drift.
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = list(fh.DEFAULT_SEEDS)              # 42..51
DEFAULT_REQUESTS = fh.DEFAULT_REQUESTS              # 200
VALIDATION_SEED = fh.VALIDATION_SEED                # 999, held out
OBSERVE_INTERVAL = fh.OBSERVE_INTERVAL              # 10
ALPHA = fh.ALPHA                                    # 0.5
GRID_BATCH_AXIS = list(fh.GRID_BATCH_AXIS)          # 2..32
GRID_CHUNK_AXIS = list(fh.GRID_CHUNK_AXIS)          # 64..512
LIVE_DIMENSIONS = fh.LIVE_DIMENSIONS
INERT_DIMENSIONS = dict(fh.INERT_DIMENSIONS)
CONVERGENCE_RULE = fh.CONVERGENCE_RULE              # the SAME object
_ARM_SET_FOR = dict(fh._ARM_SET_FOR)

CONFIGS = ["Static-Tput", "Static-Utility", "RuleOnly", "CARL-Repaired",
           "CARL-Expanded"]
_STATIC_CONFIGS = {"Static-Tput", "Static-Utility"}
_CONTROLLER_CONFIGS = {"RuleOnly", "CARL-Repaired", "CARL-Expanded"}

# Every comparison, on BOTH metrics. `metric` names the pre-registered primary;
# the other is reported as a secondary so nothing is chosen after the fact.
COMPARISONS = [
    ("A_adaptation_vs_throughput_baseline", "CARL-Repaired", "Static-Tput",
     "throughput_tps",
     "Q2. Under sub-critical Poisson arrivals, does adaptation beat the best "
     "fixed config FOR THROUGHPUT, on the metric that baseline was selected "
     "for?"),
    ("B_adaptation_vs_utility_baseline", "CARL-Repaired", "Static-Utility",
     "run_utility_v2",
     "Q3. Does adaptation beat the best fixed config FOR CARL'S OWN OBJECTIVE, "
     "on that objective?"),
    ("C_learning_isolation", "CARL-Repaired", "RuleOnly", "throughput_tps",
     "Q4. What did LEARNING buy, holding the classifier and the arm set fixed?"),
    ("D_action_space_isolation", "CARL-Expanded", "CARL-Repaired",
     "throughput_tps",
     "Q5. What did a WIDER ACTION SPACE buy, holding the learner fixed?"),
]

# H2, as amended 2026-10-22. The denominator is the COMPARATOR's mean
# throughput -- the treatment the adaptive arm is measured against -- and the
# identical definition applies at both operating points. Comparison B is
# excluded: its outcome is utility_v2, which has no natural zero, and no
# relative-utility hypothesis is created in its place.
H2_COMPARATORS = {
    "A_adaptation_vs_throughput_baseline": "Static-Tput",
    "C_learning_isolation": "RuleOnly",
    "D_action_space_isolation": "CARL-Repaired",
}
H2_DEFINITION = {
    "formula": ("abs(mean over seeds of (a_s - b_s)) / mean over seeds of b_s, "
                "on throughput_tps, where b is the comparator"),
    "comparators": dict(H2_COMPARATORS),
    "excluded": {"B_adaptation_vs_utility_baseline": (
        "outcome is utility_v2 (no natural zero); no relative utility "
        "hypothesis is created")},
    "same_definition_at_both_operating_points": True,
    "saturated_side_requires": (
        "the frozen saturated artifact's per-config throughput_tps_mean for "
        "Static-Tput, RuleOnly and CARL-Repaired over seeds 42..51, plus the "
        "per-seed throughput of each compared pair (or their means over the "
        "same seeds). Never reconstructed from rounded paired differences."),
    "saturated_side_source": (
        "saturated_reference.load_verified: the hash-verified frozen results "
        "JSON, its per-seed lists aligned to seeds through the hash-verified "
        "raw seed files, put through THIS module's h2_relative_difference"),
}

RESEARCH_QUESTIONS = {
    "Q1": ("How much flatter is the live configuration landscape under rho ~ "
           "0.40 Poisson arrivals than under the frozen saturated burst? "
           "Reported descriptively from the 28-point held-out grid; no "
           "significance test is applied to 28 deterministic search points."),
    "Q2": "Does CARL-Repaired beat Static-Tput on throughput? (comparison A)",
    "Q3": ("Does CARL-Repaired beat Static-Utility on end-of-run utility_v2? "
           "(comparison B)"),
    "Q4": "Does CARL-Repaired improve over RuleOnly? (comparison C)",
    "Q5": "Does Expanded help or hurt relative to Repaired? (comparison D)",
    "Q6": ("Does CARL satisfy the SAME pre-registered convergence rule under "
           "sub-critical load? Same diagnostic function, same thresholds, "
           "imported from final_hardening."),
    "Q7": ("Does the relationship between knob leverage and adaptation headroom "
           "change relative to the saturated operating point? Answered by "
           "reading Q1's landscape spread against A-D's paired differences."),
}

PRE_REGISTRATION = {
    "registered_before_run": True,
    "document": "docs/eval/PREREGISTRATION_subcritical_poisson.md",
    "mechanistic_hypothesis": (
        "The sub-critical Poisson operating point will expose less "
        "configuration-dependent throughput headroom than the saturated burst "
        "operating point. Therefore the value available to online adaptation "
        "should be smaller relative to its finite-horizon search and "
        "reconfiguration cost."),
    "H1_landscape_flattens": {
        "statement": (
            "The relative throughput range across the 28-configuration held-out "
            "grid will be SMALLER at rho ~ 0.40 than at the saturated burst "
            "point, because below saturation the batch cap is arrival-bounded "
            "rather than config-bounded and therefore does not bind."),
        "question": "Q1", "metric": "relative_throughput_range_pct on the held-out grid",
        "prediction": ("relative throughput range strictly smaller than the "
                       "saturated sweep's"),
        "falsified_if": ("the relative throughput range is equal to or larger "
                         "than the saturated sweep's"),
        "note": ("descriptive comparison of two deterministic 28-point grids; "
                 "no p-value is computed and none would mean anything"),
    },
    "H2_adaptation_headroom_shrinks": {
        "statement": (
            "Because less configuration-dependent headroom is available, the "
            "MAGNITUDE of every paired difference between treatments will be "
            "smaller, in relative terms, than at the saturated point. This is a "
            "prediction about effect SIZE, not about which side wins."),
        "question": "Q7",
        "metric": ("|mean paired throughput difference| / mean comparator "
                   "throughput (amended 2026-10-22)"),
        "definition": H2_DEFINITION,
        "prediction": ("|relative paired difference| smaller than the saturated "
                       "run's for comparisons A, C and D"),
        "falsified_if": ("any of A, C or D shows a LARGER relative paired "
                         "difference than at the saturated point"),
        "unresolved_if": ("the frozen saturated artifact's comparator means are "
                          "unavailable: then H2 is neither supported nor "
                          "falsified"),
        "saturated_reference_recovered": (
            "2026-10-22, after the operational T4 smoke was inspected and "
            "before the full run; the definition above is unchanged"),
        "superseded_wording": {"metric": "|mean paired difference| / mean level"},
    },
    "H3_direction_is_not_preregistered": {
        "statement": (
            "NO winner is pre-registered for comparisons A, B, C or D. The "
            "mechanistic hypothesis predicts that the STAKES shrink, and is "
            "silent on the sign. All three outcomes below were written down "
            "before the GPU ran and all three are informative."),
        "case_A": {
            "outcome": "CARL still loses to objective-matched static tuning",
            "interpretation": ("the static advantage -- the general limitation "
                               "of finite-horizon online adaptation in this "
                               "setting -- survives across offered-load regimes, "
                               "which widens the scope of the saturated result "
                               "rather than confining it"),
        },
        "case_B": {
            "outcome": ("CARL becomes competitive with, or exceeds, an "
                        "objective-matched static baseline"),
            "interpretation": ("offered load is a BOUNDARY CONDITION on the "
                               "saturated finding: the saturated result is then "
                               "scoped to rho >> 1 and must be reported as such"),
        },
        "case_C": {
            "outcome": "CARL improves over RuleOnly but not over static",
            "interpretation": ("adaptive behaviour is useful but finite-horizon "
                               "adaptation economics remain unfavourable; the "
                               "saturated conclusion holds in shape across "
                               "regimes even if its magnitudes change"),
        },
    },
    "H4_cold_start": {
        "statement": (
            "Every adaptive run will show a non-empty warm-up: at least one "
            "leading control cycle whose reward is withheld for want of a "
            "measured TTFT/TPOT. POSITIVE CONTROL on the repair carried over "
            "from final_hardening -- a warm-up of zero everywhere would mean the "
            "validity gate never engaged."),
        "metric": "reward_validity.warmup_cycles",
        "prediction": "warmup_cycles >= 1 for every controller run",
        "falsified_if": "any controller run reports warmup_cycles == 0",
        "note": ("under Poisson arrivals the warm-up may be LONGER than at the "
                 "saturated point, because fewer requests are in flight early "
                 "and the first completion arrives later. A longer warm-up is "
                 "an observation, not a failure."),
    },
    "H5_arrival_process": {
        "statement": (
            "The generated workload will be the workload described: per phase, "
            "the empirical mean interarrival time will be consistent with "
            "1/lambda for the derived lambda."),
        "metric": "arrival_validation.<phase>.empirical_mean_interarrival_s",
        "amended": "2026-10-22, before any GPU run (see the document's "
                   "corrections log, including the prior-exposure disclosure)",
        "reference_distribution": (
            "the mean of k consecutive Exp(lambda) gaps is Gamma(shape=k, "
            "rate=k*lambda); k = 99 per phase at n = 200"),
        "family": ("20 cells: evaluation seeds 42..51 x {interactive, batch}; "
                   "exact two-sided Gamma bounds at alpha = 0.05/20 per cell "
                   "(Bonferroni: familywise error <= 0.05 without assuming "
                   "independence)"),
        "held_out_seed": ("seed 999 is NOT in the family; it is reported "
                          "separately against its own exact 95% bounds"),
        "prediction": "every family cell inside its Bonferroni bounds",
        "family_rejection_means": (
            "statistical inconsistency of the realised draws with Exp(lambda) "
            "that requires investigation. It does not by itself prove a "
            "generator defect, does not automatically invalidate every "
            "conclusion, and does not authorise changing seeds, lambda or the "
            "workload. Every cell, rejected or not, is preserved and reported."),
        "descriptive_only": ("the per-cell normal-approximation 95% interval "
                             "(mean_within_ci95) and empirical rho"),
        "no_seed_rejection": "no seed is rejected, replaced or retried",
        "superseded_wording": {
            "prediction": ("empirical mean within the reported 95% interval "
                           "around 1/lambda for every phase and seed"),
            "falsified_if": ("the empirical mean is outside that interval, which "
                             "would mean the run does not sit at the stated rho "
                             "and no load-regime conclusion may be drawn from "
                             "it"),
        },
    },
    "paired_interval": {
        "amended": "2026-10-22, before any GPU run",
        "primary": ("mean_difference +/- t(0.975, n-1) * sd / sqrt(n), sd the "
                    "sample SD of the paired differences; t computed for the "
                    "actual n (2.2621571628 at n = 10)"),
        "secondary": ("mean_difference +/- 1.96 * sd / sqrt(n), labelled a "
                      "normal approximation, kept ONLY for comparison with the "
                      "frozen saturated analysis, whose intervals are unchanged"),
        "scope": ("per-comparison intervals; no familywise adjustment across "
                  "the four comparisons or two metrics"),
        "superseded_wording": "95% paired CI = mean +/- 1.96 * sd / sqrt(n)",
    },
    "engine_warmup": {
        "amended": ("2026-10-22, AFTER inspecting the T4 wiring smoke (seed 42, "
                    "40 requests, 2x2 sweep): an operational amendment to the "
                    "measurement protocol, not to any hypothesis or quantity"),
        "observed": ("sweep point mb4_cs128 -- the first measured unit of the "
                     "process -- had its first token at 8.10 s and a 7.91 s "
                     "maximum submission lag (TTFT p99 10858 ms); the other "
                     "three points ~0.07 s / ~0.05 s. Attribution (Triton JIT, "
                     "CUDA / cuBLAS initialisation, other) was NOT measured"),
        "protocol": ENGINE_WARMUP_VERSION,
        "rule": ("the declared warm-up runs once per process before the first "
                 "measured unit, on disposable scheduler state, and is excluded "
                 "from every measurement; see engine_warmup_spec"),
        "unchanged": ("rho, mu/lambda, seeds, n, grid, treatments, workload, "
                      "cadence, reward / scales / gates, arms, static-selection "
                      "objectives, convergence rule, and the controller's "
                      "reward-validity cold start"),
        "comparability": ("the saturated final-hardening runs had no engine "
                          "warm-up; whether their first unit absorbed a similar "
                          "start-up cost is not verified here"),
    },
    "saturated_reference_integration": {
        "recorded": ("2026-10-22, AFTER inspecting the operational T4 smoke "
                     "and BEFORE the full sub-critical run"),
        "what": ("the frozen final_hardening results JSON was recovered from "
                 "an archive and is now read, hash-verified, for the "
                 "saturated side of H1 and H2"),
        "source_zip_sha256": sref.RECOVERY["source_zip_sha256"],
        "artifact_sha256": sref.EXPECTED_RESULTS_SHA256,
        "unchanged": ("H1's relative-range definition and denominator, H2's "
                      "amended definition and comparators, every hypothesis, "
                      "prediction and falsification criterion, and every "
                      "experimental quantity"),
        "no_smoke_evidence": "no smoke measurement is used as evidence",
    },
    "note": (
        "The mechanistic hypothesis predicts a shrinking EFFECT SIZE, not a "
        "winner. Registering the three outcome cases before the run is what "
        "stops whichever one occurs from being narrated as the expected one."),
}

# The frozen saturated result, for the DESCRIPTIVE comparison Q1/Q7 require.
# The figures below were transcribed before the artifact was recovered and are
# kept as that historical record. H1/H2 never read them: their saturated side
# comes only from the hash-verified artifact (saturated_reference.py). NOTHING
# in this module writes to the final hardening's artifact or paths.
SATURATED_REFERENCE = {
    "source_artifact": "docs/eval/final_hardening_results.json",
    "source_artifact_present_in_repo": True,
    "recovery": dict(sref.RECOVERY),
    "verified_by": ("saturated_reference.load_verified (validator "
                    f"{sref.VALIDATOR_VERSION}); the per-run record is "
                    "results.saturated_reference_verification"),
    "superseded_note": (
        "until 2026-10-22 the results JSON was absent from this working tree "
        "and H1/H2 were recorded as unresolved_missing_saturated_artifact; the "
        "transcribed `comparisons` below are ROUNDED, are kept only as the "
        "historical transcription, and are never used for H1 or H2"),
    "operating_point": "saturated / non-stationary burst (bulk dump), rho >> 1",
    "hardware": "Tesla T4, TinyLlama-1.1B-Chat-v1.0, fp16",
    "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    "n_requests": 200,
    "held_out_seed": 999,
    "live_plane": "max_batch_size x chunk_size = 7 x 4 = 28",
    "static_tput": {"max_batch_size": 32, "chunk_size": 512},
    "static_utility": {"max_batch_size": 8, "chunk_size": 256},
    "comparisons": {
        "A_CARL_Repaired_minus_Static_Tput": {
            "metric": "throughput_tps", "mean_difference": -3.3678,
            "ci95": [-3.6633, -3.0723], "cohens_d_paired": -7.06395},
        "B_CARL_Repaired_minus_Static_Utility": {
            "metric": "run_utility_v2", "mean_difference": -0.0245,
            "ci95": [-0.0263, -0.0227], "cohens_d_paired": -8.44215},
        "C_CARL_Repaired_minus_RuleOnly": {
            "metric": "throughput_tps", "mean_difference": +3.7633,
            "ci95": [+3.2932, +4.2335], "cohens_d_paired": +4.96133,
            "secondary_run_utility_v2": {"mean_difference": +0.0363,
                                         "ci95": [+0.0347, +0.0380]}},
        "D_CARL_Expanded_minus_CARL_Repaired": {
            "metric": "throughput_tps", "mean_difference": -1.7956,
            "ci95": [-2.2754, -1.3158],
            "secondary_run_utility_v2": {"mean_difference": -0.0277,
                                         "ci95": [-0.0295, -0.0260]}},
    },
    # What H1/H2 read from the frozen artifact -- verified, full precision.
    # If verification fails, these are reported as missing, never estimated.
    "inputs_for_hypotheses": {
        "H1": ["the 28-point held-out grid's throughput (reward_calibration."
               "observations == static_selection.ranked_table), whose min and "
               "max equal static_selection.objective_spread.throughput_tps"],
        "H2": ["configs.<Static-Tput|RuleOnly|CARL-Repaired|CARL-Expanded>."
               "_per_seed.throughput_tps, aligned to seeds 42..51 through the "
               "raw seed files",
               "the artifact's SHA256, seeds, n, hardware and experiment "
               "identity"],
    },
    "convergence_supported_seeds": {"RuleOnly": 0, "CARL-Repaired": 0,
                                    "CARL-Expanded": 0, "out_of": 10},
    "cold_start": {"warmup_cycles_per_run": 3,
                   "withheld_after_first_valid": 0},
    "allowed_wording": (
        "At the saturated T4 operating point, adaptive reconfiguration is "
        "useful but insufficient: CARL-Repaired improves throughput and "
        "end-of-run utility over RuleOnly, but remains below objective-matched "
        "static configurations. Expanded live configuration coverage further "
        "reduces both metrics."),
}

# What a cross-regime comparison can and cannot isolate. Reported with H1/H2.
SATURATED_COMPARABILITY = {
    "runtime_stack": ("the frozen saturated run recorded torch 2.11.0+cu128 / "
                      "CUDA 12.8; the current T4 smoke recorded torch "
                      "2.11.0+cu130 / CUDA 13.0. The current runtime is "
                      "recorded as-is and is not downgraded"),
    "engine_warmup": ("the saturated harness had no declared engine warm-up; "
                      "subcritical /4 runs engine_warmup/1"),
    "model_revision": ("no model revision was recorded for the saturated run; "
                       "the same repository name does not prove identical "
                       "weights"),
    "interpretation": ("cross-regime comparisons are DESCRIPTIVE. They cannot "
                       "cleanly isolate offered load from every implementation "
                       "and runtime difference between the two runs"),
}


# The two utility estimators, and the exact wording permitted about the static
# utility baseline. Module-level so it is one auditable string rather than prose
# buried in a driver -- and so a test can read the words themselves.
UTILITY_ESTIMATORS = {
    "run_utility_v2": (
        "utility_v2 of a run's END-OF-RUN aggregates under the frozen scales. "
        "Defined for EVERY treatment including the statics, and the basis of "
        "comparison B."),
    "mean_valid_cycle_utility": (
        "mean over a controller's SCORED control cycles -- what the learner "
        "actually consumed. Reported for controllers only and NEVER compared "
        "against the statics, which have no control cycles. The two are not "
        "interchangeable."),
    "static_utility_selection_wording": (
        "Static-Utility is selected using the same utility_v2 function, weights, "
        "and frozen scales used by CARL, and both systems are compared on "
        "end-of-run utility_v2."),
    "forbidden_wording": (
        "It is NOT claimed to optimise the exact sequence of per-cycle rewards "
        "CARL consumed. Mean valid cycle reward and end-of-run utility_v2 are "
        "different estimators and are never substituted for one another."),
}


class ResumeMismatchError(RuntimeError):
    """A resume was asked for against state that does not belong to this run.

    Raised rather than worked around. Silently resuming across changed code,
    configuration or provenance would splice two different experiments into one
    artifact, and nothing downstream could tell.
    """


class AmbiguousServiceRateError(RuntimeError):
    """The service-rate source cannot support the derivation without guessing.

    Raised rather than defaulted. A lambda invented because an artifact was
    missing a row would put the run at an unknown rho, and the whole point of
    this experiment is that rho is a stated condition.
    """


# ===========================================================================
# 1. Service rates -> arrival rates. Derived from an artifact, never chosen.
# ===========================================================================


def execution_arm_for(sched) -> str:
    """'graph' iff a CUDA-graph runner is ACTUALLY attached; else 'eager'.

    Reads the object that will execute, not `CARLConfig.use_cuda_graphs`. The
    flag defaults True while `ablation_live._new_scheduler` attaches no runner
    and `ContinuousBatchScheduler._decode_forward` refuses to replay without
    one, so trusting the flag would select service rates ~1.9x too high and
    silently move this run from rho=0.40 to rho~0.75.
    """
    return "graph" if getattr(sched, "_graph_runner", None) is not None else "eager"


def _abs_artifact(rel_or_abs: str) -> str:
    return (rel_or_abs if os.path.isabs(rel_or_abs)
            else os.path.join(_REPO_ROOT, rel_or_abs))


def _positive_finite(value, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AmbiguousServiceRateError(
            f"{where} is {value!r}, which is not a number; refusing to guess a "
            "service rate")
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise AmbiguousServiceRateError(
            f"{where} is {v!r}; a service rate must be finite and > 0")
    return v


def assert_arm_evidence(artifact: dict, arm: str,
                        calibration_path: str | None = None) -> dict:
    """Prove the chosen arm's rows rest on a distinct, hardware-fitted cost model.

    Two things must hold before an eager mu may be used as an EAGER mu:

      1. the sweep's hardware profile must carry SEPARATE eager and CUDA-graph
         decode parameters, so the two arms are distinguishable at all;
      2. the eager parameters must equal the eager least-squares fit from the
         real T4 intervention run, so "eager" names a measured execution path
         rather than a label in a config.

    Returns the evidence block. Raises AmbiguousServiceRateError if either
    fails: a run that cannot say which execution path its capacity estimate
    describes cannot state its own rho.
    """
    hw = artifact.get("hardware_profile")
    if not isinstance(hw, dict):
        raise AmbiguousServiceRateError(
            "the service-rate artifact carries no `hardware_profile` block, so "
            "there is no way to tell which execution path its saturation rows "
            "describe")

    pairs = {
        "decode_per_row_s": ("decode_per_row_s", "cuda_graph_decode_per_row_s"),
        "decode_fixed_per_step_s": ("decode_fixed_per_step_s",
                                    "cuda_graph_decode_fixed_per_step_s"),
    }
    eager_params, graph_params = {}, {}
    for label, (ek, gk) in pairs.items():
        eager_params[label] = _positive_finite(hw.get(ek), f"hardware_profile.{ek}")
        graph_params[label] = _positive_finite(hw.get(gk), f"hardware_profile.{gk}")
        if eager_params[label] == graph_params[label]:
            raise AmbiguousServiceRateError(
                f"hardware_profile.{ek} == hardware_profile.{gk}: the eager and "
                "CUDA-graph cost models are identical, so the two saturation "
                "arms cannot be distinguished and an 'eager' mu means nothing")

    evidence = {
        "execution_arm": arm,
        "why_this_arm": (
            "read from the live scheduler object: `_graph_runner is None`, so "
            "`ContinuousBatchScheduler._decode_forward` falls back to eager on "
            "every decode regardless of `CARLConfig.use_cuda_graphs`"
            if arm == "eager" else
            "a CUDA-graph runner IS attached to the live scheduler, so the "
            "graph capacity rows are the ones that describe execution"),
        "eager_decode_parameters": eager_params,
        "cuda_graph_decode_parameters": graph_params,
        "arms_are_distinct": True,
    }

    if calibration_path is None:
        evidence["hardware_fit_crosscheck"] = {
            "checked": False,
            "why": "no calibration artifact was supplied to cross-check against",
        }
        return evidence

    path = _abs_artifact(calibration_path)
    if not os.path.isfile(path):
        # A NAMED cross-check whose evidence is absent is a failed check, not a
        # skipped one; only calibration_path=None opts out, and says so above.
        raise AmbiguousServiceRateError(
            f"calibration artifact {calibration_path} is missing, so the eager "
            "saturation rows cannot be attributed to the measured non-graph "
            "path. It is a required archive member; rebuild the archive.")

    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    fit = (cal.get("analysis") or {}).get("hardware_profile_fit_eager") or {}
    graph_fit = (cal.get("analysis") or {}).get(
        "hardware_profile_fit_cuda_graph") or {}
    matches = (fit.get("decode_per_row_s") == eager_params["decode_per_row_s"]
               and fit.get("fixed_per_step_s")
               == eager_params["decode_fixed_per_step_s"])
    if arm == "eager" and not matches:
        raise AmbiguousServiceRateError(
            "the sweep's eager decode parameters do not equal the eager "
            "least-squares fit in {}: sweep={} fit={}. The eager saturation "
            "rows can no longer be attributed to the measured non-graph "
            "execution path.".format(calibration_path, eager_params, fit))
    evidence["hardware_fit_crosscheck"] = {
        "checked": True,
        "path": calibration_path,
        "eager_fit": {k: fit.get(k) for k in
                      ("decode_per_row_s", "fixed_per_step_s", "r_squared",
                       "n_points")},
        "cuda_graph_fit": {k: graph_fit.get(k) for k in
                           ("decode_per_row_s", "fixed_per_step_s", "r_squared",
                            "n_points")},
        "eager_parameters_match_fit": bool(matches),
        "measured_on": (cal.get("_provenance") or {}).get("gpu"),
        "model": (cal.get("_provenance") or {}).get("model"),
        "dtype": (cal.get("_provenance") or {}).get("dtype"),
        "why_this_matters": (
            "it is what makes 'eager' a MEASURED execution path (54 rows, real "
            "Tesla T4, R^2 reported above) rather than a label; the CUDA-graph "
            "fit is a separate regression from the same run, which is why the "
            "two arms are distinguishable at all"),
    }
    return evidence


def lambda_for_rho(mu: float, rho: float) -> float:
    """lambda = rho * mu, with the sub-critical condition enforced, not assumed."""
    mu = _positive_finite(mu, "mu")
    if not isinstance(rho, (int, float)) or isinstance(rho, bool):
        raise ValueError(f"rho must be a number, got {rho!r}")
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        raise ValueError(
            f"rho={rho} is not sub-critical; this experiment is DEFINED by an "
            "offered load comfortably below capacity, and a rho >= 1 would make "
            "it the saturated experiment under a different name")
    return rho * mu


def derive_service_rates(arm: str, *, artifact_path: str = MU_SOURCE_ARTIFACT,
                         calibration_path: str | None = MU_CALIBRATION_ARTIFACT,
                         rho: float = TARGET_RHO) -> dict:
    """Read mu per phase for `arm` and turn it into lambda per phase.

    Mechanical: the only free choices are TARGET_RHO (stated, pre-registered)
    and the execution arm (read from the live scheduler). Every ambiguity --
    missing artifact, missing block, missing row, unusable value, indistinct
    arms -- raises rather than defaulting, because a lambda chosen to keep the
    script running puts the run at an unknown rho.
    """
    path = _abs_artifact(artifact_path)
    if not os.path.isfile(path):
        raise AmbiguousServiceRateError(
            f"service-rate artifact {artifact_path} is missing. Do NOT invent "
            "lambda and do NOT estimate mu from memory: re-run "
            "scripts/eval/repair/load_sweep.py, or point --mu-artifact at the "
            "artifact that exists.")
    with open(path, encoding="utf-8") as f:
        artifact = json.load(f)

    saturation = artifact.get("saturation")
    if not isinstance(saturation, dict) or not saturation:
        raise AmbiguousServiceRateError(
            f"{artifact_path} carries no usable `saturation` block; there is no "
            "measured capacity to derive an arrival rate from")

    evidence = assert_arm_evidence(artifact, arm, calibration_path)

    phases = {}
    for phase, shape_key in PHASE_SHAPE_KEY.items():
        key = f"{shape_key}@{arm}"
        row = saturation.get(key)
        if not isinstance(row, dict):
            raise AmbiguousServiceRateError(
                "{} has no saturation row {!r}. Available rows: {}. The "
                "execution path that will actually run has no measured "
                "capacity in this artifact, so its rho cannot be stated."
                .format(artifact_path, key, sorted(saturation)))
        mu = _positive_finite(row.get("mu_max_req_per_s"),
                              f"saturation[{key!r}].mu_max_req_per_s")
        lam = lambda_for_rho(mu, rho)
        phases[phase] = {
            "shape_key": shape_key,
            "saturation_row": key,
            "mu_max_req_per_s": mu,
            "mu_method": row.get("method"),
            "mu_n_seeds": row.get("n_seeds"),
            "mu_n_requests": row.get("n_requests"),
            "target_rho": rho,
            "lambda_rps": lam,
            "expected_mean_interarrival_s": 1.0 / lam,
        }

    return {
        "method": "lambda_phase = target_rho * mu_phase, per phase",
        "target_rho": rho,
        "execution_arm": arm,
        "source_artifact": artifact_path,
        "source_artifact_substrate": artifact.get("substrate"),
        "source_artifact_calibration_source": artifact.get("calibration_source"),
        "source_artifact_provenance": {
            k: (artifact.get("_provenance") or {}).get(k)
            for k in ("experiment", "script", "git_sha", "git_dirty",
                      "timestamp_utc", "substrate")},
        "rows_used": [p["saturation_row"] for p in phases.values()],
        "fields_used": ["saturation.<shape>@<arm>.mu_max_req_per_s",
                        "hardware_profile.decode_per_row_s",
                        "hardware_profile.decode_fixed_per_step_s",
                        "hardware_profile.cuda_graph_decode_per_row_s",
                        "hardware_profile.cuda_graph_decode_fixed_per_step_s"],
        "arm_evidence": evidence,
        "phases": phases,
        "why_phase_specific": (
            "the two phases differ in capacity by more than 2x (interactive "
            "~2.96 vs batch ~1.41 req/s eager), so a single lambda would place "
            "one phase at roughly twice the other's offered load and the "
            "sub-critical condition would hold for only one of them"),
        "known_limitation": (
            "mu is measured on the calibrated mechanistic substrate "
            "(src/eval/engine_model) whose decode parameters are fitted to a "
            "real T4 run, not measured end-to-end on the T4 under THIS "
            "harness's prompt-length windows. The load-sweep shapes are close "
            "but not identical: its interactive shape is prompt mean 32 / "
            "output mean 32 against this harness's uniform prompt 16-32 with "
            "max_new 32, and its batch shape is prompt mean 220 / output mean "
            "64 against uniform prompt 128-256 with max_new 64. rho=0.40 is "
            "therefore a DERIVED TARGET. `offered_load.per_phase[*]."
            "empirical_rho` in every run is the realised offered load of the "
            "seeded draw, and `drain_overrun` says whether the engine kept up "
            "with it on hardware. Neither is ever used to re-tune lambda."),
    }


# ===========================================================================
# 2. The Poisson workload. Same shapes, same phase split, different arrivals.
# ===========================================================================


def _arrival_rng(seed: int) -> random.Random:
    """The arrival sub-stream for `seed`.

    Deterministic in the seed and INDEPENDENT of the prompt-length draws, so a
    change to shape sampling cannot silently alter the arrival process -- which
    is the one thing this experiment varies.
    """
    return random.Random(f"{ARRIVAL_RNG_NAMESPACE}:{seed}")


def poisson_arrivals(n: int, seed: int, plan: dict) -> dict:
    """The seeded arrival offsets alone, per phase -- no tokenizer needed.

    The ONE place the arrival stream is drawn, so a resume can regenerate the
    exact offsets a raw file claims and reject one that does not match. Phase 1
    continues from where phase 0's arrival window ended, so the regime flip is a
    change in the STREAM rather than a synchronised dump.
    """
    half = n // 2
    rng_arr = _arrival_rng(seed)
    a0 = build_arrivals(half, ARRIVAL_POISSON,
                        plan["phases"]["interactive"]["lambda_rps"], rng_arr)
    t1 = max(a0) if a0 else 0.0
    a1 = build_arrivals(n - half, ARRIVAL_POISSON,
                        plan["phases"]["batch"]["lambda_rps"], rng_arr,
                        t_offset=t1)
    return {"interactive": list(a0), "batch": list(a1)}


def build_poisson_workload(tokenizer, n: int, seed: int, plan: dict):
    """The final hardening's workload with Poisson arrivals at two rates.

    Held identical to `ablation_live._build_workload`'s NON-STATIONARY case:
    the same `_SCENARIO_SPEC` prompt/output windows, the same split at `n // 2`,
    the same phase labels, and phase-1 arrivals spliced onto the end of phase
    0's arrival window via `build_arrivals(t_offset=...)`.

    Changed: each phase draws its interarrival times at ITS OWN lambda.

    Returns (specs, arrivals) where `arrivals` maps phase name -> the list of
    scheduled offsets, retained so the generated process can be audited.
    """
    half = n // 2
    inter = _SCENARIO_SPEC["INTERACTIVE"]
    batch = _SCENARIO_SPEC["BATCH"]

    rng_shape = random.Random(seed)
    arrivals = poisson_arrivals(n, seed, plan)
    a0, a1 = arrivals["interactive"], arrivals["batch"]

    specs = []
    for i in range(half):
        L = rng_shape.randint(inter["prompt_lo"], inter["prompt_hi"])
        specs.append(_ReqSpec(f"i{i}", _make_prompt(tokenizer, L),
                              inter["max_new"], 0, arrival_offset_s=a0[i]))
    for i in range(n - half):
        L = rng_shape.randint(batch["prompt_lo"], batch["prompt_hi"])
        specs.append(_ReqSpec(f"b{i}", _make_prompt(tokenizer, L),
                              batch["max_new"], 1, arrival_offset_s=a1[i]))

    # The splice is an INVARIANT, not an expectation. Two different lambdas are
    # only meaningful if each governs its own, non-overlapping stretch of the
    # stream; an overlap would mean neither phase ran at the rate it claims.
    assert_phase_splice(specs, arrivals, n)
    return specs, arrivals


class PhaseSpliceError(RuntimeError):
    """The two phases are not sequential, so neither ran at its stated lambda."""


def phase_boundary_report(specs: list, arrivals: dict, n: int) -> dict:
    """Where the INTERACTIVE stream ends and the BATCH stream begins."""
    a0 = arrivals.get("interactive") or []
    a1 = arrivals.get("batch") or []
    half = n // 2
    first_i = min(a0) if a0 else None
    last_i = max(a0) if a0 else None
    first_b = min(a1) if a1 else None
    last_b = max(a1) if a1 else None
    gap = ((first_b - last_i)
           if (first_b is not None and last_i is not None) else None)
    head = [s.phase for s in specs[:half]]
    tail = [s.phase for s in specs[half:]]
    return {
        "n_requests": len(specs),
        "split_index": half,
        "n_interactive": len(a0),
        "n_batch": len(a1),
        "first_interactive_arrival_s": first_i,
        "last_interactive_arrival_s": last_i,
        "first_batch_arrival_s": first_b,
        "last_batch_arrival_s": last_b,
        "phase_boundary_gap_s": gap,
        "phases_do_not_overlap": bool(gap is not None and gap >= 0.0),
        "head_is_all_interactive": all(p == 0 for p in head),
        "tail_is_all_batch": all(p == 1 for p in tail),
        "batch_requests_in_head": sum(1 for p in head if p == 1),
        "interactive_requests_in_tail": sum(1 for p in tail if p == 0),
        "boundary_convention": (
            "build_arrivals places the first phase-1 arrival exactly at "
            "t_offset, so first_batch_arrival_s == last_interactive_arrival_s "
            "and the gap is 0.0 by construction. The invariant is >= 0: the "
            "BATCH stream may never begin before the INTERACTIVE stream's "
            "scheduled window has ended. This is the open-loop splice -- it is "
            "NOT the historical closed-loop dump, which fired on engine "
            "progress rather than on the schedule."),
    }


def assert_phase_splice(specs: list, arrivals: dict, n: int) -> dict:
    """Raise unless the two phases are sequential and correctly indexed.

    Three ways this could silently go wrong, each of which would invalidate the
    phase-specific lambda: the BATCH stream starting before the INTERACTIVE
    window closes; a BATCH request among the first n/2; an INTERACTIVE request
    after n/2. All three are checked, and a violation raises rather than being
    recorded -- a run whose phases overlap cannot be reported at either rate.
    """
    rep = phase_boundary_report(specs, arrivals, n)
    problems = []
    if not rep["phases_do_not_overlap"]:
        problems.append(
            "the BATCH stream begins at {} but the INTERACTIVE stream runs to "
            "{} (gap {}): the phases overlap, so neither ran at its stated "
            "lambda".format(rep["first_batch_arrival_s"],
                            rep["last_interactive_arrival_s"],
                            rep["phase_boundary_gap_s"]))
    if not rep["head_is_all_interactive"]:
        problems.append(
            "{} BATCH request(s) appear among the first {} requests".format(
                rep["batch_requests_in_head"], rep["split_index"]))
    if not rep["tail_is_all_batch"]:
        problems.append(
            "{} INTERACTIVE request(s) appear after request {}".format(
                rep["interactive_requests_in_tail"], rep["split_index"]))
    if problems:
        raise PhaseSpliceError(
            "phase splice is not sequential: " + "; ".join(problems))
    return rep


def arrival_report(offsets: list, phase: str, plan: dict) -> dict:
    """Everything needed to prove the intended process was the one generated.

    Includes this phase's EMPIRICAL OFFERED LOAD, which is the only quantity in
    this experiment entitled to be called rho:

        empirical_rho = empirical_lambda_rps / mu_req_per_s

    It is computed from the SCHEDULED arrival offsets, so it is a property of
    the seeded draw and is known before any GPU time is spent. Poisson sampling
    means it will scatter around the 0.40 target; that scatter IS the process,
    not an error in it. No seed is rejected, re-drawn or re-run on the strength
    of it -- see `empirical_rho_policy`.
    """
    p = plan["phases"][phase]
    lam = p["lambda_rps"]
    mu = p["mu_max_req_per_s"]
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    duration = (offsets[-1] - offsets[0]) if len(offsets) > 1 else 0.0
    mean_gap = statistics.fmean(gaps) if gaps else None
    # The empirical rate over the phase's own arrival span: intervals divided by
    # the span they cover, which is exactly the reciprocal of the interarrival
    # mean. (n / span would be a different, slightly larger estimator and would
    # bias rho upward.)
    emp_lambda = (len(gaps) / duration) if duration > 0 else None
    emp_rho = (emp_lambda / mu) if (emp_lambda is not None and mu > 0) else None
    # Normal-approximation interval on the mean of n-1 exponential gaps; the
    # standard deviation of Exp(lambda) is 1/lambda, so the interval is a
    # function of lambda and the count alone and can be written down BEFORE the
    # run. DESCRIPTIVE since the 2026-10-22 amendment: H5 itself is decided by
    # the exact Gamma family bounds (h5_reference / h5_family_report).
    expected = 1.0 / lam
    halfwidth = (1.96 * expected / math.sqrt(len(gaps))) if gaps else None
    # An interarrival mean cannot be negative, so the lower bound is clamped at
    # 0 rather than reported as the raw normal approximation. At small gap
    # counts the interval is wide enough to accept almost anything, which is why
    # it carries a flag saying so.
    ci_lo = (max(0.0, expected - halfwidth) if halfwidth is not None else None)
    return {
        "phase": phase,
        "process": ARRIVAL_POISSON,
        # --- the operating point, requested and realised -------------------
        "target_rho": p["target_rho"],
        "mu_req_per_s": mu,
        "mu_source_row": p["saturation_row"],
        "requested_lambda_rps": lam,
        "empirical_lambda_rps": emp_lambda,
        "empirical_rho": emp_rho,
        "empirical_rho_minus_target": (
            (emp_rho - p["target_rho"]) if emp_rho is not None else None),
        "empirical_rho_definition": "empirical_lambda_rps / mu_req_per_s",
        "empirical_rho_policy": (
            "Poisson sampling makes the realised rate scatter around the "
            "target; the first seeded draw IS the evidence. No seed is "
            "rejected, re-drawn or re-run because its empirical_rho came out "
            "above or below 0.40."),
        # --- the generated process ----------------------------------------
        "n_requests": len(offsets),
        "n_interarrival_gaps": len(gaps),
        "expected_mean_interarrival_s": expected,
        "empirical_mean_interarrival_s": mean_gap,
        "empirical_median_interarrival_s": (statistics.median(gaps)
                                            if gaps else None),
        "empirical_stdev_interarrival_s": (statistics.stdev(gaps)
                                           if len(gaps) > 1 else None),
        "scheduled_arrival_window_s": duration,
        "first_arrival_offset_s": offsets[0] if offsets else None,
        "last_arrival_offset_s": offsets[-1] if offsets else None,
        "expected_mean_ci95": ([ci_lo, expected + halfwidth]
                               if halfwidth is not None else None),
        "expected_mean_ci95_method": (
            "normal approximation on the mean of n exponential gaps; the "
            "standard deviation of Exp(lambda) is 1/lambda, so the half-width "
            "is 1.96/(lambda*sqrt(n)) and is a function of lambda and n alone "
            "-- writable before the run. Lower bound clamped at 0."),
        "expected_mean_ci95_is_informative": len(gaps) >= 30,
        "mean_within_ci95": (
            bool(halfwidth is not None and mean_gap is not None
                 and abs(mean_gap - expected) <= halfwidth)),
        "rate_estimator_note": (
            "empirical_lambda_rps counts INTERVALS over the span between first "
            "and last arrival (n-1 gaps / span), which is the reciprocal of "
            "the interarrival mean; dividing n by the span would be a "
            "different, slightly larger estimator and would bias rho upward"),
    }


def offered_load_report(arrival_validation: dict, plan: dict) -> dict:
    """Per-phase EMPIRICAL OFFERED LOAD, gathered in one place.

    This is the quantity entitled to be called rho. It is a property of the
    seeded arrival draw -- rate divided by the measured service rate -- and is
    deliberately kept apart from `drain_overrun`, which is a wall-clock
    observation about the engine and is NOT a load.
    """
    per_phase = {}
    for phase, r in arrival_validation.items():
        per_phase[phase] = {
            "target_rho": r["target_rho"],
            "mu_req_per_s": r["mu_req_per_s"],
            "mu_source_row": r["mu_source_row"],
            "requested_lambda_rps": r["requested_lambda_rps"],
            "empirical_lambda_rps": r["empirical_lambda_rps"],
            "empirical_rho": r["empirical_rho"],
            "empirical_rho_minus_target": r["empirical_rho_minus_target"],
            "empirical_mean_interarrival_s": r["empirical_mean_interarrival_s"],
            "empirical_median_interarrival_s": r["empirical_median_interarrival_s"],
            "scheduled_arrival_window_s": r["scheduled_arrival_window_s"],
        }
    return {
        "definition": "empirical_rho = empirical_lambda_rps / mu_req_per_s",
        "target_rho": plan["target_rho"],
        "per_phase": per_phase,
        "seed_selection_policy": (
            "NO SEED IS EVER REJECTED OR RE-RUN on the basis of its realised "
            "Poisson rate. Every seed's first draw is the evidence. Discarding "
            "draws that missed the target would turn an arrival process into a "
            "hand-picked schedule and would make the stated rho a selection "
            "effect rather than a condition."),
        "not_to_be_confused_with": (
            "`drain_overrun`, which compares wall clock against the scheduled "
            "arrival window. That is an engine observation, not an offered "
            "load, and is not rho."),
    }


def drain_overrun(out: dict, arrivals: dict, plan: dict) -> dict:
    """How far past its own arrival schedule did the run keep working?

    THIS IS NOT RHO, AND NOT A LOAD OR UTILISATION. It is a wall-clock
    observation: `arrival_window_overrun_ratio = wall_s / scheduled_arrival_
    window_s`. Offered load is `offered_load_report`, and the two are kept
    under separate names on purpose -- the earlier draft called this quantity
    "realised load", which it is not.

    What it is good for: under a genuinely sub-critical open-loop stream the
    engine keeps up, so the run ends shortly after its last scheduled arrival
    and the ratio sits near 1.0. Materially above 1.0 says the engine, not the
    arrival process, set the pace. REPORTED, never used to re-tune lambda.
    """
    last = max((max(v) if v else 0.0) for v in arrivals.values())
    wall = out.get("wall_s") or 0.0
    records = out.get("requests") or []
    n_total = sum(len(v) for v in arrivals.values())

    # Per-phase ACTUAL timing, read off the per-request stage timestamps, so the
    # scheduled window and the observed drain are reported side by side rather
    # than collapsed into one run-level number.
    by_phase = {"interactive": "INTERACTIVE", "batch": "BATCH"}
    per_phase = {}
    lags = []
    for phase, regime in by_phase.items():
        sched = arrivals.get(phase) or []
        finishes, submits = [], []
        for r in records:
            if r.get("regime") != regime:
                continue
            ts = r.get("stage_timestamps") or {}
            if ts.get("finish_s") is not None:
                finishes.append(ts["finish_s"])
            if ts.get("submit_s") is not None and ts.get("arrival_target_s") is not None:
                lag = ts["submit_s"] - ts["arrival_target_s"]
                submits.append(lag)
                lags.append(lag)
        sched_last = max(sched) if sched else None
        last_finish = max(finishes) if finishes else None
        per_phase[phase] = {
            "scheduled_first_arrival_s": min(sched) if sched else None,
            "scheduled_last_arrival_s": sched_last,
            "scheduled_arrival_window_s": (
                (max(sched) - min(sched)) if len(sched) > 1 else 0.0),
            "requests_scheduled": len(sched),
            "requests_completed": len(finishes),
            "actual_first_finish_s": min(finishes) if finishes else None,
            "actual_last_finish_s": last_finish,
            # How long after its LAST scheduled arrival the phase was still
            # working. Small means the engine kept up with the stream.
            "drain_overrun_s": (
                (last_finish - sched_last)
                if (last_finish is not None and sched_last is not None) else None),
            "max_submission_lag_s": max(submits) if submits else None,
        }

    return {
        "metric": "arrival_window_overrun_ratio",
        "definition": "wall_s / scheduled_arrival_window_s",
        "is_not_rho": (
            "this is a wall-clock ratio, not an offered load. The offered load "
            "is empirical_lambda_rps / mu_req_per_s and lives in "
            "offered_load_report."),
        "wall_s": wall,
        "scheduled_arrival_window_s": last,
        "arrival_window_overrun_ratio": (wall / last) if last > 0 else None,
        "requests_scheduled": n_total,
        "requests_completed": len(records),
        "completion_rate_req_per_s": (len(records) / wall) if wall > 0 else None,
        "max_submission_lag_s": max(lags) if lags else None,
        "submission_lag_note": (
            "submit time minus the generator's intended arrival time. A large "
            "lag means the HARNESS, not the engine, failed to honour the "
            "schedule, and the arrival process would then be mis-stated."),
        "per_phase": per_phase,
        "interpretation": (
            "near 1.0 corroborates the derived sub-critical target: the engine "
            "drained as fast as the stream arrived. Materially above 1.0 means "
            "the engine set the pace, which bounds what the run can be used to "
            "claim -- but it is still not a value of rho."),
    }


# ===========================================================================
# 2b. Deterministic checkpoint resume. AN EXECUTION FEATURE ONLY.
#
# Nothing here may change a scientific quantity. It decides only whether a unit
# of work is RE-EXECUTED or READ BACK, and it refuses to do either unless the
# state it is reading belongs to this exact experiment. The frozen scales and
# both static baselines are recomputed from the full grid on every invocation
# and must reproduce the calibration digest every evaluation unit is bound to;
# a scale or a winner is never taken on trust from the artifact.
# ===========================================================================


def sweep_unit_id(mb: int, cs: int) -> str:
    return f"sweep:mb{int(mb)}_cs{int(cs)}"


def eval_unit_id(name: str, seed: int) -> str:
    return f"eval:{name}:seed{int(seed):03d}"


def unit_plan(batches: list, chunks: list, seeds: list) -> list:
    """Every unit of work, in the ONE deterministic order they are executed in.

    The held-out sweep first, in grid order, then every treatment in `CONFIGS`
    order crossed with `seeds` in the order given. A resume walks this same list
    and skips; it never reorders, so "resumed" and "uninterrupted" differ only
    in which units were executed in which session.
    """
    units = [sweep_unit_id(mb, cs) for mb in batches for cs in chunks]
    units += [eval_unit_id(name, s) for name in CONFIGS for s in seeds]
    return units


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_of_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def check_source(sidecar: dict | None, archive_path: str | None = None) -> dict:
    """Verify the EXECUTING tree against the sidecar's archive manifest.

    `executing_manifest_sha256` is recomputed from the files on disk, so it is
    what goes into the fingerprint: an unchanged sidecar cannot vouch for edited
    source, because the edit moves this digest and the resume then refuses.
    With `archive_path` (the uploaded ZIP) the ZIP is hashed too, and the
    report says whether its digest was VERIFIED or merely asserted.
    """
    rep = srcprov.verify_tree(sidecar, _REPO_ROOT, archive_path=archive_path)
    rep["working_tree_diff_sha256"] = (
        sidecar.get("working_tree_diff_sha256") if isinstance(sidecar, dict)
        else None)
    return rep


# --- runtime and model identity -------------------------------------------------

# Files whose bytes define the model and tokenizer this run executes. config,
# weights and tokenizer config are required; at least one tokenizer vocabulary
# file must also be present.
MODEL_IDENTITY_REQUIRED = ("config.json", "model.safetensors",
                           "tokenizer_config.json")
MODEL_IDENTITY_OPTIONAL = ("tokenizer.json", "tokenizer.model",
                           "special_tokens_map.json", "generation_config.json")


def _device_index(device) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    idx = getattr(device, "index", None)
    return int(idx) if idx is not None else int(torch.cuda.current_device())


def _cuda_device_props(index: int) -> dict:
    p = torch.cuda.get_device_properties(index)
    return {"name": p.name, "capability": f"{p.major}.{p.minor}",
            "total_memory_bytes": int(p.total_memory)}


def _library_versions() -> dict:
    import platform

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception as exc:          # recorded, never silently null
            return f"<unavailable: {type(exc).__name__}>"

    try:
        cudnn = (torch.backends.cudnn.version()
                 if torch.backends.cudnn.is_available() else None)
    except Exception:
        cudnn = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": cudnn,
            "transformers": _ver("transformers"),
            "huggingface_hub": _ver("huggingface_hub"),
            "safetensors": _ver("safetensors")}


def runtime_identity(device) -> dict:
    """The device that executes and the library stack around it."""
    idx = _device_index(device)
    return {"device_type": getattr(device, "type", str(device)),
            "device_index": idx,
            "gpu": _cuda_device_props(idx) if idx is not None else None,
            **_library_versions()}


def _model_device(model):
    """The device the model's parameters actually live on, or None."""
    try:
        return next(iter(model.parameters())).device
    except (AttributeError, StopIteration, TypeError):
        return None


def model_identity(model_name: str = MODEL_NAME) -> dict:
    """The immutable revision and file digests of the model this run loaded.

    Read OFFLINE from the local Hugging Face cache that `hf_hub_download` /
    `AutoTokenizer.from_pretrained` just used: the snapshot directory name is
    the commit the hub resolved at download time. Nothing is resolved over the
    network and no revision is assumed. Every listed file is hashed, so the
    identity is the bytes, not only the name. Unresolvable -> an explicit
    record saying why, which a published run refuses.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as exc:
        return {"resolved": False, "repo_id": model_name,
                "why": f"huggingface_hub unavailable ({type(exc).__name__})"}
    files, revisions, missing = {}, set(), []
    for name in MODEL_IDENTITY_REQUIRED + MODEL_IDENTITY_OPTIONAL:
        try:
            path = try_to_load_from_cache(model_name, name)
        except Exception:
            path = None
        if not isinstance(path, str) or not os.path.isfile(path):
            if name in MODEL_IDENTITY_REQUIRED:
                missing.append(name)
            continue
        parts = os.path.normpath(path).split(os.sep)
        revisions.add(parts[parts.index("snapshots") + 1]
                      if "snapshots" in parts else None)
        files[name] = {"sha256": _sha256_of_file(path),
                       "bytes": os.path.getsize(path)}
    if not ({"tokenizer.json", "tokenizer.model"} & set(files)):
        missing.append("tokenizer.json or tokenizer.model")
    rev = next(iter(revisions)) if len(revisions) == 1 else None
    problems = []
    if missing:
        problems.append(f"not in the local cache: {missing}")
    if len(revisions) > 1:
        problems.append(f"files come from different snapshots {sorted(map(str, revisions))}")
    if rev is not None and not srcprov.is_hex_digest(rev, 40):
        problems.append(f"snapshot directory {rev!r} is not a 40-hex commit")
    if problems or rev is None:
        return {"resolved": False, "repo_id": model_name, "files": files,
                "why": "; ".join(problems) or "no snapshot revision found"}
    body = {"repo_id": model_name, "revision": rev, "files": files}
    return {"resolved": True, **body,
            "identity_sha256": _sha256_text(_canonical(body)),
            "revision_source": ("local HF cache snapshot directory (the commit "
                                "resolved at download time); not re-resolved "
                                "over the network"),
            "registered_reference_revision": None,
            "registered_reference_note": (
                "no model revision was recorded for the saturated run, so this "
                "binds the run to ITSELF across sessions; it cannot show the "
                "weights equal the saturated run's")}


def experiment_fingerprint(plan: dict, seeds: list, n: int, batches: list,
                           chunks: list, *, model: str, dtype: str,
                           source_check: dict | None = None,
                           run_mode: str = RUN_MODE_PUBLISHED,
                           runtime: dict | None = None,
                           model_id: dict | None = None,
                           saturated_ref: dict | None = None) -> dict:
    """Everything a resume must agree on before it may skip a single unit.

    Deliberately wide. A narrow fingerprint is worse than none: it licenses a
    resume across exactly the changes that would corrupt the artifact. Anything
    here that differs aborts the resume and names itself.
    """
    sc = source_check or {}
    pi = plan["phases"]["interactive"]
    pb = plan["phases"]["batch"]
    return {
        "experiment": EXPERIMENT,
        "experiment_version": EXPERIMENT_VERSION,
        "run_mode": run_mode,
        "target_rho": plan["target_rho"],
        "execution_arm": plan["execution_arm"],
        "mu_interactive_req_per_s": pi["mu_max_req_per_s"],
        "mu_batch_req_per_s": pb["mu_max_req_per_s"],
        "lambda_interactive_rps": pi["lambda_rps"],
        "lambda_batch_rps": pb["lambda_rps"],
        "service_rate_source_artifact": plan["source_artifact"],
        "seeds": list(seeds),
        "held_out_seed": VALIDATION_SEED,
        "n_requests": n,
        "batch_grid": list(batches),
        "chunk_grid": list(chunks),
        "model": model,
        "dtype": dtype,
        # The hardware and library stack, and the model's immutable revision
        # and file digests: a resume on another GPU, another torch /
        # transformers / CUDA, or other weights is a different experiment.
        "runtime_identity": runtime,
        "model_identity": model_id,
        # The DECLARED engine warm-up (version, inputs, shapes, bounds, sync
        # policy). Never its measured duration: that is per-session evidence.
        "engine_warmup_protocol": engine_warmup_spec(batches, chunks),
        "arrival_mode": ARRIVAL_POISSON,
        "arrival_rng_namespace": ARRIVAL_RNG_NAMESPACE,
        "workload_version": provenance.WORKLOAD_VERSION,
        "phase1_open_loop": True,
        "reward_version": provenance.REWARD_VERSION,
        "reward_thresholds": [MIN_ARM_REWARD_SPREAD, MIN_LIVE_REWARD_SPREAD,
                              MIN_LIVE_DISTINCT],
        "observe_interval": OBSERVE_INTERVAL,
        "alpha": ALPHA,
        "feature_dim": FEATURE_DIM,
        "convergence_rule_sha256": _sha256_text(_canonical(CONVERGENCE_RULE)),
        "treatments": list(CONFIGS),
        "arm_space_sha256": _sha256_text(_canonical({
            "restricted": effective_arm_report(abl.all_arm_sets()),
            "expanded": effective_arm_report(abl.expanded_arm_sets()),
        })),
        # LOCAL provenance only. The archive digest is read from where the
        # producer writes it (`archive.sha256`); the source digest is the one
        # RECOMPUTED over the executing tree, not the sidecar's claim.
        "source_archive_sha256": sc.get("archive_sha256"),
        "archive_manifest_sha256": sc.get("archive_manifest_sha256"),
        "executing_source_manifest_sha256": sc.get("executing_manifest_sha256"),
        "working_tree_diff_sha256": sc.get("working_tree_diff_sha256"),
        "preregistration_sha256": _sha256_of_file(PREREG_PATH),
        # WHICH frozen saturated evidence H1/H2 are judged against, and whether
        # it verified: changed or re-verified evidence never resumes silently.
        "saturated_reference_identity": sref.fingerprint_identity(
            saturated_ref or {}),
    }


def load_saturated_reference() -> dict:
    """The verified saturated reference record, or an explicit unverified one."""
    return sref.try_load(SATURATED_REFERENCE_ROOT)


def compare_fingerprints(stored: dict, current: dict, ignore=()) -> list:
    """Keys on which a stored fingerprint disagrees with the current one.

    `ignore` names keys not yet knowable (the model identity before the model
    is loaded); the full comparison always runs again once they are."""
    if not isinstance(stored, dict):
        return ["<no fingerprint recorded in the artifact>"]
    out = []
    for key in sorted((set(stored) | set(current)) - set(ignore)):
        a, b = stored.get(key, "<absent>"), current.get(key, "<absent>")
        if _canonical(a) != _canonical(b):
            out.append(f"{key}: checkpoint={a!r} now={b!r}")
    return out


class TerminalGateFailureError(ResumeMismatchError):
    """--resume asked to continue past a reward gate that already fired.

    A gate failure is a RESULT about the operating point. Re-measuring the unit
    until timing noise lets it pass would turn the gate into a retry loop, so
    an ordinary resume refuses and the failure stays on the record.
    """


class ResumeLedger:
    """Which units are already done -- per the checkpoint AND their evidence.

    A checkpoint entry alone is not enough, and neither is a file that parses.
    The caller validates each claimed unit's durable evidence (fingerprint,
    calibration binding, identity, finiteness, nested structure, internal
    consistency) and either `accept`s it or `reject`s it, and every rejection,
    recovery or superseded file is kept in `audit` rather than swallowed.
    """

    def __init__(self, fingerprint: dict, completed=()):
        self.fingerprint = fingerprint
        self._order = list(dict.fromkeys(completed))
        self._done = set(self._order)
        self.resumed = False
        self.skipped: list = []
        self.reran: list = []
        self.audit: list = []

    def claimed(self, unit_id: str) -> bool:
        return unit_id in self._done

    def accept(self, unit_id: str) -> None:
        self.skipped.append(unit_id)

    def reject(self, unit_id: str, why: str) -> None:
        self._done.discard(unit_id)
        self._order = [u for u in self._order if u != unit_id]
        self.reran.append({"unit": unit_id, "why": why})
        self.note(unit_id, "rerun", why)
        print(f"[resume] {unit_id}: checkpoint says done but {why} -- "
              f"RE-RUNNING", flush=True)

    def note(self, unit_id: str, action: str, why: str) -> None:
        self.audit.append({"unit": unit_id, "action": action, "why": why,
                           "at_utc": _now()})

    def mark(self, unit_id: str) -> None:
        if unit_id not in self._done:
            self._done.add(unit_id)
            self._order.append(unit_id)

    def as_dict(self) -> dict:
        return {
            "units_completed": list(self._order),
            "n_completed": len(self._order),
            "resumed": self.resumed,
            "units_skipped_on_resume": list(self.skipped),
            "units_rerun_on_resume": list(self.reran),
            "recovery_audit": list(self.audit),
            "policy": (
                "a unit counts as complete only if BOTH its checkpoint entry "
                "and its durable evidence exist and validate (fingerprint, "
                "calibration digest, identity, seed, n, finite metrics, nested "
                "evidence, internal consistency); otherwise it is re-run and "
                "the reason is recorded here. Resume decides execution only -- "
                "the frozen scales and both static baselines are recomputed "
                "from the full grid and must reproduce the calibration digest."),
        }


def terminal_failure_reasons(prior: dict | None, paths: "RunPaths") -> list:
    """Every sign that a scientific gate already fired for this run."""
    reasons = []
    if isinstance(prior, dict):
        if prior.get("terminal_failure"):
            reasons.append("the checkpoint records a terminal_failure")
        ab = prior.get("aborted") or {}
        if (ab.get("status") == STATUS_TERMINAL_GATE
                or ab.get("reason") == "DegenerateRewardError"):
            reasons.append("the checkpoint's abort record is a reward-gate "
                           "failure")
        for h in prior.get("run_history") or []:
            if isinstance(h, dict) and h.get("status") == STATUS_TERMINAL_GATE:
                reasons.append(f"session {h.get('session_index')} ended in a "
                               "reward-gate failure")
    if os.path.isdir(paths.failed_dir):
        # Terminal MARKERS live here too and are written before the larger
        # evidence file, so a gate failure is recognised even when the
        # checkpoint never recorded it and the evidence write itself failed.
        failed = sorted(f for f in os.listdir(paths.failed_dir)
                        if f.endswith(".json"))
        if failed:
            reasons.append(f"terminal marker / failed-unit evidence exists: "
                           f"{failed}")
    return reasons


class InvalidFreshEvidenceError(RuntimeError):
    """Freshly measured evidence failed the validation a resume would apply."""


class InvalidEvidenceResumeError(ResumeMismatchError):
    """--resume asked to continue past invalid fresh evidence."""


def invalid_evidence_reasons(prior: dict | None, paths: "RunPaths") -> list:
    reasons = []
    if isinstance(prior, dict):
        if prior.get("invalid_fresh_evidence"):
            reasons.append("the checkpoint records invalid fresh evidence")
        if (prior.get("aborted") or {}).get("status") == STATUS_INVALID_EVIDENCE:
            reasons.append("the checkpoint's abort record is invalid fresh "
                           "evidence")
    if os.path.isdir(paths.invalid_dir):
        bad = sorted(f for f in os.listdir(paths.invalid_dir) if f.endswith(".json"))
        if bad:
            reasons.append(f"invalid fresh evidence exists: {bad}")
    return reasons


def read_resume_checkpoint(paths: "RunPaths") -> dict:
    """Read a checkpoint for `--resume`, or refuse to. Reads; never writes.

    Runs BEFORE the model is loaded, so the cheap refusals (nothing to resume,
    unreadable, unfingerprinted, a gate already fired) cost no GPU time and
    cannot be followed by any write to the checkpoint.
    """
    if not os.path.isfile(paths.results):
        raise ResumeMismatchError(
            f"--resume was given but there is no artifact at {paths.results} to "
            "resume from. Run without --resume to start the experiment.")
    try:
        with open(paths.results, encoding="utf-8") as f:
            prior = json.load(f)
    except Exception as exc:
        raise ResumeMismatchError(
            f"--resume could not parse {paths.results}: {exc}. Refusing to guess "
            "which units completed; nothing was modified.") from exc
    if not isinstance(prior, dict) or not isinstance(prior.get("fingerprint"), dict):
        raise ResumeMismatchError(
            f"{paths.results} records no experiment fingerprint, so there is no "
            "way to show it belongs to this run. Nothing was modified.")
    reasons = terminal_failure_reasons(prior, paths)
    if reasons:
        raise TerminalGateFailureError(
            "--resume REFUSED: a reward gate already fired for this run ("
            + "; ".join(reasons) + "). That is a scientific result, not an "
            "interruption: re-measuring until timing noise lets the gate pass "
            "would make it a retry loop. The evidence is left exactly as it is.")
    bad = invalid_evidence_reasons(prior, paths)
    if bad:
        raise InvalidEvidenceResumeError(
            "--resume REFUSED: freshly measured evidence failed validation ("
            + "; ".join(bad) + "). That is a validity defect to investigate, "
            "not an interruption: resuming would silently regenerate the unit. "
            "The diagnostic evidence is left exactly as it is.")
    return prior


def load_resume_state(paths: "RunPaths", fingerprint: dict) -> dict:
    """`read_resume_checkpoint` plus the full fingerprint comparison."""
    prior = read_resume_checkpoint(paths)
    diffs = compare_fingerprints(prior.get("fingerprint"), fingerprint)
    if diffs:
        raise ResumeMismatchError(
            "--resume REFUSED: the checkpoint belongs to a different "
            "experiment. Differing keys:\n  - " + "\n  - ".join(diffs) +
            "\nResuming across a changed operating point, seed list, grid, "
            "model, reward, convergence rule, run mode or executing source "
            "would splice two experiments into one artifact. Nothing was "
            "modified; start a fresh run instead.")
    return prior

# ===========================================================================
# 3. Action space: nominal arms vs unique LIVE configurations.
# ===========================================================================


def effective_arm_report(arms_by_regime: dict) -> dict:
    """Nominal arm count AND unique reachable (max_batch_size, chunk_size).

    Two nominal arms that differ only in `spec_k` or `eviction_threshold` are
    the SAME configuration in this harness, because those fields cannot change
    execution (see INERT_DIMENSIONS). Reporting only the nominal count would
    describe an action space wider than the one that exists, so "10 arms vs 6
    arms" is replaced by expanded LIVE CONFIGURATION COVERAGE.
    """
    out = {}
    for regime, arms in arms_by_regime.items():
        tuples = [(int(a.max_batch_size), int(a.chunk_size)) for a in arms]
        unique = sorted(set(tuples))
        out[str(getattr(regime, "value", regime))] = {
            "nominal_arms": len(arms),
            "unique_live_configurations": len(unique),
            "live_configuration_tuples": [list(t) for t in unique],
            "reachable_max_batch_size": sorted({t[0] for t in unique}),
            "reachable_chunk_size": sorted({t[1] for t in unique}),
            "nominal_arms_collapse": len(arms) != len(unique),
        }
    return {
        "live_dimensions": list(LIVE_DIMENSIONS),
        "per_regime": out,
        "why_both_counts": (
            "some nominal arms differ only in fields that cannot change "
            "execution in this harness; the count that describes the action "
            "space an agent can actually express is the unique "
            "(max_batch_size, chunk_size) count"),
        "preferred_wording": "expanded live configuration coverage",
    }


# ===========================================================================
# 4. One run.
# ===========================================================================


def run_static(cfg: CARLConfig, model, tokenizer, n: int, seed: int,
               plan: dict) -> dict:
    """Serve one fixed configuration once under the Poisson stream."""
    specs, arrivals = build_poisson_workload(tokenizer, n, seed, plan)
    sched = abl._new_scheduler(model)
    abl._apply_sched(sched, cfg)
    out = abl._serve(sched, specs, phase1_open_loop=True)
    out["arrivals"] = arrivals
    # Recorded here because this is where the specs live. The invariant itself
    # was already asserted inside build_poisson_workload; this is the evidence.
    out["phase_boundary"] = phase_boundary_report(specs, arrivals, n)
    return out


def sweep_live_plane(model, tokenizer, n: int, seed: int, plan: dict,
                     batches: list, chunks: list, *,
                     ledger=None, rows_by_key=None, on_row=None,
                     save_unit=None, before_measure=None) -> list:
    """FRESH exhaustive max_batch_size x chunk_size on the held-out seed.

    Not reused from the saturated run: the saturated scales and the saturated
    Static-Tput / Static-Utility are properties of a saturated queue. Carrying
    them into a sub-critical stream would give this experiment a baseline tuned
    for a condition it is not running, which is the objective-mismatch defect in
    a new costume.

    Grid points already in `rows_by_key` (verified evidence restored by a
    resume) are not re-measured and not re-announced. Each newly measured point
    is validated and written to its OWN file by `save_unit` first (which
    returns the trusted row), then marked in the ledger,
    then `on_row` checkpoints -- so a disconnect loses at most the config in
    flight, and a checkpoint can never hold fewer rows than were verified.
    `before_measure` runs before every point that is actually measured (the
    driver uses it to refuse measurement without the engine warm-up). The
    returned list is always in GRID ORDER regardless of what order the work
    actually happened in, so the frozen scales and the static selection cannot
    depend on the resume path.
    """
    total = len(batches) * len(chunks)
    have = rows_by_key if rows_by_key is not None else {}
    restored = sum(1 for mb in batches for cs in chunks
                   if f"mb{mb}_cs{cs}" in have)
    print(f"\n[held-out sweep] FRESH EXHAUSTIVE {len(batches)}x{len(chunks)} = "
          f"{total} configs x {n} requests on seed {seed} (never an evaluation "
          f"seed), Poisson arrivals at rho={plan['target_rho']}; {restored} "
          f"restored from verified evidence", flush=True)
    for mb in batches:
        for cs in chunks:
            unit = sweep_unit_id(mb, cs)
            key = unit.split(":", 1)[1]
            if key in have:
                if ledger is not None:
                    ledger.accept(unit)
                continue
            cfg = CARLConfig(max_batch_size=mb, chunk_size=cs).clamp()
            if before_measure is not None:
                before_measure()
            m = run_static(cfg, model, tokenizer, n, seed, plan)
            row = {
                "max_batch_size": mb,
                "chunk_size": cs,
                "config": cfg.as_dict(),
                "throughput_tps": m["throughput_tps"],
                "ttft_p50_ms": m["ttft_p50"], "ttft_p99_ms": m["ttft_p99"],
                "tpot_p50_ms": m["tpot_p50"], "tpot_p99_ms": m["tpot_p99"],
                "slo_rate": m["slo_rate"], "wall_s": m["wall_s"],
                "arrival_validation": {
                    ph: arrival_report(offs, ph, plan)
                    for ph, offs in m["arrivals"].items()},
                "drain_overrun": drain_overrun(m, m["arrivals"], plan),
                "phase_boundary": m["phase_boundary"],
            }
            row["offered_load"] = offered_load_report(
                row["arrival_validation"], plan)
            if save_unit is not None:
                # Validated exactly as a resume would validate it, written, and
                # handed back as the round-tripped row -- so a fresh row and a
                # resumed row are the same object.
                row = save_unit(mb, cs, row, m)
            have[key] = row
            print(f"  mb={mb:2d} cs={cs:3d} -> {m['throughput_tps']:7.2f} tok/s  "
                  f"ttftP99={m['ttft_p99']:9.1f}ms  tpotP99={m['tpot_p99']:7.1f}ms  "
                  f"({m['wall_s']:.1f}s)", flush=True)
            if ledger is not None:
                ledger.mark(unit)
            if on_row is not None:
                on_row()
    return [have[f"mb{mb}_cs{cs}"] for mb in batches for cs in chunks
            if f"mb{mb}_cs{cs}" in have]


def _rho_of(run: dict, phase: str) -> float:
    """A run's realised offered load for one phase, for the progress line."""
    return (run["offered_load"]["per_phase"][phase]["empirical_rho"]) or 0.0


def assert_sweep_complete(rows: list, batches: list, chunks: list) -> None:
    """The static selection may only run once EVERY grid point exists.

    A selection made from a partial grid is a selection from a different search
    space, and on a resumed run that would silently depend on where the previous
    session happened to stop.
    """
    have = {(r["max_batch_size"], r["chunk_size"]) for r in rows}
    want = {(mb, cs) for mb in batches for cs in chunks}
    missing = sorted(want - have)
    if missing:
        raise ResumeMismatchError(
            "the held-out sweep is incomplete: {} of {} grid points missing "
            "{}. Static-Tput and Static-Utility are argmaxes over the FULL "
            "plane and must not be computed from a partial one.".format(
                len(missing), len(want), missing))


def gate_sweep(rows: list, scales) -> dict:
    """PRE-RUN GATE: the frozen scales must separate the candidate plane.

    Same function and the SAME frozen threshold the saturated run used. If the
    sub-critical plane is so flat that the reward cannot tell 28 configurations
    apart, that is a RESULT about this operating point -- it is reported, and the
    threshold is not lowered to get past it.
    """
    arm_rewards = {
        f"mb{r['max_batch_size']}_cs{r['chunk_size']}": fh.run_utility(
            {"throughput_tps": r["throughput_tps"], "ttft_p99": r["ttft_p99_ms"],
             "tpot_p99": r["tpot_p99_ms"]}, scales)
        for r in rows
    }
    report = gate_candidate_arm_spread(
        arm_rewards, context="subcritical_poisson held-out sweep",
        threshold=MIN_ARM_REWARD_SPREAD)
    print(f"[gate:pre-run] candidate reward spread "
          f"{report['reward_spread']:.6f} >= {MIN_ARM_REWARD_SPREAD} -- PASS",
          flush=True)
    return report


def flatness_diagnostics(values: list, identities: list) -> dict:
    """Near-tie structure of one objective over the held-out grid.

    PURELY DESCRIPTIVE, AND SELECTION-FREE. This function is called AFTER
    `fh.select_static_baselines` has already chosen the winners, it returns no
    configuration for anyone to act on, and nothing downstream consumes it. It
    exists because at sub-critical offered load throughput is partly
    arrival-limited, so the gap between the best and second-best configuration
    can be far smaller than the measurement's own noise -- and an argmax over a
    near-flat surface should not be read as a discovered optimum.

    It does NOT introduce a tie-break. The pre-registered winner stands.
    """
    if not values:
        return {"n": 0}
    order = sorted(range(len(values)), key=lambda i: -values[i])
    best_i = order[0]
    best = values[best_i]
    second = values[order[1]] if len(order) > 1 else None
    lo, hi = min(values), max(values)

    def _within(pct):
        # "Within p% of best" on a positive scale. Both objectives here are
        # strictly positive; guarded anyway so a degenerate grid reports
        # nothing rather than dividing by zero.
        if best <= 0:
            return None
        thresh = best * (1.0 - pct / 100.0)
        idx = [i for i, v in enumerate(values) if v >= thresh]
        return {
            "count": len(idx),
            "configurations": [identities[i] for i in idx],
        }

    return {
        "n": len(values),
        "best": best,
        "best_configuration": identities[best_i],
        "second_best": second,
        "second_best_configuration": (identities[order[1]]
                                      if len(order) > 1 else None),
        "best_minus_second_best": (best - second) if second is not None else None,
        "best_minus_second_best_relative_pct": (
            (100.0 * (best - second) / best)
            if (second is not None and best > 0) else None),
        "min": lo, "max": hi,
        "range_absolute": hi - lo,
        "range_relative_pct": (100.0 * (hi - lo) / hi) if hi > 0 else None,
        "median": statistics.median(values),
        "within_0p5pct_of_best": _within(0.5),
        "within_1p0pct_of_best": _within(1.0),
        "selection_note": (
            "descriptive only: the pre-registered argmax is unchanged by "
            "anything in this block, and no post-hoc tie-break is applied"),
    }


def landscape_report(rows: list, scales, selection: dict) -> dict:
    """Q1: how flat is the live configuration landscape here?

    Descriptive only. These are 28 deterministic search points on one held-out
    seed; a significance test over them would be theatre, so none is computed.
    The near-tie diagnostics are computed from `selection`'s own inputs AFTER
    the winners are fixed, and change neither.
    """
    tputs = [r["throughput_tps"] for r in rows]
    utils = [fh.run_utility({"throughput_tps": r["throughput_tps"],
                             "ttft_p99": r["ttft_p99_ms"],
                             "tpot_p99": r["tpot_p99_ms"]}, scales)
             for r in rows]
    ids = [{"max_batch_size": r["max_batch_size"], "chunk_size": r["chunk_size"]}
           for r in rows]
    t_min, t_max = min(tputs), max(tputs)
    u_min, u_max = min(utils), max(utils)
    return {
        "n_grid_points": len(rows),
        "held_out_seed": VALIDATION_SEED,
        "throughput_tps": {
            "min": t_min, "max": t_max, "range_absolute": t_max - t_min,
            # H1's definition; the saturated side goes through the same function.
            "range_relative_pct": sref.relative_range_pct(t_min, t_max),
            "median": statistics.median(tputs),
            "flatness": flatness_diagnostics(tputs, ids),
        },
        "utility_v2": {
            "min": u_min, "max": u_max, "range_absolute": u_max - u_min,
            "range_relative_pct": (100.0 * (u_max - u_min) / abs(u_max)
                                   if u_max else None),
            "median": statistics.median(utils),
            "flatness": flatness_diagnostics(utils, ids),
            "relative_range_caveat": (
                "utility_v2 is a bounded weighted score with no natural zero, "
                "so its RELATIVE range is reported for completeness and the "
                "ABSOLUTE range is the meaningful figure"),
        },
        "arrival_limited_caveat": (
            "At sub-critical offered load, end-to-end throughput may be "
            "arrival-limited. Static-Tput remains the pre-registered argmax, "
            "but near-ties and the size of the near-optimal set are reported "
            "so a noise-level argmax is not over-interpreted."),
        "static_tput_identity": selection["static_tput"]["winner"],
        "static_utility_identity": selection["static_utility"]["winner"],
        "objectives_agree": selection["objectives_agree"],
        "comparison_with_saturated_point": {
            "method": ("DESCRIPTIVE. Two deterministic 28-point grids at two "
                       "operating points are compared by their ranges. No "
                       "significance test is applied and no p-value is "
                       "computed."),
            "saturated_reference": SATURATED_REFERENCE,
            "saturated_side": ("hypotheses.H1, from the hash-verified frozen "
                               "grid in saturated_reference_verification; "
                               "unresolved there if verification failed"),
        },
    }


def run_one(name: str, model, tokenizer, n: int, seed: int, plan: dict, *,
            static_cfg=None, reward_fn=None, scales=None) -> dict:
    """Serve one treatment once. Identical to the saturated run bar arrivals."""
    if name in _STATIC_CONFIGS:
        out = run_static(static_cfg, model, tokenizer, n, seed, plan)
        out["applied_config"] = static_cfg.as_dict()
        out["run_utility_v2"] = fh.run_utility(out, scales)
        out["arrival_validation"] = {
            ph: arrival_report(offs, ph, plan)
            for ph, offs in out["arrivals"].items()}
        out["offered_load"] = offered_load_report(out["arrival_validation"], plan)
        out["drain_overrun"] = drain_overrun(out, out["arrivals"], plan)
        # `run_static` owns the specs, so it records the splice itself; the
        # invariant was already asserted inside build_poisson_workload.
        return out

    specs, arrivals = build_poisson_workload(tokenizer, n, seed, plan)
    sched = abl._new_scheduler(model)
    tracker = MetricsTracker(window=max(50, n))
    controller = CARLController(
        scheduler=sched, bandit=fh._build_bandit(name),
        observe_interval=OBSERVE_INTERVAL, slo=abl._SLO, metrics=tracker,
        reward_fn=reward_fn)

    out = abl._serve(sched, specs, controller=controller, tracker=tracker,
                     phase1_open_loop=True)
    out["arrivals"] = arrivals
    out["arrival_validation"] = {ph: arrival_report(offs, ph, plan)
                                 for ph, offs in arrivals.items()}
    out["offered_load"] = offered_load_report(out["arrival_validation"], plan)
    out["drain_overrun"] = drain_overrun(out, arrivals, plan)
    out["phase_boundary"] = phase_boundary_report(specs, arrivals, n)
    out["run_utility_v2"] = fh.run_utility(out, scales)

    cycles = fh._cycle_records(controller)
    out["cycles"] = cycles
    out["arm_set"] = _ARM_SET_FOR[name]
    out["bandit_cls"] = type(controller.bandit).__name__
    out["controller_stats"] = controller.stats()
    out["reward_validity"] = controller.reward_validity_report()

    arms_played = [c["selected_arm"] for c in cycles]
    out["decisions"] = len(arms_played)
    out["arm_changes"] = sum(1 for a, b in zip(arms_played, arms_played[1:])
                             if a != b)
    out["arm_change_rate"] = (out["arm_changes"] / (len(arms_played) - 1)
                              if len(arms_played) > 1 else 0.0)
    out["arm_histogram"] = {}
    for c in cycles:
        key = f"{c['regime']}|arm{c['selected_arm']}"
        out["arm_histogram"][key] = out["arm_histogram"].get(key, 0) + 1
    # THE SAME diagnostic, imported. Not a more favourable definition.
    out["convergence"] = fh.convergence_diagnostic(cycles)

    valid = [c["reward"] for c in cycles if c["reward_valid"]]
    out["n_valid_rewards"] = len(valid)
    out["n_withheld_rewards"] = len(cycles) - len(valid)
    out["mean_valid_cycle_utility"] = (statistics.fmean(valid) if valid else None)
    out["valid_reward_spread"] = (max(valid) - min(valid)) if valid else None
    out["n_distinct_valid_rewards"] = len(set(valid))
    try:
        out["reward_gate"] = gate_live_reward_stream(
            valid, context=f"subcritical_poisson {name} seed={seed} (valid rewards only)",
            threshold=MIN_LIVE_REWARD_SPREAD, min_distinct=MIN_LIVE_DISTINCT)
    except DegenerateRewardError as exc:
        # The gate's verdict stands and propagates; the run that produced it is
        # attached so the driver can write its measurements, controller trace
        # and gate evidence before anything unwinds.
        out["reward_gate"] = {
            "passed": False, "message": str(exc), "valid_rewards": valid,
            "n": len(valid), "n_distinct": len(set(valid)),
            "reward_spread": (max(valid) - min(valid)) if valid else None,
            "threshold": MIN_LIVE_REWARD_SPREAD, "min_distinct": MIN_LIVE_DISTINCT,
        }
        exc.failed_run = out
        raise
    return out


# ===========================================================================
# 5. Statistics.
# ===========================================================================

_METRICS = ["throughput_tps", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
            "slo_rate", "run_utility_v2"]


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _midranks(values: list) -> list:
    """Ranks of `values` with ties averaged (the usual Wilcoxon midranks)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank_exact(diffs: list) -> dict:
    """EXACT two-sided Wilcoxon signed-rank test, by enumerating sign patterns.

    Reported as ROBUSTNESS only; it is never the headline. With n = 10 seeds the
    null distribution has 2^10 = 1024 points, so it is enumerated rather than
    approximated -- and enumerating over the OBSERVED midranks makes the test a
    valid conditional permutation test even when |d| has ties, which the
    textbook table does not handle.

    Zero differences are NOT dropped and NOT adjusted: the signed-rank test has
    no assumption-free treatment for them, and a bit-identical pair is an
    identity rather than a tie. When any difference is zero the test is declined
    and says so.
    """
    n = len(diffs)
    if n == 0:
        return {"applicable": False, "reason": "no paired differences"}
    zeros = sum(1 for d in diffs if d == 0.0)
    if zeros:
        return {
            "applicable": False,
            "n_zero_differences": zeros,
            "reason": (
                "exact Wilcoxon requires no zero differences; {} of {} paired "
                "differences are exactly zero, which is an IDENTITY between the "
                "two treatments on those seeds, not a tie to be broken"
            ).format(zeros, n),
        }
    ranks = _midranks([abs(d) for d in diffs])
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    total = sum(ranks)
    centre = total / 2.0
    observed = abs(w_plus - centre)

    # Enumerate every sign assignment; under H0 each is equally likely.
    counts = {0.0: 1}
    for r in ranks:
        nxt = {}
        for s, c in counts.items():
            nxt[s] = nxt.get(s, 0) + c
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    n_patterns = 2 ** n
    extreme = sum(c for s, c in counts.items() if abs(s - centre) >= observed - 1e-12)
    return {
        "applicable": True,
        "n": n,
        "w_plus": w_plus,
        "w_minus": total - w_plus,
        "null_mean_w_plus": centre,
        "p_value_two_sided_exact": extreme / n_patterns,
        "method": ("exact enumeration of all 2^n sign assignments over the "
                   "observed midranks (a conditional permutation test; valid "
                   "under ties in |d|)"),
        "reporting_note": ("robustness check only -- the paired mean difference, "
                           "its 95% CI and the per-seed signs are the result"),
    }


# --- exact reference distributions (no scipy on the runtime) ----------------
#
# Pure-Python, because the archive must not grow a dependency to state an
# interval. Each is checked in the tests against closed forms (integer-shape
# Gamma via the Poisson sum, Student-t at 1 and 2 degrees of freedom) and
# against published table values.

_Z_975 = 1.959963984540054      # Phi^-1(0.975), the secondary normal interval


def _gamma_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x): series for x < a + 1, else
    the continued fraction for Q (modified Lentz)."""
    if x <= 0.0:
        return 0.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap, term = a, 1.0 / a
        total = term
        for _ in range(100000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * 1e-17:
                break
        return min(1.0, total * math.exp(log_front))
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-16:
            break
    return max(0.0, 1.0 - math.exp(log_front) * h)


def _bisect_quantile(cdf, p: float, lo: float, hi: float) -> float:
    """Smallest x with cdf(x) >= p, to double precision, for increasing cdf."""
    while cdf(hi) < p:
        lo, hi = hi, hi * 2.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gamma_quantile(shape: float, p: float, rate: float = 1.0) -> float:
    """Quantile of Gamma(shape, rate)."""
    if not (0.0 < p < 1.0) or shape <= 0 or rate <= 0:
        raise ValueError(f"gamma_quantile({shape}, {p}, rate={rate})")
    return _bisect_quantile(lambda x: _gamma_p(shape, x), p, 0.0,
                            max(1.0, 2.0 * shape)) / rate


def gamma_mean_bounds(k: int, lam: float, alpha: float) -> tuple:
    """Exact two-sided (1 - alpha) bounds for the MEAN of k Exp(lam) gaps.

    The SUM of k gaps is Gamma(shape=k, rate=lam); the MEAN is
    Gamma(shape=k, rate=k*lam). Using the sum's distribution for the mean
    would put the bounds near k/lam -- k times too wide to reject anything.
    """
    if k < 1:
        raise ValueError(f"need at least one gap, got k={k}")
    return (gamma_quantile(k, alpha / 2.0, rate=k * lam),
            gamma_quantile(k, 1.0 - alpha / 2.0, rate=k * lam))


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 100000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-16:
            break
    return h


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    tail = 0.5 * _beta_inc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def student_t_quantile(p: float, df: int) -> float:
    """Quantile of Student's t with `df` degrees of freedom, for p in (0.5, 1)."""
    if not (0.5 < p < 1.0) or df < 1:
        raise ValueError(f"student_t_quantile({p}, df={df})")
    return _bisect_quantile(lambda x: student_t_cdf(x, df), p, 0.0, 4.0)


def _paired(a_vals: list, b_vals: list) -> dict:
    """Paired a - b over matched seeds: mean, t CI (primary), z CI (secondary),
    Cohen's d, signs, exact Wilcoxon."""
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else None
    se = (sd / math.sqrt(n)) if sd is not None else None
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    zero = sum(1 for d in diffs if d == 0.0)
    if n < 2:
        t_crit, ci_t, ci_z = None, None, None
        ci_status = ("undefined: a paired interval needs n >= 2 differences "
                     f"(n = {n}); no interval is reported")
    else:
        t_crit = student_t_quantile(0.975, n - 1)
        ci_t = [mean - t_crit * se, mean + t_crit * se]
        ci_z = [mean - _Z_975 * se, mean + _Z_975 * se]
        ci_status = ("ok" if sd > 0 else
                     "zero variance: every paired difference is identical, so "
                     "both intervals collapse to the point "
                     f"{mean!r}; this is a degenerate interval, not certainty")
    return {
        "n": n,
        "mean_difference": mean,
        "std_difference": sd,
        "ci95_t": ci_t,
        "ci95_t_critical_value": t_crit,
        "ci95_t_df": (n - 1) if n >= 2 else None,
        "ci95_t_method": ("PRIMARY: mean +/- t(0.975, n-1) * sd / sqrt(n), sd "
                          "the sample SD of the paired differences, t computed "
                          "for this n"),
        "ci95_status": ci_status,
        "ci95_z_normal_approx_secondary": ci_z,
        "ci95_z_method": ("SECONDARY normal approximation, mean +/- 1.96 * sd / "
                          "sqrt(n); reported ONLY for comparison with the frozen "
                          "saturated analysis, which used it"),
        "ci95_limitations": (
            "Assumes the paired differences are approximately normal, which "
            "ten seeds can barely check; skew or outliers (for example in drain "
            "tails) degrade coverage. It covers seed-to-seed variation on one "
            "GPU, not hardware variation. Per-comparison only: no familywise "
            "adjustment across the four comparisons or the two metrics."),
        "cohens_d_paired": (mean / sd) if sd else None,
        "all_diffs_zero": all(d == 0.0 for d in diffs),
        "per_seed_difference": diffs,
        "sign_of_per_seed_difference": [
            (1 if d > 0 else (-1 if d < 0 else 0)) for d in diffs],
        "n_positive": pos, "n_negative": neg, "n_zero": zero,
        "signs_all_same_direction": (pos == n) or (neg == n),
        "wilcoxon_signed_rank": wilcoxon_signed_rank_exact(diffs),
        "reporting_rule": ("the paired mean difference, its 95% CI, Cohen's d "
                           "and the per-seed signs are the result; the Wilcoxon "
                           "p-value is a robustness check and is never the "
                           "headline"),
    }


# --- H5, as amended 2026-10-22 ----------------------------------------------

H5_FAMILY_ALPHA = 0.05
H5_HELD_OUT_ALPHA = 0.05


def h5_reference(plan: dict, n: int, seeds: list) -> dict:
    """The H5 bounds, from lambda and n ALONE -- computed before any run is
    measured, so no threshold can be chosen by looking at which draws pass."""
    counts = {"interactive": n // 2, "batch": n - n // 2}
    n_cells = 2 * len(seeds)
    per_cell = H5_FAMILY_ALPHA / n_cells if n_cells else None
    phases = {}
    for ph, m in counts.items():
        lam = plan["phases"][ph]["lambda_rps"]
        k = m - 1
        entry = {"lambda_rps": lam, "expected_mean_interarrival_s": 1.0 / lam,
                 "n_requests": m, "k_gaps": k}
        if k >= 1 and per_cell:
            fam = gamma_mean_bounds(k, lam, per_cell)
            ho = gamma_mean_bounds(k, lam, H5_HELD_OUT_ALPHA)
            entry.update({
                "family_bounds_mean_interarrival_s": list(fam),
                "family_bounds_over_expected_mean": [fam[0] * lam, fam[1] * lam],
                "held_out_bounds_mean_interarrival_s": list(ho),
                "held_out_bounds_over_expected_mean": [ho[0] * lam, ho[1] * lam],
            })
        else:
            entry["why_no_bounds"] = "fewer than two arrivals in this phase"
        phases[ph] = entry
    return {
        "distribution": ("mean of k consecutive Exp(lambda) gaps ~ "
                         "Gamma(shape=k, rate=k*lambda); exact two-sided "
                         "quantiles"),
        "k_definition": ("k = (requests in the phase) - 1. build_arrivals places "
                         "a phase's first arrival at its offset and discards the "
                         "draw after its last, and the BATCH stream starts "
                         "exactly at the last INTERACTIVE arrival, so each phase "
                         "contributes exactly k gaps and none spans the "
                         "boundary"),
        "family": {"seeds": list(seeds), "phases": list(counts),
                   "n_cells": n_cells, "familywise_alpha": H5_FAMILY_ALPHA,
                   "per_cell_alpha": per_cell,
                   "correction": ("Bonferroni; controls familywise error at <= "
                                  "0.05 without assuming the cells are "
                                  "independent")},
        "held_out": {"seed": VALIDATION_SEED, "alpha": H5_HELD_OUT_ALPHA,
                     "in_family": False},
        "phases": phases,
        "computed_from": "lambda and n only, before any measurement",
        "interpretation": PRE_REGISTRATION["H5_arrival_process"][
            "family_rejection_means"],
    }


def _gap_mean(offsets: list):
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    return (statistics.fmean(gaps) if gaps else None), gaps


def h5_family_report(reference: dict, offsets_by_seed: dict,
                     held_out_offsets: dict, plan: dict, n: int) -> dict:
    """Every family cell and the held-out cells, plus generator invariants.

    The invariants say the recorded offsets ARE the seeded stream and obey the
    splice; they are not, and are not reported as, proof that the stream is
    exponential. That rests on the sampler itself (random.expovariate inside
    src.carl.live.build_arrivals) and on a large-sample test of it run on a
    namespace no evaluation uses.
    """
    def _cell(seed, ph, offs, bounds_key):
        ref = reference["phases"][ph]
        mean, gaps = _gap_mean(offs)
        bounds = ref.get(bounds_key)
        inside = (bounds is not None and mean is not None
                  and bounds[0] <= mean <= bounds[1])
        return {"seed": seed, "phase": ph, "k_gaps": len(gaps),
                "k_expected": ref["k_gaps"],
                "empirical_mean_interarrival_s": mean,
                "bounds_mean_interarrival_s": bounds, "inside": bool(inside)}

    cells = [_cell(s, ph, offsets_by_seed[s][ph], "family_bounds_mean_interarrival_s")
             for s in reference["family"]["seeds"] if s in offsets_by_seed
             for ph in reference["family"]["phases"]]
    held = [_cell(VALIDATION_SEED, ph, held_out_offsets[ph],
                  "held_out_bounds_mean_interarrival_s")
            for ph in reference["family"]["phases"]]
    rejected = [c for c in cells if not c["inside"]]

    def _invariants(seed, offs):
        regen = poisson_arrivals(n, seed, plan)
        all_gaps = [g for ph in _PHASES for g in _gap_mean(offs[ph])[1]]
        return {
            "regenerates_from_seeded_stream": offs == regen,
            "all_gaps_positive_and_finite": all(
                math.isfinite(g) and g > 0 for g in all_gaps),
            "batch_starts_at_last_interactive_arrival": (
                bool(offs["interactive"]) and bool(offs["batch"])
                and offs["batch"][0] == offs["interactive"][-1]),
            "counts": {ph: len(offs[ph]) for ph in _PHASES},
        }

    inv = {str(s): _invariants(s, o) for s, o in offsets_by_seed.items()}
    inv[str(VALIDATION_SEED)] = _invariants(VALIDATION_SEED, held_out_offsets)
    return {
        "n_cells": len(cells),
        "n_cells_expected": reference["family"]["n_cells"],
        "cells": cells,
        "rejected_cells": rejected,
        "family_rejected": bool(rejected),
        "held_out_cells": held,
        "generator_invariants": inv,
        "generator_invariants_note": (
            "deterministic regeneration, positive finite gaps, the splice and "
            "the counts. These show the recorded arrivals are the seeded "
            "stream with the registered structure; they do NOT by themselves "
            "show the sampling is exponential"),
        "interpretation": reference["interpretation"],
        "no_seed_rejection": "every cell is reported; no seed is dropped, "
                             "replaced or retried",
    }


# --- H2 ----------------------------------------------------------------------


def h2_relative_difference(a_vals: list, b_vals: list) -> dict:
    """|mean(a - b)| / mean(b): b is the comparator. None, with a reason,
    when it is not defined."""
    if not a_vals or len(a_vals) != len(b_vals):
        return {"value": None, "why": "no matched seeds"}
    mean_b = statistics.fmean(b_vals)
    if not (math.isfinite(mean_b) and mean_b > 0):
        return {"value": None, "why": f"comparator mean throughput is {mean_b!r}"}
    diff = statistics.fmean(a - b for a, b in zip(a_vals, b_vals))
    return {"value": abs(diff) / mean_b, "abs_mean_paired_difference": abs(diff),
            "comparator_mean": mean_b, "n": len(a_vals)}


def saturated_hypothesis_inputs(reference: dict | None) -> dict | None:
    """The saturated side of H1/H2, by THIS module's definitions, computed from
    a VERIFIED reference record only. None otherwise: there is no fallback."""
    if (reference or {}).get("status") != "verified":
        return None
    g = reference["grid"]
    h2 = {}
    for label, a, b, _metric, _q in COMPARISONS:
        if label not in H2_COMPARATORS:
            continue
        if b != H2_COMPARATORS[label]:
            raise RuntimeError(f"H2 comparator for {label} is {b!r}, "
                               f"registered {H2_COMPARATORS[label]!r}")
        h2[label] = {"a": a, "comparator": b, **h2_relative_difference(
            sref.per_seed_in_order(reference, a),
            sref.per_seed_in_order(reference, b))}
    return {
        "H1": {"n_configurations": g["n_configurations"],
               "throughput_min": g["throughput_min"],
               "throughput_max": g["throughput_max"],
               "relative_throughput_range_pct": sref.relative_range_pct(
                   g["throughput_min"], g["throughput_max"])},
        "H2": h2,
        "source": {k: reference.get(k) for k in (
            "source_member", "artifact_sha256", "raw_manifest_sha256",
            "validator_version")},
    }


def hypotheses_status(results: dict, reference: dict | None = None,
                      not_evaluable: list | tuple = ()) -> dict:
    """H1/H2: the sub-critical side, the verified saturated side, and the
    verdict the PRE-REGISTERED rule gives -- only for the registered run.

    `not_evaluable` lists why THIS run is not the registered one (a smoke, a
    deviating grid, ...): its numbers are reported, but no verdict is given.
    """
    sat = saturated_hypothesis_inputs(reference)
    ref_status = (reference or {}).get("status", "absent")
    land = (results.get("landscape") or {}).get("throughput_tps") or {}
    sub_h1 = land.get("range_relative_pct")

    def _gate():
        if sat is None:
            return "unresolved_saturated_reference_unverified"
        if not_evaluable:
            return "not_evaluated_unregistered_run"
        return None

    h1_status = _gate()
    if h1_status is None:
        if sub_h1 is None:
            h1_status = "unresolved_subcritical_value_missing"
        else:
            # Falsified if EQUAL to or larger than the saturated sweep's.
            h1_status = ("supported"
                         if sub_h1 < sat["H1"]["relative_throughput_range_pct"]
                         else "falsified")

    h2 = {}
    for label, comparator in H2_COMPARATORS.items():
        c = (results.get("comparisons") or {}).get(label) or {}
        h2[label] = {"comparator": comparator,
                     "subcritical": c.get("h2_relative_paired_difference"),
                     "saturated": (sat or {}).get("H2", {}).get(label)}
    h2_status = _gate()
    if h2_status is None:
        pairs = [((v["subcritical"] or {}).get("value"),
                  (v["saturated"] or {}).get("value")) for v in h2.values()]
        if any(s is None or t is None for s, t in pairs):
            h2_status = "unresolved_value_missing"
        elif any(s > t for s, t in pairs):
            h2_status = "falsified"
        elif all(s < t for s, t in pairs):
            h2_status = "supported"
        else:
            # Neither larger anywhere nor smaller everywhere: an exact tie.
            h2_status = "neither_supported_nor_falsified_tie"
    out = {
        "H1": {"definition": ("100 * (max - min) / max of throughput_tps over "
                              "the 28-point held-out grid, at each operating "
                              "point"),
               "subcritical_relative_throughput_range_pct": sub_h1,
               "saturated_relative_throughput_range_pct":
                   (sat or {}).get("H1", {}).get("relative_throughput_range_pct"),
               "saturated": (sat or {}).get("H1"),
               "status": h1_status},
        "H2": {"definition": H2_DEFINITION, "per_comparison": h2,
               "status": h2_status},
        "saturated_reference_status": ref_status,
        "saturated_reference_source": (sat or {}).get("source"),
        "not_evaluable_because": list(not_evaluable),
        "comparability": SATURATED_COMPARABILITY,
        "rule": ("unresolved means neither supported nor falsified; nothing on "
                 "the saturated side is estimated, reconstructed or taken from "
                 "the rounded transcription; a verdict is given only for the "
                 "registered published run"),
    }
    if sat is None:
        out["saturated_reference_problems"] = (reference or {}).get(
            "problems", ["no saturated reference record"])
        for h in ("H1", "H2"):
            out[h]["missing_inputs"] = SATURATED_REFERENCE[
                "inputs_for_hypotheses"][h]
    return out


# --- descriptive phase / regime attribution -----------------------------------

_SCHEDULED_PHASE = {"INTERACTIVE": "interactive", "BATCH": "batch"}


def phase_regime_report(run: dict) -> dict:
    """Scheduled arrival phase vs inferred controller regime, per phase drain
    timing, and periods when both phases are in flight. DESCRIPTIVE.

    Timing semantics, verified in ablation_live._serve: step_log[*].t, every
    request's stage_timestamps (arrival_target_s, submit_s, finish_s) and the
    phase boundary's scheduled offsets are all seconds since the SAME t0. A
    controller cycle is joined to time ONLY through its scheduler step: the
    controller is called with sched._step_idx right after the step that
    step_log recorded under the same index. Its own t_monotonic_s is a raw
    perf_counter with a different origin and is never compared with them.

    Arrival windows are sequential by construction; EXECUTION is not, since
    INTERACTIVE requests can still be in flight after the first BATCH
    arrival. Both attributions are therefore reported. The regime is the
    classifier's inference, never ground truth.
    """
    problems = []
    reqs = []
    for r in run.get("requests") or []:
        ts = r.get("stage_timestamps") or {}
        ph = _SCHEDULED_PHASE.get(r.get("regime"))
        times = (ts.get("arrival_target_s"), ts.get("submit_s"), ts.get("finish_s"))
        if ph is None or not all(_finite(v) for v in times):
            problems.append(f"request {r.get('request_id')!r} lacks a scheduled "
                            "phase or finite arrival/submit/finish times")
            continue
        reqs.append((ph,) + times)

    drain = {}
    for ph in _PHASES:
        mine = [q for q in reqs if q[0] == ph]
        if not mine:
            continue
        last_arr = max(q[1] for q in mine)
        last_fin = max(q[3] for q in mine)
        drain[ph] = {"last_scheduled_arrival_s": last_arr,
                     "first_submit_s": min(q[2] for q in mine),
                     "last_finish_s": last_fin,
                     "drain_after_last_arrival_s": last_fin - last_arr}
    mixed = None
    if "interactive" in drain and "batch" in drain:
        start, end = drain["batch"]["first_submit_s"], drain["interactive"]["last_finish_s"]
        mixed = {"start_s": start, "end_s": end,
                 "duration_s": max(0.0, end - start),
                 "both_phases_in_flight": end > start}

    out = {"scheduled_phase_drain": drain, "mixed_execution_window": mixed,
           "timing_origin": "seconds since ablation_live._serve t0",
           "regime_is_inferred": True}
    cycles = run.get("cycles")
    if not isinstance(cycles, list):
        out["controller_attribution"] = None
        out["problems"] = problems
        out["attribution_reliable"] = not problems
        return out

    steps = [e.get("step") for e in run.get("step_log") or []]
    t_by_step = {e.get("step"): e.get("t") for e in run.get("step_log") or []}
    if len(set(steps)) != len(steps):
        problems.append("step_log scheduler-step identifiers are not unique")
    first_batch = (run.get("phase_boundary") or {}).get("first_batch_arrival_s")
    if not _finite(first_batch):
        problems.append("phase_boundary.first_batch_arrival_s is missing")
    table: dict = {}
    for c in cycles:
        t = t_by_step.get(c.get("step"))
        if not _finite(t) or not _finite(first_batch):
            problems.append(f"cycle at step {c.get('step')!r} has no step_log time")
            continue
        window = "interactive" if t < first_batch else "batch"
        live = {q[0] for q in reqs if q[2] <= t < q[3]}
        comp = ("idle" if not live else "mixed" if len(live) == 2
                else f"{next(iter(live))}_only")
        key = f"{window}_window|{comp}|{c.get('regime')}"
        table[key] = table.get(key, 0) + 1

    def _share(pred):
        sel = {k: v for k, v in table.items() if pred(k)}
        total = sum(sel.values())
        hit = sum(v for k, v in sel.items() if k.endswith("|batch"))
        return {"cycles": total,
                "classified_batch": hit,
                "share_classified_batch": (hit / total) if total else None}

    out["controller_attribution"] = {
        "counts_by_window_composition_regime": dict(sorted(table.items())),
        "batch_arrival_window": _share(lambda k: k.startswith("batch_window|")),
        "batch_only_in_flight": _share(lambda k: "|batch_only|" in k),
        "mixed_in_flight": _share(lambda k: "|mixed|" in k),
        "composition_definition": ("requests with submit_s <= t < finish_s at "
                                   "the time t of the cycle's scheduler step"),
    }
    out["problems"] = problems
    out["attribution_reliable"] = not problems
    return out


# ===========================================================================
# 6. Artifacts. Atomic, schema'd, fingerprinted, and checkpointed at every
#    boundary.
# ===========================================================================


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str, payload) -> str:
    """Write `payload` so the destination is never a partial file.

    The JSON is serialised COMPLETELY in memory first (a serialisation error
    touches nothing), written to a temp file in the destination's directory,
    flushed and fsync'd, then moved over the destination with `os.replace`.

    Three separate statements, deliberately not merged:
      * PROCESS INTERRUPTION (exception, Ctrl-C, killed process) on the local
        filesystems tested (NTFS here, by the test suite): the destination
        is the previous complete file or the new complete file, never a
        partial one.
      * ATOMIC REPLACEMENT: `os.replace` is a single rename on POSIX and
        MoveFileEx on Windows, both atomic within one local volume.
      * NOT GUARANTEED: power loss or kernel crash (the directory entry is not
        fsync'd, so which version is visible afterwards is not promised), and
        network or FUSE filesystems such as Google Drive, whose rename and
        fsync semantics are not those of a local disk. Nothing here is claimed
        for them.
    """
    srcprov.atomic_write_json(path, payload)
    return path


def _read_json(path: str) -> tuple:
    """(payload, None), or (None, why) -- never raises for a bad file."""
    if not os.path.isfile(path):
        return None, "its evidence file is missing"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"its evidence file does not parse ({type(exc).__name__})"


def _rel(path: str) -> str:
    return os.path.relpath(path, _REPO_ROOT).replace(os.sep, "/")


def supersede(paths: RunPaths, path: str) -> str | None:
    """Move an untrusted evidence file aside, never delete it."""
    if not os.path.isfile(path):
        return None
    os.makedirs(paths.superseded_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = os.path.join(paths.superseded_dir,
                        f"{stamp}__{os.path.basename(path)}")
    os.replace(path, dest)
    return dest


class UnitContext:
    """What a unit's evidence must agree with to be trusted on resume."""

    def __init__(self, paths: RunPaths, fingerprint_sha256: str, n: int,
                 plan: dict, calibration: dict | None = None):
        self.paths = paths
        self.fingerprint_sha256 = fingerprint_sha256
        self.n = n
        self.plan = plan
        self.calibration = calibration
        self._arrivals: dict = {}

    def arrivals(self, seed: int) -> dict:
        if seed not in self._arrivals:
            self._arrivals[seed] = poisson_arrivals(self.n, seed, self.plan)
        return self._arrivals[seed]


def _finite(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


_METRIC_KEYS = ("throughput_tps", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms",
                "tpot_p99_ms", "slo_rate", "wall_s")
_PHASES = ("interactive", "batch")


def _header_problems(payload: dict, schema: str, unit_id: str,
                     ctx: UnitContext) -> list:
    p = []
    if payload.get("schema") != schema:
        p.append(f"schema is {payload.get('schema')!r}, expected {schema!r}")
    if payload.get("experiment_version") != EXPERIMENT_VERSION:
        p.append("it was written by experiment version "
                 f"{payload.get('experiment_version')!r}")
    if payload.get("status") != "complete":
        p.append(f"its status is {payload.get('status')!r}")
    if payload.get("unit_id") != unit_id:
        p.append(f"it records unit {payload.get('unit_id')!r}")
    if payload.get("fingerprint_sha256") != ctx.fingerprint_sha256:
        p.append("it belongs to a different experiment fingerprint")
    if payload.get("n_requests") != ctx.n:
        p.append(f"it records n={payload.get('n_requests')!r}, expected {ctx.n}")
    return p


def _evidence_problems(rec: dict, offsets, requests, seed: int,
                       ctx: UnitContext) -> list:
    """Metrics, raw arrivals and the nested blocks aggregation reads."""
    p = [f"{k} is not a finite number" for k in _METRIC_KEYS
         if not _finite(rec.get(k))]
    expected = ctx.arrivals(seed)
    if offsets != expected:
        p.append(f"its arrival offsets are not the seeded draw for seed {seed}")
    if not isinstance(requests, list) or len(requests) != ctx.n:
        p.append(f"it holds {len(requests) if isinstance(requests, list) else 'no'} "
                 f"request records, expected {ctx.n}")

    av = rec.get("arrival_validation")
    ol = (rec.get("offered_load") or {}).get("per_phase") if isinstance(
        rec.get("offered_load"), dict) else None
    do = rec.get("drain_overrun")
    for ph in _PHASES:
        r = av.get(ph) if isinstance(av, dict) else None
        if not isinstance(r, dict):
            p.append(f"arrival_validation.{ph} is missing")
        else:
            for k in ("empirical_rho", "empirical_mean_interarrival_s",
                      "empirical_lambda_rps"):
                if not _finite(r.get(k)):
                    p.append(f"arrival_validation.{ph}.{k} is not finite")
            if r.get("requested_lambda_rps") != ctx.plan["phases"][ph]["lambda_rps"]:
                p.append(f"arrival_validation.{ph} was generated at a different "
                         "lambda")
            if not isinstance(r.get("mean_within_ci95"), bool):
                p.append(f"arrival_validation.{ph}.mean_within_ci95 is missing")
            if r.get("n_requests") != len(expected[ph]):
                p.append(f"arrival_validation.{ph} counts the wrong requests")
        if not (isinstance(ol, dict) and isinstance(ol.get(ph), dict)
                and _finite(ol[ph].get("empirical_rho"))):
            p.append(f"offered_load.per_phase.{ph}.empirical_rho is missing")
        if not (isinstance(do, dict) and isinstance(do.get("per_phase"), dict)
                and isinstance(do["per_phase"].get(ph), dict)):
            p.append(f"drain_overrun.per_phase.{ph} is missing")
    if not (isinstance(do, dict) and _finite(do.get("arrival_window_overrun_ratio"))):
        p.append("drain_overrun.arrival_window_overrun_ratio is not finite")

    pb = rec.get("phase_boundary")
    if not isinstance(pb, dict):
        p.append("phase_boundary is missing")
    else:
        if pb.get("n_requests") != ctx.n:
            p.append("phase_boundary counts the wrong requests")
        for flag in ("phases_do_not_overlap", "head_is_all_interactive",
                     "tail_is_all_batch"):
            if pb.get(flag) is not True:
                p.append(f"phase_boundary.{flag} is not True")
    return p


def _controller_problems(payload: dict, name: str) -> list:
    """Controller evidence must be present AND agree with its own cycle trace."""
    p = []
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or not all(
            isinstance(c, dict) and _is_int(c.get("selected_arm"))
            and isinstance(c.get("regime"), str)
            and isinstance(c.get("reward_valid"), bool) for c in cycles):
        return ["its controller cycle trace is missing or malformed"]
    if payload.get("arm_set") != _ARM_SET_FOR[name]:
        p.append(f"it records arm set {payload.get('arm_set')!r}")
    arms = [c["selected_arm"] for c in cycles]
    if payload.get("decisions") != len(arms):
        p.append("decisions does not equal the number of recorded cycles")
    changes = sum(1 for a, b in zip(arms, arms[1:]) if a != b)
    if payload.get("arm_changes") != changes:
        p.append("arm_changes does not match the cycle trace")
    rate = changes / (len(arms) - 1) if len(arms) > 1 else 0.0
    if not _finite(payload.get("arm_change_rate")) or payload["arm_change_rate"] != rate:
        p.append("arm_change_rate does not match the cycle trace")
    valid = [c.get("reward") for c in cycles if c["reward_valid"]]
    if not valid or not all(_finite(v) for v in valid):
        p.append("its valid rewards are missing or not finite")
    else:
        if payload.get("mean_valid_cycle_utility") != statistics.fmean(valid):
            p.append("mean_valid_cycle_utility does not match the cycle trace")
        if payload.get("valid_reward_spread") != max(valid) - min(valid):
            p.append("valid_reward_spread does not match the cycle trace")
        if payload.get("n_distinct_valid_rewards") != len(set(valid)):
            p.append("n_distinct_valid_rewards does not match the cycle trace")
    hist: dict = {}
    for c in cycles:
        k = f"{c['regime']}|arm{c['selected_arm']}"
        hist[k] = hist.get(k, 0) + 1
    if payload.get("arm_histogram") != hist:
        p.append("arm_histogram does not match the cycle trace")
    if _canonical(payload.get("convergence")) != _canonical(
            fh.convergence_diagnostic(cycles)):
        p.append("its convergence record does not match the imported "
                 "diagnostic applied to its own cycle trace")
    rv = payload.get("reward_validity")
    if not isinstance(rv, dict):
        p.append("reward_validity is missing")
    else:
        for k in ("warmup_cycles", "valid_reward_updates",
                  "withheld_after_first_valid"):
            if not _is_int(rv.get(k)):
                p.append(f"reward_validity.{k} is not an integer")
        for k in ("first_valid_reward_cycle_index",
                  "first_valid_reward_scheduler_step", "warmup_duration_s"):
            if k not in rv:
                p.append(f"reward_validity.{k} is missing")
    gate = payload.get("reward_gate")
    if not (isinstance(gate, dict) and gate.get("passed") is True
            and _finite(gate.get("reward_spread")) and _is_int(gate.get("n_distinct"))):
        p.append("its reward gate record is missing or did not pass")
    return p


def validate_sweep_payload(payload, mb: int, cs: int, ctx: UnitContext) -> list:
    if not isinstance(payload, dict):
        return ["it is not a JSON object"]
    p = _header_problems(payload, SWEEP_UNIT_SCHEMA, sweep_unit_id(mb, cs), ctx)
    if payload.get("seed") != VALIDATION_SEED:
        p.append(f"it records seed {payload.get('seed')!r}, not the held-out "
                 f"seed {VALIDATION_SEED}")
    row = payload.get("row")
    if not isinstance(row, dict):
        return p + ["it carries no measurement row"]
    if (row.get("max_batch_size"), row.get("chunk_size")) != (mb, cs):
        p.append("its row is for a different grid point")
    want_cfg = CARLConfig(max_batch_size=mb, chunk_size=cs).clamp().as_dict()
    if _canonical(row.get("config")) != _canonical(want_cfg):
        p.append("its applied configuration is not the grid point's")
    return p + _evidence_problems(row, payload.get("arrival_offsets_s"),
                                  payload.get("requests"), VALIDATION_SEED, ctx)


def _static_cfg_dict(name: str, calibration: dict) -> dict:
    key = {"Static-Tput": "static_tput", "Static-Utility": "static_utility"}[name]
    return calibration["binding"][key]


def validate_eval_payload(payload, name: str, seed: int, ctx: UnitContext) -> list:
    """Every reason this raw unit may NOT be reused. Empty means trusted."""
    if not isinstance(payload, dict):
        return ["it is not a JSON object"]
    p = _header_problems(payload, RAW_UNIT_SCHEMA, eval_unit_id(name, seed), ctx)
    if payload.get("config") != name:
        p.append(f"it records config {payload.get('config')!r}")
    if not _is_int(payload.get("seed")) or payload.get("seed") != seed:
        p.append(f"it records seed {payload.get('seed')!r}")
    cal = ctx.calibration
    if cal is None:
        return p + ["no calibration is established to bind it to"]
    if payload.get("calibration_sha256") != cal["calibration_sha256"]:
        p.append("it was measured under a different calibration (frozen "
                 "scales / static selection)")
    p += _evidence_problems(payload, payload.get("arrival_offsets_s"),
                            payload.get("requests"), seed, ctx)
    u = payload.get("run_utility_v2")
    if not _finite(u):
        p.append("run_utility_v2 is not finite")
    elif all(_finite(payload.get(k)) for k in _METRIC_KEYS):
        again = fh.run_utility({"throughput_tps": payload["throughput_tps"],
                                "ttft_p99": payload["ttft_p99_ms"],
                                "tpot_p99": payload["tpot_p99_ms"]}, cal["scales"])
        if abs(again - u) > 1e-12 * max(1.0, abs(u)):
            p.append("run_utility_v2 does not reproduce under the frozen scales")
    if name in _STATIC_CONFIGS:
        if _canonical(payload.get("applied_config")) != _canonical(
                _static_cfg_dict(name, cal)):
            p.append("its applied configuration is not the selected static "
                     "baseline")
    else:
        p += _controller_problems(payload, name)
    return p


def _unit_body(run: dict) -> dict:
    """The measured content of one evaluation run, as written to disk."""
    return {
        "throughput_tps": run.get("throughput_tps"),
        "ttft_p50_ms": run.get("ttft_p50"), "ttft_p99_ms": run.get("ttft_p99"),
        "tpot_p50_ms": run.get("tpot_p50"), "tpot_p99_ms": run.get("tpot_p99"),
        "slo_rate": run.get("slo_rate"), "wall_s": run.get("wall_s"),
        "run_utility_v2": run.get("run_utility_v2"),
        "applied_config": run.get("applied_config"),
        "requests": run.get("requests"),
        "step_log": run.get("step_log"),
        "shift_t": run.get("shift_t"), "shift_step": run.get("shift_step"),
        "phase1_open_loop": run.get("phase1_open_loop"),
        # RAW arrival timestamps, per seed, so the process is reproducible and
        # auditable rather than summarised away.
        "arrival_offsets_s": run.get("arrivals"),
        "arrival_validation": run.get("arrival_validation"),
        "offered_load": run.get("offered_load"),
        "drain_overrun": run.get("drain_overrun"),
        "phase_boundary": run.get("phase_boundary"),
        "arm_set": run.get("arm_set"),
        "cycles": run.get("cycles"),
        "reward_gate": run.get("reward_gate"),
        "reward_validity": run.get("reward_validity"),
        "n_valid_rewards": run.get("n_valid_rewards"),
        "n_withheld_rewards": run.get("n_withheld_rewards"),
        "mean_valid_cycle_utility": run.get("mean_valid_cycle_utility"),
        "valid_reward_spread": run.get("valid_reward_spread"),
        "n_distinct_valid_rewards": run.get("n_distinct_valid_rewards"),
        "decisions": run.get("decisions"),
        "arm_changes": run.get("arm_changes"),
        "arm_change_rate": run.get("arm_change_rate"),
        "arm_histogram": run.get("arm_histogram"),
        "convergence": run.get("convergence"),
        "controller_stats": run.get("controller_stats"),
        "decision_us": run.get("decision_us"),
    }


def _roundtrip(payload: dict) -> dict:
    """Exactly what a later session would read back from disk."""
    return json.loads(json.dumps(payload, default=str))


def save_invalid_unit(unit_id: str, ctx: UnitContext, problems: list,
                      payload: dict) -> str | None:
    """Keep fresh evidence that failed validation, for diagnosis. Never a
    completed unit; its presence makes --resume refuse."""
    safe = unit_id.replace(":", "__")
    try:
        return atomic_write_json(
            os.path.join(ctx.paths.invalid_dir, f"{safe}.json"),
            {"schema": INVALID_UNIT_SCHEMA, "status": STATUS_INVALID_EVIDENCE,
             "unit_id": unit_id, "problems": problems,
             "fingerprint_sha256": ctx.fingerprint_sha256,
             "written_utc": _now(), "payload": payload})
    except Exception:
        return None


def _accept_fresh(unit_id: str, payload: dict, problems_of, path: str,
                  ctx: UnitContext) -> dict:
    """Validate fresh evidence with the RESUME rules, on its round-tripped
    form, before it may count as complete; then write it. Invalid evidence
    stops the run and is preserved -- it is never silently regenerated."""
    trusted = _roundtrip(payload)
    problems = problems_of(trusted)
    if problems:
        where = save_invalid_unit(unit_id, ctx, problems, trusted)
        err = InvalidFreshEvidenceError(
            f"{unit_id}: freshly measured evidence fails the validation a resume "
            "would apply: " + "; ".join(problems) + ". The run stops; the unit "
            "is NOT counted as complete and is NOT re-measured. Diagnostic "
            f"evidence: {_rel(where) if where else 'COULD NOT BE WRITTEN'}.")
        err.unit_id, err.evidence_file, err.problems = unit_id, where, problems
        raise err
    atomic_write_json(path, trusted)
    return trusted


def save_eval_unit(name: str, seed: int, run: dict, ctx: UnitContext) -> dict:
    """Validate, write, and return the run AS A RESUME WOULD RELOAD IT."""
    payload = {
        "schema": RAW_UNIT_SCHEMA, "experiment_version": EXPERIMENT_VERSION,
        "status": "complete", "unit_id": eval_unit_id(name, seed),
        "config": name, "seed": seed, "n_requests": ctx.n,
        "fingerprint_sha256": ctx.fingerprint_sha256,
        "calibration_sha256": ctx.calibration["calibration_sha256"],
        "written_utc": _now(),
        **_unit_body(run),
    }
    trusted = _accept_fresh(
        eval_unit_id(name, seed), payload,
        lambda p: validate_eval_payload(p, name, seed, ctx),
        ctx.paths.eval_raw(name, seed), ctx)
    return _run_from_payload(trusted, name, resumed=False)


def save_sweep_unit(mb: int, cs: int, row: dict, measured: dict,
                    ctx: UnitContext) -> dict:
    """Validate, write, and return the row AS A RESUME WOULD RELOAD IT."""
    payload = {
        "schema": SWEEP_UNIT_SCHEMA, "experiment_version": EXPERIMENT_VERSION,
        "status": "complete", "unit_id": sweep_unit_id(mb, cs),
        "seed": VALIDATION_SEED, "n_requests": ctx.n,
        "max_batch_size": mb, "chunk_size": cs,
        "fingerprint_sha256": ctx.fingerprint_sha256,
        "written_utc": _now(),
        "row": row,
        "arrival_offsets_s": measured.get("arrivals"),
        "requests": measured.get("requests"),
    }
    trusted = _accept_fresh(
        sweep_unit_id(mb, cs), payload,
        lambda p: validate_sweep_payload(p, mb, cs, ctx),
        ctx.paths.sweep_raw(mb, cs), ctx)
    return trusted["row"]


def save_terminal_marker(label: str, paths: RunPaths, *, gate: str,
                         message: str, fingerprint_sha256: str | None) -> str:
    """The SMALL record of a fired gate, written before the large evidence
    file so the verdict survives even if that write fails."""
    safe = label.replace(":", "__")
    return atomic_write_json(
        os.path.join(paths.failed_dir, f"{safe}.terminal_marker.json"),
        {"schema": TERMINAL_MARKER_SCHEMA, "status": STATUS_TERMINAL_GATE,
         "unit_id": label, "gate": gate, "message": message[:4000],
         "fingerprint_sha256": fingerprint_sha256, "written_utc": _now()})


def save_failed_unit(label: str, ctx: UnitContext, *, gate: str, message: str,
                     evidence: dict) -> str:
    """Terminal gate failures keep their evidence; nothing is discarded."""
    payload = {
        "schema": FAILED_UNIT_SCHEMA, "experiment_version": EXPERIMENT_VERSION,
        "status": STATUS_TERMINAL_GATE, "unit_id": label, "gate": gate,
        "message": message, "n_requests": ctx.n,
        "fingerprint_sha256": ctx.fingerprint_sha256,
        "calibration_sha256": (ctx.calibration or {}).get("calibration_sha256"),
        "written_utc": _now(),
        **evidence,
    }
    safe = label.replace(":", "__")
    return atomic_write_json(os.path.join(ctx.paths.failed_dir, f"{safe}.json"),
                             payload)


def record_terminal_failure(results: dict, session: dict, ctx: UnitContext,
                            exc: BaseException, *, label: str, gate: str,
                            evidence: dict) -> dict:
    """A fired gate, recorded so that no later failure can disguise it.

    Order: (1) in memory, in the results AND the session, so the top-level
    handler classifies the run as terminal whatever exception finally
    propagates; (2) a small marker file; (3) the full evidence. A failure in
    (2) or (3) -- an I/O error, an interrupt -- is recorded next to the
    verdict and swallowed, so the CALLER re-raises the original scientific
    exception. If every write fails, the verdict lives only in memory and in
    whatever checkpoint write still succeeds; that limitation is stated in the
    record rather than hidden.
    """
    tf = {"status": STATUS_TERMINAL_GATE, "gate": gate, "unit": label,
          "message": str(exc)[:4000],
          "resume_policy": "terminal: --resume refuses this run",
          "marker_file": None, "evidence_file": None, "write_errors": []}
    results["terminal_failure"] = tf
    session["terminal_failure"] = tf
    try:
        tf["marker_file"] = _rel(save_terminal_marker(
            label, ctx.paths, gate=gate, message=str(exc),
            fingerprint_sha256=ctx.fingerprint_sha256))
    except BaseException as werr:          # noqa: BLE001 -- recorded, not lost
        tf["write_errors"].append(f"terminal marker: {werr!r}")
    try:
        tf["evidence_file"] = _rel(save_failed_unit(
            label, ctx, gate=gate, message=str(exc), evidence=evidence))
    except BaseException as werr:          # noqa: BLE001 -- recorded, not lost
        tf["write_errors"].append(f"failed-unit evidence: {werr!r}")
    tf["persisted_on_disk"] = bool(tf["marker_file"] or tf["evidence_file"])
    if not tf["persisted_on_disk"]:
        tf["durability_limitation"] = (
            "neither the terminal marker nor the evidence could be written. "
            "The verdict survives only if the checkpoint write succeeds; if "
            "that fails too, a later --resume cannot know this gate fired. "
            "Do NOT resume this run.")
    return tf


def _run_from_payload(payload: dict, name: str, *, resumed: bool = True) -> dict:
    """Back to the in-memory shape `run_one` returns, so every downstream
    consumer is identical whether the unit was measured or resumed."""
    run = {
        "throughput_tps": payload["throughput_tps"],
        "ttft_p50": payload["ttft_p50_ms"], "ttft_p99": payload["ttft_p99_ms"],
        "tpot_p50": payload["tpot_p50_ms"], "tpot_p99": payload["tpot_p99_ms"],
        "slo_rate": payload["slo_rate"], "wall_s": payload["wall_s"],
        "run_utility_v2": payload["run_utility_v2"],
        "requests": payload["requests"],
        "arrivals": payload["arrival_offsets_s"],
        "arrival_validation": payload["arrival_validation"],
        "offered_load": payload["offered_load"],
        "drain_overrun": payload["drain_overrun"],
        "phase_boundary": payload["phase_boundary"],
        "applied_config": payload.get("applied_config"),
        "step_log": payload.get("step_log"),
        "resumed_from_checkpoint": resumed,
    }
    if name in _CONTROLLER_CONFIGS:
        for k in ("cycles", "reward_validity", "convergence", "decisions",
                  "arm_changes", "arm_change_rate", "arm_histogram",
                  "mean_valid_cycle_utility", "valid_reward_spread",
                  "n_distinct_valid_rewards", "reward_gate", "n_valid_rewards",
                  "n_withheld_rewards", "controller_stats", "arm_set"):
            run[k] = payload.get(k)
    return run


def load_eval_unit(name: str, seed: int, ctx: UnitContext) -> tuple:
    """(run, None) for trusted evidence, else (None, why). Never raises."""
    payload, why = _read_json(ctx.paths.eval_raw(name, seed))
    if payload is None:
        return None, why
    problems = validate_eval_payload(payload, name, seed, ctx)
    if problems:
        return None, "its evidence is not trusted: " + "; ".join(problems)
    return _run_from_payload(payload, name), None


# --- the calibration and its digest ----------------------------------------


def _scales_dict(scales) -> dict:
    return {"t_half": scales.t_half, "ttft_target": scales.ttft_target,
            "tpot_target": scales.tpot_target, "sharpness": scales.sharpness}


def compute_calibration(rows: list, n: int, fingerprint_sha256: str) -> dict:
    """MEASUREMENTS -> frozen scales -> pre-run gate -> both argmaxes -> digest.

    The order the saturated run used, with the same imported functions. The
    digest binds the full held-out measurements, the frozen scales, the reward
    weights and both selected configurations; every evaluation unit records it,
    so a unit measured under one calibration can never be reused under another.
    """
    scales, record = fh.freeze_scales(rows, n, VALIDATION_SEED)
    try:
        record["gate"] = gate_sweep(rows, scales)
    except DegenerateRewardError as exc:
        exc.frozen_scales = _scales_dict(scales)
        raise
    cfg_tput, cfg_utility, selection = fh.select_static_baselines(rows, scales)
    binding = {
        "schema": CALIBRATION_SCHEMA,
        "fingerprint_sha256": fingerprint_sha256,
        "held_out_seed": VALIDATION_SEED,
        "held_out_measurements_sha256": _sha256_text(_canonical(rows)),
        "frozen_scales": _scales_dict(scales),
        "reward_weights": dict(DEFAULT_WEIGHTS),
        "static_tput": cfg_tput.as_dict(),
        "static_utility": cfg_utility.as_dict(),
    }
    return {
        "scales": scales, "record": record, "selection": selection,
        "cfg_tput": cfg_tput, "cfg_utility": cfg_utility, "binding": binding,
        "calibration_sha256": _sha256_text(_canonical(binding)),
    }


def save_calibration(paths: RunPaths, cal: dict, rows: list,
                     *, recovered_from: str | None = None) -> str:
    return atomic_write_json(paths.calibration, {
        "schema": CALIBRATION_SCHEMA, "experiment_version": EXPERIMENT_VERSION,
        "calibration_sha256": cal["calibration_sha256"],
        "binding": cal["binding"],
        "held_out_rows": rows,
        "recovered_from": recovered_from,
        "written_utc": _now(),
        "role": ("durable record of the calibration every evaluation unit is "
                 "bound to; a resume restores the held-out rows from here and "
                 "must reproduce calibration_sha256 exactly"),
    })


def load_calibration(paths: RunPaths, fingerprint_sha256: str,
                     grid: list) -> tuple:
    """(rows, stored_digest, None) for a self-consistent record, else
    (None, None, why). Never raises; the caller decides what absence means."""
    payload, why = _read_json(paths.calibration)
    if payload is None:
        return None, None, why
    if not isinstance(payload, dict) or payload.get("schema") != CALIBRATION_SCHEMA:
        return None, None, "the calibration record has the wrong schema"
    binding, rows = payload.get("binding"), payload.get("held_out_rows")
    if not isinstance(binding, dict) or not isinstance(rows, list):
        return None, None, "the calibration record is incomplete"
    if _sha256_text(_canonical(binding)) != payload.get("calibration_sha256"):
        return None, None, "the calibration record's digest does not match it"
    if binding.get("fingerprint_sha256") != fingerprint_sha256:
        return None, None, "the calibration record belongs to another fingerprint"
    if _sha256_text(_canonical(rows)) != binding.get("held_out_measurements_sha256"):
        return None, None, "the calibration record's rows do not match its digest"
    if [(r.get("max_batch_size"), r.get("chunk_size")) for r in rows
            if isinstance(r, dict)] != list(grid):
        return None, None, "the calibration record does not cover the grid in order"
    return rows, payload["calibration_sha256"], None


def _or_pending(value, why: str):
    """`value`, or an explicit record of why it is absent -- never a bare null."""
    if value:
        return value
    return {"recorded": False, "why": why}


def _provenance_extra(seeds: list, n: int, results: dict) -> dict:
    """No silent nulls. Every required field is either a value or a statement."""
    plan = results.get("arrival_plan") or {}
    phases = plan.get("phases") or {}
    return {
        "model": MODEL_NAME,
        "dtype": str(torch.float16 if DEVICE.type == "cuda" else torch.float32),
        "device": str(DEVICE),
        "run_mode": results.get("run_mode"),
        "seeds": seeds,
        "requests": n,
        "arrival_mode": "poisson (open-loop, phase-specific lambda)",
        "workload_version": provenance.WORKLOAD_VERSION,
        "target_rho": plan.get("target_rho"),
        "execution_arm": plan.get("execution_arm"),
        "service_rate_source_artifact": plan.get("source_artifact"),
        "service_rate_rows_used": plan.get("rows_used"),
        "mu_interactive_req_per_s": (phases.get("interactive") or {}).get(
            "mu_max_req_per_s"),
        "mu_batch_req_per_s": (phases.get("batch") or {}).get("mu_max_req_per_s"),
        "lambda_interactive_rps": (phases.get("interactive") or {}).get(
            "lambda_rps"),
        "lambda_batch_rps": (phases.get("batch") or {}).get("lambda_rps"),
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
            "method": "exhaustive max_batch_size x chunk_size, re-run for THIS "
                      "operating point",
            "max_batch_size": GRID_BATCH_AXIS,
            "chunk_size": GRID_CHUNK_AXIS,
            "n_configurations": len(GRID_BATCH_AXIS) * len(GRID_CHUNK_AXIS),
            "validation_seed": VALIDATION_SEED,
            "inert_dimensions_excluded": sorted(INERT_DIMENSIONS),
            "reused_from_saturated_run": False,
        },
        "reward_version": provenance.REWARD_VERSION,
        # Explicit absence, never a bare null: on the checkpoints written before
        # the held-out sweep finishes there ARE no frozen scales yet, and a null
        # there is indistinguishable from "we forgot to record them".
        "frozen_reward_scales": _or_pending(
            results.get("reward", {}).get("scales"),
            "not yet frozen: this checkpoint predates the held-out sweep, and "
            "the scales are medians of that sweep's measurements"),
        "reward_weights": _or_pending(
            results.get("reward", {}).get("weights"),
            "not yet recorded: frozen together with the scales"),
        "cold_start_policy": _or_pending(
            results.get("reward", {}).get("cold_start_policy"),
            "not yet recorded: LiveRewardV2 is constructed after the scales are "
            "frozen; require_valid_metrics=True is fixed in run_all"),
        "calibration_sha256": _or_pending(
            (results.get("calibration") or {}).get("calibration_sha256"),
            "not yet computed: the calibration is fixed after the held-out "
            "sweep completes"),
        "convergence_rule": CONVERGENCE_RULE,
        "convergence_rule_source": ("scripts/eval/repair/final_hardening.py::"
                                    "CONVERGENCE_RULE (imported, not copied)"),
        "experiment_version": EXPERIMENT_VERSION,
        "fingerprint": _or_pending(
            results.get("fingerprint"),
            "not yet computed: this checkpoint predates the arrival "
            "derivation the fingerprint is built from"),
        "resume_policy": (
            "--resume is an EXECUTION feature only. It decides which units are "
            "re-measured and never any scientific quantity. It aborts unless "
            "the stored fingerprint -- including the digest RECOMPUTED over the "
            "executing source -- matches on every key, refuses after a "
            "reward-gate failure, and re-uses an evaluation unit only if its "
            "evidence validates and is bound to the same calibration digest."),
        "source_verification": _or_pending(
            results.get("source_verification"),
            "not yet recorded: verified at start-up"),
        # THE SIDECAR: local git HEAD, dirty state, captured git status, diff
        # SHA256, source manifest, archive SHA256 and archive member manifest.
        # Absence is RECORDED, never emitted as a bare null.
        "source_provenance": fh.load_source_provenance(),
    }


def write_artifact(results: dict, seeds: list, n: int,
                   paths: RunPaths | None = None) -> str:
    """Stamp with provenance and write ATOMICALLY to this run's own path."""
    paths = paths or RunPaths(PUBLISHED_OUTPUT_DIR)
    provenance.stamp(results, EXPERIMENT,
                     script="scripts/eval/repair/subcritical_poisson.py",
                     extra=_provenance_extra(seeds, n, results))
    return atomic_write_json(paths.results, results)
# ===========================================================================
# 7. Driver.
# ===========================================================================


class PreflightRefusal(RuntimeError):
    """Start-up refused before any evidence was read-modified-written."""


def _dtype():
    return torch.float16 if DEVICE.type == "cuda" else torch.float32


def load_model_and_tokenizer(dtype):
    """The ONE expensive initialisation. Nothing before it writes anything."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()
    return tokenizer, model


# --- engine warm-up ----------------------------------------------------------
#
# WHY. The 2026-10 T4 smoke charged the process's one-off start-up to whichever
# unit ran first: sweep point mb4_cs128 saw its first token at 8.10 s and a
# 7.91 s submission lag, every later unit in the same process ~0.07 s / ~0.05 s.
# The loaded model had never executed a forward pass; the liveness probe reads
# attributes only; and on CUDA the attention path first JIT-compiles its Triton
# kernels (fused RoPE, FA2) and loads cuBLAS kernels lazily. Which of these the
# 8 s was is NOT measured. Because `_serve` releases arrivals from the same
# synchronous loop that runs `sched.step()`, a stalled first step also delays
# every submission behind it.
#
# WHAT. A fixed, declared sequence of disposable schedulers driven through the
# SAME `_new_scheduler` / `_apply_sched` / `add_request` / `step` path the
# measured units use, on the workload's own prompt construction. No controller,
# no tracker, no `_serve`, no arrival clock. It runs once per process, before
# the first measured unit, and only when some unit will actually be measured.
#
# NOT AN ENGINE OF THE EXPERIMENT. Nothing it produces -- time, tokens,
# requests -- reaches a measured metric, an arrival timestamp, the reward
# calibration or the convergence rule. It is unrelated to the controller's
# reward-validity cold start (`reward_validity.warmup_cycles`), which is a
# property of each measured run and is left exactly as it was.

# Prefill shapes: every 16-token boundary of the workload's prompt range
# (16..256) and the length one past it, so prefill widths that are and are not
# multiples of 16 both execute. Triton specialises integer kernel arguments on
# divisibility by 16 (and on == 1); cuBLAS selects kernels by GEMM shape.
WARMUP_PREFILL_LENGTHS = tuple(sorted({v for L in range(16, 257, 16)
                                       for v in (L, L + 1) if v <= 256}))
WARMUP_PREFILL_MAX_NEW = 2       # the prefill-sampled token + one decode step
# Batched decode: a 30-token prompt and 4 new tokens walk the decode context
# through 31..34 (via 32, a multiple of 16) at every batch size.
WARMUP_DECODE_PROMPT_LEN = 30
WARMUP_DECODE_MAX_NEW = 4


class EngineWarmupError(RuntimeError):
    """The declared warm-up did not complete; no measured unit may run."""

    def __init__(self, message: str, record: dict | None = None):
        super().__init__(message)
        self.warmup_record = record


def _warmup_shapes(batches: list, chunks: list) -> tuple:
    """(chunk sizes, largest batch) any measured unit can execute: the sweep
    grid AND every arm the controllers can select, after the same clamp."""
    cfgs = [CARLConfig(max_batch_size=mb, chunk_size=cs).clamp()
            for mb in batches for cs in chunks]
    for arm_sets in (abl.all_arm_sets(), abl.expanded_arm_sets()):
        for arms in arm_sets.values():
            cfgs += [a.clamp() for a in arms]
    return (sorted({c.chunk_size for c in cfgs}),
            max(c.max_batch_size for c in cfgs))


def _stage_step_bound(prompt_lengths: list, max_new: int, chunk: int,
                      batch: int) -> int:
    """A hard upper bound on the steps one stage may take. Decode tokens can
    shrink a step's prefill budget to `chunk - batch`; every step makes
    progress, so the sequential sum bounds the batched execution."""
    eff = max(1, chunk - batch)
    return sum(-(-L // eff) + max_new for L in prompt_lengths) + len(prompt_lengths)


def engine_warmup_spec(batches: list, chunks: list) -> dict:
    """The DECLARED warm-up: a fingerprint key, so it contains nothing measured.

    A function of the grid and the arm space only -- never of an observation,
    a treatment ranking or a reward.
    """
    chunk_set, max_batch = _warmup_shapes(batches, chunks)
    top = max(chunk_set)
    longest = max(WARMUP_PREFILL_LENGTHS)
    stages = [{"stage": "prefill", "max_batch_size": 1, "chunk_size": top,
               "prompt_lengths": list(WARMUP_PREFILL_LENGTHS),
               "max_new_tokens": WARMUP_PREFILL_MAX_NEW}]
    for cs in chunk_set:
        # Continuation chunks (the masked, position-offset prefill path) exist
        # only for chunk sizes below the longest workload prompt.
        lengths = sorted({L for L in (cs + 1, cs + 17, longest) if cs < L <= longest})
        if lengths:
            stages.append({"stage": "chunked_prefill", "max_batch_size": 1,
                           "chunk_size": cs, "prompt_lengths": lengths,
                           "max_new_tokens": WARMUP_PREFILL_MAX_NEW})
    for b in range(1, max_batch + 1):
        stages.append({"stage": "batched_decode", "max_batch_size": b,
                       "chunk_size": top,
                       "prompt_lengths": [WARMUP_DECODE_PROMPT_LEN] * b,
                       "max_new_tokens": WARMUP_DECODE_MAX_NEW})
    for s in stages:
        s["step_bound"] = _stage_step_bound(s["prompt_lengths"],
                                            s["max_new_tokens"],
                                            s["chunk_size"], s["max_batch_size"])
    return {
        "version": ENGINE_WARMUP_VERSION,
        "when": ("once per process, after the resume identity / evidence checks "
                 "and BEFORE the run takes ownership of the checkpoint; only if "
                 "at least one measured unit will execute"),
        "inputs": ("src.carl.live._make_prompt(tokenizer, L): the workload's own "
                   "tiled sentence, so prefix-cache hits occur as in measured "
                   "units. Deterministic; no RNG draw; eos_token_id=None so "
                   "every request emits exactly max_new_tokens"),
        "execution_path": ("ablation_live._new_scheduler + _apply_sched + "
                           "add_request + step(): the eager prefill, chunked "
                           "(position-offset, masked) prefill, prefix-cache "
                           "sliced prefill and batched-decode forwards of "
                           "ContinuousBatchScheduler. No CARLController, no "
                           "MetricsTracker, no _serve, no arrival clock"),
        "chunk_sizes_covered": chunk_set,
        "max_batch_size_covered": max_batch,
        "stages": stages,
        "step_bound_total": sum(s["step_bound"] for s in stages),
        "termination": ("a fixed request list per stage, run to drain; a stage "
                        "that exceeds its step bound or does not emit exactly "
                        "its declared tokens aborts the run. Never repeated "
                        "and never conditioned on how fast it ran"),
        "synchronization": ("device synchronize before the warm-up clock starts "
                            "and after the last stage, before any measured unit"),
        "state_isolation": ("a new scheduler / KV pool per stage, drained and "
                            "discarded; Python, NumPy, torch CPU and CUDA RNG "
                            "states restored afterwards; the CUDA caching "
                            "allocator is NOT emptied (as between measured units)"),
        "coverage_limits": (
            "not every (prefill width, context length) the measured units can "
            "produce is executed: a step's prefill budget is chunk_size minus "
            "the decode count, so widths are arbitrary. Kernel variants or "
            "cuBLAS kernels first reached by an uncovered shape can still "
            "cost time inside a measured unit; that residue is not measured"),
    }


def _cuda_synchronize(device) -> None:
    """Wait for queued device work. A no-op off CUDA (the clock is then honest)."""
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _rng_snapshot() -> dict:
    snap = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    try:
        import numpy as np
        snap["numpy"] = np.random.get_state()
    except ImportError:          # pragma: no cover - numpy ships with torch here
        pass
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap: dict) -> None:
    random.setstate(snap["python"])
    torch.set_rng_state(snap["torch_cpu"])
    if "numpy" in snap:
        import numpy as np
        np.random.set_state(snap["numpy"])
    if "torch_cuda" in snap:
        torch.cuda.set_rng_state_all(snap["torch_cuda"])


def _rng_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b) or a["python"] != b["python"]:
        return False
    if not torch.equal(a["torch_cpu"], b["torch_cpu"]):
        return False
    if "numpy" in a:
        na, nb = a["numpy"], b["numpy"]
        if na[0] != nb[0] or list(na[1]) != list(nb[1]) or na[2:] != nb[2:]:
            return False
    if "torch_cuda" in a:
        if len(a["torch_cuda"]) != len(b["torch_cuda"]) or not all(
                torch.equal(x, y) for x, y in zip(a["torch_cuda"], b["torch_cuda"])):
            return False
    return True


def _forward_kind(args, kwargs) -> tuple | None:
    """(path, batch, width, position-offset>0) of one model forward."""
    ids = args[0] if args else kwargs.get("input_ids")
    cache = kwargs.get("kv_cache", args[1] if len(args) > 1 else None)
    shape = tuple(getattr(ids, "shape", ()))
    if len(shape) != 2:
        return None
    if isinstance(cache, list):
        return ("batched_decode", shape[0], shape[1], True)
    try:
        offset = int(cache.seq_len(0)) if cache is not None else 0
    except Exception:
        offset = None
    path = "prefill" if shape[1] > 1 else "single_token"
    return (path, shape[0], shape[1], bool(offset) if offset is not None else None)


def run_engine_warmup(model, tokenizer, spec: dict, device=None) -> dict:
    """Execute the declared warm-up once. Returns its audit record.

    Raises EngineWarmupError (carrying the partial record) if any stage fails
    to complete exactly as declared. RNG states are restored and the forward
    hook is removed whether it completes, fails or is interrupted.
    """
    device = device if device is not None else (_model_device(model) or DEVICE)
    record = {"version": spec["version"], "status": "running",
              "inference_performed": True, "device": str(device),
              "stages": [], "synchronized_before": False,
              "synchronized_after": False}
    seen: dict = {}

    def _hook(_mod, args, kwargs):
        kind = _forward_kind(args, kwargs)
        if kind is not None:
            seen[kind] = seen.get(kind, 0) + 1

    handle = None
    register = getattr(model, "register_forward_pre_hook", None)
    rng_before = _rng_snapshot()
    try:
        if register is not None:
            handle = register(_hook, with_kwargs=True)
        _cuda_synchronize(device)
        record["synchronized_before"] = True
        t_start = time.perf_counter()
        for i, st in enumerate(spec["stages"]):
            t0 = time.perf_counter()
            sched = abl._new_scheduler(model)
            abl._apply_sched(sched, CARLConfig(max_batch_size=st["max_batch_size"],
                                               chunk_size=st["chunk_size"]).clamp())
            emitted: dict = {}
            for j, L in enumerate(st["prompt_lengths"]):
                rid = f"warmup{i}_{j}"
                emitted[rid] = 0
                sched.add_request(rid, _make_prompt(tokenizer, L),
                                  max_new_tokens=st["max_new_tokens"],
                                  eos_token_id=None)
            steps = 0
            while sched.has_work():
                if steps >= st["step_bound"]:
                    raise EngineWarmupError(
                        f"warm-up stage {i} ({st['stage']}) exceeded its "
                        f"declared bound of {st['step_bound']} steps", record)
                for rid, _tok in sched.step():
                    emitted[rid] = emitted.get(rid, 0) + 1
                steps += 1
            sched.get_finished()
            short = {r: c for r, c in emitted.items() if c != st["max_new_tokens"]}
            if short or set(emitted) != {f"warmup{i}_{j}" for j in
                                         range(len(st["prompt_lengths"]))}:
                raise EngineWarmupError(
                    f"warm-up stage {i} ({st['stage']}) did not emit exactly "
                    f"{st['max_new_tokens']} tokens per request: {short}", record)
            del sched
            record["stages"].append({
                "stage": st["stage"], "max_batch_size": st["max_batch_size"],
                "chunk_size": st["chunk_size"],
                "requests": len(st["prompt_lengths"]),
                "tokens_emitted": sum(emitted.values()), "steps": steps,
                "elapsed_s": time.perf_counter() - t0})
        _cuda_synchronize(device)
        record["synchronized_after"] = True
        record["elapsed_s"] = time.perf_counter() - t_start
        record["status"] = "complete"
    except EngineWarmupError:
        record["status"] = "failed"
        raise
    except Exception as exc:
        record["status"] = "failed"
        raise EngineWarmupError(
            f"engine warm-up failed: {type(exc).__name__}: {exc}", record) from exc
    except BaseException as exc:
        record["status"] = "interrupted"
        try:
            exc.warmup_record = record
        except Exception:        # pragma: no cover - exotic exception types
            pass
        raise
    finally:
        if handle is not None:
            handle.remove()
        _rng_restore(rng_before)
        record["rng_state_restored"] = _rng_equal(rng_before, _rng_snapshot())
        record["forward_shapes_observed"] = (
            [{"path": k[0], "batch": k[1], "width": k[2],
              "position_offset": k[3], "count": v}
             for k, v in sorted(seen.items(), key=lambda kv: str(kv[0]))]
            if register is not None else
            "not observable: the model exposes no forward hook")
        record["steps"] = sum(s["steps"] for s in record["stages"])
        record["tokens_emitted"] = sum(s["tokens_emitted"] for s in record["stages"])
        record["requests"] = sum(s["requests"] for s in record["stages"])
    if not record["rng_state_restored"]:
        record["status"] = "failed"
        raise EngineWarmupError("RNG state could not be restored after warm-up",
                                record)
    return record


def measured_work_pending(ledger, restore: dict, ctx, batches: list,
                          chunks: list, seeds: list) -> dict:
    """Which units the run below WILL measure, decided by the same rules it
    applies -- read-only. Used only to decide whether the warm-up is needed."""
    if restore["calibration"] is None:
        # No calibration yet: no evaluation can exist, so every evaluation unit
        # and any sweep point without verified evidence will be measured.
        sweep = [sweep_unit_id(mb, cs) for mb in batches for cs in chunks
                 if f"mb{mb}_cs{cs}" not in restore["rows_by_key"]]
        evals = [eval_unit_id(n, s) for n in CONFIGS for s in seeds]
    else:
        probe = UnitContext(ctx.paths, ctx.fingerprint_sha256, ctx.n, ctx.plan,
                            calibration=restore["calibration"])
        sweep = []
        evals = [eval_unit_id(n, s) for n in CONFIGS for s in seeds
                 if not (ledger.claimed(eval_unit_id(n, s))
                         and load_eval_unit(n, s, probe)[0] is not None)]
    return {"pending": bool(sweep or evals), "sweep_units": sweep,
            "eval_units": evals}


def registered_deviations(seeds: list, n: int, batches: list, chunks: list,
                          rho: float, mu_artifact: str) -> list:
    """How this invocation differs from the pre-registered run, if at all."""
    out = []
    if list(seeds) != DEFAULT_SEEDS:
        out.append(f"seeds {seeds} != {DEFAULT_SEEDS}")
    if n != DEFAULT_REQUESTS:
        out.append(f"--limit {n} != {DEFAULT_REQUESTS}")
    if list(batches) != GRID_BATCH_AXIS:
        out.append(f"--grid-batches {batches} != {GRID_BATCH_AXIS}")
    if list(chunks) != GRID_CHUNK_AXIS:
        out.append(f"--grid-chunks {chunks} != {GRID_CHUNK_AXIS}")
    if rho != TARGET_RHO:
        out.append(f"--rho {rho} != {TARGET_RHO}")
    if os.path.normcase(os.path.abspath(_abs_artifact(mu_artifact))) != \
            os.path.normcase(os.path.abspath(_abs_artifact(MU_SOURCE_ARTIFACT))):
        out.append(f"--mu-artifact {mu_artifact} != {MU_SOURCE_ARTIFACT}")
    return out


def _resolved(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _same_location(a: str, b: str) -> bool:
    """True if `a` and `b` name the same place, through symlinks, junctions,
    `..`, case and (for existing paths) any other alias samefile can see."""
    if _resolved(a) == _resolved(b):
        return True
    try:
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return False


_ISOLATED_PATHS = ("output_dir", "results", "raw_dir", "sweep_dir", "failed_dir",
                   "invalid_dir", "superseded_dir", "calibration")


def published_aliases(paths: RunPaths) -> list:
    """Which of `paths` resolve onto the PUBLISHED run's own locations."""
    pub = RunPaths(PUBLISHED_OUTPUT_DIR)
    return [k for k in _ISOLATED_PATHS
            if _same_location(getattr(paths, k), getattr(pub, k))]


def preflight(*, mode: str, allow_cpu: bool, seeds: list, n: int,
              batches: list, chunks: list, rho: float, mu_artifact: str,
              resume: bool, paths: RunPaths,
              verify_archive: str | None = None) -> dict:
    """Every refusal that needs no model, in the order that costs least.

    Reads only. SystemExit for an invocation that should never have been made
    (wrong device or GPU, a 'published' run that is not the registered one, a
    destination that would mix smoke and published output, a fresh run over
    existing evidence); PreflightRefusal / ResumeMismatchError for state that
    cannot be trusted. None of them writes to the checkpoint.
    """
    if mode not in (RUN_MODE_PUBLISHED, RUN_MODE_SMOKE):
        raise SystemExit(f"unknown run mode {mode!r}")
    if DEVICE.type != "cuda" and not (mode == RUN_MODE_SMOKE and allow_cpu):
        raise SystemExit(
            "subcritical_poisson REFUSES to run without CUDA. The operating "
            "point is derived from a T4-calibrated service rate and a CPU "
            "number would sit at an entirely different rho. Pass "
            "--allow-cpu-smoke for a wiring smoke test (never a result).")
    runtime = runtime_identity(DEVICE)
    gpu_name = (runtime.get("gpu") or {}).get("name")
    if mode == RUN_MODE_PUBLISHED and gpu_name != REQUIRED_GPU_NAME:
        raise SystemExit(
            f"a PUBLISHED run must execute on a {REQUIRED_GPU_NAME}: mu, and "
            f"therefore lambda and rho, are T4 measurements. This device is "
            f"{gpu_name!r}. Use --smoke on other hardware (recorded, never a "
            "result).")

    if mode == RUN_MODE_PUBLISHED:
        dev = registered_deviations(seeds, n, batches, chunks, rho, mu_artifact)
        if dev:
            raise SystemExit(
                "a PUBLISHED run must be the pre-registered run, and this one "
                "deviates: " + "; ".join(dev) + ". Use --smoke for anything "
                "else; CUDA alone does not make a run a result.")
        if not _same_location(paths.output_dir, PUBLISHED_OUTPUT_DIR):
            raise SystemExit(
                f"a published run writes only to {PUBLISHED_OUTPUT_DIR}; "
                f"--output-dir {paths.output_dir} is for --smoke runs")
    else:
        hits = published_aliases(paths)
        if hits:
            raise SystemExit(
                "a smoke run may not write to the published destination "
                f"{PUBLISHED_OUTPUT_DIR}: {hits} resolve onto published "
                "locations (directly or through a symlink / junction / alias); "
                "choose another --output-dir")

    if not resume:
        existing = ([paths.results] if os.path.isfile(paths.results) else []) \
            + paths.evidence_files()
        if existing:
            raise SystemExit(
                "REFUSING to start a fresh run over existing evidence: "
                + ", ".join(_rel(p) for p in existing[:5])
                + (" ..." if len(existing) > 5 else "")
                + ". Use --resume to continue it, or move it aside. Nothing "
                "was modified.")

    sidecar = fh.load_source_provenance()
    source_check = check_source(sidecar, archive_path=verify_archive)
    if mode == RUN_MODE_PUBLISHED and not source_check["verified"]:
        raise PreflightRefusal(
            "the executing source could not be verified against the packaged "
            "archive manifest: " + "; ".join(source_check["problems"]) +
            ". Mismatched: {}. Missing: {}. Unexpected: {}. Rebuild the "
            "archive and capture its sidecar locally with "
            "`source_provenance.py --archive <zip>`; do not regenerate it on "
            "Colab.".format(source_check["mismatched"][:10],
                            source_check["missing"][:10],
                            source_check["unexpected"][:10]))

    # The frozen saturated side of H1/H2. A published run cannot resolve its
    # hypotheses without it, so it refuses here, before any GPU time; a smoke
    # records the failure and leaves H1/H2 unresolved.
    saturated_ref = load_saturated_reference()
    if saturated_ref["status"] != "verified":
        if mode == RUN_MODE_PUBLISHED:
            raise PreflightRefusal(
                "the frozen saturated reference failed verification: "
                + "; ".join(saturated_ref.get("problems") or ["unknown"])
                + ". H1/H2 cannot be resolved against it, and nothing falls "
                "back to the rounded transcription.")
        print("[reference] saturated reference UNVERIFIED -- H1/H2 will be "
              "unresolved: " + "; ".join(saturated_ref.get("problems") or []),
              flush=True)

    prior = None
    if resume:
        prior = read_resume_checkpoint(paths)
        arm = prior["fingerprint"].get("execution_arm")
        if arm not in ("eager", "graph"):
            raise ResumeMismatchError(
                f"the checkpoint records execution arm {arm!r}; nothing to "
                "resume against")
        # Every fingerprint key except the arm and the model identity is
        # knowable without the model -- including the GPU and library stack --
        # so this comparison runs HERE, before any GPU time; both are
        # re-checked, with every other key, once the model is loaded.
        plan = derive_service_rates(arm, artifact_path=mu_artifact, rho=rho)
        expected = experiment_fingerprint(
            plan, seeds, n, batches, chunks, model=MODEL_NAME,
            dtype=str(_dtype()), source_check=source_check, run_mode=mode,
            runtime=runtime, saturated_ref=saturated_ref)
        diffs = compare_fingerprints(prior["fingerprint"], expected,
                                     ignore=("model_identity",))
        if diffs:
            raise ResumeMismatchError(
                "--resume REFUSED: the checkpoint belongs to a different "
                "experiment. Differing keys:\n  - " + "\n  - ".join(diffs) +
                "\nNothing was modified; start a fresh run instead.")
    return {"sidecar": sidecar, "source_check": source_check, "prior": prior,
            "runtime": runtime, "saturated_reference": saturated_ref}


def plan_restore(ledger: ResumeLedger, ctx: UnitContext, batches: list,
                 chunks: list, *, checkpoint_calibration_sha256: str | None = None
                 ) -> dict:
    """Decide, WITHOUT writing, what the evidence on disk can support.

    Returns the verified sweep rows, the calibration to recover (if any), and
    the file moves to perform once the run owns the checkpoint. Raises
    ResumeMismatchError -- with nothing touched -- when the evidence cannot be
    reconciled without mixing two calibrations.

    `checkpoint_calibration_sha256` is the digest the CHECKPOINT recorded. It
    is a binding digest in its own right: the calibration record, the
    evaluation units and the checkpoint must all agree, and a recovery with no
    binding digest at all is refused rather than accepted as "consistent".
    """
    paths = ctx.paths
    grid = [(mb, cs) for mb in batches for cs in chunks]
    rows_by_key: dict = {}
    moves: list = []         # (unit, path, why) -> superseded/
    marks: list = []         # (unit, why) -> ledger.mark + audit
    calibration = None
    write_calibration = None

    disk = {}
    for mb, cs in grid:
        payload, why = _read_json(paths.sweep_raw(mb, cs))
        if payload is not None:
            problems = validate_sweep_payload(payload, mb, cs, ctx)
            if problems:
                payload, why = None, "its evidence is not trusted: " + "; ".join(problems)
        disk[(mb, cs)] = (payload, why)

    cal_rows, cal_digest, cal_why = load_calibration(
        paths, ctx.fingerprint_sha256, grid)
    eval_files = [f for f in (os.listdir(paths.raw_dir)
                              if os.path.isdir(paths.raw_dir) else [])
                  if f.endswith(".json") and f != os.path.basename(paths.calibration)]
    eval_claims = [u for u in ledger.as_dict()["units_completed"]
                   if u.startswith("eval:")]
    evaluations_started = bool(eval_claims or eval_files or cal_rows is not None
                               or os.path.isfile(paths.calibration))

    if cal_rows is not None:
        # The ORIGINAL calibration is on disk: its rows are authoritative and
        # nothing in the sweep is re-measured.
        for (mb, cs), row in zip(grid, cal_rows):
            unit, key = sweep_unit_id(mb, cs), sweep_unit_id(mb, cs).split(":", 1)[1]
            payload, why = disk[(mb, cs)]
            if payload is not None and _canonical(payload["row"]) != _canonical(row):
                raise ResumeMismatchError(
                    f"{unit}: its sweep evidence disagrees with the calibration "
                    "record the evaluations are bound to. Refusing to choose "
                    "between them; nothing was modified.")
            if payload is None:
                marks.append((unit, f"sweep evidence unavailable ({why}); row "
                                    "restored from the calibration record"))
            elif not ledger.claimed(unit):
                marks.append((unit, "valid sweep evidence bound to the "
                                    "calibration record; checkpoint entry "
                                    "restored"))
            rows_by_key[key] = row
        if (checkpoint_calibration_sha256 is not None
                and checkpoint_calibration_sha256 != cal_digest):
            raise ResumeMismatchError(
                "the checkpoint records calibration digest "
                f"{checkpoint_calibration_sha256} but the calibration record on "
                f"disk is {cal_digest}. Refusing to choose between them; nothing "
                "was modified.")
        calibration = compute_calibration(cal_rows, ctx.n, ctx.fingerprint_sha256)
        if calibration["calibration_sha256"] != cal_digest:
            raise ResumeMismatchError(
                "recomputing the calibration from its own recorded rows gives "
                f"{calibration['calibration_sha256']}, not the recorded "
                f"{cal_digest}. Refusing to continue under a different "
                "calibration; nothing was modified.")
    elif evaluations_started:
        # Evaluations exist but the calibration record is gone or corrupt
        # ({cal_why}). The original can be recovered only from a COMPLETE,
        # verified sweep -- and only if it reproduces every digest the
        # evaluation evidence carries.
        incomplete = [sweep_unit_id(mb, cs) for mb, cs in grid
                      if disk[(mb, cs)][0] is None]
        if incomplete:
            raise ResumeMismatchError(
                f"evaluation evidence exists but the calibration record is "
                f"unusable ({cal_why}) and {len(incomplete)} held-out sweep "
                f"unit(s) lack valid evidence ({incomplete[:5]}). Re-measuring "
                "them would silently create a REPLACEMENT calibration under "
                "evaluations bound to the original. Refusing; nothing was "
                "modified.")
        rows = [disk[g][0]["row"] for g in grid]
        calibration = compute_calibration(rows, ctx.n, ctx.fingerprint_sha256)
        bound = set()
        for f in eval_files:
            payload, _ = _read_json(os.path.join(paths.raw_dir, f))
            if isinstance(payload, dict) and payload.get("calibration_sha256"):
                bound.add(payload["calibration_sha256"])
        if checkpoint_calibration_sha256 is not None:
            bound.add(checkpoint_calibration_sha256)
        if not bound:
            raise ResumeMismatchError(
                f"the calibration record is unusable ({cal_why}) and neither the "
                "checkpoint nor any evaluation unit records the ORIGINAL "
                "calibration digest, so a recomputed calibration cannot be "
                "shown to be the original -- only to be self-consistent. "
                "Refusing; nothing was modified. If no evaluation was ever "
                "run, move the calibration record aside and resume.")
        if bound - {calibration["calibration_sha256"]}:
            raise ResumeMismatchError(
                "evaluation evidence and/or the checkpoint are bound to "
                f"calibration digest(s) {sorted(bound)} but the sweep evidence "
                f"reproduces {calibration['calibration_sha256']}. Refusing to "
                "mix them; nothing was modified.")
        for (mb, cs) in grid:
            unit = sweep_unit_id(mb, cs)
            rows_by_key[unit.split(":", 1)[1]] = disk[(mb, cs)][0]["row"]
            if not ledger.claimed(unit):
                marks.append((unit, "valid sweep evidence; checkpoint entry "
                                    "restored during calibration recovery"))
        write_calibration = (rows, f"recovered from complete sweep evidence "
                                   f"(calibration record: {cal_why})")
    else:
        # Before any evaluation: incomplete sweep units may simply be measured.
        for (mb, cs) in grid:
            unit = sweep_unit_id(mb, cs)
            payload, why = disk[(mb, cs)]
            exists = os.path.isfile(paths.sweep_raw(mb, cs))
            if payload is not None and ledger.claimed(unit):
                rows_by_key[unit.split(":", 1)[1]] = payload["row"]
            elif ledger.claimed(unit):
                moves.append((unit, paths.sweep_raw(mb, cs) if exists else None,
                              why, True))
            elif exists:
                moves.append((unit, paths.sweep_raw(mb, cs),
                              "evidence present without a checkpoint entry "
                              "(the session ended between the two writes); "
                              "re-measured" if payload is not None else why,
                              False))
    return {"rows_by_key": rows_by_key, "calibration": calibration,
            "moves": moves, "marks": marks, "write_calibration": write_calibration}


def apply_restore(restore: dict, ledger: ResumeLedger, paths: RunPaths) -> None:
    """Carry out what `plan_restore` decided. Runs only once the run owns the
    checkpoint, so a refused resume never reaches it."""
    for unit, why in restore["marks"]:
        ledger.mark(unit)
        ledger.note(unit, "restored", why)
    for unit, path, why, was_claimed in restore["moves"]:
        moved = supersede(paths, path) if path else None
        if was_claimed:
            ledger.reject(unit, why)
        else:
            ledger.note(unit, "superseded", why)
        if moved:
            ledger.note(unit, "moved_aside", _rel(moved))
    if restore["write_calibration"] is not None:
        rows, why = restore["write_calibration"]
        save_calibration(paths, restore["calibration"], rows, recovered_from=why)
        ledger.note("calibration", "recovered", why)


def run_all(seeds: list, n: int, *, mode: str = RUN_MODE_PUBLISHED,
            allow_cpu: bool = False,
            batches: list | None = None, chunks: list | None = None,
            rho: float = TARGET_RHO,
            mu_artifact: str = MU_SOURCE_ARTIFACT,
            resume: bool = False,
            paths: RunPaths | None = None,
            sink: dict | None = None,
            session: dict | None = None,
            verify_archive: str | None = None) -> dict:
    batches = batches or GRID_BATCH_AXIS
    chunks = chunks or GRID_CHUNK_AXIS
    if paths is None:
        paths = RunPaths(PUBLISHED_OUTPUT_DIR if mode == RUN_MODE_PUBLISHED
                         else SMOKE_OUTPUT_DIR)
    session = session if session is not None else {}
    session["owns_checkpoint"] = False

    # 0. Everything that can refuse WITHOUT a model -- including the full resume
    #    fingerprint check -- runs first, and writes nothing.
    pre = preflight(mode=mode, allow_cpu=allow_cpu, seeds=seeds, n=n,
                    batches=batches, chunks=chunks, rho=rho,
                    mu_artifact=mu_artifact, resume=resume, paths=paths,
                    verify_archive=verify_archive)
    prior, source_check = pre["prior"], pre["source_check"]
    saturated_ref = pre["saturated_reference"]

    dtype = _dtype()
    print(f"Device: {DEVICE} | dtype: {dtype} | mode: {mode} | {len(seeds)} "
          f"seeds x {n} requests | grid {len(batches)}x{len(chunks)} | seeds "
          f"{seeds} | output {_rel(paths.output_dir)}", flush=True)

    tokenizer, model = load_model_and_tokenizer(dtype)

    # 0a. IDENTITY of what actually executes: the device the model's weights
    #     are on (not merely the default device), the stack around it, and the
    #     immutable model revision + file digests. Published runs fail closed.
    model_dev = _model_device(model)
    runtime = runtime_identity(model_dev if model_dev is not None else DEVICE)
    model_id = model_identity(MODEL_NAME)
    if mode == RUN_MODE_PUBLISHED:
        why = []
        if model_dev is None:
            why.append("the device holding the model's parameters is unknown")
        elif _device_index(model_dev) != pre["runtime"]["device_index"]:
            why.append(f"the model is on {model_dev}, not the preflight device")
        if (runtime.get("gpu") or {}).get("name") != REQUIRED_GPU_NAME:
            why.append(f"the model executes on {(runtime.get('gpu') or {}).get('name')!r}, "
                       f"not a {REQUIRED_GPU_NAME}")
        if not model_id.get("resolved"):
            why.append("the model's immutable identity could not be established "
                       f"({model_id.get('why')})")
        if why:
            raise PreflightRefusal("published run refused after loading the "
                                   "model: " + "; ".join(why))
    if runtime != pre["runtime"]:
        raise PreflightRefusal(
            "the runtime identity changed between preflight and model load: "
            f"{pre['runtime']} -> {runtime}")
    if mode == RUN_MODE_SMOKE and not model_id.get("resolved"):
        print(f"[identity] model identity UNRESOLVED ({model_id.get('why')}); "
              "recorded -- a smoke run is never a result", flush=True)

    # 0b. PROVE which dimensions execute, before spending anything. Same probe,
    #     same abort: anything declared inert that turns out live stops the run.
    probe_sched = abl._new_scheduler(model)
    liveness = fh.probe_live_dimensions(probe_sched)
    print(f"[liveness] searched {liveness['live_dimensions_searched']} "
          f"exhaustively; nothing unexpectedly live -- PASS", flush=True)

    # 1. THE OPERATING POINT, derived from the arm that will actually execute.
    arm = execution_arm_for(probe_sched)
    plan = derive_service_rates(arm, artifact_path=mu_artifact, rho=rho)
    pi, pb = plan["phases"]["interactive"], plan["phases"]["batch"]
    print(f"[arrivals] execution arm = {arm} (read from the live scheduler, "
          f"not from use_cuda_graphs)", flush=True)
    print(f"[arrivals] mu_interactive = {pi['mu_max_req_per_s']:.6f} req/s -> "
          f"lambda_interactive = {pi['lambda_rps']:.6f} rps "
          f"(1/lambda = {pi['expected_mean_interarrival_s']:.4f} s)", flush=True)
    print(f"[arrivals] mu_batch       = {pb['mu_max_req_per_s']:.6f} req/s -> "
          f"lambda_batch       = {pb['lambda_rps']:.6f} rps "
          f"(1/lambda = {pb['expected_mean_interarrival_s']:.4f} s)", flush=True)

    arms_restricted = abl.all_arm_sets()
    arms_expanded = abl.expanded_arm_sets()

    # 1b. THE RESUME FINGERPRINT. Computed whether or not --resume was passed,
    #     because an uninterrupted run must leave one behind for a later session
    #     to check itself against.
    fingerprint = experiment_fingerprint(
        plan, seeds, n, batches, chunks, model=MODEL_NAME, dtype=str(dtype),
        source_check=source_check, run_mode=mode, runtime=runtime,
        model_id=model_id, saturated_ref=saturated_ref)
    fingerprint_sha256 = _sha256_text(_canonical(fingerprint))
    if prior is not None:
        diffs = compare_fingerprints(prior["fingerprint"], fingerprint)
        if diffs:
            raise ResumeMismatchError(
                "--resume REFUSED after loading the model: " + "; ".join(diffs)
                + ". Nothing was modified.")
    completed = ((prior or {}).get("checkpoint") or {}).get("units_completed") or []
    ledger = ResumeLedger(fingerprint, completed)
    ledger.resumed = bool(resume)
    ctx = UnitContext(paths, fingerprint_sha256, n, plan)
    all_units = unit_plan(batches, chunks, seeds)

    # 1c. What the evidence on disk supports. Pure read; may refuse. The
    #     checkpoint's own calibration digest is one of the binding digests.
    prior_calibration = (prior or {}).get("calibration")
    prior_cal_digest = (prior_calibration.get("calibration_sha256")
                        if isinstance(prior_calibration, dict) else None)
    restore = plan_restore(ledger, ctx, batches, chunks,
                           checkpoint_calibration_sha256=prior_cal_digest)
    rows_by_key = restore["rows_by_key"]
    grid_keys = [sweep_unit_id(mb, cs).split(":", 1)[1]
                 for mb in batches for cs in chunks]

    history = list((prior or {}).get("run_history") or [])
    if prior is not None and prior.get("aborted") and not (
            history and history[-1].get("aborted")):
        history.append({"session_index": len(history), "status":
                        prior["aborted"].get("status", STATUS_OPERATIONAL),
                        "aborted": prior["aborted"],
                        "note": "abort record carried over from the checkpoint"})
    this_session = {"session_index": len(history), "started_utc": _now(),
                    "resume": bool(resume), "mode": mode, "status": "running",
                    "archive_sha256_status": source_check.get(
                        "archive_sha256_status")}
    history.append(this_session)

    results: dict = {
        "objective": (
            "Move ONE factor -- the arrival process -- from a saturated burst to "
            "a sub-critical Poisson stream at a derived rho, and re-measure the "
            "final hardening's four comparisons. A scope experiment, not a "
            "repair: it asks whether the saturated finding is a property of "
            "adaptation or of the operating point."),
        "run_mode": mode,
        # Eligible to be reported only if published AND run_status ends "complete".
        "published_run": mode == RUN_MODE_PUBLISHED,
        "output_paths": paths.as_dict(),
        "relationship_to_final_hardening": {
            "supersedes": False,
            # Read-only, hash-verified, for the saturated side of H1/H2.
            "reads_its_artifact": True,
            "writes_its_artifact": False,
            "reuses_its_code": [
                "probe_live_dimensions", "freeze_scales",
                "select_static_baselines", "run_utility",
                "convergence_diagnostic", "CONVERGENCE_RULE", "_cycle_records",
                "_build_bandit", "load_source_provenance",
            ],
            "why": ("importing the saturated experiment's controller, reward, "
                    "selection and convergence code is what makes 'held fixed' "
                    "true by construction; the saturated artifact itself is "
                    "FROZEN EVIDENCE: it is read, hash-verified, only for the "
                    "saturated side of H1/H2, and is never modified here"),
        },
        "research_questions": RESEARCH_QUESTIONS,
        "pre_registration": PRE_REGISTRATION,
        "pre_registration_note": (
            "`pre_registration` is the structured summary held in the script, "
            "not the document text; the document is identified by "
            "preregistration_document_sha256, which is also a resume "
            "fingerprint key"),
        "preregistration_document_sha256": _sha256_of_file(PREREG_PATH),
        "saturated_reference": SATURATED_REFERENCE,
        # The verified frozen evidence H1/H2 read: source member, hashes,
        # validated identity, fields used and full-precision quantities -- or,
        # if it failed, status "unverified" and every problem found.
        "saturated_reference_verification": saturated_ref,
        "saturated_comparability": SATURATED_COMPARABILITY,
        "source_verification": source_check,
        # What actually executed. Also fingerprint keys; repeated here so a
        # smoke on other hardware says exactly what it ran on.
        "runtime_identity": runtime,
        "model_identity": model_id,
        "liveness_probe": liveness,
        "arrival_plan": plan,
        "scenario": abl.scenario_description(n),
        "workload": {
            "shapes": "ablation_live / src.carl.live _SCENARIO_SPEC, UNCHANGED",
            "phase_split": f"requests 1-{n // 2} INTERACTIVE, "
                           f"{n // 2 + 1}-{n} BATCH",
            "arrival_process": ARRIVAL_POISSON,
            "phase1_open_loop": True,
            "why_open_loop": (
                "the historical harness dumps every phase-1 request the instant "
                "half of phase 0 has finished. That is a CLOSED-LOOP trigger: "
                "the BATCH phase's arrival pattern becomes a function of how "
                "fast the engine drained phase 0, so a phase-specific arrival "
                "RATE cannot be expressed under it. Here both phases are "
                "released from their own offsets and the regime shift is "
                "recorded when the first BATCH request actually arrives."),
            "arrival_rng": ("random.Random(f'{ARRIVAL_RNG_NAMESPACE}:{seed}') "
                            "-- a named sub-stream, independent of the "
                            "prompt-length draws"),
            "deterministic_in_seed": True,
        },
        "seeds": seeds, "runs": len(seeds), "requests": n,
        "observe_interval": OBSERVE_INTERVAL,
        "validation_seed": VALIDATION_SEED,
        "held_fixed": [
            "Tesla T4", "TinyLlama-1.1B-Chat-v1.0 fp16", "seeds 42..51",
            f"{n} requests per run",
            "two-phase semantic split (INTERACTIVE shape then BATCH shape)",
            f"controller cadence OBSERVE_INTERVAL={OBSERVE_INTERVAL}",
            "restricted/expanded arm-set definitions",
            "RepairedLinUCBBandit, alpha=0.5",
            f"held-out calibration seed {VALIDATION_SEED}",
            f"context vector FEATURE_DIM={FEATURE_DIM}",
            "LiveRewardV2 with require_valid_metrics=True",
            "reward gates and their frozen thresholds",
            "convergence_diagnostic and CONVERGENCE_RULE (imported)",
        ],
        "changed_vs_final_hardening": [
            "arrivals: burst bulk dump -> open-loop Poisson at a derived, "
            "phase-specific lambda (rho = 0.40 of each phase's own measured "
            "eager capacity)",
            "reward scales: RE-DERIVED from a fresh held-out sweep at this "
            "operating point, not carried over",
            "Static-Tput / Static-Utility: RE-SELECTED from that fresh sweep, "
            "not carried over",
            "statistics: per-seed signs and an exact Wilcoxon signed-rank test "
            "added as robustness",
        ],
        "not_run_and_why": {
            "DynOracle": ("withdrawn: it ranks arms by rewards collected in a "
                          "particular cycle ORDER, so it is not an oracle"),
            "inactive ablations (spec / router / KV eviction)": (
                "these subsystems cannot execute in this harness, so freezing "
                "them measures nothing; see liveness_probe"),
        },
        "action_space": {
            "restricted": effective_arm_report(arms_restricted),
            "expanded": effective_arm_report(arms_expanded),
            "summary_restricted": abl.arm_set_summary(arms_restricted),
            "summary_expanded": abl.arm_set_summary(arms_expanded),
            "expanded_batch_axis": abl.ARM_SET_BATCH_AXIS,
        },
        "utility_estimators": dict(UTILITY_ESTIMATORS),
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
        "scope_note": (
            "ONE new operating point: open-loop Poisson arrivals at a derived "
            "rho = 0.40 on a single T4 with one 1.1B model, over the two live "
            "knobs this harness exposes. Together with the saturated burst it "
            "gives TWO points on a load axis, which is a boundary check, not a "
            "load envelope. It says nothing about speculation, routing or KV "
            "eviction, which are inactive here."
            if mode == RUN_MODE_PUBLISHED else
            "SMOKE -- wiring verification only. NOT A RESULT."),
        "configs": {},
        "run_history": history,
        "stage": "started",
    }
    if mode == RUN_MODE_SMOKE:
        results["smoke"] = True
        results["cpu_smoke"] = DEVICE.type != "cuda"
    if sink is not None:
        sink.clear()
        sink.update(results)
        results = sink

    def _verified_rows() -> list:
        """ALL verified rows, in canonical grid order -- never a visited prefix."""
        return [rows_by_key[k] for k in grid_keys if k in rows_by_key]

    # 1d. ENGINE WARM-UP, before the run owns the checkpoint: identity and the
    #     evidence on disk are already verified, and a warm-up that fails or is
    #     interrupted leaves every byte of an earlier checkpoint as it was. It
    #     runs only if some unit will actually be measured in this process.
    warmup_spec = fingerprint["engine_warmup_protocol"]
    pending = measured_work_pending(ledger, restore, ctx, batches, chunks, seeds)
    if pending["pending"]:
        print(f"[warm-up] {warmup_spec['version']}: {len(warmup_spec['stages'])} "
              f"declared stages before the first of "
              f"{len(pending['sweep_units']) + len(pending['eval_units'])} "
              f"unit(s) to measure; excluded from every measurement", flush=True)
        warmup = run_engine_warmup(model, tokenizer, warmup_spec,
                                   device=model_dev)
        print(f"[warm-up] complete: {warmup['steps']} steps, "
              f"{warmup['tokens_emitted']} tokens in {warmup['elapsed_s']:.2f}s "
              f"(NOT part of any measured unit)", flush=True)
    else:
        warmup = {"version": warmup_spec["version"],
                  "status": "skipped_no_measured_work_pending",
                  "inference_performed": False}
        print("[warm-up] skipped: every unit is restored from verified "
              "evidence, so no inference runs in this session", flush=True)
    warmup["pending_units_at_start"] = (len(pending["sweep_units"])
                                        + len(pending["eval_units"]))
    warmup["at_utc"] = _now()
    this_session["engine_warmup"] = warmup
    results["engine_warmup"] = warmup
    results["engine_warmup_protocol"] = warmup_spec

    def _require_warm() -> None:
        # Fail closed: a unit measured without the declared warm-up would be a
        # different measurement protocol. Unreachable while
        # measured_work_pending mirrors the loops below.
        if warmup.get("status") != "complete":
            raise EngineWarmupError(
                "a measured unit was about to run without the declared engine "
                f"warm-up (status {warmup.get('status')!r})", warmup)

    # IDENTITY FIRST. Before this session may write the checkpoint, the dict it
    # would write already carries the fingerprint, the ledger as the previous
    # session left it (so every prior claim survives), every verified row and
    # the checkpoint's calibration binding. An abort anywhere from here on --
    # inside apply_restore, a supersede, a calibration write, an interrupt --
    # therefore writes a checkpoint that is still resumable.
    results["fingerprint"] = fingerprint
    results["fingerprint_sha256"] = fingerprint_sha256
    results["unit_plan"] = {
        "order": all_units,
        "n_units": len(all_units),
        "note": ("the ONE deterministic execution order: the held-out sweep in "
                 "grid order, then every treatment crossed with every seed. "
                 "--resume walks this same list and skips completed units; it "
                 "never reorders."),
    }
    results["checkpoint"] = ledger.as_dict()
    results["held_out_sweep_rows"] = _verified_rows()
    if isinstance(prior_calibration, dict):
        results["calibration"] = prior_calibration
    results["h5_reference"] = h5_reference(plan, n, seeds)
    session["ledger"] = ledger

    # OWNERSHIP. From here on this session may write the checkpoint.
    session["owns_checkpoint"] = True
    apply_restore(restore, ledger, paths)
    results["checkpoint"] = ledger.as_dict()
    results["stage"] = "arrival_plan_derived"
    write_artifact(results, seeds, n, paths)

    # 2. FRESH held-out sweep -> frozen scales -> gate -> both baselines.
    def _checkpoint_rows():
        rows_now = _verified_rows()
        results["held_out_sweep_rows"] = rows_now
        results["held_out_sweep_progress"] = {
            "rows_completed": len(rows_now),
            "of": len(batches) * len(chunks),
        }
        results["checkpoint"] = ledger.as_dict()
        results["stage"] = "held_out_sweep"
        write_artifact(results, seeds, n, paths)

    calibration = restore["calibration"]
    if calibration is None:
        rows = sweep_live_plane(
            model, tokenizer, n, VALIDATION_SEED, plan, batches, chunks,
            ledger=ledger, rows_by_key=rows_by_key, on_row=_checkpoint_rows,
            save_unit=lambda mb, cs, row, m: save_sweep_unit(mb, cs, row, m, ctx),
            before_measure=_require_warm)
    else:
        rows = _verified_rows()
        print(f"[held-out sweep] all {len(rows)} rows RESTORED from the "
              f"calibration record {calibration['calibration_sha256'][:12]}",
              flush=True)
    # The static selection is an argmax over the FULL plane, so it may not be
    # computed until every grid point exists.
    assert_sweep_complete(rows, batches, chunks)
    results["held_out_sweep_rows"] = rows

    # ORDER MATTERS: scales are frozen from the MEASUREMENTS, then gated, then
    # both baselines are selected -- the same imported functions, on this run's
    # own rows. On a resume the result must reproduce the recorded digest.
    if calibration is None:
        try:
            calibration = compute_calibration(rows, n, fingerprint_sha256)
        except DegenerateRewardError as exc:
            record_terminal_failure(
                results, session, ctx, exc, label="prerun_sweep_gate",
                gate="pre_run_candidate_arm_spread",
                evidence={"held_out_rows": rows,
                          "frozen_scales": getattr(exc, "frozen_scales", None),
                          "threshold": MIN_ARM_REWARD_SPREAD})
            raise
        save_calibration(paths, calibration, rows)
    ctx.calibration = calibration
    scales = calibration["scales"]
    cal_record = calibration["record"]
    cal_record["reused_from_saturated_run"] = False
    cal_record["why_not_reused"] = (
        "the saturated scales are medians of a saturated operating range; "
        "under sub-critical arrivals the observed throughput and tail latency "
        "ranges are different quantities, and a yardstick fitted elsewhere "
        "would place this plane entirely above or below its mid-point")
    selection = calibration["selection"]
    static_cfg_for = {"Static-Tput": calibration["cfg_tput"],
                      "Static-Utility": calibration["cfg_utility"]}

    results["reward_calibration"] = cal_record
    results["static_selection"] = selection
    results["calibration"] = {
        "calibration_sha256": calibration["calibration_sha256"],
        "binding": calibration["binding"],
        "record_path": _rel(paths.calibration),
        "role": ("every evaluation unit records this digest; a unit bound to "
                 "any other digest is never reused"),
    }
    results["landscape"] = landscape_report(rows, scales, selection)
    results["reward"] = {
        "version": provenance.REWARD_VERSION,
        "function": "src.carl.reward.utility_v2",
        "adapter": "src.carl.live_reward.LiveRewardV2",
        "thresholds": {
            "min_arm_reward_spread": MIN_ARM_REWARD_SPREAD,
            "min_live_reward_spread": MIN_LIVE_REWARD_SPREAD,
            "min_live_distinct": MIN_LIVE_DISTINCT,
            "provenance": THRESHOLD_PROVENANCE,
            "changed_vs_final_hardening": False,
        },
        "weights": dict(DEFAULT_WEIGHTS),
    }

    results["static_selection_is_recomputed_every_run"] = (
        "The frozen scales and both static baselines are RECOMPUTED from the "
        "complete grid on every invocation, resumed or not, and must reproduce "
        "the calibration digest every evaluation unit carries. A resume "
        "therefore cannot move them: once evaluation has begun the rows are "
        "restored from the calibration record, and a sweep that cannot "
        "reproduce the recorded digest refuses to continue.")

    # 3. The repaired reward. Cold-start semantics preserved EXACTLY.
    reward_fn = LiveRewardV2(scales, require_valid_metrics=True,
                             min_latency_samples=1)
    results["reward"].update(reward_fn.as_dict())
    results["stage"] = "static_selection_complete"
    results["checkpoint"] = ledger.as_dict()
    write_artifact(results, seeds, n, paths)       # CHECKPOINT: after selection

    per_seed: dict = {}
    arrivals_by_config: dict = {}
    for name in CONFIGS:
        per_run = []
        for i, seed in enumerate(seeds):
            unit = eval_unit_id(name, seed)
            raw_path = paths.eval_raw(name, seed)
            run = None
            if ledger.claimed(unit):
                run, why = load_eval_unit(name, seed, ctx)
                if run is None:
                    moved = supersede(paths, raw_path)
                    ledger.reject(unit, why)
                    if moved:
                        ledger.note(unit, "moved_aside", _rel(moved))
                else:
                    ledger.accept(unit)
                    per_run.append(run)
                    print(f"  {name:<15} {i+1}/{len(seeds)} (seed {seed}): "
                          f"RESUMED from checkpoint "
                          f"({run['throughput_tps']:7.2f} tok/s)", flush=True)
                    continue
            elif os.path.isfile(raw_path):
                moved = supersede(paths, raw_path)
                ledger.note(unit, "superseded",
                            "raw evidence present without a checkpoint entry "
                            "(the session ended between the two writes); "
                            "re-measured. Moved to " + _rel(moved))
            _require_warm()
            try:
                run = run_one(
                    name, model, tokenizer, n, seed, plan,
                    static_cfg=static_cfg_for.get(name),
                    reward_fn=reward_fn if name in _CONTROLLER_CONFIGS else None,
                    scales=scales)
            except DegenerateRewardError as exc:
                # TERMINAL. Recorded in memory first, then a small marker, then
                # the failed unit's measurements, controller trace and gate
                # evidence -- all before the error propagates.
                failed = getattr(exc, "failed_run", None) or {}
                results["checkpoint"] = ledger.as_dict()
                record_terminal_failure(
                    results, session, ctx, exc, label=unit,
                    gate="per_run_live_reward_stream",
                    evidence={"config": name, "seed": seed, **_unit_body(failed)})
                raise
            # Validated by the resume rules on its round-tripped form, THEN
            # written, THEN counted; the in-memory run is what a resume would
            # reload, so fresh and resumed units aggregate identically.
            try:
                run = save_eval_unit(name, seed, run, ctx)
            except InvalidFreshEvidenceError as exc:
                results["invalid_fresh_evidence"] = {
                    "status": STATUS_INVALID_EVIDENCE, "unit": unit,
                    "problems": exc.problems,
                    "evidence_file": _rel(exc.evidence_file) if exc.evidence_file else None,
                    "resume_policy": ("--resume refuses until the defect is "
                                      "investigated; the unit is not re-measured"),
                }
                raise
            per_run.append(run)
            ledger.mark(unit)
            extra = ""
            if name in _CONTROLLER_CONFIGS:
                rv = run["reward_validity"]
                extra = (f" | u={run['mean_valid_cycle_utility']:.4f} "
                         f"valid={run['n_valid_rewards']}/{run['decisions']} "
                         f"warmup={rv['warmup_cycles']} "
                         f"updates={rv['valid_reward_updates']} "
                         f"armChg={run['arm_changes']} "
                         f"rate={run['arm_change_rate']:.3f}")
            rl = run["drain_overrun"]
            print(f"  {name:<15} {i+1}/{len(seeds)} (seed {seed}): "
                  f"{run['throughput_tps']:7.2f} tok/s "
                  f"ttftP99={run['ttft_p99']:9.1f}ms "
                  f"runU={run['run_utility_v2']:.4f} "
                  f"overrun={rl['arrival_window_overrun_ratio']:.2f} "
                  f"rhoI={_rho_of(run, 'interactive'):.3f} "
                  f"rhoB={_rho_of(run, 'batch'):.3f}{extra}",
                  flush=True)
            # CHECKPOINT: after EVERY (config, seed). A disconnect loses at most
            # the run in flight.
            results["stage"] = f"evaluating:{name}:seed{seed}"
            results["checkpoint"] = ledger.as_dict()
            write_artifact(results, seeds, n, paths)

        agg = {"arm_set": _ARM_SET_FOR.get(name),
               "policy": results["config_policy"][name]}
        for key in _METRICS:
            mean, std = _mean_std([m[key] for m in per_run])
            agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
        agg["_per_seed"] = {k: [m[k] for m in per_run]
                            for k in ("throughput_tps", "run_utility_v2")}
        agg["arrival_validation_per_seed"] = {
            str(s): m["arrival_validation"] for s, m in zip(seeds, per_run)}
        agg["offered_load_per_seed"] = {
            str(s): m["offered_load"] for s, m in zip(seeds, per_run)}
        agg["drain_overrun_per_seed"] = {
            str(s): m["drain_overrun"] for s, m in zip(seeds, per_run)}
        agg["phase_boundary_per_seed"] = {
            str(s): m["phase_boundary"] for s, m in zip(seeds, per_run)}
        # DESCRIPTIVE: scheduled arrival phase vs inferred regime, drain per
        # phase, and mixed execution. Never an outcome, never a filter.
        agg["phase_regime_per_seed"] = {
            str(s): phase_regime_report(m) for s, m in zip(seeds, per_run)}
        arrivals_by_config[name] = {s: m["arrivals"] for s, m in zip(seeds, per_run)}
        # Realised offered load across seeds, per phase. Reported as a SPREAD,
        # never as a filter: the scatter is the Poisson process, and no seed is
        # dropped for landing away from 0.40.
        agg["empirical_rho_across_seeds"] = {
            ph: {
                "values": [m["offered_load"]["per_phase"][ph]["empirical_rho"]
                           for m in per_run],
                "mean": statistics.fmean(
                    m["offered_load"]["per_phase"][ph]["empirical_rho"]
                    for m in per_run),
                "min": min(m["offered_load"]["per_phase"][ph]["empirical_rho"]
                           for m in per_run),
                "max": max(m["offered_load"]["per_phase"][ph]["empirical_rho"]
                           for m in per_run),
                "target": plan["target_rho"],
                "policy": "no seed is excluded on the basis of these values",
            } for ph in ("interactive", "batch")}
        if name in _STATIC_CONFIGS:
            agg["applied_config"] = per_run[0]["applied_config"]
        if name in _CONTROLLER_CONFIGS:
            agg["decisions_mean"] = statistics.fmean(m["decisions"] for m in per_run)
            agg["arm_changes_mean"] = statistics.fmean(
                m["arm_changes"] for m in per_run)
            agg["arm_change_rate_mean"] = statistics.fmean(
                m["arm_change_rate"] for m in per_run)
            agg["mean_valid_cycle_utility"] = statistics.fmean(
                m["mean_valid_cycle_utility"] for m in per_run)
            agg["mean_valid_cycle_reward_per_seed"] = {
                str(s): m["mean_valid_cycle_utility"]
                for s, m in zip(seeds, per_run)}
            agg["valid_reward_spread_per_seed"] = {
                str(s): m["valid_reward_spread"] for s, m in zip(seeds, per_run)}
            agg["n_distinct_valid_rewards_per_seed"] = {
                str(s): m["n_distinct_valid_rewards"]
                for s, m in zip(seeds, per_run)}
            agg["valid_reward_updates_mean"] = statistics.fmean(
                m["reward_validity"]["valid_reward_updates"] for m in per_run)
            agg["valid_reward_updates_min"] = min(
                m["reward_validity"]["valid_reward_updates"] for m in per_run)
            # THE COLD-START RECORD, per seed. Pre-registered H4 reads this.
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
                "h4_positive_control_holds": all(
                    m["reward_validity"]["warmup_cycles"] >= 1 for m in per_run),
            }
            # Q6, by the SAME rule as the saturated run.
            agg["convergence_supported_seeds"] = sum(
                1 for m in per_run if m["convergence"]["supported_in_any_regime"])
            agg["convergence_per_seed"] = {
                str(s): {
                    "supported_in_any_regime": m["convergence"][
                        "supported_in_any_regime"],
                    "supported_in_all_regimes": m["convergence"][
                        "supported_in_all_regimes"],
                    "per_regime": {
                        r: {"supported": v["supported"],
                            "why_not": v["why_not"],
                            "decisions": v["decisions"],
                            "first_half_distinct_arms": v[
                                "first_half_distinct_arms"],
                            "second_half_modal_share": v[
                                "second_half_modal_share"]}
                        for r, v in m["convergence"]["per_regime"].items()},
                } for s, m in zip(seeds, per_run)}
            agg["convergence_claim"] = (
                "supported in at least one regime on {}/{} seeds by the "
                "pre-registered rule (unchanged from final_hardening)".format(
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
        results["stage"] = f"completed:{name}"
        results["checkpoint"] = ledger.as_dict()
        write_artifact(results, seeds, n, paths)   # CHECKPOINT after every config

    results["comparisons"] = {
        label: {
            "a": a, "b": b, "metric": metric, "question": q,
            metric: _paired(per_seed[a][metric], per_seed[b][metric]),
            "secondary": {
                k: _paired(per_seed[a][k], per_seed[b][k])
                for k in ("throughput_tps", "run_utility_v2") if k != metric
            },
            # The frozen run's OWN statistics for the same comparison, verbatim
            # from the verified artifact (never recomputed); None if it failed.
            "saturated_reference": (
                (saturated_ref.get("frozen_comparisons_verbatim") or {}).get(label)
                if saturated_ref.get("status") == "verified" else None),
        }
        for label, a, b, metric, q in COMPARISONS
        if a in per_seed and b in per_seed
    }
    for label, comparator in H2_COMPARATORS.items():
        c = results["comparisons"].get(label)
        if c is not None:
            if c["b"] != comparator:
                raise RuntimeError(f"H2 comparator for {label} is {c['b']!r}, "
                                   f"registered {comparator!r}")
            c["h2_relative_paired_difference"] = h2_relative_difference(
                per_seed[c["a"]]["throughput_tps"],
                per_seed[c["b"]]["throughput_tps"])
    not_evaluable = registered_deviations(seeds, n, batches, chunks, rho,
                                          mu_artifact)
    if mode != RUN_MODE_PUBLISHED:
        not_evaluable = [f"run_mode is {mode!r}, not a published run"] + not_evaluable
    # Matched design with the FROZEN run, checked against its recorded facts
    # (not against this module's constants, which a caller could change).
    if (list(seeds) != sref.EXPECTED_SEEDS or n != sref.EXPECTED_REQUESTS
            or list(batches) != sref.EXPECTED_GRID_BATCHES
            or list(chunks) != sref.EXPECTED_GRID_CHUNKS):
        not_evaluable.append("seeds / n / grid differ from the saturated run's "
                             "(42..51, 200, 7x4)")
    results["hypotheses"] = hypotheses_status(results, saturated_ref,
                                              not_evaluable)

    # H5 (amended): the 20-cell family on the evaluation seeds' arrivals --
    # identical across treatments by construction, which is checked, not
    # assumed -- and seed 999 separately.
    first = arrivals_by_config[CONFIGS[0]]
    held_out = poisson_arrivals(n, VALIDATION_SEED, plan)
    results["h5"] = h5_family_report(results["h5_reference"], first, held_out,
                                     plan, n)
    results["h5"]["arrivals_identical_across_treatments"] = all(
        arr == first for arr in arrivals_by_config.values())
    results["h5"]["held_out_matches_recorded_sweep"] = all(
        r["arrival_validation"][c["phase"]]["empirical_mean_interarrival_s"]
        == c["empirical_mean_interarrival_s"]
        for r in rows for c in results["h5"]["held_out_cells"])
    results["stage"] = "complete"
    results["checkpoint"] = ledger.as_dict()
    this_session.update(status="complete", ended_utc=_now())
    return results


def _print(results: dict) -> None:
    plan = results["arrival_plan"]
    print(f"\n=== SUB-CRITICAL POISSON (rho={plan['target_rho']}, arm="
          f"{plan['execution_arm']}; mean +/- std over {results['runs']} seeds, "
          f"{results['requests']} requests) ===")
    for ph, p in plan["phases"].items():
        print(f"  {ph:<12} mu={p['mu_max_req_per_s']:.4f} req/s  "
              f"lambda={p['lambda_rps']:.4f} rps  "
              f"1/lambda={p['expected_mean_interarrival_s']:.4f} s  "
              f"[{p['saturation_row']}]")

    lg = results.get("landscape")
    if lg:
        t, u = lg["throughput_tps"], lg["utility_v2"]
        print(f"\n=== Q1: held-out landscape over {lg['n_grid_points']} configs ===")
        print(f"  throughput  min={t['min']:.2f}  max={t['max']:.2f}  "
              f"range={t['range_absolute']:.2f} ({t['range_relative_pct']:.2f}%)")
        print(f"  utility_v2  min={u['min']:.4f}  max={u['max']:.4f}  "
              f"range={u['range_absolute']:.4f}")
        hyp = results.get("hypotheses") or {}
        if hyp:
            sat_h1 = hyp["H1"].get("saturated_relative_throughput_range_pct")
            print("  H1 saturated relative range: "
                  + (f"{sat_h1:.4f}%" if sat_h1 is not None else "UNRESOLVED")
                  + f" | H1 {hyp['H1']['status']} | H2 {hyp['H2']['status']}"
                  + " (descriptive cross-regime comparison)")
        for label, blk in (("throughput", t), ("utility_v2", u)):
            fl = blk.get("flatness") or {}
            if not fl.get("n"):
                continue
            w05 = (fl.get("within_0p5pct_of_best") or {}).get("count")
            w10 = (fl.get("within_1p0pct_of_best") or {}).get("count")
            gap = fl.get("best_minus_second_best")
            pct = fl.get("best_minus_second_best_relative_pct")
            print(f"  {label:<11} best-second={gap:.4f} ({pct:.3f}%)  "
                  f"within 0.5%: {w05}/{fl['n']}  within 1.0%: {w10}/{fl['n']}")
        print("  (near-ties are DESCRIPTIVE; the pre-registered argmax stands)")
        print(f"  Static-Tput    mb={lg['static_tput_identity']['max_batch_size']} "
              f"cs={lg['static_tput_identity']['chunk_size']}")
        print(f"  Static-Utility mb={lg['static_utility_identity']['max_batch_size']} "
              f"cs={lg['static_utility_identity']['chunk_size']}")
        print(f"  objectives agree: {lg['objectives_agree']}")

    hdr = ["config", "tok/s", "ttft p50/p99 ms", "tpot p50/p99 ms", "run util",
           "cycle reward", "validUpd", "decisions", "armChg rate"]
    print("\n| " + " | ".join(hdr) + " |")
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
            f"{a.get('arm_change_rate_mean', nan):.3f}",
        ]) + " |")
    print("  (`cycle reward` is the mean VALID per-cycle reward the learner "
          "consumed; it is NOT `run util` and the two are never substituted)")

    print("\n=== Arrival-process validation (seed 42 of the first config) ===")
    for name in CONFIGS:
        av = (results["configs"].get(name) or {}).get("arrival_validation_per_seed")
        if not av:
            continue
        first = av[sorted(av)[0]]
        for ph, r in first.items():
            print(f"  {ph:<12} lambda={r['requested_lambda_rps']:.4f} "
                  f"1/lambda={r['expected_mean_interarrival_s']:.4f}s "
                  f"empirical mean={r['empirical_mean_interarrival_s']:.4f}s "
                  f"median={r['empirical_median_interarrival_s']:.4f}s "
                  f"emp lambda={r['empirical_lambda_rps']:.4f}rps "
                  f"within CI95: {r['mean_within_ci95']}")
        break

    print("\n=== Realised OFFERED LOAD (rho = empirical lambda / mu) ===")
    print("    (the Poisson draw scatters around the target; NO seed is ever "
          "excluded for it)")
    for name in CONFIGS:
        er = (results["configs"].get(name) or {}).get("empirical_rho_across_seeds")
        if not er:
            continue
        for ph, r in er.items():
            print(f"  {name:<15} {ph:<12} rho mean={r['mean']:.4f} "
                  f"[{r['min']:.4f}, {r['max']:.4f}] target={r['target']:.2f}")

    print("\n=== Phase splice (BATCH may not begin before INTERACTIVE ends) ===")
    for name in CONFIGS:
        pbs = (results["configs"].get(name) or {}).get("phase_boundary_per_seed")
        if not pbs:
            continue
        ok = all(b["phases_do_not_overlap"] and b["head_is_all_interactive"]
                 and b["tail_is_all_batch"] for b in pbs.values())
        gaps = [b["phase_boundary_gap_s"] for b in pbs.values()]
        print(f"  {name:<15} sequential on all seeds: {ok} | boundary gap "
              f"min={min(gaps):.4f}s max={max(gaps):.4f}s")

    print("\n=== Drain overrun (NOT rho: wall clock / arrival window) ===")
    for name in CONFIGS:
        dos = (results["configs"].get(name) or {}).get("drain_overrun_per_seed")
        if not dos:
            continue
        ratios = [d["arrival_window_overrun_ratio"] for d in dos.values()]
        print(f"  {name:<15} overrun ratio mean={statistics.fmean(ratios):.3f} "
              f"[{min(ratios):.3f}, {max(ratios):.3f}] (near 1.0 = engine kept "
              f"up with the stream)")

    print("\n=== Cold start (pre-registered H4: warmup_cycles >= 1) ===")
    for name in CONFIGS:
        cs = (results["configs"].get(name) or {}).get("cold_start")
        if not cs:
            continue
        w = list(cs["warmup_cycles_per_seed"].values())
        print(f"  {name:<15} warmup cycles per seed {w} | H4 holds: "
              f"{cs['h4_positive_control_holds']} | withheld after first valid: "
              f"{cs['withheld_after_first_valid_total']}")

    print("\n=== Q6 Convergence (SAME pre-registered rule as final_hardening) ===")
    for name in CONFIGS:
        a = results["configs"].get(name) or {}
        if "convergence_claim" in a:
            print(f"  {name:<15} {a['convergence_claim']}")

    print("\n=== Comparisons (paired over seeds; p-values are robustness only) ===")
    for label, c in results.get("comparisons", {}).items():
        t = c[c["metric"]]
        w = t["wilcoxon_signed_rank"]
        p = (f"wilcoxon p={w['p_value_two_sided_exact']:.5f}"
             if w.get("applicable") else "wilcoxon n/a (zero differences)")
        print(f"  {label}")
        print(f"    {c['a']} - {c['b']} on {c['metric']}: "
              f"{t['mean_difference']:+.4f} "
              f"t-CI95 {_fmt_ci(t.get('ci95_t'))} "
              f"(z-approx {_fmt_ci(t.get('ci95_z_normal_approx_secondary'))}) "
              f"d={t['cohens_d_paired']} "
              f"signs +{t['n_positive']}/-{t['n_negative']}/0{t['n_zero']} {p}")
        sat = c.get("saturated_reference") or {}
        if sat:
            s = sat[sat["metric"]]
            print(f"    (saturated reference, {sat['metric']}: "
                  f"{s['mean_difference']:+.4f} "
                  f"z-CI95 [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}], "
                  "frozen, verified)")
        for k, s in c["secondary"].items():
            print(f"    (secondary) on {k}: {s['mean_difference']:+.4f} "
                  f"t-CI95 {_fmt_ci(s.get('ci95_t'))}")
        h2 = (c.get("h2_relative_paired_difference") or {}).get("value")
        if h2 is not None:
            hyp = ((results.get("hypotheses") or {}).get("H2") or {})
            sat_h2 = (((hyp.get("per_comparison") or {}).get(label) or {})
                      .get("saturated") or {}).get("value")
            print(f"    H2 relative difference (vs {c['b']}): {h2:.5f}; "
                  + (f"saturated {sat_h2:.5f}" if sat_h2 is not None
                     else "saturated side UNRESOLVED"))

    h5 = results.get("h5")
    if h5:
        print(f"\n=== H5 (amended): {h5['n_cells']} family cells, "
              f"{len(h5['rejected_cells'])} outside their Bonferroni bounds; "
              f"family rejected: {h5['family_rejected']} ===")
        for c in h5["held_out_cells"]:
            print(f"  held-out seed {c['seed']} {c['phase']:<12} mean "
                  f"{c['empirical_mean_interarrival_s']:.4f}s in "
                  f"{_fmt_ci(c['bounds_mean_interarrival_s'])}: {c['inside']}")


def _fmt_ci(ci) -> str:
    return "n/a" if not ci else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def record_preflight_failure(paths: RunPaths, exc: BaseException, *,
                             resume: bool, mode: str) -> str | None:
    """A start-up failure gets its OWN file. The checkpoint is not touched."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = os.path.join(paths.preflight_dir, f"{stamp}_{os.getpid()}.json")
    try:
        return atomic_write_json(path, {
            "status": "preflight_failure",
            "reason": type(exc).__name__,
            "message": str(exc)[:4000],
            "resume": resume, "run_mode": mode, "at_utc": _now(),
            "checkpoint": _rel(paths.results),
            "checkpoint_modified": False,
            "note": ("the run never took ownership of the checkpoint, so "
                     "nothing under it -- results JSON, raw units, calibration "
                     "record -- was written, moved or deleted"),
            # A failed or interrupted engine warm-up: what it reached.
            "engine_warmup": getattr(exc, "warmup_record", None),
        })
    except Exception:
        return None


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Sub-critical Poisson operating point: the final hardening's "
                    "comparisons re-measured at a derived rho.")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--limit", type=int, default=DEFAULT_REQUESTS,
                    help="requests per run (the saturated run used 200)")
    ap.add_argument("--grid-batches", default=",".join(map(str, GRID_BATCH_AXIS)),
                    help="max_batch_size axis of the FRESH held-out sweep")
    ap.add_argument("--grid-chunks", default=",".join(map(str, GRID_CHUNK_AXIS)),
                    help="chunk_size axis of the FRESH held-out sweep")
    ap.add_argument("--rho", type=float, default=TARGET_RHO,
                    help="target offered load; pre-registered at 0.40")
    ap.add_argument("--mu-artifact", default=MU_SOURCE_ARTIFACT,
                    help="artifact the service rates are derived from")
    ap.add_argument("--resume", action="store_true",
                    help="continue a run from its existing artifact. EXECUTION "
                         "ONLY: it decides which units are re-measured, never "
                         "any scientific quantity. Aborts unless the stored "
                         "experiment fingerprint matches this invocation "
                         "exactly, and refuses after a reward-gate failure.")
    ap.add_argument("--smoke", action="store_true",
                    help="a WIRING smoke run (CUDA or not): never a result, "
                         "never written to the published paths")
    ap.add_argument("--allow-cpu-smoke", action="store_true",
                    help="permit a CPU run for WIRING VERIFICATION ONLY; implies "
                         "--smoke")
    ap.add_argument("--output-dir", default=None,
                    help="where this run writes. Published runs: docs/eval "
                         "only. Smoke runs: default smoke_runs/"
                         "subcritical_poisson, never docs/eval")
    ap.add_argument("--verify-archive", default=None,
                    help="path to the uploaded source ZIP: it is hashed and "
                         "compared with the sidecar's archive.sha256, so the "
                         "ZIP digest is VERIFIED rather than asserted. The "
                         "extracted-tree check runs regardless.")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    batches = [int(s) for s in args.grid_batches.split(",") if s.strip()]
    chunks = [int(s) for s in args.grid_chunks.split(",") if s.strip()]
    mode = (RUN_MODE_SMOKE if (args.smoke or args.allow_cpu_smoke)
            else RUN_MODE_PUBLISHED)
    paths = RunPaths(args.output_dir or (
        PUBLISHED_OUTPUT_DIR if mode == RUN_MODE_PUBLISHED else SMOKE_OUTPUT_DIR))

    t_start = time.time()
    partial: dict = {}
    session: dict = {"owns_checkpoint": False}
    try:
        results = run_all(seeds, args.limit, mode=mode,
                          allow_cpu=args.allow_cpu_smoke,
                          batches=batches, chunks=chunks, rho=args.rho,
                          mu_artifact=args.mu_artifact, resume=args.resume,
                          paths=paths, sink=partial, session=session,
                          verify_archive=args.verify_archive)
    except SystemExit:
        # A refusal of the INVOCATION (no CUDA, an unregistered 'published'
        # run, a destination collision). Nothing was measured or read-modified,
        # and writing anything would leave a file that looks like a result.
        raise
    except BaseException as exc:
        if not session.get("owns_checkpoint"):
            # START-UP FAILURE: model/tokenizer load, an invalid path, the
            # liveness probe, the service-rate derivation, unverifiable source,
            # a refused resume, an interrupt. The checkpoint -- if there is
            # one -- belongs to an earlier session and is left byte-for-byte
            # as it was; the failure is recorded in its own file.
            where = record_preflight_failure(paths, exc, resume=args.resume,
                                             mode=mode)
            label = ("--resume REFUSED" if isinstance(exc, ResumeMismatchError)
                     else f"START-UP FAILED ({type(exc).__name__})")
            print(f"\n{label}. The checkpoint was NOT modified."
                  + (f" Recorded in {_rel(where)}." if where else "")
                  + f"\n{exc}", flush=True)
            raise
        # Classified by RECORDED STATE, not only by the exception that happened
        # to propagate: a gate that fired stays terminal even if an evidence
        # write or an interrupt raised something else afterwards.
        tf = partial.get("terminal_failure") or session.get("terminal_failure")
        terminal = bool(tf) or any(
            isinstance(e, DegenerateRewardError)
            for e in (exc, exc.__context__, exc.__cause__))
        invalid = isinstance(exc, InvalidFreshEvidenceError) or bool(
            partial.get("invalid_fresh_evidence"))
        status = (STATUS_TERMINAL_GATE if terminal else
                  STATUS_INVALID_EVIDENCE if invalid else STATUS_OPERATIONAL)
        if tf:
            partial["terminal_failure"] = tf
        if invalid and not partial.get("invalid_fresh_evidence"):
            where = getattr(exc, "evidence_file", None)
            partial["invalid_fresh_evidence"] = {
                "status": STATUS_INVALID_EVIDENCE,
                "unit": getattr(exc, "unit_id", None),
                "problems": getattr(exc, "problems", None),
                "evidence_file": _rel(where) if where else None,
                "resume_policy": ("--resume refuses until the defect is "
                                  "investigated; the unit is not re-measured"),
            }
        partial["aborted"] = {
            "status": status,
            "reason": type(exc).__name__,
            "message": str(exc)[:4000],
            "at_utc": _now(),
            "completed_configs": list(partial.get("configs", {})),
            "interpretation": {
                STATUS_TERMINAL_GATE: (
                    "A reward gate fired: the reward could not distinguish "
                    "configurations at this operating point. That is a RESULT "
                    "about this operating point, not a crash. It is TERMINAL: "
                    "--resume refuses this run. Do not re-run with a lowered "
                    "threshold; report the degeneracy."),
                STATUS_INVALID_EVIDENCE: (
                    "Freshly measured evidence failed the validation a resume "
                    "applies. A validity defect to investigate, not an "
                    "interruption: --resume refuses, and the unit is not "
                    "re-measured. NOT a reportable result."),
                STATUS_OPERATIONAL: (
                    "The run was interrupted. Completed units are on disk with "
                    "their evidence; --resume continues from them. NOT a "
                    "reportable result."),
            }[status],
        }
        if terminal and type(exc).__name__ != "DegenerateRewardError":
            partial["aborted"]["note"] = (
                "the propagating exception is not the gate's own; the run is "
                "terminal because the gate's verdict was recorded first")
        partial["run_status"] = status
        ledger = session.get("ledger")
        if ledger is not None:
            partial["checkpoint"] = ledger.as_dict()
        history = partial.get("run_history") or []
        if history:
            history[-1].update(status=status, ended_utc=_now(),
                               aborted=partial["aborted"])
        partial["elapsed_s"] = time.time() - t_start
        if not (isinstance(partial.get("fingerprint"), dict)
                and isinstance(partial.get("checkpoint"), dict)):
            # Never replace a checkpoint with a record that lacks its identity:
            # that would make the run unresumable. Record the abort aside.
            where = record_preflight_failure(paths, exc, resume=args.resume,
                                             mode=mode)
            print(f"\nABORTED ({status}) before the checkpoint carried its "
                  "identity; the checkpoint was NOT modified."
                  + (f" Recorded in {_rel(where)}." if where else ""), flush=True)
            raise
        try:
            write_artifact(partial, seeds, args.limit, paths)
            print(f"\nABORTED ({status}: {type(exc).__name__}). Checkpoint + "
                  f"reason written to {_rel(paths.results)}", flush=True)
        except Exception as wexc:
            print(f"\nABORTED ({status}); the abort record could NOT be written "
                  f"({wexc!r}). The previous checkpoint is intact.", flush=True)
            if terminal and not (tf or {}).get("persisted_on_disk"):
                print("WARNING: a reward gate fired and NO record of it could "
                      "be written. A later --resume cannot know; do NOT resume "
                      "this run.", flush=True)
        raise

    results["elapsed_s"] = time.time() - t_start
    results["run_status"] = "complete"
    write_artifact(results, seeds, args.limit, paths)
    _print(results)
    print(f"\nSaved to {_rel(paths.results)}", flush=True)
    print(f"Per-seed + per-cycle + raw arrivals in {_rel(paths.raw_dir)}",
          flush=True)


if __name__ == "__main__":
    main()
