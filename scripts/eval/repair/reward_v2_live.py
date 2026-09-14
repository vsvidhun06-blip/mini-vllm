"""THE REWARD-v2 LIVE EXPERIMENT -- does a non-degenerate reward change anything?

SCIENTIFIC OBJECTIVE
====================
Isolate the effect of restoring a NON-DEGENERATE REWARD to the live repaired
controller, holding every other experimental factor fixed.

This experiment exists because of one line in the 2026-09-02 T4 artifact:

    "per_arm_mean_reward": {"0": 0.29999999999999993, "1": 0.29999999999999993, ...}

Every arm, in every regime, scored exactly 0.3. That is reward v1 fully
saturated (`0.3 = 0.3*1.0 + 0.3*0 + 0.2*0 + 0.2*0`): throughput clipped at the
50 tok/s reference against ~96 measured, TTFT and TPOT violated 100% of the time
against 200 ms / 50 ms SLOs, and no prefix reuse. `src/carl/controller.py`
imported `utility` from `src.carl.bandit` (v1); `utility_v2` existed only inside
`scripts/eval/repair/*`. The repair reached the simulator and never reached the
controller.

Consequence: in that run EVERY learner was blind. The static-vs-adaptive result
stands (Static-Best is a fixed configuration and does not consume a reward), but
the sentence the paper needs -- "we repaired the learner and it STILL ties" --
was not tested on hardware, because the repaired learner received a constant.

THE CONTROLLED PAIR
===================
    Run A  reward v1  docs/eval/ablation_live_results.json     (2026-09-02, T4)
    Run B  reward v2  this script                              (to be run, T4)

Held IDENTICAL by construction, by calling Run A's own code rather than
reimplementing it (`ablation_live._build_workload`, `._new_scheduler`,
`._serve`, `._frozen_arms`, `.select_static_best`, `.OBSERVE_INTERVAL`):

    hardware              Tesla T4
    model                 TinyLlama-1.1B-Chat-v1.0, fp16
    seeds                 42..51 (10 runs)
    requests              200 per run
    workload construction NON-STATIONARY, regime flip at n//2, burst arrivals
    controller cadence    OBSERVE_INTERVAL = 10 scheduler steps
    arm sets              restricted / expanded, byte-identical definitions
    learner               RepairedLinUCBBandit, alpha = 0.5
    static search         LHS, 16 candidates, wide space, validation seed 999
    context vector        FEATURE_DIM = 10, untouched

CHANGED, deliberately, and nothing else:

    reward                v1 (bandit.utility, saturating)
                       -> v2 (reward.utility_v2, frozen calibrated scales)

REWARD SCALES ARE CALIBRATED ON HELD-OUT DATA, THEN FROZEN
==========================================================
The scales are NOT derived from the evaluation seeds. Doing so would let the
yardstick move with the thing being measured: `t_half` set to the median of the
test runs' own throughput puts the mid-point at 0.5 by construction, which
manufactures discrimination rather than measuring it.

Instead, a dedicated CALIBRATION SWEEP runs first, on the held-out validation
seed 999 (the same seed the Static-Best search has always used, and which is
documented as never used for an evaluation run), at the SAME request count as
evaluation. It sweeps `max_batch_size` across the validated batch axis
{2,4,8,12,16,24,32} -- the axis the B7 load sweep showed spans the achievable
operating range -- and records throughput, TTFT p99 and TPOT p99 for each. The
scales are the medians of those observations, frozen before a single evaluation
run starts, and written into the artifact verbatim.

Calibrating at the SAME n as evaluation matters: under a burst arrival the TTFT
p99 scales with how many requests are queued, so scales fitted at n=100 would be
wrong by roughly a factor of two at n=200.

THE HARD GATE
=============
Two gates, both of which ABORT rather than warn (see `src/carl/live_reward.py`
for the thresholds and their provenance):

  1. PRE-RUN, on the calibration sweep. If the frozen scales do not separate the
     candidate configurations by at least MIN_ARM_REWARD_SPREAD, the reward
     cannot distinguish the operating points the controller will choose among,
     and no learning result could mean anything. Abort BEFORE spending GPU time.
  2. PER-RUN, on each controller's logged reward stream. Catches scales that
     discriminate across the calibration sweep but collapse at the operating
     point one run happens to reach.

Gate 2 aborts for EVERY controller-driven config, RuleOnly included. RuleOnly
ignores the reward, so a degenerate stream would not corrupt its policy -- but
the reward stream is a property of the substrate and the frozen scales, not of
the policy, so degeneracy anywhere is evidence the scales do not discriminate at
this operating point, which invalidates the learners' runs too.

CONFIGS -- deliberately only four
=================================
    Static-Best     the tuned fixed configuration (no controller)
    RuleOnly        classify -> DEFAULT_CONFIGS[regime]; no learning
    CARL-Repaired   RepairedLinUCB + restricted arms
    CARL-Expanded   RepairedLinUCB + expanded arms

The five inactive-subsystem ablations (NoSpec / NoCache / NoRouter, and their
NoSched / NoChunk siblings) are NOT re-run. Run A's own `scope_note` records
that speculation is pinned off, no router is present and KV eviction never
triggers here, so three of them measure CARL-Full by design; re-running them
would spend GPU time reproducing a known no-op. CARL-Full is not re-run either:
it is the AS-PUBLISHED learner, which is provably inert
(`test_controllers.py::test_as_published_linucb_locks_on_arm_zero`), so it
cannot benefit from a better reward and is not part of this question.

THE THREE COMPARISONS
=====================
    primary          CARL-Repaired vs Static-Best     does adaptation pay?
    learning         CARL-Repaired vs RuleOnly        what did LEARNING buy?
    action space     CARL-Expanded vs CARL-Repaired   what did WIDER ARMS buy?

Hypotheses H1-H3 are pre-registered in
`docs/eval/PREREGISTRATION_reward_v2_live.md` and copied verbatim into the
artifact under `pre_registration`, before the run.

SUPERSEDED -- KEPT DELIBERATELY
===============================
Per-cycle inspection of this experiment's own artifact found two defects that
this script does NOT fix, because fixing them here would destroy the evidence:

  1. COLD-START OPTIMISM. In every seed the first three control cycles logged
     ttft_p99_ms = 0 and tpot_p99_ms = 0 -- no request had completed, so the
     tracker's windows were empty and `_percentile` returned its NaN-free 0.0.
     Reward v2 scores 0 ms as PERFECT latency, so those cycles earned close to
     the maximum reward and fed it to LinUCB, biasing arm 0 before any latency
     existed. This script now pins `require_valid_metrics=False` so it keeps
     reproducing that behaviour verbatim; see `run_all`.

  2. OBJECTIVE MISMATCH. Its Static-Best is selected on THROUGHPUT while CARL
     optimises utility_v2, and the calibration sweep itself shows the two
     objectives disagree. A utility comparison against a throughput-selected
     baseline is not an adaptive-control result.

The repaired experiment is `scripts/eval/repair/final_hardening.py`. It writes
to different paths and never touches this one's artifact. This script and its
artifact remain the record of how both defects were found.

Run:
    python scripts/eval/repair/reward_v2_live.py
    python scripts/eval/repair/reward_v2_live.py --seeds 42 --limit 20 \\
        --allow-cpu-smoke        # wiring smoke only; never a result
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

# Run A's own harness. Imported, never reimplemented -- that is what makes the
# two runs a controlled pair rather than two similar experiments.
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

EXPERIMENT = "reward_v2_live"
DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "reward_v2_live")
RESULTS_PATH = os.path.join(DOCS_EVAL, "reward_v2_live_results.json")
PREREG_PATH = os.path.join(DOCS_EVAL, "PREREGISTRATION_reward_v2_live.md")

# Held fixed against Run A. Read from ablation_live so they cannot drift.
DEFAULT_SEEDS = list(abl.DEFAULT_SEEDS)              # 42..51
DEFAULT_REQUESTS = 200                               # Run A used --limit 200
VALIDATION_SEED = abl.VALIDATION_SEED                # 999, held out
OBSERVE_INTERVAL = abl.OBSERVE_INTERVAL              # 10
CALIBRATION_BATCH_AXIS = list(abl.ARM_SET_BATCH_AXIS)  # {2,4,8,12,16,24,32}
ALPHA = 0.5

CONFIGS = ["Static-Best", "RuleOnly", "CARL-Repaired", "CARL-Expanded"]
_CONTROLLER_CONFIGS = {"RuleOnly", "CARL-Repaired", "CARL-Expanded"}
_ARM_SET_FOR = {"CARL-Repaired": "restricted", "CARL-Expanded": "expanded",
                "RuleOnly": "restricted"}

COMPARISONS = [
    ("primary", "CARL-Repaired", "Static-Best",
     "Does online adaptation pay against a tuned fixed configuration?"),
    ("learning_isolation", "CARL-Repaired", "RuleOnly",
     "What did LEARNING buy, holding the classifier and arm set fixed?"),
    ("action_space_isolation", "CARL-Expanded", "CARL-Repaired",
     "What did a WIDER ACTION SPACE buy, holding the learner fixed?"),
]

PRE_REGISTRATION = {
    "registered_before_run": True,
    "document": "docs/eval/PREREGISTRATION_reward_v2_live.md",
    "H1": {
        "statement": (
            "CARL-Repaired will not materially outperform Static-Best at this "
            "saturated operating point, because the calibrated substrate "
            "predicts max_batch_size=32 as the throughput optimum and the "
            "Static-Best search selects it."),
        "primary_comparison": "CARL-Repaired vs Static-Best",
        "prediction": "throughput delta <= 0, or within noise of 0",
        "falsified_if": (
            "CARL-Repaired exceeds Static-Best throughput by more than 2% with "
            "a paired 95% CI excluding 0."),
    },
    "H2": {
        "statement": (
            "With a non-degenerate reward, CARL-Repaired should differ "
            "measurably from RuleOnly if the learner extracts useful "
            "information. Under reward v1 the two were behaviourally "
            "indistinguishable because the reward was constant."),
        "primary_comparison": "CARL-Repaired vs RuleOnly",
        "prediction": (
            "arm-change count and played-config histogram differ from "
            "RuleOnly's; the throughput delta may be positive, zero or "
            "negative -- H2 is about whether learning DOES anything, not "
            "whether it HELPS."),
        "falsified_if": (
            "CARL-Repaired is bit-identical to RuleOnly on every seed, which "
            "would mean the repaired learner is still inert on hardware."),
    },
    "H3": {
        "statement": (
            "CARL-Expanded will not recover enough benefit to offset its "
            "increased exploration cost."),
        "primary_comparison": "CARL-Expanded vs CARL-Repaired",
        "prediction": "throughput delta <= 0",
        "falsified_if": (
            "CARL-Expanded exceeds CARL-Repaired with a paired 95% CI "
            "excluding 0."),
    },
    "note": (
        "H1 and H3 predict null/negative results and H2 predicts a behavioural "
        "difference of unspecified sign. Registering them before the run is "
        "what makes a null result evidence rather than an absence of evidence."),
}


# ===========================================================================
# Calibration -- held out, explicit, frozen.
# ===========================================================================


def calibrate_reward_scales(model, tokenizer, n: int, seed: int) -> tuple:
    """Sweep the validated batch axis on the HELD-OUT seed; freeze the scales.

    Returns (RewardScales, calibration_record). The record carries every
    observation the scales were derived from, so a reader can recompute them.
    """
    print(f"\n[calibration] {len(CALIBRATION_BATCH_AXIS)} configs x {n} requests "
          f"on HELD-OUT seed {seed} (never an evaluation seed)", flush=True)
    rows = []
    for mb in CALIBRATION_BATCH_AXIS:
        cfg = CARLConfig(max_batch_size=mb).clamp()
        specs = abl._build_workload(tokenizer, "NON-STATIONARY", n,
                                    random.Random(seed))
        sched = abl._new_scheduler(model)
        abl._apply_sched(sched, cfg)
        m = abl._serve(sched, specs)
        rows.append({
            "max_batch_size": mb,
            "config": cfg.as_dict(),
            "throughput_tps": m["throughput_tps"],
            "ttft_p50_ms": m["ttft_p50"],
            "ttft_p99_ms": m["ttft_p99"],
            "tpot_p50_ms": m["tpot_p50"],
            "tpot_p99_ms": m["tpot_p99"],
            "wall_s": m["wall_s"],
        })
        print(f"  mb={mb:2d} -> {m['throughput_tps']:7.2f} tok/s  "
              f"ttftP99={m['ttft_p99']:9.1f}ms  tpotP99={m['tpot_p99']:7.1f}ms",
              flush=True)

    scales = RewardScales.from_measurements(
        [r["throughput_tps"] for r in rows],
        [r["ttft_p99_ms"] for r in rows],
        [r["tpot_p99_ms"] for r in rows],
    )
    record = {
        "method": "median of a held-out batch-axis sweep, frozen before evaluation",
        "held_out_seed": seed,
        "requests_per_calibration_run": n,
        "batch_axis": CALIBRATION_BATCH_AXIS,
        "why_same_n_as_evaluation": (
            "Under a burst arrival the TTFT p99 scales with the number of queued "
            "requests, so scales fitted at a smaller n would mis-set ttft_target."),
        "why_held_out": (
            "Seed 999 is documented in ablation_live as never used for an "
            "evaluation run. Deriving scales from the evaluation seeds would set "
            "the reward's mid-point from the data being measured."),
        "observations": rows,
        "frozen_scales": {
            "t_half": scales.t_half,
            "ttft_target": scales.ttft_target,
            "tpot_target": scales.tpot_target,
            "sharpness": scales.sharpness,
        },
        "weights": dict(DEFAULT_WEIGHTS),
    }
    print(f"[calibration] FROZEN scales: t_half={scales.t_half:.4f} "
          f"ttft_target={scales.ttft_target:.4f} tpot_target={scales.tpot_target:.4f} "
          f"sharpness={scales.sharpness}", flush=True)
    return scales, record


def gate_calibration(scales: RewardScales, record: dict) -> dict:
    """PRE-RUN GATE. Score every calibration config under the frozen scales."""
    arm_rewards = {}
    for r in record["observations"]:
        arm_rewards[f"max_batch_size={r['max_batch_size']}"] = utility_v2(
            {"throughput_tps": r["throughput_tps"],
             "ttft_p99_ms": r["ttft_p99_ms"],
             "tpot_p99_ms": r["tpot_p99_ms"],
             "cache_hit_rate": 0.0},
            DEFAULT_WEIGHTS, scales)
    report = gate_candidate_arm_spread(
        arm_rewards, context="reward_v2_live calibration sweep",
        threshold=MIN_ARM_REWARD_SPREAD)
    print(f"[gate:pre-run] candidate-arm reward spread "
          f"{report['reward_spread']:.6f} >= {MIN_ARM_REWARD_SPREAD} -- PASS",
          flush=True)
    return report


# ===========================================================================
# One evaluation run.
# ===========================================================================


def _build_bandit(name: str):
    """The policy for `name`, from ablation_live's own arm-set definitions."""
    arm_set = _ARM_SET_FOR[name]
    arms = abl._frozen_arms(None, arm_set)
    if name == "RuleOnly":
        return RuleOnlyBandit(arms, d=FEATURE_DIM)
    return PerRegimeBandit(arms, d=FEATURE_DIM,
                           bandit_cls=RepairedLinUCBBandit, alpha=ALPHA)


def _cycle_records(controller, t0_note: str = "") -> list:
    """Per-control-cycle rows: every field the analysis needs, nothing derived.

    `reward` at row t scores the config in row t-1 (delayed-reward timing), which
    is why `rewarded_arm` is recorded alongside `arm` instead of leaving a reader
    to re-derive the attribution.
    """
    out = []
    for e in controller.controller_log:
        terms = dict(e.reward_terms)
        raw = terms.pop("raw_metrics", None)
        out.append({
            "step": e.step,
            "t_monotonic_s": e.t_monotonic_s,
            "regime": e.regime.value,
            "selected_arm": e.arm,
            "rewarded_arm": e.rewarded_arm,
            "config": e.config.as_dict(),
            "context": e.state_features,
            "reward": e.reward,
            "reward_terms": terms,
            "reward_raw_metrics": raw,
            "observed": e.observed,
        })
    return out


def run_one(name: str, model, tokenizer, n: int, seed: int, *,
            static_cfg=None, reward_fn=None) -> dict:
    """Serve one configuration once. Same workload construction as Run A."""
    specs = abl._build_workload(tokenizer, "NON-STATIONARY", n,
                                random.Random(seed))
    sched = abl._new_scheduler(model)
    controller = tracker = None

    if name == "Static-Best":
        abl._apply_sched(sched, static_cfg or CARLConfig())
    else:
        tracker = MetricsTracker(window=max(50, n))
        controller = CARLController(
            scheduler=sched, bandit=_build_bandit(name),
            observe_interval=OBSERVE_INTERVAL, slo=abl._SLO, metrics=tracker,
            reward_fn=reward_fn)

    out = abl._serve(sched, specs, controller=controller, tracker=tracker)

    if controller is not None:
        cycles = _cycle_records(controller)
        out["cycles"] = cycles
        out["arm_set"] = _ARM_SET_FOR[name]
        out["bandit_cls"] = type(controller.bandit).__name__
        out["controller_stats"] = controller.stats()
        arms_played = [c["selected_arm"] for c in cycles]
        out["arm_changes"] = sum(1 for a, b in zip(arms_played, arms_played[1:])
                                 if a != b)
        out["arm_histogram"] = {
            f"{c['regime']}|arm{c['selected_arm']}": 0 for c in cycles}
        for c in cycles:
            out["arm_histogram"][f"{c['regime']}|arm{c['selected_arm']}"] += 1
        # PER-RUN GATE. Aborts the experiment; see the module docstring.
        out["reward_gate"] = gate_live_reward_stream(
            [c["reward"] for c in cycles],
            context=f"reward_v2_live {name} seed={seed}",
            threshold=MIN_LIVE_REWARD_SPREAD, min_distinct=MIN_LIVE_DISTINCT)
        out["mean_reward"] = statistics.fmean(c["reward"] for c in cycles)
    return out


# ===========================================================================
# Aggregation.
# ===========================================================================

_METRICS = ["throughput_tps", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
            "slo_rate"]


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _paired(a_vals: list, b_vals: list) -> dict:
    """Paired difference a - b over matched seeds, with a 95% CI and Cohen's d.

    `all_diffs_zero` is reported explicitly so a bit-identical IDENTITY is never
    mistaken for a statistical tie -- the distinction the reward-v1 run could not
    make, and the one H2 turns on.
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
        "requests": run["requests"],
        "step_log": run.get("step_log"),
        "shift_t": run.get("shift_t"), "shift_step": run.get("shift_step"),
        "cycles": run.get("cycles"),
        "reward_gate": run.get("reward_gate"),
        "arm_changes": run.get("arm_changes"),
        "arm_histogram": run.get("arm_histogram"),
        "controller_stats": run.get("controller_stats"),
        "decision_us": run.get("decision_us"),
    }
    with open(os.path.join(RAW_DIR, f"{name}_seed{seed:03d}.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


# ===========================================================================
# Driver.
# ===========================================================================


def _provenance_extra(seeds: list, n: int, results: dict) -> dict:
    """Everything run-identifying that is not already in `provenance.capture`."""
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
            "method": f"latin_hypercube_{abl.N_LHS_CANDIDATES}_candidates",
            "space_name": "wide",
            "space": abl.SEARCH_SPACE_WIDE,
            "validation_seed": VALIDATION_SEED,
        },
        "frozen_reward_scales": results.get("reward", {}).get("scales"),
        "reward_weights": results.get("reward", {}).get("weights"),
    }


def write_artifact(results: dict, seeds: list, n: int) -> str:
    """Stamp with provenance and write. Called after EVERY config, not once.

    WHY INCREMENTAL. Two artifacts in this repository were lost to a Colab VM
    torn down before the JSON was downloaded, and were later reconstructed by
    hand -- which is why they are quarantined. `batch_intervention.py` writes
    after every row for the same reason. A gate firing on the last config, or a
    disconnect, must not destroy the GPU hours already spent.
    """
    return provenance.write_result(
        RESULTS_PATH, results, EXPERIMENT,
        script="scripts/eval/repair/reward_v2_live.py",
        extra=_provenance_extra(seeds, n, results))


def run_all(seeds: list, n: int, allow_cpu: bool = False,
            sink: dict | None = None) -> dict:
    if DEVICE.type != "cuda" and not allow_cpu:
        raise SystemExit(
            "reward_v2_live REFUSES to run without CUDA. This experiment exists "
            "to answer a hardware question and a CPU number would not answer it. "
            "Pass --allow-cpu-smoke for a wiring smoke test (never a result).")

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {len(seeds)} runs x {n} requests "
          f"| seeds {seeds}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    # 1. CALIBRATE on held-out data, FREEZE, then GATE. Before any eval run.
    scales, calibration = calibrate_reward_scales(model, tokenizer, n,
                                                  VALIDATION_SEED)
    calibration["gate"] = gate_calibration(scales, calibration)
    # require_valid_metrics=False PINS THE PRE-REPAIR COLD-START BEHAVIOUR.
    #
    # This experiment's artifact is the evidence that reward v2 awarded full
    # TTFT/TPOT credit for latencies that had never been measured: in every seed
    # its first three cycles logged ttft_p99_ms = 0 and tpot_p99_ms = 0, and
    # `latency_term` scores 0 ms as 1.0. `LiveRewardV2` now DEFAULTS to refusing
    # to score an unmeasured metric, so leaving this line as `LiveRewardV2(scales)`
    # would silently change what this script produces and destroy the ability to
    # reproduce the artifact that exposed the bug.
    #
    # The flag is therefore explicit and deliberate here, and ONLY here. The
    # repaired experiment is `scripts/eval/repair/final_hardening.py`, which
    # takes the default. Do not copy this line.
    reward_fn = LiveRewardV2(scales, require_valid_metrics=False)

    # 2. Static-Best, by Run A's own search, on the same wide space.
    static_cfg, selection = abl.select_static_best(
        model, tokenizer, max(10, n // 2),
        space=abl.SEARCH_SPACES["wide"], space_name="wide")

    results: dict = {
        "objective": (
            "Isolate the effect of restoring a non-degenerate reward to the "
            "live repaired controller, holding all other factors fixed."),
        "controlled_pair": {
            "run_a": {
                "artifact": "docs/eval/ablation_live_results.json",
                "reward_version": "v1",
                "note": ("bandit.utility; saturated to a constant 0.3 on every "
                         "arm in every regime on 2026-09-02"),
            },
            "run_b": {"artifact": os.path.relpath(RESULTS_PATH, _REPO_ROOT),
                      "reward_version": "v2"},
            "changed": "the reward, and nothing else",
            "held_fixed": [
                "Tesla T4", "TinyLlama-1.1B-Chat-v1.0 fp16", "seeds 42..51",
                f"{n} requests per run", "workload construction (ablation_live)",
                f"controller cadence OBSERVE_INTERVAL={OBSERVE_INTERVAL}",
                "restricted/expanded arm-set definitions",
                "RepairedLinUCBBandit, alpha=0.5",
                "Static-Best LHS search, 16 candidates, wide space, seed 999",
                f"context vector FEATURE_DIM={FEATURE_DIM}",
            ],
        },
        "pre_registration": PRE_REGISTRATION,
        "scenario": abl.scenario_description(n),
        "seeds": seeds, "runs": len(seeds), "requests": n,
        "arrival_mode": "burst (bulk dump; ablation_live default)",
        "observe_interval": OBSERVE_INTERVAL,
        "validation_seed": VALIDATION_SEED,
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
        "static_best_selection": selection,
        "configs_run": list(CONFIGS),
        "config_arm_sets": dict(_ARM_SET_FOR),
        "config_policy": {
            "Static-Best": "fixed configuration, no controller",
            "RuleOnly": "src.carl.rule_only.RuleOnlyBandit (no learning)",
            "CARL-Repaired": "RepairedLinUCBBandit + restricted arms",
            "CARL-Expanded": "RepairedLinUCBBandit + expanded arms",
        },
        "arm_sets": {
            "restricted": abl.arm_set_summary(abl.all_arm_sets()),
            "expanded": abl.arm_set_summary(abl.expanded_arm_sets()),
            "expanded_batch_axis": abl.ARM_SET_BATCH_AXIS,
        },
        "not_run": {
            "CARL-Full": ("as-published LinUCB is provably inert (200/200 on "
                          "arm 0); a better reward cannot help an inert learner "
                          "and it is not part of this question"),
            "CARL-NoSpec/NoCache/NoRouter": (
                "measure CARL-Full by design in this harness -- speculation "
                "pinned off, no router, KV eviction inactive"),
            "CARL-NoSched/NoChunk": "knob-freeze ablations, not a reward question",
            "AutoTuner": "not a reward-driven controller",
            "DynOracle": ("meaningless under reward v1 (all arms tied at 0.3) "
                          "and not required by any of the three comparisons"),
        },
        "scope_note": (
            "ONE operating point: burst arrivals at rho >> 1 on a single T4 with "
            "one 1.1B model. This experiment resolves the reward-version "
            "confound in Run A; it does NOT establish a load envelope."),
        "configs": {},
    }
    if sink is not None:
        # Same object, so an abort in main() sees everything written so far.
        sink.clear()
        sink.update(results)
        results = sink

    per_seed_tput: dict = {}
    for name in CONFIGS:
        per_run = []
        for i, seed in enumerate(seeds):
            run = run_one(name, model, tokenizer, n, seed,
                          static_cfg=static_cfg if name == "Static-Best" else None,
                          reward_fn=reward_fn if name in _CONTROLLER_CONFIGS else None)
            per_run.append(run)
            _save_raw(name, seed, run)
            extra = ""
            if "mean_reward" in run:
                extra = (f" | reward mean={run['mean_reward']:.4f} "
                         f"spread={run['reward_gate']['reward_spread']:.4f} "
                         f"arm_changes={run['arm_changes']}")
            print(f"  {name:<14} {i+1}/{len(seeds)} (seed {seed}): "
                  f"{run['throughput_tps']:7.2f} tok/s "
                  f"ttftP99={run['ttft_p99']:9.1f}ms{extra}", flush=True)

        agg = {"arm_set": _ARM_SET_FOR.get(name),
               "policy": results["config_policy"][name]}
        for key in _METRICS:
            mean, std = _mean_std([m[key] for m in per_run])
            agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
        agg["_per_seed_throughput_tps"] = [m["throughput_tps"] for m in per_run]
        if name in _CONTROLLER_CONFIGS:
            agg["mean_reward"] = statistics.fmean(m["mean_reward"] for m in per_run)
            agg["arm_changes_mean"] = statistics.fmean(m["arm_changes"] for m in per_run)
            agg["reward_spread_min"] = min(m["reward_gate"]["reward_spread"]
                                           for m in per_run)
            agg["reward_distinct_min"] = min(m["reward_gate"]["n_distinct"]
                                             for m in per_run)
            agg["seeds_distinct"] = len({tuple(
                round(c["reward"], 12) for c in m["cycles"]) for m in per_run}) > 1
        results["configs"][name] = agg
        per_seed_tput[name] = agg["_per_seed_throughput_tps"]
        results["completed_configs"] = list(results["configs"])
        write_artifact(results, seeds, n)      # checkpoint after every config

    results["comparisons"] = {
        label: {"a": a, "b": b, "question": q,
                "throughput_tps": _paired(per_seed_tput[a], per_seed_tput[b])}
        for label, a, b, q in COMPARISONS
        if a in per_seed_tput and b in per_seed_tput
    }
    return results


def _print(results: dict) -> None:
    print("\n=== REWARD-v2 LIVE (mean +/- std over "
          f"{results['runs']} seeds, {results['requests']} requests) ===")
    hdr = ["config", "tok/s", "ttftP99 ms", "tpotP99 ms", "reward", "armChg"]
    print("| " + " | ".join(hdr) + " |")
    print("| " + " | ".join("---" for _ in hdr) + " |")
    for name in CONFIGS:
        a = results["configs"].get(name)
        if not a:
            continue
        print("| " + " | ".join([
            name,
            f"{a['throughput_tps_mean']:.2f} +/- {a['throughput_tps_std']:.2f}",
            f"{a['ttft_p99_mean']:.1f}", f"{a['tpot_p99_mean']:.1f}",
            f"{a.get('mean_reward', float('nan')):.4f}",
            f"{a.get('arm_changes_mean', float('nan')):.1f}",
        ]) + " |")
    print("\n=== Comparisons (paired, throughput tok/s) ===")
    for label, c in results.get("comparisons", {}).items():
        t = c["throughput_tps"]
        print(f"  {label:<24} {c['a']} - {c['b']}: "
              f"{t['mean_difference']:+.3f} tok/s "
              f"CI95 [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] "
              f"d={t['cohens_d_paired']} all_zero={t['all_diffs_zero']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reward-v2 live experiment (controlled pair against the "
                    "reward-v1 T4 run).")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--limit", type=int, default=DEFAULT_REQUESTS,
                    help="requests per run (Run A used 200)")
    ap.add_argument("--allow-cpu-smoke", action="store_true",
                    help="permit a CPU run for WIRING VERIFICATION ONLY; the "
                         "artifact is written with a cpu_smoke marker and is "
                         "never a result")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    t_start = time.time()
    # `partial` is filled in by run_all as it goes, so an abort still has
    # something to write. A gate that fires on the last config must not destroy
    # the GPU hours already spent -- and the FAILURE ITSELF is a result worth
    # keeping, since a degenerate reward is exactly what this run is testing for.
    partial: dict = {}
    try:
        results = run_all(seeds, args.limit, allow_cpu=args.allow_cpu_smoke,
                          sink=partial)
    except DegenerateRewardError as exc:
        partial["aborted"] = {
            "reason": "DegenerateRewardError",
            "message": str(exc),
            "completed_configs": list(partial.get("configs", {})),
            "interpretation": (
                "A gate fired: the reward could not distinguish configurations. "
                "This is a RESULT, not a crash -- it says the frozen scales do "
                "not discriminate at this operating point. Do not re-run with a "
                "lowered threshold; re-derive the scales, or report the "
                "degeneracy."),
        }
        partial["elapsed_s"] = time.time() - t_start
        write_artifact(partial, seeds, args.limit)
        print(f"\nABORTED (gate fired). Partial artifact + reason written to "
              f"{RESULTS_PATH}", flush=True)
        raise
    except BaseException as exc:
        # Any other failure -- OOM, disconnect, KeyboardInterrupt -- must still
        # leave the completed configs on disk. Seeds are NOT skipped and the run
        # is NOT continued: the paired comparisons require complete seed sets,
        # so a partial config is evidence to inspect, never a result to report.
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
