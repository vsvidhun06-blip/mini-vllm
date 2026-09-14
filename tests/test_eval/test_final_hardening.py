"""The final hardening experiment: is it actually the experiment it claims to be?

These tests are structural. They do not run the GPU experiment -- they check the
properties that determine whether its OUTPUT would mean anything:

  * the static search covers the dimensions that execute, exhaustively, and
    excludes only dimensions proven inert;
  * two static baselines exist and are ranked by two DIFFERENT objectives;
  * the reward scales are frozen from held-out measurements BEFORE the utility
    baseline is selected from them;
  * the cold-start repair is switched ON here and pinned OFF in the superseded
    experiment, so that artifact stays reproducible;
  * a convergence claim requires exploration followed by exploitation;
  * every factor said to be held fixed is READ from the earlier harness rather
    than retyped.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_DIR = os.path.join(_REPO_ROOT, "scripts", "eval")
for _p in (_REPO_ROOT, _EVAL_DIR, os.path.join(_EVAL_DIR, "repair")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch", reason="final_hardening imports torch at module level")

import ablation_live as abl  # noqa: E402
import final_hardening as fh  # noqa: E402
import reward_v2_live as rv2  # noqa: E402

from src.carl.config import CARLConfig  # noqa: E402
from src.carl.reward import DEFAULT_WEIGHTS, RewardScales, utility_v2  # noqa: E402
from src.carl.state import FEATURE_DIM  # noqa: E402

SCALES = RewardScales(t_half=96.0, ttft_target=60000.0, tpot_target=300.0)


def _row(mb, cs, tps, ttft, tpot):
    return {"max_batch_size": mb, "chunk_size": cs,
            "config": CARLConfig(max_batch_size=mb, chunk_size=cs).clamp().as_dict(),
            "throughput_tps": tps, "ttft_p50_ms": ttft / 3, "ttft_p99_ms": ttft,
            "tpot_p50_ms": tpot / 2, "tpot_p99_ms": tpot, "slo_rate": 0.0,
            "wall_s": 100.0}


# ---------------------------------------------------------------------------
# ISSUE 1 -- the objective mismatch.
# ---------------------------------------------------------------------------


def test_there_are_two_static_baselines_not_one():
    assert "Static-Tput" in fh.CONFIGS
    assert "Static-Utility" in fh.CONFIGS
    assert "Static-Best" not in fh.CONFIGS, (
        "a single throughput-selected baseline is the defect being repaired")
    assert fh._STATIC_CONFIGS == {"Static-Tput", "Static-Utility"}


def test_the_two_baselines_are_ranked_by_different_objectives():
    """The whole point: when throughput and utility disagree, the two baselines
    must diverge. Rows are constructed so the disagreement is unambiguous -- the
    fastest config has by far the worst tail latency."""
    rows = [
        _row(32, 512, 140.0, 200000.0, 900.0),   # fastest, terrible latency
        _row(12, 256, 120.0, 30000.0, 250.0),    # best utility
        _row(2, 64, 60.0, 20000.0, 200.0),
    ]
    cfg_t, cfg_u, rec = fh.select_static_baselines(rows, SCALES)

    assert cfg_t.max_batch_size == 32
    assert cfg_u.max_batch_size == 12
    assert cfg_t != cfg_u
    assert rec["objectives_agree"] is False
    assert "objective mismatch" in rec["objective_disagreement_note"]


def test_the_baselines_coincide_when_the_objectives_agree():
    """And when they DO agree, the experiment must say so rather than
    manufacturing a difference."""
    rows = [_row(32, 512, 140.0, 30000.0, 250.0),
            _row(4, 128, 90.0, 60000.0, 400.0)]
    cfg_t, cfg_u, rec = fh.select_static_baselines(rows, SCALES)
    assert cfg_t == cfg_u
    assert rec["objectives_agree"] is True


def test_static_utility_is_ranked_by_exactly_carls_objective():
    """Not "a utility-like score" -- the same function, weights and frozen scales
    the controller is driven by. Anything else reintroduces the mismatch."""
    rows = [_row(12, 256, 120.0, 30000.0, 250.0), _row(32, 512, 140.0, 200000.0, 900.0)]
    _t, _u, rec = fh.select_static_baselines(rows, SCALES)
    for entry in rec["ranked_table"]:
        assert entry["utility_v2"] == pytest.approx(utility_v2(
            {"throughput_tps": entry["throughput_tps"],
             "ttft_p99_ms": entry["ttft_p99_ms"],
             "tpot_p99_ms": entry["tpot_p99_ms"], "cache_hit_rate": 0.0},
            DEFAULT_WEIGHTS, SCALES))


def test_the_live_plane_is_swept_exhaustively_not_sampled():
    assert fh.GRID_BATCH_AXIS == abl.ARM_SET_BATCH_AXIS == [2, 4, 8, 12, 16, 24, 32]
    assert fh.GRID_CHUNK_AXIS == abl.SEARCH_SPACE_WIDE["chunk_size"]
    assert len(fh.GRID_BATCH_AXIS) * len(fh.GRID_CHUNK_AXIS) == 28

    rows = fh.sweep_live_plane.__doc__
    assert "EXHAUSTIVE" in rows
    # And it must not have quietly become an LHS sample again.
    assert not hasattr(fh, "N_LHS_CANDIDATES")


def test_only_provably_inert_dimensions_are_excluded_from_the_search():
    """Every CARLConfig field is either searched or has a recorded reason it
    cannot change execution. A field in neither set would be an unsearched live
    dimension -- the defect this experiment exists to remove."""
    fields = set(CARLConfig().as_dict())
    covered = set(fh.LIVE_DIMENSIONS) | set(fh.INERT_DIMENSIONS)
    assert fields == covered, f"unaccounted config dimensions: {fields - covered}"
    assert all(fh.INERT_DIMENSIONS[k] for k in fh.INERT_DIMENSIONS), (
        "an inert dimension without a stated reason is an assertion, not evidence")


def test_the_liveness_claim_is_probed_against_a_real_scheduler_object():
    """A scheduler stub shaped like the live one: two live knobs, the CUDA-graph
    flag present but with no runner attached, speculation gated off."""
    class _Sched:
        max_batch_size = 8
        chunk_size = 256
        use_cuda_graphs = True
        enable_spec_decode = False
        spec_decode_k = 2
        _graph_runner = None

    report = fh.probe_live_dimensions(_Sched())
    assert report["passed"] is True
    assert report["unexpectedly_live"] == []
    assert report["per_field"]["use_cuda_graphs"]["graph_runner_attached"] is False
    assert report["per_field"]["preemption_enabled"]["scheduler_has_attribute"] is False
    assert report["grid_is_exhaustive_over"]["n_configurations"] == 28


def test_a_dimension_that_became_live_aborts_the_run():
    """The probe must FAIL LOUDLY, not annotate. If a CUDA-graph runner is
    attached, use_cuda_graphs starts changing execution and the exhaustive grid
    is no longer exhaustive -- which is precisely how an earlier T4 smoke shipped
    two 'CUDA graph' arms with zero graph hits."""
    class _SchedWithGraphs:
        max_batch_size = 8
        chunk_size = 256
        use_cuda_graphs = True
        enable_spec_decode = False
        _graph_runner = object()

    with pytest.raises(RuntimeError, match="use_cuda_graphs"):
        fh.probe_live_dimensions(_SchedWithGraphs())


def test_live_speculation_would_also_abort():
    class _SchedWithSpec:
        max_batch_size = 8
        chunk_size = 256
        use_cuda_graphs = True
        enable_spec_decode = True
        spec_k = 2
        _graph_runner = None

    with pytest.raises(RuntimeError, match="spec_k"):
        fh.probe_live_dimensions(_SchedWithSpec())


# ---------------------------------------------------------------------------
# Held-out calibration and its ordering.
# ---------------------------------------------------------------------------


def test_the_calibration_seed_is_never_an_evaluation_seed():
    assert fh.VALIDATION_SEED == abl.VALIDATION_SEED == 999
    assert fh.VALIDATION_SEED not in fh.DEFAULT_SEEDS


def test_the_scales_come_only_from_the_held_out_rows():
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0),
            _row(32, 512, 140.0, 200000.0, 900.0)]
    scales, record = fh.freeze_scales(rows, 200, 999)
    assert scales.t_half == 120.0          # median of the three throughputs
    assert scales.ttft_target == 30000.0
    assert record["held_out_seed"] == 999
    assert record["requests_per_sweep_run"] == 200
    assert record["n_observations"] == 3


def test_scales_are_frozen_before_the_utility_baseline_is_selected_from_them():
    """Ordering, not just provenance. If the utility ranking could move the
    scales, the baseline would be chosen under a yardstick it had shaped.
    `freeze_scales` reads only measured columns; `select_static_baselines` takes
    the scales as an argument and cannot return new ones."""
    import inspect
    # co_names is the set of globals/attributes the COMPILED function touches,
    # so this cannot be satisfied by a docstring the way a source-text search
    # can. `freeze_scales` must reach no utility function at all.
    touched = set(fh.freeze_scales.__code__.co_names)
    assert not touched & {"utility_v2", "run_utility", "term_breakdown"}, (
        f"the scales must not be derived from any utility: {touched}")
    assert "from_measurements" in touched

    sig = inspect.signature(fh.select_static_baselines)
    assert list(sig.parameters) == ["rows", "scales"], (
        "the ranking must CONSUME frozen scales, never produce them")

    driver = inspect.getsource(fh.run_all)
    assert driver.index("freeze_scales") < driver.index("select_static_baselines")


def test_calibration_uses_the_evaluation_request_count():
    """Under burst arrivals the TTFT p99 scales with queue depth, so scales
    fitted at a smaller n would mis-set ttft_target."""
    import inspect
    src = inspect.getsource(fh.run_all)
    assert "sweep_live_plane(model, tokenizer, n, VALIDATION_SEED" in src


# ---------------------------------------------------------------------------
# ISSUE 2 -- the cold-start repair, at the experiment level.
# ---------------------------------------------------------------------------


def test_this_experiment_switches_the_cold_start_repair_ON():
    import inspect
    src = inspect.getsource(fh.run_all)
    assert "require_valid_metrics=True" in src


def test_the_superseded_experiment_keeps_the_repair_OFF():
    """Its artifact is the evidence for the defect. Taking the new default would
    silently change what that script produces and destroy the reproduction."""
    import inspect
    src = inspect.getsource(rv2.run_all)
    assert "require_valid_metrics=False" in src


def test_the_two_experiments_write_to_different_paths():
    """The reward-v2 artifact must be preserved, so nothing here may overwrite it."""
    assert fh.RESULTS_PATH != rv2.RESULTS_PATH
    assert fh.RAW_DIR != rv2.RAW_DIR
    assert "final_hardening" in fh.RESULTS_PATH


def test_the_per_run_gate_sees_only_valid_rewards():
    """A withheld cycle carries no number. Padding the stream would make the
    gate pass or fail for a reason unrelated to whether the reward
    discriminates."""
    import inspect
    src = inspect.getsource(fh.run_one)
    assert 'valid = [c["reward"] for c in cycles if c["reward_valid"]]' in src
    assert "gate_live_reward_stream(\n        valid," in src


def test_the_cold_start_positive_control_is_pre_registered():
    """A warm-up of zero everywhere would mean the gate never engaged and the
    run says nothing about the defect. H5 makes that outcome a falsification
    rather than a quiet pass."""
    h5 = fh.PRE_REGISTRATION["H5_cold_start"]
    assert "warmup_cycles" in h5["metric"]
    assert "warmup_cycles == 0" in h5["falsified_if"]


# ---------------------------------------------------------------------------
# Convergence: claimed only when the trace shows it.
# ---------------------------------------------------------------------------


def _cycles(arms, regime="interactive"):
    return [{"regime": regime, "selected_arm": a} for a in arms]


def test_a_learner_that_never_moved_is_not_convergence():
    """The as-published LinUCB scored 200/200 on arm 0 while learning nothing.
    Any 'it settled' test would pass that trace; this one must not."""
    d = fh.convergence_diagnostic(_cycles([0] * 20))
    r = d["per_regime"]["interactive"]
    assert r["second_half_modal_share"] == 1.0
    assert r["explored"] is False
    assert r["supported"] is False
    assert "no exploration" in r["why_not"]
    assert d["supported_in_any_regime"] is False


def test_exploration_then_exploitation_is_convergence():
    d = fh.convergence_diagnostic(_cycles([0, 3, 1, 2, 4, 1, 2, 2, 2, 2, 2, 2]))
    r = d["per_regime"]["interactive"]
    assert r["explored"] is True and r["exploited"] is True
    assert r["supported"] is True


def test_continuous_thrashing_is_not_convergence():
    d = fh.convergence_diagnostic(_cycles([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))
    r = d["per_regime"]["interactive"]
    assert r["explored"] is True
    assert r["exploited"] is False
    assert r["supported"] is False
    assert "no exploitation" in r["why_not"]


def test_too_short_a_trace_cannot_support_a_convergence_claim():
    d = fh.convergence_diagnostic(_cycles([0, 1, 1, 1]))
    r = d["per_regime"]["interactive"]
    assert r["supported"] is False
    assert "fewer than" in r["why_not"]


def test_the_convergence_rule_is_fixed_before_the_run():
    """Fixed at module level and copied into the artifact, so it cannot be
    chosen after the traces are in hand."""
    assert fh.CONVERGENCE_RULE["min_decisions_in_regime"] == 6
    assert fh.CONVERGENCE_RULE["min_first_half_distinct_arms"] == 2
    assert fh.CONVERGENCE_RULE["requires_increase_in_modal_share"] is True
    assert "must not be reported as convergence" in fh.CONVERGENCE_RULE["statement"]


def test_terminal_run_length():
    assert fh._terminal_run([0, 1, 2, 2, 2]) == 3
    assert fh._terminal_run([1]) == 1
    assert fh._terminal_run([]) == 0


# ---------------------------------------------------------------------------
# What must be held fixed against the earlier runs.
# ---------------------------------------------------------------------------


def test_every_held_fixed_factor_is_read_from_the_earlier_harness():
    assert fh.DEFAULT_SEEDS == abl.DEFAULT_SEEDS == [42, 43, 44, 45, 46, 47, 48,
                                                     49, 50, 51]
    assert fh.OBSERVE_INTERVAL == abl.OBSERVE_INTERVAL == 10
    assert fh.DEFAULT_REQUESTS == rv2.DEFAULT_REQUESTS == 200
    assert fh.ALPHA == rv2.ALPHA == 0.5
    assert fh.VALIDATION_SEED == rv2.VALIDATION_SEED


def test_it_reuses_the_serving_and_workload_code_paths():
    """Importing the earlier harness rather than reimplementing it is what makes
    'held fixed' true by construction instead of by inspection."""
    import inspect
    src = inspect.getsource(fh)
    for call in ("abl._build_workload", "abl._new_scheduler", "abl._serve",
                 "abl._frozen_arms", "abl._apply_sched"):
        assert call in src, f"{call} should be reused, not reimplemented"


def test_context_dimension_is_unchanged():
    assert FEATURE_DIM == 10
    assert fh.FEATURE_DIM == FEATURE_DIM


def test_the_arm_sets_are_the_same_definitions():
    assert fh._ARM_SET_FOR == {"CARL-Repaired": "restricted",
                               "CARL-Expanded": "expanded",
                               "RuleOnly": "restricted"}
    assert fh._ARM_SET_FOR == rv2._ARM_SET_FOR


# ---------------------------------------------------------------------------
# The four comparisons, and the two utility estimators.
# ---------------------------------------------------------------------------


def test_the_four_comparisons_are_the_ones_asked_for():
    labels = {c[0]: (c[1], c[2], c[3]) for c in fh.COMPARISONS}
    assert labels["A_adaptation_vs_throughput_baseline"] == (
        "CARL-Repaired", "Static-Tput", "throughput_tps")
    assert labels["B_adaptation_vs_utility_baseline"] == (
        "CARL-Repaired", "Static-Utility", "run_utility_v2")
    assert labels["C_learning_isolation"] == (
        "CARL-Repaired", "RuleOnly", "throughput_tps")
    assert labels["D_action_space_isolation"] == (
        "CARL-Expanded", "CARL-Repaired", "throughput_tps")


def test_each_comparison_judges_a_baseline_on_the_metric_it_was_selected_for():
    """The repair, stated as an invariant: a throughput-selected baseline is
    never the primary comparator for a utility metric, or vice versa."""
    selected_on = {"Static-Tput": "throughput_tps",
                   "Static-Utility": "run_utility_v2"}
    for _label, _a, b, metric, _q in fh.COMPARISONS:
        if b in selected_on:
            assert metric == selected_on[b], (
                f"{b} was selected on {selected_on[b]} but is compared on {metric}")


def test_the_hypotheses_match_the_comparisons():
    for key in ("H1", "H2", "H3", "H4"):
        h = fh.PRE_REGISTRATION[key]
        assert h["comparison"] in {c[0][0] for c in fh.COMPARISONS}
        assert h["falsified_if"]


def test_run_utility_is_defined_for_a_static_run_too():
    """Comparison B needs a utility for a config that has no control cycles."""
    u = fh.run_utility({"throughput_tps": 120.0, "ttft_p99": 30000.0,
                        "tpot_p99": 250.0}, SCALES)
    assert 0.0 < u < sum(DEFAULT_WEIGHTS.values())


def test_the_two_utility_estimators_are_kept_distinct():
    """`mean_valid_cycle_utility` is a controller-only quantity and must never
    stand in for the run-level number the statics are compared on."""
    assert "run_utility_v2" in fh._METRICS
    assert "mean_valid_cycle_utility" not in fh._METRICS


def test_paired_reports_a_bit_identical_result_explicitly():
    """H3 turns on identity vs a statistical tie; the two must not look alike."""
    same = fh._paired([1.0, 2.0], [1.0, 2.0])
    assert same["all_diffs_zero"] is True
    assert fh._paired([1.0, 2.0], [1.0, 2.1])["all_diffs_zero"] is False


def test_arm_change_rate_normalises_by_opportunities():
    """A longer run gets more chances to change arms, so the raw count is not
    comparable across configs; the rate is."""
    import inspect
    src = inspect.getsource(fh.run_one)
    assert 'out["arm_change_rate"] = (out["arm_changes"] / (len(arms_played) - 1)' in src


# ---------------------------------------------------------------------------
# Provenance and safety rails.
# ---------------------------------------------------------------------------


def test_source_provenance_absence_is_recorded_not_nulled():
    """A null git_sha is what this whole mechanism exists to remove; an absent
    sidecar must not reintroduce one."""
    real = fh.SOURCE_PROVENANCE_PATH
    try:
        fh.SOURCE_PROVENANCE_PATH = os.path.join(_REPO_ROOT, "no_such_file.json")
        block = fh.load_source_provenance()
    finally:
        fh.SOURCE_PROVENANCE_PATH = real
    assert block["present"] is False
    assert block["why_missing"] and block["remedy"]


def test_it_refuses_to_run_without_cuda_unless_explicitly_smoked():
    import inspect
    src = inspect.getsource(fh.run_all)
    assert "REFUSES to run without CUDA" in src
    assert "allow_cpu" in src


def test_a_firing_gate_preserves_the_partial_artifact():
    import inspect
    src = inspect.getsource(fh.main)
    assert "DegenerateRewardError" in src
    assert "write_artifact(partial" in src
    # A gate firing is a RESULT. Re-running it with a weaker threshold would
    # turn the finding into a knob, so the refusal is stated in the artifact.
    assert "lowered" in src and "threshold" in src


def test_the_artifact_is_checkpointed_after_every_config():
    import inspect
    src = inspect.getsource(fh.run_all)
    assert "write_artifact(results, seeds, n)      # checkpoint after every config" in src


def test_the_preregistration_document_exists_and_names_every_hypothesis():
    path = os.path.join(_REPO_ROOT, "docs", "eval",
                        "PREREGISTRATION_final_hardening.md")
    assert os.path.isfile(path)
    text = open(path, encoding="utf-8").read()
    for key in ("H1", "H2", "H3", "H4", "H5"):
        assert f"### {key}" in text
    assert "Static-Tput" in text and "Static-Utility" in text
    assert fh.PRE_REGISTRATION["document"].endswith(
        "PREREGISTRATION_final_hardening.md")
