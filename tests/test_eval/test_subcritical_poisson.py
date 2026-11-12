"""The sub-critical Poisson experiment: is it the experiment it claims to be?

These tests are structural and behavioural, never a GPU run. They check the
properties that decide whether the artifact would MEAN anything:

  * the operating point is DERIVED from a measured service rate, per phase, for
    the execution path that will actually run -- and every ambiguity aborts
    rather than defaulting to a convenient lambda;
  * the arrival process generated is the arrival process described, and is
    deterministic in the seed;
  * the static baselines and reward scales are RE-DERIVED here, in the order
    that stops Static-Utility shaping the yardstick it is ranked under;
  * the cold-start repair, the reward gates and the convergence rule are the
    saturated experiment's, unchanged;
  * the frozen saturated artifact is neither read nor written.

Every test names the wrong behaviour it forbids, because a test that merely
asserts today's behaviour will keep passing after that behaviour regresses into
a different-but-equally-wrong one.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import random
import statistics
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_DIR = os.path.join(_REPO_ROOT, "scripts", "eval")
for _p in (_REPO_ROOT, _EVAL_DIR, os.path.join(_EVAL_DIR, "repair")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch", reason="the experiment imports torch at module level")

import torch  # noqa: E402

import ablation_live as abl  # noqa: E402
import final_hardening as fh  # noqa: E402
import subcritical_poisson as sp  # noqa: E402

from src.carl.config import CARLConfig, all_arm_sets  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.live import ARRIVAL_POISSON  # noqa: E402
from src.carl.live_reward import LiveRewardV2  # noqa: E402
from src.carl.reward import (  # noqa: E402
    DEFAULT_WEIGHTS, RewardScales, utility_v2,
)
from src.carl.state import FEATURE_DIM, MetricsTracker, RuntimeState  # noqa: E402

SCALES = RewardScales(t_half=96.0, ttft_target=60000.0, tpot_target=300.0)


# ---------------------------------------------------------------------------
# Fixtures and doubles.
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Eleven real-looking token ids; `_make_prompt` tiles them to length."""

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.arange(1, 12).unsqueeze(0)}


def _plan(lam_i=2.0, lam_b=0.5, rho=0.4, mu_i=5.0, mu_b=1.25) -> dict:
    """An arrival plan shaped like derive_service_rates()'s output."""
    return {
        "target_rho": rho, "execution_arm": "eager",
        "phases": {
            "interactive": {"shape_key": "interactive",
                            "saturation_row": "interactive@eager",
                            "mu_max_req_per_s": mu_i, "target_rho": rho,
                            "lambda_rps": lam_i,
                            "expected_mean_interarrival_s": 1.0 / lam_i},
            "batch": {"shape_key": "batch", "saturation_row": "batch@eager",
                      "mu_max_req_per_s": mu_b, "target_rho": rho,
                      "lambda_rps": lam_b,
                      "expected_mean_interarrival_s": 1.0 / lam_b},
        },
    }


def _artifact(**over) -> dict:
    """A minimal load-sweep-shaped artifact; `over` perturbs it per test."""
    base = {
        "hardware_profile": {
            "decode_per_row_s": 0.009448482794434556,
            "decode_fixed_per_step_s": 0.02546177948065939,
            "cuda_graph_decode_per_row_s": 0.0051007379055177885,
            "cuda_graph_decode_fixed_per_step_s": 0.010862193551853631,
        },
        "saturation": {
            "interactive@eager": {"mu_max_req_per_s": 2.9613881136988742,
                                  "method": "burst dump at max_batch=32",
                                  "n_seeds": 3, "n_requests": 300},
            "batch@eager": {"mu_max_req_per_s": 1.4135291672527908,
                            "method": "burst dump at max_batch=32",
                            "n_seeds": 3, "n_requests": 300},
            "interactive@graph": {"mu_max_req_per_s": 5.563800006488338},
            "batch@graph": {"mu_max_req_per_s": 2.564880811317019},
        },
    }
    base.update(over)
    return base


def _write(tmp_path, artifact: dict) -> str:
    path = os.path.join(str(tmp_path), "sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f)
    return path


def _row(mb, cs, tps, ttft, tpot):
    return {"max_batch_size": mb, "chunk_size": cs,
            "config": CARLConfig(max_batch_size=mb, chunk_size=cs).clamp().as_dict(),
            "throughput_tps": tps, "ttft_p50_ms": ttft / 3, "ttft_p99_ms": ttft,
            "tpot_p50_ms": tpot / 2, "tpot_p99_ms": tpot, "slo_rate": 0.0,
            "wall_s": 100.0}


def _gaps(offsets):
    return [b - a for a, b in zip(offsets, offsets[1:])]


# ===========================================================================
# 1-3. rho -> lambda, per phase, from DIFFERENT service capacities.
# ===========================================================================


def test_1_lambda_is_rho_times_mu():
    assert sp.lambda_for_rho(10.0, 0.4) == pytest.approx(4.0)
    assert sp.lambda_for_rho(2.9613881136988742, 0.40) == pytest.approx(
        1.1845552454795498, abs=1e-15)
    assert sp.TARGET_RHO == 0.40


def test_1b_a_supercritical_or_nonsense_rho_is_refused():
    """The experiment is DEFINED by an offered load comfortably below capacity.
    A rho >= 1 would make it the saturated experiment under a different name."""
    for bad in (1.0, 1.5, 0.0, -0.2):
        with pytest.raises(ValueError, match="sub-critical|rho"):
            sp.lambda_for_rho(10.0, bad)
    with pytest.raises(sp.AmbiguousServiceRateError):
        sp.lambda_for_rho(0.0, 0.4)
    with pytest.raises(sp.AmbiguousServiceRateError):
        sp.lambda_for_rho(float("nan"), 0.4)


def test_2_lambdas_are_phase_specific(tmp_path):
    plan = sp.derive_service_rates("eager", artifact_path=_write(tmp_path, _artifact()),
                                   calibration_path=None)
    li = plan["phases"]["interactive"]["lambda_rps"]
    lb = plan["phases"]["batch"]["lambda_rps"]
    assert li == pytest.approx(0.40 * 2.9613881136988742)
    assert lb == pytest.approx(0.40 * 1.4135291672527908)
    assert li != lb, (
        "one lambda for both phases would place one of them at roughly twice "
        "the other's offered load, and 'sub-critical' would hold for only one")


def test_3_each_phase_uses_its_own_measured_capacity(tmp_path):
    plan = sp.derive_service_rates("eager", artifact_path=_write(tmp_path, _artifact()),
                                   calibration_path=None)
    pi, pb = plan["phases"]["interactive"], plan["phases"]["batch"]
    assert pi["saturation_row"] == "interactive@eager"
    assert pb["saturation_row"] == "batch@eager"
    assert pi["mu_max_req_per_s"] != pb["mu_max_req_per_s"]
    # And each lambda is rho times ITS OWN mu, not the other phase's.
    assert pi["lambda_rps"] == pytest.approx(plan["target_rho"] * pi["mu_max_req_per_s"])
    assert pb["lambda_rps"] == pytest.approx(plan["target_rho"] * pb["mu_max_req_per_s"])
    assert pi["expected_mean_interarrival_s"] == pytest.approx(1.0 / pi["lambda_rps"])


# ===========================================================================
# 4. Execution reality, not config labels.
# ===========================================================================


def test_4_no_graph_service_rate_when_no_graph_runner_exists(tmp_path):
    """THE TRAP. `CARLConfig.use_cuda_graphs` defaults True, but
    `_new_scheduler` attaches no runner and `_decode_forward` will not replay
    without one. Taking the flag at face value picks mu ~1.9x too high, and a
    nominal rho of 0.40 silently becomes ~0.75."""
    assert CARLConfig().use_cuda_graphs is True, (
        "if this default ever changes the trap changes shape; the test below "
        "must still read the SCHEDULER, not the config")

    class _Sched:
        max_batch_size = 8
        chunk_size = 256
        use_cuda_graphs = True          # says yes
        enable_spec_decode = False
        _graph_runner = None            # does no

    assert sp.execution_arm_for(_Sched()) == "eager"

    plan = sp.derive_service_rates(sp.execution_arm_for(_Sched()),
                                   artifact_path=_write(tmp_path, _artifact()),
                                   calibration_path=None)
    assert plan["execution_arm"] == "eager"
    assert plan["rows_used"] == ["interactive@eager", "batch@eager"]
    graph_mus = {5.563800006488338, 2.564880811317019}
    for p in plan["phases"].values():
        assert p["mu_max_req_per_s"] not in graph_mus


def test_4b_an_attached_runner_would_select_the_graph_rows(tmp_path):
    """The converse, so the arm is read from the object in BOTH directions and
    an 'eager' answer is a measurement rather than a constant."""
    class _SchedWithGraphs:
        _graph_runner = object()

    assert sp.execution_arm_for(_SchedWithGraphs()) == "graph"
    plan = sp.derive_service_rates("graph",
                                   artifact_path=_write(tmp_path, _artifact()),
                                   calibration_path=None)
    assert plan["phases"]["interactive"]["mu_max_req_per_s"] == pytest.approx(
        5.563800006488338)


def test_4c_the_live_harness_really_does_attach_no_graph_runner():
    """Read against the source of the scheduler this experiment serves on, so
    the eager choice is pinned to the harness rather than to a comment."""
    src = inspect.getsource(abl._new_scheduler)
    assert "_graph_runner" not in src
    assert fh.INERT_DIMENSIONS["use_cuda_graphs"].startswith(
        "the scheduler has the flag")


# ===========================================================================
# 5. Every ambiguity aborts. None of them defaults.
# ===========================================================================


def test_5_a_missing_artifact_aborts_rather_than_inventing_lambda():
    with pytest.raises(sp.AmbiguousServiceRateError, match="missing"):
        sp.derive_service_rates("eager", artifact_path="docs/eval/no_such.json")


def test_5b_a_missing_saturation_block_aborts(tmp_path):
    art = _artifact()
    del art["saturation"]
    with pytest.raises(sp.AmbiguousServiceRateError, match="saturation"):
        sp.derive_service_rates("eager", artifact_path=_write(tmp_path, art),
                                calibration_path=None)


def test_5c_a_missing_row_for_the_executing_arm_aborts(tmp_path):
    art = _artifact()
    del art["saturation"]["batch@eager"]
    with pytest.raises(sp.AmbiguousServiceRateError, match="batch@eager"):
        sp.derive_service_rates("eager", artifact_path=_write(tmp_path, art),
                                calibration_path=None)


def test_5d_an_unusable_mu_aborts(tmp_path):
    for bad in (0.0, -1.0, None, "fast", float("inf")):
        art = _artifact()
        art["saturation"]["interactive@eager"]["mu_max_req_per_s"] = bad
        with pytest.raises(sp.AmbiguousServiceRateError):
            sp.derive_service_rates("eager", artifact_path=_write(tmp_path, art),
                                    calibration_path=None)


def test_5e_indistinguishable_arms_abort(tmp_path):
    """If the eager and CUDA-graph cost models are identical, the two saturation
    arms cannot be told apart and an 'eager' mu means nothing."""
    art = _artifact()
    hw = art["hardware_profile"]
    hw["cuda_graph_decode_per_row_s"] = hw["decode_per_row_s"]
    with pytest.raises(sp.AmbiguousServiceRateError, match="identical"):
        sp.derive_service_rates("eager", artifact_path=_write(tmp_path, art),
                                calibration_path=None)


def test_5f_a_missing_hardware_profile_aborts(tmp_path):
    art = _artifact()
    del art["hardware_profile"]
    with pytest.raises(sp.AmbiguousServiceRateError, match="hardware_profile"):
        sp.derive_service_rates("eager", artifact_path=_write(tmp_path, art),
                                calibration_path=None)


def test_5g_eager_parameters_that_stop_matching_the_t4_fit_abort(tmp_path):
    """What makes 'eager' a MEASURED path is that the sweep's eager decode
    parameters equal the eager least-squares fit from the real T4 run. If that
    stops holding, the capacity estimate can no longer be attributed to the
    non-graph execution path."""
    cal = os.path.join(str(tmp_path), "cal.json")
    with open(cal, "w", encoding="utf-8") as f:
        json.dump({"analysis": {"hardware_profile_fit_eager": {
            "decode_per_row_s": 0.123, "fixed_per_step_s": 0.456}}}, f)
    with pytest.raises(sp.AmbiguousServiceRateError, match="eager"):
        sp.derive_service_rates("eager",
                                artifact_path=_write(tmp_path, _artifact()),
                                calibration_path=cal)


def test_5h_the_real_artifacts_satisfy_the_crosscheck():
    """The derivation this experiment will actually run, end to end, against the
    artifacts in the repository."""
    plan = sp.derive_service_rates("eager")
    x = plan["arm_evidence"]["hardware_fit_crosscheck"]
    assert x["checked"] is True
    assert x["eager_parameters_match_fit"] is True
    assert x["measured_on"] == "Tesla T4"
    assert plan["phases"]["interactive"]["mu_max_req_per_s"] == pytest.approx(
        2.9613881136988742)
    assert plan["phases"]["batch"]["mu_max_req_per_s"] == pytest.approx(
        1.4135291672527908)
    assert plan["phases"]["interactive"]["lambda_rps"] == pytest.approx(
        1.1845552454795498)
    assert plan["phases"]["batch"]["lambda_rps"] == pytest.approx(
        0.5654116669011163)
    # The known limitation is STATED in the artifact, not left for a reader to
    # discover: mu is simulator-derived and the probe shapes are close but not
    # identical to this harness's prompt windows.
    assert "calibrated mechanistic substrate" in plan["known_limitation"]


# ===========================================================================
# 6-8. The generator produces the process it says it does.
# ===========================================================================


def test_6_arrivals_are_reproducible_under_a_seed():
    tok, plan = _Tokenizer(), _plan()
    _s1, a1 = sp.build_poisson_workload(tok, 40, 42, plan)
    _s2, a2 = sp.build_poisson_workload(tok, 40, 42, plan)
    assert a1 == a2
    assert a1["interactive"] and a1["batch"]


def test_7_different_seeds_produce_different_arrival_sequences():
    tok, plan = _Tokenizer(), _plan()
    seqs = [tuple(sp.build_poisson_workload(tok, 40, s, plan)[1]["interactive"])
            for s in (42, 43, 44, 45, 46)]
    assert len(set(seqs)) == 5, (
        "five seeds producing fewer than five arrival streams would mean the "
        "seed cannot reach the arrival process")


def test_7b_the_arrival_stream_is_isolated_from_the_prompt_draws():
    """Arrivals are the independent variable. If they shared a generator with
    the prompt-length draws, a change to shape sampling would silently move the
    operating point."""
    plan = _plan()
    _specs, arr = sp.build_poisson_workload(_Tokenizer(), 40, 42, plan)
    # The workload's arrivals ARE the standalone arrival draw, which needs no
    # tokenizer and never touches the prompt-length stream.
    assert arr == sp.poisson_arrivals(40, 42, plan)
    # Draining the shape stream must not move the arrival stream.
    a = sp._arrival_rng(42).expovariate(1.0)
    junk = random.Random(42)
    for _ in range(1000):
        junk.randint(0, 100)
    assert sp._arrival_rng(42).expovariate(1.0) == a


def test_8_the_empirical_interarrival_mean_approaches_one_over_lambda():
    """A statistical check, not an eyeball one: the mean of n exponential gaps
    has standard error (1/lambda)/sqrt(n), so a large sample must land inside a
    narrow interval around 1/lambda, and the median must land near ln(2)/lambda
    -- which a uniform or deterministic generator would fail."""
    lam = 1.1845552454795498
    rng = sp._arrival_rng(42)
    from src.carl.live import build_arrivals
    offsets = build_arrivals(20000, ARRIVAL_POISSON, lam, rng)
    gaps = _gaps(offsets)
    expected = 1.0 / lam
    se = expected / math.sqrt(len(gaps))
    assert abs(statistics.fmean(gaps) - expected) < 3.0 * se
    assert statistics.median(gaps) == pytest.approx(math.log(2) / lam, rel=0.05)
    assert min(gaps) >= 0.0


def test_8b_the_reported_arrival_stats_carry_every_field_the_preregistration_promises():
    tok, plan = _Tokenizer(), _plan()
    _specs, arr = sp.build_poisson_workload(tok, 200, 42, plan)
    r = sp.arrival_report(arr["interactive"], "interactive", plan)
    for field in ("requested_lambda_rps", "target_rho", "mu_req_per_s",
                  "mu_source_row", "n_requests",
                  "empirical_mean_interarrival_s",
                  "empirical_median_interarrival_s",
                  "empirical_lambda_rps", "empirical_rho",
                  "empirical_rho_minus_target", "scheduled_arrival_window_s",
                  "expected_mean_interarrival_s", "expected_mean_ci95",
                  "expected_mean_ci95_is_informative", "mean_within_ci95"):
        assert field in r, f"arrival stat {field} is not logged"
    assert r["process"] == ARRIVAL_POISSON
    assert r["expected_mean_interarrival_s"] == pytest.approx(
        1.0 / r["requested_lambda_rps"])
    assert r["empirical_lambda_rps"] == pytest.approx(
        1.0 / r["empirical_mean_interarrival_s"], rel=1e-9)
    assert r["n_requests"] == 100
    assert r["expected_mean_ci95"][0] >= 0.0, (
        "an interarrival mean cannot be negative; the normal approximation's "
        "lower bound must be clamped rather than reported raw")
    assert r["expected_mean_ci95_is_informative"] is True    # 99 gaps
    # At a tiny gap count the interval accepts almost anything, and the record
    # must SAY so rather than letting H5 pass quietly.
    tiny = sp.arrival_report([0.0, 0.5], "interactive", plan)
    assert tiny["expected_mean_ci95_is_informative"] is False
    assert tiny["expected_mean_ci95"][0] >= 0.0


# ===========================================================================
# 9-10. The two phases, at their two rates, in the right order.
# ===========================================================================


def _phase_means(n=400, lam_i=8.0, lam_b=0.5):
    tok = _Tokenizer()
    plan = _plan(lam_i=lam_i, lam_b=lam_b)
    specs, arr = sp.build_poisson_workload(tok, n, 42, plan)
    return specs, arr, statistics.fmean(_gaps(arr["interactive"])), \
        statistics.fmean(_gaps(arr["batch"]))


def test_9_the_first_half_arrives_at_the_interactive_lambda():
    specs, arr, mean_i, _mean_b = _phase_means()
    assert len(arr["interactive"]) == 200
    assert [s.phase for s in specs[:200]] == [0] * 200
    assert all(s.rid.startswith("i") for s in specs[:200])
    assert mean_i == pytest.approx(1.0 / 8.0, rel=0.20)


def test_10_the_second_half_arrives_at_the_batch_lambda():
    specs, arr, mean_i, mean_b = _phase_means()
    assert len(arr["batch"]) == 200
    assert [s.phase for s in specs[200:]] == [1] * 200
    assert all(s.rid.startswith("b") for s in specs[200:])
    assert mean_b == pytest.approx(1.0 / 0.5, rel=0.20)
    # The two rates are not merely different constants in a dict; they produce
    # measurably different streams, in the stated direction.
    assert mean_b > mean_i * 4


def test_10b_the_batch_phase_continues_the_stream_rather_than_restarting_it():
    """The regime flip must be a change in the STREAM, not a synchronised dump
    at t=0 and not a second run beginning at zero."""
    _specs, arr, _mi, _mb = _phase_means()
    assert arr["interactive"][0] == 0.0
    assert arr["batch"][0] == pytest.approx(max(arr["interactive"]))
    assert all(b >= max(arr["interactive"]) for b in arr["batch"])


def test_10c_the_semantic_phase_shapes_are_unchanged():
    """Only ARRIVALS move. The prompt/output windows are the harness's own."""
    from src.carl.live import _SCENARIO_SPEC
    src = inspect.getsource(sp.build_poisson_workload)
    assert '_SCENARIO_SPEC["INTERACTIVE"]' in src
    assert '_SCENARIO_SPEC["BATCH"]' in src
    specs, _arr = sp.build_poisson_workload(_Tokenizer(), 200, 42, _plan())
    lo, hi = (_SCENARIO_SPEC["INTERACTIVE"]["prompt_lo"],
              _SCENARIO_SPEC["INTERACTIVE"]["prompt_hi"])
    for s in specs[:100]:
        assert lo <= int(s.prompt_ids.shape[-1]) <= hi
        assert s.max_new == _SCENARIO_SPEC["INTERACTIVE"]["max_new"]
    for s in specs[100:]:
        assert s.max_new == _SCENARIO_SPEC["BATCH"]["max_new"]


def test_10d_phase_one_is_released_open_loop_not_dumped():
    """The historical harness dumps phase 1 when half of phase 0 has finished --
    a CLOSED-LOOP trigger under which a phase-specific arrival RATE cannot be
    expressed, because the pattern becomes a function of engine speed."""
    assert "phase1_open_loop" in inspect.signature(abl._serve).parameters
    assert inspect.signature(abl._serve).parameters[
        "phase1_open_loop"].default is False, (
        "the historical bulk-dump path must remain the default so every earlier "
        "experiment stays reproducible")
    for fn in (sp.run_static, sp.run_one):
        assert "phase1_open_loop=True" in inspect.getsource(fn)


def test_10e_the_default_serve_path_is_byte_for_byte_the_old_one():
    """An additive change that altered the default would silently invalidate
    every artifact produced by the saturated experiment."""
    src = inspect.getsource(abl._serve)
    # The closed-loop dump still exists and is still reached when the flag is off.
    assert "if not phase1_done and finished_count >= max(1, len(phase0) // 2):" in src
    assert "phase1_done = (len(phase1) == 0) or phase1_open_loop" in src


# ===========================================================================
# 11-15. The held-out search, re-derived here, in the right order.
# ===========================================================================


def test_11_the_held_out_seed_is_never_an_evaluation_seed(engine, tmp_path):
    assert sp.VALIDATION_SEED == fh.VALIDATION_SEED == abl.VALIDATION_SEED == 999
    assert sp.VALIDATION_SEED not in sp.DEFAULT_SEEDS
    assert sp.DEFAULT_SEEDS == [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    # Behaviourally: every sweep run serves seed 999's arrival stream, and no
    # evaluation run does.
    _run_smoke(tmp_path / "out")
    plan = sp.derive_service_rates("eager")
    held_out = sp.poisson_arrivals(_N, 999, plan)
    on_held_out = [ctl for ctl, a in engine.served if a == held_out]
    assert on_held_out == [False] * 4, "exactly the 4 static sweep runs"
    assert len(engine.served) == 4 + 10


def test_12_the_static_sweep_is_a_fresh_exhaustive_28_configurations():
    assert sp.GRID_BATCH_AXIS == [2, 4, 8, 12, 16, 24, 32]
    assert sp.GRID_CHUNK_AXIS == [64, 128, 256, 512]
    assert len(sp.GRID_BATCH_AXIS) * len(sp.GRID_CHUNK_AXIS) == 28
    assert "FRESH EXHAUSTIVE" in inspect.getsource(sp.sweep_live_plane)
    assert not hasattr(sp, "N_LHS_CANDIDATES")


def test_12b_the_saturated_selections_are_not_reused():
    """The saturated Static-Tput (mb=32, cs=512) and Static-Utility (mb=8,
    cs=256) are properties of a saturated queue. Carrying either here would give
    this experiment a baseline tuned for a condition it is not running."""
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0),
            _row(4, 128, 140.0, 25000.0, 300.0)]
    cal = sp.compute_calibration(rows, 200, "f" * 64)
    # The winners are whatever THIS grid says, not the saturated identities.
    assert cal["binding"]["static_tput"]["max_batch_size"] == 4
    assert cal["binding"]["static_tput"] != sp.SATURATED_REFERENCE["static_tput"]
    ref = sp.SATURATED_REFERENCE
    assert ref["static_tput"] == {"max_batch_size": 32, "chunk_size": 512}
    assert ref["static_utility"] == {"max_batch_size": 8, "chunk_size": 256}


def test_13_scales_are_frozen_from_measurements_before_any_utility_is_computed(
        monkeypatch):
    """Ordering, not just provenance. If the utility ranking could move the
    scales, Static-Utility would be chosen under a yardstick it had shaped."""
    touched = set(fh.freeze_scales.__code__.co_names)
    assert not touched & {"utility_v2", "run_utility", "term_breakdown"}
    assert "from_measurements" in touched
    assert list(inspect.signature(fh.select_static_baselines).parameters) == [
        "rows", "scales"], "the ranking must CONSUME frozen scales, never produce them"

    calls = []
    for name in ("freeze_scales", "select_static_baselines"):
        real = getattr(fh, name)
        monkeypatch.setattr(fh, name, lambda *a, _r=real, _n=name, **k: (
            calls.append(_n), _r(*a, **k))[1])
    real_gate = sp.gate_sweep
    monkeypatch.setattr(sp, "gate_sweep", lambda *a, **k: (
        calls.append("gate_sweep"), real_gate(*a, **k))[1])
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0),
            _row(32, 512, 140.0, 200000.0, 900.0)]
    sp.compute_calibration(rows, 200, "f" * 64)
    assert calls == ["freeze_scales", "gate_sweep", "select_static_baselines"]


def test_13b_the_scales_come_only_from_this_operating_points_rows():
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0),
            _row(32, 512, 140.0, 200000.0, 900.0)]
    scales, record = fh.freeze_scales(rows, 200, 999)
    assert scales.t_half == 120.0           # median of these three throughputs
    assert record["held_out_seed"] == 999
    assert record["n_observations"] == 3


def test_14_static_tput_maximises_throughput():
    rows = [_row(32, 512, 140.0, 200000.0, 900.0),
            _row(12, 256, 120.0, 30000.0, 250.0),
            _row(2, 64, 60.0, 20000.0, 200.0)]
    cfg_t, _cfg_u, rec = fh.select_static_baselines(rows, SCALES)
    assert cfg_t.max_batch_size == 32 and cfg_t.chunk_size == 512
    assert rec["static_tput"]["ranked_by"] == "throughput_tps"
    assert rec["static_tput"]["row"]["throughput_tps"] == 140.0


def test_15_static_utility_maximises_exactly_carls_objective():
    """Not 'a utility-like score': the same function, weights and frozen scales
    the controller is driven by. Anything else reintroduces the mismatch."""
    rows = [_row(32, 512, 140.0, 200000.0, 900.0),
            _row(12, 256, 120.0, 30000.0, 250.0),
            _row(2, 64, 60.0, 20000.0, 200.0)]
    cfg_t, cfg_u, rec = fh.select_static_baselines(rows, SCALES)
    assert cfg_u.max_batch_size == 12
    assert cfg_t != cfg_u and rec["objectives_agree"] is False
    for entry in rec["ranked_table"]:
        assert entry["utility_v2"] == pytest.approx(utility_v2(
            {"throughput_tps": entry["throughput_tps"],
             "ttft_p99_ms": entry["ttft_p99_ms"],
             "tpot_p99_ms": entry["tpot_p99_ms"], "cache_hit_rate": 0.0},
            DEFAULT_WEIGHTS, SCALES))


def test_15b_the_two_estimators_of_utility_are_never_substituted():
    assert "run_utility_v2" in sp._METRICS
    assert "mean_valid_cycle_utility" not in sp._METRICS
    u = sp.UTILITY_ESTIMATORS
    assert "same utility_v2 function, weights, and frozen scales" in u[
        "static_utility_selection_wording"]
    assert "end-of-run utility_v2" in u["static_utility_selection_wording"]
    assert "NOT claimed to optimise the exact sequence of per-cycle rewards" in u[
        "forbidden_wording"], (
        "the artifact must state that Static-Utility does not optimise the "
        "per-cycle reward SEQUENCE CARL consumed")
    assert "never substituted" in u["forbidden_wording"]
    assert "NEVER compared" in u["mean_valid_cycle_utility"]
    assert "dict(UTILITY_ESTIMATORS)" in inspect.getsource(sp.run_all)


def test_15c_the_landscape_report_answers_q1_descriptively():
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0),
            _row(32, 512, 140.0, 200000.0, 900.0)]
    _t, _u, sel = fh.select_static_baselines(rows, SCALES)
    lg = sp.landscape_report(rows, SCALES, sel)
    t = lg["throughput_tps"]
    assert t["min"] == 60.0 and t["max"] == 140.0
    assert t["range_absolute"] == pytest.approx(80.0)
    assert t["range_relative_pct"] == pytest.approx(100.0 * 80.0 / 140.0)
    assert lg["utility_v2"]["range_absolute"] > 0
    assert lg["static_tput_identity"]["max_batch_size"] == 32
    assert lg["static_utility_identity"]["max_batch_size"] == 12
    # No significance theatre over 28 deterministic search points.
    method = lg["comparison_with_saturated_point"]["method"]
    assert "DESCRIPTIVE" in method and "No significance test" in method
    assert "p_value" not in json.dumps(lg)


# ===========================================================================
# 16-18. The cold-start repair, preserved exactly.
# ===========================================================================


def test_16_the_cold_start_repair_is_switched_ON():
    src = inspect.getsource(sp.run_all)
    assert "require_valid_metrics=True" in src
    assert "min_latency_samples=1" in src


class _SpyBandit:
    """PerRegimeBandit-shaped, recording every update it is handed.

    Hand-written rather than a mock double on purpose: the controller drives its
    policy through a duck-typed interface, so this exercises the same path the
    real bandit takes.
    """

    def __init__(self) -> None:
        self.arms_by_regime = all_arm_sets()
        self.updates: list = []

    def arms(self, regime):
        return self.arms_by_regime[regime]

    def select(self, regime, context):
        return 0, self.arms_by_regime[regime][0]

    def update(self, regime, arm, reward, context):
        self.updates.append((regime, arm, reward))


def _cold_tracker():
    t = MetricsTracker(window=200)
    t.record_throughput(90.0)
    t.record_batch(8)
    return t


def _cold_state():
    return RuntimeState(queue_depth=40, avg_prompt_len=48.0, throughput_tps=90.0,
                        p50_ttft_ms=0.0, p99_tpot_ms=0.0, active_requests=8)


def test_17_an_invalid_reward_performs_no_learner_update():
    """Not an update carrying 0.0, and not a penalty: NO call. A missing
    measurement must not be fabricated in either direction."""
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = CARLController(scheduler=None, bandit=spy, observe_interval=1,
                       metrics=tracker,
                       reward_fn=LiveRewardV2(SCALES, require_valid_metrics=True,
                                              min_latency_samples=1))
    for _ in range(3):
        c.step(state=_cold_state())

    assert spy.updates == []
    assert [e.reward for e in c.controller_log] == [None, None, None]
    assert [e.reward_valid for e in c.controller_log] == [False] * 3
    # rewarded_arm = -1 says the learner received nothing for that interval, and
    # stops a later reader attributing a withheld cycle to some arm.
    assert [e.rewarded_arm for e in c.controller_log] == [-1, -1, -1]


def test_18_withheld_after_first_valid_is_reported_and_starts_at_zero():
    spy, tracker = _SpyBandit(), _cold_tracker()
    c = CARLController(scheduler=None, bandit=spy, observe_interval=1,
                       metrics=tracker,
                       reward_fn=LiveRewardV2(SCALES, require_valid_metrics=True,
                                              min_latency_samples=1))
    for _ in range(3):
        c.step(state=_cold_state())
    tracker.record_request(ttft_ms=21000.0, tpot_ms=275.0)
    for _ in range(3):
        c.step(state=_cold_state())

    r = c.reward_validity_report()
    assert r["warmup_cycles"] == 3
    assert r["withheld_after_first_valid"] == 0
    assert r["first_valid_reward_cycle_index"] == 3
    assert r["valid_reward_updates"] >= 1
    for field in ("warmup_cycles", "first_valid_reward_cycle_index",
                  "first_valid_reward_scheduler_step", "warmup_duration_s",
                  "valid_reward_updates", "withheld_after_first_valid"):
        assert field in r


def test_18b_the_experiment_records_every_cold_start_field_per_seed():
    src = inspect.getsource(sp.run_all)
    for field in ("warmup_cycles_per_seed", "first_valid_reward_cycle_per_seed",
                  "first_valid_reward_step_per_seed",
                  "warmup_duration_s_per_seed",
                  "withheld_after_first_valid_total",
                  "h4_positive_control_holds"):
        assert field in src
    h4 = sp.PRE_REGISTRATION["H4_cold_start"]
    assert "warmup_cycles == 0" in h4["falsified_if"]


def test_18c_reward_non_degeneracy_uses_the_existing_frozen_thresholds():
    """Thresholds are not silently changed. If the sub-critical plane is too
    flat for the reward to separate 28 configs, that is a RESULT, not a knob."""
    from src.carl.live_reward import (
        MIN_ARM_REWARD_SPREAD, MIN_LIVE_DISTINCT, MIN_LIVE_REWARD_SPREAD,
    )
    assert (MIN_ARM_REWARD_SPREAD, MIN_LIVE_REWARD_SPREAD, MIN_LIVE_DISTINCT) == (
        0.02, 0.01, 3)
    sweep_src = inspect.getsource(sp.gate_sweep)
    assert "threshold=MIN_ARM_REWARD_SPREAD" in sweep_src
    run_src = inspect.getsource(sp.run_one)
    assert "threshold=MIN_LIVE_REWARD_SPREAD" in run_src
    assert "min_distinct=MIN_LIVE_DISTINCT" in run_src
    # The per-run gate sees only VALID rewards; padding a withheld cycle would
    # make it pass or fail for a reason unrelated to discrimination.
    assert 'valid = [c["reward"] for c in cycles if c["reward_valid"]]' in run_src
    for field in ("valid_reward_spread", "n_distinct_valid_rewards",
                  "mean_valid_cycle_utility"):
        assert field in run_src


# ===========================================================================
# 19-21. Treatments, and the rules that must not move.
# ===========================================================================


def test_19_all_five_treatments_are_present_and_no_more():
    assert sp.CONFIGS == ["Static-Tput", "Static-Utility", "RuleOnly",
                          "CARL-Repaired", "CARL-Expanded"]
    assert sp._STATIC_CONFIGS == {"Static-Tput", "Static-Utility"}
    assert sp._CONTROLLER_CONFIGS == {"RuleOnly", "CARL-Repaired", "CARL-Expanded"}
    assert sp._ARM_SET_FOR == fh._ARM_SET_FOR == {
        "CARL-Repaired": "restricted", "CARL-Expanded": "expanded",
        "RuleOnly": "restricted"}
    labels = {c[0]: (c[1], c[2], c[3]) for c in sp.COMPARISONS}
    assert labels["A_adaptation_vs_throughput_baseline"] == (
        "CARL-Repaired", "Static-Tput", "throughput_tps")
    assert labels["B_adaptation_vs_utility_baseline"] == (
        "CARL-Repaired", "Static-Utility", "run_utility_v2")
    assert labels["C_learning_isolation"] == (
        "CARL-Repaired", "RuleOnly", "throughput_tps")
    assert labels["D_action_space_isolation"] == (
        "CARL-Expanded", "CARL-Repaired", "throughput_tps")


def test_19b_each_baseline_is_judged_on_the_metric_it_was_selected_for():
    selected_on = {"Static-Tput": "throughput_tps",
                   "Static-Utility": "run_utility_v2"}
    for _label, _a, b, metric, _q in sp.COMPARISONS:
        if b in selected_on:
            assert metric == selected_on[b]


def test_20_there_is_no_dynoracle_and_no_inactive_ablation():
    text = inspect.getsource(sp)
    assert "DynOracle" not in sp.CONFIGS
    for name in ("CARL-NoSpec", "CARL-NoCache", "CARL-NoRouter", "Static-Best",
                 "AutoTuner", "CARL-Thompson"):
        assert name not in sp.CONFIGS
    # Named once, in the record of what was deliberately NOT run and why.
    assert text.count("DynOracle") <= 3
    assert "not an oracle" in text


def test_21_the_convergence_rule_is_the_saturated_experiments_own_object():
    """Imported, not copied. A copy could drift; a more favourable definition
    would make Q6 unanswerable."""
    assert sp.CONVERGENCE_RULE is fh.CONVERGENCE_RULE
    assert sp.CONVERGENCE_RULE["min_decisions_in_regime"] == 6
    assert sp.CONVERGENCE_RULE["min_first_half_distinct_arms"] == 2
    assert sp.CONVERGENCE_RULE["min_second_half_modal_share"] == 0.75
    assert sp.CONVERGENCE_RULE["requires_increase_in_modal_share"] is True
    assert "fh.convergence_diagnostic(cycles)" in inspect.getsource(sp.run_one)
    assert not hasattr(sp, "convergence_diagnostic"), (
        "defining a local diagnostic is how a more favourable rule gets in")


def test_21b_a_learner_that_never_moved_is_still_not_convergence():
    cycles = [{"regime": "interactive", "selected_arm": 0} for _ in range(20)]
    d = fh.convergence_diagnostic(cycles)
    assert d["per_regime"]["interactive"]["second_half_modal_share"] == 1.0
    assert d["supported_in_any_regime"] is False


def test_21c_convergence_is_reported_per_seed_and_per_regime():
    src = inspect.getsource(sp.run_all)
    assert "convergence_per_seed" in src
    assert "convergence_supported_seeds" in src
    assert "supported_in_all_regimes" in src
    assert sp.SATURATED_REFERENCE["convergence_supported_seeds"] == {
        "RuleOnly": 0, "CARL-Repaired": 0, "CARL-Expanded": 0, "out_of": 10}


# ===========================================================================
# 22-23. Liveness probe and effective action space.
# ===========================================================================


def test_22_the_live_dimension_probe_runs_and_aborts_on_a_surprise():
    src = inspect.getsource(sp.run_all)
    assert "fh.probe_live_dimensions(probe_sched)" in src
    # It must run BEFORE any GPU time is spent on the sweep.
    assert src.index("probe_live_dimensions") < src.index("sweep_live_plane")

    class _Sched:
        max_batch_size = 8
        chunk_size = 256
        use_cuda_graphs = True
        enable_spec_decode = False
        _graph_runner = None

    report = fh.probe_live_dimensions(_Sched())
    assert report["passed"] is True
    assert report["grid_is_exhaustive_over"]["n_configurations"] == 28

    class _SchedWithGraphs(_Sched):
        _graph_runner = object()

    with pytest.raises(RuntimeError, match="use_cuda_graphs"):
        fh.probe_live_dimensions(_SchedWithGraphs())


def test_22b_every_config_dimension_is_searched_or_has_a_stated_inert_reason():
    fields = set(CARLConfig().as_dict())
    assert fields == set(sp.LIVE_DIMENSIONS) | set(sp.INERT_DIMENSIONS)
    assert all(sp.INERT_DIMENSIONS.values())


def test_23_the_action_space_reports_nominal_arms_and_unique_live_configs():
    """Some nominal arms differ only in inert fields, so a bare arm count
    describes a space wider than the one that exists."""
    rep = sp.effective_arm_report(abl.expanded_arm_sets())
    assert rep["preferred_wording"] == "expanded live configuration coverage"
    for regime, r in rep["per_regime"].items():
        assert r["nominal_arms"] >= r["unique_live_configurations"] >= 1
        assert len(r["live_configuration_tuples"]) == r["unique_live_configurations"]
        assert r["reachable_max_batch_size"] == sorted(
            {t[0] for t in r["live_configuration_tuples"]})
        assert r["reachable_chunk_size"] == sorted(
            {t[1] for t in r["live_configuration_tuples"]})
        assert isinstance(r["nominal_arms_collapse"], bool)
        assert r["nominal_arms_collapse"] == (
            r["nominal_arms"] != r["unique_live_configurations"])

    # Expanded really does reach further than restricted on the live plane --
    # otherwise comparison D is not an action-space comparison at all.
    restricted = sp.effective_arm_report(all_arm_sets())
    for regime, r in restricted["per_regime"].items():
        e = rep["per_regime"][regime]
        assert set(r["reachable_max_batch_size"]) <= set(e["reachable_max_batch_size"])


def test_23b_a_collapsing_arm_set_is_reported_as_collapsing():
    """Two nominal arms that differ only in an INERT field are one live
    configuration, and the report must say so rather than counting two."""
    from src.carl.state import WorkloadRegime
    arms = [CARLConfig(max_batch_size=8, chunk_size=256, spec_k=0).clamp(),
            CARLConfig(max_batch_size=8, chunk_size=256, spec_k=4).clamp()]
    rep = sp.effective_arm_report({WorkloadRegime.INTERACTIVE: arms})
    r = rep["per_regime"]["interactive"]
    assert r["nominal_arms"] == 2
    assert r["unique_live_configurations"] == 1
    assert r["nominal_arms_collapse"] is True


# ===========================================================================
# 24-26. Logging, checkpointing, provenance.
# ===========================================================================


def test_24_arrival_stats_and_raw_offsets_are_both_persisted(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    payload = _load(_paths(out).eval_raw("CARL-Repaired", 42))
    plan = sp.derive_service_rates("eager")
    assert payload["arrival_offsets_s"] == sp.poisson_arrivals(_N, 42, plan), (
        "summary statistics alone cannot be re-checked; the raw offsets must "
        "survive")
    for field in ("arrival_validation", "offered_load", "drain_overrun",
                  "phase_boundary"):
        assert payload[field], field
    agg = _load(_paths(out).results)["configs"]["CARL-Repaired"]
    for field in ("arrival_validation_per_seed", "offered_load_per_seed",
                  "drain_overrun_per_seed", "phase_boundary_per_seed",
                  "empirical_rho_across_seeds"):
        assert agg[field], field


def test_24b_drain_overrun_is_not_called_rho_or_load():
    """The wall-clock ratio is an ENGINE observation. Naming it a realised load
    or a rho -- as an earlier draft did -- silently reassigns the experiment's
    defining quantity to something that is not it."""
    out = {"wall_s": 260.0, "requests": [{}] * 200}
    arr = {"interactive": [0.0, 84.0], "batch": [84.0, 255.0]}
    d = sp.drain_overrun(out, arr, {"target_rho": 0.4})

    assert d["metric"] == "arrival_window_overrun_ratio"
    assert d["definition"] == "wall_s / scheduled_arrival_window_s"
    assert d["scheduled_arrival_window_s"] == 255.0
    assert d["arrival_window_overrun_ratio"] == pytest.approx(260.0 / 255.0)
    assert d["requests_completed"] == 200
    assert "not an offered load" in d["is_not_rho"]

    # The forbidden names must not reappear anywhere in this block, and the
    # old function must be gone rather than left as an alias.
    blob = json.dumps(d).lower()
    for banned in ("realised_load", "realized_load", "utilisation",
                   "utilization"):
        assert banned not in blob
    for key in d:
        assert "rho" not in key or key == "is_not_rho"
    assert not hasattr(sp, "realised_load")
    assert "never used to re-tune" in inspect.getdoc(sp.drain_overrun)


def test_24c_empirical_rho_is_rate_over_mu():
    """THE definition, checked arithmetically rather than trusted."""
    plan = _plan(lam_i=2.0, lam_b=0.5, mu_i=5.0, mu_b=1.25)
    _specs, arr = sp.build_poisson_workload(_Tokenizer(), 400, 42, plan)
    for phase, mu in (("interactive", 5.0), ("batch", 1.25)):
        r = sp.arrival_report(arr[phase], phase, plan)
        assert r["empirical_rho"] == pytest.approx(
            r["empirical_lambda_rps"] / mu)
        assert r["mu_req_per_s"] == mu
        assert r["empirical_rho_minus_target"] == pytest.approx(
            r["empirical_rho"] - 0.4)
        assert r["empirical_rho_definition"] == (
            "empirical_lambda_rps / mu_req_per_s")
        # The realised value is near the target but NOT pinned to it -- if it
        # were exactly 0.40 the generator would not be sampling.
        assert 0.25 < r["empirical_rho"] < 0.60

    rep = sp.offered_load_report(
        {ph: sp.arrival_report(arr[ph], ph, plan) for ph in arr}, plan)
    assert rep["definition"] == "empirical_rho = empirical_lambda_rps / mu_req_per_s"
    assert set(rep["per_phase"]) == {"interactive", "batch"}
    for ph, blk in rep["per_phase"].items():
        for field in ("target_rho", "mu_req_per_s", "requested_lambda_rps",
                      "empirical_lambda_rps", "empirical_rho",
                      "empirical_rho_minus_target",
                      "empirical_mean_interarrival_s",
                      "empirical_median_interarrival_s",
                      "scheduled_arrival_window_s"):
            assert field in blk, f"{ph} is missing {field}"


def test_24d_no_seed_is_ever_rejected_for_its_realised_poisson_rate():
    """The first seeded draw is the evidence. Re-drawing seeds that missed 0.40
    would make the stated rho a selection effect."""
    plan = sp.derive_service_rates("eager")
    r = sp.arrival_report(
        sp.build_poisson_workload(_Tokenizer(), 200, 42, plan)[1]["interactive"],
        "interactive", plan)
    assert "No seed is rejected" in r["empirical_rho_policy"]
    rep = sp.offered_load_report({"interactive": r}, plan)
    assert "NO SEED IS EVER REJECTED" in rep["seed_selection_policy"]

    # Structural: the driver has no seed-filtering or retry path at all.
    driver = inspect.getsource(sp.run_all)
    for smell in ("if rho", "resample", "redraw", "re-draw", "retry",
                  "skip_seed", "continue  # rho", "while abs("):
        assert smell not in driver
    assert "no seed is excluded on the basis of these values" in driver
    # Seeds are consumed exactly as given, in order.
    assert "for i, seed in enumerate(seeds):" in driver
    prereg = open(sp.PREREG_PATH, encoding="utf-8").read()
    assert "no seed will be rejected" in prereg.lower()


def test_25_the_artifact_is_checkpointed_at_every_boundary(engine, tmp_path,
                                                          monkeypatch):
    """Spy on every checkpoint write of a real run and read the stages back."""
    stages = []
    real = sp.write_artifact
    monkeypatch.setattr(sp, "write_artifact", lambda r, *a, **k: (
        stages.append(r.get("stage")), real(r, *a, **k))[1])
    _run_smoke(tmp_path / "out")
    assert stages[0] == "arrival_plan_derived"
    assert stages.count("held_out_sweep") == 4            # every grid point
    assert "static_selection_complete" in stages
    assert sum(s.startswith("evaluating:") for s in stages) == 10   # every unit
    assert sum(s.startswith("completed:") for s in stages) == 5     # every config
    assert stages[-1] == "complete"


def test_25c_a_preflight_refusal_writes_no_artifact(engine, tmp_path):
    """A no-CUDA refusal measured nothing. Writing a file for it would leave
    something that looks like a result directory entry."""
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="REFUSES to run without CUDA"):
        sp.main(["--smoke", *_SMALL, "--output-dir", str(out)])
    assert not out.exists()
    assert engine.loads == 0


def test_25b_a_firing_gate_is_a_result_not_a_knob(engine, tmp_path, monkeypatch):
    _fail_gate_for(monkeypatch, "RuleOnly", 42)
    out = tmp_path / "out"
    with pytest.raises(sp.DegenerateRewardError):
        _run_smoke(out)
    ab = _load(_paths(out).results)["aborted"]
    assert ab["status"] == sp.STATUS_TERMINAL_GATE
    assert "lowered threshold" in ab["interpretation"]
    assert "report the degeneracy" in ab["interpretation"]


def test_26_provenance_is_embedded_with_no_silent_nulls():
    extra = sp._provenance_extra([42], 200, {
        "arrival_plan": sp.derive_service_rates("eager")})
    for field in ("model", "dtype", "device", "seeds", "requests",
                  "arrival_mode", "workload_version", "target_rho",
                  "execution_arm", "service_rate_source_artifact",
                  "service_rate_rows_used", "mu_interactive_req_per_s",
                  "mu_batch_req_per_s", "lambda_interactive_rps",
                  "lambda_batch_rps", "observe_interval", "validation_seed",
                  "learner_class", "alpha", "configs_run",
                  "arm_set_definition", "static_search_definition",
                  "reward_version", "frozen_reward_scales", "reward_weights",
                  "cold_start_policy", "convergence_rule",
                  "convergence_rule_source", "source_provenance"):
        assert field in extra, f"provenance is missing {field}"
    assert extra["target_rho"] == 0.40
    assert extra["execution_arm"] == "eager"
    assert extra["mu_interactive_req_per_s"] == pytest.approx(2.9613881136988742)
    assert extra["lambda_batch_rps"] == pytest.approx(0.5654116669011163)
    assert extra["static_search_definition"]["n_configurations"] == 28
    assert extra["static_search_definition"]["reused_from_saturated_run"] is False
    # An absent value is an explicit record, never a bare null.
    assert extra["frozen_reward_scales"]["recorded"] is False
    assert extra["frozen_reward_scales"]["why"]
    assert extra["source_provenance"]  # present block, or a stated absence


def test_26b_an_absent_source_sidecar_is_recorded_rather_than_nulled():
    real = fh.SOURCE_PROVENANCE_PATH
    try:
        fh.SOURCE_PROVENANCE_PATH = os.path.join(_REPO_ROOT, "no_such_file.json")
        block = fh.load_source_provenance()
    finally:
        fh.SOURCE_PROVENANCE_PATH = real
    assert block["present"] is False
    assert block["why_missing"] and block["remedy"]
    assert "fh.load_source_provenance()" in inspect.getsource(sp._provenance_extra)


def test_26c_provenance_never_reaches_a_network():
    """Local provenance only. The sidecar mechanism exists precisely so nothing
    has to contact a remote for a git SHA."""
    text = inspect.getsource(sp).lower()
    for forbidden in ("api.github.com", "requests.get", "requests.post",
                      "urllib", "http://", "https://", "subprocess",
                      "git push", "git fetch", "git clone", "git remote",
                      "gh pr", "gh api"):
        assert forbidden not in text, f"{forbidden!r} has no place in this module"
    # The one place a git SHA comes from is the LOCAL sidecar.
    assert "fh.load_source_provenance()" in inspect.getsource(sp._provenance_extra)


# ===========================================================================
# 27. The frozen artifact is not touched.
# ===========================================================================


def test_27_the_final_hardening_artifact_is_neither_read_nor_written(engine,
                                                                    tmp_path):
    assert sp.RESULTS_PATH != fh.RESULTS_PATH
    assert sp.RAW_DIR != fh.RAW_DIR
    assert "subcritical_poisson" in sp.RESULTS_PATH
    assert "subcritical_poisson" in sp.RAW_DIR
    text = inspect.getsource(sp)
    assert "fh.RESULTS_PATH" not in text
    assert "fh.RAW_DIR" not in text
    assert "final_hardening_results.json" in text, (
        "the path is named in SATURATED_REFERENCE, as the artifact this "
        "experiment reads (verified) but never writes")
    # The published layout is exactly the historical one...
    pub = sp.RunPaths(sp.PUBLISHED_OUTPUT_DIR)
    assert (pub.results, pub.raw_dir) == (sp.RESULTS_PATH, sp.RAW_DIR)
    # ...and a real run writes nowhere outside its own output directory.
    before = _snapshot_tree(os.path.join(_REPO_ROOT, "docs"))
    _run_smoke(tmp_path / "out")
    assert _snapshot_tree(os.path.join(_REPO_ROOT, "docs")) == before
    assert sp.EXPERIMENT == "subcritical_poisson"


def test_27b_the_saturated_reference_is_read_only_data_and_states_its_wording():
    ref = sp.SATURATED_REFERENCE
    # Recovered 2026-10-22: present, and the transcription is labelled as
    # the rounded historical record H1/H2 never read.
    assert ref["source_artifact_present_in_repo"] is True
    assert "never used for H1 or H2" in ref["superseded_note"]
    assert ref["recovery"]["source_zip_sha256"] == (
        "39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae")
    assert "missing_for_hypotheses" not in ref
    a = ref["comparisons"]["A_CARL_Repaired_minus_Static_Tput"]
    assert a["mean_difference"] == -3.3678 and a["ci95"] == [-3.6633, -3.0723]
    assert a["cohens_d_paired"] == -7.06395
    b = ref["comparisons"]["B_CARL_Repaired_minus_Static_Utility"]
    assert b["metric"] == "run_utility_v2" and b["mean_difference"] == -0.0245
    assert "useful but insufficient" in ref["allowed_wording"]
    for retracted in ("108.2", "23.2", "2.3e-41", "0.45%", "cycle-9", "28.51"):
        assert retracted not in json.dumps(ref)


# ===========================================================================
# Statistics: what is reported, and what is not headlined.
# ===========================================================================


def test_paired_reports_signs_effect_size_and_an_identity_explicitly():
    p = sp._paired([3.0, 4.0, 5.0], [1.0, 5.0, 2.0])
    assert p["per_seed_difference"] == [2.0, -1.0, 3.0]
    assert p["sign_of_per_seed_difference"] == [1, -1, 1]
    assert (p["n_positive"], p["n_negative"], p["n_zero"]) == (2, 1, 0)
    assert p["signs_all_same_direction"] is False
    assert p["cohens_d_paired"] is not None
    same = sp._paired([1.0, 2.0], [1.0, 2.0])
    assert same["all_diffs_zero"] is True
    assert same["wilcoxon_signed_rank"]["applicable"] is False, (
        "a bit-identical pair is an IDENTITY, not a tie to be broken")


def test_the_exact_wilcoxon_matches_the_textbook_and_is_never_headlined():
    # n = 5, every difference positive: the most extreme outcome, p = 2/2^5.
    w = sp.wilcoxon_signed_rank_exact([1.0, 2.0, 3.0, 4.0, 5.0])
    assert w["applicable"] is True
    assert w["w_plus"] == 15.0 and w["w_minus"] == 0.0
    assert w["p_value_two_sided_exact"] == pytest.approx(2 / 32)
    # A perfectly symmetric sample cannot be extreme.
    assert sp.wilcoxon_signed_rank_exact(
        [1.0, -1.0, 2.0, -2.0])["p_value_two_sided_exact"] == pytest.approx(1.0)
    assert "robustness" in w["reporting_note"]
    assert "never the headline" in sp._paired([1.0], [0.0])["reporting_rule"]


def test_the_midranks_average_ties():
    assert sp._midranks([5.0, 5.0, 1.0]) == [2.5, 2.5, 1.0]


# ===========================================================================
# Pre-registration: written down before the run, and not outcome-shopped.
# ===========================================================================


def test_the_preregistration_document_exists_and_names_every_hypothesis():
    assert os.path.isfile(sp.PREREG_PATH)
    text = open(sp.PREREG_PATH, encoding="utf-8").read()
    for key in ("H1", "H2", "H3", "H4", "H5"):
        assert f"### {key}" in text
    assert sp.PRE_REGISTRATION["document"].endswith(
        "PREREGISTRATION_subcritical_poisson.md")
    for phrase in ("lambda_phase = TARGET_RHO * mu_phase", "interactive@eager",
                   "batch@eager", "2.9613881136988742", "1.4135291672527908",
                   "Static-Tput", "Static-Utility"):
        assert phrase in text
    assert "will not be rewritten" in text.lower()


def test_no_winner_is_preregistered_and_all_three_cases_are_written_down():
    h3 = sp.PRE_REGISTRATION["H3_direction_is_not_preregistered"]
    assert set(h3) >= {"statement", "case_A", "case_B", "case_C"}
    for case in ("case_A", "case_B", "case_C"):
        assert h3[case]["outcome"] and h3[case]["interpretation"]
    assert "boundary condition" in h3["case_B"]["interpretation"].lower()
    assert "survives across offered-load regimes" in h3["case_A"]["interpretation"]
    mech = sp.PRE_REGISTRATION["mechanistic_hypothesis"]
    assert "less configuration-dependent throughput headroom" in mech
    assert "smaller relative to its finite-horizon search" in mech


def test_the_research_questions_are_the_seven_asked_for():
    assert sorted(sp.RESEARCH_QUESTIONS) == [f"Q{i}" for i in range(1, 8)]
    assert "no significance test" in sp.RESEARCH_QUESTIONS["Q1"].lower()


def test_every_held_fixed_factor_is_read_from_the_saturated_harness():
    """Reading them rather than retyping them is what makes 'held fixed' true by
    construction instead of by inspection."""
    assert sp.DEFAULT_SEEDS == fh.DEFAULT_SEEDS
    assert sp.DEFAULT_REQUESTS == fh.DEFAULT_REQUESTS == 200
    assert sp.OBSERVE_INTERVAL == fh.OBSERVE_INTERVAL == abl.OBSERVE_INTERVAL == 10
    assert sp.ALPHA == fh.ALPHA == 0.5
    assert sp.GRID_BATCH_AXIS == fh.GRID_BATCH_AXIS
    assert sp.GRID_CHUNK_AXIS == fh.GRID_CHUNK_AXIS
    assert FEATURE_DIM == 10
    text = inspect.getsource(sp)
    for call in ("abl._new_scheduler", "abl._serve", "abl._apply_sched",
                 "fh._build_bandit", "fh._cycle_records", "fh.run_utility",
                 "fh.freeze_scales", "fh.select_static_baselines",
                 "fh.convergence_diagnostic", "fh.probe_live_dimensions"):
        assert call in text, f"{call} should be reused, not reimplemented"


def test_it_refuses_to_run_without_cuda_unless_explicitly_smoked(engine, tmp_path):
    for argv in (["--output-dir", str(tmp_path / "a")],
                 ["--smoke", "--output-dir", str(tmp_path / "b")]):
        with pytest.raises(SystemExit, match="REFUSES to run without CUDA"):
            sp.main([*_SMALL, *argv])
    assert engine.loads == 0
    _run_smoke(tmp_path / "c")                     # --allow-cpu-smoke runs
    assert _load(_paths(tmp_path / "c").results)["cpu_smoke"] is True


# ===========================================================================
# HARDENING 2. The phases are sequential, and that is an invariant.
# ===========================================================================


def test_h2_phase_boundary_is_recorded_with_every_field():
    plan = _plan(lam_i=8.0, lam_b=0.5)
    specs, arr = sp.build_poisson_workload(_Tokenizer(), 200, 42, plan)
    b = sp.phase_boundary_report(specs, arr, 200)
    for field in ("first_interactive_arrival_s", "last_interactive_arrival_s",
                  "first_batch_arrival_s", "last_batch_arrival_s",
                  "phase_boundary_gap_s"):
        assert field in b, f"phase boundary field {field} is not recorded"
    assert b["first_interactive_arrival_s"] == 0.0
    assert b["n_interactive"] == 100 and b["n_batch"] == 100
    assert b["split_index"] == 100


def test_h2_batch_never_begins_before_interactive_ends():
    """THE INVARIANT. Two lambdas mean nothing if the stretches they govern
    overlap: neither phase would then have run at its stated rate."""
    plan = sp.derive_service_rates("eager")
    for seed in (42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 999):
        specs, arr = sp.build_poisson_workload(_Tokenizer(), 200, seed, plan)
        b = sp.phase_boundary_report(specs, arr, 200)
        assert b["first_batch_arrival_s"] >= b["last_interactive_arrival_s"], (
            f"seed {seed}: the BATCH stream starts inside the INTERACTIVE one")
        assert b["phase_boundary_gap_s"] >= 0.0
        assert b["phases_do_not_overlap"] is True
        assert all(x >= b["last_interactive_arrival_s"] for x in arr["batch"])
        assert all(x <= b["first_batch_arrival_s"] for x in arr["interactive"])


def test_h2_request_indices_match_their_phases():
    plan = sp.derive_service_rates("eager")
    specs, arr = sp.build_poisson_workload(_Tokenizer(), 200, 42, plan)
    b = sp.phase_boundary_report(specs, arr, 200)
    assert b["head_is_all_interactive"] is True
    assert b["tail_is_all_batch"] is True
    assert b["batch_requests_in_head"] == 0
    assert b["interactive_requests_in_tail"] == 0
    assert [s.phase for s in specs[:100]] == [0] * 100
    assert [s.phase for s in specs[100:]] == [1] * 100


def test_h2_an_overlapping_splice_raises():
    """Phase 2 beginning before phase 1's scheduled window ends must abort, not
    be annotated -- the run could not be reported at either rate."""
    plan = _plan()
    specs, arr = sp.build_poisson_workload(_Tokenizer(), 40, 42, plan)
    bad = {"interactive": list(arr["interactive"]),
           "batch": [arr["interactive"][-1] - 1.0] + list(arr["batch"][1:])}
    with pytest.raises(sp.PhaseSpliceError, match="overlap"):
        sp.assert_phase_splice(specs, bad, 40)


def test_h2_a_batch_request_among_the_first_half_raises():
    plan = _plan()
    specs, arr = sp.build_poisson_workload(_Tokenizer(), 40, 42, plan)
    specs[5].phase = 1
    with pytest.raises(sp.PhaseSpliceError, match="BATCH request"):
        sp.assert_phase_splice(specs, arr, 40)


def test_h2_an_interactive_request_after_the_split_raises():
    plan = _plan()
    specs, arr = sp.build_poisson_workload(_Tokenizer(), 40, 42, plan)
    specs[-1].phase = 0
    with pytest.raises(sp.PhaseSpliceError, match="INTERACTIVE request"):
        sp.assert_phase_splice(specs, arr, 40)


def test_h2_the_generator_enforces_the_invariant_itself():
    """Not merely available to a caller: build_poisson_workload asserts it, so
    no run can ever serve an overlapping splice."""
    assert "assert_phase_splice(specs, arrivals, n)" in inspect.getsource(
        sp.build_poisson_workload)
    # And the open-loop release is still what carries it -- the historical
    # closed-loop dump is NOT restored.
    assert inspect.signature(abl._serve).parameters[
        "phase1_open_loop"].default is False
    for fn in (sp.run_static, sp.run_one):
        assert "phase1_open_loop=True" in inspect.getsource(fn)


def test_h2_the_boundary_is_reported_per_run_and_per_seed(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    raw = _load(_paths(out).eval_raw("Static-Tput", 43))
    assert raw["phase_boundary"]["phases_do_not_overlap"] is True
    agg = _load(_paths(out).results)["configs"]["Static-Tput"]
    assert set(agg["phase_boundary_per_seed"]) == {"42", "43"}


# ===========================================================================
# HARDENING 3. Flatness diagnostics that cannot touch the selection.
# ===========================================================================


def _ids(n):
    return [{"max_batch_size": 2 * (i + 1), "chunk_size": 64} for i in range(n)]


def test_h3_flatness_reports_every_required_field():
    vals = [10.0, 9.98, 9.95, 9.0, 5.0]
    f = sp.flatness_diagnostics(vals, _ids(5))
    assert f["best"] == 10.0
    assert f["second_best"] == 9.98
    assert f["best_minus_second_best"] == pytest.approx(0.02)
    assert f["best_minus_second_best_relative_pct"] == pytest.approx(0.2)
    assert f["min"] == 5.0 and f["max"] == 10.0
    assert f["range_absolute"] == pytest.approx(5.0)
    assert f["range_relative_pct"] == pytest.approx(50.0)
    # Within 0.5% of 10.0 means >= 9.95: three configs. Within 1.0% means
    # >= 9.90: the same three.
    assert f["within_0p5pct_of_best"]["count"] == 3
    assert f["within_1p0pct_of_best"]["count"] == 3
    assert len(f["within_0p5pct_of_best"]["configurations"]) == 3
    assert f["best_configuration"] == {"max_batch_size": 2, "chunk_size": 64}
    assert f["second_best_configuration"] == {"max_batch_size": 4,
                                              "chunk_size": 64}


def test_h3_a_near_flat_surface_is_visible_as_one():
    """The case the diagnostics exist for: an argmax separated from the field by
    less than the measurement could resolve."""
    f = sp.flatness_diagnostics([10.0, 9.999, 9.998, 9.997], _ids(4))
    assert f["best_minus_second_best_relative_pct"] < 0.02
    assert f["within_0p5pct_of_best"]["count"] == 4, (
        "every configuration is within half a percent -- the argmax is noise")


def test_h3_diagnostics_do_not_change_the_baselines(monkeypatch):
    """Computed AFTER the winners are fixed, consumed by nothing, and returning
    no configuration anyone acts on."""
    rows = [_row(2, 64, 60.0, 20000.0, 200.0),
            _row(12, 256, 120.0, 30000.0, 250.0),
            _row(32, 512, 140.0, 200000.0, 900.0)]
    cfg_t, cfg_u, sel = fh.select_static_baselines(rows, SCALES)
    lg = sp.landscape_report(rows, SCALES, sel)
    cfg_t2, cfg_u2, sel2 = fh.select_static_baselines(rows, SCALES)
    assert (cfg_t, cfg_u) == (cfg_t2, cfg_u2)
    assert sel["static_tput"]["winner"] == sel2["static_tput"]["winner"]
    assert lg["static_tput_identity"] == sel["static_tput"]["winner"]
    assert lg["static_utility_identity"] == sel["static_utility"]["winner"]

    # The calibration (which fixes the winners) cannot reach the diagnostics:
    # make them explode and the calibration is unaffected.
    monkeypatch.setattr(sp, "flatness_diagnostics",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cal = sp.compute_calibration(rows, 200, "f" * 64)
    frozen, _ = fh.freeze_scales(rows, 200, 999)
    want_t, want_u, _ = fh.select_static_baselines(rows, frozen)
    assert cal["binding"]["static_tput"] == want_t.as_dict()
    assert cal["binding"]["static_utility"] == want_u.as_dict()
    assert "flatness" not in inspect.getsource(fh.select_static_baselines)
    assert "flatness" not in inspect.getsource(sp.run_one)
    assert "flatness" not in inspect.getsource(sp.gate_sweep)


def test_h3_no_post_hoc_tiebreak_is_introduced():
    src = inspect.getsource(sp.flatness_diagnostics)
    assert "does NOT introduce a tie-break" in src
    for smell in ("tie_break", "tiebreak", "prefer_smaller", "reselect",
                  "re-select"):
        assert smell not in inspect.getsource(sp)
    assert "arrival_limited_caveat" in inspect.getsource(sp.landscape_report)


def test_h3_the_arrival_limited_caveat_is_preregistered():
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0)]
    _t, _u, sel = fh.select_static_baselines(rows, SCALES)
    lg = sp.landscape_report(rows, SCALES, sel)
    assert "arrival-limited" in lg["arrival_limited_caveat"]
    assert "pre-registered argmax" in lg["arrival_limited_caveat"]
    text = open(sp.PREREG_PATH, encoding="utf-8").read()
    assert "arrival-limited" in text
    assert "within 0.5%" in text and "within 1.0%" in text
    assert "no post-hoc tie-break" in text.lower()


def test_h3_both_objectives_get_the_diagnostics():
    rows = [_row(2, 64, 60.0, 20000.0, 200.0), _row(12, 256, 120.0, 30000.0, 250.0),
            _row(32, 512, 140.0, 200000.0, 900.0)]
    _t, _u, sel = fh.select_static_baselines(rows, SCALES)
    lg = sp.landscape_report(rows, SCALES, sel)
    for objective in ("throughput_tps", "utility_v2"):
        f = lg[objective]["flatness"]
        assert f["n"] == 3
        assert f["best_minus_second_best"] is not None
        assert f["within_0p5pct_of_best"]["count"] >= 1
        assert f["within_1p0pct_of_best"]["count"] >= 1


# ===========================================================================
# HARDENING 4. Deterministic checkpoint resume. AN EXECUTION FEATURE ONLY.
#
# Unit-level checks of the fingerprint and the resume reader, then the real
# orchestration: everything below "HARDENING 5" drives the ACTUAL main() and
# run_all() on CPU, with only the model loader, the scheduler and the harness
# clock replaced by fakes. No GPU, no model download, no network.
# ===========================================================================


_SOURCE_CHECK = {"archive_sha256": "aa" * 32,
                 "archive_manifest_sha256": "dd" * 32,
                 "executing_manifest_sha256": "cc" * 32,
                 "working_tree_diff_sha256": "bb" * 32}


def _fp(**over):
    base = sp.experiment_fingerprint(
        sp.derive_service_rates("eager"), [42, 43], 200, [4, 8], [128, 256],
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="torch.float16",
        source_check=dict(_SOURCE_CHECK))
    base.update(over)
    return base


def test_h4_unit_plan_is_one_deterministic_order():
    units = sp.unit_plan([4, 8], [128, 256], [42, 43])
    assert units[:4] == ["sweep:mb4_cs128", "sweep:mb4_cs256",
                         "sweep:mb8_cs128", "sweep:mb8_cs256"]
    assert units[4:8] == ["eval:Static-Tput:seed042", "eval:Static-Tput:seed043",
                          "eval:Static-Utility:seed042",
                          "eval:Static-Utility:seed043"]
    assert len(units) == 4 + len(sp.CONFIGS) * 2
    assert len(set(units)) == len(units), "unit ids must be unique"
    assert units == sp.unit_plan([4, 8], [128, 256], [42, 43])


def test_h4_fingerprint_covers_everything_a_resume_must_agree_on():
    fp = _fp()
    for field in ("experiment", "experiment_version", "run_mode", "target_rho",
                  "mu_interactive_req_per_s", "mu_batch_req_per_s",
                  "lambda_interactive_rps", "lambda_batch_rps", "seeds",
                  "held_out_seed", "n_requests", "batch_grid", "chunk_grid",
                  "model", "dtype", "arrival_mode", "workload_version",
                  "reward_version", "convergence_rule_sha256",
                  "arm_space_sha256", "source_archive_sha256",
                  "archive_manifest_sha256", "executing_source_manifest_sha256",
                  "working_tree_diff_sha256", "preregistration_sha256"):
        assert field in fp, f"fingerprint is missing {field}"
    assert fp["target_rho"] == 0.40
    assert fp["held_out_seed"] == 999
    assert fp["lambda_interactive_rps"] == pytest.approx(1.1845552454795498)
    assert fp["source_archive_sha256"] == "aa" * 32
    assert fp["executing_source_manifest_sha256"] == "cc" * 32
    assert fp["convergence_rule_sha256"] == sp._sha256_text(
        sp._canonical(fh.CONVERGENCE_RULE))


def test_h4_identical_fingerprints_compare_equal():
    assert sp.compare_fingerprints(_fp(), _fp()) == []


@pytest.mark.parametrize("key,value", [
    ("target_rho", 0.7), ("lambda_interactive_rps", 2.0),
    ("lambda_batch_rps", 0.1), ("mu_batch_req_per_s", 9.9),
    ("seeds", [42, 43, 44]), ("n_requests", 100),
    ("batch_grid", [2, 4, 8, 12, 16, 24, 32]), ("chunk_grid", [64, 128]),
    ("model", "some/other-model"), ("dtype", "torch.float32"),
    ("reward_version", "v1"), ("convergence_rule_sha256", "deadbeef"),
    ("arm_space_sha256", "deadbeef"), ("source_archive_sha256", "ff" * 32),
    ("archive_manifest_sha256", "ff" * 32),
    ("executing_source_manifest_sha256", "ff" * 32),
    ("preregistration_sha256", "ff" * 32), ("run_mode", "smoke"),
    ("experiment_version", "subcritical_poisson/1"),
])
def test_h4_any_material_change_is_detected(key, value):
    diffs = sp.compare_fingerprints(_fp(), _fp(**{key: value}))
    assert len(diffs) == 1 and diffs[0].startswith(key + ":")


def test_h4_a_missing_fingerprint_is_a_mismatch():
    assert sp.compare_fingerprints(None, _fp()) == [
        "<no fingerprint recorded in the artifact>"]
    assert sp.compare_fingerprints({}, _fp())


def _write_checkpoint(out, fingerprint, completed=(), **extra):
    paths = sp.RunPaths(str(out))
    os.makedirs(paths.output_dir, exist_ok=True)
    with open(paths.results, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint,
                   "checkpoint": {"units_completed": list(completed)}, **extra}, f)
    return paths


def test_h4_resume_loads_matching_state(tmp_path):
    fp = _fp()
    paths = _write_checkpoint(tmp_path, fp, ["sweep:mb4_cs128"])
    prior = sp.load_resume_state(paths, fp)
    assert prior["checkpoint"]["units_completed"] == ["sweep:mb4_cs128"]


def test_h4_resume_with_nothing_to_resume_from_aborts(tmp_path):
    with pytest.raises(sp.ResumeMismatchError, match="no artifact"):
        sp.load_resume_state(sp.RunPaths(str(tmp_path)), _fp())


def test_h4_resume_across_an_unparseable_artifact_aborts(tmp_path):
    paths = sp.RunPaths(str(tmp_path))
    with open(paths.results, "w", encoding="utf-8") as f:
        f.write("{ not json")
    with pytest.raises(sp.ResumeMismatchError, match="could not parse"):
        sp.load_resume_state(paths, _fp())


@pytest.mark.parametrize("key", ["lambda_batch_rps", "seeds",
                                 "source_archive_sha256",
                                 "executing_source_manifest_sha256",
                                 "working_tree_diff_sha256",
                                 "convergence_rule_sha256", "arm_space_sha256"])
def test_h4_a_changed_key_aborts_the_resume(tmp_path, key):
    paths = _write_checkpoint(tmp_path, _fp(**{key: "changed"}))
    with pytest.raises(sp.ResumeMismatchError, match=key) as exc:
        sp.load_resume_state(paths, _fp())
    assert "REFUSED" in str(exc.value)


def test_h4_a_prior_gate_failure_refuses_the_resume(tmp_path):
    """Including artifacts written by the previous version, whose abort record
    named the reason but had no status."""
    for extra in ({"aborted": {"reason": "DegenerateRewardError"}},
                  {"aborted": {"status": sp.STATUS_TERMINAL_GATE}},
                  {"terminal_failure": {"unit": "eval:RuleOnly:seed042"}},
                  {"run_history": [{"status": sp.STATUS_TERMINAL_GATE}]}):
        paths = _write_checkpoint(tmp_path / str(len(str(extra))), _fp(), **extra)
        with pytest.raises(sp.TerminalGateFailureError):
            sp.load_resume_state(paths, _fp())


def test_h4_the_ledger_records_every_rejection_and_recovery():
    led = sp.ResumeLedger(_fp(), ["sweep:mb4_cs128", "eval:Static-Tput:seed042"])
    assert led.claimed("sweep:mb4_cs128") and not led.claimed("sweep:mb8_cs128")
    led.accept("sweep:mb4_cs128")
    led.reject("eval:Static-Tput:seed042", "its evidence file is missing")
    led.note("calibration", "recovered", "from sweep evidence")
    d = led.as_dict()
    assert d["units_completed"] == ["sweep:mb4_cs128"]
    assert d["units_skipped_on_resume"] == ["sweep:mb4_cs128"]
    assert d["units_rerun_on_resume"][0]["unit"] == "eval:Static-Tput:seed042"
    assert [a["action"] for a in d["recovery_audit"]] == ["rerun", "recovered"]


def test_h4_resume_is_documented_as_execution_only():
    assert "EXECUTION" in sp.main.__code__.co_consts.__repr__() or \
        "EXECUTION" in inspect.getsource(sp.main)
    text = open(sp.PREREG_PATH, encoding="utf-8").read()
    assert "execution feature only" in text.lower()
    assert "changes **no scientific quantity**" in text


# ===========================================================================
# HARDENING 5. The orchestration itself -- real main() / run_all() on fakes.
# ===========================================================================


class _Clock:
    """The harness's `time` module, virtualised: deterministic and instant."""

    def __init__(self):
        self.t = 1000.0

    def perf_counter(self):
        return self.t

    def sleep(self, s):
        # +1e-6 so a sleep "until the next arrival" lands past it despite float
        # rounding, as a real sleep would.
        self.t += max(0.0, s) + 1e-6

    def time(self):
        return self.t


class _FakeReq:
    def __init__(self, rid, plen, max_new):
        self.request_id = rid
        self.prompt_len = plen
        self.remaining = max_new
        self.prefilled = 0


class _FakeScheduler:
    """ContinuousBatchScheduler-shaped: admission by max_batch_size, prefill by
    chunk_size, one token per active request per step, and a step cost that
    depends on both knobs -- so the live plane is not flat."""

    def __init__(self, engine):
        self.clock = engine.clock
        self.clock.t = 1000.0             # every run starts from the same clock
        self.max_batch_size = 8
        self.chunk_size = 256
        self.enable_spec_decode = False
        self._graph_runner = None
        self._step_idx = 0
        self.waiting, self.active, self._finished = [], [], []

    def add_request(self, rid, prompt_ids, max_new_tokens, eos_token_id=None):
        self.waiting.append(_FakeReq(rid, int(prompt_ids.shape[-1]), max_new_tokens))

    def has_work(self):
        return bool(self.waiting or self.active)

    def step(self):
        while self.waiting and len(self.active) < self.max_batch_size:
            self.active.append(self.waiting.pop(0))
        budget, prefill, emitted = self.chunk_size, 0, []
        for r in self.active:
            if r.prefilled < r.prompt_len:
                take = min(budget, r.prompt_len - r.prefilled)
                r.prefilled += take
                budget -= take
                prefill += take
        for r in self.active:
            if r.prefilled >= r.prompt_len and r.remaining > 0:
                r.remaining -= 1
                emitted.append((r.request_id, 0))
        self.clock.t += (0.02 + 0.006 * len(self.active) + 0.0004 * prefill
                         + 0.00002 * self.chunk_size)
        done = [r for r in self.active if r.remaining == 0]
        self.active = [r for r in self.active if r.remaining > 0]
        self._finished += done
        self._step_idx += 1
        return emitted

    def get_finished(self):
        out, self._finished = self._finished, []
        return out


class _Param:
    def __init__(self, device):
        self.device = device


class _FakeModel:
    """Only what the driver asks of a model: where its parameters live."""

    def __init__(self, engine):
        self._engine = engine

    def parameters(self):
        dev = self._engine.model_device
        return iter([_Param(sp.DEVICE if dev is None else dev)])


_FAKE_MODEL_ID = {"resolved": True, "repo_id": sp.MODEL_NAME,
                  "revision": "a" * 40,
                  "files": {"config.json": {"sha256": "b" * 64, "bytes": 1}},
                  "identity_sha256": "c" * 64}


class _Engine:
    def __init__(self):
        self.clock = _Clock()
        self.loads = 0
        self.load_error = None
        self.served = []          # (has_controller, arrivals by phase)
        self.serves = 0
        self.fail_at_serve = None
        self.model_device = None   # None: wherever sp.DEVICE says
        self.gpu = {"name": "Tesla T4", "capability": "7.5",
                    "total_memory_bytes": 15828320256}
        self.model_id = dict(_FAKE_MODEL_ID)

    def load(self, dtype):
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error
        return _Tokenizer(), _FakeModel(self)


@pytest.fixture
def engine(monkeypatch):
    eng = _Engine()
    real_serve = abl._serve
    monkeypatch.setattr(sp, "_cuda_device_props", lambda idx: dict(eng.gpu))
    monkeypatch.setattr(sp, "model_identity", lambda name=None: dict(eng.model_id))

    def _serve(sched, specs, **kw):
        eng.serves += 1
        if eng.fail_at_serve is not None and eng.serves >= eng.fail_at_serve:
            raise KeyboardInterrupt("simulated Colab disconnect")
        eng.served.append((kw.get("controller") is not None, {
            "interactive": [s.arrival_offset_s for s in specs if s.phase == 0],
            "batch": [s.arrival_offset_s for s in specs if s.phase == 1]}))
        return real_serve(sched, specs, **kw)

    monkeypatch.setattr(abl, "time", eng.clock)
    monkeypatch.setattr(abl, "_new_scheduler", lambda model: _FakeScheduler(eng))
    monkeypatch.setattr(abl, "_serve", _serve)
    monkeypatch.setattr(sp, "load_model_and_tokenizer", eng.load)
    return eng


_N = 20
_SMALL = ["--seeds", "42,43", "--limit", str(_N), "--grid-batches", "4,8",
          "--grid-chunks", "128,256"]
_N_SERVES = 4 + 5 * 2


def _paths(out):
    return sp.RunPaths(str(out))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_smoke(out, *extra):
    sp.main(["--allow-cpu-smoke", *_SMALL, "--output-dir", str(out), *extra])


def _snapshot_tree(root):
    """{relative path: bytes} of every file under root."""
    snap = {}
    if os.path.isdir(root):
        for d, _dirs, files in os.walk(root):
            for f in files:
                full = os.path.join(d, f)
                with open(full, "rb") as fh_:
                    snap[os.path.relpath(full, root)] = fh_.read()
    return snap


def _interrupted(engine, out, fail_at, *extra):
    engine.serves, engine.fail_at_serve = 0, fail_at
    with pytest.raises(KeyboardInterrupt):
        _run_smoke(out, *extra)
    engine.fail_at_serve = None


def _science(results):
    """Everything a resume must not be able to move."""
    return {
        "calibration": results["calibration"]["calibration_sha256"],
        "binding": results["calibration"]["binding"],
        "rows": [(r["max_batch_size"], r["chunk_size"], r["throughput_tps"],
                  r["ttft_p99_ms"], r["tpot_p99_ms"])
                 for r in results["held_out_sweep_rows"]],
        "comparisons": {k: (v[v["metric"]]["mean_difference"],
                            v[v["metric"]]["per_seed_difference"])
                        for k, v in results["comparisons"].items()},
        "per_seed": {k: v["_per_seed"] for k, v in results["configs"].items()},
    }


# --- 1. startup failures never touch an existing checkpoint ----------------


@pytest.mark.parametrize("failure", ["model_load", "keyboard_interrupt",
                                     "liveness", "bad_mu_path"])
def test_o1_startup_failure_during_resume_preserves_checkpoint_bytes(
        engine, tmp_path, monkeypatch, failure):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)                 # a real, resumable checkpoint
    before = _snapshot_tree(str(out))
    extra = ["--resume"]
    if failure == "model_load":
        engine.load_error = OSError("HF download timed out")
        expected = OSError
    elif failure == "keyboard_interrupt":
        engine.load_error = KeyboardInterrupt()
        expected = KeyboardInterrupt
    elif failure == "liveness":
        monkeypatch.setattr(fh, "probe_live_dimensions", lambda s: (
            _ for _ in ()).throw(RuntimeError("spec_k is live")))
        expected = RuntimeError
    else:
        extra += ["--mu-artifact", "docs/eval/no_such_sweep.json"]
        expected = sp.AmbiguousServiceRateError
    with pytest.raises(expected):
        _run_smoke(out, *extra)

    after = _snapshot_tree(str(out))
    records = [k for k in after if k.startswith(
        "subcritical_poisson_preflight_errors")]
    assert len(records) == 1, "the start-up failure is recorded in its own file"
    assert {k: v for k, v in after.items() if k not in records} == before, (
        "a start-up failure overwrote evidence it never owned")
    assert _load(os.path.join(str(out), records[0]))["checkpoint_modified"] is False


def test_o1b_a_refused_resume_leaves_every_byte_untouched(engine, tmp_path):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)
    before = _snapshot_tree(str(out))
    with pytest.raises(sp.ResumeMismatchError, match="seeds"):
        sp.main(["--allow-cpu-smoke", "--seeds", "42", "--limit", str(_N),
                 "--grid-batches", "4,8", "--grid-chunks", "128,256",
                 "--output-dir", str(out), "--resume"])
    after = _snapshot_tree(str(out))
    assert {k: v for k, v in after.items()
            if "preflight_errors" not in k} == before
    assert engine.loads == 1, "the refusal came BEFORE loading the model"


# --- 2. atomic writes --------------------------------------------------------


class _Unserialisable:
    def __str__(self):
        raise RuntimeError("cannot serialise")


def test_o2_failed_serialisation_preserves_the_previous_checkpoint(tmp_path):
    path = str(tmp_path / "ck.json")
    sp.atomic_write_json(path, {"good": 1})
    before = open(path, "rb").read()
    with pytest.raises(RuntimeError, match="cannot serialise"):
        sp.atomic_write_json(path, {"bad": _Unserialisable()})
    assert open(path, "rb").read() == before
    assert os.listdir(str(tmp_path)) == ["ck.json"], "no temp file left behind"


def test_o2b_an_interrupted_replace_preserves_the_previous_checkpoint(
        tmp_path, monkeypatch):
    path = str(tmp_path / "ck.json")
    sp.atomic_write_json(path, {"good": 1})
    before = open(path, "rb").read()
    import source_provenance as srcprov
    monkeypatch.setattr(srcprov.os, "replace", lambda a, b: (
        _ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        sp.atomic_write_json(path, {"good": 2})
    assert open(path, "rb").read() == before
    assert os.listdir(str(tmp_path)) == ["ck.json"]


def test_o2c_a_mid_run_serialisation_failure_keeps_the_last_good_checkpoint(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    real = sp.write_artifact
    calls = {"n": 0}

    def _poisoned(results, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            results["poison"] = _Unserialisable()
        return real(results, *a, **k)

    monkeypatch.setattr(sp, "write_artifact", _poisoned)
    with pytest.raises(RuntimeError, match="cannot serialise"):
        _run_smoke(out)
    ck = _load(_paths(out).results)          # still parses: the last good one
    assert ck["stage"] == "held_out_sweep"
    assert len(ck["held_out_sweep_rows"]) == 1


# --- 3. every completed sweep row survives repeated interruption -------------


def test_o3_repeated_interruption_during_sweep_replay_preserves_every_row(
        engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 4)                 # sweep units 1-3 done
    assert len(_load(paths.results)["held_out_sweep_rows"]) == 3
    # Lose the FIRST unit's evidence: the restored set is now NOT a prefix of
    # the grid, which is exactly when a visited-prefix checkpoint loses rows.
    os.remove(paths.sweep_raw(4, 128))
    _interrupted(engine, out, 1, "--resume")     # dies re-measuring unit 1
    ck = _load(paths.results)
    keys = [(r["max_batch_size"], r["chunk_size"]) for r in ck["held_out_sweep_rows"]]
    assert keys == [(4, 256), (8, 128)], (
        "the checkpoint must hold EVERY verified row in grid order")
    rerun = ck["checkpoint"]["units_rerun_on_resume"]
    assert [r["unit"] for r in rerun] == ["sweep:mb4_cs128"]
    _interrupted(engine, out, 1, "--resume")     # and again
    assert _load(paths.results)["held_out_sweep_rows"] == ck["held_out_sweep_rows"]


# --- 4. resumed == uninterrupted ---------------------------------------------


def test_o4_uninterrupted_and_interrupted_resumed_runs_match(engine, tmp_path):
    whole, split = tmp_path / "whole", tmp_path / "split"
    _run_smoke(whole)
    assert engine.serves == _N_SERVES

    _interrupted(engine, split, 3)               # inside the sweep
    _interrupted(engine, split, 7, "--resume")   # inside the evaluations
    engine.serves = 0
    _run_smoke(split, "--resume")
    a, b = _load(_paths(whole).results), _load(_paths(split).results)
    assert _science(a) == _science(b)
    assert [h["status"] for h in b["run_history"]] == [
        sp.STATUS_OPERATIONAL, sp.STATUS_OPERATIONAL, "complete"]
    assert b["checkpoint"]["units_completed"] == sp.unit_plan(
        [4, 8], [128, 256], [42, 43])
    assert b["run_status"] == "complete"


def test_o5_valid_completed_units_are_skipped(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    first = _load(_paths(out).results)
    engine.serves = 0
    _run_smoke(out, "--resume")
    assert engine.serves == 0, "nothing re-measured"
    second = _load(_paths(out).results)
    assert second["checkpoint"]["units_rerun_on_resume"] == []
    assert len(second["checkpoint"]["units_skipped_on_resume"]) == 10
    assert _science(first) == _science(second)


def test_o6_invalid_or_missing_units_are_rerun_and_audited(engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _run_smoke(out)
    nan = _load(paths.eval_raw("Static-Tput", 42))
    nan["throughput_tps"] = float("nan")
    with open(paths.eval_raw("Static-Tput", 42), "w", encoding="utf-8") as f:
        json.dump(nan, f)
    os.remove(paths.eval_raw("RuleOnly", 43))
    foreign = _load(paths.eval_raw("CARL-Expanded", 42))
    foreign["seed"] = 43
    with open(paths.eval_raw("CARL-Expanded", 43), "w", encoding="utf-8") as f:
        json.dump(foreign, f)

    engine.serves = 0
    _run_smoke(out, "--resume")
    assert engine.serves == 3
    rerun = {r["unit"] for r in
             _load(paths.results)["checkpoint"]["units_rerun_on_resume"]}
    assert rerun == {"eval:Static-Tput:seed042", "eval:RuleOnly:seed043",
                     "eval:CARL-Expanded:seed043"}
    # Untrusted files are moved aside, never deleted.
    assert len(os.listdir(paths.superseded_dir)) == 2
    for name, seed in (("Static-Tput", 42), ("RuleOnly", 43),
                       ("CARL-Expanded", 43)):
        assert _load(paths.eval_raw(name, seed))["seed"] == seed


# --- 5. the calibration binding ----------------------------------------------


def test_o7_evaluations_are_bound_to_the_calibration_digest(engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _run_smoke(out)
    digest = _load(paths.results)["calibration"]["calibration_sha256"]
    cal = _load(paths.calibration)
    assert cal["calibration_sha256"] == digest
    assert set(cal["binding"]) >= {"held_out_measurements_sha256",
                                   "frozen_scales", "reward_weights",
                                   "static_tput", "static_utility"}
    for name in sp.CONFIGS:
        for seed in (42, 43):
            assert _load(paths.eval_raw(name, seed))["calibration_sha256"] == digest


def test_o7b_an_altered_sweep_cannot_create_a_replacement_calibration(
        engine, tmp_path):
    """Calibration record lost AND a sweep row changed: the recomputed
    calibration no longer matches the evaluations, so the resume refuses
    rather than mixing them -- and touches nothing."""
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)                 # evaluations have started
    os.remove(paths.calibration)
    unit = _load(paths.sweep_raw(8, 256))
    unit["row"]["throughput_tps"] *= 1.5
    with open(paths.sweep_raw(8, 256), "w", encoding="utf-8") as f:
        json.dump(unit, f)
    before = _snapshot_tree(str(out))
    with pytest.raises(sp.ResumeMismatchError, match="calibration"):
        _run_smoke(out, "--resume")
    assert {k: v for k, v in _snapshot_tree(str(out)).items()
            if "preflight_errors" not in k} == before


def test_o7c_a_sweep_that_disagrees_with_the_calibration_record_refuses(
        engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    unit = _load(paths.sweep_raw(4, 128))
    unit["row"]["ttft_p99_ms"] += 1.0
    with open(paths.sweep_raw(4, 128), "w", encoding="utf-8") as f:
        json.dump(unit, f)
    with pytest.raises(sp.ResumeMismatchError, match="disagrees"):
        _run_smoke(out, "--resume")


def test_o7d_a_lost_calibration_record_is_recovered_only_from_identical_evidence(
        engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    digest = _load(paths.calibration)["calibration_sha256"]
    os.remove(paths.calibration)
    engine.serves = 0
    _run_smoke(out, "--resume")
    assert _load(paths.calibration)["calibration_sha256"] == digest
    audit = _load(paths.results)["checkpoint"]["recovery_audit"]
    assert any(a["unit"] == "calibration" and a["action"] == "recovered"
               for a in audit)
    assert engine.serves == _N_SERVES - 6, "no sweep unit was re-measured"


def test_o7e_sweep_rows_are_restored_from_the_calibration_record(engine, tmp_path):
    """After evaluation starts, a lost sweep file is not re-measured (that
    would be a new calibration); its row comes back from the record."""
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    os.remove(paths.sweep_raw(8, 128))
    engine.serves = 0
    _run_smoke(out, "--resume")
    assert engine.serves == _N_SERVES - 6
    rows = _load(paths.results)["held_out_sweep_rows"]
    assert len(rows) == 4


# --- 6. raw evidence is validated, not merely parsed --------------------------


@pytest.fixture
def finished_run(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    results = _load(_paths(out).results)
    plan = results["arrival_plan"]
    cal = _load(_paths(out).calibration)
    ctx = sp.UnitContext(_paths(out), results["fingerprint_sha256"], _N, plan)
    rows = cal["held_out_rows"]
    ctx.calibration = sp.compute_calibration(rows, _N, results["fingerprint_sha256"])
    return out, ctx


def _mutate(payload, dotted, value=None, delete=False):
    keys = dotted.split(".")
    target = payload
    for k in keys[:-1]:
        target = target[int(k)] if isinstance(target, list) else target[k]
    last = int(keys[-1]) if isinstance(target, list) else keys[-1]
    if delete:
        del target[last]
    else:
        target[last] = value


@pytest.mark.parametrize("name,dotted,value,delete,why", [
    ("Static-Tput", "throughput_tps", float("nan"), False, "finite"),
    ("Static-Tput", "ttft_p99_ms", None, False, "finite"),
    ("Static-Tput", "seed", 43, False, "seed"),
    ("Static-Tput", "n_requests", 200, False, "n="),
    ("Static-Tput", "config", "Static-Utility", False, "config"),
    ("Static-Tput", "fingerprint_sha256", "0" * 64, False, "fingerprint"),
    ("Static-Tput", "calibration_sha256", "0" * 64, False, "calibration"),
    ("Static-Tput", "schema", "something/else", False, "schema"),
    ("Static-Tput", "applied_config.max_batch_size", 999, False, "static"),
    ("Static-Tput", "arrival_validation.batch", None, True, "arrival_validation"),
    ("Static-Tput", "offered_load.per_phase.interactive", None, True, "offered_load"),
    ("Static-Tput", "phase_boundary.tail_is_all_batch", False, False, "phase_boundary"),
    ("Static-Tput", "arrival_offsets_s.interactive.3", 0.123, False, "seeded draw"),
    ("Static-Tput", "run_utility_v2", 0.5, False, "reproduce"),
    ("CARL-Repaired", "cycles", None, True, "cycle trace"),
    ("CARL-Repaired", "convergence.supported_in_any_regime", True, False,
     "convergence"),
    ("CARL-Repaired", "decisions", 1, False, "decisions"),
    ("CARL-Repaired", "reward_validity.warmup_cycles", None, False, "warmup"),
    ("CARL-Repaired", "reward_gate.passed", False, False, "gate"),
    ("CARL-Repaired", "mean_valid_cycle_utility", 0.0, False, "mean_valid"),
])
def test_o8_untrustworthy_raw_evidence_is_rejected(finished_run, name, dotted,
                                                   value, delete, why):
    out, ctx = finished_run
    payload = _load(_paths(out).eval_raw(name, 42))
    assert sp.validate_eval_payload(payload, name, 42, ctx) == [], (
        "the untouched unit must validate, or this test proves nothing")
    _mutate(payload, dotted, value, delete)
    problems = sp.validate_eval_payload(payload, name, 42, ctx)
    assert problems and any(why in p for p in problems), problems


def test_o8b_a_foreign_raw_file_is_rejected(finished_run):
    out, ctx = finished_run
    other = _load(_paths(out).eval_raw("Static-Tput", 43))
    problems = sp.validate_eval_payload(other, "Static-Tput", 42, ctx)
    assert any("seed" in p for p in problems)


def test_o8c_sweep_units_are_validated_too(finished_run):
    out, ctx = finished_run
    payload = _load(_paths(out).sweep_raw(4, 128))
    assert sp.validate_sweep_payload(payload, 4, 128, ctx) == []
    assert sp.validate_sweep_payload(payload, 8, 128, ctx)
    payload["seed"] = 42
    assert any("held-out" in p for p in
               sp.validate_sweep_payload(payload, 4, 128, ctx))


# --- 7. terminal reward-gate failures -----------------------------------------


def _fail_gate_for(monkeypatch, name, seed):
    real = sp.gate_live_reward_stream

    def _gate(valid, *, context="", **kw):
        if f"{name} seed={seed}" in context:
            raise sp.DegenerateRewardError(f"{context}: forced for the test")
        return real(valid, context=context, **kw)

    monkeypatch.setattr(sp, "gate_live_reward_stream", _gate)


def test_o9_a_gate_failure_saves_its_evidence_and_is_terminal(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    paths = _paths(out)
    _fail_gate_for(monkeypatch, "CARL-Repaired", 43)
    with pytest.raises(sp.DegenerateRewardError):
        _run_smoke(out)
    failed = _load(os.path.join(paths.failed_dir,
                                "eval__CARL-Repaired__seed043.json"))
    assert failed["status"] == sp.STATUS_TERMINAL_GATE
    assert failed["cycles"], "the controller trace is kept"
    assert len(failed["requests"]) == _N, "the raw measurements are kept"
    assert failed["reward_gate"]["passed"] is False
    assert failed["reward_gate"]["valid_rewards"]
    ck = _load(paths.results)
    assert ck["aborted"]["status"] == sp.STATUS_TERMINAL_GATE
    assert ck["terminal_failure"]["unit"] == "eval:CARL-Repaired:seed043"
    assert ck["run_history"][-1]["status"] == sp.STATUS_TERMINAL_GATE

    # An ordinary resume refuses -- even once the gate would pass -- and
    # touches nothing.
    monkeypatch.undo()
    monkeypatch.setattr(abl, "time", engine.clock)
    monkeypatch.setattr(abl, "_new_scheduler", lambda m: _FakeScheduler(engine))
    monkeypatch.setattr(sp, "load_model_and_tokenizer", engine.load)
    before = _snapshot_tree(str(out))
    with pytest.raises(sp.TerminalGateFailureError):
        _run_smoke(out, "--resume")
    assert {k: v for k, v in _snapshot_tree(str(out)).items()
            if "preflight_errors" not in k} == before


def test_o9b_the_pre_run_sweep_gate_is_terminal_too(engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sp, "gate_candidate_arm_spread", lambda *a, **k: (
        _ for _ in ()).throw(sp.DegenerateRewardError("flat plane")))
    with pytest.raises(sp.DegenerateRewardError):
        _run_smoke(out)
    failed = _load(os.path.join(_paths(out).failed_dir, "prerun_sweep_gate.json"))
    assert len(failed["held_out_rows"]) == 4 and failed["frozen_scales"]
    assert not os.path.exists(_paths(out).calibration)
    with pytest.raises(sp.TerminalGateFailureError):
        _run_smoke(out, "--resume")


def test_o9c_an_interruption_is_not_terminal(engine, tmp_path):
    out = tmp_path / "out"
    _interrupted(engine, out, 9)
    assert _load(_paths(out).results)["aborted"]["status"] == sp.STATUS_OPERATIONAL
    _run_smoke(out, "--resume")
    assert _load(_paths(out).results)["run_status"] == "complete"


# --- 8. smoke and published runs never share a destination ------------------


class _Cuda:
    type = "cuda"
    index = 0

    def __str__(self):
        return "cuda"


def test_o10_a_cuda_smoke_is_still_a_smoke(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "DEVICE", _Cuda())
    out = tmp_path / "out"
    sp.main(["--smoke", *_SMALL, "--output-dir", str(out)])
    r = _load(_paths(out).results)
    assert r["run_mode"] == "smoke" and r["published_run"] is False
    assert r["smoke"] is True and r["cpu_smoke"] is False
    assert "NOT A RESULT" in r["scope_note"]
    assert r["fingerprint"]["run_mode"] == "smoke"


def test_o10b_a_smoke_may_not_target_the_published_paths(engine, monkeypatch):
    before = _snapshot_tree(os.path.join(_REPO_ROOT, "docs", "eval"))
    with pytest.raises(SystemExit, match="published destination"):
        sp.main(["--allow-cpu-smoke", *_SMALL, "--output-dir", sp.DOCS_EVAL])
    assert _snapshot_tree(os.path.join(_REPO_ROOT, "docs", "eval")) == before
    assert os.path.normcase(sp.SMOKE_OUTPUT_DIR) != os.path.normcase(sp.DOCS_EVAL)


def test_o10c_cuda_alone_does_not_make_a_published_run(engine, monkeypatch):
    monkeypatch.setattr(sp, "DEVICE", _Cuda())
    with pytest.raises(SystemExit, match="pre-registered run"):
        sp.main(_SMALL)                       # published mode, unregistered args
    assert engine.loads == 0


def test_o10d_a_fresh_run_refuses_to_overwrite_an_existing_one(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    before = _snapshot_tree(str(out))
    with pytest.raises(SystemExit, match="REFUSING to start a fresh run"):
        _run_smoke(out)
    assert _snapshot_tree(str(out)) == before


# --- 9. provenance: the producer's real schema, verified without git ---------


def _zip_tree(zip_path, root, members, *, transform=None, prefix=""):
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in members:
            with open(os.path.join(root, *rel.split("/")), "rb") as f:
                data = f.read()
            if transform:
                data = transform(rel, data)
            z.writestr(prefix + rel, data)


def _sidecar_for(zip_path):
    """What source_provenance.capture(--archive) writes, minus the git block."""
    import source_provenance as srcprov
    return {"schema": srcprov.SCHEMA,
            "archive": {"path": os.path.basename(zip_path),
                        "bytes": os.path.getsize(zip_path),
                        "sha256": srcprov._sha256_file(zip_path)},
            "archive_manifest": srcprov.archive_manifest(zip_path),
            "working_tree_diff_sha256": "e3" * 32}


@pytest.fixture(scope="module")
def repo_zip(tmp_path_factory):
    import source_provenance as srcprov
    d = tmp_path_factory.mktemp("zip")
    members = [e["path"] for e in srcprov.tree_manifest(_REPO_ROOT)["files"]]
    z = str(d / "upload.zip")
    _zip_tree(z, _REPO_ROOT, members)
    return z


def test_o11_the_real_sidecar_schema_is_consumed(repo_zip):
    """Producer -> consumer with the structure the producer actually writes:
    the archive digest is nested under `archive`, not at the top level."""
    side = _sidecar_for(repo_zip)
    assert "archive_sha256" not in side
    check = sp.check_source(side)
    assert check["verified"] is True, check["problems"]
    assert check["archive_sha256"] == side["archive"]["sha256"]
    fp = sp.experiment_fingerprint(sp.derive_service_rates("eager"), [42], 20,
                                   [4], [128], model="m", dtype="d",
                                   source_check=check)
    assert fp["source_archive_sha256"] == side["archive"]["sha256"]
    assert fp["executing_source_manifest_sha256"] == (
        side["archive_manifest"]["manifest_sha256"])


@pytest.mark.parametrize("damage", ["no_archive", "no_manifest", "bad_sha",
                                    "absent"])
def test_o11b_missing_or_altered_provenance_is_rejected(repo_zip, damage):
    side = _sidecar_for(repo_zip)
    if damage == "no_archive":
        side["archive"] = None
    elif damage == "no_manifest":
        side["archive_manifest"] = None
    elif damage == "bad_sha":
        side["archive"]["sha256"] = "not-a-digest"
    else:
        side = {"present": False}
    assert sp.check_source(side)["verified"] is False


def test_o11c_the_existing_v1_sidecar_cannot_clear_a_published_run():
    """The sidecar in this tree predates the archive manifest."""
    real = fh.load_source_provenance()
    if not real.get("present"):
        pytest.skip("no sidecar in this tree")
    check = sp.check_source(real)
    assert check["verified"] is False
    assert any("archive_manifest" in p for p in check["problems"])


def test_o11d_a_source_edit_is_detected_with_an_unchanged_sidecar(
        repo_zip, tmp_path, monkeypatch):
    import source_provenance as srcprov
    side = _sidecar_for(repo_zip)
    import zipfile
    tree = tmp_path / "extracted"
    with zipfile.ZipFile(repo_zip) as z:
        z.extractall(str(tree))
    monkeypatch.setattr(sp, "_REPO_ROOT", str(tree))
    clean = sp.check_source(side)
    assert clean["verified"] is True
    target = tree / "src" / "carl" / "live.py"
    target.write_bytes(target.read_bytes() + b"\n# edited on Colab\n")
    edited = sp.check_source(side)
    assert edited["verified"] is False
    assert edited["mismatched"] == ["src/carl/live.py"]
    assert edited["executing_manifest_sha256"] != clean["executing_manifest_sha256"]
    (tree / "scripts" / "helper.py").write_text("x = 1\n")
    assert sp.check_source(side)["unexpected"] == ["scripts/helper.py"]
    # ... and the fingerprint therefore refuses a resume across the edit.
    plan = sp.derive_service_rates("eager")
    a = sp.experiment_fingerprint(plan, [42], 20, [4], [128], model="m",
                                  dtype="d", source_check=clean)
    b = sp.experiment_fingerprint(plan, [42], 20, [4], [128], model="m",
                                  dtype="d", source_check=edited)
    assert [d.split(":")[0] for d in sp.compare_fingerprints(a, b)] == [
        "executing_source_manifest_sha256"]
    assert srcprov  # imported for the side effect of sharing the module


def test_o11e_a_published_run_end_to_end_requires_verified_source(
        engine, tmp_path, monkeypatch, repo_zip):
    """The published path, exercised for real: CUDA (faked), the registered
    parameters (shrunk for the test), a verified archive -- and a refusal when
    the sidecar cannot vouch for the tree."""
    pub = tmp_path / "published"
    monkeypatch.setattr(sp, "DEVICE", _Cuda())
    monkeypatch.setattr(sp, "PUBLISHED_OUTPUT_DIR", str(pub))
    monkeypatch.setattr(sp, "DEFAULT_SEEDS", [42, 43])
    monkeypatch.setattr(sp, "DEFAULT_REQUESTS", _N)
    monkeypatch.setattr(sp, "GRID_BATCH_AXIS", [4, 8])
    monkeypatch.setattr(sp, "GRID_CHUNK_AXIS", [128, 256])
    side_path = tmp_path / "SOURCE_PROVENANCE.json"
    side_path.write_text(json.dumps({"schema": "source_provenance/1"}))
    monkeypatch.setattr(fh, "SOURCE_PROVENANCE_PATH", str(side_path))
    with pytest.raises(sp.PreflightRefusal, match="archive manifest"):
        sp.main([])
    assert not os.path.exists(os.path.join(str(pub), sp.RESULTS_BASENAME))
    assert engine.loads == 0

    side_path.write_text(json.dumps(_sidecar_for(repo_zip)))
    sp.main([])
    r = _load(os.path.join(str(pub), sp.RESULTS_BASENAME))
    assert r["run_mode"] == "published" and r["published_run"] is True
    assert r["run_status"] == "complete"
    assert r["source_verification"]["verified"] is True
    assert r["fingerprint"]["source_archive_sha256"] == _sidecar_for(
        repo_zip)["archive"]["sha256"]


# --- 10. the scientific constants and the seeded draws did not move ---------


# Digests of (arrival offsets, prompt lengths, max_new, phase) per seed at n=200
# under the derived plan, recorded from the code BEFORE this hardening pass.
# They contain no H5 statistic; they only pin that nothing here moved a draw.
_PINNED_WORKLOAD_SHA256 = {
    42: "7102bc1122da2ccba7c0e250f9169d8927bd8968b6d51ee44da77ee05c5e84d6",
    43: "f085499328e4537c63603fdf2c03e7f59b950ca85190e553ae2c23f439bdfe07",
    44: "22dedb2ddb03f2242b8c3fc9864334d7ee4fbf6ab20abb9d89ceffd07fac3946",
    45: "6a3d83d3876b20d6996732d77178ba64d6196e03d65ca8187684c2530b8a753f",
    46: "62e49d406f6187b2a23bb588675ad6f67090218d3cd04a2042aa7ec6201c54fe",
    47: "382c084457c6dc6f1d74e16d37c52d3ea59305337d16a63331b60bcb24703c67",
    48: "20f09a8d82b15c6d989b67b2ff6e0a215c5faa732782f7605367c2eb328511a3",
    49: "fba706bb3b48ab8570b79495d9946e9282cba69091df690651dab71f95dcea4d",
    50: "9f1b85ffb21ef6ec44674e459e88edd1ca8765201e9651fac2985749813187a3",
    51: "e5c6e32e6cf619ab89fdcd8d61ffd79d8dc31a133153b22a93f4c76cb21ac436",
    999: "d3ef5a1307237c71a7b8705c5328e6efef366c24f430b8e71cd16474204cfb88",
}


def test_o12_seeded_workloads_are_byte_identical_to_before():
    import hashlib
    plan = sp.derive_service_rates("eager")
    for seed, want in _PINNED_WORKLOAD_SHA256.items():
        specs, arr = sp.build_poisson_workload(_Tokenizer(), 200, seed, plan)
        blob = json.dumps({"arr": arr,
                           "plens": [int(s.prompt_ids.shape[-1]) for s in specs],
                           "maxnew": [s.max_new for s in specs],
                           "phase": [s.phase for s in specs]}, sort_keys=True)
        assert hashlib.sha256(blob.encode()).hexdigest() == want, seed


def test_o12b_the_registered_constants_are_unchanged():
    from src.carl.live import _SCENARIO_SPEC
    from src.carl.live_reward import (
        MIN_ARM_REWARD_SPREAD, MIN_LIVE_DISTINCT, MIN_LIVE_REWARD_SPREAD,
    )
    assert sp.TARGET_RHO == 0.40
    assert sp.DEFAULT_SEEDS == list(range(42, 52))
    assert sp.VALIDATION_SEED == 999 and sp.DEFAULT_REQUESTS == 200
    assert (sp.GRID_BATCH_AXIS, sp.GRID_CHUNK_AXIS) == (
        [2, 4, 8, 12, 16, 24, 32], [64, 128, 256, 512])
    assert sp.CONFIGS == ["Static-Tput", "Static-Utility", "RuleOnly",
                          "CARL-Repaired", "CARL-Expanded"]
    assert sp.OBSERVE_INTERVAL == 10 and sp.ALPHA == 0.5
    assert sp.CONVERGENCE_RULE is fh.CONVERGENCE_RULE
    assert (MIN_ARM_REWARD_SPREAD, MIN_LIVE_REWARD_SPREAD, MIN_LIVE_DISTINCT) == (
        0.02, 0.01, 3)
    assert _SCENARIO_SPEC["INTERACTIVE"] == {"prompt_lo": 16, "prompt_hi": 32,
                                             "max_new": 32}
    assert _SCENARIO_SPEC["BATCH"] == {"prompt_lo": 128, "prompt_hi": 256,
                                       "max_new": 64}
    plan = sp.derive_service_rates("eager")
    assert plan["phases"]["interactive"]["lambda_rps"] == 0.4 * 2.9613881136988742
    assert plan["phases"]["batch"]["lambda_rps"] == 0.4 * 1.4135291672527908


def test_o12c_a_missing_calibration_crosscheck_artifact_aborts(tmp_path):
    """A NAMED cross-check whose evidence is absent is a failure, not a skip."""
    with pytest.raises(sp.AmbiguousServiceRateError, match="missing"):
        sp.derive_service_rates(
            "eager", calibration_path=str(tmp_path / "absent.json"))


# ===========================================================================
# HARDENING 6. Identity, the ownership window, terminal durability, fresh
# validation, calibration binding, path aliases, and the 2026-10-22
# statistical amendments -- through the real main() / run_all() where the
# claim is about orchestration.
# ===========================================================================


def _without_preflight(snap):
    return {k: v for k, v in snap.items() if "preflight_errors" not in k}


@pytest.fixture
def published(engine, tmp_path, monkeypatch, repo_zip):
    """The published path with registered parameters shrunk for the test, a
    faked T4 and a sidecar that verifies the real tree."""
    pub = tmp_path / "published"
    monkeypatch.setattr(sp, "DEVICE", _Cuda())
    monkeypatch.setattr(sp, "PUBLISHED_OUTPUT_DIR", str(pub))
    monkeypatch.setattr(sp, "DEFAULT_SEEDS", [42, 43])
    monkeypatch.setattr(sp, "DEFAULT_REQUESTS", _N)
    monkeypatch.setattr(sp, "GRID_BATCH_AXIS", [4, 8])
    monkeypatch.setattr(sp, "GRID_CHUNK_AXIS", [128, 256])
    side_path = tmp_path / "SOURCE_PROVENANCE.json"
    side_path.write_text(json.dumps(_sidecar_for(repo_zip)))
    monkeypatch.setattr(fh, "SOURCE_PROVENANCE_PATH", str(side_path))
    return pub, side_path


# --- 1. hardware, runtime and model identity ---------------------------------


def test_p1_a_published_run_on_another_gpu_is_refused(engine, published):
    pub, _ = published
    engine.gpu["name"] = "NVIDIA L4"
    with pytest.raises(SystemExit, match="Tesla T4"):
        sp.main([])
    assert engine.loads == 0 and not os.path.exists(str(pub))


def test_p1b_the_gpu_that_holds_the_model_is_the_one_checked(
        engine, published, monkeypatch):
    """Preflight sees a T4 on device 0; the weights land on device 1, an L4."""
    pub, _ = published

    class _Cuda1(_Cuda):
        index = 1

    names = {0: "Tesla T4", 1: "NVIDIA L4"}
    monkeypatch.setattr(sp, "_cuda_device_props", lambda idx: {
        "name": names[idx], "capability": "x", "total_memory_bytes": 1})
    engine.model_device = _Cuda1()
    with pytest.raises(sp.PreflightRefusal, match="not the preflight device"):
        sp.main([])
    assert not os.path.exists(os.path.join(str(pub), sp.RESULTS_BASENAME))


def test_p1c_an_unresolvable_model_identity_fails_closed_when_published(
        engine, published):
    pub, _ = published
    engine.model_id = {"resolved": False, "repo_id": sp.MODEL_NAME,
                       "why": "not in the local cache: ['model.safetensors']"}
    with pytest.raises(sp.PreflightRefusal, match="immutable identity"):
        sp.main([])
    assert not os.path.exists(os.path.join(str(pub), sp.RESULTS_BASENAME))


def test_p1d_a_smoke_records_its_actual_identity_and_stays_unpublished(
        engine, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "DEVICE", _Cuda())
    engine.gpu["name"] = "NVIDIA A100-SXM4-40GB"
    engine.model_id = {"resolved": False, "repo_id": sp.MODEL_NAME, "why": "x"}
    out = tmp_path / "out"
    sp.main(["--smoke", *_SMALL, "--output-dir", str(out)])
    r = _load(_paths(out).results)
    assert r["run_mode"] == "smoke" and r["published_run"] is False
    assert r["runtime_identity"]["gpu"]["name"] == "NVIDIA A100-SXM4-40GB"
    assert r["fingerprint"]["runtime_identity"] == r["runtime_identity"]
    assert r["model_identity"]["resolved"] is False


def test_p1e_a_changed_library_stack_is_refused_before_the_model_loads(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)
    before = _snapshot_tree(str(out))
    real = sp._library_versions
    monkeypatch.setattr(sp, "_library_versions",
                        lambda: {**real(), "transformers": "999.0.0"})
    loads = engine.loads
    with pytest.raises(sp.ResumeMismatchError, match="runtime_identity"):
        _run_smoke(out, "--resume")
    assert engine.loads == loads, "refused BEFORE loading the model"
    assert _without_preflight(_snapshot_tree(str(out))) == before


def test_p1f_a_changed_gpu_is_refused_on_resume(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "DEVICE", _Cuda())
    out = tmp_path / "out"
    engine.serves, engine.fail_at_serve = 0, 8
    with pytest.raises(KeyboardInterrupt):
        sp.main(["--smoke", *_SMALL, "--output-dir", str(out)])
    engine.fail_at_serve = None
    before = _snapshot_tree(str(out))
    engine.gpu["name"] = "NVIDIA L4"
    with pytest.raises(sp.ResumeMismatchError, match="runtime_identity"):
        sp.main(["--smoke", *_SMALL, "--output-dir", str(out), "--resume"])
    assert _without_preflight(_snapshot_tree(str(out))) == before


def test_p1g_changed_model_weights_are_refused_on_resume(engine, tmp_path):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)
    before = _snapshot_tree(str(out))
    engine.model_id = {**engine.model_id, "revision": "d" * 40,
                       "identity_sha256": "e" * 64}
    with pytest.raises(sp.ResumeMismatchError, match="model_identity"):
        _run_smoke(out, "--resume")
    assert _without_preflight(_snapshot_tree(str(out))) == before


def test_p1h_model_identity_is_read_offline_from_the_cache_snapshot(
        tmp_path, monkeypatch):
    """Revision = the snapshot directory the hub resolved at download time;
    every file is hashed; nothing is fetched."""
    import huggingface_hub
    rev = "f" * 40
    snap = tmp_path / "models--x" / "snapshots" / rev
    snap.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer_config.json",
                 "tokenizer.json"):
        (snap / name).write_bytes(name.encode())
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, name: (str(snap / name)
                                            if (snap / name).exists() else None))
    ident = sp.model_identity("x/y")
    assert ident["resolved"] is True and ident["revision"] == rev
    assert ident["files"]["model.safetensors"]["sha256"] == (
        sp._sha256_text("model.safetensors"))
    (snap / "model.safetensors").unlink()
    gone = sp.model_identity("x/y")
    assert gone["resolved"] is False and "model.safetensors" in gone["why"]


# --- 2. the ownership window ---------------------------------------------------


def _assert_resumable_to_completion(engine, out, reference):
    ck = _load(_paths(out).results)
    assert isinstance(ck["fingerprint"], dict), "checkpoint lost its identity"
    assert isinstance(ck["checkpoint"], dict)
    engine.serves = 0
    _run_smoke(out, "--resume")
    done = _load(_paths(out).results)
    assert done["run_status"] == "complete"
    assert _science(done) == _science(reference)


@pytest.fixture
def reference_run(engine, tmp_path):
    whole = tmp_path / "whole"
    _run_smoke(whole)
    return _load(_paths(whole).results)


def _corrupt_sweep_unit(out, mb, cs):
    p = _paths(out).sweep_raw(mb, cs)
    unit = _load(p)
    unit["row"]["wall_s"] = None
    with open(p, "w", encoding="utf-8") as f:
        json.dump(unit, f)


@pytest.mark.parametrize("where,exc", [
    ("supersede", OSError("sharing violation")),
    ("supersede", KeyboardInterrupt()),
    ("save_calibration", KeyboardInterrupt()),
    ("ledger_note", KeyboardInterrupt()),
    ("first_checkpoint_write", KeyboardInterrupt()),
])
def test_p2_a_failure_inside_the_ownership_window_leaves_a_resumable_checkpoint(
        engine, tmp_path, monkeypatch, reference_run, where, exc):
    out = tmp_path / "out"
    paths = _paths(out)
    if where == "supersede":
        _interrupted(engine, out, 4)              # 3 sweep units claimed
        _corrupt_sweep_unit(out, 4, 128)          # claimed, now invalid -> move
        monkeypatch.setattr(sp, "supersede", lambda *a: (_ for _ in ()).throw(exc))
    elif where == "save_calibration":
        _interrupted(engine, out, 7)              # evaluations started
        os.remove(paths.calibration)              # -> recovery writes it
        monkeypatch.setattr(sp, "save_calibration",
                            lambda *a, **k: (_ for _ in ()).throw(exc))
    elif where == "ledger_note":
        _interrupted(engine, out, 7)
        os.remove(paths.sweep_raw(8, 128))        # -> restored from the record
        real_note = sp.ResumeLedger.note

        def _note(self, unit, action, why):
            if action == "restored":
                raise exc
            return real_note(self, unit, action, why)

        monkeypatch.setattr(sp.ResumeLedger, "note", _note)
    else:
        _interrupted(engine, out, 7)
        real_write = sp.write_artifact
        calls = {"n": 0}

        def _write(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise exc
            return real_write(*a, **k)

        monkeypatch.setattr(sp, "write_artifact", _write)
    prior_claims = set(_load(paths.results)["checkpoint"]["units_completed"])
    evidence_before = {os.path.basename(p) for p in paths.evidence_files()}

    with pytest.raises(type(exc)):
        _run_smoke(out, "--resume")
    monkeypatch.undo()
    monkeypatch.setattr(abl, "time", engine.clock)
    monkeypatch.setattr(abl, "_new_scheduler", lambda m: _FakeScheduler(engine))
    monkeypatch.setattr(sp, "load_model_and_tokenizer", engine.load)
    monkeypatch.setattr(sp, "model_identity", lambda name=None: dict(engine.model_id))

    ck = _load(paths.results)
    assert isinstance(ck["fingerprint"], dict)
    assert prior_claims <= set(ck["checkpoint"]["units_completed"]), (
        "the abort dropped claims the previous session had made")
    kept = {os.path.basename(p) for p in paths.evidence_files()}
    moved = set(os.listdir(paths.superseded_dir)) if os.path.isdir(
        paths.superseded_dir) else set()
    assert all(f in kept or any(m.endswith(f) for m in moved)
               for f in evidence_before), "evidence was lost"
    _assert_resumable_to_completion(engine, out, reference_run)


def test_p2b_an_abort_record_without_identity_never_replaces_a_checkpoint(
        engine, tmp_path, monkeypatch):
    """Belt and braces: even if the identity were somehow absent, main() writes
    the abort aside instead of over the checkpoint."""
    out = tmp_path / "out"
    _interrupted(engine, out, 7)
    before = _snapshot_tree(str(out))

    def _apply(restore, ledger, paths):
        _apply.sink.pop("fingerprint", None)
        raise KeyboardInterrupt()

    real_run_all = sp.run_all

    def _run_all(*a, **k):
        _apply.sink = k["sink"]
        return real_run_all(*a, **k)

    monkeypatch.setattr(sp, "apply_restore", _apply)
    monkeypatch.setattr(sp, "run_all", _run_all)
    with pytest.raises(KeyboardInterrupt):
        _run_smoke(out, "--resume")
    assert _without_preflight(_snapshot_tree(str(out))) == before


# --- 3. a fired gate survives failed evidence writes -------------------------


def test_p3_a_failed_evidence_write_keeps_the_run_terminal(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    paths = _paths(out)
    _fail_gate_for(monkeypatch, "CARL-Repaired", 43)
    monkeypatch.setattr(sp, "save_failed_unit", lambda *a, **k: (
        _ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(sp.DegenerateRewardError):     # the ORIGINAL exception
        _run_smoke(out)
    ck = _load(paths.results)
    assert ck["aborted"]["status"] == sp.STATUS_TERMINAL_GATE
    tf = ck["terminal_failure"]
    assert tf["evidence_file"] is None and tf["marker_file"]
    assert any("disk full" in e for e in tf["write_errors"])
    assert os.listdir(paths.failed_dir) == [
        "eval__CARL-Repaired__seed043.terminal_marker.json"]
    with pytest.raises(sp.TerminalGateFailureError):
        _run_smoke(out, "--resume")


def test_p3b_every_record_write_failing_still_raises_the_gate_and_says_so(
        engine, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _fail_gate_for(monkeypatch, "CARL-Repaired", 43)
    for fn in ("save_failed_unit", "save_terminal_marker"):
        monkeypatch.setattr(sp, fn, lambda *a, **k: (
            _ for _ in ()).throw(KeyboardInterrupt()))
    real_write = sp.write_artifact

    def _write(results, *a, **k):
        if results.get("aborted"):
            raise OSError("storage gone")
        return real_write(results, *a, **k)

    monkeypatch.setattr(sp, "write_artifact", _write)
    with pytest.raises(sp.DegenerateRewardError):
        _run_smoke(out)
    text = capsys.readouterr().out
    assert sp.STATUS_TERMINAL_GATE in text and "do NOT resume" in text


def test_p3c_the_checkpoint_alone_keeps_the_run_terminal(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _fail_gate_for(monkeypatch, "CARL-Repaired", 43)
    for fn in ("save_failed_unit", "save_terminal_marker"):
        monkeypatch.setattr(sp, fn, lambda *a, **k: (
            _ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(sp.DegenerateRewardError):
        _run_smoke(out)
    ck = _load(_paths(out).results)
    assert ck["terminal_failure"]["persisted_on_disk"] is False
    assert "Do NOT resume" in ck["terminal_failure"]["durability_limitation"]
    with pytest.raises(sp.TerminalGateFailureError):
        _run_smoke(out, "--resume")


def test_p3d_a_terminal_marker_blocks_resume_without_a_terminal_checkpoint(
        engine, tmp_path):
    """The process died after the marker, before the checkpoint said so."""
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 9)
    assert _load(paths.results)["aborted"]["status"] == sp.STATUS_OPERATIONAL
    sp.save_terminal_marker("eval:CARL-Repaired:seed043", paths,
                            gate="per_run_live_reward_stream", message="fired",
                            fingerprint_sha256=None)
    before = _snapshot_tree(str(out))
    with pytest.raises(sp.TerminalGateFailureError, match="terminal marker"):
        _run_smoke(out, "--resume")
    assert _without_preflight(_snapshot_tree(str(out))) == before


# --- 4. published resume, source verification ---------------------------------


def test_p4_a_published_resume_with_unverifiable_source_touches_nothing(
        engine, published):
    pub, side_path = published
    engine.serves, engine.fail_at_serve = 0, 6
    with pytest.raises(KeyboardInterrupt):
        sp.main([])
    engine.fail_at_serve = None
    before = _snapshot_tree(str(pub))
    side = json.loads(side_path.read_text())
    side["archive"]["sha256"] = "Z" * 64          # 64 chars, not a digest
    side_path.write_text(json.dumps(side))
    loads = engine.loads
    with pytest.raises(sp.PreflightRefusal, match="well-formed"):
        sp.main(["--resume"])
    assert engine.loads == loads
    assert _without_preflight(_snapshot_tree(str(pub))) == before


def test_p4b_an_uploaded_zip_that_does_not_match_refuses_a_published_run(
        engine, published, tmp_path):
    pub, _ = published
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"PK not the uploaded archive")
    with pytest.raises(sp.PreflightRefusal, match="uploaded ZIP"):
        sp.main(["--verify-archive", str(bogus)])
    assert engine.loads == 0


def test_p4c_a_matching_uploaded_zip_is_recorded_as_verified(
        engine, published, repo_zip):
    pub, _ = published
    sp.main(["--verify-archive", repo_zip])
    r = _load(os.path.join(str(pub), sp.RESULTS_BASENAME))
    assert r["source_verification"]["archive_sha256_status"] == (
        "verified_against_uploaded_zip")
    assert r["run_history"][-1]["archive_sha256_status"] == (
        "verified_against_uploaded_zip")


# --- 5. fresh evidence is validated like resumed evidence ----------------------


def test_p5_fresh_and_resumed_units_pass_through_the_same_validator(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    seen = []
    real = sp.validate_eval_payload

    def _spy(payload, name, seed, ctx):
        json.dumps(payload, allow_nan=True)        # JSON-native: round-tripped
        seen.append((name, seed))
        return real(payload, name, seed, ctx)

    monkeypatch.setattr(sp, "validate_eval_payload", _spy)
    _run_smoke(out)
    fresh = sorted(seen)
    seen.clear()
    _run_smoke(out, "--resume")
    # A resume validates each claimed unit twice with the SAME validator: once,
    # read-only, to decide whether the engine warm-up is needed, and once in
    # the loop, which remains authoritative.
    assert fresh == sorted(set(seen)) == sorted(
        (n, s) for n in sp.CONFIGS for s in (42, 43))


def test_p5b_invalid_fresh_evaluation_evidence_stops_the_run(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    paths = _paths(out)
    real = sp.run_one

    def _bad(name, *a, **k):
        run = real(name, *a, **k)
        if name == "RuleOnly" and a[3] == 43:
            run["tpot_p99"] = float("nan")
        return run

    monkeypatch.setattr(sp, "run_one", _bad)
    with pytest.raises(sp.InvalidFreshEvidenceError, match="tpot_p99_ms"):
        _run_smoke(out)
    assert not os.path.exists(paths.eval_raw("RuleOnly", 43)), (
        "invalid evidence was written as a completed unit")
    diag = _load(os.path.join(paths.invalid_dir, "eval__RuleOnly__seed043.json"))
    assert diag["status"] == sp.STATUS_INVALID_EVIDENCE and diag["problems"]
    ck = _load(paths.results)
    assert ck["aborted"]["status"] == sp.STATUS_INVALID_EVIDENCE
    assert "eval:RuleOnly:seed043" not in ck["checkpoint"]["units_completed"]
    monkeypatch.setattr(sp, "run_one", real)
    before = _snapshot_tree(str(out))
    with pytest.raises(sp.InvalidEvidenceResumeError):
        _run_smoke(out, "--resume")
    assert _without_preflight(_snapshot_tree(str(out))) == before


def test_p5c_invalid_fresh_sweep_evidence_stops_the_run(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    paths = _paths(out)
    real = sp.run_static

    def _bad(cfg, *a, **k):
        m = real(cfg, *a, **k)
        if (cfg.max_batch_size, cfg.chunk_size) == (8, 128):
            m["wall_s"] = float("inf")
        return m

    monkeypatch.setattr(sp, "run_static", _bad)
    with pytest.raises(sp.InvalidFreshEvidenceError, match="wall_s"):
        _run_smoke(out)
    assert not os.path.exists(paths.sweep_raw(8, 128))
    assert os.path.isfile(os.path.join(paths.invalid_dir, "sweep__mb8_cs128.json"))
    assert _load(paths.results)["aborted"]["status"] == sp.STATUS_INVALID_EVIDENCE


# --- 6. calibration recovery is bound to the checkpoint's digest ---------------


def _drop_eval_files(paths):
    for f in os.listdir(paths.raw_dir):
        if f.endswith(".json") and f != os.path.basename(paths.calibration):
            os.remove(os.path.join(paths.raw_dir, f))


def test_p6_recovery_must_reproduce_the_checkpoint_calibration_digest(
        engine, tmp_path):
    """No evaluation file is left to carry a digest; only the checkpoint
    remembers the original. A changed sweep must not become a new
    calibration."""
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    os.remove(paths.calibration)
    _drop_eval_files(paths)
    unit = _load(paths.sweep_raw(8, 256))
    unit["row"]["throughput_tps"] *= 1.5
    with open(paths.sweep_raw(8, 256), "w", encoding="utf-8") as f:
        json.dump(unit, f)
    before = _snapshot_tree(str(out))
    with pytest.raises(sp.ResumeMismatchError, match="checkpoint"):
        _run_smoke(out, "--resume")
    assert _without_preflight(_snapshot_tree(str(out))) == before


def test_p6b_recovery_from_untouched_evidence_matches_the_checkpoint(
        engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    digest = _load(paths.results)["calibration"]["calibration_sha256"]
    os.remove(paths.calibration)
    _drop_eval_files(paths)
    _run_smoke(out, "--resume")
    assert _load(paths.calibration)["calibration_sha256"] == digest
    assert _load(paths.results)["run_status"] == "complete"


def test_p6c_recovery_with_no_binding_digest_at_all_is_refused(engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    os.remove(paths.calibration)
    _drop_eval_files(paths)
    ck = _load(paths.results)
    del ck["calibration"]
    with open(paths.results, "w", encoding="utf-8") as f:
        json.dump(ck, f)
    with pytest.raises(sp.ResumeMismatchError, match="neither the checkpoint"):
        _run_smoke(out, "--resume")


def test_p6d_a_record_that_disagrees_with_the_checkpoint_is_refused(
        engine, tmp_path):
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    ck = _load(paths.results)
    ck["calibration"]["calibration_sha256"] = "0" * 64
    with open(paths.results, "w", encoding="utf-8") as f:
        json.dump(ck, f)
    with pytest.raises(sp.ResumeMismatchError, match="checkpoint records"):
        _run_smoke(out, "--resume")


def test_p6e_an_early_abort_keeps_the_checkpoint_calibration_binding(
        engine, tmp_path, monkeypatch):
    """A resumed session that dies at its FIRST checkpoint write -- before the
    calibration is recomputed -- must not write a checkpoint that has
    forgotten the digest."""
    out = tmp_path / "out"
    paths = _paths(out)
    _interrupted(engine, out, 7)
    digest = _load(paths.results)["calibration"]["calibration_sha256"]
    real_write = sp.write_artifact
    calls = {"n": 0}

    def _write(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt()
        return real_write(*a, **k)

    monkeypatch.setattr(sp, "write_artifact", _write)
    with pytest.raises(KeyboardInterrupt):
        _run_smoke(out, "--resume")
    ck = _load(paths.results)
    assert ck["stage"] == "arrival_plan_derived", (
        "the abort came before the calibration was recomputed")
    assert ck["calibration"]["calibration_sha256"] == digest


# --- 7. output-path aliases ------------------------------------------------------


def _dir_link(link, target):
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return "junction"
    except Exception:
        pytest.skip("this platform/user cannot create directory symlinks or "
                    "junctions")


def test_p7_a_smoke_cannot_reach_published_evidence_through_a_link(
        engine, tmp_path, monkeypatch):
    pub = tmp_path / "pub"
    pub.mkdir()
    monkeypatch.setattr(sp, "PUBLISHED_OUTPUT_DIR", str(pub))
    alias = tmp_path / "alias"
    _dir_link(alias, pub)
    with pytest.raises(SystemExit, match="published destination"):
        _run_smoke(alias)
    assert os.listdir(str(pub)) == []


def test_p7b_a_linked_raw_directory_is_caught_too(engine, tmp_path, monkeypatch):
    pub = tmp_path / "pub"
    (pub / "raw" / "subcritical_poisson").mkdir(parents=True)
    monkeypatch.setattr(sp, "PUBLISHED_OUTPUT_DIR", str(pub))
    out = tmp_path / "out"
    (out / "raw").mkdir(parents=True)
    _dir_link(out / "raw" / "subcritical_poisson", pub / "raw" / "subcritical_poisson")
    with pytest.raises(SystemExit, match="raw_dir"):
        _run_smoke(out)
    assert os.listdir(str(pub / "raw" / "subcritical_poisson")) == []


def test_p7c_dotdot_and_case_aliases_are_caught(engine, tmp_path, monkeypatch):
    pub = tmp_path / "pub"
    pub.mkdir()
    monkeypatch.setattr(sp, "PUBLISHED_OUTPUT_DIR", str(pub))
    with pytest.raises(SystemExit, match="published destination"):
        _run_smoke(os.path.join(str(pub), "..", "pub"))
    if os.path.normcase("A") == os.path.normcase("a"):
        with pytest.raises(SystemExit, match="published destination"):
            _run_smoke(str(pub).upper())


# --- 8. H5 as amended -------------------------------------------------------------


def _poisson_cdf_complement(k, x):
    """P(Gamma(k, 1) <= x) for integer k, via the Poisson sum -- an exact
    closed form independent of the implementation under test."""
    return 1.0 - sum(math.exp(j * math.log(x) - x - math.lgamma(j + 1))
                     for j in range(k))


def test_p8_the_mean_of_k_gaps_is_gamma_k_rate_k_lambda():
    lam = 0.4 * 2.9613881136988742
    lo, hi = sp.gamma_mean_bounds(99, lam, 0.05 / 20)
    # The MEAN's bounds bracket 1/lambda; the SUM's would sit near 99/lambda.
    assert lo < 1.0 / lam < hi < 2.0 / lam
    assert _poisson_cdf_complement(99, lo * 99 * lam) == pytest.approx(0.00125, abs=1e-12)
    assert _poisson_cdf_complement(99, hi * 99 * lam) == pytest.approx(0.99875, abs=1e-12)
    assert (lo * lam, hi * lam) == pytest.approx(
        (0.7233298293230, 1.3314135661416), abs=1e-11)
    # Published chi-square table, df = 100 (Gamma shape 50, scale 2).
    assert 2 * sp.gamma_quantile(50, 0.005) == pytest.approx(67.328, abs=5e-4)
    assert 2 * sp.gamma_quantile(50, 0.995) == pytest.approx(140.169, abs=5e-4)
    assert sp.gamma_quantile(1, 0.5) == pytest.approx(math.log(2), abs=1e-14)


def test_p8b_the_family_is_twenty_cells_and_999_is_separate():
    plan = sp.derive_service_rates("eager")
    ref = sp.h5_reference(plan, 200, sp.DEFAULT_SEEDS)
    fam = ref["family"]
    assert fam["n_cells"] == 20 and fam["seeds"] == list(range(42, 52))
    assert fam["per_cell_alpha"] == 0.05 / 20
    assert ref["held_out"] == {"seed": 999, "alpha": 0.05, "in_family": False}
    for ph in ("interactive", "batch"):
        p = ref["phases"][ph]
        assert p["k_gaps"] == 99
        assert p["family_bounds_over_expected_mean"] == pytest.approx(
            [0.7233298293230, 1.3314135661416], abs=1e-11)
        assert p["held_out_bounds_over_expected_mean"] == pytest.approx(
            [0.8127506550929, 1.2063698009408], abs=1e-11)


def test_p8c_bounds_do_not_depend_on_the_draws():
    plan = sp.derive_service_rates("eager")
    a = sp.h5_reference(plan, 200, sp.DEFAULT_SEEDS)
    random.seed(12345)
    b = sp.h5_reference(plan, 200, sp.DEFAULT_SEEDS)
    assert a == b
    assert "offsets" not in inspect.signature(sp.h5_reference).parameters


def test_p8d_a_rejected_cell_is_reported_not_dropped():
    plan = sp.derive_service_rates("eager")
    ref = sp.h5_reference(plan, 20, [42, 43])
    offs = {s: sp.poisson_arrivals(20, s, plan) for s in (42, 43)}
    held = sp.poisson_arrivals(20, 999, plan)
    ok = sp.h5_family_report(ref, offs, held, plan, 20)
    assert ok["n_cells"] == 4 and all(
        v["regenerates_from_seeded_stream"] for v in ok["generator_invariants"].values())
    # Stretch one cell's gaps x10: it must be flagged AND kept, and the
    # stretched stream must no longer pass for the seeded draw.
    stretched = {s: dict(v) for s, v in offs.items()}
    base = stretched[43]["batch"][0]
    stretched[43]["batch"] = [base + 10 * (t - base) for t in stretched[43]["batch"]]
    bad = sp.h5_family_report(ref, stretched, held, plan, 20)
    assert bad["family_rejected"] is True and bad["n_cells"] == 4
    assert [(c["seed"], c["phase"]) for c in bad["rejected_cells"]] == [(43, "batch")]
    assert bad["generator_invariants"]["43"]["regenerates_from_seeded_stream"] is False


def test_p8e_the_sampler_itself_is_exponential_on_a_separate_namespace():
    """Regeneration proves reproducibility, not exponential sampling. This
    checks the sampler on a stream no evaluation uses: mean, coefficient of
    variation and the median of Exp(lambda) at large n."""
    from src.carl.live import build_arrivals
    lam, n = 1.7, 200_001
    rng = random.Random("test-only/expovariate-check")
    offs = build_arrivals(n, ARRIVAL_POISSON, lam, rng)
    gaps = [b - a for a, b in zip(offs, offs[1:])]
    mean = statistics.fmean(gaps)
    se = (1.0 / lam) / math.sqrt(len(gaps))
    assert abs(mean - 1.0 / lam) < 5 * se
    assert statistics.stdev(gaps) / mean == pytest.approx(1.0, abs=0.01)
    below_median = sum(1 for g in gaps if g < math.log(2) / lam) / len(gaps)
    assert below_median == pytest.approx(0.5, abs=0.005)
    assert min(gaps) > 0


def test_p8f_a_finished_run_reports_the_family(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    r = _load(_paths(out).results)
    assert r["h5_reference"]["family"]["n_cells"] == 4
    assert r["h5"]["n_cells"] == 4 and len(r["h5"]["held_out_cells"]) == 2
    assert r["h5"]["arrivals_identical_across_treatments"] is True
    assert r["h5"]["held_out_matches_recorded_sweep"] is True


# --- 9. the paired interval -------------------------------------------------------


@pytest.mark.parametrize("df,ref", [
    (1, math.tan(math.pi * 0.475)),                          # closed form
    (2, 0.95 * math.sqrt(2.0 / (4 * 0.975 * 0.025))),        # closed form
    (4, 2.7764451052), (9, 2.2621571628), (30, 2.0422724563),
])
def test_p9_student_t_critical_values(df, ref):
    assert sp.student_t_quantile(0.975, df) == pytest.approx(ref, abs=1e-9)


def test_p9b_the_primary_interval_is_t_and_the_z_one_is_labelled_secondary():
    a = [10.0 + 0.1 * i for i in range(10)]
    b = [9.0 + 0.13 * i + (0.05 if i % 2 else -0.04) for i in range(10)]
    r = sp._paired(a, b)
    d = [x - y for x, y in zip(a, b)]
    se = statistics.stdev(d) / math.sqrt(10)
    m = statistics.fmean(d)
    assert r["ci95_t_df"] == 9
    assert r["ci95_t_critical_value"] == pytest.approx(2.2621571628, abs=1e-9)
    assert r["ci95_t"] == pytest.approx([m - 2.2621571628 * se, m + 2.2621571628 * se])
    assert r["ci95_z_normal_approx_secondary"] == pytest.approx(
        [m - 1.959963984540054 * se, m + 1.959963984540054 * se])
    assert "SECONDARY" in r["ci95_z_method"] and "PRIMARY" in r["ci95_t_method"]
    assert "ci95" not in r, "an unlabelled interval would be ambiguous"
    assert r["wilcoxon_signed_rank"]["applicable"] is True


def test_p9c_the_critical_value_follows_n_not_ten():
    r = sp._paired([1.0, 2.0, 4.0, 3.5, 2.2], [0.5, 1.0, 3.0, 3.0, 2.0])
    assert r["ci95_t_df"] == 4
    assert r["ci95_t_critical_value"] == pytest.approx(2.7764451052, abs=1e-9)


def test_p9d_small_n_and_zero_variance_are_explicit():
    one = sp._paired([1.0], [0.5])
    assert one["ci95_t"] is None and one["ci95_z_normal_approx_secondary"] is None
    assert one["ci95_status"].startswith("undefined")
    flat = sp._paired([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert flat["ci95_t"] == [1.0, 1.0]
    assert flat["ci95_status"].startswith("zero variance")
    assert flat["cohens_d_paired"] is None


def test_p9e_the_frozen_saturated_intervals_are_untouched():
    c = sp.SATURATED_REFERENCE["comparisons"]
    assert c["A_CARL_Repaired_minus_Static_Tput"]["ci95"] == [-3.6633, -3.0723]
    assert c["B_CARL_Repaired_minus_Static_Utility"]["ci95"] == [-0.0263, -0.0227]
    assert c["C_CARL_Repaired_minus_RuleOnly"]["ci95"] == [+3.2932, +4.2335]
    assert c["D_CARL_Expanded_minus_CARL_Repaired"]["ci95"] == [-2.2754, -1.3158]


# --- 10. H2 and the unresolved saturated side -------------------------------------


def test_p10_h2_divides_by_the_comparator_and_excludes_b():
    assert sp.H2_COMPARATORS == {
        "A_adaptation_vs_throughput_baseline": "Static-Tput",
        "C_learning_isolation": "RuleOnly",
        "D_action_space_isolation": "CARL-Repaired"}
    r = sp.h2_relative_difference([9.0, 11.0], [10.0, 10.0])
    assert r["value"] == 0.0
    r = sp.h2_relative_difference([12.0, 8.0, 13.0], [10.0, 10.0, 10.0])
    assert r["value"] == pytest.approx(1.0 / 10.0)


def test_p10b_a_smoke_reports_both_sides_but_gives_no_verdict(engine, tmp_path):
    """The saturated side is now read from the verified artifact. A smoke
    reports it, and its own side, but is never judged by the H1/H2 rule."""
    out = tmp_path / "out"
    _run_smoke(out)
    r = _load(_paths(out).results)
    for label in sp.H2_COMPARATORS:
        assert r["comparisons"][label]["h2_relative_paired_difference"]["value"] >= 0
    assert "h2_relative_paired_difference" not in r["comparisons"][
        "B_adaptation_vs_utility_baseline"]
    hyp = r["hypotheses"]
    assert hyp["saturated_reference_status"] == "verified"
    for h in ("H1", "H2"):
        assert hyp[h]["status"] == "not_evaluated_unregistered_run"
    assert any("smoke" in why for why in hyp["not_evaluable_because"])
    assert hyp["H1"]["saturated_relative_throughput_range_pct"] == pytest.approx(
        53.15154401495218, rel=1e-12)


# --- 11. phase / regime attribution ------------------------------------------------


def _req(rid, phase, arr, sub, fin):
    return {"request_id": rid, "regime": phase,
            "stage_timestamps": {"arrival_target_s": arr, "submit_s": sub,
                                 "finish_s": fin}}


def test_p11_cycles_join_through_scheduler_steps_not_their_own_clock():
    run = {
        "requests": [_req(0, "INTERACTIVE", 0.0, 0.0, 5.0),
                     _req(1, "INTERACTIVE", 1.0, 1.0, 3.0),
                     _req(2, "BATCH", 4.0, 4.0, 9.0)],
        "step_log": [{"step": 10, "t": 2.0}, {"step": 20, "t": 4.5},
                     {"step": 30, "t": 6.0}, {"step": 40, "t": 9.5}],
        "phase_boundary": {"first_batch_arrival_s": 4.0},
        "cycles": [{"step": 10, "regime": "interactive", "t_monotonic_s": 1e9},
                   {"step": 20, "regime": "interactive", "t_monotonic_s": -7},
                   {"step": 30, "regime": "batch", "t_monotonic_s": 0},
                   {"step": 40, "regime": "interactive", "t_monotonic_s": 3}],
    }
    rep = sp.phase_regime_report(run)
    assert rep["attribution_reliable"] is True
    assert rep["controller_attribution"]["counts_by_window_composition_regime"] == {
        "interactive_window|interactive_only|interactive": 1,
        "batch_window|mixed|interactive": 1,
        "batch_window|batch_only|batch": 1,
        "batch_window|idle|interactive": 1,
    }
    assert rep["controller_attribution"]["batch_only_in_flight"] == {
        "cycles": 1, "classified_batch": 1, "share_classified_batch": 1.0}
    assert rep["mixed_execution_window"] == {
        "start_s": 4.0, "end_s": 5.0, "duration_s": 1.0,
        "both_phases_in_flight": True}
    assert rep["scheduled_phase_drain"]["batch"]["drain_after_last_arrival_s"] == 5.0


def test_p11b_unjoinable_cycles_are_reported_not_guessed():
    run = {"requests": [_req(0, "INTERACTIVE", 0.0, 0.0, 1.0)],
           "step_log": [{"step": 10, "t": 0.5}, {"step": 10, "t": 0.6}],
           "phase_boundary": {"first_batch_arrival_s": 2.0},
           "cycles": [{"step": 99, "regime": "interactive"}]}
    rep = sp.phase_regime_report(run)
    assert rep["attribution_reliable"] is False
    assert any("not unique" in p for p in rep["problems"])
    assert any("no step_log time" in p for p in rep["problems"])


def test_p11c_a_finished_run_carries_the_attribution(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    r = _load(_paths(out).results)
    for name in ("RuleOnly", "CARL-Repaired", "CARL-Expanded"):
        for seed, rep in r["configs"][name]["phase_regime_per_seed"].items():
            assert rep["attribution_reliable"] is True, rep["problems"]
            counted = sum(rep["controller_attribution"][
                "counts_by_window_composition_regime"].values())
            unit = _load(_paths(out).eval_raw(name, int(seed)))
            assert counted == unit["decisions"]
    st = r["configs"]["Static-Tput"]["phase_regime_per_seed"]["42"]
    assert st["controller_attribution"] is None
    assert set(st["scheduled_phase_drain"]) == {"interactive", "batch"}


# ===========================================================================
# ENGINE WARM-UP (engine_warmup/1). The T4 smoke charged ~8 s of process
# start-up to the first measured unit only. These tests pin the protocol that
# moves that cost out of every measurement: declared, bounded, isolated, run
# before the first measured unit of every process that measures anything.
# ===========================================================================


class _RecordingScheduler(_FakeScheduler):
    """The fake engine, with every step logged to a shared event list."""

    def __init__(self, engine, events):
        super().__init__(engine)
        self._events = events

    def step(self):
        self._events.append("step")
        return super().step()


def _record_events(engine, monkeypatch):
    events = []
    monkeypatch.setattr(abl, "_new_scheduler",
                        lambda model: _RecordingScheduler(engine, events))
    monkeypatch.setattr(sp, "_cuda_synchronize",
                        lambda device: events.append("sync"))
    fixture_serve = abl._serve

    def _serve(sched, specs, **kw):
        events.append("serve")
        return fixture_serve(sched, specs, **kw)

    monkeypatch.setattr(abl, "_serve", _serve)
    real = sp.run_engine_warmup

    def _warm(*a, **k):
        events.append("warmup_begin")
        try:
            return real(*a, **k)
        finally:
            events.append("warmup_end")

    monkeypatch.setattr(sp, "run_engine_warmup", _warm)
    return events


def test_w1_the_warmup_runs_before_the_first_measured_unit_and_before_ownership(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    events = _record_events(engine, monkeypatch)
    state = {}
    wrapped = sp.run_engine_warmup

    def _check(*a, **k):
        # Before ownership: nothing of this run exists on disk yet.
        state["results_existed"] = os.path.exists(_paths(out).results)
        state["evidence"] = _paths(out).evidence_files()
        return wrapped(*a, **k)

    monkeypatch.setattr(sp, "run_engine_warmup", _check)
    _run_smoke(out)
    first_serve = events.index("serve")
    assert events.count("warmup_begin") == 1, "exactly once per process"
    begin, end = events.index("warmup_begin"), events.index("warmup_end")
    assert begin < end < first_serve
    warm = events[begin:end + 1]
    # Synchronised before its clock starts and after its last step.
    assert warm[1] == "sync" and warm[-2] == "sync"
    steps = [i for i, e in enumerate(warm) if e == "step"]
    assert steps and steps[0] > 1 and steps[-1] < len(warm) - 2
    assert state == {"results_existed": False, "evidence": []}
    r = _load(_paths(out).results)
    assert r["engine_warmup"]["status"] == "complete"
    assert r["engine_warmup"]["synchronized_before"] is True
    assert r["engine_warmup"]["synchronized_after"] is True
    assert r["engine_warmup"]["pending_units_at_start"] == _N_SERVES


def test_w2_the_declared_coverage_executes_through_the_real_scheduler():
    """A tiny random LlamaModel on CPU through the REAL ContinuousBatchScheduler
    (the same constructor the measured units use): every declared stage runs,
    exactly as declared, and the eager prefill, position-offset prefill and
    batched-decode forwards all execute."""
    from src.engine.model import LlamaConfig, LlamaModel
    torch.manual_seed(1234)
    model = LlamaModel(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False)).eval()
    spec = sp.engine_warmup_spec([4, 8], [128, 256])
    params_before = [p.detach().clone() for p in model.parameters()]
    rec = sp.run_engine_warmup(model, _Tokenizer(), spec,
                               device=torch.device("cpu"))
    assert rec["status"] == "complete" and rec["rng_state_restored"] is True
    assert [(s["stage"], s["max_batch_size"], s["chunk_size"])
            for s in rec["stages"]] == [
        (s["stage"], s["max_batch_size"], s["chunk_size"]) for s in spec["stages"]]
    for got, want in zip(rec["stages"], spec["stages"]):
        assert got["requests"] == len(want["prompt_lengths"])
        assert got["tokens_emitted"] == (len(want["prompt_lengths"])
                                         * want["max_new_tokens"])
        assert 0 < got["steps"] <= want["step_bound"]
    seen = rec["forward_shapes_observed"]
    prefill = [s for s in seen if s["path"] == "prefill"]
    assert any(s["position_offset"] is False for s in prefill), "fresh prefill"
    assert any(s["position_offset"] is True for s in prefill), \
        "chunk-continuation / prefix-cache (masked) prefill"
    assert {s["width"] % 16 == 0 for s in prefill} == {True, False}
    decode_batches = {s["batch"] for s in seen if s["path"] == "batched_decode"}
    assert set(range(1, spec["max_batch_size_covered"] + 1)) <= decode_batches
    assert all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    assert not model._forward_pre_hooks, "the audit hook is removed"


def test_w2b_the_warmup_is_bounded_even_if_the_engine_never_drains(
        engine, monkeypatch):
    steps = []

    class _Stuck(_FakeScheduler):
        def has_work(self):
            return True

        def step(self):
            steps.append(1)
            return []

    monkeypatch.setattr(abl, "_new_scheduler", lambda model: _Stuck(engine))
    spec = sp.engine_warmup_spec([4], [128])
    with pytest.raises(sp.EngineWarmupError,
                       match="exceeded its declared bound") as ei:
        sp.run_engine_warmup(_FakeModel(engine), _Tokenizer(), spec)
    assert len(steps) == spec["stages"][0]["step_bound"]
    assert ei.value.warmup_record["status"] == "failed"


def test_w2c_the_spec_is_declarative_and_covers_every_executable_shape():
    spec = sp.engine_warmup_spec([2, 4, 8, 12, 16, 24, 32], [64, 128, 256, 512])
    assert spec == sp.engine_warmup_spec([2, 4, 8, 12, 16, 24, 32],
                                         [64, 128, 256, 512])
    assert spec["version"] == sp.ENGINE_WARMUP_VERSION == "engine_warmup/1"
    arm_chunks = {a.chunk_size for sets in (abl.all_arm_sets(),
                                            abl.expanded_arm_sets())
                  for arms in sets.values() for a in arms}
    assert set(spec["chunk_sizes_covered"]) == {64, 128, 256, 512} | arm_chunks
    # The smoke grid still covers the arms the controllers can select.
    smoke = sp.engine_warmup_spec([4, 8], [128, 256])
    assert smoke["max_batch_size_covered"] == 32
    assert {384, 512} <= set(smoke["chunk_sizes_covered"])
    lengths = spec["stages"][0]["prompt_lengths"]
    lo = sp._SCENARIO_SPEC["INTERACTIVE"]["prompt_lo"]
    hi = sp._SCENARIO_SPEC["BATCH"]["prompt_hi"]
    assert min(lengths) == lo and max(lengths) == hi
    assert {L % 16 == 0 for L in lengths} == {True, False}
    decode = [s for s in spec["stages"] if s["stage"] == "batched_decode"]
    assert [s["max_batch_size"] for s in decode] == list(range(1, 33))
    chunked = {s["chunk_size"] for s in spec["stages"]
               if s["stage"] == "chunked_prefill"}
    assert chunked == {c for c in spec["chunk_sizes_covered"] if c < hi}
    assert "elapsed" not in json.dumps(spec), \
        "nothing measured or outcome-dependent in the declared protocol"
    assert spec["step_bound_total"] == sum(s["step_bound"] for s in spec["stages"])


def test_w3_no_scheduler_cache_or_controller_state_leaks(engine, tmp_path,
                                                        monkeypatch):
    out = tmp_path / "out"
    created, in_warmup = [], {"on": False}
    built = {"controller": [], "tracker": [], "reward": []}

    def _new(model):
        s = _FakeScheduler(engine)
        s.during_warmup = in_warmup["on"]
        created.append(s)
        return s

    monkeypatch.setattr(abl, "_new_scheduler", _new)
    fixture_serve = abl._serve

    def _serve(sched, specs, **kw):
        assert not sched.during_warmup, "a warm-up scheduler was measured"
        assert sched._step_idx == 0 and not sched.has_work(), "not fresh"
        return fixture_serve(sched, specs, **kw)

    monkeypatch.setattr(abl, "_serve", _serve)
    for attr, key in (("CARLController", "controller"),
                      ("MetricsTracker", "tracker"), ("LiveRewardV2", "reward")):
        real_cls = getattr(sp, attr)

        def _spy(*a, _real=real_cls, _key=key, **k):
            built[_key].append(in_warmup["on"])
            return _real(*a, **k)

        monkeypatch.setattr(sp, attr, _spy)
    real = sp.run_engine_warmup

    def _warm(*a, **k):
        in_warmup["on"] = True
        try:
            return real(*a, **k)
        finally:
            in_warmup["on"] = False

    monkeypatch.setattr(sp, "run_engine_warmup", _warm)
    _run_smoke(out)
    warm = [s for s in created if s.during_warmup]
    assert len(warm) == len(sp.engine_warmup_spec([4, 8], [128, 256])["stages"])
    assert all(not s.has_work() for s in warm), "every warm-up scheduler drained"
    assert all(v is False for vs in built.values() for v in vs)
    assert built["controller"] and built["reward"] == [False], \
        "one LiveRewardV2 per run, constructed after the warm-up"


def test_w4_rng_states_are_preserved(engine, monkeypatch):
    np = pytest.importorskip("numpy")

    class _Noisy(_FakeScheduler):
        def step(self):
            random.random(), np.random.rand(), torch.rand(1)
            return super().step()

    monkeypatch.setattr(abl, "_new_scheduler", lambda model: _Noisy(engine))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state()[1].copy(),
              torch.get_rng_state().clone())
    rec = sp.run_engine_warmup(_FakeModel(engine), _Tokenizer(),
                               sp.engine_warmup_spec([4], [128]))
    assert rec["rng_state_restored"] is True
    assert random.getstate() == before[0]
    assert (np.random.get_state()[1] == before[1]).all()
    assert torch.equal(torch.get_rng_state(), before[2])


def test_w4b_cuda_rng_and_synchronisation_with_a_mocked_cuda(engine,
                                                           monkeypatch):
    """No GPU: torch.cuda is mocked. The CUDA RNG state is restored and the
    device is synchronised before the warm-up clock starts and after its
    last step."""
    events = []
    cuda_state = {"s": [torch.tensor([1, 2, 3], dtype=torch.uint8)]}

    class _CudaNoisy(_FakeScheduler):
        def step(self):
            events.append("step")
            cuda_state["s"] = [torch.tensor([9, 9, 9], dtype=torch.uint8)]
            return super().step()

    monkeypatch.setattr(abl, "_new_scheduler", lambda model: _CudaNoisy(engine))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda device=None: events.append(("sync", str(device))))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all",
                        lambda: [t.clone() for t in cuda_state["s"]])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all",
                        lambda states: cuda_state.update(
                            s=[t.clone() for t in states]))
    real_clock = sp.time.perf_counter
    monkeypatch.setattr(sp.time, "perf_counter",
                        lambda: events.append("clock") or real_clock())
    dev = torch.device("cuda", 0)
    rec = sp.run_engine_warmup(_FakeModel(engine), _Tokenizer(),
                               sp.engine_warmup_spec([4], [128]), device=dev)
    assert torch.equal(cuda_state["s"][0],
                       torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert rec["rng_state_restored"] is True
    syncs = [i for i, e in enumerate(events) if isinstance(e, tuple)]
    steps = [i for i, e in enumerate(events) if e == "step"]
    assert len(syncs) == 2 and all(events[i] == ("sync", "cuda:0") for i in syncs)
    assert syncs[0] < events.index("clock") < steps[0], \
        "sync, then the warm-up clock, then work"
    assert steps[-1] < syncs[1], "the final sync follows the last warm-up step"
    assert rec["synchronized_before"] and rec["synchronized_after"]


def _no_wallclock(obj):
    """Drop the controller's REAL-clock stamps (time.monotonic, not the
    virtual harness clock): they differ between any two runs."""
    if isinstance(obj, dict):
        return {k: _no_wallclock(v) for k, v in obj.items()
                if "monotonic" not in k and k != "warmup_duration_s"}
    if isinstance(obj, list):
        return [_no_wallclock(v) for v in obj]
    return obj


def test_w5_the_warmup_never_reaches_a_measurement(engine, tmp_path,
                                                   monkeypatch):
    """Same run with and without the warm-up executing: every measured number,
    arrival, request, reward and convergence verdict is identical, and no
    warm-up request appears in any unit."""
    with_warm = tmp_path / "with"
    _run_smoke(with_warm)
    without = tmp_path / "without"
    monkeypatch.setattr(sp, "run_engine_warmup", lambda *a, **k: {
        "version": sp.ENGINE_WARMUP_VERSION, "status": "complete",
        "steps": 0, "tokens_emitted": 0, "elapsed_s": 0.0})
    _run_smoke(without)
    a, b = _load(_paths(with_warm).results), _load(_paths(without).results)
    assert _science(a) == _science(b)
    for name in sp.CONFIGS:
        for seed in (42, 43):
            ua = _load(_paths(with_warm).eval_raw(name, seed))
            ub = _load(_paths(without).eval_raw(name, seed))
            for key in ("arrival_offsets_s", "requests", "throughput_tps",
                        "ttft_p99_ms", "run_utility_v2", "reward_validity",
                        "convergence", "cycles", "step_log"):
                assert _no_wallclock(ua.get(key)) == _no_wallclock(ub.get(key)),                     (name, seed, key)
            assert len(ua["requests"]) == _N
            assert not any("warmup" in str(r.get("request_id"))
                           for r in ua["requests"])
    for mb in (4, 8):
        for cs in (128, 256):
            sa = _load(_paths(with_warm).sweep_raw(mb, cs))
            sb = _load(_paths(without).sweep_raw(mb, cs))
            assert sa["row"] == sb["row"] and sa["requests"] == sb["requests"]


def test_w6_the_reward_validity_cold_start_is_unchanged(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    r = _load(_paths(out).results)
    assert r["reward"]["require_valid_metrics"] is True
    for name in ("RuleOnly", "CARL-Repaired", "CARL-Expanded"):
        for seed in (42, 43):
            rv = _load(_paths(out).eval_raw(name, seed))["reward_validity"]
            assert rv["warmup_cycles"] >= 1, \
                "the controller's cold start is per measured run, not engine warm-up"
    # Two distinct records under distinct names.
    assert "warmup_cycles" not in r["engine_warmup"]
    assert r["engine_warmup_protocol"]["version"] == "engine_warmup/1"


def test_w7_a_resume_warms_up_before_missing_work_and_skips_completed_units(
        engine, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)
    done = list(_load(_paths(out).results)["checkpoint"]["units_completed"])
    assert done, "the interrupted session completed some units"
    events = _record_events(engine, monkeypatch)
    engine.serves = 0
    _run_smoke(out, "--resume")
    assert events.count("warmup_begin") == 1
    assert events.index("warmup_end") < events.index("serve")
    assert engine.serves == _N_SERVES - len(done), "completed units were skipped"
    r = _load(_paths(out).results)
    hist = r["run_history"]
    assert [h["engine_warmup"]["status"] for h in hist] == ["complete", "complete"]
    assert hist[0]["status"] == sp.STATUS_OPERATIONAL, \
        "the first session's warm-up audit survives the resume"
    assert hist[1]["engine_warmup"]["pending_units_at_start"] == engine.serves
    assert r["run_status"] == "complete"


def test_w8_a_fully_completed_resume_performs_no_inference(engine, tmp_path,
                                                          monkeypatch):
    out = tmp_path / "out"
    _run_smoke(out)
    events = _record_events(engine, monkeypatch)
    engine.serves = 0
    _run_smoke(out, "--resume")
    assert "step" not in events and "warmup_begin" not in events
    assert engine.serves == 0
    r = _load(_paths(out).results)
    assert r["engine_warmup"]["status"] == "skipped_no_measured_work_pending"
    assert r["engine_warmup"]["inference_performed"] is False
    assert [h["engine_warmup"]["status"] for h in r["run_history"]] == [
        "complete", "skipped_no_measured_work_pending"]


@pytest.mark.parametrize("failure,expected", [
    (RuntimeError("CUDA error: an illegal memory access"), sp.EngineWarmupError),
    (KeyboardInterrupt(), KeyboardInterrupt),
])
def test_w9_a_failed_or_interrupted_warmup_preserves_the_checkpoint(
        engine, tmp_path, monkeypatch, failure, expected):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)
    before = _snapshot_tree(str(out))
    rng = random.getstate()

    class _Failing(_FakeScheduler):
        def step(self):
            raise failure

    monkeypatch.setattr(abl, "_new_scheduler", lambda model: _Failing(engine))
    engine.serves = 0
    with pytest.raises(expected):
        _run_smoke(out, "--resume")
    assert engine.serves == 0, "no measured unit ran"
    assert random.getstate() == rng
    after = _snapshot_tree(str(out))
    assert {k: v for k, v in after.items()
            if "preflight_errors" not in k} == before
    aside = [json.loads(v) for k, v in after.items() if "preflight_errors" in k]
    assert len(aside) == 1 and aside[0]["checkpoint_modified"] is False
    assert aside[0]["engine_warmup"]["status"] == (
        "failed" if expected is sp.EngineWarmupError else "interrupted")
    # The untouched checkpoint then resumes normally.
    monkeypatch.setattr(abl, "_new_scheduler",
                        lambda model: _FakeScheduler(engine))
    _run_smoke(out, "--resume")
    assert _load(_paths(out).results)["run_status"] == "complete"


def test_w10_a_changed_warmup_protocol_is_a_fingerprint_mismatch():
    fp = _fp()
    assert fp["engine_warmup_protocol"] == sp.engine_warmup_spec([4, 8], [128, 256])
    changed = dict(fp["engine_warmup_protocol"], version="engine_warmup/0")
    diffs = sp.compare_fingerprints(fp, _fp(engine_warmup_protocol=changed))
    assert len(diffs) == 1 and diffs[0].startswith("engine_warmup_protocol:")
    assert "elapsed" not in json.dumps(fp), \
        "measured warm-up duration is never a fingerprint key"
    assert sp.EXPERIMENT_VERSION == "subcritical_poisson/4"


def test_w10b_a_checkpoint_from_the_previous_protocol_is_refused(
        engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    path = _paths(out).results
    old = _load(path)
    old["fingerprint"].pop("engine_warmup_protocol")
    old["fingerprint"]["experiment_version"] = "subcritical_poisson/3"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f)
    before = _snapshot_tree(str(out))
    loads = engine.loads
    with pytest.raises(sp.ResumeMismatchError) as ei:
        _run_smoke(out, "--resume")
    assert "engine_warmup_protocol" in str(ei.value)
    assert "experiment_version" in str(ei.value)
    assert engine.loads == loads, "refused before loading the model"
    after = _snapshot_tree(str(out))
    assert {k: v for k, v in after.items()
            if "preflight_errors" not in k} == before


def test_w10c_units_measured_under_the_previous_protocol_are_never_reused(
        engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    paths = _paths(out)
    r = _load(paths.results)
    ctx = sp.UnitContext(paths, r["fingerprint_sha256"], _N, r["arrival_plan"],
                         calibration=None)
    unit = _load(paths.sweep_raw(4, 128))
    assert sp.validate_sweep_payload(unit, 4, 128, ctx) == []
    unit["experiment_version"] = "subcritical_poisson/3"
    problems = sp.validate_sweep_payload(unit, 4, 128, ctx)
    assert any("experiment version 'subcritical_poisson/3'" in p for p in problems)


def test_w11_measurement_without_the_warmup_fails_closed(engine, tmp_path,
                                                        monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sp, "measured_work_pending", lambda *a, **k: {
        "pending": False, "sweep_units": [], "eval_units": []})
    with pytest.raises(sp.EngineWarmupError,
                       match="without the declared engine warm-up"):
        _run_smoke(out)
    assert engine.serves == 0


def test_w12_the_pending_decision_mirrors_what_is_measured(engine, tmp_path):
    out = tmp_path / "out"
    _interrupted(engine, out, 8)
    done = set(_load(_paths(out).results)["checkpoint"]["units_completed"])
    missing = [u for u in sp.unit_plan([4, 8], [128, 256], [42, 43])
               if u not in done]
    engine.serves = 0
    _run_smoke(out, "--resume")
    r = _load(_paths(out).results)
    assert r["run_history"][-1]["engine_warmup"]["pending_units_at_start"] \
        == len(missing) == engine.serves


# ===========================================================================
# The recovered frozen saturated reference: H1/H2's saturated side.
# ===========================================================================

import hashlib  # noqa: E402
import shutil  # noqa: E402

import saturated_reference as sref  # noqa: E402


def _copy_reference(root):
    """A byte-for-byte copy of the recovered reference under `root`."""
    for rel in sref.required_files():
        dst = os.path.join(str(root), *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(_REPO_ROOT, *rel.split("/")), dst)
    return str(root)


def _fake_results(h1_pct, h2_values):
    return {"landscape": {"throughput_tps": {"range_relative_pct": h1_pct}},
            "comparisons": {label: {"h2_relative_paired_difference": {"value": v}}
                            for label, v in h2_values.items()}}


def test_r1_h1_uses_the_existing_relative_range_formula():
    """100 * (max - min) / max, the SAME function at both operating points;
    the denominator is the grid maximum, unchanged."""
    rec = sref.load_verified()
    sat = sp.saturated_hypothesis_inputs(rec)["H1"]
    lo, hi = 44.97500716097522, 96.00104467760788
    assert (sat["throughput_min"], sat["throughput_max"]) == (lo, hi)
    assert sat["relative_throughput_range_pct"] == 100.0 * (hi - lo) / hi
    assert sat["relative_throughput_range_pct"] == pytest.approx(53.151544014952)
    assert sat["n_configurations"] == 28
    # The sub-critical landscape goes through the identical function.
    assert sref.relative_range_pct(60.0, 80.0) == 100.0 * 20.0 / 80.0
    assert sref.relative_range_pct(0.0, 0.0) is None
    assert "sref.relative_range_pct(t_min, t_max)" in inspect.getsource(
        sp.landscape_report)


def test_r2_h2_uses_exactly_the_a_c_d_comparators_and_excludes_b():
    rec = sref.load_verified()
    h2 = sp.saturated_hypothesis_inputs(rec)["H2"]
    assert set(h2) == {"A_adaptation_vs_throughput_baseline",
                       "C_learning_isolation", "D_action_space_isolation"}
    pairs = {"A_adaptation_vs_throughput_baseline": ("CARL-Repaired", "Static-Tput"),
             "C_learning_isolation": ("CARL-Repaired", "RuleOnly"),
             "D_action_space_isolation": ("CARL-Expanded", "CARL-Repaired")}
    for label, (a, b) in pairs.items():
        av = sref.per_seed_in_order(rec, a)
        bv = sref.per_seed_in_order(rec, b)
        expect = abs(statistics.fmean(x - y for x, y in zip(av, bv))) / \
            statistics.fmean(bv)
        assert (h2[label]["a"], h2[label]["comparator"]) == (a, b)
        assert h2[label]["value"] == expect
        assert h2[label]["comparator_mean"] == rec["throughput_tps_mean"][b]
        assert h2[label]["n"] == 10
    assert h2["A_adaptation_vs_throughput_baseline"]["value"] == pytest.approx(
        0.0350242571445, rel=1e-10)
    assert h2["C_learning_isolation"]["value"] == pytest.approx(
        0.0422731152326, rel=1e-10)
    assert h2["D_action_space_isolation"]["value"] == pytest.approx(
        0.0193515168703, rel=1e-10)


def test_r3_the_registered_rule_decides_only_with_a_verified_reference():
    rec = sref.load_verified()
    sat = sp.saturated_hypothesis_inputs(rec)
    s1 = sat["H1"]["relative_throughput_range_pct"]
    s2 = {k: v["value"] for k, v in sat["H2"].items()}
    smaller = _fake_results(s1 / 2, {k: v / 2 for k, v in s2.items()})
    st = sp.hypotheses_status(smaller, rec)
    assert (st["H1"]["status"], st["H2"]["status"]) == ("supported", "supported")
    # H1 is falsified by EQUAL as well as larger (its registered wording).
    st = sp.hypotheses_status(_fake_results(s1, s2), rec)
    assert st["H1"]["status"] == "falsified"
    assert st["H2"]["status"] == "neither_supported_nor_falsified_tie"
    one_larger = dict({k: v / 2 for k, v in s2.items()},
                      C_learning_isolation=s2["C_learning_isolation"] * 1.01)
    assert sp.hypotheses_status(_fake_results(s1 / 2, one_larger), rec)[
        "H2"]["status"] == "falsified"
    # Not the registered run: numbers reported, no verdict.
    st = sp.hypotheses_status(smaller, rec, ["run_mode is 'smoke'"])
    assert {st["H1"]["status"], st["H2"]["status"]} == {
        "not_evaluated_unregistered_run"}
    assert st["H1"]["saturated_relative_throughput_range_pct"] == s1
    assert st["comparability"] is sp.SATURATED_COMPARABILITY


def test_r4_no_rounded_value_fallback(tmp_path):
    """An unverified reference leaves H1/H2 unresolved; nothing from the rounded
    transcription is put in their place."""
    bad = sref.try_load(str(tmp_path))
    assert bad["status"] == "unverified"
    assert sp.saturated_hypothesis_inputs(bad) is None
    st = sp.hypotheses_status(_fake_results(1.0, {
        k: 0.001 for k in sp.H2_COMPARATORS}), bad)
    for h in ("H1", "H2"):
        assert st[h]["status"] == "unresolved_saturated_reference_unverified"
        assert st[h]["missing_inputs"]
    assert st["H1"]["saturated_relative_throughput_range_pct"] is None
    assert all(v["saturated"] is None for v in st["H2"]["per_comparison"].values())
    assert st["saturated_reference_problems"]
    text = json.dumps(st)
    for rounded in ("3.3678", "3.7633", "1.7956", "-3.6633"):
        assert rounded not in text
    assert "SATURATED_REFERENCE[\"comparisons\"]" not in inspect.getsource(
        sp.hypotheses_status)


def test_r5_the_reference_provenance_is_embedded_in_the_artifact(engine, tmp_path):
    out = tmp_path / "out"
    _run_smoke(out)
    r = _load(_paths(out).results)
    v = r["saturated_reference_verification"]
    assert v["status"] == "verified"
    assert v["source_member"] == "docs/eval/final_hardening_results.json"
    assert v["artifact_sha256"] == sref.EXPECTED_RESULTS_SHA256
    assert v["recovery"]["source_zip_sha256"] == (
        "39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae")
    assert v["identity"]["gpu"] == "Tesla T4" and v["fields_used"]
    assert v["raw_manifest_sha256"] == sref.raw_manifest_sha256(
        sref.EXPECTED_RAW_SHA256)
    assert v["throughput_tps_mean"]["Static-Tput"] == 96.15569365964514
    assert r["fingerprint"]["saturated_reference_identity"] == \
        sref.fingerprint_identity(v)
    assert r["relationship_to_final_hardening"]["reads_its_artifact"] is True
    assert r["relationship_to_final_hardening"]["writes_its_artifact"] is False
    comp = r["saturated_comparability"]
    assert "2.11.0+cu128" in comp["runtime_stack"]
    assert "2.11.0+cu130" in comp["runtime_stack"]
    # The frozen statistics are carried VERBATIM, never recomputed.
    a = r["comparisons"]["A_adaptation_vs_throughput_baseline"]["saturated_reference"]
    assert a["throughput_tps"]["ci95"] == [-3.663278087591392, -3.072285393696247]
    assert a["throughput_tps"]["cohens_d_paired"] == -7.063954496437444
    assert r["hypotheses"]["H2"]["per_comparison"]["D_action_space_isolation"][
        "saturated"]["value"] == pytest.approx(0.0193515168703, rel=1e-10)


def test_r6_a_changed_reference_refuses_to_resume_before_the_model_loads(
        engine, tmp_path, monkeypatch):
    root = _copy_reference(tmp_path / "ref")
    monkeypatch.setattr(sp, "SATURATED_REFERENCE_ROOT", root)
    out = tmp_path / "out"
    engine.serves, engine.fail_at_serve = 0, 3
    with pytest.raises(KeyboardInterrupt):
        _run_smoke(out)
    engine.fail_at_serve = None
    before = _snapshot_tree(str(out))
    raw = os.path.join(root, *sref.RAW_REL_DIR.split("/"), "RuleOnly_seed042.json")
    with open(raw, "ab") as f:
        f.write(b" ")
    loads = engine.loads
    with pytest.raises(sp.ResumeMismatchError, match="saturated_reference_identity"):
        _run_smoke(out, "--resume")
    assert engine.loads == loads, "refused before any model load"
    # The checkpoint and every unit are untouched; only the refusal is
    # recorded, aside, as every start-up refusal is.
    after = _snapshot_tree(str(out))
    aside = [k for k in after if k not in before]
    assert aside and all(k.startswith("subcritical_poisson_preflight_errors")
                         for k in aside)
    assert {k: after[k] for k in before} == before
    # Restoring the exact bytes restores resumability.
    with open(raw, "rb+") as f:
        data = f.read()[:-1]
        f.seek(0)
        f.write(data)
        f.truncate()
    _run_smoke(out, "--resume")
    assert _load(_paths(out).results)["run_status"].startswith("complete")


def test_r6b_the_fingerprint_key_names_the_reference_identity():
    plan = sp.derive_service_rates("eager")
    good = sref.load_verified()
    a = sp.experiment_fingerprint(plan, [42], 20, [4], [128], model="m",
                                  dtype="d", saturated_ref=good)
    b = sp.experiment_fingerprint(plan, [42], 20, [4], [128], model="m",
                                  dtype="d", saturated_ref=dict(
                                      good, artifact_sha256="0" * 64))
    assert [d.split(":")[0] for d in sp.compare_fingerprints(a, b)] == [
        "saturated_reference_identity"]
    assert a["saturated_reference_identity"]["status"] == "verified"
    assert "grid" not in a["saturated_reference_identity"]


def test_r7_a_published_run_refuses_an_unverified_reference(
        engine, published, monkeypatch, tmp_path):
    pub, _ = published
    root = _copy_reference(tmp_path / "ref")
    path = os.path.join(root, *sref.RESULTS_REL.split("/"))
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data.replace(b"96.15569365964514", b"96.15569365964515", 1))
    monkeypatch.setattr(sp, "SATURATED_REFERENCE_ROOT", root)
    with pytest.raises(sp.PreflightRefusal, match="saturated reference"):
        sp.main([])
    assert engine.loads == 0
    assert not os.path.exists(os.path.join(str(pub), sp.RESULTS_BASENAME))


def test_r7b_a_smoke_with_an_unverified_reference_records_it_unresolved(
        engine, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SATURATED_REFERENCE_ROOT", str(tmp_path / "none"))
    out = tmp_path / "out"
    _run_smoke(out)
    r = _load(_paths(out).results)
    assert r["saturated_reference_verification"]["status"] == "unverified"
    for h in ("H1", "H2"):
        assert r["hypotheses"][h]["status"] == \
            "unresolved_saturated_reference_unverified"
    assert all(c["saturated_reference"] is None
               for c in r["comparisons"].values())


def test_r8_a_published_registered_run_still_needs_the_frozen_design(
        engine, published):
    """The `published` fixture shrinks the registered seeds and grid; the
    frozen run's design is checked against ITS facts, so no verdict."""
    pub, _ = published
    sp.main([])
    r = _load(os.path.join(str(pub), sp.RESULTS_BASENAME))
    hyp = r["hypotheses"]
    assert hyp["saturated_reference_status"] == "verified"
    assert hyp["H1"]["status"] == hyp["H2"]["status"] == \
        "not_evaluated_unregistered_run"
    assert any("saturated run's" in why for why in hyp["not_evaluable_because"])


def test_r9_the_reference_changes_no_measured_behaviour(engine, tmp_path,
                                                        monkeypatch):
    """Serving, controller, reward and static selection: identical whether the
    reference verifies or not. It only feeds H1/H2 and the report."""
    a_out = tmp_path / "verified"
    _run_smoke(a_out)
    monkeypatch.setattr(sp, "SATURATED_REFERENCE_ROOT", str(tmp_path / "none"))
    b_out = tmp_path / "unverified"
    _run_smoke(b_out)
    a, b = _load(_paths(a_out).results), _load(_paths(b_out).results)
    sa, sb = _science(a), _science(b)
    # The calibration binding carries the fingerprint digest, which differs
    # BY DESIGN (the reference identity is a fingerprint key); nothing else may.
    for s_ in (sa, sb):
        s_.pop("calibration")
        s_["binding"] = {k: v for k, v in s_["binding"].items()
                         if k != "fingerprint_sha256"}
    assert sa == sb
    assert a["fingerprint_sha256"] != b["fingerprint_sha256"]
    assert a["reward"] == b["reward"]
    for name in sp.CONFIGS:
        for seed in (42, 43):
            ua = _load(_paths(a_out).eval_raw(name, seed))
            ub = _load(_paths(b_out).eval_raw(name, seed))
            for key in ("arrival_offsets_s", "requests", "throughput_tps",
                        "run_utility_v2", "cycles", "step_log", "convergence"):
                assert _no_wallclock(ua.get(key)) == _no_wallclock(ub.get(key))


def test_r10_the_frozen_reference_is_byte_identical_after_a_run(engine, tmp_path):
    def digests():
        out = {}
        for rel in sref.required_files():
            with open(os.path.join(_REPO_ROOT, *rel.split("/")), "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
        return out
    before = digests()
    assert before[sref.RESULTS_REL] == sref.EXPECTED_RESULTS_SHA256
    _run_smoke(tmp_path / "out")
    assert digests() == before


def test_r11_the_source_check_covers_the_reference(tmp_path):
    """The executing tree's reference bytes are verified against the archive
    manifest like source: an edited copy on the runtime is a mismatch."""
    import source_provenance as srcprov
    tree = tmp_path / "tree"
    for rel in [*srcprov.ARCHIVE_REQUIRED_DATA_FILES, "src/m.py"]:
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        src = os.path.join(_REPO_ROOT, *rel.split("/"))
        p.write_bytes(open(src, "rb").read() if os.path.isfile(src) else b"x\n")
    side = {"archive": {"sha256": "a" * 64},
            "archive_manifest": {"schema": srcprov.ARCHIVE_MANIFEST_SCHEMA,
                                 **srcprov.tree_manifest(str(tree))}}
    assert srcprov.verify_tree(side, str(tree))["verified"] is True
    target = tree / sref.RESULTS_REL
    target.write_bytes(target.read_bytes() + b"\n")
    rep = srcprov.verify_tree(side, str(tree))
    assert rep["verified"] is False
    assert sref.RESULTS_REL in rep["mismatched"]
