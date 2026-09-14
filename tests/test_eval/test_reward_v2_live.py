"""The reward-v2 live experiment must be a CONTROLLED PAIR against the v1 run.

Everything asserted here is a property that, if it silently drifted, would turn
the experiment from "one factor changed" into "two experiments that look
similar" -- which is exactly how the 2026-09-02 artifact came to misdescribe its
own workload and its own reward.
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "scripts", "eval"),
           os.path.join(_ROOT, "scripts", "eval", "repair")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch", reason="the live harness imports torch at import time")

import ablation_live as abl  # noqa: E402
import reward_v2_live as rv2  # noqa: E402

from src.carl.bandit import RepairedLinUCBBandit  # noqa: E402
from src.carl.rule_only import RuleOnlyBandit  # noqa: E402
from src.carl.state import FEATURE_DIM  # noqa: E402


def _source_of(fn) -> str:
    """inspect.getsource, guarded against a STALE IMPORT.

    `getsource` applies the line numbers baked into the code object at IMPORT
    time to the file as it exists NOW. Edit the module while pytest is running
    -- watch mode, or a developer saving mid-run -- and it silently returns some
    OTHER function's body, turning every source assertion in this file into a
    bewildering false failure. Observed in practice: getsource(run_all) came
    back holding the body of write_artifact, and six unrelated tests "failed".

    The assertions below deliberately inspect source text, because that is the
    only way to enforce the controlled-pair property -- that Run B CALLS Run A's
    functions rather than reimplementing them. Importing proves nothing; calling
    is the claim. So the guard keeps the technique and removes its failure mode.
    """
    import inspect
    src = inspect.getsource(fn)
    head = src.lstrip().split("(")[0].strip()
    if head != "def " + fn.__name__:
        raise AssertionError(
            "source out of sync: _source_of(" + fn.__name__ + ") "
            "returned " + repr(head) + ". The module was edited after it was "
            "imported -- re-run pytest. This is NOT a failure of the assertion "
            "that follows.")
    return src


# --- only the four decisive configs ----------------------------------------

def test_runs_exactly_the_four_decisive_configs():
    assert rv2.CONFIGS == ["Static-Best", "RuleOnly",
                           "CARL-Repaired", "CARL-Expanded"]
    for dead in ("CARL-Full", "CARL-NoSpec", "CARL-NoCache", "CARL-NoRouter",
                 "CARL-NoSched", "CARL-NoChunk", "AutoTuner", "DynOracle",
                 "CARL-Thompson"):
        assert dead not in rv2.CONFIGS


def test_policies_are_the_intended_ones():
    assert type(rv2._build_bandit("RuleOnly")) is RuleOnlyBandit
    for name in ("CARL-Repaired", "CARL-Expanded"):
        b = rv2._build_bandit(name)
        for per_regime in b.bandits.values():
            assert isinstance(per_regime, RepairedLinUCBBandit)
            assert per_regime.use_intercept is True
            assert per_regime.center_rewards is True
            assert per_regime.alpha == rv2.ALPHA == 0.5


def test_the_two_carl_treatments_differ_only_in_the_arm_set():
    assert rv2._ARM_SET_FOR["CARL-Repaired"] == "restricted"
    assert rv2._ARM_SET_FOR["CARL-Expanded"] == "expanded"
    r = rv2._build_bandit("CARL-Repaired").arms_by_regime
    e = rv2._build_bandit("CARL-Expanded").arms_by_regime
    for regime in r:
        assert all(a in e[regime] for a in r[regime]), "restricted must be a subset"
        assert len(e[regime]) > len(r[regime])


def test_rule_only_draws_the_same_arm_zero_as_carl():
    """RuleOnly must play the arm CARL warm-starts from, or the learning
    isolation compares two different things."""
    rule = rv2._build_bandit("RuleOnly").arms_by_regime
    carl = rv2._build_bandit("CARL-Repaired").arms_by_regime
    for regime in carl:
        assert rule[regime][0] == carl[regime][0]


# --- the controlled pair ---------------------------------------------------

def test_every_held_fixed_factor_is_read_from_the_v1_harness():
    """Read, never re-declared: a constant that drifted would silently break
    the pair. These must be the SAME objects/values ablation_live uses."""
    assert rv2.DEFAULT_SEEDS == abl.DEFAULT_SEEDS == [42, 43, 44, 45, 46, 47,
                                                      48, 49, 50, 51]
    assert rv2.OBSERVE_INTERVAL == abl.OBSERVE_INTERVAL == 10
    assert rv2.VALIDATION_SEED == abl.VALIDATION_SEED == 999
    assert rv2.CALIBRATION_BATCH_AXIS == abl.ARM_SET_BATCH_AXIS
    assert rv2.DEFAULT_REQUESTS == 200


def test_it_reuses_the_v1_serving_and_workload_code_paths():
    """Guards against a reimplementation that would look identical and not be."""
    import inspect
    src = _source_of(rv2.run_one) + _source_of(rv2.calibrate_reward_scales)
    for call in ("abl._build_workload(", "abl._new_scheduler(", "abl._serve("):
        assert call in src, call
    assert "abl.select_static_best(" in _source_of(rv2.run_all)
    assert "abl._frozen_arms(" in _source_of(rv2._build_bandit)


def test_context_dimension_is_unchanged():
    assert FEATURE_DIM == 10


# --- calibration is held out and frozen ------------------------------------

def test_calibration_seed_is_never_an_evaluation_seed():
    assert rv2.VALIDATION_SEED not in rv2.DEFAULT_SEEDS


def test_calibration_does_not_read_the_evaluation_seeds():
    import inspect
    src = _source_of(rv2.calibrate_reward_scales)
    assert "from_measurements" in src
    assert "seed" in inspect.signature(rv2.calibrate_reward_scales).parameters
    # It is called with the held-out seed and nothing else.
    assert "calibrate_reward_scales(model, tokenizer, n,\n" in _source_of(rv2.run_all)
    assert "VALIDATION_SEED)" in _source_of(rv2.run_all)


def test_calibration_uses_the_evaluation_request_count():
    """Scales fitted at a different n would mis-set ttft_target, because under a
    burst arrival the TTFT p99 scales with the number of queued requests."""
    import inspect
    src = _source_of(rv2.run_all)
    assert "calibrate_reward_scales(model, tokenizer, n," in src


# --- gates are wired, and abort ---------------------------------------------

def test_both_gates_are_wired_into_the_run():
    import inspect
    assert "gate_calibration(" in _source_of(rv2.run_all)
    assert "gate_live_reward_stream(" in _source_of(rv2.run_one)
    assert "gate_candidate_arm_spread(" in _source_of(rv2.gate_calibration)


def test_the_live_gate_covers_every_controller_driven_config():
    assert rv2._CONTROLLER_CONFIGS == {"RuleOnly", "CARL-Repaired",
                                       "CARL-Expanded"}
    assert "Static-Best" not in rv2._CONTROLLER_CONFIGS


# --- pre-registration is present and complete ------------------------------

def test_hypotheses_are_pre_registered_in_the_artifact_payload():
    pr = rv2.PRE_REGISTRATION
    assert pr["registered_before_run"] is True
    for h in ("H1", "H2", "H3"):
        assert set(pr[h]) >= {"statement", "primary_comparison", "prediction",
                              "falsified_if"}
        assert pr[h]["statement"].strip()
        assert pr[h]["falsified_if"].strip()
    assert pr["H1"]["primary_comparison"] == "CARL-Repaired vs Static-Best"
    assert pr["H2"]["primary_comparison"] == "CARL-Repaired vs RuleOnly"
    assert pr["H3"]["primary_comparison"] == "CARL-Expanded vs CARL-Repaired"


def test_pre_registration_document_exists_and_names_the_hypotheses():
    path = os.path.join(_ROOT, "docs", "eval",
                        "PREREGISTRATION_reward_v2_live.md")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    for token in ("H1", "H2", "H3", "Falsified if", "frozen", "held-out"):
        assert token in text, token


def test_the_three_comparisons_match_the_hypotheses():
    labels = {label: (a, b) for label, a, b, _q in rv2.COMPARISONS}
    assert labels["primary"] == ("CARL-Repaired", "Static-Best")
    assert labels["learning_isolation"] == ("CARL-Repaired", "RuleOnly")
    assert labels["action_space_isolation"] == ("CARL-Expanded", "CARL-Repaired")


# --- the scenario string is generated, not hard-coded ----------------------

def test_scenario_description_is_generated_from_n():
    """The v1 artifact was produced with n=200 and still said '1-25 / 26-50'."""
    assert abl.scenario_description(50) == (
        "NON-STATIONARY (1-25 INTERACTIVE prompt16-64/max32, "
        "26-50 BATCH prompt128-256/max64)")
    assert abl.scenario_description(200) == (
        "NON-STATIONARY (1-100 INTERACTIVE prompt16-64/max32, "
        "101-200 BATCH prompt128-256/max64)")
    import inspect
    assert '"scenario": scenario_description(n)' in _source_of(abl.run_all)


# --- provenance -------------------------------------------------------------

def test_provenance_records_every_required_field():
    import inspect
    from src.eval.provenance import REQUIRED_FIELDS
    assert "provenance.write_result(" in _source_of(rv2.write_artifact)
    assert "write_artifact(" in _source_of(rv2.main)
    # Written after EVERY config, not once at the end: two artifacts in this
    # repo were lost to a Colab VM torn down before the JSON was downloaded.
    assert "write_artifact(results, seeds, n)" in _source_of(rv2.run_all)
    src = _source_of(rv2._provenance_extra)
    for field in ("model", "seeds", "requests", "arrival_mode",
                  "observe_interval", "learner_class", "arm_set_definition",
                  "static_search_definition", "frozen_reward_scales",
                  "reward_weights"):
        assert f'"{field}"' in src, field
    # git sha / dirty / argv / timestamp / gpu / cuda / torch / python /
    # reward_version / workload_version come from provenance.capture itself.
    for field in ("git_sha", "git_dirty", "argv", "timestamp_utc", "gpu",
                  "cuda", "torch", "python", "reward_version",
                  "workload_version"):
        assert field in REQUIRED_FIELDS, field


# --- it refuses to produce a CPU "result" ----------------------------------

def test_refuses_to_run_without_cuda_unless_explicitly_smoked():
    import inspect
    src = _source_of(rv2.run_all)
    assert 'DEVICE.type != "cuda" and not allow_cpu' in src
    assert "REFUSES to run without CUDA" in src
    assert "--allow-cpu-smoke" in _source_of(rv2.main)
    assert 'results["cpu_smoke"] = True' in _source_of(rv2.main)


# --- paired statistics distinguish an identity from a tie ------------------

def test_paired_reports_all_diffs_zero_explicitly():
    ident = rv2._paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ident["all_diffs_zero"] is True
    assert ident["mean_difference"] == 0.0
    tie = rv2._paired([1.0, 2.0, 3.0], [1.1, 1.9, 3.0])
    assert tie["all_diffs_zero"] is False
    assert tie["n"] == 3
    assert len(tie["per_seed_difference"]) == 3


def test_a_firing_gate_preserves_the_partial_artifact():
    """A gate firing on the last config must not destroy the GPU hours already
    spent -- and the degeneracy itself is a result worth keeping."""
    import inspect
    src = _source_of(rv2.main)
    assert "except DegenerateRewardError" in src
    assert 'partial["aborted"]' in src
    assert "write_artifact(partial, seeds, args.limit)" in src
    assert "raise" in src, "the abort must still fail loudly"
    assert "sink" in inspect.signature(rv2.run_all).parameters
