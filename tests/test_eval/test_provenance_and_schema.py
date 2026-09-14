"""Every committed result must carry machine-verifiable provenance.

The project shipped two flagship 'GPU' result files that no script could have
produced (see docs/eval/legacy/README.md). These tests make that class of
artifact impossible to introduce again without the test suite going red.
"""
from __future__ import annotations

import json
import os

import pytest

from src.eval.provenance import (
    PROVENANCE_KEY, REQUIRED_FIELDS, capture, validate, write_result,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPAIR_DIR = os.path.join(_ROOT, "docs", "eval", "raw", "repair")
LEGACY_DIR = os.path.join(_ROOT, "docs", "eval", "legacy")


def test_capture_has_every_required_field():
    blk = capture("unit_test")
    for f in REQUIRED_FIELDS:
        assert f in blk, f"missing {f}"
    assert blk["generated_by"] == "script"
    assert blk["git_sha"], "git sha must resolve inside the repo"


def test_capture_records_semantic_versions():
    """Two results that disagree must be distinguishable from two results that
    measure different things."""
    blk = capture("unit_test")
    assert blk["reward_version"]
    assert blk["workload_version"]


def test_write_result_round_trips(tmp_path):
    p = tmp_path / "r.json"
    write_result(str(p), {"value": 1}, "unit_test")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["value"] == 1
    assert validate(loaded) == []


def test_validate_rejects_missing_provenance():
    assert "missing" in validate({"value": 1})[0]


def test_validate_rejects_hand_authored():
    """The exact shape of the quarantined artifacts."""
    payload = {"value": 1, PROVENANCE_KEY: dict(capture("x"),
                                                generated_by="hand")}
    problems = validate(payload)
    assert any("hand-authored" in p for p in problems)


# --- gate on the actual repository ----------------------------------------

def _json_files(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")]


@pytest.mark.parametrize("path", _json_files(REPAIR_DIR))
def test_every_repair_result_carries_provenance(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert validate(payload) == [], f"{os.path.basename(path)} failed validation"


def test_quarantined_artifacts_are_not_in_the_live_results_directory():
    """RECONSTRUCTED files must live under legacy/, never beside real results."""
    live = os.path.join(_ROOT, "docs", "eval")
    for f in os.listdir(live):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(live, f), encoding="utf-8") as fh:
            txt = fh.read()
        assert "RECONSTRUCTED" not in txt, (
            f"{f} claims to be reconstructed but sits in the live results "
            "directory; move it to docs/eval/legacy/")


def test_legacy_directory_documents_why_each_file_is_quarantined():
    readme = os.path.join(LEGACY_DIR, "README.md")
    assert os.path.exists(readme), "quarantine must be documented, not silent"
    txt = open(readme, encoding="utf-8").read()
    for f in _json_files(LEGACY_DIR):
        assert os.path.basename(f) in txt, f"{f} quarantined without explanation"
