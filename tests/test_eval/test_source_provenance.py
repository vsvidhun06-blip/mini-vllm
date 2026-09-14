"""SOURCE_PROVENANCE: the sidecar that replaces the runtime's null git fields.

The T4 artifacts carry `git_sha: null` and `git_dirty: null` because the Colab
archive excludes `.git`, so every git call on the runtime fails and
`provenance._run_git` returns None -- which is indistinguishable from "clean".
These tests pin the properties that make the sidecar a repair rather than
another place for a null to hide.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "eval", "repair")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import source_provenance as sp  # noqa: E402


def _have_git() -> bool:
    return sp._run_git_bytes("rev-parse", "HEAD") is not None


requires_git = pytest.mark.skipif(not _have_git(), reason="not a git repository")


# ---------------------------------------------------------------------------
# The fields the request asked for.
# ---------------------------------------------------------------------------


@requires_git
def test_it_captures_every_required_field():
    block = sp.capture()
    for field in ("git_head", "git_dirty", "git_status_porcelain",
                  "working_tree_diff_sha256", "source_manifest", "created_utc"):
        assert field in block, field
    assert block["git_head"] and len(block["git_head"]) == 40
    assert isinstance(block["git_dirty"], bool)
    assert block["source_manifest"]["manifest_sha256"]


@requires_git
def test_the_porcelain_status_is_captured_in_full_including_untracked():
    """`git_dirty` alone is not enough. It is false for a tree whose only
    changes are untracked -- and this repository's experiment scripts and
    `src/carl/live_reward.py` are untracked right now, so a reader who saw only
    the flag would conclude the run came from a clean HEAD."""
    block = sp.git_state()
    assert block["git_status_porcelain"] is not None
    assert block["git_status_porcelain_untracked_included"] is True
    assert "untracked" in block["git_dirty_definition"]


# ---------------------------------------------------------------------------
# No silent nulls. This is the whole point.
# ---------------------------------------------------------------------------


@requires_git
def test_the_diff_digest_is_not_silently_the_empty_hash_on_a_dirty_tree():
    """THE REGRESSION THIS MODULE ALREADY SUFFERED.

    The first version decoded git output as text. On this Windows host the
    locale codec is cp1252, `git diff HEAD` over this tree contains byte 0x81,
    the decode raised inside subprocess's reader thread, the exception was
    swallowed, and the digest came back as SHA256("") sitting next to
    `git_dirty: True`. That is the same class of silent null the sidecar exists
    to remove, one layer down.
    """
    block = sp.git_state()
    if not block["git_dirty"]:
        pytest.skip("clean tree: an empty diff digest is correct here")
    assert block["working_tree_diff_captured"] is True
    assert block["working_tree_diff_sha256"] != hashlib.sha256(b"").hexdigest()
    assert block["working_tree_diff_bytes"] > 0


def test_the_empty_diff_digest_is_published_so_the_two_cases_are_separable():
    """A clean tree and a failed capture must be distinguishable by inspection,
    not by trusting that the capture worked."""
    block = sp.git_state()
    assert block["working_tree_diff_empty_sha256"] == hashlib.sha256(b"").hexdigest()
    if not block["working_tree_diff_captured"]:
        assert block["working_tree_diff_sha256"] is None, (
            "a failed diff must be null, never the digest of a substituted empty "
            "string")


def test_a_missing_repository_says_so_instead_of_looking_clean(monkeypatch):
    monkeypatch.setattr(sp, "_run_git_bytes", lambda *a: None)
    block = sp.git_state()
    assert block["git_available"] is False
    assert block["git_head"] is None
    assert block["git_dirty"] is None
    assert "null for that reason and not because" in block["notes"]


def test_git_output_with_undecodable_bytes_does_not_blank_the_capture(monkeypatch):
    monkeypatch.setattr(sp, "_run_git_bytes", lambda *a: b"caf\x81 subject\n")
    assert sp._run_git("log") == "caf� subject\n"


# ---------------------------------------------------------------------------
# The manifest: untracked source must not slip through.
# ---------------------------------------------------------------------------


def test_the_manifest_covers_untracked_source_the_diff_cannot():
    """`git diff HEAD` says nothing about untracked files. The experiment depends
    on several -- so the manifest hashes what is on disk, tracked or not, and the
    two digests together are what makes a run reproducible."""
    m = sp.source_manifest()
    paths = {e["path"] for e in m["files"]}
    for expected in ("src/carl/live_reward.py", "src/carl/controller.py",
                     "src/carl/state.py",
                     "scripts/eval/repair/final_hardening.py"):
        assert expected in paths, expected
    assert m["n_files"] == len(m["files"]) > 0


def test_the_manifest_excludes_outputs_so_a_mismatch_still_means_the_code_changed():
    m = sp.source_manifest()
    for e in m["files"]:
        assert e["path"].endswith(".py")
        assert "__pycache__" not in e["path"]
        assert not e["path"].startswith("docs/")


def test_the_manifest_digest_is_recomputable_by_hand():
    """The recipe is published in the artifact, so it must actually be the
    recipe -- a digest nobody can reproduce is a decoration."""
    m = sp.source_manifest()
    expected = hashlib.sha256("".join(
        f"{e['sha256']}  {e['path']}\n" for e in m["files"]).encode("utf-8")
    ).hexdigest()
    assert m["manifest_sha256"] == expected
    assert "sha256sum" in m["manifest_digest_recipe"]


def test_the_manifest_is_order_independent():
    a, b = sp.source_manifest(), sp.source_manifest()
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert [e["path"] for e in a["files"]] == sorted(e["path"] for e in a["files"])


def test_editing_a_source_file_changes_the_digest(tmp_path, monkeypatch):
    """A manifest that does not move when the code moves would certify nothing."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    target = root / "src" / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(sp, "_REPO_ROOT", str(root))

    before = sp.source_manifest()["manifest_sha256"]
    target.write_text("x = 2\n", encoding="utf-8")
    assert sp.source_manifest()["manifest_sha256"] != before


# ---------------------------------------------------------------------------
# The archive digest.
# ---------------------------------------------------------------------------


def test_the_archive_digest_matches_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_REPO_ROOT", str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("y = 0\n", encoding="utf-8")
    archive = tmp_path / "upload.zip"
    archive.write_bytes(b"PK\x03\x04not-really-a-zip")

    block = sp.capture(str(archive))
    assert block["archive"]["sha256"] == hashlib.sha256(
        archive.read_bytes()).hexdigest()
    assert block["archive"]["bytes"] == archive.stat().st_size


def test_a_named_archive_that_does_not_exist_stops_the_capture(tmp_path, monkeypatch):
    """Silently omitting it would leave the artifact claiming a provenance it
    does not have."""
    monkeypatch.setattr(sp, "_REPO_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match="no such file"):
        sp.capture(str(tmp_path / "missing.zip"))


def test_no_archive_records_why_rather_than_leaving_a_bare_null(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_REPO_ROOT", str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("y = 0\n", encoding="utf-8")
    block = sp.capture(None)
    assert block["archive"] is None
    assert "--archive" in block["archive_note"]


# ---------------------------------------------------------------------------
# End to end, and the consumer.
# ---------------------------------------------------------------------------


@requires_git
def test_the_cli_writes_valid_json(tmp_path):
    out = tmp_path / "SOURCE_PROVENANCE.json"
    r = subprocess.run(
        [sys.executable, os.path.join("scripts", "eval", "repair",
                                      "source_provenance.py"),
         "--out", str(out)],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    block = json.loads(out.read_text(encoding="utf-8"))
    assert block["schema"] == "source_provenance/1"
    assert block["git_head"]
    assert block["source_manifest"]["n_files"] > 0


@requires_git
def test_the_experiment_embeds_the_sidecar_verbatim():
    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "eval"))
    import final_hardening as fh

    if not os.path.isfile(fh.SOURCE_PROVENANCE_PATH):
        pytest.skip("no sidecar captured yet; run source_provenance.py")
    block = fh.load_source_provenance()
    on_disk = json.load(open(fh.SOURCE_PROVENANCE_PATH, encoding="utf-8"))

    assert block["present"] is True
    assert block["git_head"] == on_disk["git_head"]
    assert block["source_manifest"]["manifest_sha256"] == (
        on_disk["source_manifest"]["manifest_sha256"])
    assert "embedded_by" in block
