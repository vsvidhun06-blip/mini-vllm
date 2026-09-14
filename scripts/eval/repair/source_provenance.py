"""SOURCE_PROVENANCE.json -- what git state a Colab run was actually built from.

WHY THIS EXISTS
===============
Every result this repository produces carries a `_provenance` block with a
`git_sha` and a `git_dirty` flag (`src/eval/provenance.py`). On the T4 runs both
came back null:

    "git_sha": null,
    "git_dirty": null

`provenance._run_git` shells out to git in the repo root and returns None on any
failure. The Colab archive is built without `.git`, so there is no repository on
the runtime, every git call fails, and the fields go null -- silently, because a
null is indistinguishable from "git not installed" and from "not a repo". The
artifact then cannot say which source produced it, which is precisely the
question a reader asks first.

Adding git to the archive is the wrong fix: the archive would grow by the whole
object store, and it would still not record which of the many untracked working
files were included. The right fix is to capture the state HERE, on the machine
that has the repository, at the moment the archive is built, and to carry that
capture INTO the run as a sidecar the experiment embeds verbatim.

WHAT IT CAPTURES
================
    git_head                 full SHA of HEAD (and the short SHA, branch,
                             subject and commit timestamp, for a human)
    git_dirty                bool -- tracked files differ from HEAD
    git_status_porcelain     the full `git status --porcelain` text, so the
                             reader sees exactly WHICH files were modified or
                             untracked, not merely that some were
    working_tree_diff_sha256 SHA256 of `git diff HEAD` (tracked changes)
    source_manifest          every source file that would run, each with its
                             size and SHA256, plus a single manifest digest
    archive_sha256           SHA256 of the archive actually uploaded, when one
                             is named with --archive
    created_utc              when this capture was taken

WHY BOTH A DIFF HASH AND A MANIFEST
===================================
They cover disjoint failure modes. `git diff HEAD` covers edits to TRACKED
files and is empty when the tree is clean -- but it says nothing about UNTRACKED
files, and this repository currently has fourteen untracked source files
including `src/carl/live_reward.py` and the experiment scripts themselves. The
manifest hashes what is actually on disk, tracked or not, so an untracked module
cannot slip into a run unrecorded. A run is reproducible if HEAD matches AND
both digests match.

`manifest_sha256` is computed over the canonical text `"<sha256>  <path>\\n"`
sorted by path, so it is stable across filesystems and orderings and can be
recomputed by hand from the listed entries.

Run BEFORE building the Colab archive:

    python scripts/eval/repair/source_provenance.py

or, to include the archive's own digest (build the zip first):

    python scripts/eval/repair/source_provenance.py --archive mini-vllm-final.zip

Writes `docs/eval/SOURCE_PROVENANCE.json` by default. `final_hardening.py` reads
it and embeds it under `_provenance.source_provenance`; if it is absent, that
script records why it is absent rather than emitting a null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_OUT = os.path.join(_REPO_ROOT, "docs", "eval", "SOURCE_PROVENANCE.json")

# What counts as "the source that runs". Deliberately narrow: the engine, the
# controller, the eval helpers and the experiment scripts. Result JSON, docs,
# notebooks and archives are excluded because they are OUTPUTS -- hashing them
# would make the manifest change every time a run is written, so a mismatch
# would stop meaning "the code changed".
SOURCE_ROOTS = ("src", "scripts", "tests")
SOURCE_SUFFIXES = (".py",)
EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints"}


def _run_git_bytes(*args: str) -> bytes | None:
    """`git *args` in the repo root, RAW BYTES; None on any failure.

    Bytes, not text, deliberately. `subprocess(text=True)` decodes with the
    locale codec, which on this Windows host is cp1252 -- and `git diff HEAD`
    over this tree contains a byte (0x81) that cp1252 cannot decode. The first
    version of this module used text=True, the decode raised inside the reader
    thread, `_run_git` swallowed it, and the diff digest came back as the SHA256
    of the empty string next to `git_dirty: True`. That is exactly the silent
    null this module exists to eliminate, reproduced one layer down. Hashing the
    raw bytes is also the more correct thing: a diff can legitimately contain
    binary hunks, which no text codec round-trips.
    """
    try:
        out = subprocess.run(("git", *args), cwd=_REPO_ROOT,
                             capture_output=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _run_git(*args: str) -> str | None:
    """`git *args` decoded as UTF-8 with replacement; None on any failure.

    `errors="replace"` rather than a raise: a path or commit subject with an
    undecodable byte must not blank out the whole capture. Digests are always
    taken from the BYTES (`_run_git_bytes`), never from this, so replacement
    characters can never change a hash.
    """
    raw = _run_git_bytes(*args)
    return None if raw is None else raw.decode("utf-8", errors="replace")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streamed, so a multi-megabyte archive does not have to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_source_files() -> list[str]:
    """Every source file under SOURCE_ROOTS, as repo-relative POSIX paths.

    Sorted, so the manifest and its digest are order-independent.
    """
    found: list[str] = []
    for root in SOURCE_ROOTS:
        base = os.path.join(_REPO_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            for name in filenames:
                if not name.endswith(SOURCE_SUFFIXES):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, _REPO_ROOT).replace(os.sep, "/")
                found.append(rel)
    return sorted(found)


def source_manifest() -> dict:
    """Per-file size + SHA256 for every source file, plus one manifest digest.

    The digest is over the canonical text `"<sha256>  <path>\\n"` for each entry
    in path order -- the same shape `sha256sum` emits -- so it is reproducible
    outside this script and independent of JSON key ordering.
    """
    entries = []
    lines = []
    for rel in iter_source_files():
        full = os.path.join(_REPO_ROOT, rel)
        digest = _sha256_file(full)
        entries.append({"path": rel, "bytes": os.path.getsize(full),
                        "sha256": digest})
        lines.append(f"{digest}  {rel}\n")
    return {
        "roots": list(SOURCE_ROOTS),
        "suffixes": list(SOURCE_SUFFIXES),
        "excluded_dir_names": sorted(EXCLUDE_DIR_NAMES),
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "manifest_sha256": _sha256_bytes("".join(lines).encode("utf-8")),
        "manifest_digest_recipe": (
            "sha256 over the concatenation of '<sha256>  <path>\\n' for every "
            "entry, sorted by path (the sha256sum output format)"),
        "files": entries,
    }


def git_state() -> dict:
    """HEAD, dirtiness, the full porcelain status, and a diff digest.

    Every field is either a real value or an explicit null WITH a reason in
    `git_available` / `notes`. A silent null is the failure mode this whole
    module exists to remove, so it is not reintroduced here.
    """
    head = _run_git("rev-parse", "HEAD")
    available = head is not None
    porcelain = _run_git("status", "--porcelain")
    tracked_only = _run_git("status", "--porcelain", "--untracked-files=no")
    # RAW BYTES. See _run_git_bytes for why this must not go through a text
    # decode, and for the failure it already caused once.
    diff = _run_git_bytes("diff", "HEAD")
    # A clean tree and a tree whose only changes are untracked both produce an
    # EMPTY diff, so the SHA256 of b"" is a meaningful expected value here and is
    # recorded as such. A FAILED diff is a different thing entirely and must not
    # be allowed to look identical to an empty one -- so it nulls the digest and
    # says so, instead of hashing a substituted empty string.
    diff_ok = diff is not None
    return {
        "git_available": available,
        "git_head": head.strip() if head else None,
        "git_head_short": (head.strip()[:12] if head else None),
        "git_branch": (_run_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "git_head_subject": (_run_git("log", "-1", "--pretty=%s") or "").strip() or None,
        "git_head_committed_utc": (
            _run_git("log", "-1", "--pretty=%cI") or "").strip() or None,
        "git_dirty": (bool(tracked_only.strip()) if tracked_only is not None else None),
        "git_dirty_definition": (
            "tracked files differ from HEAD; untracked files do NOT set this "
            "flag, which is why git_status_porcelain is captured in full"),
        "git_status_porcelain": porcelain if porcelain is not None else None,
        "git_status_porcelain_untracked_included": True,
        "working_tree_diff_sha256": (_sha256_bytes(diff) if diff_ok else None),
        "working_tree_diff_bytes": (len(diff) if diff_ok else None),
        "working_tree_diff_captured": diff_ok,
        "working_tree_diff_command": "git diff HEAD (raw bytes, no text decode)",
        "working_tree_diff_covers": (
            "TRACKED file modifications only. Untracked source files are covered "
            "by source_manifest, not by this digest."),
        "working_tree_diff_empty_sha256": _sha256_bytes(b""),
        "working_tree_diff_empty_note": (
            "the digest of an empty diff, listed so a clean tree is "
            "distinguishable from a failed capture by inspection"),
        "notes": (None if available else
                  "git unavailable or not a repository at capture time; every "
                  "git_* field above is null for that reason and not because "
                  "the state was clean"),
    }


def capture(archive_path: str | None = None) -> dict:
    """The whole sidecar. Pure read; writes nothing."""
    block = {
        "schema": "source_provenance/1",
        "purpose": (
            "Record the exact source state a run was built from, on the machine "
            "that HAS the git repository, because the Colab archive excludes "
            ".git and therefore reports git_sha=null / git_dirty=null."),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": _REPO_ROOT,
        "captured_by": "scripts/eval/repair/source_provenance.py",
        "argv": list(sys.argv),
        **git_state(),
        "source_manifest": source_manifest(),
    }
    if archive_path:
        full = archive_path if os.path.isabs(archive_path) else os.path.join(
            _REPO_ROOT, archive_path)
        if not os.path.isfile(full):
            raise SystemExit(
                f"--archive {archive_path!r}: no such file. Build the archive "
                "BEFORE capturing provenance, so the digest describes the file "
                "that is actually uploaded.")
        block["archive"] = {
            "path": os.path.relpath(full, _REPO_ROOT).replace(os.sep, "/"),
            "bytes": os.path.getsize(full),
            "sha256": _sha256_file(full),
        }
    else:
        block["archive"] = None
        block["archive_note"] = (
            "No archive named at capture time. Re-run with --archive <zip> "
            "AFTER building the upload so the artifact records the digest of "
            "the file that was actually uploaded.")
    return block


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Capture git + source state into docs/eval/SOURCE_PROVENANCE.json")
    ap.add_argument("--archive", default=None,
                    help="path to the archive being uploaded; its SHA256 is "
                         "recorded. Build the archive first.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    block = capture(args.archive)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(block, f, indent=2)

    print(f"git_head   : {block['git_head']}")
    print(f"git_dirty  : {block['git_dirty']}")
    print(f"diff sha256: {block['working_tree_diff_sha256']}")
    print(f"manifest   : {block['source_manifest']['manifest_sha256']} "
          f"({block['source_manifest']['n_files']} files)")
    if block.get("archive"):
        print(f"archive    : {block['archive']['sha256']}  "
              f"{block['archive']['path']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
