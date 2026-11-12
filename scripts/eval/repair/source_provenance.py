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
    archive.sha256           SHA256 of the archive actually uploaded, when one
                             is named with --archive (nested under `archive`;
                             there is no top-level `archive_sha256` key)
    archive_manifest         per-MEMBER size + SHA256 of every source file and
                             required runtime data file INSIDE that archive, plus
                             one manifest digest -- what Colab verifies the
                             extracted tree against, without git
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

WHY AN ARCHIVE-MEMBER MANIFEST AS WELL
=====================================
`source_manifest` hashes the Windows WORKING TREE. What runs on Colab is the
archive's members after extraction, and nothing else checks that the two agree
or that the extracted files are still the ones that were packaged. So with
--archive this module also opens the ZIP and hashes every member that is source
(`SOURCE_ROOTS` x `SOURCE_SUFFIXES`) or a required runtime data file
(`ARCHIVE_REQUIRED_DATA_FILES`). `verify_tree()` -- pure Python, no git --
recomputes the same digests over an extracted tree and reports every mismatched,
missing or unexpected file. The experiment calls it at start-up and puts the
RECOMPUTED digest, not the sidecar's claim, into its resume fingerprint, so an
unchanged sidecar cannot vouch for edited source.

Line endings are handled by hashing RAW BYTES everywhere. The archive is built
on Windows from a `core.autocrlf=true` checkout, so most `.py` members carry
CRLF; ZIP extraction does not translate line endings, so the bytes on Colab are
the bytes that were packaged and the raw digests match exactly. Each entry also
records `sha256_lf`, the digest after CRLF -> LF normalisation, so a member can
be compared against the LF blob git stores -- informational only, never used
for verification.

Run AFTER building the Colab archive, locally, against that archive:

    python scripts/eval/repair/source_provenance.py --archive <the zip>

Without --archive only the working-tree capture is written, and the
sub-critical experiment will refuse a published run from it.

Writes `docs/eval/SOURCE_PROVENANCE.json` by default (atomically). The
experiments read it and embed it under `_provenance.source_provenance`; if it is
absent, they record why it is absent rather than emitting a null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import saturated_reference as _saturated_reference  # noqa: E402

DEFAULT_OUT = os.path.join(_REPO_ROOT, "docs", "eval", "SOURCE_PROVENANCE.json")

# What counts as "the source that runs". Deliberately narrow: the engine, the
# controller, the eval helpers and the experiment scripts. Result JSON, docs,
# notebooks and archives are excluded because they are OUTPUTS -- hashing them
# would make the manifest change every time a run is written, so a mismatch
# would stop meaning "the code changed".
SOURCE_ROOTS = ("src", "scripts", "tests")
SOURCE_SUFFIXES = (".py",)
EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints"}

SCHEMA = "source_provenance/2"
ARCHIVE_MANIFEST_SCHEMA = "archive_manifest/1"

# Non-source files a run READS at runtime and therefore must be in the archive
# and verified like source. A missing one would otherwise surface hours into a
# run, or -- worse -- be silently skipped (the eager-fit cross-check used to
# skip when its file was absent).
#
# The frozen saturated reference (final_hardening's results JSON and the raw
# seed files H2 aligns through) is listed by saturated_reference.py, its one
# definition, so this list cannot drift from what the experiment verifies.
ARCHIVE_REQUIRED_DATA_FILES = (
    "docs/eval/raw/repair/load_sweep_b7.json",
    "docs/eval/raw/repair/batch_intervention_phase15_16_s42_s43_s44.json",
    "docs/eval/PREREGISTRATION_subcritical_poisson.md",
    *_saturated_reference.required_files(),
)


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


def is_hex_digest(value, length: int = 64) -> bool:
    """A lowercase hexadecimal digest of exactly `length` characters."""
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


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


def _is_source_path(rel: str) -> bool:
    parts = rel.split("/")
    return (len(parts) > 1 and parts[0] in SOURCE_ROOTS
            and rel.endswith(SOURCE_SUFFIXES)
            and not any(p in EXCLUDE_DIR_NAMES for p in parts[:-1]))


def _entry(rel: str, data: bytes) -> dict:
    return {"path": rel, "bytes": len(data), "sha256": _sha256_bytes(data),
            "sha256_lf": _sha256_bytes(data.replace(b"\r\n", b"\n"))}


def _manifest_digest(entries: list) -> str:
    """The `source_manifest` recipe: sha256sum lines, sorted by path."""
    lines = [f"{e['sha256']}  {e['path']}\n"
             for e in sorted(entries, key=lambda e: e["path"])]
    return _sha256_bytes("".join(lines).encode("utf-8"))


def tree_manifest(root: str | None = None) -> dict:
    """Raw-byte manifest of the source + required data files under `root`.

    The same file selection `archive_manifest` makes inside the ZIP, applied to
    a directory, so the two digests are directly comparable. Pure Python: this
    is what runs on Colab, where there is no git.
    """
    root = root or _REPO_ROOT
    entries = []
    for top in SOURCE_ROOTS:
        base = os.path.join(root, top)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            for name in filenames:
                if name.endswith(SOURCE_SUFFIXES):
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    with open(full, "rb") as f:
                        entries.append(_entry(rel, f.read()))
    for rel in ARCHIVE_REQUIRED_DATA_FILES:
        full = os.path.join(root, *rel.split("/"))
        if os.path.isfile(full):
            with open(full, "rb") as f:
                entries.append(_entry(rel, f.read()))
    entries.sort(key=lambda e: e["path"])
    return {"n_files": len(entries), "manifest_sha256": _manifest_digest(entries),
            "files": entries}


class ArchiveError(SystemExit):
    """The archive cannot be used as a portable, complete source bundle."""


def archive_manifest(archive_path: str) -> dict:
    """Open the ZIP and hash every source / required-data MEMBER, raw bytes.

    Refuses (ArchiveError) an archive that would not extract to the same tree on
    Linux: a member name with a backslash (PowerShell 5.1 `Compress-Archive`
    can write these, and `unzip` then creates files literally named
    `src\\carl\\live.py`), an absolute or `..` path, or two members that collide
    case-insensitively. Also refuses one that is missing a required runtime
    data file or has no source under `src/`.

    A single top-level directory wrapping the whole tree (`mini-vllm/src/...`)
    is tolerated and recorded as `root_prefix`, because that is what zipping the
    folder rather than its contents produces.
    """
    try:
        zf = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"--archive {archive_path!r} is not a ZIP: {exc}") from exc
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        problems, seen = [], {}
        for info in infos:
            # orig_filename, not filename: on Windows, zipfile silently rewrites
            # "\\" to "/" in `filename`, which would hide exactly the defect
            # this check exists for on the machine where it runs.
            name = info.orig_filename
            if "\\" in name:
                problems.append(f"member {name!r} uses a backslash separator")
            if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
                problems.append(f"member {name!r} is an absolute path")
            if ".." in name.replace("\\", "/").split("/"):
                problems.append(f"member {name!r} escapes the extraction root")
            low = name.lower()
            if low in seen:
                problems.append(f"members {seen[low]!r} and {name!r} collide "
                                "on a case-insensitive filesystem")
            seen[low] = name
        if problems:
            raise ArchiveError(
                "the archive is not portable to the Colab (Linux) runtime:\n  - "
                + "\n  - ".join(problems))

        names = [i.filename for i in infos]
        tops = {n.split("/", 1)[0] for n in names}
        prefix = ""
        if len(tops) == 1 and all("/" in n for n in names):
            only = next(iter(tops))
            if any(n.startswith(f"{only}/src/") for n in names):
                prefix = only + "/"

        entries = []
        for info in infos:
            rel = info.filename[len(prefix):] if prefix else info.filename
            if _is_source_path(rel) or rel in ARCHIVE_REQUIRED_DATA_FILES:
                entries.append(_entry(rel, zf.read(info)))
    entries.sort(key=lambda e: e["path"])
    paths = {e["path"] for e in entries}
    missing = [r for r in ARCHIVE_REQUIRED_DATA_FILES if r not in paths]
    if missing:
        raise ArchiveError(
            f"the archive is missing required runtime data file(s) {missing}; a "
            "run would fail or skip a check hours in. Rebuild the archive.")
    if not any(p.startswith("src/") for p in paths):
        raise ArchiveError("the archive contains no source under src/")
    return {
        "schema": ARCHIVE_MANIFEST_SCHEMA,
        "root_prefix": prefix,
        "selection": {"roots": list(SOURCE_ROOTS), "suffixes": list(SOURCE_SUFFIXES),
                      "excluded_dir_names": sorted(EXCLUDE_DIR_NAMES),
                      "required_data_files": list(ARCHIVE_REQUIRED_DATA_FILES)},
        "n_files": len(entries),
        "manifest_sha256": _manifest_digest(entries),
        "digest_recipe": (
            "sha256 over '<sha256>  <path>\\n' per entry sorted by path, where "
            "<sha256> is over the member's RAW bytes (no line-ending "
            "translation). sha256_lf is informational: the digest after CRLF -> "
            "LF, comparable with the blob git stores."),
        "files": entries,
    }


def verify_tree(sidecar: dict | None, root: str | None = None, *,
                archive_path: str | None = None) -> dict:
    """Check the tree that is ABOUT TO EXECUTE against the packaged manifest.

    No git, no network: recomputes `tree_manifest(root)` and compares it, file
    by file, with the sidecar's `archive_manifest`. `verified` is True only if
    the sidecar carries a well-formed archive digest and an archive manifest
    and every file matches with nothing missing and nothing extra.
    `executing_manifest_sha256` is always the RECOMPUTED digest, so a caller
    can fingerprint what actually runs regardless of what the sidecar claims.

    The ZIP digest itself is only ASSERTED by the sidecar unless the uploaded
    ZIP is supplied as `archive_path`: then it is hashed and compared, and
    `archive_sha256_status` says which of the two it is.
    """
    live = tree_manifest(root)
    sc = sidecar if isinstance(sidecar, dict) else {}
    present = bool(sc) and sc.get("present", True) is not False
    archive = sc.get("archive") if isinstance(sc.get("archive"), dict) else {}
    am = sc.get("archive_manifest") if isinstance(sc.get("archive_manifest"), dict) else {}
    archive_sha = archive.get("sha256")
    problems = []
    if not present:
        problems.append("no source-provenance sidecar is present")
    if not is_hex_digest(archive_sha):
        problems.append("the sidecar records no well-formed archive.sha256 "
                        "(64 lowercase hex characters; captured without "
                        "--archive?)")
    if am.get("schema") != ARCHIVE_MANIFEST_SCHEMA or not isinstance(
            am.get("files"), list):
        problems.append("the sidecar carries no archive_manifest to verify the "
                        "extracted tree against")
    elif not is_hex_digest(am.get("manifest_sha256")):
        problems.append("archive_manifest.manifest_sha256 is not a SHA256 digest")

    zip_sha = None
    if archive_path is None:
        zip_status = ("asserted_by_sidecar" if is_hex_digest(archive_sha)
                      else "absent")
    elif not os.path.isfile(archive_path):
        zip_status = "uploaded_zip_missing"
        problems.append(f"the ZIP to verify, {archive_path!r}, does not exist")
    else:
        zip_sha = _sha256_file(archive_path)
        if zip_sha == archive_sha:
            zip_status = "verified_against_uploaded_zip"
        else:
            zip_status = "mismatch"
            problems.append(f"the uploaded ZIP hashes to {zip_sha}, not the "
                            f"sidecar's archive.sha256 {archive_sha}")

    mismatched, missing, unexpected = [], [], []
    if isinstance(am.get("files"), list):
        want = {e.get("path"): e for e in am["files"] if isinstance(e, dict)}
        have = {e["path"]: e for e in live["files"]}
        for path, e in sorted(want.items()):
            if path not in have:
                missing.append(path)
            elif have[path]["sha256"] != e.get("sha256"):
                mismatched.append(path)
        unexpected = sorted(set(have) - set(want))
        if _manifest_digest(list(want.values())) != am.get("manifest_sha256"):
            problems.append("archive_manifest.manifest_sha256 does not match "
                            "its own file list")
    if mismatched:
        problems.append(f"{len(mismatched)} file(s) differ from the archive")
    if missing:
        problems.append(f"{len(missing)} packaged file(s) are missing")
    if unexpected:
        problems.append(f"{len(unexpected)} source file(s) are not in the archive")
    return {
        "method": ("raw-byte SHA256 of every source and required data file under "
                   "the executing root, compared with the sidecar's "
                   "archive_manifest; no git, no network"),
        "sidecar_present": present,
        "sidecar_schema": sc.get("schema"),
        "archive_sha256": archive_sha if is_hex_digest(archive_sha) else None,
        "archive_sha256_status": zip_status,
        "uploaded_zip_sha256": zip_sha,
        "archive_manifest_sha256": am.get("manifest_sha256"),
        "executing_manifest_sha256": live["manifest_sha256"],
        "executing_n_files": live["n_files"],
        "mismatched": mismatched,
        "missing": missing,
        "unexpected": unexpected,
        "problems": problems,
        "verified": not problems,
    }


def atomic_write_json(path: str, payload) -> None:
    """Serialise fully, write a same-directory temp file, fsync, os.replace.

    On a process interruption (exception, Ctrl-C, killed process) on a local
    filesystem the destination is the previous complete file or the new
    complete file. Not promised: which version is visible after a power loss
    (the directory entry is not fsync'd), or anything on network / FUSE
    filesystems such as Google Drive.
    """
    text = json.dumps(payload, indent=2, default=str)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".",
                               suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
        "schema": SCHEMA,
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
        am = archive_manifest(full)
        block["archive_manifest"] = am
        # Informational: does the archive hold the bytes the working tree holds?
        # A mismatch is not fatal (the archive is what runs), but a reader should
        # not have to discover it.
        wt = tree_manifest(_REPO_ROOT)
        wt_files = {e["path"]: e["sha256"] for e in wt["files"]}
        block["archive_vs_working_tree"] = {
            "identical": wt["manifest_sha256"] == am["manifest_sha256"],
            "differing_files": sorted(
                e["path"] for e in am["files"]
                if wt_files.get(e["path"]) != e["sha256"]),
            "working_tree_only_files": sorted(
                set(wt_files) - {e["path"] for e in am["files"]}),
        }
    else:
        block["archive"] = None
        block["archive_manifest"] = None
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
    atomic_write_json(args.out, block)

    print(f"git_head   : {block['git_head']}")
    print(f"git_dirty  : {block['git_dirty']}")
    print(f"diff sha256: {block['working_tree_diff_sha256']}")
    print(f"manifest   : {block['source_manifest']['manifest_sha256']} "
          f"({block['source_manifest']['n_files']} files)")
    if block.get("archive"):
        print(f"archive    : {block['archive']['sha256']}  "
              f"{block['archive']['path']}")
        am = block["archive_manifest"]
        print(f"members    : {am['manifest_sha256']} ({am['n_files']} files, "
              f"root prefix {am['root_prefix']!r})")
        cmp = block["archive_vs_working_tree"]
        print(f"vs tree    : identical={cmp['identical']} "
              f"differing={len(cmp['differing_files'])} "
              f"tree-only={len(cmp['working_tree_only_files'])}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
