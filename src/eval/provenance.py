"""Experiment provenance that travels INSIDE the result file.

WHY THIS EXISTS
---------------
Before this module, environment capture lived in docs/eval/environment.json --
a single, host-specific, .gitignore'd file that every script overwrote. Three
things went wrong as a direct consequence:

  1. A stale CPU environment record was shipped alongside GPU results.
  2. Because the file is gitignored, NO committed result had a committed
     environment record at all.
  3. Two flagship result files (ablation_live_results.json,
     failure_cases_results.json) were hand-reconstructed from terminal output
     after a Colab VM reset, and nothing in the artifact could distinguish them
     from script-generated output.

The fix is structural, not procedural: provenance is embedded in the result
payload itself, under a reserved "_provenance" key, and a schema test asserts
every committed result carries one. A result without provenance is not a
result.

WHAT IS CAPTURED
----------------
Repository state (git SHA + dirty flag), wall-clock time, host, interpreter,
numpy/torch/CUDA/GPU, plus experiment identity (name, script, argv) and the
VERSIONED semantics that make numbers comparable across runs: reward_version
and workload_version. Those last two matter because this repo has already
changed its reward and its workload generator mid-project; without a version
stamp, two result files that disagree are indistinguishable from two result
files that measure different things.

Torch is imported lazily and defensively: this module must work on a CPU-only
box with no torch installed.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Semantic versions. BUMP THESE when the meaning of a number changes.
# ---------------------------------------------------------------------------
#
# reward_version   -- bump when utility()/SLO normalization changes, because
#                     rewards (and therefore regret, oracle capture, and every
#                     bandit decision) become incomparable across the change.
# workload_version -- bump when the request stream changes shape (arrival
#                     process, prompt distribution, phase structure), because
#                     throughput/TTFT become incomparable across the change.
#
# History:
#   reward v1   : throughput_norm = min(1, tps/50); fixed 200ms TTFT / 50ms TPOT
#                 SLOs. Saturates on real GPU (see docs/eval/reward_diagnostics.md).
#   reward v2   : operating-range normalization + soft SLO margins (this pass).
#   workload v1 : bulk-dump live workload (all phase-0 at t0, phase-1 dumped
#                 when half of phase 0 finishes).
#   workload v2 : configurable arrival process (deterministic / Poisson / burst)
#                 with per-stage timestamps; v1 retained as the "burst" stress mode.
# ---------------------------------------------------------------------------

REWARD_VERSION = "v2"
WORKLOAD_VERSION = "v2"

PROVENANCE_KEY = "_provenance"

# Keys every provenance block must carry. The schema test enforces this list.
REQUIRED_FIELDS = (
    "experiment", "script", "argv", "git_sha", "git_dirty", "timestamp_utc",
    "hostname", "python", "platform", "numpy", "torch", "cuda", "gpu",
    "reward_version", "workload_version", "generated_by",
)


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo root; None on any failure."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        out = subprocess.run(
            ("git", *args), cwd=root, capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_sha() -> str | None:
    return _run_git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes to tracked files.

    Recorded rather than forbidden: during an engineering pass results are
    legitimately generated from a dirty tree. What matters is that the reader
    can tell.
    """
    out = _run_git("status", "--porcelain", "--untracked-files=no")
    if out is None:
        return None
    return bool(out.strip())


def _torch_info() -> tuple[str | None, str | None, str | None]:
    """(torch_version, cuda_version, gpu_name); Nones on a torch-free box."""
    try:
        import torch  # noqa: PLC0415 - deliberately lazy
    except Exception:
        return None, None, None
    cuda = getattr(torch.version, "cuda", None)
    gpu = None
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            gpu = "CPU"
    except Exception:
        gpu = None
    return torch.__version__, cuda, gpu


def _numpy_version() -> str | None:
    try:
        import numpy  # noqa: PLC0415
        return numpy.__version__
    except Exception:
        return None


def capture(experiment: str, *, script: str | None = None, extra: dict | None = None) -> dict:
    """Build the provenance block for one experiment run.

    Args:
        experiment: stable experiment name (e.g. "bandit_null_check"). This is
            what the claim ledger cites, so keep it stable across reruns.
        script: path of the generating script. Defaults to sys.argv[0], which is
            correct for `python scripts/eval/x.py` invocation.
        extra: additional run-identifying fields (model, seeds, alpha, ...).
            Merged in at the top level of the block.

    Returns:
        A JSON-serialisable dict with at least REQUIRED_FIELDS.
    """
    torch_v, cuda_v, gpu = _torch_info()
    block = {
        "experiment": experiment,
        "script": script if script is not None else (sys.argv[0] or "<interactive>"),
        "argv": list(sys.argv),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": _safe_user(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": _numpy_version(),
        "torch": torch_v,
        "cuda": cuda_v,
        "gpu": gpu,
        "reward_version": REWARD_VERSION,
        "workload_version": WORKLOAD_VERSION,
        # Explicit, machine-checkable statement that this file came out of a
        # script rather than being typed by hand. The schema test refuses any
        # committed result whose generated_by is not "script".
        "generated_by": "script",
    }
    if extra:
        block.update(extra)
    return block


def _safe_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:
        return None


def stamp(payload: dict, experiment: str, *, script: str | None = None,
          extra: dict | None = None) -> dict:
    """Attach a provenance block to `payload` under PROVENANCE_KEY, in place."""
    payload[PROVENANCE_KEY] = capture(experiment, script=script, extra=extra)
    return payload


def write_result(path: str, payload: dict, experiment: str, *,
                 script: str | None = None, extra: dict | None = None) -> str:
    """Stamp `payload` with provenance and write it as indented JSON.

    Returns the path written, so callers can log it.
    """
    stamp(payload, experiment, script=script, extra=extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def validate(payload: dict) -> list[str]:
    """Return a list of provenance problems with `payload`; empty means valid.

    Used by tests/test_eval/test_result_schema.py to gate every committed
    result file.
    """
    problems: list[str] = []
    block = payload.get(PROVENANCE_KEY)
    if not isinstance(block, dict):
        return [f"missing '{PROVENANCE_KEY}' block"]
    for field in REQUIRED_FIELDS:
        if field not in block:
            problems.append(f"provenance missing required field '{field}'")
    if block.get("generated_by") != "script":
        problems.append(
            f"generated_by is {block.get('generated_by')!r}, expected 'script' "
            "(hand-authored results are not admissible)"
        )
    if not block.get("git_sha"):
        problems.append("git_sha is empty")
    return problems
