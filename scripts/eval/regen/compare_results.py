#!/usr/bin/env python3
"""
Compare pre-fix vs post-fix CARL eval artifacts and flag spec-decode impact.

Context
-------
The speculative-decoding regression (`9a5a4c9`, fixed in `2e11321`) corrupted
decode whenever `enable_spec_decode` was active. The original eval artifacts are
unrecoverable (see docs/eval/spec_decode_contamination.md), so the regeneration
package produces BOTH sides fresh on GPU:

    before/  -> evals run at the pre-fix commit  (f23d6ff, contains the bug)
    after/   -> evals run at the post-fix commit (2e11321, fixed)

This script reads the two directories and reports, per artifact, the deltas the
contamination assessment needs. It has NO heavy dependencies (pure stdlib), so it
runs anywhere -- including the CPU-only box -- once the two JSON sets exist.

Spec_k identification
---------------------
* ablation_live: spec_k is a real per-config field (`config.spec_k`); rows are
  tagged exactly.
* failure_cases / cross_model: CARL-Full controls spec_k DYNAMICALLY, so there is
  no fixed per-row spec_k. We treat every CARL-Full row as "spec-capable" and
  report all of them; the before/after delta itself reveals whether speculation
  was active and corrupted. Static-Best rows are tagged spec_k>0 only if their
  selected static config carries spec_k>0.

Usage
-----
    python compare_results.py --before OUT/before --after OUT/after \
        [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Be portable: Colab/Linux stdout is UTF-8, but Windows consoles default to
# cp1252 and choke on non-ASCII. Force UTF-8 where supported; the report itself
# is ASCII-only below regardless.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FILES = [
    "failure_cases_results.json",
    "ablation_live_results.json",
    "cross_model_results.json",
]

# Metric keys as written by the eval scripts (means where aggregated).
METRICS = [
    ("throughput_tps", "throughput (tok/s)"),
    ("ttft_p50", "TTFT p50 (ms)"),
    ("ttft_p99", "TTFT p99 (ms)"),
    ("slo_rate", "SLO rate (%)"),
]
# Material-change threshold (relative) for highlighting a metric delta.
MATERIAL_REL = 0.05  # 5%


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _load(d: str, name: str) -> dict | None:
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _get(d: dict | None, *keys, default=None):
    """Nested get; first key found among aliases at each level wins."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if isinstance(k, (list, tuple)):
            hit = next((kk for kk in k if kk in cur), None)
            if hit is None:
                return default
            cur = cur[hit]
        else:
            if k not in cur:
                return default
            cur = cur[k]
    return cur


def _metric(node: dict | None, key: str):
    """Read a metric that may be stored as `key`, `key_mean`, or `key_tps_mean`."""
    if node is None:
        return None
    for cand in (key, f"{key}_mean", key.replace("_tps", "") + "_tps_mean",
                 f"{key}_tps_mean"):
        if cand in node:
            return node[cand]
    # throughput special-case
    if key == "throughput_tps":
        for cand in ("throughput_tps_mean", "throughput_tps", "throughput"):
            if cand in node:
                return node[cand]
    return None


def _fmt_delta(b, a, pct=True) -> str:
    if b is None or a is None:
        return f"{b} -> {a}  (incomparable)"
    d = a - b
    rel = (d / b * 100.0) if b else float("inf")
    flag = "  **MATERIAL**" if (b and abs(d) / abs(b) >= MATERIAL_REL) else ""
    if pct:
        return f"{b:.3f} -> {a:.3f}  (d= {d:+.3f}, {rel:+.1f}%){flag}"
    return f"{b:.3f} -> {a:.3f}  (d= {d:+.3f}){flag}"


def _find_acceptance(node: dict) -> dict:
    """Recursively collect any key that looks like a spec acceptance rate."""
    out = {}

    def walk(n, path=""):
        if isinstance(n, dict):
            for k, v in n.items():
                kp = f"{path}.{k}" if path else k
                if "accept" in k.lower() and isinstance(v, (int, float)):
                    out[kp] = v
                walk(v, kp)
        elif isinstance(n, list):
            for i, v in enumerate(n):
                walk(v, f"{path}[{i}]")

    walk(node)
    return out


# --------------------------------------------------------------------------- #
# per-artifact comparisons
# --------------------------------------------------------------------------- #
def compare_failure_cases(b: dict, a: dict, lines: list, flags: list) -> None:
    lines.append("\n## failure_cases_results.json\n")
    bs = {s["scenario"]: s for s in b.get("scenarios", [])}
    as_ = {s["scenario"]: s for s in a.get("scenarios", [])}
    for scen in sorted(set(bs) | set(as_)):
        sb, sa = bs.get(scen), as_.get(scen)
        lines.append(f"\n### scenario: {scen}  (CARL-Full is spec-capable)\n")
        if sb is None or sa is None:
            lines.append(f"- present only in {'after' if sb is None else 'before'}\n")
            continue
        wb, wa = sb.get("winner"), sa.get("winner")
        mb, ma = sb.get("margin_pct"), sa.get("margin_pct")
        lines.append(f"- winner: {wb} -> {wa}" + ("  **FLIP**" if wb != wa else ""))
        if wb != wa:
            flags.append(f"WINNER FLIP [failure_cases/{scen}]: {wb} -> {wa}")
        if mb is not None and ma is not None:
            lines.append(f"- margin_pct: {mb:+.2f}% -> {ma:+.2f}%  (d= {ma - mb:+.2f} pp)")
        for side in ("carl", "static"):
            nb, na = sb.get(side), sa.get(side)
            tag = "CARL-Full" if side == "carl" else "Static-Best"
            sk = _get(sa, "static_best_config", "spec_k") if side == "static" else "dynamic"
            lines.append(f"- {tag} (spec_k={sk}):")
            for key, label in METRICS:
                vb, va = _metric(nb, key), _metric(na, key)
                lines.append(f"    - {label}: {_fmt_delta(vb, va)}")


def compare_ablation(b: dict, a: dict, lines: list, flags: list) -> None:
    lines.append("\n## ablation_live_results.json\n")
    bc, ac = b.get("configs", {}), a.get("configs", {})
    for name in sorted(set(bc) | set(ac)):
        nb, na = bc.get(name), ac.get(name)
        sk = _get(na or nb, "config", "spec_k")
        spec_tag = f"spec_k={sk}" + ("  <-- SPEC>0" if isinstance(sk, (int, float)) and sk > 0 else "")
        lines.append(f"\n### config: {name}  ({spec_tag})\n")
        if nb is None or na is None:
            lines.append(f"- present only in {'after' if nb is None else 'before'}\n")
            continue
        for key, label in METRICS:
            vb, va = _metric(nb, key), _metric(na, key)
            lines.append(f"- {label}: {_fmt_delta(vb, va)}")
    # subsystem-contribution ranking flip
    rb = list((b.get("subsystem_contributions") or {}).keys())
    ra = list((a.get("subsystem_contributions") or {}).keys())
    if rb and ra:
        lines.append(f"\n- subsystem ranking: {rb} -> {ra}"
                     + ("  **RANK CHANGE**" if rb != ra else ""))
        if rb != ra:
            flags.append(f"RANKING CHANGE [ablation_live/subsystem]: {rb} -> {ra}")


def compare_cross_model(b: dict, a: dict, lines: list, flags: list) -> None:
    lines.append("\n## cross_model_results.json\n")
    bm = {m.get("short", m.get("model")): m for m in b.get("models", [])}
    am = {m.get("short", m.get("model")): m for m in a.get("models", [])}
    for short in sorted(set(bm) | set(am)):
        mb, ma = bm.get(short), am.get(short)
        lines.append(f"\n### model: {short}  (CARL-Full is spec-capable)\n")
        if mb is None or ma is None:
            lines.append(f"- present only in {'after' if mb is None else 'before'}\n")
            continue
        methods = set(_get(mb, "methods", default={})) | set(_get(ma, "methods", default={}))
        for method in sorted(methods):
            nb = _get(mb, "methods", method)
            na = _get(ma, "methods", method)
            lines.append(f"- method {method}:")
            for key, label in METRICS:
                vb, va = _metric(nb, key), _metric(na, key)
                lines.append(f"    - {label}: {_fmt_delta(vb, va)}")
        # winner-per-model flip (best throughput method)
        def best(m):
            ms = _get(m, "methods", default={})
            if not ms:
                return None
            return max(ms, key=lambda k: _metric(ms[k], "throughput_tps") or -1)
        wb, wa = best(mb), best(ma)
        if wb != wa:
            lines.append(f"- best method: {wb} -> {wa}  **FLIP**")
            flags.append(f"WINNER FLIP [cross_model/{short}]: {wb} -> {wa}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="dir with pre-fix JSONs")
    ap.add_argument("--after", required=True, help="dir with post-fix JSONs")
    ap.add_argument("--out", default=None, help="write markdown report here")
    args = ap.parse_args()

    lines: list[str] = ["# Spec-decode fix: pre-fix vs post-fix eval comparison\n"]
    lines.append("`before/` = pre-fix commit `f23d6ff` (contains regression `9a5a4c9`).  ")
    lines.append("`after/`  = post-fix commit `2e11321`.\n")
    flags: list[str] = []

    dispatch = {
        "failure_cases_results.json": compare_failure_cases,
        "ablation_live_results.json": compare_ablation,
        "cross_model_results.json": compare_cross_model,
    }
    accept_notes: list[str] = []
    for name in FILES:
        b, a = _load(args.before, name), _load(args.after, name)
        if b is None or a is None:
            lines.append(f"\n## {name}\n- SKIPPED (missing "
                         f"{'before' if b is None else 'after'} copy)\n")
            continue
        dispatch[name](b, a, lines, flags)
        for tag, node in (("before", b), ("after", a)):
            acc = _find_acceptance(node)
            if acc:
                accept_notes.append(f"{name} [{tag}]: " +
                                    ", ".join(f"{k}={v:.4f}" for k, v in acc.items()))

    lines.append("\n## Speculative-decoding acceptance rate\n")
    if accept_notes:
        lines += [f"- {n}" for n in accept_notes]
    else:
        lines.append("- No acceptance-rate field found in any result JSON. The eval "
                     "scripts do not persist it. Capture it separately on GPU via "
                     "`scripts/probe_spec_acceptance.py` (depth-8 default), or wire "
                     "`spec_decode_observer` into the run. Pre-fix value was 0.0% "
                     "(0/196); post-fix restored to ~1.1% at depth 8.")

    lines.append("\n## TOP-LEVEL FLAGS\n")
    if flags:
        lines += [f"- [FLAG] {f}" for f in flags]
    else:
        lines.append("- No winner flips or ranking changes detected. If spec-capable "
                     "rows nonetheless show MATERIAL metric deltas, the affected "
                     "conclusions still require revalidation before publication.")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
