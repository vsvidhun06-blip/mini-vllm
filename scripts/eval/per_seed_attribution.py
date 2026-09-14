"""
READ-ONLY per-seed analysis for the per-knob scheduler attribution.

This script does NOT run any inference and does NOT touch the eval implementation.
It post-processes the raw per-run JSON that scripts/eval/knob_attribution.py
already writes to docs/eval/raw/knob_attribution/, building the per-seed table
that distinguishes a real causal contribution from measurement variance.

Paired by seed, it computes:

    delta_seed = CARL-Full[seed].throughput_tps - freeze_<knob>[seed].throughput_tps

for the ambiguous knob (preemption_enabled) and the live-effective baseline
(max_batch_size), then reports mean, std, and a Student-t 95% CI across seeds.
A CI that crosses zero is consistent with noise (an inert knob); a tight CI well
clear of zero is a real, low-variance causal effect (the live baseline).

Run AFTER `python scripts/eval/knob_attribution.py` has produced the raw files:
    python scripts/eval/per_seed_attribution.py                       # default raw dir
    python scripts/eval/per_seed_attribution.py path/to/raw/knob_attribution
"""
import json
import math
import os
import sys

# Repo root = two levels up from scripts/eval/, so the default raw path resolves
# regardless of the caller's working directory (e.g. a fresh Colab clone).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_RAW = os.path.join(_REPO_ROOT, "docs", "eval", "raw", "knob_attribution")

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
# Student-t 0.975 quantile by dof (n-1). 9 dof (n=10) = 2.262.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 19: 2.093, 29: 2.045}


def _tput(raw_dir, tag, seed):
    path = os.path.join(raw_dir, f"{tag}_run_{seed:03d}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["throughput_tps"]


def _summary(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    se = std / math.sqrt(n) if n else 0.0
    t = _T975.get(n - 1, 2.262)
    return mean, std, mean - t * se, mean + t * se


def table(raw_dir, knob):
    tag = f"freeze_{knob}"
    rows, deltas = [], []
    for s in SEEDS:
        try:
            full = _tput(raw_dir, "CARL-Full", s)
            frozen = _tput(raw_dir, tag, s)
        except FileNotFoundError as e:
            print(f"  seed {s}: MISSING ({e.filename})")
            continue
        d = full - frozen
        deltas.append(d)
        rows.append((s, full, frozen, d))
    print(f"\n=== {knob} (freeze@static-best vs CARL-Full), paired per seed ===")
    print("| seed | CARL-Full tps | frozen tps | delta_vs_full |")
    print("| ---- | ------------- | ---------- | ------------- |")
    for s, full, frozen, d in rows:
        print(f"| {s} | {full:8.2f} | {frozen:8.2f} | {d:+8.3f} |")
    if deltas:
        mean, std, lo, hi = _summary(deltas)
        crosses = "YES (consistent with noise)" if lo <= 0 <= hi else "no"
        print(f"  mean={mean:+.3f}  std={std:.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]"
              f"  CI crosses 0: {crosses}")
    return deltas


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_RAW
    print(f"raw dir: {raw}")
    table(raw, "preemption_enabled")
    table(raw, "max_batch_size")


if __name__ == "__main__":
    main()
