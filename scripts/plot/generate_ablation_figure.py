"""
Generate the live-ablation figure for the CARL paper from the measured results
written by scripts/eval/ablation_live.py.

Reads  docs/eval/ablation_live_results.json  and emits, into docs/eval/figures/:
  * ablation_live_throughput.pdf / .png -- grouped bar chart of mean throughput
    (tok/s) per config with +/- std error bars, live-effective configs in solid
    colour and the no-live-effect ablations hatched/greyed (they measure ~=
    CARL-Full by design in this single-model harness), CARL-Full highlighted,
    and DynOracle drawn as a dashed upper-bound line.
  * ablation_live_contributions.pdf / .png -- ranked subsystem contribution bars
    (delta tput = CARL-Full - CARL-NoX).

Design notes:
  - Pure matplotlib (Agg backend), no seaborn, so it runs headless on Colab/CI.
  - Everything is driven off the JSON; nothing is hard-coded except the canonical
    config order, which mirrors ablation_live.CONFIGS so the bars read left->right
    the same way the table prints.
  - Fails LOUDLY with a clear message if the results file is missing, so a stale
    figure is never silently regenerated from nothing.

Run:
  python scripts/plot/generate_ablation_figure.py
  python scripts/plot/generate_ablation_figure.py --results path/to/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_RESULTS = os.path.join(_REPO_ROOT, "docs", "eval", "ablation_live_results.json")
FIG_DIR = os.path.join(_REPO_ROOT, "docs", "eval", "figures")

# Canonical left->right order (mirrors ablation_live.CONFIGS).
CONFIG_ORDER = ["CARL-Full", "CARL-NoSched", "CARL-NoSpec", "CARL-NoCache",
                "CARL-NoRouter", "CARL-NoChunk", "Static-Best", "AutoTuner",
                "CARL-Thompson", "DynOracle"]


def _load(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"ERROR: results file not found: {path}\n"
                 f"Run scripts/eval/ablation_live.py first to produce it.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, stem: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"  wrote {out}", flush=True)


def plot_throughput(results: dict, plt) -> None:
    cfgs = results["configs"]
    names = [n for n in CONFIG_ORDER if n in cfgs]
    means = [cfgs[n]["throughput_tps_mean"] for n in names]
    stds = [cfgs[n]["throughput_tps_std"] for n in names]
    live = [bool(cfgs[n].get("live_effective")) for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(names))
    for i, n in enumerate(names):
        if n == "CARL-Full":
            colour, hatch = "#1f77b4", None          # highlight the headline config
        elif n == "DynOracle":
            colour, hatch = "#2ca02c", None          # oracle upper bound
        elif live[i]:
            colour, hatch = "#4c78a8", None          # live-effective baseline/ablation
        else:
            colour, hatch = "#bbbbbb", "//"          # no live effect (~= CARL-Full)
        ax.bar(i, means[i], yerr=stds[i], capsize=4, color=colour, hatch=hatch,
               edgecolor="black", linewidth=0.6, label=None)

    # DynOracle as a dashed upper-bound reference line across the panel.
    if "DynOracle" in cfgs:
        dyn = cfgs["DynOracle"]["throughput_tps_mean"]
        ax.axhline(dyn, ls="--", lw=1.0, color="#2ca02c", alpha=0.7,
                   label=f"DynOracle UB ({dyn:.0f} tok/s)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Throughput (tok/s)")
    gap = results.get("oracle_gap_pct")
    title = "CARL live ablation (real TinyLlama, NON-STATIONARY): throughput mean +/- std"
    if gap is not None:
        title += f"\noracle gap = {gap:+.1f}%"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Legend: explain the hatch convention + oracle line.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="CARL-Full"),
        Patch(facecolor="#4c78a8", edgecolor="black", label="live-effective"),
        Patch(facecolor="#bbbbbb", edgecolor="black", hatch="//",
              label="no live effect (~= CARL-Full)"),
    ]
    if "DynOracle" in cfgs:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], ls="--", color="#2ca02c", label="DynOracle UB"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    _save(fig, "ablation_live_throughput")
    plt.close(fig)


def plot_contributions(results: dict, plt) -> None:
    contrib = results.get("subsystem_contributions", {})
    if not contrib:
        print("  (no subsystem_contributions in results; skipping contribution plot)")
        return
    subs = list(contrib.keys())            # already ranked desc by the eval harness
    vals = [contrib[s] for s in subs]

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#d62728" if v >= 0 else "#7f7f7f" for v in vals]
    ax.bar(range(len(subs)), vals, color=colours, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=20, ha="right")
    ax.set_ylabel("delta throughput vs CARL-Full (tok/s)")
    ax.set_title("Subsystem contributions (CARL-Full - CARL-NoX), ranked", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "ablation_live_contributions")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the CARL live-ablation figure.")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="path to ablation_live_results.json")
    args = parser.parse_args()

    results = _load(args.results)
    try:
        import matplotlib
        matplotlib.use("Agg")              # headless: no display needed on Colab/CI
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("ERROR: matplotlib is required (pip install matplotlib).")

    print(f"Plotting from {args.results} -> {FIG_DIR}", flush=True)
    plot_throughput(results, plt)
    plot_contributions(results, plt)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
