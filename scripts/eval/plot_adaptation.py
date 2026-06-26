"""
Publication-quality plots of CARL's adaptation dynamics.

INPUT  : docs/eval/raw/adaptation/decisions_<seed>.csv  (produced by
         scripts/eval/adaptation_analysis.py)
OUTPUT : docs/eval/figures/adaptation_<plot>_seed<seed>.png

Three figures per seed:
  1. reward vs cycle          -- realised reward and the oracle reward, with
                                 arm-switch dots, regime boundaries, convergence.
  2. cumulative regret vs cycle
  3. arm timeline             -- selected arm index per cycle, regime shaded,
                                 boundaries and convergence marked.

This script READS ONLY the generated CSVs. It never imports the engine, never
touches torch, and needs no GPU -- every marker it draws (regime transitions,
arm switches, convergence point) is read straight from the CSV's boolean
columns, so plotting never has to re-infer an event.

Matplotlib only (Agg backend, so it works headless on a CI/Colab box).

Run:
  python scripts/eval/plot_adaptation.py                 # all seeds found
  python scripts/eval/plot_adaptation.py --seeds 42      # just seed 42
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")            # headless: render straight to PNG, no display
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(_REPO_ROOT, "docs", "eval", "raw", "adaptation")
FIG_DIR = os.path.join(_REPO_ROOT, "docs", "eval", "figures")

# A calm, print-friendly palette (colour-blind-safe blue/orange/green).
_C_REWARD = "#1f77b4"
_C_ORACLE = "#7f7f7f"
_C_REGRET = "#d62728"
_C_ARM = "#2ca02c"
_C_SWITCH = "#ff7f0e"
_C_TRANSITION = "#9467bd"
_C_CONVERGE = "#111111"


def _to_float(s, default=0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _to_bool(s) -> bool:
    """CSV stores Python bools as the strings 'True'/'False'."""
    return str(s).strip().lower() in ("true", "1")


def _read_csv(path: str) -> list:
    """Load a decisions CSV into a list of typed row dicts."""
    rows: list = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "cycle": int(_to_float(r.get("cycle"))),
                "step": int(_to_float(r.get("step"))),
                "regime": r.get("regime", ""),
                "arm": int(_to_float(r.get("arm"))),
                "reward": _to_float(r.get("reward")),
                "oracle_reward": _to_float(r.get("oracle_reward")),
                "cumulative_regret": _to_float(r.get("cumulative_regret")),
                "is_regime_transition": _to_bool(r.get("is_regime_transition")),
                "is_arm_switch": _to_bool(r.get("is_arm_switch")),
                "is_convergence_point": _to_bool(r.get("is_convergence_point")),
            })
    return rows


def _regime_spans(rows: list) -> list:
    """Contiguous (regime, start_cycle, end_cycle) spans for background shading."""
    spans: list = []
    if not rows:
        return spans
    start = rows[0]
    cur = start["regime"]
    for r in rows[1:]:
        if r["regime"] != cur:
            spans.append((cur, start["cycle"], r["cycle"]))
            start, cur = r, r["regime"]
    spans.append((cur, start["cycle"], rows[-1]["cycle"]))
    return spans


def _mark_transitions(ax, rows: list) -> None:
    """Vertical lines at regime boundaries + a single convergence line."""
    for r in rows:
        if r["is_regime_transition"]:
            ax.axvline(r["cycle"], color=_C_TRANSITION, ls="--", lw=1.2, alpha=0.8,
                       label="_regime boundary")
        if r["is_convergence_point"]:
            ax.axvline(r["cycle"], color=_C_CONVERGE, ls=":", lw=1.6, alpha=0.9,
                       label="_convergence")


def _legend_once(ax, handles_labels: list) -> None:
    """Attach a de-duplicated legend from explicit (handle, label) pairs."""
    seen, hs, ls = set(), [], []
    for h, l in handles_labels:
        if l and l not in seen:
            seen.add(l)
            hs.append(h)
            ls.append(l)
    if hs:
        ax.legend(hs, ls, fontsize=8, loc="best", framealpha=0.9)


def plot_reward(rows: list, seed: int, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cycles = [r["cycle"] for r in rows]
    ax.plot(cycles, [r["reward"] for r in rows], color=_C_REWARD, lw=1.6,
            marker="o", ms=3, label="CARL reward")
    ax.plot(cycles, [r["oracle_reward"] for r in rows], color=_C_ORACLE, lw=1.3,
            ls="--", label="oracle reward")
    sw = [r for r in rows if r["is_arm_switch"]]
    if sw:
        ax.scatter([r["cycle"] for r in sw], [r["reward"] for r in sw],
                   color=_C_SWITCH, s=42, zorder=5, label="arm switch")
    _mark_transitions(ax, rows)
    ax.set_xlabel("control cycle")
    ax.set_ylabel("reward (utility)")
    ax.set_title(f"CARL reward vs cycle (seed {seed})")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    extra = [(plt.Line2D([], [], color=_C_TRANSITION, ls="--"), "regime boundary"),
             (plt.Line2D([], [], color=_C_CONVERGE, ls=":"), "convergence")]
    _legend_once(ax, list(zip(handles, labels)) + extra)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_regret(rows: list, seed: int, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cycles = [r["cycle"] for r in rows]
    ax.plot(cycles, [r["cumulative_regret"] for r in rows], color=_C_REGRET,
            lw=1.8, marker="o", ms=3, label="cumulative regret")
    _mark_transitions(ax, rows)
    ax.set_xlabel("control cycle")
    ax.set_ylabel("cumulative regret vs oracle")
    ax.set_title(f"CARL cumulative regret vs cycle (seed {seed})")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    extra = [(plt.Line2D([], [], color=_C_TRANSITION, ls="--"), "regime boundary"),
             (plt.Line2D([], [], color=_C_CONVERGE, ls=":"), "convergence")]
    _legend_once(ax, list(zip(handles, labels)) + extra)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_arm_timeline(rows: list, seed: int, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cycles = [r["cycle"] for r in rows]

    # Shade each regime span so the regime context is obvious behind the arms.
    spans = _regime_spans(rows)
    regimes = sorted({s[0] for s in spans})
    shade = {rg: c for rg, c in zip(regimes, ["#e8f0fe", "#fff3e0", "#e8f5e9",
                                              "#fce4ec", "#f3e5f5"])}
    for rg, lo, hi in spans:
        ax.axvspan(lo, hi, color=shade.get(rg, "#f0f0f0"), alpha=0.6, lw=0)

    ax.step(cycles, [r["arm"] for r in rows], where="post", color=_C_ARM, lw=1.8,
            marker="o", ms=3, label="selected arm")
    sw = [r for r in rows if r["is_arm_switch"]]
    if sw:
        ax.scatter([r["cycle"] for r in sw], [r["arm"] for r in sw],
                   color=_C_SWITCH, s=42, zorder=5, label="arm switch")
    _mark_transitions(ax, rows)
    ax.set_xlabel("control cycle")
    ax.set_ylabel("arm index")
    ax.set_title(f"CARL arm timeline (seed {seed})")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    extra = [(plt.Line2D([], [], color=_C_TRANSITION, ls="--"), "regime boundary"),
             (plt.Line2D([], [], color=_C_CONVERGE, ls=":"), "convergence")]
    extra += [(plt.Rectangle((0, 0), 1, 1, color=shade.get(rg, "#f0f0f0")), f"regime: {rg}")
              for rg in regimes]
    _legend_once(ax, list(zip(handles, labels)) + extra)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_seed(csv_path: str) -> list:
    """Render all three figures for one decisions CSV. Returns the PNG paths."""
    base = os.path.basename(csv_path)
    seed_token = base.replace("decisions_", "").replace(".csv", "")
    try:
        seed = int(seed_token)
    except ValueError:
        seed = seed_token
    rows = _read_csv(csv_path)
    if not rows:
        print(f"  {base}: empty, skipped", flush=True)
        return []
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = f"seed{seed:03d}" if isinstance(seed, int) else f"seed{seed}"
    outs = {
        "reward": os.path.join(FIG_DIR, f"adaptation_reward_{tag}.png"),
        "regret": os.path.join(FIG_DIR, f"adaptation_regret_{tag}.png"),
        "arms": os.path.join(FIG_DIR, f"adaptation_arms_{tag}.png"),
    }
    plot_reward(rows, seed, outs["reward"])
    plot_regret(rows, seed, outs["regret"])
    plot_arm_timeline(rows, seed, outs["arms"])
    print(f"  {base} -> {', '.join(os.path.basename(p) for p in outs.values())}",
          flush=True)
    return list(outs.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CARL adaptation dynamics from decision CSVs.")
    parser.add_argument("--seeds", default="",
                        help="comma-separated seeds to plot (default: all CSVs found)")
    args = parser.parse_args()

    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        csvs = [os.path.join(RAW_DIR, f"decisions_{s:03d}.csv") for s in seeds]
        csvs = [p for p in csvs if os.path.exists(p)]
    else:
        csvs = sorted(glob.glob(os.path.join(RAW_DIR, "decisions_*.csv")))

    if not csvs:
        print(f"No decision CSVs found under {RAW_DIR}. "
              f"Run scripts/eval/adaptation_analysis.py first.", flush=True)
        return

    print(f"Plotting {len(csvs)} seed(s) -> {FIG_DIR}", flush=True)
    for p in csvs:
        plot_seed(p)


if __name__ == "__main__":
    main()
