"""
Render the 100k long-horizon INTERACTIVE stability figures (A-F) from the
existing results JSON. FIGURE-ONLY: reads docs/eval/stability_100k_results.json
and writes PNGs -- it NEVER reruns the experiment.

Figure roles (see the protocol): A (oracle capture) and B (regret RATE) are the
main-paper figures; C-F are appendix. B is the regret RATE (incremental
oracle-regret per checkpoint), NOT cumulative regret -- the stability quantity.

Design: publication line charts, one axis per panel (no dual-axis; C and E are
split into two panels), Okabe-Ito colourblind-safe categorical palette for the
4-series capture figure (validated: CVD sep 17.9), single-series figures carry no
legend (the title names the series). Captions are written to
docs/eval/figures/stability_captions.md.

Run:
  python scripts/plot/plot_stability_100k.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                       # headless render, no display needed.
import matplotlib.pyplot as plt             # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(_REPO_ROOT, "docs", "eval", "stability_100k_results.json")
FIG_DIR = os.path.join(_REPO_ROOT, "docs", "eval", "figures")

# Okabe-Ito colourblind-safe palette (validated categorical subset). CARL is the
# emphasised series; baselines are the muted references.
C_CARL = "#0072B2"
C_STATIC = "#E69F00"
C_UCB1 = "#009E73"
C_EPS = "#CC79A7"
INK = "#222222"
MUTED = "#666666"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
})


def _kreq(requests: list) -> list:
    """Requests -> thousands, for a compact x-axis."""
    return [r / 1000.0 for r in requests]


def load() -> dict:
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Figures.
# ===========================================================================


def fig_A(d: dict, path: str) -> None:
    """A (MAIN): cumulative oracle-capture % vs requests -- CARL + baselines."""
    ck = d["carl_checkpoints"]
    x = _kreq([c["requests"] for c in ck])
    carl = [c["cumulative"]["oracle_capture_pct"] for c in ck]
    base = d["figures"]["baseline_capture"]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(x, base["Static-Best"], color=C_STATIC, lw=1.3, label="Static-Best")
    ax.plot(x, base["UCB1"], color=C_UCB1, lw=1.3, label="UCB1")
    ax.plot(x, base["EpsilonGreedy"], color=C_EPS, lw=1.3, label="$\\epsilon$-greedy")
    ax.plot(x, carl, color=C_CARL, lw=2.2, label="CARL")
    ax.annotate("CARL", (x[-1], carl[-1]), color=C_CARL, fontweight="bold",
                fontsize=9, xytext=(4, 0), textcoords="offset points", va="center")
    ax.set_xlabel("Requests (thousands)")
    ax.set_ylabel("Oracle capture (%)")
    ax.set_title("A  Oracle capture holds near-optimal over 100k requests")
    ax.set_ylim(98.5, 100.15)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_B(d: dict, path: str) -> None:
    """B (MAIN): regret RATE = incremental oracle-regret per checkpoint.

    The reframe: not cumulative regret (which is linear only because of an
    irreducible per-cycle noise floor), but the regret RATE -- flat/bounded ->
    the true stability signal.
    """
    ck = d["carl_checkpoints"]
    x = _kreq([c["requests"] for c in ck])
    inc = [c["incremental_regret"] for c in ck]
    mean = sum(inc) / len(inc)
    mk = d["analysis"]["incremental_regret_mk"]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(x, inc, color=C_CARL, lw=1.6)
    ax.axhline(mean, color=MUTED, lw=1.0, ls="--")
    ax.annotate(f"mean {mean:.2f}", (x[-1], mean), color=MUTED, fontsize=8,
                xytext=(4, 4), textcoords="offset points", va="bottom", ha="right")
    ax.set_xlabel("Requests (thousands)")
    ax.set_ylabel("Regret rate\n(incremental regret per 1k requests)")
    ax.set_title("B  Regret rate stays flat (bounded, non-increasing)")
    ax.set_ylim(0, max(inc) * 1.3)
    ax.annotate(f"Mann-Kendall trend: none (p={mk['p_value']:.2f})",
                (0.03, 0.06), xycoords="axes fraction", fontsize=8, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def fig_C(d: dict, path: str) -> None:
    """C (appendix): SLO satisfaction + TTFT p99 -- TWO panels (no dual axis)."""
    ck = d["carl_checkpoints"]
    x = _kreq([c["requests"] for c in ck])
    slo = [c["window"]["slo_rate"] * 100 for c in ck]
    ttft = [c["window"]["ttft_p99_ms"] for c in ck]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.plot(x, slo, color=C_CARL, lw=1.6)
    a1.set_xlabel("Requests (thousands)"); a1.set_ylabel("SLO satisfaction (%)")
    a1.set_title("C1  SLO satisfaction")
    a2.plot(x, ttft, color=C_CARL, lw=1.6)
    a2.set_xlabel("Requests (thousands)"); a2.set_ylabel("TTFT p99 (ms)")
    a2.set_title("C2  TTFT p99")
    fig.savefig(path)
    plt.close(fig)


def fig_D(d: dict, path: str) -> None:
    """D (appendix): arm-switch rate vs requests (single series)."""
    ck = d["carl_checkpoints"]
    x = _kreq([c["requests"] for c in ck])
    sw = [c["arm_switch_rate"] for c in ck]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(x, sw, color=C_CARL, lw=1.6)
    ax.set_xlabel("Requests (thousands)")
    ax.set_ylabel("Arm-switch rate (per cycle)")
    ax.set_title("D  Arm-switch rate")
    ax.set_ylim(-0.02, max(0.1, max(sw) * 1.3))
    if max(sw) == 0:
        ax.annotate("0 across all checkpoints\n(converged before first checkpoint)",
                    (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                    fontsize=9, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def fig_E(d: dict, path: str) -> None:
    """E (appendix): LinUCB condition number + ||theta|| -- TWO panels."""
    ck = d["carl_checkpoints"]
    x = _kreq([c["requests"] for c in ck])
    cond = [c["linucb"]["cond_max"] for c in ck]
    theta = [c["linucb"]["theta_norm_max"] for c in ck]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.plot(x, cond, color=C_CARL, lw=1.6)
    a1.set_xlabel("Requests (thousands)")
    a1.set_ylabel("Max condition number")
    a1.set_title("E1  LinUCB conditioning")
    a2.plot(x, theta, color=C_CARL, lw=1.6)
    a2.set_xlabel("Requests (thousands)")
    a2.set_ylabel(r"Max $\|\theta\|$")
    a2.set_title(r"E2  LinUCB $\|\theta\|$")
    fig.savefig(path)
    plt.close(fig)


def fig_F(d: dict, path: str) -> None:
    """F (appendix): workload validity -- regime mix + queue percentiles."""
    v = d["validity"]
    mix = v["regime_mix"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    labels = list(mix.keys())
    vals = [mix[k] * 100 for k in labels]
    colors = [C_CARL if k == "interactive" else C_STATIC for k in labels]
    ax.barh(labels, vals, color=colors, height=0.55)
    for i, val in enumerate(vals):
        ax.annotate(f"{val:.2f}%", (val, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Share of requests (%)")
    ax.set_title("F  Workload validity: regime mix "
                 f"(queue p99={v['queue_p99']:.0f}, max={v['queue_max']:.0f})")
    ax.set_xlim(0, 108)
    fig.savefig(path)
    plt.close(fig)


CAPTIONS = {
    "A": ("Figure A (main). Cumulative oracle-capture (%) over the 100k-request "
          "INTERACTIVE horizon (seed 42, simulation). CARL holds ~99.97% of the "
          "hindsight DynOracle throughout (slope 95% CI excludes negative), "
          "matching Static-Best and exceeding the context-free UCB1/epsilon-greedy "
          "baselines."),
    "B": ("Figure B (main). Regret RATE -- incremental oracle-regret per "
          "1,000-request checkpoint -- for CARL over the 100k horizon. The rate is "
          "flat and bounded (Mann-Kendall trend not significant, p=0.10); the "
          "residual reflects the irreducible per-cycle noise floor of the "
          "clipped regret against a stochastic cost model, not a growing learning "
          "cost. (Cumulative regret is linear for this reason; the RATE is the "
          "stability quantity.)"),
    "C": ("Figure C (appendix). Operational stability: SLO satisfaction (C1) and "
          "TTFT p99 (C2) per trailing-window checkpoint remain steady across the "
          "horizon."),
    "D": ("Figure D (appendix). Per-cycle arm-switch rate is 0 across all "
          "checkpoints -- CARL settles on its arm before the first checkpoint."),
    "E": ("Figure E (appendix). LinUCB numerical health: max design-matrix "
          "condition number (E1) and max ||theta|| (E2) across the INTERACTIVE "
          "arms. The condition number is small and numerically safe at this "
          "horizon; ||theta|| stays bounded."),
    "F": ("Figure F (appendix). Workload validity: the calibrated arrival model "
          "(rho=0.6, single-turn, lognormal(48,24)) yields a 99.9% INTERACTIVE "
          "regime mix with a shallow queue (p99=5), confirming the experiment "
          "operates in the intended regime."),
}


def main() -> None:
    if not os.path.exists(RESULTS):
        sys.exit(f"results not found: {RESULTS} (run stability_100k.py first)")
    os.makedirs(FIG_DIR, exist_ok=True)
    d = load()

    renderers = {"A": fig_A, "B": fig_B, "C": fig_C, "D": fig_D, "E": fig_E, "F": fig_F}
    written = []
    for key, fn in renderers.items():
        path = os.path.join(FIG_DIR, f"stability_{key}.png")
        fn(d, path)
        written.append(path)
        print(f"  wrote {os.path.relpath(path, _REPO_ROOT)}")

    cap_path = os.path.join(FIG_DIR, "stability_captions.md")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write("# 100k stability figures -- captions\n\n")
        for k in "ABCDEF":
            f.write(f"**{k}.** {CAPTIONS[k]}\n\n")
    written.append(cap_path)
    print(f"  wrote {os.path.relpath(cap_path, _REPO_ROOT)}")
    print(f"\n{len(written)} artifacts written to {os.path.relpath(FIG_DIR, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
