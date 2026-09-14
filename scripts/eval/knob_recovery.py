"""
Causal recovery under regime shift: what adaptation is LOST if one knob is frozen?

WHAT THIS MEASURES (read before trusting any number)
----------------------------------------------------
A TRUE CLOSED-LOOP COUNTERFACTUAL, on real TinyLlama inference. For each
live-effective scheduler knob k in {max_batch_size, chunk_size} we rerun the
FULL controller with k pinned at its Static-Best value while every OTHER knob
stays under CARL's control (regime classification, reward, and LinUCB updates
all unchanged). We then compare that frozen-knob rollout against CARL-Full over
a NON-STATIONARY workload whose second half is a regime shift (INTERACTIVE ->
BATCH), and measure how much slower / worse CARL recovers.

WHY A RERUN, NOT LOG ANALYSIS (the methodological crux)
-------------------------------------------------------
Freezing k changes the config applied at a control cycle -> changes the realised
reward -> changes the LinUCB A/b update -> changes the next arm selection. The
frozen trajectory therefore DIVERGES from CARL-Full in workload state, bandit
state, and action sequence simultaneously; it is a genuinely different rollout,
NOT a subset or re-labelling of the CARL-Full log. So this cannot be
reconstructed off-policy from a CARL-Full log -- it is measured by RERUNNING the
controller under the frozen-knob policy. The counterfactual answered is:

    "What adaptation performance is lost if knob k is frozen at Static-Best
     while all OTHER knobs remain under CARL's control, across a regime shift?"

The recovery ANALYSIS (curves, t2a, AURC, deficits) is post-processing over the
reran trajectories; the trajectories themselves are controlled reruns.

REUSED HARNESS (nothing duplicated)
-----------------------------------
  knob_attribution._run_carl        -- the closed-loop rerun (frozen or full)
  ablation_live._serve              -- serving loop + per-step throughput log
  ablation_live._build_workload     -- the NON-STATIONARY (shift) workload
  ablation_live._frozen_arms        -- pins one knob across every bandit arm
  ablation_live.select_static_best  -- the pin value (held-out LHS validation)
  src.carl.*                        -- the UNCHANGED controller / bandit / reward

SCOPE (honest, same as knob_attribution)
----------------------------------------
This single-model live harness acts on max_batch_size and chunk_size only (no
router / KV / spec), so those are the only knobs whose freeze changes inference.
The recovery claim is therefore made for exactly those two knobs.

Run (GPU box / Colab T4; CPU works but is a correctness smoke, not a timing):
  python scripts/eval/knob_recovery.py --seeds 42,43,44 --limit 120   # smoke
  python scripts/eval/knob_recovery.py                                 # 10 seeds
Nothing is committed; results are written only to --out (default under docs/eval,
override to a scratch path for an unreviewed smoke).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from datetime import datetime

# --- path bootstrap: run standalone as `python scripts/eval/knob_recovery.py` --
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# Reuse the live harness wholesale (importing runs only import-time code; their
# main() is __main__-guarded). We do NOT modify ablation_live beyond the additive
# per-step log it now returns.
import ablation_live as abl  # noqa: E402
import knob_attribution as ka  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

# The two live-effective scheduler knobs (the only ones whose freeze changes
# inference in this single-model harness -- see module docstring / scope note).
KNOBS = ["max_batch_size", "chunk_size"]

DOCS_EVAL = abl.DOCS_EVAL
DEFAULT_OUT = os.path.join(DOCS_EVAL, "knob_recovery_results.json")

# Recovery-detection parameters (documented in the schema so a reader can see how
# t2a was defined). A metric is "recovered" once it returns within THRESHOLD of
# the run's own post-shift steady value and stays there for HOLD samples.
RECOVERY_THRESHOLD = 0.10     # within 10% of steady.
RECOVERY_HOLD = 3             # must stay recovered this many consecutive samples.
STEADY_TAIL_FRAC = 0.33       # steady value = mean over the last third post-shift.
TPUT_WINDOW_STEPS = 8         # sliding-window width for throughput(t) smoothing.
TTFT_WINDOW_REQS = 8          # sliding-window width for TTFT-P99(request) smoothing.


# ===========================================================================
# Rerun: CARL-Full + one config per frozen knob, over one seed.
# ===========================================================================


def _run_full(model, tokenizer, n: int, seed: int) -> dict:
    """CARL-Full rollout (every knob adaptive) -- the counterfactual reference."""
    return ka._run_carl(model, tokenizer, n, seed, None)


def _run_frozen(model, tokenizer, n: int, seed: int, knob: str, value) -> dict:
    """CARL rollout with exactly `knob` pinned at `value`; all else adaptive."""
    return ka._run_carl(model, tokenizer, n, seed, {knob: value})


# ===========================================================================
# Curve construction from the per-step log and per-request records.
# ===========================================================================


def throughput_curve(step_log: list, shift_t: float | None,
                     window: int = TPUT_WINDOW_STEPS) -> tuple[list, list]:
    """Smoothed throughput(t) relative to the shift, in tok/s.

    For each step i, throughput = sum(tokens[i-w+1..i]) / sum(dt[i-w+1..i]) over a
    sliding window of `window` steps (smooths the per-step token bursts). x is the
    step's wall time MINUS the shift time, so x<0 is pre-shift and x>=0 is the
    post-shift recovery region. Returns (x_since_shift, tput).
    """
    if not step_log:
        return [], []
    s0 = shift_t if shift_t is not None else 0.0
    xs, ys = [], []
    for i in range(len(step_log)):
        lo = max(0, i - window + 1)
        toks = sum(step_log[j]["tokens"] for j in range(lo, i + 1))
        dt = sum(step_log[j]["dt"] for j in range(lo, i + 1))
        if dt <= 0:
            continue
        xs.append(step_log[i]["t"] - s0)
        ys.append(toks / dt)
    return xs, ys


def ttft_curve(records: list, n_phase0: int,
               window: int = TTFT_WINDOW_REQS) -> tuple[list, list]:
    """Sliding-window TTFT-P99 (ms) vs post-shift request index.

    records carry request_id in submission order; request_id >= n_phase0 are the
    post-shift (BATCH) requests. We take a trailing `window` of requests and
    report their TTFT p99, indexed by (request_id - n_phase0). Returns
    (req_since_shift, ttft_p99).
    """
    recs = sorted(records, key=lambda r: r["request_id"])
    xs, ys = [], []
    for i, r in enumerate(recs):
        if r["request_id"] < n_phase0:
            continue
        lo = max(0, i - window + 1)
        w = [recs[j]["ttft_ms"] for j in range(lo, i + 1)]
        xs.append(r["request_id"] - n_phase0)
        ys.append(abl._percentile(w, 99))
    return xs, ys


def _steady_value(ys: list, tail_frac: float = STEADY_TAIL_FRAC) -> float:
    """Steady reference = mean of the last `tail_frac` of a post-shift series."""
    if not ys:
        return 0.0
    k = max(1, int(len(ys) * tail_frac))
    return statistics.fmean(ys[-k:])


def _post(xs: list, ys: list) -> list:
    """The post-shift (x >= 0) points of a curve, as (x, y) pairs."""
    return [(x, y) for x, y in zip(xs, ys) if x >= 0]


def t2a_to_target(post: list, target: float, *, higher_is_better: bool,
                  hold: int = RECOVERY_HOLD) -> tuple[float | None, bool]:
    """First x at which the metric reaches and HOLDS a COMMON `target`.

    This is the fixed methodology: recovery is judged against CARL-Full's
    post-shift steady value (a common reference passed in as `target`), NOT
    against each run's own steady. So a frozen policy stuck at a degraded level
    is correctly reported as NEVER recovering (returns (None, False)) instead of
    trivially "recovering" to its own bad steady.

    higher_is_better=True for throughput (reached = y >= target); False for
    TTFT-P99 (reached = y <= target). Returns (t2a_x, recovered_bool).
    """
    if not post or target <= 0:
        return None, False

    def ok(y: float) -> bool:
        return y >= target if higher_is_better else y <= target

    for i in range(len(post)):
        if all(ok(post[j][1]) for j in range(i, min(len(post), i + hold))):
            return post[i][0], True
    return None, False


def shortfall_integral(post: list, ref: float, hi: float,
                       *, higher_is_better: bool = True,
                       n_grid: int = 80) -> float | None:
    """Trapezoidal integral of the shortfall vs a COMMON reference over [0, hi].

    Shortfall = max(0, ref - y) for throughput (below the full-steady rate is
    bad), or max(0, y - ref) for TTFT (above the full-steady latency is bad). We
    interpolate the curve onto a uniform [0, hi] grid so runs of DIFFERENT
    post-shift durations are compared over the SAME fixed horizon -- removing the
    time-alignment confound the first smoke exposed. Units: tok (throughput) or
    ms*s (latency).
    """
    if hi <= 0 or len(post) < 2:
        return None
    xs = [x for x, _y in post]
    ys = [y for _x, y in post]
    grid = [hi * i / (n_grid - 1) for i in range(n_grid)]
    yi = _interp(xs, ys, grid)
    sf = [max(0.0, (ref - y) if higher_is_better else (y - ref)) for y in yi]
    return sum((sf[i] + sf[i + 1]) / 2.0 * (grid[i + 1] - grid[i])
               for i in range(n_grid - 1))


# ===========================================================================
# Aligned deltas: frozen vs CARL-Full on a common post-shift grid.
# ===========================================================================


def _interp(xs: list, ys: list, grid: list) -> list:
    """Linear interpolation of (xs, ys) onto `grid` (xs assumed ascending)."""
    out, j = [], 0
    for g in grid:
        while j < len(xs) - 1 and xs[j + 1] < g:
            j += 1
        if g <= xs[0]:
            out.append(ys[0])
        elif g >= xs[-1]:
            out.append(ys[-1])
        else:
            x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
            out.append(y0 + (y1 - y0) * (g - x0) / (x1 - x0) if x1 > x0 else y0)
    return out


def recovery_metrics(full: dict, frozen: dict) -> dict:
    """All per-seed recovery metrics for one (full, frozen) pair.

    COMMON-REFERENCE methodology (fixed after the first smoke):
      * The recovery TARGET is CARL-Full's post-shift steady value -- the same
        reference for both runs -- so a degraded frozen policy is scored as NOT
        recovering, rather than trivially recovering to its own bad steady.
      * Deficit / AURC / regret are SHORTFALLS vs that common reference,
        integrated over FIXED horizons, so runs of different post-shift durations
        are comparable:
          aurc_tokens  -- shortfall over [0, H], H = CARL-Full's post-shift
                          duration (the reference recovery window).
          regret_tokens -- shortfall over [0, T_frozen], the frozen run's OWN
                          post-shift duration (captures the extra time a stuck
                          policy keeps grinding below the full-steady rate).
          transient_deficit_tps -- mean rate shortfall over [0, H].
      * t2a is the time/requests to REACH AND HOLD the common target; None
        (censored) means "did not recover within the horizon".
    """
    fx, fy = throughput_curve(full["step_log"], full["shift_t"])
    zx, zy = throughput_curve(frozen["step_log"], frozen["shift_t"])
    f_post, z_post = _post(fx, fy), _post(zx, zy)

    # Common reference: CARL-Full's post-shift steady throughput, and horizons.
    full_steady = _steady_value([y for _x, y in f_post])
    tput_target = full_steady * (1.0 - RECOVERY_THRESHOLD)
    H = f_post[-1][0] if f_post else 0.0            # reference horizon (full).
    T_frozen = z_post[-1][0] if z_post else 0.0     # frozen's own duration.

    t2a_full_s, rec_full = t2a_to_target(f_post, tput_target, higher_is_better=True)
    t2a_frozen_s, rec_frozen = t2a_to_target(z_post, tput_target, higher_is_better=True)

    aurc = shortfall_integral(z_post, full_steady, H, higher_is_better=True)
    regret = shortfall_integral(z_post, full_steady, T_frozen, higher_is_better=True)
    deficit = (aurc / H) if (aurc is not None and H > 0) else None

    # TTFT: common target = CARL-Full's post-shift steady TTFT-p99; recovery in
    # request space; "during adapt" = frozen TTFT-p99 up to its t2a (or all
    # post-shift if censored).
    ftx, fty = ttft_curve(full["requests"], full["n_phase0"])
    ztx, zty = ttft_curve(frozen["requests"], frozen["n_phase0"])
    ft_post, zt_post = _post(ftx, fty), _post(ztx, zty)
    full_steady_ttft = _steady_value([y for _x, y in ft_post])
    ttft_target = full_steady_ttft * (1.0 + RECOVERY_THRESHOLD)
    t2a_full_req, rec_full_ttft = t2a_to_target(ft_post, ttft_target, higher_is_better=False)
    t2a_frozen_req, rec_frozen_ttft = t2a_to_target(zt_post, ttft_target, higher_is_better=False)

    ttft_adapt = None
    if zt_post:
        cut = t2a_frozen_req if t2a_frozen_req is not None else max(x for x, _y in zt_post)
        w = [y for x, y in zt_post if x <= cut]
        ttft_adapt = abl._percentile(w, 99) if w else None

    return {
        "full_steady_tps": full_steady,
        "horizon_full_s": H,
        "horizon_frozen_s": T_frozen,
        "t2a_full_s": t2a_full_s,
        "t2a_frozen_s": t2a_frozen_s,
        "frozen_recovered": rec_frozen,
        "t2a_full_req": float(t2a_full_req) if t2a_full_req is not None else None,
        "t2a_frozen_req": float(t2a_frozen_req) if t2a_frozen_req is not None else None,
        "frozen_ttft_recovered": rec_frozen_ttft,
        "transient_tput_deficit_tps": deficit,
        "aurc_tokens": aurc,
        "cumulative_regret_tokens": regret,
        "ttft_p99_during_adapt_ms": ttft_adapt,
        "full_throughput_tps": full["throughput_tps"],
        "frozen_throughput_tps": frozen["throughput_tps"],
        "full_ttft_p99_ms": full["ttft_p99"],
        "frozen_ttft_p99_ms": frozen["ttft_p99"],
        # Down-sampled curves for plotting / the mock (keep the JSON small).
        "curve": {
            "tput_x_full": fx, "tput_y_full": fy,
            "tput_x_frozen": zx, "tput_y_frozen": zy,
            "ttft_x_full": ftx, "ttft_y_full": fty,
            "ttft_x_frozen": ztx, "ttft_y_frozen": zty,
        },
    }


# ===========================================================================
# Cross-seed statistics (self-contained; no scipy).
# ===========================================================================

# Student-t 97.5% critical values for small df (two-sided 95% CI). df -> t.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def agg(vals: list) -> dict:
    """{mean, std, ci95, n} over per-seed values (None-tolerant, t-based CI)."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return {"mean": None, "std": None, "ci95": [None, None], "n": 0}
    mean = statistics.fmean(xs)
    if len(xs) < 2:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean], "n": len(xs)}
    std = statistics.stdev(xs)
    t = _T975.get(len(xs) - 1, 1.96)
    half = t * std / math.sqrt(len(xs))
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half], "n": len(xs)}


def paired_test(a: list, b: list) -> dict:
    """Paired t-test + Cohen's d for a (frozen) vs b (full) over matched seeds."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return {"n": n, "mean_diff": None, "t_statistic": None,
                "p_value": None, "cohens_d": None, "significant_0.05": False}
    diffs = [x - y for x, y in pairs]
    md = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return {"n": n, "mean_diff": md, "t_statistic": None, "p_value": None,
                "cohens_d": None, "significant_0.05": False}
    t = md / (sd / math.sqrt(n))
    d = md / sd
    # Two-sided p via the same regularised incomplete beta as the rest of the
    # suite -- but for a compact smoke we report |t| and let the reviewer read it
    # against the small-df critical value; p is approximated as significant iff
    # |t| exceeds the df-appropriate 97.5% critical value.
    crit = _T975.get(n - 1, 1.96)
    return {"n": n, "mean_diff": md, "t_statistic": t, "cohens_d": d,
            "t_crit_0.975": crit, "significant_0.05": bool(abs(t) > crit)}


# ===========================================================================
# Driver.
# ===========================================================================


def run(seeds: list, n: int, out_path: str, pins: dict | None) -> dict:
    env = abl.capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype {dtype} | {len(seeds)} seeds x {n} reqs "
          f"| knobs {KNOBS}", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU correctness smoke only; timing is NOT a "
              "T4 estimate.", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    # Pin values = Static-Best (held-out LHS), unless overridden for a fast smoke.
    if pins:
        static_dict = dict(pins)
        print(f"[static-best] OVERRIDE pins (smoke): {static_dict}", flush=True)
    else:
        static_cfg, _sel = abl.select_static_best(model, tokenizer, max(10, n // 2))
        static_dict = static_cfg.as_dict()
    pin_vals = {k: static_dict[k] for k in KNOBS}
    print(f"[pins] {pin_vals}", flush=True)

    t_start = time.perf_counter()
    per_seed: dict = {k: [] for k in KNOBS}       # knob -> list of recovery_metrics
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        try:
            t0 = time.perf_counter()
            full = _run_full(model, tokenizer, n, seed)
            print(f"  CARL-Full: {full['throughput_tps']:.1f} tok/s "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            for knob in KNOBS:
                tk = time.perf_counter()
                frozen = _run_frozen(model, tokenizer, n, seed, knob, pin_vals[knob])
                m = recovery_metrics(full, frozen)
                per_seed[knob].append(m)
                print(f"  freeze {knob}={pin_vals[knob]}: "
                      f"{frozen['throughput_tps']:.1f} tok/s, "
                      f"t2a_frozen={m['t2a_frozen_s']}, "
                      f"regret={m['cumulative_regret_tokens']} "
                      f"({time.perf_counter()-tk:.1f}s)", flush=True)
        except Exception:
            print(f"  seed {seed} FAILED", flush=True)
            traceback.print_exc()
    wall = time.perf_counter() - t_start

    # Aggregate each metric across seeds; paired frozen-vs-full where meaningful.
    metric_keys = ["full_steady_tps", "horizon_full_s", "horizon_frozen_s",
                   "t2a_frozen_s", "t2a_full_s", "t2a_frozen_req", "t2a_full_req",
                   "transient_tput_deficit_tps", "aurc_tokens",
                   "cumulative_regret_tokens", "ttft_p99_during_adapt_ms",
                   "frozen_throughput_tps", "full_throughput_tps",
                   "frozen_ttft_p99_ms", "full_ttft_p99_ms"]
    knobs_out: dict = {}
    for knob in KNOBS:
        runs = per_seed[knob]
        agg_metrics = {k: agg([r[k] for r in runs]) for k in metric_keys}
        # Fraction of seeds whose frozen run recovered to the common target (a
        # censored t2a means "did not recover" -- the correct verdict for a
        # degraded freeze).
        n_runs = len(runs) or 1
        agg_metrics["frozen_recovered_fraction"] = {
            "throughput": sum(1 for r in runs if r["frozen_recovered"]) / n_runs,
            "ttft": sum(1 for r in runs if r["frozen_ttft_recovered"]) / n_runs,
            "n": len(runs),
        }
        tests = {
            "t2a_frozen_vs_full_s": paired_test([r["t2a_frozen_s"] for r in runs],
                                                [r["t2a_full_s"] for r in runs]),
            "throughput_frozen_vs_full": paired_test(
                [r["frozen_throughput_tps"] for r in runs],
                [r["full_throughput_tps"] for r in runs]),
            "ttft_p99_frozen_vs_full": paired_test(
                [r["frozen_ttft_p99_ms"] for r in runs],
                [r["full_ttft_p99_ms"] for r in runs]),
        }
        knobs_out[knob] = {
            "frozen_value": pin_vals[knob], "live_effective": True,
            "metrics": agg_metrics, "paired_tests": tests,
            "curves_per_seed": [{"seed": s, **r["curve"]}
                                for s, r in zip(seeds, runs)],
        }

    results = {
        "description": ("Causal recovery under regime shift: freeze one "
                        "live-effective knob at Static-Best; all other knobs stay "
                        "adaptive; measure recovery vs CARL-Full."),
        "method": ("closed-loop RERUN of the controller with one knob pinned "
                   "(NOT reconstructed from CARL-Full logs)"),
        "counterfactual": ("adaptation performance lost if knob k is frozen while "
                           "all other knobs remain under CARL control, across "
                           "INTERACTIVE->BATCH"),
        "environment": env,
        "workload": {"scenario": "NON-STATIONARY", "n_requests": n,
                     "pre": "INTERACTIVE", "post": "BATCH"},
        "seeds": seeds,
        "recovery": {"threshold_pct": RECOVERY_THRESHOLD * 100, "hold": RECOVERY_HOLD,
                     "steady_tail_frac": STEADY_TAIL_FRAC,
                     "tput_window_steps": TPUT_WINDOW_STEPS,
                     "ttft_window_reqs": TTFT_WINDOW_REQS},
        "static_best_source": "override" if pins else "latin_hypercube_validation",
        "knobs": knobs_out,
        "runtime_s": wall,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    print(f"\nWall time: {wall:.1f}s on {DEVICE} "
          f"({'NOT a T4 estimate' if DEVICE.type != 'cuda' else 'GPU'}).")
    print(f"Saved (UNCOMMITTED) to {out_path}", flush=True)
    return results


def _print_summary(results: dict) -> None:
    print("\n=== KNOB-FREEZE RECOVERY (frozen vs CARL-Full, mean +/- std) ===")
    for knob, v in results["knobs"].items():
        m = v["metrics"]

        def show(key, unit=""):
            a = m[key]
            if a["mean"] is None:
                return "n/a"
            return f"{a['mean']:.2f}+/-{a['std']:.2f}{unit}"

        rec = v["metrics"]["frozen_recovered_fraction"]
        print(f"\n[{knob} pinned @ {v['frozen_value']}]")
        print(f"  full-steady target     : {show('full_steady_tps',' tok/s')}")
        print(f"  frozen recovered?      : tput {rec['throughput']*100:.0f}% of seeds, "
              f"ttft {rec['ttft']*100:.0f}% of seeds")
        print(f"  t2a  full={show('t2a_full_s','s')}  frozen={show('t2a_frozen_s','s')} "
              f"(None = never reached full-steady)")
        print(f"  transient tput deficit : {show('transient_tput_deficit_tps',' tok/s')}")
        print(f"  AURC (tokens)          : {show('aurc_tokens')}")
        print(f"  cumulative regret (tok): {show('cumulative_regret_tokens')}")
        print(f"  TTFT-p99 during adapt  : {show('ttft_p99_during_adapt_ms',' ms')}")
        pt = v["paired_tests"]["throughput_frozen_vs_full"]
        if pt["t_statistic"] is not None:
            print(f"  paired tput frozen-vs-full: mean_diff={pt['mean_diff']:.2f} "
                  f"tok/s, t={pt['t_statistic']:.2f}, d={pt['cohens_d']:.2f}, "
                  f"{'SIG' if pt['significant_0.05'] else 'n.s.'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Causal knob-freeze recovery study.")
    p.add_argument("--seeds", default="42,43,44",
                   help="comma-separated seeds (smoke default 42,43,44)")
    p.add_argument("--limit", type=int, default=120, help="requests per run")
    p.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    p.add_argument("--pin-mb", type=int, default=None,
                   help="override Static-Best max_batch_size pin (fast smoke)")
    p.add_argument("--pin-cs", type=int, default=None,
                   help="override Static-Best chunk_size pin (fast smoke)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pins = None
    if args.pin_mb is not None and args.pin_cs is not None:
        pins = {"max_batch_size": args.pin_mb, "chunk_size": args.pin_cs}
    run(seeds, args.limit, args.out, pins)


if __name__ == "__main__":
    main()
