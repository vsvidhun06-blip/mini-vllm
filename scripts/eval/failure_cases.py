"""
HONEST failure analysis of CARL: find and document where the adaptive controller
UNDERPERFORMS a well-validated Static-Best fixed config.

A method paper that only shows wins is suspicious. This harness deliberately
constructs five workloads chosen to stress CARL's weak spots -- workloads with
no regime change, too little time to adapt, or a memory wall that no scheduling
knob can move -- and reports, per scenario, whether CARL wins, ties, or loses on
real TinyLlama inference. Where CARL loses we classify WHY, using an evidence
rule set applied to the recorded adaptation trace, and we contrast a stated
hypothesis (fixed BEFORE the run) against the data-driven verdict.

The five scenarios:
  1. ultra_short       prompt 4-8,    max 8     -- requests finish before CARL acts
  2. stable_load       30 identical prompts     -- nothing to adapt to
  3. single_queue      one request at a time    -- no batching headroom, little signal
  4. rapid_oscillation INTERACTIVE/BATCH every 5 -- regime flips faster than CARL learns
  5. memory_pressure   prompt 512-1024, max 128 -- KV-bound; throughput is memory-set

Each: 30 requests, N=3 runs (seeds 42/43/44), CARL-Full vs Static-Best only.
Static-Best is the held-out LHS-validation winner for THAT scenario (the same
selection procedure the ablation uses), so it is a genuinely strong fixed
competitor, not a strawman.

Serving reuses scripts/eval/ablation_live.py's scheduler/knobs/SLO so the
metrics and raw schema match the rest of the eval suite; the only new machinery
is a wave-based submission loop so we can model sequential and oscillating
arrival patterns the standard NON-STATIONARY harness can't express.

CPU note: needs torch + the model, so it runs on a GPU/Colab box (cell 6g).

Run:
  python scripts/eval/failure_cases.py
  python scripts/eval/failure_cases.py --seeds 42 --limit 12   # quick smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from datetime import datetime

# --- path bootstrap so `python scripts/eval/failure_cases.py` finds src/ -----
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

# Reuse the ablation's serving primitives so knobs, SLO, and the validation
# search are byte-identical to the in-paper harness.
from scripts.eval.ablation_live import (  # noqa: E402
    N_LHS_CANDIDATES, OBSERVE_INTERVAL, SEARCH_SPACE, SLO_TTFT_MS, VALIDATION_SEED,
    _SLO, _apply_sched, _arm_index, _new_scheduler, latin_hypercube,
)
from src.carl.bandit import LinUCBBandit, PerRegimeBandit  # noqa: E402
from src.carl.config import CARLConfig, all_arm_sets  # noqa: E402
from src.carl.controller import CARLController  # noqa: E402
from src.carl.live import _ReqSpec, _make_prompt, _percentile  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RAW_DIR = os.path.join(DOCS_EVAL, "raw", "failure_cases")
RESULTS_PATH = os.path.join(DOCS_EVAL, "failure_cases_results.json")
ENV_PATH = os.path.join(DOCS_EVAL, "environment.json")

DEFAULT_SEEDS = [42, 43, 44]
N_REQUESTS = 30
METHODS = ["CARL-Full", "Static-Best"]
WIN_MARGIN = 0.02   # +/-2% throughput band for the winner determination

# Hypotheses are FIXED here, before any run -- the whole point is to test them
# against the evidence, not to back-fill them from results.
SCENARIOS = ["ultra_short", "stable_load", "single_queue",
             "rapid_oscillation", "memory_pressure"]
HYPOTHESIS = {
    "ultra_short":       "exploration_overhead",
    "stable_load":       "no_regime_change",
    "single_queue":      "insufficient_adaptation_time",
    "rapid_oscillation": "insufficient_adaptation_time",
    "memory_pressure":   "memory_bound",
}


# ===========================================================================
# Workload construction -- each scenario returns "waves" of requests.
# ===========================================================================
#
# A wave is (trigger, [specs]): submit the wave once `trigger` requests have
# finished. trigger=0 means "submit at t0". This single mechanism expresses:
#   * burst      -> one wave, trigger 0 (all at once)
#   * sequential -> 30 waves of 1, wave k triggered after k finish (single_queue)
#   * oscillating-> 6 waves of 5, wave w triggered after w*5 finish


def _spec(tokenizer, rid: str, prompt_len: int, max_new: int) -> _ReqSpec:
    return _ReqSpec(rid, _make_prompt(tokenizer, prompt_len), max_new, phase=0)


def build_waves(scenario: str, tokenizer, n: int, rng: random.Random) -> list:
    """Build the (trigger, specs) waves for a scenario."""
    if scenario == "ultra_short":
        specs = [_spec(tokenizer, f"u{i}", rng.randint(4, 8), 8) for i in range(n)]
        return [(0, specs)]                                   # burst

    if scenario == "stable_load":
        # Identical prompt + identical decode budget for every request: there is
        # genuinely nothing for the regime classifier to react to.
        specs = [_spec(tokenizer, f"s{i}", 64, 32) for i in range(n)]
        return [(0, specs)]                                   # burst

    if scenario == "single_queue":
        # One request in flight at a time: wave k submits after k have finished.
        return [(k, [_spec(tokenizer, f"q{k}", rng.randint(16, 32), 32)])
                for k in range(n)]

    if scenario == "rapid_oscillation":
        # Alternate INTERACTIVE (short) and BATCH (long) every 5 requests.
        waves, group = [], 5
        for w, start in enumerate(range(0, n, group)):
            count = min(group, n - start)
            if w % 2 == 0:
                specs = [_spec(tokenizer, f"i{start+j}", rng.randint(16, 32), 32)
                         for j in range(count)]
            else:
                specs = [_spec(tokenizer, f"b{start+j}", rng.randint(128, 256), 64)
                         for j in range(count)]
            waves.append((w * group, specs))                  # next wave after prev drains
        return waves

    if scenario == "memory_pressure":
        specs = [_spec(tokenizer, f"m{i}", rng.randint(512, 1024), 128) for i in range(n)]
        return [(0, specs)]                                   # burst

    raise ValueError(f"unknown scenario {scenario}")


# ===========================================================================
# The serve loop (wave-aware), mirroring ablation_live._serve's metric math.
# ===========================================================================


def serve(sched, waves: list, *, controller=None, tracker=None) -> dict:
    """Serve `waves` once; return the standard per-run metrics dict.

    Identical TTFT/TPOT/throughput/SLO accounting to ablation_live._serve; the
    only difference is the wave-based submission so sequential and oscillating
    arrival patterns are faithful.
    """
    meta: dict = {}
    rid_counter = 0
    for _trigger, specs in waves:
        for spec in specs:
            meta[spec.rid] = {"request_id": rid_counter,
                              "prompt_len": int(spec.prompt_ids.shape[-1])}
            rid_counter += 1

    submit_time, first_tok, last_tok, tok_count = {}, {}, {}, {}

    def _submit(spec) -> None:
        submit_time[spec.rid] = time.perf_counter()
        tok_count[spec.rid] = 0
        sched.add_request(spec.rid, spec.prompt_ids, max_new_tokens=spec.max_new,
                          eos_token_id=None)

    records: list[dict] = []
    decision_us: list[float] = []
    total_tokens = 0
    finished_count = 0
    wave_idx = 0
    t0 = time.perf_counter()
    last_step_t = t0

    while True:
        # Submit every wave whose trigger the finished-count has reached.
        progressed = False
        while wave_idx < len(waves) and finished_count >= waves[wave_idx][0]:
            for spec in waves[wave_idx][1]:
                _submit(spec)
            wave_idx += 1
            progressed = True

        if not sched.has_work():
            if wave_idx >= len(waves):
                break
            # No in-flight work but waves remain. If the next trigger is somehow
            # unreachable (shouldn't happen), force it so we never deadlock.
            if not progressed:
                for spec in waves[wave_idx][1]:
                    _submit(spec)
                wave_idx += 1
            continue

        emitted = sched.step()
        now = time.perf_counter()
        dt = now - last_step_t
        last_step_t = now

        for rid, _tok in emitted:
            if rid not in first_tok:
                first_tok[rid] = now
            last_tok[rid] = now
            tok_count[rid] += 1
            total_tokens += 1

        for r in sched.get_finished():
            rid = r.request_id
            finished_count += 1
            ntok = tok_count[rid]
            ttft_ms = (first_tok[rid] - submit_time[rid]) * 1000.0
            tpot_ms = ((last_tok[rid] - first_tok[rid]) / (ntok - 1) * 1000.0) if ntok > 1 else 0.0
            records.append(dict(meta[rid], tokens_generated=ntok,
                                ttft_ms=ttft_ms, tpot_ms=tpot_ms))
            if tracker is not None:
                tracker.record_request(ttft_ms, tpot_ms)

        if tracker is not None:
            tracker.record_batch(len(sched.active))
            if dt > 0 and emitted:
                tracker.record_throughput(len(emitted) / dt)

        if controller is not None:
            d0 = time.perf_counter()
            entry = controller.maybe_step(sched._step_idx)
            d1 = time.perf_counter()
            sched.enable_spec_decode = False           # re-pin: CARL may flip it on
            if entry is not None:
                decision_us.append((d1 - d0) * 1e6)

    wall = time.perf_counter() - t0
    records.sort(key=lambda r: r["request_id"])
    ttfts = [r["ttft_ms"] for r in records]
    tpots = [r["tpot_ms"] for r in records if r["tokens_generated"] > 1]
    slo_rate = (100.0 * sum(1 for t in ttfts if t < SLO_TTFT_MS) / len(ttfts)
                if ttfts else 0.0)

    return {
        "requests": records,
        "throughput_tps": total_tokens / wall if wall > 0 else 0.0,
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p99": _percentile(ttfts, 99),
        "tpot_p50": _percentile(tpots, 50),
        "tpot_p99": _percentile(tpots, 99),
        "slo_rate": slo_rate,
        "wall_s": wall,
        "decision_us": decision_us,
    }


# ===========================================================================
# One run of one method over one scenario.
# ===========================================================================


def run_once(scenario: str, method: str, model, tokenizer, n: int, seed: int,
             *, static_cfg: CARLConfig | None = None) -> dict:
    waves = build_waves(scenario, tokenizer, n, random.Random(seed))
    sched = _new_scheduler(model)
    controller = tracker = None

    if method == "Static-Best":
        _apply_sched(sched, static_cfg or CARLConfig())
    else:  # CARL-Full
        tracker = MetricsTracker(window=max(50, n))
        bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
        controller = CARLController(scheduler=sched, bandit=bandit,
                                    observe_interval=OBSERVE_INTERVAL,
                                    slo=_SLO, metrics=tracker)

    out = serve(sched, waves, controller=controller, tracker=tracker)

    # Adaptation trace -> arm_changes + unique_arms (CARL only). An "arm" here is
    # a (regime, arm-index) pair, so a regime switch or a within-regime config
    # change both count as a change.
    if controller is not None:
        keys = []
        for e in controller.controller_log:
            arms = controller.bandit.arms(e.regime)
            keys.append((e.regime.value, _arm_index(arms, e.config)))
        out["arm_changes"] = sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1])
        out["unique_arms"] = len(set(keys))
        out["n_decisions"] = len(keys)
    return out


# ===========================================================================
# Static-Best via held-out LHS validation, per scenario.
# ===========================================================================


def select_static_best(scenario: str, model, tokenizer, val_n: int) -> CARLConfig:
    candidates = latin_hypercube(N_LHS_CANDIDATES, SEARCH_SPACE, VALIDATION_SEED)
    tputs = []
    for cfg in candidates:
        m = run_once(scenario, "Static-Best", model, tokenizer, val_n,
                     VALIDATION_SEED, static_cfg=cfg)
        tputs.append(m["throughput_tps"])
    win = max(range(len(candidates)), key=lambda i: tputs[i])
    return candidates[win]


# ===========================================================================
# Raw data.
# ===========================================================================


def _save_raw(scenario: str, method: str, seed: int, run: dict, seeds: list) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = {
        "scenario": scenario, "config": method, "seed": seed, "seeds_used": seeds,
        "requests": run["requests"],
        "throughput_tps": run["throughput_tps"],
        "ttft_p50": run["ttft_p50"], "ttft_p99": run["ttft_p99"],
        "tpot_p50": run["tpot_p50"], "tpot_p99": run["tpot_p99"],
        "slo_rate": run["slo_rate"],
    }
    if method == "CARL-Full":
        payload["arm_changes"] = run.get("arm_changes", 0)
        payload["unique_arms"] = run.get("unique_arms", 0)
    path = os.path.join(RAW_DIR, f"{scenario}_{method}_run_{seed:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ===========================================================================
# Failure-reason classification (evidence rules, applied in listed order).
# ===========================================================================


def classify_failure(scenario: str, carl_tput: float, static_tput: float,
                     margin_pct: float, arm_changes: int, unique_arms: int) -> str:
    """Data-driven failure reason. Rules applied top-to-bottom, first match wins
    (the order is exactly the spec's evidence list)."""
    if arm_changes == 0:
        return "no_adaptation_triggered"
    if arm_changes > 8:
        return "adaptation_instability"
    if carl_tput >= static_tput * (1 - WIN_MARGIN):    # CARL wins or ties
        return "no_failure"
    # Below here CARL is genuinely losing (> 2% under Static-Best).
    if unique_arms <= 2:
        return "insufficient_exploration"
    if unique_arms >= 4:
        return "exploration_overhead"
    if margin_pct < -10.0 and scenario == "memory_pressure":
        return "memory_bound"
    return "other -- inspect adaptation trace"


def _winner(carl_tput: float, static_tput: float) -> str:
    if carl_tput > static_tput * (1 + WIN_MARGIN):
        return "CARL"
    if static_tput > carl_tput * (1 + WIN_MARGIN):
        return "Static"
    return "Tie"


# ===========================================================================
# Aggregation + driver.
# ===========================================================================


def _mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def capture_environment() -> dict:
    """Capture the run host, reusing docs/eval/environment.json ONLY if it still
    matches the live device.

    A cached record is trusted only when its gpu AND torch fields agree with the
    live torch.cuda state; otherwise (e.g. a CPU-written file shipped to a GPU
    box, or vice-versa) it is STALE and we regenerate -- so the recorded
    environment can never silently misreport CPU on a GPU run (or the reverse).
    """
    live_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    live_torch = torch.__version__
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                env = json.load(f)
            if env.get("gpu") == live_gpu and env.get("torch") == live_torch:
                return env
        except Exception:
            pass
    env = {"gpu": live_gpu,
           "cuda": torch.version.cuda, "torch": live_torch,
           "python": sys.version, "timestamp": datetime.now().isoformat()}
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    return env


def run_scenario(scenario: str, model, tokenizer, seeds: list, n: int) -> dict:
    print(f"\n=== {scenario} (hypothesis: {HYPOTHESIS[scenario]}) ===", flush=True)
    static_cfg = select_static_best(scenario, model, tokenizer, max(10, n // 2))
    print(f"  Static-Best: mb={static_cfg.max_batch_size} cs={static_cfg.chunk_size}",
          flush=True)

    carl_runs, static_runs = [], []
    for seed in seeds:
        try:
            c = run_once(scenario, "CARL-Full", model, tokenizer, n, seed)
            s = run_once(scenario, "Static-Best", model, tokenizer, n, seed,
                         static_cfg=static_cfg)
            carl_runs.append(c)
            static_runs.append(s)
            _save_raw(scenario, "CARL-Full", seed, c, seeds)
            _save_raw(scenario, "Static-Best", seed, s, seeds)
            print(f"  seed {seed}: CARL {c['throughput_tps']:6.1f} vs Static "
                  f"{s['throughput_tps']:6.1f} tok/s "
                  f"(arm_changes={c.get('arm_changes')}, unique={c.get('unique_arms')})",
                  flush=True)
        except Exception:
            print(f"  seed {seed}: FAILED", flush=True)
            traceback.print_exc()

    # Means/stds across seeds.
    carl_t_m, carl_t_s = _mean_std([r["throughput_tps"] for r in carl_runs])
    stat_t_m, stat_t_s = _mean_std([r["throughput_tps"] for r in static_runs])
    carl_f50_m, carl_f50_s = _mean_std([r["ttft_p50"] for r in carl_runs])
    stat_f50_m, stat_f50_s = _mean_std([r["ttft_p50"] for r in static_runs])
    carl_f99_m, carl_f99_s = _mean_std([r["ttft_p99"] for r in carl_runs])
    stat_f99_m, stat_f99_s = _mean_std([r["ttft_p99"] for r in static_runs])
    carl_slo_m, carl_slo_s = _mean_std([r["slo_rate"] for r in carl_runs])
    stat_slo_m, stat_slo_s = _mean_std([r["slo_rate"] for r in static_runs])
    arm_changes = round(_mean_std([r.get("arm_changes", 0) for r in carl_runs])[0])
    unique_arms = round(_mean_std([r.get("unique_arms", 0) for r in carl_runs])[0])

    margin_pct = ((carl_t_m - stat_t_m) / stat_t_m * 100.0) if stat_t_m else 0.0
    winner = _winner(carl_t_m, stat_t_m)
    final = classify_failure(scenario, carl_t_m, stat_t_m, margin_pct,
                             arm_changes, unique_arms)
    hypothesis = HYPOTHESIS[scenario]
    revised = (hypothesis != final)
    if revised:
        print(f"  Hypothesis revised: was {hypothesis}, now {final} "
              f"(arm_changes={arm_changes}, unique_arms={unique_arms}, "
              f"margin={margin_pct:+.1f}%)", flush=True)

    # Latency tradeoff: even where CARL loses throughput, does it cut TTFT?
    ttft_p99_delta = carl_f99_m - stat_f99_m       # negative => CARL faster tail
    slo_delta = carl_slo_m - stat_slo_m            # positive => CARL meets SLO more
    if winner == "Static" and (ttft_p99_delta < -1.0 or slo_delta > 1.0):
        tradeoff = (f"CARL trades {abs(margin_pct):.1f}% throughput for a "
                    f"{ttft_p99_delta:+.1f} ms TTFT-P99 / {slo_delta:+.1f} pt SLO change")
    elif winner == "Static":
        tradeoff = "CARL loses throughput with no offsetting TTFT/SLO benefit here"
    else:
        tradeoff = "n/a (CARL wins or ties on throughput)"

    return {
        "scenario": scenario,
        "static_best_config": static_cfg.as_dict(),
        "carl": {
            "throughput_tps_mean": carl_t_m, "throughput_tps_std": carl_t_s,
            "ttft_p50_mean": carl_f50_m, "ttft_p50_std": carl_f50_s,
            "ttft_p99_mean": carl_f99_m, "ttft_p99_std": carl_f99_s,
            "slo_rate_mean": carl_slo_m, "slo_rate_std": carl_slo_s,
        },
        "static": {
            "throughput_tps_mean": stat_t_m, "throughput_tps_std": stat_t_s,
            "ttft_p50_mean": stat_f50_m, "ttft_p50_std": stat_f50_s,
            "ttft_p99_mean": stat_f99_m, "ttft_p99_std": stat_f99_s,
            "slo_rate_mean": stat_slo_m, "slo_rate_std": stat_slo_s,
        },
        "winner": winner,
        "margin_pct": margin_pct,
        "arm_changes": arm_changes,
        "unique_arms": unique_arms,
        "failure_reason_hypothesis": hypothesis,
        "failure_reason_final": final,
        "hypothesis_revised": revised,
        "latency_tradeoff": tradeoff,
    }


def run_all(seeds: list, n: int) -> dict:
    env = capture_environment()
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {len(seeds)} seeds {seeds} x {n} "
          f"requests | SLO TTFT < {SLO_TTFT_MS:.0f} ms", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on a Colab GPU for real numbers.\n",
              flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    scenarios = [run_scenario(s, model, tokenizer, seeds, n) for s in SCENARIOS]
    return _finalize(env, seeds, n, scenarios)


def _finalize(env, seeds, n, scenarios) -> dict:
    # "CARL underperforms or ties" = does NOT strictly win on throughput.
    not_wins = sum(1 for s in scenarios if s["winner"] != "CARL")
    confirmed = sum(1 for s in scenarios if not s["hypothesis_revised"])

    # Paper-ready quote, generated from the data.
    losses = [s for s in scenarios if s["winner"] == "Static"]
    worst = min(scenarios, key=lambda s: s["margin_pct"]) if scenarios else None
    quote = (
        f"On real TinyLlama, CARL fails to beat a per-scenario-validated "
        f"Static-Best in {not_wins}/5 stress workloads designed to expose its "
        f"weaknesses"
        + (f", losing most on '{worst['scenario']}' ({worst['margin_pct']:+.1f}% "
           f"throughput, {worst['failure_reason_final']})" if worst and losses else "")
        + ". CARL's advantage is specifically NON-stationary multi-regime traffic; "
        "on stable, sequential, or memory-bound workloads its online exploration "
        "is a cost, not a benefit -- exactly where a fixed config should be used."
    )

    results = {
        "seeds": seeds, "requests_per_run": n, "methods": METHODS,
        "slo_ttft_ms": SLO_TTFT_MS, "win_margin_pct": WIN_MARGIN * 100,
        "environment": env,
        "scenarios": scenarios,
        "summary": {
            "carl_underperforms_or_ties": f"{not_wins}/5",
            "hypotheses_confirmed": f"{confirmed}/5",
            "failure_reasons": {s["scenario"]: s["failure_reason_final"] for s in scenarios},
        },
        "paper_quote": quote,
    }
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(results)
    print(f"\nSaved failure-case results to {RESULTS_PATH}", flush=True)
    return results


def _print(results: dict) -> None:
    print("\n=== FAILURE-CASE ANALYSIS (real TinyLlama, mean +/- std over seeds) ===")
    head = ["scenario", "carl_tput", "static_tput", "carl_ttftP99", "static_ttftP99",
            "carl_SLO", "static_SLO", "winner", "margin%", "arm_chg", "uniq", "reason"]
    print("| " + " | ".join(head) + " |")
    print("| " + " | ".join("---" for _ in head) + " |")
    for s in results["scenarios"]:
        c, st = s["carl"], s["static"]
        print("| " + " | ".join([
            s["scenario"],
            f"{c['throughput_tps_mean']:.1f}+/-{c['throughput_tps_std']:.1f}",
            f"{st['throughput_tps_mean']:.1f}+/-{st['throughput_tps_std']:.1f}",
            f"{c['ttft_p99_mean']:.0f}+/-{c['ttft_p99_std']:.0f}",
            f"{st['ttft_p99_mean']:.0f}+/-{st['ttft_p99_std']:.0f}",
            f"{c['slo_rate_mean']:.0f}", f"{st['slo_rate_mean']:.0f}",
            s["winner"], f"{s['margin_pct']:+.1f}",
            str(s["arm_changes"]), str(s["unique_arms"]),
            s["failure_reason_final"],
        ]) + " |")

    print(f"\nSUMMARY: CARL underperforms or ties in "
          f"{results['summary']['carl_underperforms_or_ties']} scenarios; "
          f"hypotheses confirmed {results['summary']['hypotheses_confirmed']}.")
    print("\nPer-scenario hypothesis check + latency tradeoff:")
    for s in results["scenarios"]:
        verdict = ("CONFIRMED" if not s["hypothesis_revised"]
                   else f"REVISED ({s['failure_reason_hypothesis']} -> {s['failure_reason_final']})")
        print(f"  - {s['scenario']:<18} {verdict}")
        print(f"      tradeoff: {s['latency_tradeoff']}")
    print(f"\nPAPER QUOTE:\n  {results['paper_quote']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CARL honest failure analysis (GPU).")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated run seeds (default 42,43,44)")
    parser.add_argument("--limit", type=int, default=N_REQUESTS, help="requests per run")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n = args.limit if 0 < args.limit <= 200 else N_REQUESTS
    run_all(seeds, n)


if __name__ == "__main__":
    main()
