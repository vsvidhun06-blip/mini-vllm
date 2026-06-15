"""
CARL ablation on REAL TinyLlama inference (GPU) -- the measured counterpart to
the simulation in scripts/eval/ablation.py.

WHAT THIS IS
------------
This reuses src/carl/live.py's harness pattern: real TinyLlama weights, the real
ContinuousBatchScheduler, real prefill/decode forward passes, with CARL driving
the scheduler's knobs LIVE while requests are served. It runs eight
configurations over the NON-STATIONARY scenario (25 INTERACTIVE then 25 BATCH
requests injected mid-run, no notice) and reports wall-clock throughput / TTFT /
TPOT, mean +/- std over N runs. Results go to docs/eval/ablation_live_results.json.

HOW THE ABLATIONS ARE APPLIED (no src/ changes)
-----------------------------------------------
Each "CARL-NoX" config freezes one knob group at the global default by building
the per-regime bandit from FROZEN arm sets: every arm has the frozen knob pinned,
so the controller can never vary it. CARL-Full uses the unmodified arms. Oracle
switches the scheduler to DEFAULT_CONFIGS[regime] exactly at the regime boundary
(perfect knowledge); Static-Best applies one fixed config (the best of the
candidate set) for the whole run.

!!! HONEST LIMITATION -- READ BEFORE INTERPRETING THE TABLE !!!
--------------------------------------------------------------
In this SINGLE-MODEL live harness (exactly like live.py) the controller is wired
to the SCHEDULER ONLY, so only the scheduling knobs (max_batch_size, chunk_size)
actually change what the GPU does. Consequently:

  * CARL-NoSched, CARL-NoChunk  -> freeze scheduling knobs -> REAL, measurable
    effect vs CARL-Full. These are the live-effective ablations.
  * CARL-NoSpec    -> speculation is pinned OFF here (TinyLlama self-spec is below
    break-even; see live.py), so freezing spec_k=0 changes nothing -> ~= CARL-Full.
  * CARL-NoCache   -> the KV cache / H2O eviction is not wired to the controller
    and never triggers at these sizes -> ~= CARL-Full.
  * CARL-NoRouter  -> there is no router (one model is served) -> ~= CARL-Full.

So rows that come out statistically identical to CARL-Full are EXPECTED: this
table measures which subsystems matter for real single-GPU TinyLlama serving
(answer: the scheduler), and is the honest hardware complement to the simulation
ablation, which can vary all five subsystems. See docs/eval/README.md.

Run:
  python scripts/eval/ablation_live.py                 # 3 runs x 50 requests
  python scripts/eval/ablation_live.py --runs 2 --limit 30   # quicker
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import replace

# --- path bootstrap so `python scripts/eval/ablation_live.py` finds src/ -----
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from src.carl.bandit import LinUCBBandit, PerRegimeBandit  # noqa: E402
from src.carl.config import CARLConfig, DEFAULT_CONFIGS, all_arm_sets  # noqa: E402
from src.carl.controller import SLO, CARLController  # noqa: E402
from src.carl.state import FEATURE_DIM, MetricsTracker, WorkloadRegime  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
from src.engine.scheduler import ContinuousBatchScheduler  # noqa: E402

# Reuse live.py's prompt/workload/percentile helpers + pool sizing, so this
# harness is byte-for-byte the same serving setup as the real-inference cell 6c.
from src.carl.live import (  # noqa: E402
    BLOCK_SIZE, NUM_BLOCKS, _build_workload, _make_prompt, _percentile,
)

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
RESULTS_PATH = os.path.join(DOCS_EVAL, "ablation_live_results.json")

# TTFT-only SLO targets mirror the rest of the eval suite (TTFT < 200ms).
_SLO = SLO(ttft_ms=200.0, tpot_ms=50.0, throughput_ref=50.0)

# Knob group each ablation freezes at the global default (None = CARL-Full).
_DEF = CARLConfig()
FREEZE = {
    "CARL-Full": None,
    "CARL-NoSched": dict(max_batch_size=_DEF.max_batch_size, chunk_size=_DEF.chunk_size),
    "CARL-NoSpec": dict(spec_k=0),
    "CARL-NoCache": dict(eviction_threshold=0.8),
    "CARL-NoRouter": dict(routing_threshold=0.5),
    "CARL-NoChunk": dict(chunk_size=256),
}
# Configs that actually change what the GPU does in this single-model harness.
_LIVE_EFFECTIVE = {"CARL-Full", "CARL-NoSched", "CARL-NoChunk", "Static-Best", "Oracle"}

CONFIGS = list(FREEZE.keys()) + ["Static-Best", "Oracle"]


def _frozen_arms(freeze: dict | None) -> dict:
    """all_arm_sets() with `freeze` applied (and clamped) to every arm."""
    base = all_arm_sets()
    if not freeze:
        return base
    return {r: [replace(a, **freeze).clamp() for a in arms] for r, arms in base.items()}


def _apply_sched(sched, cfg: CARLConfig) -> None:
    """Push a config's SCHEDULING knobs into the live scheduler (spec stays off).

    Only the knobs the scheduler reads matter here; speculation is force-pinned
    off exactly as in live.py, so a config's spec_k never turns it on.
    """
    sched.max_batch_size = int(cfg.max_batch_size)
    sched.chunk_size = int(cfg.chunk_size)
    sched.enable_spec_decode = False


def _serve(sched, specs, *, controller=None, tracker=None, oracle_phase1=None) -> dict:
    """Serve `specs` through `sched` once; return throughput + TTFT/TPOT metrics.

    Mirrors live.py's _run_config loop. `controller` (if given) runs one CARL
    cycle every 10 steps and we re-pin speculation off after it. `oracle_phase1`
    (if given) is the config applied the instant phase-1 (BATCH) requests are
    injected -- the perfect-knowledge regime switch.
    """
    phase0 = [s for s in specs if s.phase == 0]
    phase1 = [s for s in specs if s.phase == 1]
    submit_time: dict[str, float] = {}
    first_tok: dict[str, float] = {}
    last_tok: dict[str, float] = {}
    tok_count: dict[str, int] = {}

    def _submit(spec) -> None:
        submit_time[spec.rid] = time.perf_counter()
        tok_count[spec.rid] = 0
        # eos_token_id=None -> every request emits exactly max_new tokens, so all
        # configs generate the same total and throughput is comparable.
        sched.add_request(spec.rid, spec.prompt_ids, max_new_tokens=spec.max_new,
                          eos_token_id=None)

    t0 = time.perf_counter()
    for spec in phase0:
        _submit(spec)

    ttft_list: list[float] = []
    tpot_list: list[float] = []
    total_tokens = 0
    finished_count = 0
    phase1_done = (len(phase1) == 0)
    last_step_t = time.perf_counter()

    while sched.has_work():
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
            ttft_ms = (first_tok[rid] - submit_time[rid]) * 1000.0
            n = tok_count[rid]
            tpot_ms = ((last_tok[rid] - first_tok[rid]) / (n - 1) * 1000.0) if n > 1 else 0.0
            ttft_list.append(ttft_ms)
            if n > 1:
                tpot_list.append(tpot_ms)
            if tracker is not None:
                tracker.record_request(ttft_ms, tpot_ms)

        if tracker is not None:
            tracker.record_batch(len(sched.active))
            if dt > 0 and emitted:
                tracker.record_throughput(len(emitted) / dt)

        # Inject the BATCH phase once half of phase 0 has finished (the surprise
        # regime flip); the Oracle switches its config at exactly this moment.
        if not phase1_done and finished_count >= max(1, len(phase0) // 2):
            for spec in phase1:
                _submit(spec)
            phase1_done = True
            if oracle_phase1 is not None:
                _apply_sched(sched, oracle_phase1)

        if controller is not None:
            controller.maybe_step(sched._step_idx)
            sched.enable_spec_decode = False   # re-pin: CARL may have flipped it on

    wall = time.perf_counter() - t0
    throughput = total_tokens / wall if wall > 0 else 0.0
    return {
        "throughput_tok_s": throughput,
        "ttft_p50_ms": _percentile(ttft_list, 50),
        "ttft_p99_ms": _percentile(ttft_list, 99),
        "tpot_p50_ms": _percentile(tpot_list, 50),
        "tpot_p99_ms": _percentile(tpot_list, 99),
        "total_tokens": total_tokens,
        "wall_s": wall,
    }


def _new_scheduler(model) -> ContinuousBatchScheduler:
    """A scheduler at the global defaults, speculation off (live.py settings)."""
    return ContinuousBatchScheduler(
        model, max_batch_size=_DEF.max_batch_size, num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE, chunk_size=_DEF.chunk_size, enable_spec_decode=False,
    )


def run_config(name: str, model, tokenizer, n: int, seed: int,
               static_cfg: CARLConfig | None = None) -> dict:
    """Serve one configuration once over a fresh NON-STATIONARY workload."""
    import random

    specs = _build_workload(tokenizer, "NON-STATIONARY", n, random.Random(seed))
    sched = _new_scheduler(model)
    controller = tracker = oracle_phase1 = None

    if name == "Static-Best":
        _apply_sched(sched, static_cfg or CARLConfig())
    elif name == "Oracle":
        # Perfect knowledge: INTERACTIVE config for phase 0, BATCH at the flip.
        _apply_sched(sched, DEFAULT_CONFIGS[WorkloadRegime.INTERACTIVE])
        oracle_phase1 = DEFAULT_CONFIGS[WorkloadRegime.BATCH]
    else:  # CARL-Full or a CARL-NoX ablation: a bandit over (frozen) arm sets.
        tracker = MetricsTracker(window=max(50, n))
        bandit = PerRegimeBandit(_frozen_arms(FREEZE[name]), d=FEATURE_DIM,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
        controller = CARLController(scheduler=sched, bandit=bandit,
                                    observe_interval=10, slo=_SLO, metrics=tracker)

    return _serve(sched, specs, controller=controller, tracker=tracker,
                  oracle_phase1=oracle_phase1)


def _resolve_static_best(model, tokenizer, n: int, seed: int) -> tuple:
    """Pick the single best static config for the NON-STATIONARY workload.

    Probes the natural candidates (global default + the INTERACTIVE/BATCH
    hand-tuned defaults) once each and keeps the highest-throughput one.
    """
    candidates = {
        "default": CARLConfig(),
        "interactive": DEFAULT_CONFIGS[WorkloadRegime.INTERACTIVE],
        "batch": DEFAULT_CONFIGS[WorkloadRegime.BATCH],
    }
    best_name, best_cfg, best_tps = None, None, float("-inf")
    for cname, cfg in candidates.items():
        m = run_config("Static-Best", model, tokenizer, n, seed, static_cfg=cfg)
        if m["throughput_tok_s"] > best_tps:
            best_name, best_cfg, best_tps = cname, cfg, m["throughput_tok_s"]
    print(f"Static-Best resolved to '{best_name}' "
          f"(mb={best_cfg.max_batch_size}, cs={best_cfg.chunk_size}, "
          f"{best_tps:.1f} tok/s)")
    return best_cfg, best_name


def _mean_std(vals: list[float]) -> tuple:
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


_METRICS = [
    ("throughput_tok_s", "tok/s"),
    ("ttft_p50_ms", "ttftP50"),
    ("ttft_p99_ms", "ttftP99"),
    ("tpot_p50_ms", "tpotP50"),
    ("tpot_p99_ms", "tpotP99"),
]


def run_ablation_live(n: int = 50, runs: int = 3, seed0: int = 0) -> dict:
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | {runs} runs x {n} requests", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA device -- running on CPU. Numbers are a smoke test "
              "only; run on a Colab T4 GPU for representative results.\n", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    static_best_cfg, static_best_name = _resolve_static_best(model, tokenizer, n, seed0)

    results: dict = {
        "device": str(DEVICE), "dtype": str(dtype), "model": MODEL_NAME,
        "scenario": "NON-STATIONARY (25 INTERACTIVE -> 25 BATCH)",
        "runs": runs, "requests": n, "seeds": list(range(seed0, seed0 + runs)),
        "slo_ttft_ms": _SLO.ttft_ms, "static_best": static_best_name,
        "live_effective_configs": sorted(_LIVE_EFFECTIVE),
        "note": ("Single-model live harness wires CARL to the SCHEDULER only, "
                 "speculation is pinned off, there is no router, and KV eviction "
                 "does not trigger at these sizes. So only NoSched/NoChunk (and "
                 "Static-Best/Oracle) differ from CARL-Full; NoSpec/NoCache/"
                 "NoRouter are expected to match CARL-Full. See docs/eval/README.md."),
        "configs": {},
    }

    for name in CONFIGS:
        per_run = []
        for r in range(runs):
            seed = seed0 + r
            try:
                cfg = static_best_cfg if name == "Static-Best" else None
                m = run_config(name, model, tokenizer, n, seed, static_cfg=cfg)
                per_run.append(m)
                print(f"  {name:<14} run {r+1}/{runs}: "
                      f"{m['throughput_tok_s']:6.1f} tok/s, "
                      f"ttftP99={m['ttft_p99_ms']:6.1f}ms", flush=True)
            except Exception:
                # Never fail silently (matches the live.py hardening): print the
                # traceback and keep going so other configs still produce a table.
                print(f"  {name:<14} run {r+1}/{runs}: FAILED", flush=True)
                traceback.print_exc()
        if not per_run:
            continue
        agg = {}
        for key, _label in _METRICS:
            mean, std = _mean_std([m[key] for m in per_run])
            agg[f"{key}_mean"], agg[f"{key}_std"] = mean, std
        results["configs"][name] = agg

    _print(results)
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved live ablation results to {RESULTS_PATH}", flush=True)
    return results


def _print(results: dict) -> None:
    headers = ["config", "live?"] + [label for _k, label in _METRICS]
    print("\n=== LIVE ABLATION: NON-STATIONARY on real TinyLlama (mean +/- std) ===")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for name in CONFIGS:
        agg = results["configs"].get(name)
        if agg is None:
            continue
        live = "yes" if name in _LIVE_EFFECTIVE else "no*"
        cells = [name, live]
        for key, _label in _METRICS:
            cells.append(f"{agg[f'{key}_mean']:.1f} +/- {agg[f'{key}_std']:.1f}")
        print("| " + " | ".join(cells) + " |")
    print("\n* 'no' = config has no live effect in this single-model harness "
          "(spec pinned off / no router / eviction inactive); expected to match "
          "CARL-Full. See the module docstring and docs/eval/README.md.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL ablation on real TinyLlama inference (GPU).")
    parser.add_argument("--runs", type=int, default=3, help="runs per config")
    parser.add_argument("--limit", type=int, default=50, help="requests per run")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    args = parser.parse_args()
    # Guard against an accidental huge real-inference run from a stray --limit.
    n = args.limit if 0 < args.limit <= 200 else 50
    run_ablation_live(n=n, runs=args.runs, seed0=args.seed)


if __name__ == "__main__":
    main()
