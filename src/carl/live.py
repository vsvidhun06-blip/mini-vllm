"""
CARL LIVE harness -- real TinyLlama inference, CARL adaptive vs a fixed baseline.

WHAT THIS IS (and how it differs from scripts/benchmark_carl.py)
----------------------------------------------------------------
scripts/benchmark_carl.py is a CONTROL-LOOP SIMULATION: it drives the real CARL
controller/bandits over an *analytical* serving cost model, with no GPU in the
loop. That isolates the control policy but the throughput/latency numbers are
illustrative, not measured.

THIS module is the opposite end: it runs the REAL serving engine -- TinyLlama
weights, the real ContinuousBatchScheduler, real prefill/decode forward passes --
and lets CARL drive the scheduler's knobs LIVE while requests are being served.
The throughput/TTFT/TPOT it reports are wall-clock measurements off the actual
model, so it needs a GPU to be meaningful (it will still *run* on CPU, just
slowly). It is imported lazily by `benchmark_carl.py --live` so the default
(torch-free) simulation path keeps working on a box without torch.

WHAT CARL ACTUALLY CONTROLS HERE
--------------------------------
In this harness CARL is wired to the scheduler only (router / kv_cache / spec
decoder are not passed), so the knobs that actually move are the SCHEDULING ones
the scheduler reads live every step:

    * max_batch_size  -- rows per forward pass (compute budget)
    * chunk_size      -- chunked-prefill token budget per step

Those are exactly the levers that separate an interactive (small batch, low
latency) operating point from a batch/throughput (large batch) one, so they are
the honest thing to measure for "does online adaptation beat a fixed config".

SPECULATIVE DECODING IS FORCED OFF -- on purpose.
CARL's DEFAULT_CONFIGS enable speculation (spec_k>0) for several regimes, but
TinyLlama's self-speculation acceptance is ~1% (below the early-exit break-even
documented in src/engine/spec_decode.py and the server defaults), so turning it
on would make BOTH the controller and any spec-enabled config look *slower* for
reasons that have nothing to do with the scheduling adaptation under test. We
therefore pin `enable_spec_decode = False` on the scheduler after every control
cycle, so the baseline-vs-CARL comparison is purely about batch/chunk scheduling.
Both arms run with speculation off -- it is an apples-to-apples control.

METRICS
-------
Per request we timestamp:
    submit -> first token  ........  TTFT (time to first token)
    first token -> last token /(n-1)  TPOT (per-output-token time, decode pace)
and across the whole run: total generated tokens / wall time = throughput.
Requests are submitted with eos_token_id=None so every request emits EXACTLY its
max_new_tokens budget -- both arms generate the same number of tokens, so a
throughput difference reflects scheduling efficiency, not different stopping.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.carl.bandit import LinUCBBandit, PerRegimeBandit
from src.carl.config import CARLConfig, all_arm_sets
from src.carl.controller import SLO, CARLController
from src.carl.state import FEATURE_DIM, MetricsTracker
from src.engine.device import DEVICE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler

DOCS = Path(__file__).resolve().parent.parent.parent / "docs"
# Deliberately a SEPARATE file from docs/carl_results.json (which holds the
# control-loop simulation the paper/figures reference). Overwriting that would
# destroy the simulation artifact, so the live numbers get their own file.
LIVE_RESULTS_PATH = DOCS / "carl_live_results.json"

# KV pool sizing: 1024 blocks x 16 tokens = 16384 tokens of capacity. Comfortably
# holds the worst case here (max_batch_size up to 32, each request <= ~256 prompt
# + 64 generated ~= 20 blocks; 32 * 20 = 640 < 1024).
NUM_BLOCKS = 1024
BLOCK_SIZE = 16

# Per-scenario prompt-length window (in tokens) and decode budget. The lengths
# are what make classify_regime read the intended regime off the LIVE scheduler:
# short prompts + a shallow-ish queue -> INTERACTIVE; >=256-token prompts -> BATCH.
_SCENARIO_SPEC = {
    "INTERACTIVE":    dict(prompt_lo=16,  prompt_hi=32,  max_new=32),
    "BATCH":          dict(prompt_lo=128, prompt_hi=256, max_new=64),
    # NON-STATIONARY is built from the two above; see _build_workload.
}


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 on an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _make_prompt(tokenizer, length: int) -> torch.Tensor:
    """A length-`length` (1, length) prompt, built by tiling a base sentence.

    Same construction as scripts/benchmark_chunked_prefill.py: real tokens (so the
    forward pass is representative) but content-free, so nothing depends on prompt
    semantics. Returned on CPU; the scheduler moves it to the model's device at
    add_request time.
    """
    base = tokenizer("The quick brown fox jumps over the lazy dog. ",
                     return_tensors="pt")["input_ids"][0]
    reps = (length + len(base) - 1) // len(base)
    return base.repeat(reps)[:length].unsqueeze(0)


@dataclass
class _ReqSpec:
    """One request to serve: its prompt, decode budget, and submission phase.

    phase 0 requests are submitted at t0; phase 1 requests are submitted partway
    through the run (used only by NON-STATIONARY to inject a regime shift with no
    notice to the controller).
    """
    rid: str
    prompt_ids: torch.Tensor
    max_new: int
    phase: int


def _build_workload(tokenizer, scenario: str, n: int, rng) -> list[_ReqSpec]:
    """Build the `n` request specs for a scenario.

    INTERACTIVE / BATCH: all phase 0, lengths drawn from the scenario's window.
    NON-STATIONARY: first half INTERACTIVE (phase 0), second half BATCH (phase 1),
    so the workload's regime flips mid-run without warning -- the exact thing an
    online controller is supposed to track.
    """
    specs: list[_ReqSpec] = []
    if scenario == "NON-STATIONARY":
        half = n // 2
        inter = _SCENARIO_SPEC["INTERACTIVE"]
        batch = _SCENARIO_SPEC["BATCH"]
        for i in range(half):
            L = rng.randint(inter["prompt_lo"], inter["prompt_hi"])
            specs.append(_ReqSpec(f"i{i}", _make_prompt(tokenizer, L), inter["max_new"], 0))
        for i in range(n - half):
            L = rng.randint(batch["prompt_lo"], batch["prompt_hi"])
            specs.append(_ReqSpec(f"b{i}", _make_prompt(tokenizer, L), batch["max_new"], 1))
        return specs

    spec = _SCENARIO_SPEC[scenario]
    for i in range(n):
        L = rng.randint(spec["prompt_lo"], spec["prompt_hi"])
        specs.append(_ReqSpec(f"{scenario[:3].lower()}{i}", _make_prompt(tokenizer, L),
                              spec["max_new"], 0))
    return specs


def _run_config(model, tokenizer, specs: list[_ReqSpec], use_carl: bool, seed: int) -> dict:
    """Serve `specs` once, either with CARL adapting or with a fixed baseline.

    Returns a metrics dict (throughput + TTFT/TPOT percentiles). Both arms start
    from the same scheduler defaults (max_batch_size / chunk_size = CARLConfig
    defaults) and run with speculation OFF; the only difference is whether a
    CARLController is mutating max_batch_size / chunk_size live every 10 steps.
    """
    default = CARLConfig()
    sched = ContinuousBatchScheduler(
        model,
        max_batch_size=default.max_batch_size,   # 8
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        chunk_size=default.chunk_size,            # 256
        enable_spec_decode=False,                 # forced off (see module docstring)
    )

    # CARL: a per-regime LinUCB controller wired to the scheduler only. Its metric
    # windows are fed below as requests complete, so its reward and regime
    # classification reflect the REAL run, not a synthetic state.
    controller = None
    tracker = None
    if use_carl:
        tracker = MetricsTracker(window=max(50, len(specs)))
        bandit = PerRegimeBandit(all_arm_sets(), d=FEATURE_DIM,
                                 bandit_cls=LinUCBBandit, alpha=0.5)
        controller = CARLController(
            scheduler=sched, bandit=bandit, observe_interval=10,
            slo=SLO(ttft_ms=100.0, tpot_ms=50.0, throughput_ref=50.0),
            metrics=tracker,
        )

    # Per-request bookkeeping for the latency math.
    phase0 = [s for s in specs if s.phase == 0]
    phase1 = [s for s in specs if s.phase == 1]
    submit_time: dict[str, float] = {}
    first_tok: dict[str, float] = {}
    last_tok: dict[str, float] = {}
    tok_count: dict[str, int] = {}

    def _submit(spec: _ReqSpec) -> None:
        submit_time[spec.rid] = time.perf_counter()
        tok_count[spec.rid] = 0
        # eos_token_id=None -> generate exactly max_new tokens (fixed token budget
        # so both arms emit the same total and throughput is comparable).
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

        # Harvest just-finished requests into the latency lists (and feed CARL).
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

        # Feed CARL's live signals: current batch occupancy and the step's
        # instantaneous token rate, so observe()/reward see the real engine.
        if tracker is not None:
            tracker.record_batch(len(sched.active))
            if dt > 0 and emitted:
                tracker.record_throughput(len(emitted) / dt)

        # NON-STATIONARY: inject phase 1 once half of phase 0 has finished, so the
        # regime flips mid-stream and the controller has to notice.
        if not phase1_done and finished_count >= max(1, len(phase0) // 2):
            for spec in phase1:
                _submit(spec)
            phase1_done = True

        # Run one CARL control cycle if we're on an observe_interval boundary, then
        # re-pin speculation off (CARL's chosen config may have flipped it on).
        if controller is not None:
            controller.maybe_step(sched._step_idx)
            sched.enable_spec_decode = False

    wall = time.perf_counter() - t0
    throughput = total_tokens / wall if wall > 0 else 0.0

    out = {
        "throughput_tok_s": throughput,
        "ttft_p50_ms": _percentile(ttft_list, 50),
        "ttft_p99_ms": _percentile(ttft_list, 99),
        "tpot_p50_ms": _percentile(tpot_list, 50),
        "tpot_p99_ms": _percentile(tpot_list, 99),
        "total_tokens": total_tokens,
        "wall_s": wall,
        "requests": len(specs),
    }
    if controller is not None:
        # The policy CARL converged on (most-selected arm per visited regime) and
        # how many times it actually changed the applied config -- evidence it
        # adapted rather than sat still.
        stats = controller.stats()
        out["adaptations"] = stats["total_adaptations"]
        out["regime_distribution"] = stats["regime_distribution"]
    return out


# Columns for the printed pipe-table (parsed by the notebook's to_md_table).
_COLUMNS = [
    ("scenario", "scenario", "{:>14}"),
    ("config", "config", "{:>14}"),
    ("throughput_tok_s", "tok/s", "{:>8.1f}"),
    ("ttft_p50_ms", "ttftP50", "{:>8.1f}"),
    ("ttft_p99_ms", "ttftP99", "{:>8.1f}"),
    ("tpot_p50_ms", "tpotP50", "{:>8.1f}"),
    ("tpot_p99_ms", "tpotP99", "{:>8.1f}"),
]


def _print_table(rows: list[dict]) -> None:
    """Print a pipe-delimited table the notebook can turn into markdown."""
    header = "| " + " | ".join(label for _, label, _ in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    print(header)
    print(sep)
    for r in rows:
        cells = []
        for key, _label, fmt in _COLUMNS:
            v = r.get(key, "")
            try:
                cells.append(fmt.format(v).strip())
            except (ValueError, KeyError, TypeError):
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def run_live(n_requests: int = 50, seed: int = 0) -> dict:
    """Run all three live scenarios (baseline vs CARL) and return/print results.

    This is the entry point `benchmark_carl.py --live` calls and the notebook
    drives. It loads TinyLlama once, then serves each scenario twice.
    """
    import random
    import sys
    import traceback

    # fp16 on GPU (the real serving dtype); fp32 on CPU where half is unsupported
    # / slow. The harness still runs on CPU so it can be smoke-tested without a GPU.
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    # flush=True everywhere below: this module runs as a subprocess captured by
    # the notebook (docs/run_benchmarks.ipynb cell 6c). Without flushing, a long
    # GPU run that later errors or is killed (e.g. OOM) would surface NOTHING --
    # the buffered stdout dies with the process. Flushing makes progress visible
    # as it happens, so the cell can never appear to "fail silently".
    # NB: no '|' in this line. The notebook's to_md_table() (cell 5) scans every
    # stdout line containing a pipe and treats the FIRST one as the table header;
    # a "Device: ... | dtype: ..." line would hijack that and render the real
    # 7-column metrics table as a broken 2-column one in docs/benchmarks.md.
    print(f"Device: {DEVICE}, dtype: {dtype}", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA device -- running on CPU. Numbers are for a smoke "
              "test only; run on a Colab GPU for representative results.\n", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    scenarios = ["INTERACTIVE", "BATCH", "NON-STATIONARY"]
    rows: list[dict] = []
    results: dict = {"device": str(DEVICE), "dtype": str(dtype),
                     "n_requests": n_requests, "scenarios": {}}

    for scenario in scenarios:
        # Same workload (same seed) for both arms, so the only variable is CARL.
        specs_base = _build_workload(tokenizer, scenario, n_requests, random.Random(seed))
        specs_carl = _build_workload(tokenizer, scenario, n_requests, random.Random(seed))

        print(f"\n--- {scenario}: {n_requests} requests, baseline vs CARL ---", flush=True)
        # Guard each scenario: if one arm raises on a given box (a kernel dtype
        # mismatch, an OOM, a driver hiccup), print the FULL traceback and carry
        # on, so the surviving scenarios still produce a table. Failing loudly
        # but partially beats killing the whole cell with no output.
        try:
            base = _run_config(model, tokenizer, specs_base, use_carl=False, seed=seed)
            carl = _run_config(model, tokenizer, specs_carl, use_carl=True, seed=seed)
        except Exception:
            print(f"[{scenario}] FAILED -- skipping (traceback below):", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            results["scenarios"][scenario] = {"error": traceback.format_exc()}
            continue

        base_row = {"scenario": scenario, "config": "baseline", **base}
        carl_row = {"scenario": scenario, "config": "carl_adaptive", **carl}
        rows.extend([base_row, carl_row])
        results["scenarios"][scenario] = {"baseline": base, "carl_adaptive": carl}

    print(flush=True)
    if rows:
        _print_table(rows)
    else:
        # Every scenario errored -- make that unmistakable rather than emitting an
        # empty table the notebook would silently render as nothing.
        print("ERROR: no scenario completed; see the tracebacks above.", flush=True)

    # Per-scenario CARL-vs-baseline throughput delta (prose; ignored by the table
    # parser but useful in the raw output and to a human reader). Only scenarios
    # that actually ran (have a 'baseline' entry) get a delta line.
    print(flush=True)
    for scenario in scenarios:
        sc = results["scenarios"].get(scenario, {})
        if "baseline" not in sc:
            continue
        b = sc["baseline"]["throughput_tok_s"]
        c = sc["carl_adaptive"]["throughput_tok_s"]
        delta = (c - b) / b * 100.0 if b > 0 else 0.0
        adapt = sc["carl_adaptive"].get("adaptations", 0)
        print(f"{scenario:>14}: CARL throughput {c:6.1f} vs baseline {b:6.1f} tok/s "
              f"({delta:+.1f}%), {adapt} live config changes", flush=True)

    DOCS.mkdir(exist_ok=True)
    LIVE_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved live results to {LIVE_RESULTS_PATH}")
    return results


def main_live(args) -> None:
    """Adapter for benchmark_carl.py --live (reads --limit as the request count)."""
    n = getattr(args, "limit", 50)
    # --limit defaults to 500 in benchmark_carl's parser (sized for the LMSYS
    # simulation); that's far too many real generations for a notebook, so cap it.
    if n is None or n > 200:
        n = 50
    run_live(n_requests=n, seed=getattr(args, "seed", 0))


if __name__ == "__main__":
    run_live()
