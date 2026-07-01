"""
Speculative-decoding break-even micro-benchmark (self-speculation only).

WHY THIS EXISTS
---------------
The paper claims adaptive online scheduling --- NOT speculative decoding --- is
what drives performance under workload change, and Sec 4.6 states speculation is
single-request only (the scheduler runs the greedy self-speculative early-exit
path solely when one request is in DECODE, and falls back to vanilla under
concurrent decode). This benchmark supplies the empirical backing for that
claim: it characterizes EXACTLY where the engine's existing self-speculative
decoding helps, breaks even, and hurts, as a function of the draft depth
(early-exit layer) and the draft length K. The intended takeaway is honest and
quantitative: on the model/hardware we evaluate, self-speculation's acceptance is
too low to clear break-even over most of the (L, K) grid, so disabling it in the
main experiments costs little --- speculation is simply not the lever.

WHAT IT MEASURES (and what it does NOT)
---------------------------------------
It drives the REAL ContinuousBatchScheduler with a SINGLE request at a time
(max_batch_size=1), so the engine's integrated self-spec path is genuinely
active --- this is the same code the limitation sentence describes, not a
reimplementation. For each prompt it times two configurations:

  * vanilla : enable_spec_decode=False (the baseline decode path).
  * spec    : enable_spec_decode=True at draft length K and early-exit layer L,
              with a spec_decode_observer recording (accepted, K) per round.

CUDA graphs are disabled for BOTH so the comparison is apples-to-apples (graph
capture would otherwise advantage vanilla's fixed-shape decode). EOS is disabled
so every request emits exactly --decode-tokens tokens, making vanilla and spec
emit the same number of tokens (clean TPOT/throughput ratio).

It does NOT touch the scheduler/engine, does NOT exercise batched speculation
(unimplemented), and does NOT use the draft/target decoder (Part B). Scope is
deliberately the single-request self-spec path the engine actually runs.

METRICS (per prompt, aggregated over the L x K sweep and prompt-length buckets)
-------------------------------------------------------------------------------
  acceptance rate   = accepted / (rounds * K)
  throughput        = tokens / wall   (vanilla and spec)
  TTFT              = submit -> first token (ms)
  TPOT              = mean inter-token time over the decode span (ms)
  speedup           = vanilla_TPOT / spec_TPOT   (>1 helps, =1 break-even, <1 hurts)
  empirical break-even K = the K at which mean speedup crosses 1.0 (interpolated)
  analytical alpha* = the acceptance rate that WOULD be needed for break-even at
                      (L, K), from the measured per-round and per-token costs:
                      a spec round costs `round_time`; to break even it must emit
                      round_time / c_vanilla tokens, i.e. accept
                      required = round_time/c_vanilla - 1 draft tokens; we invert
                      E[accepted](alpha, K) = sum_{i=1..K} alpha^i for alpha*.

OUTPUTS
-------
  docs/eval/spec_breakeven_results.json   -- full grid + per-bucket + aggregates
  docs/eval/figures/spec_breakeven_curve.png -- speedup vs K (line per exit layer,
                                              break-even line) + acceptance vs K

Run (on the Colab T4 used for the other live evals; TinyLlama only):
  python scripts/eval/spec_breakeven.py
  python scripts/eval/spec_breakeven.py --synthetic            # skip LMSYS
  python scripts/eval/spec_breakeven.py --per-bucket 4 --seeds 42  # quick

Does NOT modify any existing evaluation script.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime

# --- path bootstrap so `python scripts/eval/spec_breakeven.py` finds src/ ----
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

from src.carl.live import BLOCK_SIZE, NUM_BLOCKS, _make_prompt  # noqa: E402
from src.engine.device import DEVICE  # noqa: E402
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf  # noqa: E402
from src.engine.scheduler import ContinuousBatchScheduler  # noqa: E402

DOCS_EVAL = os.path.join(_REPO_ROOT, "docs", "eval")
FIG_DIR = os.path.join(DOCS_EVAL, "figures")
RESULTS_PATH = os.path.join(DOCS_EVAL, "spec_breakeven_results.json")
FIG_PATH = os.path.join(FIG_DIR, "spec_breakeven_curve.png")

# --- sweep configuration (approved design) -----------------------------------
EXIT_LAYERS = [4, 6, 8, 10, 12]          # early-exit draft depth (of 22 layers)
K_VALUES = [1, 2, 4, 6, 8]               # draft length per spec round
DEFAULT_SEEDS = [42, 43, 44]
DECODE_TOKENS = 128                      # tokens generated per request (EOS off)
# Prompt-length buckets (tokenized prompt length). short/medium/long.
BUCKETS = {"short": (1, 64), "medium": (65, 256), "long": (257, 1024)}
DEFAULT_PER_BUCKET = 6                   # prompts per bucket per seed
# Representative synthetic length per bucket (fallback only).
_SYNTH_LEN = {"short": 48, "medium": 160, "long": 512}


# ===========================================================================
# Prompt sourcing: real LMSYS (realistic acceptance) with synthetic fallback.
# ===========================================================================


def _bucket_of(length: int) -> str | None:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= length <= hi:
            return name
    return None


def build_prompts(tokenizer, per_bucket: int, seed: int, use_lmsys: bool) -> tuple:
    """Return ({bucket: [prompt_ids (1,L), ...]}, source_str).

    Realistic acceptance needs realistic prompts: self-speculation acceptance is
    inflated by repetitive text, so we prefer real LMSYS-Chat-1M prompts (reusing
    benchmark_carl._load_lmsys_prompts) and only fall back to the tiled synthetic
    prompt when LMSYS is unavailable -- flagging that the synthetic acceptance is
    optimistic and not representative.
    """
    rng = random.Random(seed)
    buckets: dict = {b: [] for b in BUCKETS}

    if use_lmsys:
        try:
            import benchmark_carl as bc
            # Pull a generous pool, tokenize, and bin by length until each bucket
            # is filled. The pool size is heuristic (buckets thin out for long).
            pool = bc._load_lmsys_prompts(max(200, per_bucket * 60))
            rng.shuffle(pool)
            for text in pool:
                ids = tokenizer(text, return_tensors="pt")["input_ids"]
                L = int(ids.shape[-1])
                b = _bucket_of(L)
                if b is not None and len(buckets[b]) < per_bucket:
                    buckets[b].append(ids)
                if all(len(v) >= per_bucket for v in buckets.values()):
                    break
            if all(len(v) >= per_bucket for v in buckets.values()):
                return buckets, "lmsys-chat-1m"
            print("  LMSYS pool did not fill every bucket; filling remainder "
                  "with synthetic prompts (flagged).", flush=True)
        except Exception as exc:  # gated / no datasets / offline
            print(f"  LMSYS unavailable ({exc}); using synthetic prompts "
                  f"(acceptance will be OPTIMISTIC).", flush=True)

    # Synthetic fallback (or top-up). Tiled real tokens at a representative length.
    src = "synthetic" if not use_lmsys else "lmsys+synthetic-topup"
    for b in BUCKETS:
        while len(buckets[b]) < per_bucket:
            buckets[b].append(_make_prompt(tokenizer, _SYNTH_LEN[b]))
    return buckets, src


# ===========================================================================
# Drive ONE request through the real scheduler (single-request -> spec active).
# ===========================================================================


def _serve_one(model, prompt_ids, max_new: int, *, spec: bool, k: int,
               exit_layer: int) -> dict:
    """Serve a single request once; return timing + (for spec) acceptance counts."""
    acc = {"accepted": 0, "rounds": 0}

    def _obs(accepted: int, kk: int) -> None:
        acc["accepted"] += accepted
        acc["rounds"] += 1

    sched = ContinuousBatchScheduler(
        model, max_batch_size=1, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
        enable_spec_decode=spec, spec_decode_k=k, spec_decode_exit_layer=exit_layer,
        spec_decode_observer=(_obs if spec else None),
        use_cuda_graphs=False,   # off for BOTH paths -> apples-to-apples timing
    )
    submit = time.perf_counter()
    sched.add_request("r", prompt_ids, max_new_tokens=max_new, eos_token_id=None)
    first = last = None
    n = 0
    while sched.has_work():
        for _rid, _tok in sched.step():
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            n += 1
    finish = time.perf_counter()

    ttft_ms = (first - submit) * 1000.0 if first is not None else 0.0
    decode_s = (last - first) if (first is not None and last is not None and n > 1) else 0.0
    tpot_ms = (decode_s / (n - 1) * 1000.0) if n > 1 else 0.0
    throughput = n / (finish - submit) if finish > submit else 0.0
    return {
        "tokens": n, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
        "throughput_tps": throughput, "decode_s": decode_s,
        "accepted": acc["accepted"], "rounds": acc["rounds"],
    }


# ===========================================================================
# Analytical break-even.
# ===========================================================================


def _expected_accepted(alpha: float, k: int) -> float:
    """E[draft tokens accepted before first reject] over K, for per-token accept
    probability alpha: sum_{i=1..K} alpha^i (a token is accepted only if all
    earlier ones were). Monotone in alpha from 0 (alpha=0) to K (alpha=1)."""
    if alpha >= 1.0:
        return float(k)
    if alpha <= 0.0:
        return 0.0
    return alpha * (1.0 - alpha ** k) / (1.0 - alpha)


def _breakeven_alpha(required_accepted: float, k: int) -> float | None:
    """Acceptance rate alpha* at which a spec round breaks even.

    None  -> unreachable (even accepting all K is not enough; spec slower at any
             acceptance for this L, K).
    0.0   -> already faster at any acceptance (round is cheaper than 1 vanilla tok).
    else  -> bisection inverse of _expected_accepted.
    """
    if required_accepted <= 0.0:
        return 0.0
    if required_accepted >= k:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _expected_accepted(mid, k) < required_accepted:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ===========================================================================
# Driver.
# ===========================================================================


def run(seeds: list, per_bucket: int, decode_tokens: int, use_lmsys: bool) -> dict:
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"Device: {DEVICE} | dtype: {dtype} | model: {MODEL_NAME}", flush=True)
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA -- CPU smoke only; run on the Colab T4.\n", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)
    model.eval()

    # Warmup: trigger lazy init / kernels so the first measured run isn't skewed.
    warm = _make_prompt(tokenizer, 32)
    _serve_one(model, warm, 16, spec=False, k=4, exit_layer=8)
    _serve_one(model, warm, 16, spec=True, k=4, exit_layer=8)

    # records[(bucket, L, K)] = list of per-prompt dicts (speedup, acceptance, ...)
    records: dict = {}
    vanilla_by_bucket: dict = {b: [] for b in BUCKETS}
    prompt_source = None

    for seed in seeds:
        prompts, src = build_prompts(tokenizer, per_bucket, seed, use_lmsys)
        prompt_source = src if prompt_source is None else prompt_source
        for bucket, plist in prompts.items():
            for pi, prompt_ids in enumerate(plist):
                # Vanilla baseline once per prompt (independent of L, K).
                v = _serve_one(model, prompt_ids, decode_tokens, spec=False, k=4, exit_layer=8)
                vanilla_by_bucket[bucket].append(v)
                c_van_tpot_ms = v["tpot_ms"]
                for L in EXIT_LAYERS:
                    for K in K_VALUES:
                        s = _serve_one(model, prompt_ids, decode_tokens,
                                       spec=True, k=K, exit_layer=L)
                        accept = (s["accepted"] / (s["rounds"] * K)
                                  if s["rounds"] > 0 and K > 0 else 0.0)
                        speedup = (c_van_tpot_ms / s["tpot_ms"]
                                   if s["tpot_ms"] > 0 else 0.0)
                        round_time_s = (s["decode_s"] / s["rounds"]
                                        if s["rounds"] > 0 else 0.0)
                        records.setdefault((bucket, L, K), []).append({
                            "acceptance": accept, "speedup": speedup,
                            "spec_tpot_ms": s["tpot_ms"], "spec_ttft_ms": s["ttft_ms"],
                            "spec_throughput_tps": s["throughput_tps"],
                            "vanilla_tpot_ms": c_van_tpot_ms,
                            "vanilla_throughput_tps": v["throughput_tps"],
                            "vanilla_ttft_ms": v["ttft_ms"],
                            "round_time_s": round_time_s,
                        })
                print(f"  seed {seed} {bucket}[{pi+1}/{len(plist)}] done "
                      f"(vanilla {c_van_tpot_ms:.1f} ms/tok)", flush=True)

        # --- Per-seed GPU cleanup (see below: does NOT touch any measurement) --
        # Every _serve_one() above allocated a fresh ContinuousBatchScheduler,
        # each owning a PagedKVCache -- one large GPU tensor sized num_blocks x
        # block_size. Those schedulers are locals inside _serve_one() and are
        # already unreferenced here (the scheduler points at the model, never the
        # reverse), so nothing in this scope still holds a CUDA tensor. What keeps
        # their memory from being reused by the next seed is twofold, and each
        # line below addresses one cause:
        #   1. `del prompts` -- drops this seed's prompt tensors (the only tensor
        #      objects still named in this scope) so they are not pinned across
        #      the collection below.
        #   2. `gc.collect()` -- a scheduler/request/KV-pool object graph can
        #      contain reference cycles, which plain refcounting never frees; the
        #      cyclic collector must run so those GPU tensors are actually
        #      released back to the caching allocator before the next step.
        #   3. `torch.cuda.empty_cache()` -- returns the allocator's now-free but
        #      reserved (and fragmented) blocks to the CUDA driver, so seed N+1
        #      starts from a clean pool instead of OOM-ing on fragmentation. This
        #      is the line that actually fixes the seed-43 OOM.
        # All three run OUTSIDE every timed region (timing lives inside
        # _serve_one, around the step loop) and touch no RNG, so seed results are
        # bit-for-bit unchanged; only when the seed is fully finished do we run.
        del prompts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = _aggregate(records, vanilla_by_bucket, prompt_source, seeds,
                         per_bucket, decode_tokens)
    os.makedirs(DOCS_EVAL, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved break-even results to {RESULTS_PATH}", flush=True)
    _make_figure(results)
    _print(results)
    return results


def _aggregate(records, vanilla_by_bucket, prompt_source, seeds, per_bucket,
               decode_tokens) -> dict:
    # Per (L, K) aggregates across ALL prompts/buckets, plus per-bucket detail.
    grid: dict = {}            # f"L{L}_K{K}" -> aggregate
    per_bucket_grid: dict = {} # bucket -> {f"L{L}_K{K}": aggregate}
    for L in EXIT_LAYERS:
        for K in K_VALUES:
            allp = [r for b in BUCKETS for r in records.get((b, L, K), [])]
            if not allp:
                continue
            agg = _cell(allp, K)
            grid[f"L{L}_K{K}"] = agg
            for b in BUCKETS:
                bp = records.get((b, L, K), [])
                if bp:
                    per_bucket_grid.setdefault(b, {})[f"L{L}_K{K}"] = _cell(bp, K)

    # Empirical break-even K per exit layer (aggregated over buckets): the K where
    # mean speedup first crosses 1.0, linearly interpolated between grid Ks.
    breakeven_k = {}
    for L in EXIT_LAYERS:
        ks, sp = [], []
        for K in K_VALUES:
            cell = grid.get(f"L{L}_K{K}")
            if cell:
                ks.append(K)
                sp.append(cell["speedup_mean"])
        breakeven_k[f"L{L}"] = _crossing(ks, sp, 1.0)

    return {
        "description": ("Break-even characterization of the engine's single-request "
                        "self-speculative (early-exit) decoding: where it helps "
                        "(speedup>1), breaks even (=1), or hurts (<1), across draft "
                        "depth (exit layer L) and draft length K."),
        "scope_note": ("Single-request self-speculation only (max_batch_size=1, "
                       "the path scheduler.py runs when one request decodes). No "
                       "batched speculation, no draft/target. CUDA graphs off for "
                       "both vanilla and spec. EOS disabled so token counts match."),
        "model": MODEL_NAME, "seeds": seeds, "prompts_per_bucket_per_seed": per_bucket,
        "decode_tokens": decode_tokens, "prompt_source": prompt_source,
        "exit_layers": EXIT_LAYERS, "k_values": K_VALUES, "buckets": BUCKETS,
        "grid": grid,
        "per_bucket": per_bucket_grid,
        "empirical_breakeven_k_by_exit_layer": breakeven_k,
        "vanilla_by_bucket": {
            b: {"tpot_ms_mean": _mean([v["tpot_ms"] for v in vs]),
                "ttft_ms_mean": _mean([v["ttft_ms"] for v in vs]),
                "throughput_tps_mean": _mean([v["throughput_tps"] for v in vs]),
                "n": len(vs)}
            for b, vs in vanilla_by_bucket.items() if vs
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda": torch.version.cuda, "torch": torch.__version__,
        },
        "timestamp": datetime.now().isoformat(),
    }


def _cell(rows: list, k: int) -> dict:
    """Aggregate one (L, K) cell + its analytical break-even acceptance."""
    accept = [r["acceptance"] for r in rows]
    speed = [r["speedup"] for r in rows]
    round_t = _mean([r["round_time_s"] for r in rows])
    c_van_s = _mean([r["vanilla_tpot_ms"] for r in rows]) / 1000.0
    # Tokens a spec round must emit to break even, then the acceptance needed.
    req_tokens = (round_t / c_van_s) if c_van_s > 0 else float("inf")
    alpha_star = _breakeven_alpha(req_tokens - 1.0, k)
    return {
        "n": len(rows),
        "acceptance_mean": _mean(accept), "acceptance_std": _std(accept),
        "speedup_mean": _mean(speed), "speedup_std": _std(speed),
        "spec_tpot_ms_mean": _mean([r["spec_tpot_ms"] for r in rows]),
        "spec_ttft_ms_mean": _mean([r["spec_ttft_ms"] for r in rows]),
        "spec_throughput_tps_mean": _mean([r["spec_throughput_tps"] for r in rows]),
        "vanilla_tpot_ms_mean": _mean([r["vanilla_tpot_ms"] for r in rows]),
        "vanilla_throughput_tps_mean": _mean([r["vanilla_throughput_tps"] for r in rows]),
        "round_time_s_mean": round_t,
        "breakeven_tokens_per_round": req_tokens,
        "analytical_breakeven_alpha": alpha_star,  # None = unreachable at this K
        "helps": _mean(speed) > 1.0,
    }


def _crossing(xs: list, ys: list, level: float):
    """First x where y crosses `level` (linear interp). Returns a dict describing
    the crossing, or a status string when y is wholly above/below `level`."""
    if not xs:
        return None
    if all(y < level for y in ys):
        return {"status": "never_reaches_breakeven (always slower)"}
    if all(y >= level for y in ys):
        return {"status": f"breaks_even_at_or_below_K{xs[0]} (already faster)"}
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 < level) != (y1 < level):
            x0, x1 = xs[i - 1], xs[i]
            frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
            return {"breakeven_k": round(x0 + frac * (x1 - x0), 2)}
    return {"status": "no_clean_crossing"}


# ===========================================================================
# Figure.
# ===========================================================================


def _make_figure(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); skipping figure.", flush=True)
        return
    grid = results["grid"]
    Ks = results["k_values"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for L in results["exit_layers"]:
        sp, ac = [], []
        for K in Ks:
            cell = grid.get(f"L{L}_K{K}")
            sp.append(cell["speedup_mean"] if cell else float("nan"))
            ac.append(cell["acceptance_mean"] if cell else float("nan"))
        ax1.plot(Ks, sp, marker="o", label=f"exit L={L}")
        ax2.plot(Ks, ac, marker="o", label=f"exit L={L}")

    ax1.axhline(1.0, ls="--", color="k", lw=1, label="break-even")
    ax1.set_xlabel("draft length K")
    ax1.set_ylabel("speedup vs vanilla (TPOT ratio)")
    ax1.set_title("Self-spec speedup vs K")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("draft length K")
    ax2.set_ylabel("acceptance rate")
    ax2.set_title("Self-spec acceptance vs K")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    src = results.get("prompt_source", "?")
    fig.suptitle(f"Self-speculative decoding break-even --- {results['model']} "
                 f"(prompts: {src})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {FIG_PATH}", flush=True)


def _print(results: dict) -> None:
    print("\n=== SELF-SPEC BREAK-EVEN (TinyLlama, single-request) ===")
    print(f"prompts: {results['prompt_source']} | "
          f"{results['prompts_per_bucket_per_seed']}/bucket/seed x "
          f"{len(results['seeds'])} seeds x {len(results['buckets'])} buckets")
    print("\nspeedup (mean) by exit layer x K  [* = helps, i.e. >1]")
    print("| L \\ K | " + " | ".join(f"K={k}" for k in results["k_values"]) + " |")
    for L in results["exit_layers"]:
        cells = []
        for K in results["k_values"]:
            c = results["grid"].get(f"L{L}_K{K}")
            if c:
                cells.append(f"{c['speedup_mean']:.2f}{'*' if c['helps'] else ' '}")
            else:
                cells.append("  -  ")
        print(f"| L={L:<3} | " + " | ".join(cells) + " |")
    print("\nempirical break-even K by exit layer:")
    for L, v in results["empirical_breakeven_k_by_exit_layer"].items():
        print(f"  {L}: {v}")
    print("\nacceptance (mean) by exit layer x K:")
    for L in results["exit_layers"]:
        row = [results["grid"].get(f"L{L}_K{K}") for K in results["k_values"]]
        print(f"  L={L:<3}: " + " ".join(
            f"{c['acceptance_mean']:.2f}" if c else " - " for c in row))


def main() -> None:
    p = argparse.ArgumentParser(description="Self-spec break-even micro-benchmark (TinyLlama).")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                   help="comma-separated seeds (default 42,43,44)")
    p.add_argument("--per-bucket", type=int, default=DEFAULT_PER_BUCKET,
                   help="prompts per length bucket per seed")
    p.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS,
                   help="tokens generated per request")
    p.add_argument("--synthetic", action="store_true",
                   help="skip LMSYS; use synthetic prompts (acceptance is optimistic)")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run(seeds, max(1, args.per_bucket), max(8, args.decode_tokens),
        use_lmsys=not args.synthetic)


if __name__ == "__main__":
    main()
