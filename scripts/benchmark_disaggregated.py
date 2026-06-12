"""
Benchmark: disaggregated prefill/decode vs the unified engine.

Run:
    python scripts/benchmark_disaggregated.py

Needs TinyLlama in the HF cache (run `python -m src.engine.model` once). Runs on
whatever DEVICE the engine picks (GPU if present, else CPU).

THE WORKLOAD
------------
A deliberately adversarial mix for a UNIFIED engine:
  * 4 LONG prompts (512 tokens, 32 new tokens each) -- heavy prefill.
  * 8 SHORT prompts (16 tokens, 64 new tokens each) -- decode-dominated, the
    latency-sensitive traffic.

In a unified worker the long prefills land in the same step loop as the short
requests' decode, so a decode token can sit behind a 512-token prefill forward.
That is exactly the contention disaggregation removes: decode steps on the
decode worker never carry prefill work.

WHAT WE REPORT
--------------
  * prefill throughput  -- prompt tokens processed per second.
  * decode TPOT         -- mean time-per-output-token, measured only over a
                           request's DECODE phase (the inter-token latency a
                           user feels once streaming starts).
  * P99 TTFT            -- 99th-percentile time-to-first-token across all
                           requests (tail latency, the SLO that matters).

HONESTY ABOUT THE SIMULATION
----------------------------
This is a SINGLE-PROCESS simulation sharing one model instance, so the two paths
do not get real wall-clock overlap across separate GPUs. The robust, hardware-
independent signal here is decode TPOT STABILITY: on the disaggregated path
every timed decode step is pure decode, so TPOT does not inflate with prompt
length the way it does when a unified step absorbs a long prefill. True
end-to-end speedup needs the real two-GPU + RDMA deployment Mooncake describes.
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE
from src.engine.disaggregated import DisaggregatedEngine, _Request
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler

N_LONG, LONG_PROMPT, LONG_NEW = 4, 512, 32
N_SHORT, SHORT_PROMPT, SHORT_NEW = 8, 16, 64


def _percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0,100]); xs need not be sorted."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    frac = rank - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def _make_workload(vocab_size: int) -> list[tuple[str, torch.Tensor, int]]:
    """(request_id, prompt_ids (1,P), max_new_tokens) for the mixed workload."""
    g = torch.Generator().manual_seed(20240601)
    work: list[tuple[str, torch.Tensor, int]] = []
    for i in range(N_LONG):
        ids = torch.randint(0, vocab_size, (1, LONG_PROMPT), generator=g)
        work.append((f"long{i}", ids, LONG_NEW))
    for i in range(N_SHORT):
        ids = torch.randint(0, vocab_size, (1, SHORT_PROMPT), generator=g)
        work.append((f"short{i}", ids, SHORT_NEW))
    return work


# ---------------------------------------------------------------------------
# Unified baseline: one ContinuousBatchScheduler interleaving everything.
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_unified(model, work) -> dict:
    total_prompt_tokens = sum(int(ids.shape[1]) for _, ids, _ in work)
    sched = ContinuousBatchScheduler(
        model, max_batch_size=len(work), num_blocks=4096, block_size=16,
        chunk_size=10_000,   # no chunking: full prefill in one step (the worst case)
    )
    submit_t: dict[str, float] = {}
    first_token_t: dict[str, float] = {}
    last_token_t: dict[str, float] = {}
    out_tokens: dict[str, int] = {}

    t0 = time.perf_counter()
    for rid, ids, n in work:
        sched.add_request(request_id=rid, prompt_ids=ids, max_new_tokens=n)
        submit_t[rid] = t0  # all submitted up front

    while sched.has_work():
        emitted = sched.step()
        now = time.perf_counter()
        for rid, _tok in emitted:
            first_token_t.setdefault(rid, now)
            last_token_t[rid] = now
            out_tokens[rid] = out_tokens.get(rid, 0) + 1
        sched.get_finished()
    wall = time.perf_counter() - t0

    return _summarise(work, submit_t, first_token_t, last_token_t, out_tokens,
                      total_prompt_tokens, wall, prefill_wall=wall)


# ---------------------------------------------------------------------------
# Disaggregated path: prefill worker -> transfer queue -> decode worker.
# ---------------------------------------------------------------------------


def run_disaggregated(model, work) -> dict:
    import asyncio

    total_prompt_tokens = sum(int(ids.shape[1]) for _, ids, _ in work)
    submit_t: dict[str, float] = {}
    first_token_t: dict[str, float] = {}
    last_token_t: dict[str, float] = {}
    out_tokens: dict[str, int] = {}
    prefill_done_t: dict[str, float] = {}

    def on_token(rid: str, _tok: int) -> None:
        now = time.perf_counter()
        first_token_t.setdefault(rid, now)
        last_token_t[rid] = now
        out_tokens[rid] = out_tokens.get(rid, 0) + 1

    def on_prefill(rid: str, _seq_len: int) -> None:
        prefill_done_t[rid] = time.perf_counter()

    engine = DisaggregatedEngine(
        model, decode_blocks=4096, token_observer=on_token, prefill_observer=on_prefill,
    )
    requests = [_Request(rid, ids, n) for rid, ids, n in work]

    t0 = time.perf_counter()
    for rid, _ids, _n in work:
        submit_t[rid] = t0
    asyncio.run(engine.run_batch(requests))
    wall = time.perf_counter() - t0

    # Prefill wall = when the last prompt's KV became available. The prefill
    # worker runs them back-to-back, so this is the prefill stage's own time.
    prefill_wall = (max(prefill_done_t.values()) - t0) if prefill_done_t else wall
    return _summarise(work, submit_t, first_token_t, last_token_t, out_tokens,
                      total_prompt_tokens, wall, prefill_wall=prefill_wall)


# ---------------------------------------------------------------------------
# Shared metric reduction.
# ---------------------------------------------------------------------------


def _summarise(work, submit_t, first_token_t, last_token_t, out_tokens,
               total_prompt_tokens, wall, prefill_wall) -> dict:
    ttfts = [
        (first_token_t[rid] - submit_t[rid]) * 1e3       # ms
        for rid, _ids, _n in work if rid in first_token_t
    ]
    # TPOT per request: decode-phase time / decode tokens (= all but the first).
    tpots: list[float] = []
    for rid, _ids, _n in work:
        n_out = out_tokens.get(rid, 0)
        if n_out >= 2:
            decode_span = last_token_t[rid] - first_token_t[rid]
            tpots.append((decode_span / (n_out - 1)) * 1e3)   # ms/token
    return {
        "wall_s": wall,
        "prefill_throughput": total_prompt_tokens / prefill_wall if prefill_wall > 0 else float("nan"),
        "decode_tpot_ms": sum(tpots) / len(tpots) if tpots else float("nan"),
        "p99_ttft_ms": _percentile(ttfts, 99),
        "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else float("nan"),
    }


def _print_row(name: str, m: dict) -> None:
    print(f"{name:>14} | {m['wall_s']:>8.3f} | {m['prefill_throughput']:>12.1f} | "
          f"{m['decode_tpot_ms']:>11.3f} | {m['p99_ttft_ms']:>11.2f} | {m['mean_ttft_ms']:>11.2f}")


def main() -> None:
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    work = _make_workload(model.config.vocab_size)

    print(f"Device: {DEVICE}")
    print(f"Workload: {N_LONG} long ({LONG_PROMPT} tok, +{LONG_NEW}) + "
          f"{N_SHORT} short ({SHORT_PROMPT} tok, +{SHORT_NEW})\n")

    unified = run_unified(model, work)
    disagg = run_disaggregated(model, work)

    header = (f"{'engine':>14} | {'wall (s)':>8} | {'prefill tok/s':>12} | "
              f"{'TPOT (ms)':>11} | {'P99 TTFT':>11} | {'mean TTFT':>11}")
    print(header)
    print("-" * len(header))
    _print_row("unified", unified)
    _print_row("disaggregated", disagg)

    print("\nReading the table:")
    print("  * prefill tok/s -- disaggregated isolates prefill onto its own stage,")
    print("    so its throughput reflects pure prefill compute (no decode mixed in).")
    print("  * TPOT -- disaggregated decode steps are PURE decode; the number does")
    print("    not inflate with the 512-token prefills the way the unified one can.")
    print("  * P99 TTFT -- tail time-to-first-token; the SLO disaggregation protects.")
    print("\nNote: single-process simulation (shared model), so wall-clock overlap is")
    print("limited; true end-to-end speedup needs the two-GPU + RDMA deployment.")


if __name__ == "__main__":
    main()
