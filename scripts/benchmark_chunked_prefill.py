"""
Benchmark: full prefill vs chunked prefill under a mixed workload.

Run:
    python scripts/benchmark_chunked_prefill.py
    CHUNK_SIZE=128 python scripts/benchmark_chunked_prefill.py   # (informational)

Needs TinyLlama in the HF cache (run `python -m src.engine.model` once). Runs
on whatever DEVICE the engine picks (GPU if present, else CPU -- slower but
still illustrative).

The scenario -- the exact thing chunked prefill is designed to fix:

    Four short requests are already DECODING (emitting one token per
    iteration). Then a single 1024-token prompt arrives.

      * Full prefill: that prompt is one giant forward pass. For that whole
        iteration the four decoders emit nothing -- a visible STALL in their
        token stream (head-of-line blocking).
      * Chunked prefill (chunk_size=256): the prompt is spread over ~4
        iterations, and decode runs in every one. The decoders keep emitting;
        the stall is broken into small pieces.

We measure, for each policy:
    * TTFT (long)      -- submit -> first token of the long prompt.
    * P50 decode lat   -- median inter-token latency across the 4 decoders.
    * max decode stall -- the WORST inter-token gap a decoder saw (this is the
                          number chunked prefill is meant to shrink).
    * throughput       -- total tokens / wall time.

Table: metric | full prefill | chunked (256) | improvement
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler

LONG_LEN = 1024
N_DECODERS = 4
DECODER_MAX_NEW = 48
LONG_MAX_NEW = 8
NUM_BLOCKS = 256          # 4096 tokens of capacity -- comfortably fits the workload
BIG_CHUNK = 10_000_000    # effectively unbounded -> one-shot full prefill


def _percentile(xs, q):
    """q in [0,100]. Simple nearest-rank percentile; xs need not be sorted."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def _make_prompt(tokenizer, length: int) -> torch.Tensor:
    """A length-`length` prompt, built by tiling a base sentence's tokens."""
    base = tokenizer("The quick brown fox jumps over the lazy dog. ",
                     return_tensors="pt")["input_ids"][0]
    reps = (length + len(base) - 1) // len(base)
    ids = base.repeat(reps)[:length].unsqueeze(0)
    return ids.to(DEVICE)


def _run(model, tokenizer, chunk_size):
    """Drive the scheduler through the mixed workload, timestamping tokens.

    Returns (ttft_long, decode_latencies, total_tokens, wall_time)."""
    sched = ContinuousBatchScheduler(
        model, max_batch_size=N_DECODERS + 1, num_blocks=NUM_BLOCKS,
        chunk_size=chunk_size,
    )

    short = _make_prompt(tokenizer, 8)
    times: dict[str, list[float]] = {f"dec{i}": [] for i in range(N_DECODERS)}
    times["long"] = []

    t_start = time.perf_counter()
    for i in range(N_DECODERS):
        sched.add_request(f"dec{i}", short.clone(), max_new_tokens=DECODER_MAX_NEW,
                          eos_token_id=None)

    # Phase 1: let the decoders get into steady-state decode (a few tokens each)
    # BEFORE the long prompt lands, so the stall it causes is mid-stream.
    long_added = False
    long_add_time = None
    total_tokens = 0
    steps = 0
    while sched.has_work():
        steps += 1
        if (not long_added
                and all(len(times[f"dec{i}"]) >= 3 for i in range(N_DECODERS))):
            long_add_time = time.perf_counter()
            sched.add_request("long", _make_prompt(tokenizer, LONG_LEN),
                              max_new_tokens=LONG_MAX_NEW, eos_token_id=None)
            long_added = True

        emitted = sched.step()
        now = time.perf_counter()
        for rid, _tok in emitted:
            times.setdefault(rid, []).append(now)
            total_tokens += 1

        # Stop once the long prompt has produced its first token and the
        # decoders have run well past it -- enough to characterise the stall.
        if long_added and times["long"] and steps > 0:
            if all(len(times[f"dec{i}"]) >= DECODER_MAX_NEW for i in range(N_DECODERS)) \
               or not sched.has_work():
                break

    wall = time.perf_counter() - t_start

    ttft_long = (times["long"][0] - long_add_time) if times["long"] and long_add_time \
        else float("nan")

    # Inter-token latencies for the decoders, restricted to AFTER the long
    # prompt arrived (that's the window where stalls can happen).
    decode_lat: list[float] = []
    for i in range(N_DECODERS):
        ts = [t for t in times[f"dec{i}"] if long_add_time and t >= long_add_time]
        decode_lat.extend((b - a) * 1000.0 for a, b in zip(ts, ts[1:]))  # ms

    return ttft_long * 1000.0, decode_lat, total_tokens, wall


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Workload: {N_DECODERS} decoders + one {LONG_LEN}-token prompt\n")

    ttft_full, lat_full, tok_full, wall_full = _run(model, tokenizer, BIG_CHUNK)
    ttft_chunk, lat_chunk, tok_chunk, wall_chunk = _run(model, tokenizer, 256)

    rows = [
        ("TTFT long (ms)",        ttft_full,                 ttft_chunk,
         lambda a, b: f"{a / b:.2f}x" if b else "n/a"),
        ("P50 decode lat (ms)",   _percentile(lat_full, 50), _percentile(lat_chunk, 50),
         lambda a, b: f"{a / b:.2f}x" if b else "n/a"),
        ("max decode stall (ms)", max(lat_full or [0]),      max(lat_chunk or [0]),
         lambda a, b: f"{a / b:.2f}x less" if b else "n/a"),
        ("throughput (tok/s)",    tok_full / wall_full,      tok_chunk / wall_chunk,
         lambda a, b: f"{b / a:.2f}x" if a else "n/a"),
    ]

    header = f"{'metric':>22} | {'full prefill':>13} | {'chunked (256)':>14} | {'improvement':>12}"
    print(header)
    print("-" * len(header))
    for name, a, b, imp in rows:
        print(f"{name:>22} | {a:>13.3f} | {b:>14.3f} | {imp(a, b):>12}")

    print(
        "\nThe headline is 'max decode stall': full prefill makes the decoders "
        "wait through one 1024-token forward;\nchunked prefill breaks that into "
        "~4 pieces so decode keeps flowing."
    )


if __name__ == "__main__":
    main()
