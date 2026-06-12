"""
Honest head-to-head: mini-vLLM vs production vLLM.

Run:
    pip install vllm            # Linux/Colab + CUDA; see docs/vllm_comparison.md
    python scripts/benchmark_vs_vllm.py

Both engines run TinyLlama-1.1B-Chat on IDENTICAL prompts with IDENTICAL
max_tokens, greedy decode (temperature 0), and EOS ignored so every request
generates exactly max_tokens -- that makes the throughput numbers apples-to-
apples (no engine wins by stopping early).

WORKLOADS
---------
  * Single request -- TTFT (submit -> first token) and TPOT (mean inter-token
    latency). The latency story.
  * Batch of 4    -- aggregate throughput (tokens/sec). The concurrency story.
  * Batch of 8    -- aggregate throughput (tokens/sec). Pushes the scheduler
                     harder; this is where vLLM's C++ scheduler + CUDA graphs
                     pull furthest ahead.

GRACEFUL DEGRADATION
--------------------
If vLLM is not importable, the script prints install instructions and reports
mini-vLLM's own numbers, so it is still useful on a box without vLLM (e.g. the
CPU-only dev machine). The comparison columns simply read "n/a".

THIS IS A DELIBERATELY UNFLATTERING BENCHMARK. mini-vLLM is an educational,
pure-PyTorch engine; vLLM is a production system with a C++ scheduler, CUDA
graphs, fused kernels, and an async frontend. We expect to lose -- the point is
to measure the gap precisely and explain it (docs/vllm_comparison.md), which is
worth more than a flattering microbenchmark.
"""
from __future__ import annotations

import time
from statistics import mean

import torch

from src.engine.device import DEVICE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler

# A few realistic chat-style prompts. Batch-of-N takes the first N (cycling).
PROMPTS = [
    "Explain what a transformer neural network is in two sentences.",
    "Write a haiku about the ocean at dawn.",
    "List three reasons why caching speeds up web servers.",
    "What is the difference between a process and a thread?",
    "Summarise the plot of Romeo and Juliet in one paragraph.",
    "Give me a simple recipe for pancakes.",
    "Why is the sky blue? Answer briefly.",
    "Translate 'good morning, my friend' into French.",
]

MAX_TOKENS = 32


def _prompts(n: int) -> list[str]:
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


# ---------------------------------------------------------------------------
# mini-vLLM measurement (via the ContinuousBatchScheduler).
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure_mini(model, tokenizer, prompts: list[str], max_tokens: int) -> dict:
    """Drive the scheduler over `prompts`, timestamping every emitted token.

    EOS is disabled (eos_token_id=None) so each request emits exactly
    max_tokens -- the same fixed-length contract we give vLLM (ignore_eos).
    Returns mean TTFT (ms), mean TPOT (ms), throughput (tok/s), and wall (s).
    """
    ids = [tokenizer(p, return_tensors="pt")["input_ids"] for p in prompts]
    # Generous block pool so admission control never throttles -- we are
    # measuring engine speed, not back-pressure behaviour.
    total_cap = sum(int(x.shape[1]) for x in ids) + len(ids) * max_tokens
    num_blocks = (total_cap + 15) // 16 + len(ids) + 8
    sched = ContinuousBatchScheduler(
        model, max_batch_size=len(prompts), num_blocks=num_blocks,
        block_size=16, chunk_size=10_000_000,   # one-shot prefill
    )

    submit: dict[str, float] = {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    count: dict[str, int] = {}

    t0 = time.perf_counter()
    for i, x in enumerate(ids):
        rid = f"r{i}"
        sched.add_request(rid, x, max_new_tokens=max_tokens, eos_token_id=None)
        submit[rid] = t0
    while sched.has_work():
        emitted = sched.step()
        now = time.perf_counter()
        for rid, _tok in emitted:
            first.setdefault(rid, now)
            last[rid] = now
            count[rid] = count.get(rid, 0) + 1
        sched.get_finished()
    wall = time.perf_counter() - t0

    ttfts = [(first[r] - submit[r]) * 1e3 for r in first]
    tpots = [
        (last[r] - first[r]) / (count[r] - 1) * 1e3
        for r in first if count[r] >= 2
    ]
    total = sum(count.values())
    return {
        "ttft_ms": mean(ttfts) if ttfts else float("nan"),
        "tpot_ms": mean(tpots) if tpots else float("nan"),
        "throughput": total / wall if wall > 0 else float("nan"),
        "wall": wall,
        "total_tokens": total,
    }


# ---------------------------------------------------------------------------
# vLLM measurement (offline LLM engine).
# ---------------------------------------------------------------------------


def measure_vllm(llm, prompts: list[str], max_tokens: int) -> dict:
    """Run `prompts` through a (pre-initialised) vLLM engine, greedy, fixed
    length. Pulls per-request TTFT/TPOT from vLLM's own RequestOutput.metrics
    when available; always reports wall-clock throughput.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    ttfts: list[float] = []
    tpots: list[float] = []
    total = 0
    for o in outs:
        n = len(o.outputs[0].token_ids)
        total += n
        m = getattr(o, "metrics", None)
        if m is not None:
            arr = getattr(m, "arrival_time", None)
            ftt = getattr(m, "first_token_time", None)
            fin = getattr(m, "finished_time", None)
            if arr is not None and ftt is not None:
                ttfts.append((ftt - arr) * 1e3)
            if ftt is not None and fin is not None and n >= 2:
                tpots.append((fin - ftt) / (n - 1) * 1e3)
    return {
        "ttft_ms": mean(ttfts) if ttfts else float("nan"),
        "tpot_ms": mean(tpots) if tpots else float("nan"),
        "throughput": total / wall if wall > 0 else float("nan"),
        "wall": wall,
        "total_tokens": total,
    }


def _try_init_vllm():
    """Return an initialised vLLM engine, or None with printed guidance."""
    try:
        from vllm import LLM
    except Exception as exc:  # ImportError, or CUDA/driver import-time failures
        print("vLLM not available -- reporting mini-vLLM results only.")
        print(f"  (import failed: {type(exc).__name__}: {exc})")
        print("\nTo run the full comparison (needs Linux + CUDA, e.g. Colab):")
        print("    pip install vllm")
        print("    python scripts/benchmark_vs_vllm.py\n")
        return None
    try:
        # gpu_memory_utilization kept modest so this coexists with mini-vLLM's
        # own weights already resident on the same GPU.
        llm = LLM(model=MODEL_NAME, dtype="float16", gpu_memory_utilization=0.45,
                  enforce_eager=False, max_model_len=2048)
        # Warm up: first generate() pays one-off compilation / CUDA-graph
        # capture cost we don't want polluting the timed runs.
        from vllm import SamplingParams
        llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
        return llm
    except Exception as exc:
        print(f"vLLM import succeeded but engine init failed: "
              f"{type(exc).__name__}: {exc}")
        print("Reporting mini-vLLM results only.\n")
        return None


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _ratio_slower(mini: float, prod: float) -> str:
    """How many times SLOWER mini is than prod for a latency metric."""
    if not prod or prod != prod or mini != mini:   # 0 or NaN
        return "n/a"
    return f"{mini / prod:.1f}x slower"


def _ratio_throughput(mini: float, prod: float) -> str:
    """How many times MORE throughput prod has than mini."""
    if not mini or mini != mini or prod != prod:
        return "n/a"
    return f"{prod / mini:.1f}x"


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_NAME}")
    print(f"max_tokens={MAX_TOKENS}, greedy, EOS ignored (fixed-length)\n")

    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    llm = _try_init_vllm()

    def both(prompts):
        m = measure_mini(model, tokenizer, prompts, MAX_TOKENS)
        v = measure_vllm(llm, prompts, MAX_TOKENS) if llm is not None else None
        return m, v

    single_m, single_v = both(_prompts(1))
    batch4_m, batch4_v = both(_prompts(4))
    batch8_m, batch8_v = both(_prompts(8))

    # ---- Latency table (single request) --------------------------------
    print("SINGLE REQUEST (latency)")
    head = f"{'metric':>12} | {'mini-vLLM':>10} | {'vLLM':>10} | {'gap':>14}"
    print(head); print("-" * len(head))
    vt = single_v["ttft_ms"] if single_v else float("nan")
    vp = single_v["tpot_ms"] if single_v else float("nan")
    print(f"{'TTFT (ms)':>12} | {single_m['ttft_ms']:>10.2f} | "
          f"{vt:>10.2f} | {_ratio_slower(single_m['ttft_ms'], vt):>14}")
    print(f"{'TPOT (ms)':>12} | {single_m['tpot_ms']:>10.2f} | "
          f"{vp:>10.2f} | {_ratio_slower(single_m['tpot_ms'], vp):>14}")

    # ---- Throughput table (batches) ------------------------------------
    print("\nBATCH THROUGHPUT (tokens/sec)")
    head = f"{'workload':>12} | {'mini-vLLM':>10} | {'vLLM':>10} | {'vLLM faster':>12}"
    print(head); print("-" * len(head))
    for label, m, v in [("batch=4", batch4_m, batch4_v), ("batch=8", batch8_m, batch8_v)]:
        vthru = v["throughput"] if v else float("nan")
        print(f"{label:>12} | {m['throughput']:>10.1f} | {vthru:>10.1f} | "
              f"{_ratio_throughput(m['throughput'], vthru):>12}")

    # ---- Honest commentary ---------------------------------------------
    print("\nWHERE mini-vLLM LOSES, AND WHY")
    print("  * TPOT: every decode step is an eager PyTorch dispatch; vLLM replays")
    print("    a captured CUDA graph (one launch vs hundreds of kernel launches).")
    print("  * Throughput at batch=8: vLLM's C++ scheduler + paged attention CUDA")
    print("    kernels keep the GPU saturated; our Python step loop has per-step")
    print("    host overhead that grows with batch size.")
    print("  * TTFT: vLLM overlaps tokenisation/scheduling with an async frontend;")
    print("    mini-vLLM tokenises and admits synchronously on one thread.")
    if llm is None:
        print("\n(vLLM column is 'n/a' -- install vLLM on a CUDA box for the full table.)")
    print("\nFull methodology and analysis: docs/vllm_comparison.md")


if __name__ == "__main__":
    main()
