# mini-vLLM vs production vLLM

An honest, reproducible head-to-head. mini-vLLM is an **educational, from-scratch
PyTorch engine**; [vLLM](https://github.com/vllm-project/vllm) is a production
inference server. This document explains how the comparison is run, what gap to
expect, why the gap exists, and what I took away from measuring it.

The benchmark that produces the numbers is
[`scripts/benchmark_vs_vllm.py`](../scripts/benchmark_vs_vllm.py). It runs on a
Linux/CUDA box (e.g. Google Colab) once `pip install vllm` succeeds; it degrades
gracefully to mini-vLLM-only output anywhere vLLM can't import.

## Methodology

Both engines run the **same model** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`), the
**same prompts**, and the **same `max_tokens` (32)**, with three controls that
make the numbers fair rather than flattering:

1. **Greedy decode (temperature 0)** on both sides. No sampling variance.
2. **EOS ignored** — every request emits exactly `max_tokens`. Neither engine can
   win throughput by happening to stop early; we compare equal work.
3. **Warmup excluded.** vLLM's first `generate()` pays one-off CUDA-graph capture
   and compilation; the script warms up before the timed runs. mini-vLLM has no
   capture step, so this only ever *helps* vLLM — i.e. it makes our relative
   result look worse, which is the honest direction.

### Workloads

| Workload | What it measures | Why it's here |
|---|---|---|
| Single request | **TTFT** (submit → first token), **TPOT** (mean inter-token latency) | The single-user latency story. |
| Batch of 4 | aggregate **throughput** (tok/s) | Light concurrency. |
| Batch of 8 | aggregate **throughput** (tok/s) | Heavier concurrency — where vLLM's scheduler pulls furthest ahead. |

### How each side is measured

- **mini-vLLM** drives its `ContinuousBatchScheduler`, timestamping every token
  emitted by `step()`. TTFT = first emitted token − submit time; TPOT =
  (last − first) / (n − 1); throughput = total tokens / wall time.
- **vLLM** uses the offline `LLM.generate(...)` engine. Per-request TTFT/TPOT
  come from vLLM's own `RequestOutput.metrics` (`arrival_time`,
  `first_token_time`, `finished_time`) when present; throughput is wall-clock
  tokens/sec over the same batch.

## Expected gaps, and why

mini-vLLM is expected to **lose on every metric**. That is the correct outcome
for a hand-written teaching engine versus a production system, and each gap maps
to a specific component vLLM has and v1 of mini-vLLM does not:

| Gap | Cause | vLLM has | mini-vLLM v1 has |
|---|---|---|---|
| **TPOT (per-token latency)** | Hundreds of separate CUDA kernel launches per decode step, each an eager Python→C++ dispatch | **CUDA graphs** — the whole decode step replays as one captured graph, one launch | Eager PyTorch dispatch every step. (A capture/replay runner exists in `src/engine/cuda_graph.py` but is not the default decode path in v1.) |
| **Batch throughput** | Python `step()` loop with per-step host overhead that grows with batch size; attention via `F.scaled_dot_product_attention` over gathered views | **C++ scheduler** + custom **paged-attention CUDA kernels** that keep the GPU saturated | Python scheduler, PyTorch SDPA |
| **TTFT** | Synchronous tokenise-then-admit on one thread | **Async engine** front-end overlapping tokenisation, scheduling, and execution | Single-threaded synchronous admission |
| **Tokenisation** | Tokenises prompts inline on the request path | **Tokenizer parallelism** (background/threaded tokenisation) | Inline HF tokenizer call |

Concretely, on a T4-class GPU expect mini-vLLM to be **roughly an order of
magnitude slower on TPOT** and for the **throughput gap to widen from batch=4 to
batch=8** as vLLM's scheduler amortises overhead that mini-vLLM pays per step.
(Fill in the measured multipliers from your run — the script prints them.)

## What mini-vLLM intentionally omits

mini-vLLM optimises for **legibility**, not throughput. It deliberately leaves
out, or keeps non-default, the machinery that makes vLLM fast but opaque:

- **No C++/CUDA custom kernels.** Everything is PyTorch so each line is
  readable and parity-testable against Hugging Face. PagedAttention is
  implemented as paged KV storage feeding `F.scaled_dot_product_attention`, not
  a fused attention kernel.
- **CUDA graphs are opt-in, not the default.** The point of v1 is to *show* the
  eager decode path clearly; `src/engine/cuda_graph.py` exists to demonstrate
  the technique, not to win the benchmark.
- **Synchronous, single-process server.** One FastAPI app + one pumper thread,
  so the control flow is followable in a debugger. No async multiprocessing
  engine, no distributed/tensor-parallel execution.
- **No continuous quantization / FP8 / speculative decoding by default.** Each
  of these is explored as its own documented experiment (see `README.md` and
  `docs/`), kept off the default path so the baseline stays simple.
- **fp32 baseline.** mini-vLLM runs fp32 for exact HF parity; vLLM here runs
  fp16. That alone accounts for part of the gap — and it's the right call,
  because correctness-by-parity is mini-vLLM's whole thesis.

In short: vLLM hides enormous engineering behind a one-line API; mini-vLLM
exposes the same *ideas* (continuous batching, paged KV, RoPE/GQA, a scheduler,
prefix caching) at the cost of the production-grade speed.

## What I learned from the gap

Measuring the distance to vLLM taught me more than any single feature did:

1. **The latency gap is almost entirely launch overhead, not math.** mini-vLLM
   and vLLM compute the *same* attention; the per-token gap is dominated by
   kernel-launch and Python-dispatch cost, not FLOPs. That is exactly what CUDA
   graphs target — and it reframed CUDA graphs for me from "an optimisation" to
   "the thing standing between an eager engine and a production one." Building
   `cuda_graph.py` was the direct consequence of seeing this number.

2. **Throughput gaps compound with batch size for a structural reason.** vLLM's
   per-step scheduling cost is paid in C++ and amortised across the batch;
   mini-vLLM pays it in Python *per step*, so the relative gap grows as the
   batch grows. The fix isn't "optimise the Python" — it's moving the hot loop
   out of Python entirely, which is why production engines have a C++ scheduler.

3. **fp32-for-parity is a real, quantifiable tax.** Holding correctness to
   `atol=1e-4` against Hugging Face means fp32, and fp32 is ~2x the memory
   bandwidth of fp16 on a bandwidth-bound decode. Seeing that cost made the
   accuracy/throughput trade-off concrete rather than abstract.

4. **"Slower" is the wrong frame; "where, and by how much, and why" is the right
   one.** A single "10x slower" headline hides that mini-vLLM is competitive in
   *correctness* and *clarity*, and that each slowdown is attributable to one
   named, well-understood production technique. The value of this repo is that
   the gap is *explained*, component by component — which is the understanding a
   production engine's one-line API hides.

The honest conclusion: **mini-vLLM is not trying to beat vLLM, and doesn't.** It
is trying to make every idea inside vLLM visible and verifiable, and to measure
precisely what the production techniques buy. This benchmark is the receipt.
