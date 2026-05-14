# Mini-vLLM Talking Points (v0.1)

Interview cheat sheet. One- or two-sentence answers, with the file or
commit that owns the detail. Read `docs/design.md` for the long form.

## Architecture

**Q: Walk me through the forward pass.**
A: Embedding lookup -> 22 transformer blocks -> final RMSNorm -> LM head.
Each block is `RMSNorm -> attention (RoPE on Q/K, KV cache append, GQA
repeat_interleave, SDPA) -> residual -> RMSNorm -> SwiGLU MLP ->
residual`. RMSNorm is computed in fp32 for numerical stability; the LM
head is *not* tied to the input embedding (TinyLlama config). See
`src/engine/model.py`.

**Q: How does the KV cache work?**
A: Per-layer K and V buffers per request. K is rotated by RoPE *before*
being cached; queries are rotated each step at the current position.
Each layer exposes its own `seq_len(layer_idx)` so RoPE position stays
correct across layers during a single forward pass. Without a cache,
generation is O(N²) attention work; with it, each decode step is O(1).
See `src/engine/kv_cache.py` and commit `ea0c1b4`.

**Q: What is PagedAttention and why use it?**
A: Fixed-size physical blocks in a pool, plus a per-request "block
table" mapping logical positions to physical block indices. Two
indirections per token lookup (`block_table[T // block_size]`, then
slot `T % block_size`). Solves memory fragmentation (any free block
fits any request) and reallocation churn (no grow/copy of contiguous
buffers). See `PagedKVCache` in `src/engine/kv_cache.py`.

**Q: What's in the pool exactly?**
A: Two tensors, `K_pool` and `V_pool`, each shaped
`(num_layers=22, num_blocks, block_size=16, num_kv_heads=4, head_dim=64)`.
The split, the layer-major axis order, and `block_size` before
`num_kv_heads` are all chosen so the per-request gather produces views,
not strided copies, going into SDPA.

## Concurrency

**Q: How does the scheduler decide what to run each step?**
A: Continuous batching. At every step: (1) evict DONE requests; (2)
admit WAITING requests if free-block budget allows; (3) run one forward
pass per PREFILL request; (4) run one batched forward pass over all
DECODE requests. Steady-state is decode-only. See
`ContinuousBatchScheduler.step()` in `src/engine/scheduler.py`.

**Q: Why mixed prefill/decode batching instead of fully separating them?**
A: Prefill prompts have diverse lengths so packing them into a single
batch wastes attention work on padding. Decode steps are all length-1,
so they batch perfectly. Option (a) — sequential prefill, batched
decode — gets 3.10x over solo while keeping the code legible. Full
mixed batching (packed K/V across phases) is a real optimisation but
adds significant complexity.

**Q: How do you bridge sync scheduler -> async WebSocket?**
A: `loop.call_soon_threadsafe(queue.put_nowait, evt)` from the
scheduler thread into a per-subscriber `asyncio.Queue`. The scheduler
emits synchronously and never blocks; the WS handler awaits
`queue.get()` and forwards. The bus itself has no asyncio dependency.
See `src/server/api.py`.

**Q: What if a client is slow or dead?**
A: v0.1 has unbounded per-subscriber queues — a dead client grows its
queue silently until the next `send_json` failure. v0.2 adds bounded
queues with drop-on-full. This is in the "what's not done" list, not
an oversight.

## Bugs solved

**Q: Tell me about a bug you fixed.**
A: Per-layer position off-by-one. The original cache exposed a single
`seq_len()`. Inside a forward pass, layer 0 reads N, appends, then
returns N+1. Layer 1 then reads N+1 and its RoPE position is one ahead
of layer 0's. Logits diverged from HF by a small amount, worse at
deeper layers. Found via a layer-wise logit-diff trace; fixed by making
`seq_len(layer_idx)` per-layer. Commit `ea0c1b4`.

**Q: How did you debug it?**
A: HF parity test was failing at `atol=1e-4`. I instrumented the
forward pass to dump intermediate logits per layer and diffed against
the HF model's same intermediates. The diff was zero at layer 0 and
grew with depth — that's the signature of a position drift.

**Q: Any dtype gotchas?**
A: RMSNorm computes `mean(x²)` which over/underflows in bf16/fp16 for
realistic activations. The fix is to cast to fp32 inside RMSNorm,
normalise, and cast back. All LLaMA-family implementations do this;
forgetting it gives subtle accuracy loss that's hard to spot without a
parity test.

## Decisions

**Q: Why split K and V into two pools?**
A: SDPA takes K and V as separate arguments, so a packed pool would
force a strided slice that materialises a copy on every attention call.
Two pools let me pass views directly.

**Q: Why is the pool layer-major?**
A: `K_pool[layer]` is a contiguous view, and the layer is the natural
iteration unit for a forward pass. Putting `num_blocks` outermost would
have made per-layer indexing a strided gather.

**Q: Why `block_size` before `num_kv_heads` in the pool shape?**
A: After gathering a request's blocks the shape is
`(n_blocks, block_size, NKV, D)`. `.view(-1, NKV, D)[:seq_len]` is then
a free flatten into `(seq_len, NKV, D)`. Reversing the two middle axes
would force a copy.

**Q: Why is the event bus synchronous?**
A: The scheduler emits events from whichever thread is running
`step()`. Making the bus async would mean every emit awaits, every
state transition becomes a yield point, and the scheduler suddenly has
to be an asyncio citizen. A sync bus plus
`loop.call_soon_threadsafe` keeps the scheduler simple and gives each
WS subscriber its own queue for free.

**Q: Why prefill one-at-a-time and decode batched?**
A: Prefills have diverse prompt lengths so they don't pack; decodes are
all length-1 so they pack perfectly. The asymmetry comes from the data,
not the scheduler design.

## Measured numbers

**Q: What's the speedup over solo?**
A: 3.10x for continuous batching with ample blocks (`SimpleKVCache`),
3.27x with paged KV cache (paged overhead amortised). 1.79x with tight
blocks (`num_blocks=6`) — that last one isn't a raw speedup story; it's
the *correctness-under-back-pressure* story.

**Q: What does the 1.79x tight-blocks number mean?**
A: With only 6 blocks for 4 requests, the scheduler can't admit all of
them at once. Request 4 stays WAITING until request 0 finishes and frees
its blocks; then it admits and runs. Wall time goes up, but every
request still produces byte-identical tokens to its solo baseline. The
parity test (`test_paged_scheduler_with_tight_blocks_still_parity`)
proves the admission path is correct, which is what matters under load.

**Q: Why fp32 on CPU? Isn't that slow?**
A: Yes, ~12 tokens/sec on a laptop CPU. v0.1's point is correctness and
architecture, not throughput. fp32 makes the HF parity tests work at
`atol=1e-4` without dtype gymnastics. v0.2 moves to GPU with CUDA SDPA;
requirements already pin `torch==2.5.1+cu121`.

## GPU batching intuition

**Q: Why does the batching speedup drop from 3.10x on CPU to 1.89x on GPU?**
A: On CPU, solo decode is **matmul-bound** — each step has to stream the
1.1B weights from RAM through the BLAS kernels for a single row of work,
and that bandwidth is the bottleneck. Batching N requests through one
forward pass amortises the weight load across N rows, so the speedup
tracks N closely (3.10x for 4 requests).

On GPU with a 1.1B model, solo decode is **bandwidth- and launch-overhead-
bound** rather than matmul-bound — the SMs are mostly idle waiting on
HBM reads and kernel launch latency, not saturating arithmetic. Single-row
matmul barely uses the tensor cores. Batching 4 requests does help, but
there's less waste for it to amortise, so the speedup compresses to 1.89x.
The absolute throughput is still way higher (~12 tok/s per request batched
on GPU vs ~3 tok/s per request batched on CPU); the speedup *ratio* just
looks less dramatic because solo on GPU is already much faster than solo
on CPU.

This is why production inference engines focus on KV-cache bandwidth
(PagedAttention's whole point — fewer cache-memory copies per token) and
on reducing launch overhead (CUDA graphs, fused FlashAttention kernels,
torch.compile) rather than chasing larger batch sizes for their own sake.
Doubling the batch size when you're already bandwidth-bound buys you very
little; halving the per-token bandwidth or fusing the per-step kernel
launches buys you a lot.

**Q: What's the canonical v0.2 numbers source?**
A: `docs/benchmarks/v0.2_gpu.txt` is the raw output of
`scripts/bench_gpu.py` on the RTX 4060 Laptop GPU. It's the source of
truth for the v0.2 README rewrite. Re-running the script on a different
GPU and committing the new file is how those numbers should be refreshed.
