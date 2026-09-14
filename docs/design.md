# Mini-vLLM Design Notes (v0.1)

The README has the headline features; this is the long-form *why*.
Engineering tone, not academic — every section points at the file that
owns the decision, and is meant to be read alongside the source.

## 1. Why this project exists

The goal was to understand modern LLM serving — vLLM specifically — at
the level where I can defend every line in an interview, not just
describe what the system does. That meant writing the whole forward
pass from scratch on top of `torch.nn.functional`, building the cache,
the scheduler, and the wire protocol myself, with the Hugging Face
implementation as the reference. Reading other people's inference code
is fast; building it is slow and pays much better.

TinyLlama-1.1B-Chat is the target model: small enough to run on a CPU
in fp32, large enough that all the structural features of a modern LLM
(RoPE, GQA, SwiGLU, RMSNorm, KV cache) are present and load-bearing.

## 2. HF parity as the correctness anchor

The single most important architectural decision was making every
component testable against `transformers` at the tensor level. The
parity tests form a hierarchy:

1. **Forward-pass parity** (`test_model_parity.py`). Load TinyLlama
   into both our `LlamaModel` and `transformers.AutoModelForCausalLM`,
   run identical inputs, assert logits match at `atol=1e-4`. Pins
   down RoPE, GQA, SwiGLU, RMSNorm, and the weight loader at once.
2. **Generation parity** (`test_generation_parity.py`). Greedy decode
   matches HF token-for-token over 50 generated tokens.
3. **Cached-decode parity** (`test_kv_cache_parity.py`). Same tokens
   as the non-cached path *and* as HF's cached path.
4. **Scheduler parity** (`test_scheduler_parity.py`,
   `test_paged_kv_parity.py`). Each request driven through the
   continuous-batch scheduler produces the same tokens as a solo
   `generate()` call. Two variants: ample blocks, and tight blocks
   (forces admission control).
5. **Event stream + visualiser** (`test_events.py`). End-to-end
   WebSocket test that the scheduler emits all expected event types
   and that the visualiser page is served.

This hierarchy means bugs surface at the lowest layer that introduces
them. A scheduler test failure can't be silently masking a forward-pass
bug, because the forward pass is independently pinned to HF.

## 3. From-scratch LLaMA forward pass

A "naive transformer" tutorial does not get you to LLaMA. The deltas:

- **RMSNorm, not LayerNorm.** Drop the mean-centring step and the
  bias. The RMSNorm paper's empirical claim — that recentring
  contributes nothing to model quality — has held up across every
  LLaMA-family model. Computed in fp32 to avoid `mean(x²)` over/under-
  flow even when the rest of the model runs lower-precision.
- **SwiGLU MLP, not GELU.** Two parallel linear projections, one gated
  through SiLU and multiplied element-wise with the other, then a
  down-projection. Triple the parameter count of a GELU MLP for the
  same hidden size, but better quality per parameter at LLM scale.
- **GPT-J-style RoPE, not learned positional embeddings.** Rotate
  pairs of dimensions in Q and K by a position-dependent angle. The
  GPT-J variant pairs adjacent dims `(0,1), (2,3), …`; the GPT-NeoX
  variant pairs halves `(0, D/2), (1, D/2+1), …`. Easy to get wrong;
  the parity test catches this immediately.
- **Grouped-query attention (GQA).** TinyLlama has 32 Q heads and 4
  KV heads — the KV cache is 8x smaller as a result. Implementation:
  `K = K.repeat_interleave(num_q_per_kv, dim=1)` before SDPA. This
  materialises the broadcast; a memory-optimal implementation would
  push the repeat into the attention kernel itself.
- **No weight tying.** TinyLlama keeps the LM head separate from the
  embedding matrix (`tie_word_embeddings: false` in the HF config).
  Getting this wrong silently halves your vocab parameter count.

## 4. KV cache: O(N²) -> O(1) per decode token

Without a cache, every decode step re-runs attention over the entire
prefix, so generating N tokens is O(N²) attention work in total. With
a cache, each decode step is O(1) — the new token attends to the
cached K/V of the prefix without recomputing them.

Two non-obvious things this taught me:

**Per-layer position handling.** Each transformer block has its own K
and V buffers. A cache that exposes a single `seq_len()` to the caller
breaks when used inside a forward pass: layer 0 reads N, appends,
returns N+1; then layer 1 reads N+1 and its RoPE position is one ahead
of layer 0. Symptom: logits diverge from HF by a small but non-trivial
amount, worse at deeper layers. Fix: `seq_len(layer_idx)` is
per-layer. Caught by adding a per-layer logit-diff trace; see commit
`ea0c1b4` for the full story.

**Where rotation happens.** K is rotated by RoPE *before* being
cached; queries are rotated each step at the current position. If you
cached the un-rotated K and rotated at attention time, you would
re-rotate the entire history every step — defeating the cache.

## 5. Continuous batching state machine

Naive batching takes N prompts, runs prefill on all of them, decodes
in lockstep. Two failure modes: head-of-line blocking (one long
request stalls everyone), and idle slots (short requests finish but
the slot is "claimed" until the whole batch drains).

Continuous batching (Orca / vLLM) makes the scheduling unit a single
decode step. After every step: finished requests evicted, new
requests admitted. The batch composition is fluid.

State machine per request:

```
WAITING --(capacity + block budget)--> PREFILL
PREFILL --(one prefill forward, emit 1 token)--> DECODE
DECODE  --(decode step, emit 1 token)--> DECODE (loop)
DECODE  --(EOS or max_new_tokens)--> DONE
```

Implementation choice: mixed prefill + decode batching, option (a).
Prefill requests are processed one forward pass each (a small prompt
batch wouldn't pack well across diverse lengths). Decode requests are
batched into one forward pass. Per step we do `(n_prefill + 1)`
forward passes; in steady state most steps are decode-only. A more
sophisticated implementation (full mixed batching with a packed K/V
tensor across both phases) is a real optimisation, but option (a)
keeps the code legible and still beats solo by 3.10x.

## 6. PagedAttention: physical pool, logical mapping, admission control

The cache layout that ships in vLLM solves two problems
simultaneously: memory fragmentation and reallocation churn. Our
`PagedKVCache` is the same design, sized for TinyLlama:

```
K_pool: (num_layers=22, num_blocks=N, block_size=16, num_kv_heads=4, head_dim=64)
V_pool: (num_layers=22, num_blocks=N, block_size=16, num_kv_heads=4, head_dim=64)
```

Layout decisions (load-bearing for performance):

- **Split K and V** into two pools — SDPA takes them as separate
  arguments, so a packed pool would force a strided slice that
  materialises a copy.
- **Layer-major** (num_layers outermost) — `K_pool[layer]` is a
  contiguous view; the layer is the natural iteration unit.
- **`block_size` before `num_kv_heads`** — after gathering a
  request's blocks the shape is `(n_blocks, block_size, NKV, D)`, and
  `.view(-1, NKV, D)[:seq_len]` is a free flatten. Reversing the two
  middle axes would force a copy.

Per-request bookkeeping is a *block table*: an int list mapping
logical position to physical block index. To look up token T:
`physical = block_table[T // block_size]; slot = T % block_size`.
Two indirections, both cheap.

**Admission control.** A request is admitted only if the pool has
enough free blocks for its worst-case footprint
(`ceil((prompt_len + max_new_tokens) / block_size)`). If not, it
stays WAITING until other requests finish and return their blocks.
The tight-blocks parity test (`num_blocks=6`) exercises this path and
proves correctness under back-pressure: with only 6 blocks available,
request 4 cannot be admitted alongside the others; once request 0
finishes and frees its blocks, request 4 admits and runs to
completion with byte-identical tokens to its solo baseline.

This is the production admission story: under load, requests queue
rather than thrash memory.

## 7. WebSocket event stream + threading model

The scheduler emits structured events synchronously at every state
transition: `request_admitted`, `prefill_started`, `decode_step`,
`request_finished`, `block_allocated`, `block_freed`, `pool_state`.
Events are dataclasses; the bus is a list of synchronous subscribers;
the scheduler doesn't know what asyncio is.

The bridge from sync to async lives in the FastAPI WebSocket handler:

1. On WS accept, capture `asyncio.get_running_loop()`.
2. Create a per-subscriber `asyncio.Queue`.
3. Subscribe to the bus with a callback that runs in the *emitting*
   thread: `lambda evt: loop.call_soon_threadsafe(queue.put_nowait, evt)`.
4. The handler loop awaits `queue.get()` and sends each event.

`call_soon_threadsafe` is the documented thread-safe primitive for
getting work onto an event loop from a non-loop thread. It returns
immediately, so `emit()` never blocks scheduler work. Each
subscriber's queue is independent — a slow client cannot back up
another subscriber's stream.

The `/generate` endpoint is synchronous: each request acquires the
scheduler lock, pumps `step()` until its `request_id` shows up in the
finished-results map, and releases the lock between iterations so
concurrent callers interleave. Concurrent `POST /generate` calls
share one scheduler instance — that's how the demo batches.

## 8. Visualiser design

The visualiser is a single-page app served at `GET /`. Two rendering
regimes split by data shape:

- **DOM for low-cardinality, semantic data.** Active-request rows
  and the recent-events log are plain DOM elements — at most a few
  dozen rows, each with text, a status badge, and a colour swatch.
  DOM gives us flexbox, ellipsis truncation, and ARIA-friendly
  semantics for free.
- **D3 SVG for the cache grid.** The block pool is a regular grid of
  64+ rectangles whose fills change on every `block_allocated`,
  `block_freed`, and `pool_state` event. D3's data-join pattern
  (`selectAll().data().join()`) is the right tool for this: bind
  block index -> rect, update fill on event, no manual diffing.

Colour mapping is **FNV-1a hash of `request_id` -> hue**. A 32-bit
hash modulo 360 gives a deterministic hue per request; HSL with fixed
S/L makes the palette visually consistent. The same request_id
therefore lights up the same colour in the row, the cache grid, and
the event log — the eye can trace one request through the system
without reading a single text label.

The visualiser subscribes once to `/events` on page load and
processes events synchronously; no state is recomputed from the
server beyond the event stream.

## 9. v0.2 roadmap (what's not done)

- **GPU.** Everything runs on CPU in fp32. CUDA SDPA plus paged-
  attention kernels (xformers / FlashAttention-2) is the immediate
  next step; the cache layout is already what those kernels expect.
  Requirements pin `torch==2.5.1+cu121` in anticipation.
- **Streaming responses.** `/generate` currently blocks until the
  request finishes; v0.2 will yield tokens via Server-Sent Events as
  soon as each `decode_step` fires.
- **Prefix caching.** Multiple requests that share a system prompt
  could share the prefix's KV blocks copy-on-write. vLLM 0.3+ ships
  this; ours doesn't.
- **Metrics.** Prometheus counters for tokens/sec, queue depth,
  admission rejections, block-pool utilisation. The event bus is
  already the right place to hook these.
- **Sampling beyond greedy.** Temperature, top-k, top-p — a one-day
  addition once the scheduler-side plumbing is in.
- **Backpressure on the event WebSocket.** Per-subscriber queue is
  unbounded; a dead client would grow its queue silently until the
  next `send_json` failure. Bounded queue + drop-on-full is the
  production move.

These are deliberate omissions for v0.1, not unknowns. v0.2 brings
them in one at a time with the same parity-test discipline.

## 10. v0.3 custom kernels: from-scratch Triton vs. vendor libraries

v0.3 adds hand-written CUDA-path kernels — a fused RoPE kernel, a
from-scratch FlashAttention-2 forward, and INT8 (W8A8) quantized
projections — each as the CUDA path with the existing PyTorch op kept as
the CPU fallback and the parity reference. Two findings from running them
on real hardware (Colab T4, sm_75) are worth recording, because both are
"the textbook result," not bugs.

### 10.1 Our FlashAttention-2 is slower than `F.scaled_dot_product_attention`

The from-scratch FA2 kernel reproduces SDPA's output to ~1e-5 but is
**slower** than `F.scaled_dot_product_attention` on this hardware. That is
expected, and chasing parity on speed was never the goal — understanding the
algorithm was. The gap is structural:

- **cuDNN / FA2 fusion.** SDPA dispatches to hand-written CUDA C++ backends
  (cuDNN fused attention, or the official FlashAttention-2 kernels) that fuse
  QK^T, the scale, the softmax, and PV into one tightly scheduled kernel with
  optimal HBM↔SRAM movement. Our Triton version expresses the same math at a
  higher level and leans on the compiler for that fusion.
- **No persistent kernels.** The fast backends keep a fixed set of CTAs
  resident and stream many tiles through them, amortising launch cost and
  keeping every SM busy. We launch one program per `(batch, head, M-block)`
  and exit, so for small/medium sizes launch overhead and partly-filled
  "tail" waves dominate.
- **No register-level tiling / pipelining tuning.** Production kernels stage
  operands through registers + shared memory with multi-stage software
  pipelining and tuned warp-level MMA schedules. We use Triton defaults and
  never autotuned `BLOCK_M/BLOCK_N`, `num_warps`, or `num_stages` per
  head-dim/arch.
- **True fp32 by choice.** We set `allow_tf32=False` to hold the engine's
  atol=1e-4 parity contract; SDPA is free to take faster reduced-/mixed-
  precision tensor-core paths.

Takeaway: matching a cuDNN/FA2 kernel needs persistent scheduling plus
autotuned register tiling that a *readable* kernel deliberately omits. The
deliverable is the correctness win and the algorithm walk-through; the speed
gap is the cost of clarity. (The full reasoning also lives at the top of
`src/engine/kernels/flash_attention.py`.)

### 10.2 INT8 GEMM backend: `torch._int_mm`, not a Triton int8 `tl.dot`

The first INT8 implementation used a Triton kernel with `tl.dot` on int8
inputs. It **fails to compile on Turing (T4 / sm_75)**: Triton's
`TritonGPUAccelerateMatmul` pass has no int8 MMA lowering for sm_75, so the
int8 `tl.dot` errors. It would work on Ampere+/Ada, but T4 is the most common
free-tier GPU, so "Ampere-only" is a bad default.

The fix swaps the compute backend to **`torch._int_mm`**, PyTorch's native
int8 matmul, which dispatches to cuBLAS/cuBLASLt int8 IMMA — supported on
sm_75. Everything around it (per-tensor symmetric quantize/dequantize,
`QuantizedLinear`, `QuantizedMultiHeadAttention`, `LlamaModel.quantize()`) is
unchanged. `torch._int_mm` has hard shape constraints on CUDA — **M > 16** and
**K, N multiples of 8** — so:

- **Prefill** (large M = prompt length, K/N multiples of 8 for TinyLlama) takes
  the true int8 path.
- **Decode** (M == 1) violates M > 16, so it falls back to dequantizing the
  still-int8-stored weight and doing an fp matmul. M == 1 is memory-bound
  anyway, so dequant-on-read costs little, and the 4× weight-memory win holds
  on both paths.

This is a recurring lesson with custom kernels: the hardware's supported MMA
shapes/dtypes — not the math — decide the implementation, and the portable,
vendor-tuned primitive often beats the bespoke one once you account for which
GPUs you actually run on.
