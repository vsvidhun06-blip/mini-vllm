# Mini-vLLM — from-scratch LLM inference engine

> **v0.2: GPU + streaming + prefix caching + metrics shipped.** v0.3 (speculative decoding) in progress.

An educational, from-scratch reimplementation of vLLM's core ideas on
TinyLlama-1.1B-Chat. Continuous batching, paged KV cache (PagedAttention),
and a live WebSocket visualiser of every scheduler decision and cache
block. Built to understand modern LLM inference internals at the
production-engineering level — every line written by hand, with Hugging
Face parity tests as the correctness anchor at each layer of the stack.

## Hero

![Mini-vLLM live visualiser](docs/screenshots/hero_v2.png)

*Four concurrent requests in the DECODE phase. Each colour is one
request_id, hashed deterministically so the same hue appears in the
status row, the cache grid, and the event log. 9 of 64 KV blocks in
use; ~12.6 tokens/sec on CPU.*

## v0.1 benchmarks

TinyLlama-1.1B-Chat, fp32, CPU. Four concurrent requests, 32 tokens
each. Speedups measured against a sequential `generate()` loop over the
same prompts.

| Configuration | Time | Speedup | Notes |
|---|---|---|---|
| Solo (sequential) | 16.32s | 1.00x | baseline |
| Continuous batching, ample blocks | 5.26s | 3.10x | scheduler parity test |
| + Paged KV cache, ample blocks | 5.35s | 3.27x | paged overhead amortised |
| Paged + tight blocks (6) | 8.78s | 1.79x | admission control under back-pressure |

The fourth row is the interesting one: with only 6 cache blocks the
scheduler must queue some requests until earlier ones free theirs.
Total wall time goes up but correctness is preserved
(`test_paged_scheduler_with_tight_blocks_still_parity`).

## Architecture

```
    Prompt -> FastAPI -> ContinuousBatchScheduler
                              |
                              +-- LlamaModel (from-scratch)
                              |     |
                              |     +-- RoPE + GQA Attention
                              |     +-- SwiGLU MLP
                              |     +-- RMSNorm (fp32)
                              |
                              +-- PagedKVCache (block_size=16)
                              |
                              +-- EventBus -> WebSocket -> Visualiser
```

## Key technical decisions

See [`docs/design.md`](docs/design.md) for the long version, and
[`docs/talking_points.md`](docs/talking_points.md) for the interview
cheat sheet.

- **HF parity as correctness anchor.** Every engine layer has a parity
  test against `transformers` at `atol=1e-4` — forward pass, cached
  generation, paged-cache generation, scheduler.
- **Per-layer KV cache position handling.** Each layer reads its own
  cache length before appending, not a shared counter. See commit
  `ea0c1b4` for the off-by-one war story.
- **Mixed prefill/decode batching.** Sequential prefill (one forward
  per admitted request that turn) plus a single batched decode pass
  over all DECODE requests. Steady-state is decode-only.
- **Paged cache layout: split K/V pools, layer-major, block_size
  before num_kv_heads.** SDPA receives views, not strided copies. See
  the module docstring in `src/engine/kv_cache.py`.
- **Sync EventBus + `loop.call_soon_threadsafe` bridge.** Scheduler
  emits events synchronously from worker threads; each WebSocket
  subscriber owns an `asyncio.Queue`. Bus never knows about asyncio.

## How to run

Prerequisites: Python 3.11, ~2.5 GB disk for TinyLlama weights on
first run, ~4 GB RAM during inference.

```bash
git clone https://github.com/vsvidhun06-blip/mini-vllm.git
cd mini-vllm
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tests:

```bash
pytest tests/ -v
```

Expected: **10 passed**. The two server tests skip if TinyLlama isn't
cached locally; the eight engine tests run unconditionally and
exercise HF parity at every layer.

Demo:

```bash
uvicorn src.server.api:app        # http://localhost:8000
```

Open the URL, then send `POST /generate` requests (e.g. from DevTools)
and watch the visualiser drive WAITING -> PREFILL -> DECODE -> DONE
with the cache grid lighting up per request.

## Repo structure

```
mini-vllm/
├── src/
│   ├── engine/
│   │   ├── attention.py        # RoPE + GQA + SDPA
│   │   ├── events.py           # Event dataclass + sync EventBus
│   │   ├── kv_cache.py         # PagedKVCache + PagedRequestCache
│   │   ├── model.py            # RMSNorm, SwiGLU, LlamaModel
│   │   └── scheduler.py        # ContinuousBatchScheduler
│   ├── server/
│   │   └── api.py              # FastAPI /generate + WS /events
│   └── visualiser/
│       └── index.html          # single-file SPA, served at GET /
├── tests/
│   ├── test_engine/            # 8 HF parity tests
│   └── test_server/            # 2 server / visualiser tests
├── docs/
│   ├── design.md
│   ├── talking_points.md
│   └── screenshots/hero.png
└── requirements.txt
```

## Author

Vidhun Vijayakumar Suja
MSc Software Engineering, Heriot-Watt Edinburgh (graduating June 2026)
Dissertation: *WEAKEST Execution Visualiser* — abstract submitted to CONCUR 2026

- GitHub: <https://github.com/vsvidhun06-blip>
- Portfolio: <https://vsvidhun06-blip.github.io/>
