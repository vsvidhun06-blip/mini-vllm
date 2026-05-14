"""
Prefix-cache benchmark.

Setup: 4 requests, all sharing a ~100-token "system prompt" prefix
followed by a ~30-token unique user query. Without prefix caching,
every request prefills the system prompt from scratch. With prefix
caching, request 0 computes it once, registers the block hashes; the
other 3 each get cache hits on the system-prompt blocks and only
prefill the user query.

What we measure:
    * Per-request prefill wall time (captured via prefill_started /
      prefill_done timestamps on the event bus -- the bracket is the
      model.forward() call plus its argmax).
    * Aggregate end-to-end time (admit-to-finish).
    * Per-request cached_blocks / total_prefill_blocks ratio.

Why this isn't a pytest:
    Benchmarks are noisy enough that the natural reporting shape is
    "print numbers and let the reader judge"; pass/fail isn't useful
    here. Parity is asserted by tests/test_engine/test_prefix_cache_parity.py.

Run:
    python scripts/bench_prefix_cache.py
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE, DTYPE
from src.engine.events import EventBus
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler


# A system prompt + user query. The system prompt is intentionally long
# enough to fill several block_size=16 blocks; tokenisation gives us
# roughly one token per word on Llama-family models so ~100 tokens lands
# around the 600-character mark.
SYSTEM_PROMPT = (
    "You are a helpful, careful, and concise assistant. You answer "
    "questions using only the information provided to you. You do not "
    "make up facts, and if you don't know something, you say so. When "
    "the user's question is ambiguous, you ask for clarification before "
    "answering. When you give code, you keep the code minimal, "
    "well-commented, and consistent with the surrounding style. You "
    "prefer clear, direct prose over jargon. You never repeat yourself. "
    "You are running on a small open-weights model and should be aware "
    "that long-form answers are expensive; keep responses tight."
)
USER_QUERIES = [
    "What's the capital of France, and what river runs through it?",
    "Write a haiku about the smell of rain on hot pavement in summer.",
    "Explain why the sky appears blue in two sentences.",
    "List three Python standard library modules useful for parsing CSVs.",
]
MAX_NEW = 16
NUM_BLOCKS = 96  # plenty of room for 4 reqs * (~130 + 16) tokens at bs=16


def _build_requests(tokenizer) -> list[tuple[str, torch.Tensor]]:
    """Build (request_id, prompt_ids) pairs.

    We tokenise SYSTEM_PROMPT + USER_QUERY end-to-end. The system-prompt
    PREFIX of the resulting token ids is identical across the 4 prompts
    (tokenisation of a stable string is deterministic). That's what
    makes prefix-cache sharing possible: the first ceil(len(system)/16)
    full blocks of each prompt's token-id list contain the exact same
    ids.
    """
    out: list[tuple[str, torch.Tensor]] = []
    for i, q in enumerate(USER_QUERIES):
        text = SYSTEM_PROMPT + "\n\n" + q
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        out.append((f"req-{i}", ids))
    return out


def _run(
    model,
    tokenizer,
    *,
    enable_prefix_cache: bool,
) -> dict:
    """Drive a fresh scheduler over the 4 prompts, capture prefill timings.

    We subscribe to the event bus for prefill_started / prefill_done /
    request_admitted. The bus's timestamps bracket the model.forward()
    call inside the scheduler's prefill phase, so subtracting gives
    per-request prefill wall time.
    """
    bus = EventBus()

    started: dict[str, float] = {}
    done: dict[str, float] = {}
    admit_meta: dict[str, tuple[int, int]] = {}

    def on_event(ev) -> None:
        if ev.event_type == "prefill_started":
            started[ev.payload["request_id"]] = ev.timestamp
        elif ev.event_type == "prefill_done":
            done[ev.payload["request_id"]] = ev.timestamp
        elif ev.event_type == "request_admitted":
            admit_meta[ev.payload["request_id"]] = (
                ev.payload["cached_blocks"],
                ev.payload["total_prefill_blocks"],
            )

    bus.subscribe(on_event)

    scheduler = ContinuousBatchScheduler(
        model,
        max_batch_size=4,
        num_blocks=NUM_BLOCKS,
        block_size=16,
        event_bus=bus,
        enable_prefix_cache=enable_prefix_cache,
    )

    eos = tokenizer.eos_token_id
    for rid, ids in _build_requests(tokenizer):
        scheduler.add_request(
            request_id=rid,
            prompt_ids=ids,
            max_new_tokens=MAX_NEW,
            eos_token_id=eos,
        )

    t0 = time.perf_counter()
    while scheduler.has_work():
        scheduler.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    total_wall = time.perf_counter() - t0

    return {
        "total_wall": total_wall,
        "per_request_prefill_s": {
            rid: done[rid] - started[rid] for rid in done
        },
        "admit_meta": admit_meta,
    }


def main() -> None:
    from transformers import AutoTokenizer

    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("  TF32 matmul: off (preserve fp32 parity)")
    print()

    print("Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=DTYPE)
    model.eval()

    # Report the prompt-length geometry once.
    sample_ids = tokenizer(
        SYSTEM_PROMPT + "\n\n" + USER_QUERIES[0], return_tensors="pt"
    )["input_ids"]
    sys_only = tokenizer(SYSTEM_PROMPT, return_tensors="pt")["input_ids"]
    print(f"System-prompt length: {sys_only.shape[1]} tokens")
    print(f"Full prompt length (with first user query): {sample_ids.shape[1]} tokens")
    print(f"Block size: 16. Full blocks in system prompt: {sys_only.shape[1] // 16}")
    print()

    # Warmup. Two passes:
    #   1. Solo generate -- warms the cached-path attention kernel
    #      (S_q == S_k, is_causal=True path).
    #   2. A throwaway prefix-cache run -- warms the SLICED-prefill
    #      attention kernel (S_q < S_k, explicit attn_mask path). The
    #      first sliced-prefill in a process pays a noticeable CUDA
    #      kernel JIT cost; without this warmup the first cached request
    #      in the timed run gets attributed an artifact-y prefill time.
    print("Warmup (solo)...")
    _ = model.generate(
        tokenizer("warmup", return_tensors="pt")["input_ids"],
        max_new_tokens=4,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup (sliced prefill)...")
    _ = _run(model, tokenizer, enable_prefix_cache=True)
    print()

    # ---- Cache OFF ---------------------------------------------------
    print("=== Prefix cache OFF ===")
    off = _run(model, tokenizer, enable_prefix_cache=False)
    for rid in sorted(off["per_request_prefill_s"]):
        cached, total = off["admit_meta"][rid]
        print(
            f"  {rid}: prefill={off['per_request_prefill_s'][rid]*1000:7.1f} ms  "
            f"cached_blocks={cached}/{total}"
        )
    print(f"  total wall time: {off['total_wall']:.2f}s")
    print()

    # ---- Cache ON ----------------------------------------------------
    print("=== Prefix cache ON ===")
    on = _run(model, tokenizer, enable_prefix_cache=True)
    for rid in sorted(on["per_request_prefill_s"]):
        cached, total = on["admit_meta"][rid]
        print(
            f"  {rid}: prefill={on['per_request_prefill_s'][rid]*1000:7.1f} ms  "
            f"cached_blocks={cached}/{total}"
        )
    print(f"  total wall time: {on['total_wall']:.2f}s")
    print()

    # ---- Comparison --------------------------------------------------
    print("=== Per-request prefill speedup (cache OFF / ON) ===")
    for rid in sorted(off["per_request_prefill_s"]):
        a = off["per_request_prefill_s"][rid]
        b = on["per_request_prefill_s"][rid]
        if b > 0:
            print(f"  {rid}: {a*1000:7.1f} ms -> {b*1000:7.1f} ms  ({a/b:5.2f}x)")
        else:
            print(f"  {rid}: {a*1000:7.1f} ms -> {b*1000:7.1f} ms")
    print()

    # Aggregate hit-rate across requests.
    total_cached = sum(c for c, _ in on["admit_meta"].values())
    total_blocks = sum(t for _, t in on["admit_meta"].values())
    rate = total_cached / total_blocks if total_blocks > 0 else 0.0
    print(
        f"Aggregate prefix-cache hit rate: "
        f"{total_cached}/{total_blocks} prefill blocks ({rate*100:.1f}%)"
    )
    print(f"End-to-end wall: OFF={off['total_wall']:.2f}s  ON={on['total_wall']:.2f}s")


if __name__ == "__main__":
    main()
