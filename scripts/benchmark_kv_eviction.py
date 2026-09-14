"""
Benchmark: H2O KV-cache eviction for long-context generation.

Run:
    python scripts/benchmark_kv_eviction.py

Needs TinyLlama in the HF cache (run `python -m src.engine.model` once). Runs on
whatever DEVICE the engine picks (GPU if present, else CPU -- slower but fine;
this is a memory/quality demo, not a speed one).

THE SCENARIO
------------
A 1024-token context with a cache budget of only 512 tokens.

  * Standard cache (budget 512): runs out of blocks at 512 tokens. It physically
    cannot hold the full context -- generation past the budget fails.
  * Evicting cache (budget 512): each time the cache fills it drops the cold
    (low-attention) tokens, keeping the heavy hitters + a recency window, and
    keeps going all the way to 1024 -- 2x the budget.

We teacher-force the SAME reference token stream through each policy and report
the perplexity of next-token prediction, so quality is directly comparable. The
"full cache" reference (a cache large enough for all 1024 tokens) is the
no-eviction baseline; the eviction perplexity should be only slightly higher --
that small gap is the price of the 2x context.

Table: config | tokens generated | cache hits | evictions | perplexity
       ("cache hits" = tokens still resident in the cache at the end.)
"""
from __future__ import annotations

import torch

from src.engine.device import DEVICE
from src.engine.kv_cache import PagedKVCache, PagedRequestCache
from src.engine.kv_eviction import run_sequence_with_eviction
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf

SEQ_LEN = 1024
BUDGET = 512
RECENT_WINDOW = 32


def _make_tokens(tokenizer, length: int) -> torch.Tensor:
    """A length-`length` token sequence built by tiling a base passage."""
    base = tokenizer(
        "In a distant land beyond the mountains there lived a clever fox who "
        "loved to solve riddles and tell long stories by the fire. ",
        return_tensors="pt",
    )["input_ids"][0]
    reps = (length + len(base) - 1) // len(base)
    return base.repeat(reps)[:length].reshape(1, -1)


def _build_plain_cache(model, capacity_blocks):
    cfg = model.config
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_blocks=capacity_blocks, block_size=16,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=next(model.parameters()).dtype, device=next(model.parameters()).device,
    )
    pool.admit_request("r", prefill_blocks_needed=0, total_blocks_needed=capacity_blocks)
    return PagedRequestCache(pool, "r", num_layers=cfg.num_hidden_layers)


@torch.no_grad()
def _teacher_force_plain(model, ids, capacity_blocks):
    """Teacher-force `ids` through a plain (non-evicting) cache of the given
    block budget. Returns (tokens_processed, resident_tokens, perplexity). Stops
    cleanly when the cache runs out of blocks."""
    ids = ids.to(DEVICE)
    T = ids.shape[1]
    cache = _build_plain_cache(model, capacity_blocks)
    nll_sum, nll_count, processed = 0.0, 0, 0
    for t in range(T):
        try:
            logits = model(ids[:, t:t + 1], kv_cache=cache)
        except RuntimeError:
            # Out of blocks: the standard cache cannot hold this much context.
            break
        processed += 1
        if t + 1 < T:
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            nll_sum += -logp[int(ids[0, t + 1])].item()
            nll_count += 1
    ppl = float(torch.tensor(nll_sum / max(1, nll_count)).exp())
    return processed, cache.seq_len(), ppl


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()

    ids = _make_tokens(tokenizer, SEQ_LEN)
    print(f"Device: {DEVICE}")
    print(f"Context: {SEQ_LEN} tokens, cache budget: {BUDGET} tokens "
          f"(recency window {RECENT_WINDOW})\n")

    # Reference: a cache big enough for the whole context (no eviction).
    full_blocks = (SEQ_LEN + 15) // 16
    full_tokens, full_resident, full_ppl = _teacher_force_plain(model, ids, full_blocks)

    # Standard cache capped at the budget: it cannot reach the full context.
    budget_blocks = (BUDGET + 15) // 16
    std_tokens, std_resident, std_ppl = _teacher_force_plain(model, ids, budget_blocks)

    # Evicting cache at the same budget: continues to the full context.
    ev = run_sequence_with_eviction(
        model, ids, capacity_tokens=BUDGET, recent_window=RECENT_WINDOW,
    )

    rows = [
        ("full cache (ref)",  full_tokens, full_resident,     0,                 full_ppl),
        ("standard (512)",    std_tokens,  std_resident,      0,                 std_ppl),
        ("evicting (512)",    ev["tokens"], ev["resident_tokens"], ev["num_evictions"], ev["perplexity"]),
    ]

    header = (f"{'config':>18} | {'tokens':>7} | {'cache hits':>10} | "
              f"{'evictions':>9} | {'perplexity':>10}")
    print(header)
    print("-" * len(header))
    for name, toks, hits, ev_n, ppl in rows:
        print(f"{name:>18} | {toks:>7} | {hits:>10} | {ev_n:>9} | {ppl:>10.3f}")

    if std_tokens < SEQ_LEN <= ev["tokens"]:
        print(f"\nThe standard cache stalls at {std_tokens} tokens (out of blocks); "
              f"the evicting cache reaches all {ev['tokens']}.")
    degradation = (ev["perplexity"] / full_ppl - 1.0) * 100.0 if full_ppl else float("nan")
    print(f"Eviction perplexity vs full-cache reference: {degradation:+.1f}% "
          f"(small is the H2O promise -- 2x context for a little quality).")


if __name__ == "__main__":
    main()
