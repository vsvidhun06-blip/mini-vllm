"""
Benchmark: no speculation vs self-speculation vs draft-model speculation.

Run:
    python scripts/benchmark_spec_decode.py

This benchmark is SELF-CONTAINED and fast: it builds random-weight models on
whatever DEVICE the engine picks (CPU is fine), so it needs no TinyLlama
download. Three decode strategies, each over k in {1, 2, 4, 8}:

  * none        -- plain autoregressive sampling from the target (the baseline;
                   k is irrelevant, reported once).
  * self-spec   -- SpeculativeDecoder with SelfSpecDraftModel: the target run
                   shallow (early exit) as its own draft. No second model.
  * draft-model -- SpeculativeDecoder with TinyDraftModel: a separate 2-layer
                   model proposes, the full target verifies.

Metrics per config: tokens/sec, mean acceptance rate, TPOT (ms/token).

HONESTY UP FRONT
----------------
The weights here are RANDOM, so the draft does not actually mimic the target --
acceptance rates reflect the MECHANISM and its overhead, not the speedups a
trained draft delivers. On real weights, net speedup needs the mean acceptance
rate alpha to clear the draft's relative cost: roughly alpha > c_draft/c_target.
A 2-layer draft against a 12-layer target needs alpha > ~1/6 to win; an
early-exit-at-half draft needs alpha > ~1/2. This harness measures alpha and the
per-token wall-clock so you can see where each strategy lands; it is not a claim
that random-weight speculation is faster.
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE
from src.engine.model import LlamaConfig, LlamaModel
from src.engine.spec_decode import (
    FullModelTarget,
    SelfSpecDraftModel,
    SpeculativeDecoder,
    TinyDraftModel,
)

VOCAB = 512
PROMPT_LEN = 16
NEW_TOKENS = 64
K_VALUES = (1, 2, 4, 8)


def _build_models() -> tuple[LlamaModel, LlamaModel]:
    """A 12-layer 'target' and a 2-layer 'draft', same vocab/dims so the draft's
    proposals are valid target inputs."""
    torch.manual_seed(0)
    common = dict(
        vocab_size=VOCAB, hidden_size=256, intermediate_size=512,
        num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=512, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=False,
    )
    target = LlamaModel(LlamaConfig(num_hidden_layers=12, **common)).eval().to(DEVICE)
    draft = LlamaModel(LlamaConfig(num_hidden_layers=2, **common)).eval().to(DEVICE)
    return target, draft


@torch.no_grad()
def _bench_decoder(dec: SpeculativeDecoder, ids: torch.Tensor, n_new: int) -> dict:
    """Run a SpeculativeDecoder until >= n_new tokens are emitted; time it."""
    is_cuda = ids.is_cuda
    ctx = ids
    emitted = 0
    token_times: list[float] = []
    t0 = time.perf_counter()
    while emitted < n_new:
        toks = dec.decode_step(ctx)
        ctx = torch.cat([ctx, torch.tensor([toks], dtype=torch.long, device=ctx.device)], dim=1)
        emitted += len(toks)
        if is_cuda:
            torch.cuda.synchronize()
        token_times.append(time.perf_counter())
    total = token_times[-1] - t0
    # TPOT: mean gap between successive decode_step completions, in ms.
    if len(token_times) > 1:
        gaps = [(b - a) * 1e3 for a, b in zip(token_times, token_times[1:])]
        tpot = sum(gaps) / len(gaps)
    else:
        tpot = total * 1e3
    return {
        "tok_per_s": emitted / total if total else float("nan"),
        "tpot_ms": tpot,
        "accept": dec.mean_acceptance_rate,
    }


@torch.no_grad()
def _bench_no_spec(target: LlamaModel, ids: torch.Tensor, n_new: int) -> dict:
    """Plain autoregressive sampling baseline (no draft, no verify)."""
    is_cuda = ids.is_cuda
    ctx = ids
    gen = torch.Generator(device=ctx.device).manual_seed(123)
    t0 = time.perf_counter()
    last_t = t0
    gaps: list[float] = []
    for _ in range(n_new):
        logits = target(ctx)
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=gen))
        ctx = torch.cat([ctx, torch.tensor([[tok]], dtype=torch.long, device=ctx.device)], dim=1)
        if is_cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        gaps.append((now - last_t) * 1e3)
        last_t = now
    total = last_t - t0
    return {
        "tok_per_s": n_new / total if total else float("nan"),
        "tpot_ms": sum(gaps) / len(gaps),
        "accept": float("nan"),
    }


def main() -> None:
    target, draft = _build_models()
    g = torch.Generator().manual_seed(2024)
    ids = torch.randint(0, VOCAB, (1, PROMPT_LEN), generator=g).to(DEVICE)

    print(f"Device: {DEVICE}")
    print(f"Target: 12 layers, Draft: 2 layers, vocab={VOCAB}")
    print(f"Prompt {PROMPT_LEN} tokens, generating ~{NEW_TOKENS} (sampled)\n")

    rows: list[tuple[str, str, dict]] = []

    # Baseline: no speculation (k not applicable).
    rows.append(("none", "-", _bench_no_spec(target, ids, NEW_TOKENS)))

    # Self-spec and draft-model spec across k.
    for k in K_VALUES:
        self_dec = SpeculativeDecoder(
            SelfSpecDraftModel(target, n_layers=6),   # exit at half depth
            FullModelTarget(target), k=k,
            generator=torch.Generator().manual_seed(1),
        )
        rows.append(("self-spec", str(k), _bench_decoder(self_dec, ids, NEW_TOKENS)))

        draft_dec = SpeculativeDecoder(
            TinyDraftModel(draft),
            FullModelTarget(target), k=k,
            generator=torch.Generator().manual_seed(1),
        )
        rows.append(("draft-model", str(k), _bench_decoder(draft_dec, ids, NEW_TOKENS)))

    header = f"{'strategy':>12} | {'k':>2} | {'tok/s':>8} | {'TPOT (ms)':>10} | {'accept':>7}"
    print(header)
    print("-" * len(header))
    for name, k, m in rows:
        acc = "  n/a" if m["accept"] != m["accept"] else f"{m['accept']:6.1%}"  # NaN check
        print(f"{name:>12} | {k:>2} | {m['tok_per_s']:>8.1f} | {m['tpot_ms']:>10.2f} | {acc:>7}")

    print("\nWeights are RANDOM: acceptance reflects mechanism overhead, not the")
    print("speedup a trained draft gives. Net win needs mean acceptance alpha to")
    print("beat the draft's relative cost (~1/6 for 2-of-12 layers, ~1/2 for")
    print("early-exit-at-half). See the module docstring.")


if __name__ == "__main__":
    main()
