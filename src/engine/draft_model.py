"""
Draft-model implementations for true draft/target speculative decoding.

Two concrete DraftModels live here (the protocol + the orchestrator are in
spec_decode.py):

  * SmallModelDraft  -- a separately-loaded, smaller LlamaModel used as the
    draft. Unlike TinyDraftModel (the cache-free wrapper in spec_decode.py),
    this one maintains its OWN paged KV cache per propose() call, so the K
    autoregressive draft steps cost O(S + K) instead of O(S * K). Its cache is
    completely independent of the target's -- the two models never share state.

  * RandomDraftModel -- proposes uniformly-random tokens with uniform
    probabilities. It exists ONLY for testing: it lets us exercise the
    acceptance-rejection math (spec_decode.speculative_sample) and the whole
    SpeculativeDecoder loop on CPU without any real weights, because q_i is the
    trivially-known uniform distribution and the accept ratio reduces to
    min(1, V * p_i(x_i)).

Both satisfy spec_decode.DraftModel: propose(input_ids, k, kv_cache=None) ->
(token_ids (K,), draft_probs (K, V)).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.engine.model import LlamaModel


# ---------------------------------------------------------------------------
# SmallModelDraft -- a separate smaller checkpoint, with its own KV cache.
# ---------------------------------------------------------------------------
#
# In a real deployment the draft is a distilled / pruned model that shares the
# target's tokenizer (e.g. a 1-2 layer head trained to mimic the target). We
# don't ship such weights for TinyLlama, so `from_checkpoint` loads any
# LlamaModel checkpoint and you can also hand it a hand-built small model. The
# acceptance rate is only as good as how well the draft mimics the target --
# that's the whole game in speculative decoding -- but the MECHANISM here is
# the production one: separate model, separate KV cache, K-step cached draft.
# ---------------------------------------------------------------------------


class SmallModelDraft:
    """A smaller LlamaModel as the draft, with an independent paged KV cache."""

    def __init__(self, model: "LlamaModel", temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature
        # Optional reproducibility hook, read by spec_decode._gen via the
        # SpeculativeDecoder (which sets _generator on its draft/target).
        self._generator: torch.Generator | None = None

    @classmethod
    def from_checkpoint(cls, model_path: str, temperature: float = 1.0) -> "SmallModelDraft":
        """Load a separate (ideally smaller) checkpoint as the draft model.

        Uses the same loader as the target. For a genuinely smaller draft you'd
        point `model_path` at a distilled checkpoint; here it accepts any
        LlamaModel-compatible repo so the path is exercised end to end.
        """
        from src.engine.model import load_tinyllama_from_hf

        model, _ = load_tinyllama_from_hf(model_path)
        model.eval()
        return cls(model, temperature=temperature)

    @torch.no_grad()
    def propose(
        self, input_ids: torch.Tensor, k: int, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive K-step draft over the model's OWN paged KV cache.

        We build a fresh one-request cache sized for (S + k), prefill the
        context once, then take k cached decode steps. The cache is the draft's
        alone; the target's verify() never sees it (the `kv_cache` argument here
        is ignored on purpose -- it would be the *target's* cache).
        """
        device = input_ids.device
        s = input_ids.shape[1]
        # The model's own paged-cache helper: a tiny pool sized to this draft's
        # worst-case footprint. Independent of any target cache by construction.
        cache = self.model._make_single_request_cache(prompt_len=s, max_new_tokens=k)  # noqa: SLF001

        # Prefill: absorb the whole context, read logits for the first slot.
        logits = self.model(input_ids, kv_cache=cache)      # (1, S, V)
        last = logits[0, -1, :].to(torch.float32)

        tokens: list[int] = []
        prob_rows: list[torch.Tensor] = []
        for _ in range(k):
            q = F.softmax(last / self.temperature, dim=-1)
            tok = int(torch.multinomial(q, num_samples=1, generator=self._generator))
            tokens.append(tok)
            prob_rows.append(q)
            # Cached decode: feed only the new token; the cache supplies the past.
            nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
            logits = self.model(nxt, kv_cache=cache)         # (1, 1, V)
            last = logits[0, -1, :].to(torch.float32)

        token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        draft_probs = torch.stack(prob_rows, dim=0)          # (K, V)
        return token_ids, draft_probs


# ---------------------------------------------------------------------------
# RandomDraftModel -- testing only.
# ---------------------------------------------------------------------------
#
# Proposes uniform-random tokens with a uniform probability distribution. This
# makes the acceptance-rejection algorithm fully testable WITHOUT real weights:
#   * q_i(x) = 1/V for all x, so q_i(x_i) = 1/V is known exactly.
#   * accept iff r < min(1, p_i(x_i) / (1/V)) = min(1, V * p_i(x_i)).
#   * the residual max(0, p_i - q_i) is just p_i shifted down by 1/V then
#     clamped -- a clean, hand-checkable distribution.
# A peaked target therefore rejects most random proposals (low acceptance), and
# a uniform target accepts ~half -- both easy to assert in a test.
# ---------------------------------------------------------------------------


class RandomDraftModel:
    """Uniformly-random draft proposals with uniform probabilities (tests only)."""

    def __init__(self, vocab_size: int, seed: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.seed = seed
        # Per-device generators, created lazily so a CPU test and a (hypothetical)
        # CUDA test each get a correctly-placed RNG. torch requires the generator
        # to live on the same device as the tensor it drives.
        self._gens: dict[str, torch.Generator] = {}
        # Reproducibility hook for SpeculativeDecoder (unused here -- this model
        # has its own seeded generators -- but present for protocol symmetry).
        self._generator: torch.Generator | None = None

    def _gen_for(self, device: torch.device) -> torch.Generator | None:
        if self.seed is None:
            return None
        key = str(device)
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=device).manual_seed(self.seed)
            self._gens[key] = g
        return g

    @torch.no_grad()
    def propose(
        self, input_ids: torch.Tensor, k: int, kv_cache=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        gen = self._gen_for(device)
        token_ids = torch.randint(
            0, self.vocab_size, (k,), generator=gen, device=device, dtype=torch.long
        )
        # Uniform q over the vocab for every drafted slot.
        draft_probs = torch.full(
            (k, self.vocab_size), 1.0 / self.vocab_size, device=device
        )
        return token_ids, draft_probs
