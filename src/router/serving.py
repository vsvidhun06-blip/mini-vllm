"""
The execution layer behind the router.

MultiModelServer holds the actual LlamaModel instances and runs generation on
whichever model the LLMRouter selects. It is the only file in the router package
that touches torch / the real checkpoint, so it imports lazily and loads models
on first use -- importing this module (e.g. for type hints in api.py) must not
drag a 1.1 B-parameter model into memory.

Two logical models, one checkpoint:

  We don't have two differently-sized checkpoints on hand, so we *simulate* a
  small fast model and a large capable model from the SAME TinyLlama weights by
  giving them different generation parameters:

    "small" -> greedy (temperature 0): deterministic, fast, what you'd run on a
               cheap distilled model for easy queries.
    "large" -> temperature/top-p sampling: a slightly slower, more exploratory
               decode that stands in for a bigger, more capable model.

  This is an honest stand-in: the point of the router exercise is the routing
  decision and the cost/latency accounting, not literally hosting two
  checkpoints. The weights are shared (loaded once) so we don't pay for two
  copies of TinyLlama; only the decode policy differs by name. Swapping in a
  genuinely larger checkpoint for "large" is a one-line change to MODEL_SPECS.
"""
from __future__ import annotations

import time

# Generation parameters per logical model. Both point at the same underlying
# TinyLlama; only the decode behaviour differs. temperature 0.0 == greedy.
MODEL_SPECS: dict[str, dict] = {
    "small": {"temperature": 0.0, "top_p": 1.0},
    "large": {"temperature": 0.7, "top_p": 0.9},
}


class MultiModelServer:
    """Owns the LlamaModel instances and executes router-selected generation."""

    def __init__(self, model_specs: dict[str, dict] | None = None) -> None:
        self.model_specs = model_specs if model_specs is not None else dict(MODEL_SPECS)
        # name -> LlamaModel, populated lazily by _get_model on first request.
        self._models: dict[str, object] = {}
        # The shared underlying checkpoint + tokenizer, loaded once. Both "small"
        # and "large" reference the same weights (see module docstring).
        self._base_model = None
        self._tokenizer = None

    # -- lazy loading ------------------------------------------------------

    def _ensure_base(self) -> None:
        """Load TinyLlama + tokenizer exactly once, on first real request.

        Imports are inside the method so that merely importing serving.py (which
        api.py does at module load) never triggers a multi-GB model load or
        forces torch/transformers onto a caller that only wants type hints.
        """
        if self._base_model is not None:
            return
        from transformers import AutoTokenizer

        from src.engine.model import MODEL_NAME, load_tinyllama_from_hf

        model, _config = load_tinyllama_from_hf(MODEL_NAME)
        model.eval()
        self._base_model = model
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _get_model(self, name: str):
        """Return the LlamaModel for `name`, loading the shared base if needed.

        Both logical models share the one base instance -- the per-name
        difference is the decode policy in MODEL_SPECS, applied at generation
        time, not a separate set of weights.
        """
        if name not in self.model_specs:
            raise KeyError(f"unknown model {name!r}; known: {list(self.model_specs)}")
        if name not in self._models:
            self._ensure_base()
            self._models[name] = self._base_model
        return self._models[name]

    @property
    def loaded_models(self) -> list[str]:
        """Names that have been lazy-loaded so far (for introspection/stats)."""
        return list(self._models)

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int, router) -> tuple[str, str, float]:
        """Route `prompt`, run generation on the chosen model, return outcome.

        Returns:
            (response_text, model_used, latency_ms).

        Side effect: reports the realised latency + token count back to the
        router via record_outcome, so the router's EMA latency and stats stay
        live. The wall-clock timer brackets ONLY the decode work, not routing
        (routing overhead is measured separately in benchmark_router.py).
        """
        config = router.route(prompt)
        model = self._get_model(config.name)
        spec = self.model_specs[config.name]

        input_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.embed.weight.device)

        t0 = time.perf_counter()
        out_ids = self._run(model, input_ids, max_tokens, spec)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Decode only the newly generated tail, not the echoed prompt.
        new_ids = out_ids[input_ids.shape[1]:]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        router.record_outcome(config.name, latency_ms, len(new_ids))
        return response, config.name, latency_ms

    def generate_stream(self, prompt: str, max_tokens: int, router):
        """Token-streaming variant of generate(); a generator of dict events.

        Yields one event per token:
            {"token": str, "token_id": int, "step": int}
        then a final event:
            {"done": True, "response": str, "model_used": str,
             "complexity": str, "latency_ms": float, "cost_weight": float}

        The router decision (and its complexity + cost weight) is taken once up
        front so the SSE consumer can show which model is answering before the
        first token lands -- exactly what a UX wants from a routed endpoint.
        """
        config = router.route(prompt)
        # Re-derive the complexity label for the client. route() already counted
        # this request; we classify again only to report the label, which is
        # cheap (pure-Python) and avoids changing route()'s return contract.
        complexity = router._classify(prompt)
        model = self._get_model(config.name)
        spec = self.model_specs[config.name]

        input_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.embed.weight.device)

        t0 = time.perf_counter()
        pieces: list[str] = []
        n_new = 0
        for step, token_id in enumerate(self._iter_tokens(model, input_ids, max_tokens, spec)):
            piece = self._tokenizer.decode([token_id], skip_special_tokens=True)
            pieces.append(piece)
            n_new += 1
            yield {"token": piece, "token_id": int(token_id), "step": step}
        latency_ms = (time.perf_counter() - t0) * 1000.0

        router.record_outcome(config.name, latency_ms, n_new)
        yield {
            "done": True,
            "response": "".join(pieces),
            "model_used": config.name,
            "complexity": complexity.name,
            "latency_ms": latency_ms,
            "cost_weight": config.cost_per_token,
        }

    # -- decode internals --------------------------------------------------
    #
    # We can't reuse LlamaModel.generate for "large" because it is greedy-only.
    # Instead both paths share _iter_tokens, a simple no-KV-cache decode loop
    # (O(N^2) over the generation, which is fine for the short demo completions
    # the router serves). temperature 0.0 collapses to greedy argmax, so "small"
    # is byte-identical to a greedy decode; "large" samples with temperature/
    # top-p. Keeping one loop means both logical models exercise identical
    # plumbing and differ only in the sampling step.
    # ---------------------------------------------------------------------

    def _run(self, model, input_ids, max_tokens: int, spec: dict):
        """Buffered decode: return the full (prompt + completion) id list."""
        import torch

        generated = input_ids
        for token_id in self._iter_tokens(model, input_ids, max_tokens, spec):
            generated = torch.cat(
                [generated, torch.tensor([[token_id]], device=generated.device)], dim=1
            )
        return generated[0].tolist()

    def _iter_tokens(self, model, input_ids, max_tokens: int, spec: dict):
        """Yield generated token ids one at a time (no KV cache, greedy or sampled)."""
        import torch

        eos = self._tokenizer.eos_token_id
        temperature = spec.get("temperature", 0.0)
        top_p = spec.get("top_p", 1.0)
        ctx = input_ids
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = model(ctx)[:, -1, :]               # (1, vocab)
                token_id = _sample(logits, temperature, top_p)
                yield token_id
                if eos is not None and token_id == eos:
                    break
                ctx = torch.cat(
                    [ctx, torch.tensor([[token_id]], device=ctx.device)], dim=1
                )


def _sample(logits, temperature: float, top_p: float) -> int:
    """Pick a next token id from (1, vocab) logits.

    temperature == 0 -> greedy argmax (deterministic, the "small" model).
    Otherwise        -> temperature scaling + nucleus (top-p) sampling, the
                        more exploratory decode standing in for a larger model.
    """
    import torch

    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    # Temperature scaling then softmax to a probability distribution.
    probs = torch.softmax(logits / temperature, dim=-1).squeeze(0)   # (vocab,)

    # Nucleus filtering: keep the smallest set of tokens whose cumulative
    # probability mass reaches top_p, zero the rest, renormalise, then sample.
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep everything up to and including the first token that crosses top_p.
        keep = cumulative <= top_p
        keep[0] = True                                   # always keep the top-1
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered / filtered.sum()

    return int(torch.multinomial(probs, num_samples=1).item())
