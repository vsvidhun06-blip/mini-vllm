"""
Parity test: my from-scratch LlamaModel vs HF AutoModelForCausalLM.

What we're testing:
    Given the same input_ids, both models must produce the same logits
    to a small absolute tolerance (atol=1e-4) in fp32. If they do, our
    implementation of RMSNorm, RoPE, GQA, SwiGLU, residual structure,
    and weight loading is correct end-to-end.

How we're testing:
    1. Skip if the HF checkpoint isn't already cached locally. We don't
       want pytest to silently pull 2GB on a fresh clone.
    2. Build both models in fp32 (HF defaults to fp16/bf16; we force fp32
       so numeric drift can't hide a real bug).
    3. Feed both the same input_ids.
    4. Compare full logits (every position, every vocab entry) AND the
       greedy argmax of the last position.

If the test fails, the error message dumps max-abs-diff so we can localize
which component drifted.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.model import MODEL_NAME, load_tinyllama_from_hf


def _checkpoint_is_cached(model_name: str) -> bool:
    """Return True iff config.json and model.safetensors are in the HF cache.

    We only check existence -- not freshness -- because we want the test
    suite to be runnable offline once the user has done a single warm run.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    config_path = try_to_load_from_cache(model_name, "config.json")
    weights_path = try_to_load_from_cache(model_name, "model.safetensors")
    return bool(config_path) and bool(weights_path)


@pytest.fixture(scope="module")
def cached_or_skip() -> None:
    """Skip the whole module if TinyLlama isn't cached."""
    if not _checkpoint_is_cached(MODEL_NAME):
        pytest.skip(
            f"{MODEL_NAME} not in HF cache. "
            "Run `python -m src.engine.model` once to download it, "
            "then re-run pytest.",
            allow_module_level=False,
        )


@pytest.fixture(scope="module")
def models(cached_or_skip):
    """Load both models once and share across tests in this module.

    Loading TinyLlama-1.1B in fp32 burns ~4.4 GB of RAM per copy, so we
    really do not want to do this twice.
    """
    from transformers import AutoModelForCausalLM

    # Reference: HF's own implementation. Force fp32 + eval to match ours.
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    hf_model.eval()

    # Ours.
    my_model, _config = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    my_model.eval()

    return hf_model, my_model


def test_logits_match(models) -> None:
    """Full-logit parity across the whole sequence."""
    hf_model, my_model = models

    # A short, deterministic prompt. Short keeps the test fast; deterministic
    # makes failures reproducible.
    # Using arbitrary token ids (not from a tokenizer) so this test has zero
    # dependence on tokenizer behavior -- it isolates the model code.
    input_ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], dtype=torch.long)

    with torch.no_grad():
        hf_out = hf_model(input_ids).logits  # (1, S, V)
        my_out = my_model(input_ids)         # (1, S, V)

    assert hf_out.shape == my_out.shape, f"shape mismatch: {hf_out.shape} vs {my_out.shape}"

    max_abs_diff = (hf_out - my_out).abs().max().item()
    # 1e-4 is tight but achievable in fp32 across 22 layers when every
    # numeric choice (RoPE layout, RMSNorm dtype, GQA expansion order) is
    # correct. If this fails it's almost certainly a real bug, not noise.
    assert torch.allclose(hf_out, my_out, atol=1e-4), (
        f"Logits diverged. max |hf - mine| = {max_abs_diff:.3e}. "
        f"atol budget = 1e-4. Likely culprits: RoPE layout, RMSNorm dtype "
        f"handling, or GQA repeat order."
    )


def test_greedy_argmax_matches(models) -> None:
    """Greedy next-token id matches at the last position.

    A weaker but very legible check: even if some tiny logit drift exists,
    the argmax of the last position must agree -- this is what generation
    actually consumes.
    """
    hf_model, my_model = models

    input_ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], dtype=torch.long)

    with torch.no_grad():
        hf_next = int(torch.argmax(hf_model(input_ids).logits[0, -1, :]))
        my_next = int(torch.argmax(my_model(input_ids)[0, -1, :]))

    assert hf_next == my_next, f"Greedy mismatch: HF={hf_next}, mine={my_next}"
