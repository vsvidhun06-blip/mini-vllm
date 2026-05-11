"""
Generation parity test: my greedy generate() vs HF's generate().

What we're testing:
    Token-by-token equality between our naive greedy decode and HF's
    `model.generate(do_sample=False, use_cache=False, ...)`. Both are
    deterministic argmax decoding over the same model weights, so they
    must produce identical sequences.

Why use_cache=False on the HF side:
    Our Day 4 implementation has no KV cache -- we recompute the full
    forward each step. To compare apples to apples, we ask HF to do the
    same. (HF's cached and uncached decode produce the same tokens in
    fp32, but disabling cache removes one variable from the diff.)

Skip behavior:
    Reuses the same "skip if not cached" logic as test_model_parity --
    fresh clones don't surprise-download 2GB.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.model import MODEL_NAME, load_tinyllama_from_hf


def _checkpoint_is_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    config_path = try_to_load_from_cache(model_name, "config.json")
    weights_path = try_to_load_from_cache(model_name, "model.safetensors")
    return bool(config_path) and bool(weights_path)


@pytest.fixture(scope="module")
def cached_or_skip() -> None:
    if not _checkpoint_is_cached(MODEL_NAME):
        pytest.skip(
            f"{MODEL_NAME} not in HF cache. "
            "Run `python -m src.engine.model` once to download it, "
            "then re-run pytest."
        )


@pytest.fixture(scope="module")
def models(cached_or_skip):
    """Load both models in fp32 once for the module."""
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    hf_model.eval()

    my_model, _config = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    my_model.eval()

    return hf_model, my_model


def test_greedy_generation_matches_hf(models) -> None:
    """Generate 30 tokens with both, assert exact token sequence equality."""
    hf_model, my_model = models

    # Hardcoded ids for "The capital of France is" under TinyLlama's BPE.
    # Hardcoding lets the test isolate the model + generation logic and
    # not depend on the tokenizer being loaded.
    input_ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], dtype=torch.long)

    # TinyLlama's EOS is id=2 ("</s>"). Pass it to both so any early stop
    # happens at the same place. If EOS doesn't appear in 30 tokens (likely
    # for this prompt), both just run the full budget.
    eos_id = 2
    max_new = 30

    hf_out = hf_model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=False,
        eos_token_id=eos_id,
        pad_token_id=eos_id,        # silences a "no pad token" warning
        repetition_penalty=1.0,     # explicit: no logit reweighting
        temperature=1.0,            # explicit: irrelevant with do_sample=False
    )

    my_out = my_model.generate(
        input_ids,
        max_new_tokens=max_new,
        eos_token_id=eos_id,
    )

    assert hf_out.shape == my_out.shape, (
        f"Sequence length mismatch: HF={tuple(hf_out.shape)} "
        f"vs mine={tuple(my_out.shape)}. One stopped early and the other "
        f"didn't, which means the first-diverging token differed."
    )

    # Token-by-token equality. We compare the full sequence (prompt +
    # generated) so any difference -- including a bug that corrupts the
    # echoed prompt -- shows up.
    if not torch.equal(hf_out, my_out):
        # Find the first divergence to produce a useful failure message.
        diff_mask = (hf_out != my_out)[0]
        first_diff = int(torch.argmax(diff_mask.long()))
        raise AssertionError(
            f"Generated tokens differ.\n"
            f"  First mismatch at position {first_diff} "
            f"(prompt length = {input_ids.shape[1]}).\n"
            f"  HF:   {hf_out[0].tolist()}\n"
            f"  Mine: {my_out[0].tolist()}"
        )
