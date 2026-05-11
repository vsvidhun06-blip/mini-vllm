"""
KV cache parity tests.

Two distinct things to verify, both of which can hide a real bug:

  1. Internal parity: my cached path vs my naive path. The cache is an
     OPTIMIZATION -- it must not change outputs. If these diverge, the
     KV-cache code itself has a bug (most likely RoPE position offset,
     causal mask flip, or post-rotation caching).

  2. External parity: my cached path vs HF's cached path. Catches the
     case where mine and naive both agree but both disagree with HF.

We push max_new_tokens to 50 to give RoPE position drift time to compound
if there's an off-by-one. A single-token divergence near the start would
cascade into completely different sequences by token 50.
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
            "Run `python -m src.engine.model` once to download it."
        )


@pytest.fixture(scope="module")
def my_model(cached_or_skip):
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_model(cached_or_skip):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model


# A short, deterministic prompt. Hardcoded ids -- no tokenizer dependency.
# Corresponds to "The capital of France is" under TinyLlama's BPE.
PROMPT_IDS = torch.tensor([[1, 450, 7483, 310, 3444, 338]], dtype=torch.long)
EOS_ID = 2
MAX_NEW = 50


def _first_diverging_position(a: torch.Tensor, b: torch.Tensor) -> int:
    """Index of the first column where rows differ. Helper for failure messages."""
    diff = (a != b)[0]
    return int(torch.argmax(diff.long()))


def test_cached_matches_naive(my_model) -> None:
    """The cache must not change my model's outputs vs naive recompute.

    If this fails, the cache code has a bug. Likely suspects, in order:
      1. RoPE position offset not using cache.seq_len()
      2. is_causal not flipped between prefill and decode
      3. K cached before RoPE instead of after
    """
    with_cache    = my_model.generate(PROMPT_IDS, max_new_tokens=MAX_NEW,
                                       eos_token_id=EOS_ID, use_cache=True)
    without_cache = my_model.generate(PROMPT_IDS, max_new_tokens=MAX_NEW,
                                       eos_token_id=EOS_ID, use_cache=False)

    assert with_cache.shape == without_cache.shape, (
        f"Shape mismatch: cached={tuple(with_cache.shape)} "
        f"vs naive={tuple(without_cache.shape)}. One stopped early."
    )
    if not torch.equal(with_cache, without_cache):
        first_diff = _first_diverging_position(with_cache, without_cache)
        raise AssertionError(
            f"Cached path diverges from naive path at position {first_diff} "
            f"(prompt length = {PROMPT_IDS.shape[1]}). "
            f"This is a bug in the cache, not in the model.\n"
            f"  cached: {with_cache[0].tolist()}\n"
            f"  naive:  {without_cache[0].tolist()}"
        )


def test_cached_matches_hf(my_model, hf_model) -> None:
    """My cached generate vs HF cached generate, same prompt, 50 tokens."""
    my_out = my_model.generate(PROMPT_IDS, max_new_tokens=MAX_NEW,
                                eos_token_id=EOS_ID, use_cache=True)

    hf_out = hf_model.generate(
        PROMPT_IDS,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        use_cache=True,          # HF cached decode
        eos_token_id=EOS_ID,
        pad_token_id=EOS_ID,
        repetition_penalty=1.0,
        temperature=1.0,
    )

    assert my_out.shape == hf_out.shape, (
        f"Shape mismatch: mine={tuple(my_out.shape)} vs HF={tuple(hf_out.shape)}."
    )
    if not torch.equal(my_out, hf_out):
        first_diff = _first_diverging_position(my_out, hf_out)
        raise AssertionError(
            f"My cached output diverges from HF cached at position {first_diff} "
            f"(prompt length = {PROMPT_IDS.shape[1]}).\n"
            f"  mine: {my_out[0].tolist()}\n"
            f"  HF:   {hf_out[0].tolist()}"
        )
