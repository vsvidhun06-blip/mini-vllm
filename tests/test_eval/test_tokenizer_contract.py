"""Regression: `load_tinyllama_from_hf` returns a CONFIG, not a tokenizer.

THE BUG. `scripts/eval/repair/batch_intervention.py` did:

    model, tokenizer = load_tinyllama_from_hf()

but the loader's signature is `-> tuple[LlamaModel, LlamaConfig]`. It uses only
huggingface_hub + safetensors and never builds a tokenizer. So `tokenizer` was
bound to a LlamaConfig, and the first call into `src.carl.live._make_prompt`,
which does `tokenizer("The quick brown fox ...")`, died on the T4 with

    TypeError: 'LlamaConfig' object is not callable

The working callers (src/carl/live.py, scripts/eval/ablation_live.py) discard the
second value and build the tokenizer separately via transformers.AutoTokenizer.

These tests are CPU-only and download nothing: they exercise the CONTRACT
(_make_prompt's requirement on its argument) and statically guard the specific
misuse, rather than loading a 1.1B checkpoint.
"""
from __future__ import annotations

import ast
import inspect
import os

import pytest
import torch

from src.carl.live import _make_prompt
from src.engine.model import LlamaConfig, load_tinyllama_from_hf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_INTERVENTION = os.path.join(
    _ROOT, "scripts", "eval", "repair", "batch_intervention.py")


def _tiny_config() -> LlamaConfig:
    """A minimal valid LlamaConfig. All 10 fields are required positionally."""
    return LlamaConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )


class _FakeTokenizer:
    """Minimal stand-in with the HF __call__ contract _make_prompt relies on."""

    def __call__(self, text, return_tensors=None):
        ids = torch.arange(1, len(text.split()) + 1, dtype=torch.long)
        return {"input_ids": ids.unsqueeze(0)}


# --- the failure ----------------------------------------------------------

def test_make_prompt_rejects_a_config_the_way_the_t4_run_did():
    """Reproduces the exact T4 traceback without a GPU or a checkpoint."""
    cfg = _tiny_config()
    with pytest.raises(TypeError, match="not callable"):
        _make_prompt(cfg, 16)


def test_loader_second_return_value_is_a_config_and_is_not_callable():
    """The contract that makes the misuse possible. Asserted from the signature
    so it holds without downloading weights."""
    ann = inspect.signature(load_tinyllama_from_hf).return_annotation
    text = ann if isinstance(ann, str) else str(ann)
    assert "LlamaConfig" in text, f"unexpected return annotation: {text!r}"
    assert "Tokenizer" not in text
    assert not callable(_tiny_config()), "a config must not be callable"


# --- the corrected behaviour ----------------------------------------------

def test_make_prompt_works_with_a_real_tokenizer_interface():
    out = _make_prompt(_FakeTokenizer(), 16)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 16), f"expected (1,16), got {tuple(out.shape)}"
    assert out.dtype == torch.long


@pytest.mark.parametrize("length", [1, 7, 16, 64, 257])
def test_make_prompt_honours_requested_length(length):
    assert _make_prompt(_FakeTokenizer(), length).shape == (1, length)


# --- static guard against reintroducing the misuse -------------------------

def _loader_unpack_targets(path):
    """Names bound by `... = load_tinyllama_from_hf(...)` in `path`."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "load_tinyllama_from_hf"):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Tuple):
                found.append([e.id if isinstance(e, ast.Name) else "?"
                              for e in tgt.elts])
    return found


@pytest.mark.parametrize("path", [
    BATCH_INTERVENTION,
    os.path.join(_ROOT, "src", "carl", "live.py"),
    os.path.join(_ROOT, "scripts", "eval", "ablation_live.py"),
])
def test_no_caller_binds_the_config_to_a_tokenizer_name(path):
    unpacks = _loader_unpack_targets(path)
    assert unpacks, f"no loader call found in {os.path.basename(path)}"
    for names in unpacks:
        assert len(names) == 2, f"loader returns a 2-tuple; got {names}"
        second = names[1].lower()
        assert "tok" not in second, (
            f"{os.path.basename(path)} binds the loader's SECOND return value "
            f"to {names[1]!r}. That value is a LlamaConfig, not a tokenizer; "
            "calling it raises TypeError. Build the tokenizer separately with "
            "transformers.AutoTokenizer.")


def test_batch_intervention_builds_a_real_tokenizer():
    src = open(BATCH_INTERVENTION, encoding="utf-8").read()
    assert "AutoTokenizer.from_pretrained" in src, (
        "batch_intervention must construct a tokenizer explicitly")


def test_batch_intervention_loads_fp16_on_cuda():
    """Every other GPU measurement in this repo is fp16. Loading fp32 would make
    the fitted step-time constants incomparable to all of them."""
    src = open(BATCH_INTERVENTION, encoding="utf-8").read()
    assert "torch.float16 if DEVICE.type ==" in src
    assert "load_tinyllama_from_hf(MODEL_NAME, dtype=dtype)" in src
