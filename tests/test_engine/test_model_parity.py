"""
Parity test: my from-scratch LlamaModel vs HF AutoModelForCausalLM.

What we're testing:
    Given the same input_ids, both models must produce the same logits
    to a small absolute tolerance (atol=1e-4 on CPU, 1e-3 on CUDA) in
    fp32. If they do, our implementation of RMSNorm, RoPE, GQA, SwiGLU,
    residual structure, and weight loading is correct end-to-end.

CPU vs GPU tolerance:
    Our model loads onto DEVICE (CUDA when available); HF stays on CPU.
    fp32 matmul on CUDA with TF32 disabled produces results that differ
    from CPU fp32 by ~1e-5 per layer; across 22 layers the accumulated
    drift can reach ~1e-3 even though both paths are "correct". We bump
    the atol on CUDA to 1e-3 to accept this hardware-level rounding
    while still catching real bugs (a structural mistake like the wrong
    RoPE layout produces diffs of 1e-1 or larger, well above either
    threshold).

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

from src.engine.device import DEVICE


@pytest.fixture
def models(hf_model, my_model):
    """Bundle the two session-scoped models for tests that need both."""
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
        hf_out = hf_model(input_ids).logits          # (1, S, V) on CPU
        my_out = my_model(input_ids).cpu()           # mine may be on CUDA

    assert hf_out.shape == my_out.shape, f"shape mismatch: {hf_out.shape} vs {my_out.shape}"

    # See module docstring for tolerance rationale. Real bugs produce
    # diffs of 1e-1 or larger; either threshold catches them.
    atol = 1e-3 if DEVICE.type == "cuda" else 1e-4
    max_abs_diff = (hf_out - my_out).abs().max().item()
    assert torch.allclose(hf_out, my_out, atol=atol), (
        f"Logits diverged. max |hf - mine| = {max_abs_diff:.3e}. "
        f"atol budget = {atol:.0e} ({DEVICE.type}). Likely culprits: "
        f"RoPE layout, RMSNorm dtype handling, or GQA repeat order."
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
        # my_model returns on DEVICE; .cpu() before extracting the int is
        # technically unnecessary (int() of a 0-dim tensor handles either
        # device) but keeps the comparison style consistent.
        my_next = int(torch.argmax(my_model(input_ids)[0, -1, :].cpu()))

    assert hf_next == my_next, f"Greedy mismatch: HF={hf_next}, mine={my_next}"
