"""
Scheduler parity test + throughput sanity check.

What we're testing:
    Greedy decoding is deterministic. Each request has its OWN KV cache.
    So whether we run a prompt solo through `model.generate()` or run it
    concurrently through `ContinuousBatchScheduler` alongside other prompts,
    the generated token sequence MUST be identical.

    If it diverges, something in the batched-decode path corrupted state
    across rows -- the kind of bug that's catastrophic in a real serving
    system (one request's tokens leaking into another's).

What this test does NOT prove:
    Speedup is reported for visibility. It's not a hard assertion -- on a
    CPU-only setup with a 1.1B model the wall-clock dominator is the
    matmuls, and batching N requests through one forward pass gives an
    obvious win. On GPU you'd see a much bigger gap; here we just want
    "batched is not slower" plus the parity check.
"""
from __future__ import annotations

import time

import pytest
import torch

from src.engine.scheduler import ContinuousBatchScheduler


# Four prompts with intentionally different lengths and topics, to exercise
# different cache lengths and different generation behavior.
PROMPTS = [
    "The capital of France is",
    "The largest ocean on Earth is the",
    "Python is a programming language designed",
    "In 1969, the first humans landed on",
]
MAX_NEW = 20


def test_scheduler_matches_solo(model_and_tokenizer) -> None:
    """Each request's batched output == its solo output, token by token."""
    model, tokenizer = model_and_tokenizer
    eos_id = tokenizer.eos_token_id

    prompt_ids_list = [
        tokenizer(p, return_tensors="pt")["input_ids"] for p in PROMPTS
    ]

    # ---- Solo runs ---------------------------------------------------------
    # Each prompt through generate() on its own. Record only the generated
    # tokens (everything after the prompt).
    solo_outputs: list[list[int]] = []
    t0 = time.perf_counter()
    for prompt_ids in prompt_ids_list:
        out = model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW,
            eos_token_id=eos_id,
            use_cache=True,
        )
        gen_only = out[0, prompt_ids.shape[1]:].tolist()
        solo_outputs.append(gen_only)
    solo_time = time.perf_counter() - t0

    # ---- Batched run via scheduler ----------------------------------------
    # num_blocks=64 is comfortable headroom for these 4 short prompts (each
    # needs ~2 blocks at block_size=16, so 8 total suffices); the parity
    # path is not the place to exercise admission control.
    scheduler = ContinuousBatchScheduler(model, max_batch_size=4, num_blocks=64)
    for i, prompt_ids in enumerate(prompt_ids_list):
        scheduler.add_request(
            request_id=f"req-{i}",
            prompt_ids=prompt_ids,
            max_new_tokens=MAX_NEW,
            eos_token_id=eos_id,
        )

    batched_outputs: dict[str, list[int]] = {f"req-{i}": [] for i in range(len(PROMPTS))}
    t0 = time.perf_counter()
    while scheduler.has_work():
        for rid, tok in scheduler.step():
            batched_outputs[rid].append(tok)
    batched_time = time.perf_counter() - t0

    # ---- Compare ----------------------------------------------------------
    failures = []
    for i, prompt in enumerate(PROMPTS):
        rid = f"req-{i}"
        solo = solo_outputs[i]
        batched = batched_outputs[rid]
        if solo != batched:
            failures.append(
                f"\nRequest {rid} (prompt: {prompt!r})\n"
                f"  solo:    {solo}\n"
                f"  batched: {batched}"
            )
    if failures:
        raise AssertionError(
            "Batched generation diverged from solo generation:" + "".join(failures)
        )

    # ---- Timing (informational) -------------------------------------------
    speedup = solo_time / batched_time if batched_time > 0 else float("inf")
    # pytest -s will show this; otherwise it's captured.
    print(
        f"\n[timing] solo total = {solo_time:.2f}s  "
        f"batched = {batched_time:.2f}s  "
        f"speedup = {speedup:.2f}x"
    )

    # Sanity assertion: batched should not be dramatically slower than solo.
    # We don't assert a minimum speedup because that's hardware-dependent;
    # we just protect against catastrophic regressions.
    assert batched_time < solo_time * 1.5, (
        f"Batched is significantly slower than solo: "
        f"{batched_time:.2f}s vs {solo_time:.2f}s. "
        f"Something is wrong with the batched-decode path."
    )
