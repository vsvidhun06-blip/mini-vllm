"""
Paged KV cache parity + admission-control test.

The Day 5 SimpleKVCache is gone; both solo `model.generate(use_cache=True)`
and the scheduler now go through the paged pool. So "byte-identical to
the simple cache" reduces to: scheduler outputs == solo outputs, exactly.
That tests the per-request paged view in isolation under one cache
*plus* the multi-request shared pool under the scheduler.

Two scenarios:

  1. Ample blocks. Pool is sized so all 4 requests fit concurrently.
     This exercises the steady-state paged path: every request gets
     admitted on the first step, prefill runs, batched decode runs,
     blocks JIT-allocate as decode steps fill the current tail block.

  2. Tight blocks. Pool is sized so only some requests fit at once.
     Request 4 sits in the WAITING queue while requests 1-3 occupy the
     pool. As earlier requests finish and return their blocks, request 4
     gets admitted and runs. Its output must still match a solo run --
     delayed admission is invisible to the request itself.

  Bonus: print scheduler throughput in both modes vs solo. Not asserted;
  tight-pool mode is necessarily slower than ample-pool, because the
  scheduler runs through fewer concurrent rows per step.
"""
from __future__ import annotations

import time

import pytest
import torch

from src.engine.scheduler import ContinuousBatchScheduler


# Same prompts as the Day 6 scheduler test, so the parity story is consistent
# across days. Different lengths give different per-request block counts.
PROMPTS = [
    "The capital of France is",
    "The largest ocean on Earth is the",
    "Python is a programming language designed",
    "In 1969, the first humans landed on",
]
MAX_NEW = 20


def _solo_outputs(model, tokenizer, prompt_ids_list: list[torch.Tensor]) -> tuple[list[list[int]], float]:
    """Run each prompt solo through model.generate; return tokens + wall time."""
    eos_id = tokenizer.eos_token_id
    outs: list[list[int]] = []
    t0 = time.perf_counter()
    for prompt_ids in prompt_ids_list:
        out = model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW,
            eos_token_id=eos_id,
            use_cache=True,
        )
        outs.append(out[0, prompt_ids.shape[1]:].tolist())
    return outs, time.perf_counter() - t0


def _scheduler_outputs(
    model,
    tokenizer,
    prompt_ids_list: list[torch.Tensor],
    num_blocks: int,
    max_batch_size: int = 4,
) -> tuple[dict[str, list[int]], float, int]:
    """Drive prompts through the scheduler. Return tokens, wall time, and
    the number of step()s taken (a proxy for how much queueing happened)."""
    eos_id = tokenizer.eos_token_id
    scheduler = ContinuousBatchScheduler(
        model,
        max_batch_size=max_batch_size,
        num_blocks=num_blocks,
    )
    for i, prompt_ids in enumerate(prompt_ids_list):
        scheduler.add_request(
            request_id=f"req-{i}",
            prompt_ids=prompt_ids,
            max_new_tokens=MAX_NEW,
            eos_token_id=eos_id,
        )
    outs: dict[str, list[int]] = {f"req-{i}": [] for i in range(len(prompt_ids_list))}
    n_steps = 0
    t0 = time.perf_counter()
    while scheduler.has_work():
        for rid, tok in scheduler.step():
            outs[rid].append(tok)
        n_steps += 1
    return outs, time.perf_counter() - t0, n_steps


def _compare(prompts: list[str], solo: list[list[int]], batched: dict[str, list[int]]) -> None:
    failures: list[str] = []
    for i, prompt in enumerate(prompts):
        rid = f"req-{i}"
        if solo[i] != batched[rid]:
            failures.append(
                f"\nRequest {rid} (prompt: {prompt!r})\n"
                f"  solo:    {solo[i]}\n"
                f"  batched: {batched[rid]}"
            )
    if failures:
        raise AssertionError(
            "Paged-scheduler tokens diverged from solo tokens:" + "".join(failures)
        )


def test_paged_scheduler_matches_solo_with_ample_blocks(model_and_tokenizer) -> None:
    """Steady-state paged path: enough blocks for everyone, parity must hold.

    Each of the four short prompts needs ~2 blocks at block_size=16 (prompt
    ~6-9 tokens + 20 generated, /16 = 2). 64 blocks is comfortable headroom;
    all four requests are admitted on step 1.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids_list = [tokenizer(p, return_tensors="pt")["input_ids"] for p in PROMPTS]

    solo, solo_time = _solo_outputs(model, tokenizer, prompt_ids_list)
    batched, batched_time, n_steps = _scheduler_outputs(
        model, tokenizer, prompt_ids_list, num_blocks=64,
    )

    _compare(PROMPTS, solo, batched)

    speedup = solo_time / batched_time if batched_time > 0 else float("inf")
    print(
        f"\n[ample blocks] solo={solo_time:.2f}s  "
        f"paged={batched_time:.2f}s  speedup={speedup:.2f}x  steps={n_steps}"
    )


def test_paged_scheduler_with_tight_blocks_still_parity(model_and_tokenizer) -> None:
    """Admission control: scarce blocks force request 4 to wait, output must still match.

    Each request needs 2 blocks (~). With num_blocks=6 only 3 fit at once
    -- request 4 stays in the waiting queue until one of {1,2,3} finishes
    and frees its blocks. The whole point of admission control is that
    delayed admission is *transparent* to the request: same tokens, just
    later wall-clock.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids_list = [tokenizer(p, return_tensors="pt")["input_ids"] for p in PROMPTS]

    solo, solo_time = _solo_outputs(model, tokenizer, prompt_ids_list)
    batched, batched_time, n_steps = _scheduler_outputs(
        model, tokenizer, prompt_ids_list, num_blocks=6,
    )

    # Every request still terminates (otherwise something is wrong with the
    # eviction-and-readmit path).
    for rid, toks in batched.items():
        assert len(toks) > 0, f"{rid} produced no tokens -- starved by admission control?"
        # The scheduler should never silently truncate -- it should run until
        # the request emits an EOS or hits max_new_tokens.
        eos = tokenizer.eos_token_id
        assert toks[-1] == eos or len(toks) == MAX_NEW, (
            f"{rid} stopped at len {len(toks)} without EOS"
        )

    # Tokens identical to solo, regardless of when each request got to run.
    _compare(PROMPTS, solo, batched)

    speedup = solo_time / batched_time if batched_time > 0 else float("inf")
    print(
        f"\n[tight blocks] solo={solo_time:.2f}s  "
        f"paged={batched_time:.2f}s  speedup={speedup:.2f}x  steps={n_steps}"
    )
