"""
Benchmark: FIFO scheduler vs SLA scheduler on a mixed-priority workload.

Run:
    python scripts/benchmark_sla_scheduler.py

Workload: 4 INTERACTIVE + 8 BATCH + 4 BACKGROUND requests, all submitted up
front in an ADVERSARIAL order (batch + background first, interactive last) and
a batch size smaller than the request count, so admission order actually
matters. This is the situation the SLA scheduler exists for: under FIFO the
interactive requests queue behind a wall of batch work; under the SLA scheduler
they jump the queue (and preempt batch when the batch is full).

Metrics (measured in scheduler STEPS so the comparison is independent of raw
model speed -- TTFT is the step at which a request emits its first token):

  * INTERACTIVE P99 TTFT  -- should be much lower under the SLA scheduler.
  * BATCH throughput       -- tokens per step; expected to degrade modestly.
  * BACKGROUND completion  -- fraction finished by the end of the run.
  * Deadline miss rate     -- interactive requests whose first-token step blew
                              a fixed TTFT budget (same budget for both
                              schedulers, an apples-to-apples comparison).

The point: the SLA scheduler trades a little batch throughput to keep
interactive TTFT low and within SLA.
"""
from __future__ import annotations

import time

import torch

from src.engine.model import LlamaConfig, LlamaModel
from src.engine.scheduler import ContinuousBatchScheduler
from src.engine.sla_scheduler import RequestPriority, SLAScheduler

VOCAB = 128
MAX_BATCH = 4                 # < 16 requests, so admission order matters
NUM_BLOCKS = 4096
# A generous TTFT budget in STEPS used to score deadline misses identically for
# both schedulers (the SLA scheduler also enforces a real-clock deadline live).
TTFT_STEP_BUDGET = 6

# (priority, count, prompt_len, max_new_tokens)
WORKLOAD = [
    (RequestPriority.BATCH, 8, 12, 24),
    (RequestPriority.BACKGROUND, 4, 12, 16),
    (RequestPriority.INTERACTIVE, 4, 8, 8),   # enqueued LAST -> worst case for FIFO
]


def _model() -> LlamaModel:
    torch.manual_seed(0)
    return LlamaModel(LlamaConfig(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )).eval()


def _build_requests():
    """Materialise the workload as a flat list, preserving WORKLOAD order."""
    reqs = []
    g = torch.Generator().manual_seed(7)
    idx = 0
    for priority, count, plen, mnt in WORKLOAD:
        for _ in range(count):
            rid = f"{priority.name.lower()}-{idx}"
            prompt = torch.randint(0, VOCAB, (1, plen), generator=g)
            reqs.append((rid, prompt, mnt, priority))
            idx += 1
    return reqs


def _percentile(values, pct):
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _run(scheduler, reqs, is_sla):
    """Submit all requests, run to completion, return per-request step stats."""
    for rid, prompt, mnt, priority in reqs:
        if is_sla:
            scheduler.add_request(rid, prompt, max_new_tokens=mnt, priority=priority)
        else:
            scheduler.add_request(rid, prompt, max_new_tokens=mnt)

    first_token_step: dict[str, int] = {}
    finish_step: dict[str, int] = {}
    seen_first: set[str] = set()
    step = 0
    t0 = time.perf_counter()
    while scheduler.has_work():
        emitted = scheduler.step()
        step += 1
        for rid, _tok in emitted:
            if rid not in seen_first:
                first_token_step[rid] = step
                seen_first.add(rid)
        for r in scheduler.get_finished():
            finish_step[r.request_id] = step
        if step > 10_000:                     # safety valve
            break
    wall = time.perf_counter() - t0
    return first_token_step, finish_step, step, wall


def _summarise(name, reqs, first_token_step, finish_step, total_steps):
    by_prio = {p: [r for r in reqs if r[3] is p] for p in RequestPriority}

    inter_ttfts = [first_token_step.get(r[0], total_steps) for r in by_prio[RequestPriority.INTERACTIVE]]
    misses = sum(1 for t in inter_ttfts if t > TTFT_STEP_BUDGET)

    # Batch throughput: total batch tokens emitted / steps.
    batch_reqs = by_prio[RequestPriority.BATCH]
    batch_tokens = sum(r[2] for r in batch_reqs if r[0] in finish_step)
    # (use requested max_new as proxy for finished requests' output length)
    batch_tput = batch_tokens / total_steps if total_steps else 0.0

    bg_reqs = by_prio[RequestPriority.BACKGROUND]
    bg_done = sum(1 for r in bg_reqs if r[0] in finish_step)
    bg_rate = bg_done / len(bg_reqs) if bg_reqs else 0.0

    return {
        "name": name,
        "inter_p99_ttft": _percentile(inter_ttfts, 99),
        "inter_mean_ttft": sum(inter_ttfts) / len(inter_ttfts),
        "batch_tput": batch_tput,
        "bg_rate": bg_rate,
        "miss_rate": misses / len(inter_ttfts),
        "total_steps": total_steps,
    }


def main() -> None:
    model = _model()
    reqs = _build_requests()
    print(f"Workload: 4 INTERACTIVE + 8 BATCH + 4 BACKGROUND, "
          f"max_batch_size={MAX_BATCH}")
    print(f"TTFT budget for miss-scoring: {TTFT_STEP_BUDGET} steps "
          f"(interactive enqueued LAST)\n")

    fifo = ContinuousBatchScheduler(model, max_batch_size=MAX_BATCH, num_blocks=NUM_BLOCKS)
    fifo_stats = _run(fifo, reqs, is_sla=False)
    fifo_sum = _summarise("FIFO", reqs, *fifo_stats[:3])

    sla = SLAScheduler(model, max_batch_size=MAX_BATCH, num_blocks=NUM_BLOCKS)
    sla_stats = _run(sla, reqs, is_sla=True)
    sla_sum = _summarise("SLA", reqs, *sla_stats[:3])

    cols = (f"{'sched':>5} | {'INT P99 TTFT':>12} | {'INT mean TTFT':>13} | "
            f"{'BATCH tput':>10} | {'BG done':>7} | {'INT miss':>8}")
    print(cols)
    print("-" * len(cols))
    for s in (fifo_sum, sla_sum):
        print(f"{s['name']:>5} | {s['inter_p99_ttft']:>12.1f} | "
              f"{s['inter_mean_ttft']:>13.1f} | {s['batch_tput']:>10.2f} | "
              f"{s['bg_rate']:>6.0%} | {s['miss_rate']:>8.0%}")

    print(f"\nSLA preemptions: {sla.preemptions} | "
          f"SLA live deadline misses: {sla.deadline_misses}")
    print("INT P99 TTFT is measured in scheduler steps (lower = snappier first "
          "token).\nThe SLA scheduler front-runs interactive requests (and "
          "preempts batch when\nthe batch is full), cutting interactive TTFT at "
          "a modest cost to batch\nthroughput -- exactly the SLA trade-off.")


if __name__ == "__main__":
    main()
