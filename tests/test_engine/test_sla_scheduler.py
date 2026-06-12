"""
SLA-aware scheduler tests.

CPU-compatible: a tiny random-weight LlamaModel stands in for the real model,
and a controllable clock makes the deadline tests deterministic (no sleeps).
They pin the five policy behaviours:

  1. INTERACTIVE is admitted before BATCH in the same step.
  2. Within a priority class, earliest-deadline-first (EDF).
  3. BACKGROUND runs only when no higher-priority work is queued.
  4. An arriving INTERACTIVE request preempts a BATCH request when the batch is
     full (the BATCH request is requeued with its state preserved).
  5. deadline_miss_callback fires when a TTFT deadline is blown.
"""
from __future__ import annotations

import torch

from src.engine.model import LlamaConfig, LlamaModel
from src.engine.scheduler import RequestStatus
from src.engine.sla_scheduler import RequestPriority, SLAScheduler, SLARequest


def _tiny_model() -> LlamaModel:
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


def _prompt(n: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(n)
    return torch.randint(0, 64, (1, n), generator=g)


def _ids(reqs) -> list[str]:
    return [r.request_id for r in reqs]


class _Clock:
    """A manually-advanced monotonic clock (seconds)."""
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _make_sched(model, max_batch_size=4, clock=None, on_miss=None) -> SLAScheduler:
    return SLAScheduler(
        model,
        max_batch_size=max_batch_size,
        num_blocks=256,
        time_fn=clock,
        deadline_miss_callback=on_miss,
    )


# ---------------------------------------------------------------------------
# 1. INTERACTIVE before BATCH.
# ---------------------------------------------------------------------------


def test_interactive_scheduled_before_batch():
    model = _tiny_model()
    sched = _make_sched(model, max_batch_size=1)
    # BATCH enqueued FIRST -- FIFO would admit it; SLA must admit interactive.
    sched.add_request("batch", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.BATCH)
    sched.add_request("chat", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.INTERACTIVE)

    sched.step()

    assert "chat" in _ids(sched.active)
    assert "batch" in _ids(sched.waiting)
    assert sched.scheduled_by_priority[RequestPriority.INTERACTIVE] == 1
    assert sched.scheduled_by_priority[RequestPriority.BATCH] == 0


# ---------------------------------------------------------------------------
# 2. EDF within a priority class.
# ---------------------------------------------------------------------------


def test_edf_earlier_deadline_first():
    model = _tiny_model()
    clock = _Clock()
    sched = _make_sched(model, max_batch_size=1, clock=clock)
    # Both BATCH (same class). Enqueue the LATER deadline first; EDF must still
    # admit the earlier-deadline request.
    sched.add_request("late", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.BATCH, ttft_deadline_ms=900)
    sched.add_request("early", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.BATCH, ttft_deadline_ms=50)

    sched.step()

    assert "early" in _ids(sched.active)
    assert "late" in _ids(sched.waiting)


# ---------------------------------------------------------------------------
# 3. BACKGROUND only when the queue is otherwise empty.
# ---------------------------------------------------------------------------


def test_background_waits_while_higher_priority_queued():
    model = _tiny_model()
    sched = _make_sched(model, max_batch_size=4)
    sched.add_request("batch", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.BATCH)
    sched.add_request("bg", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.BACKGROUND)

    sched.step()
    # Even though a batch slot is free, background is held while batch is queued
    # (here batch gets admitted; background must NOT be co-admitted this step).
    assert "batch" in _ids(sched.active)
    assert "bg" in _ids(sched.waiting)


def test_background_runs_when_queue_empty():
    model = _tiny_model()
    sched = _make_sched(model, max_batch_size=4)
    # Only background work waiting -> the queue is "otherwise empty", so it runs.
    sched.add_request("bg", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.BACKGROUND)

    sched.step()
    assert "bg" in _ids(sched.active)
    assert sched.scheduled_by_priority[RequestPriority.BACKGROUND] == 1


# ---------------------------------------------------------------------------
# 4. Preemption.
# ---------------------------------------------------------------------------


def test_interactive_preempts_batch_when_full():
    model = _tiny_model()
    sched = _make_sched(model, max_batch_size=1)
    # Admit a BATCH request and let it start decoding (fills the single slot).
    sched.add_request("batch", _prompt(), max_new_tokens=20,
                      priority=RequestPriority.BATCH)
    sched.step()
    assert "batch" in _ids(sched.active)
    batch_req = sched.active[0]
    assert batch_req.status is RequestStatus.DECODE
    tokens_before = batch_req.total_emitted
    assert tokens_before >= 1

    # An INTERACTIVE request arrives; the batch is full (size 1) -> preempt.
    sched.add_request("chat", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.INTERACTIVE)
    sched.step()

    assert "chat" in _ids(sched.active), "interactive must take the freed slot"
    assert "batch" in _ids(sched.waiting), "batch must be requeued, not dropped"
    assert sched.preemptions == 1
    # State preserved: the recompute fold kept every token emitted so far.
    preempted = next(r for r in sched.waiting if r.request_id == "batch")
    assert preempted.total_emitted >= tokens_before
    assert preempted.preempt_count == 1


# ---------------------------------------------------------------------------
# 5. deadline_miss_callback.
# ---------------------------------------------------------------------------


def test_deadline_miss_callback_fires():
    model = _tiny_model()
    clock = _Clock()
    misses: list[tuple[str, float]] = []
    sched = _make_sched(model, max_batch_size=1, clock=clock,
                        on_miss=lambda rid, by: misses.append((rid, by)))

    # Occupy the only slot with a long-running batch job so the interactive
    # request can't be admitted and its TTFT clock runs out while it waits.
    sched.add_request("hog", _prompt(), max_new_tokens=50,
                      priority=RequestPriority.BATCH)
    sched.step()

    # Interactive request with a 100ms TTFT deadline, enqueued at t=0.
    sched.add_request("chat", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.INTERACTIVE, ttft_deadline_ms=100)

    # Advance the clock well past the deadline, then step. With the slot still
    # held, the interactive request has emitted nothing -> a miss must fire.
    clock.advance(0.30)   # 300 ms > 100 ms deadline
    sched.step()

    assert sched.deadline_misses >= 1
    assert any(rid == "chat" for rid, _ in misses)
    rid, missed_by = next((m for m in misses if m[0] == "chat"))
    assert missed_by > 0
    # Roughly 300 - 100 = 200 ms late (allow slack for ordering).
    assert 150 < missed_by < 260, f"missed_by={missed_by}"


def test_no_miss_when_no_deadline():
    """A request with no TTFT deadline never triggers the miss callback."""
    model = _tiny_model()
    clock = _Clock()
    misses: list = []
    sched = _make_sched(model, max_batch_size=1, clock=clock,
                        on_miss=lambda rid, by: misses.append(rid))
    sched.add_request("hog", _prompt(), max_new_tokens=50,
                      priority=RequestPriority.BATCH)
    sched.step()
    sched.add_request("chat", _prompt(), max_new_tokens=5,
                      priority=RequestPriority.INTERACTIVE)  # no deadline
    clock.advance(10.0)
    sched.step()
    assert sched.deadline_misses == 0
    assert misses == []
