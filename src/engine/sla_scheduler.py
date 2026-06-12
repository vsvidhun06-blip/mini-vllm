"""
SLA-aware scheduling: respect per-request latency targets while keeping
throughput high.

The base ContinuousBatchScheduler admits requests FIFO -- it maximises
throughput but is blind to WHO is waiting. A production serving system mixes
two very different traffic classes:

  * latency-sensitive (chat, interactive autocomplete) -- a human is watching a
    spinner; time-to-first-token (TTFT) and inter-token latency (TPOT) are the
    product, and a few hundred ms of head-of-line blocking is a bad experience.
  * throughput-optimised (batch eval jobs, offline summarisation) -- nobody is
    watching; only aggregate tokens/sec matters, and they should yield to
    interactive traffic.

SLAScheduler adds a priority + deadline policy on top of the existing engine:

  1. INTERACTIVE requests are always admitted first.
  2. Within one priority class, earliest-deadline-first (EDF) ordering -- the
     request whose TTFT deadline expires soonest goes first.
  3. BATCH requests fill whatever batch slots interactive traffic leaves.
  4. BACKGROUND requests run only when the queue is otherwise empty (no
     interactive or batch work waiting).
  5. If an INTERACTIVE request arrives and the batch is full, a lower-priority
     (BATCH/BACKGROUND) request is PREEMPTED -- requeued with its state
     preserved -- to free a slot.

Design choice -- reuse, don't rewrite
-------------------------------------
The base step() already does admission/prefill/decode/eviction correctly and is
heavily tested. Rather than reimplement it, SLAScheduler shapes the inputs the
base admission loop sees:

    preempt -> reorder self.waiting by policy -> hide BACKGROUND when higher
    work waits -> super().step() -> restore hidden BACKGROUND -> deadline +
    metrics bookkeeping.

Because the base loop pops self.waiting front-to-back, an SLA-sorted queue makes
it admit in priority/EDF order with zero changes to the core loop. With the env
flag off and no SLA fields set, behaviour is byte-identical to the base
scheduler, so existing tests are unaffected.

Preemption strategy -- recomputation
------------------------------------
A preempted request folds its already-generated tokens into its prompt, frees
its KV blocks, and goes back to the queue. On re-admission it re-prefills the
combined context and continues exactly where it left off (this is vLLM's
"recompute" preemption, the simpler sibling of KV swapping). Its remaining
max_new_tokens budget and cumulative emitted count are preserved, so the output
stream is seamless.

The pure scheduling policy (ordering, victim selection, deadline math) lives in
module-level functions so it can be unit-tested without a model or a GPU.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import torch

from src.engine import events
from src.engine.scheduler import (
    ContinuousBatchScheduler,
    Request,
    RequestStatus,
)


class RequestPriority(IntEnum):
    """Traffic class. Lower value == higher priority (sorts first)."""
    INTERACTIVE = 0
    BATCH = 1
    BACKGROUND = 2


@dataclass
class SLARequest(Request):
    """A Request annotated with its SLA: priority and latency deadlines.

    All SLA fields have defaults so this stays a valid dataclass subclass of
    Request (whose own optional fields precede these). The trailing underscore
    fields are internal bookkeeping the scheduler maintains.
    """
    priority: RequestPriority = RequestPriority.INTERACTIVE
    # Max acceptable time-to-first-token, in ms from enqueue. None == no TTFT SLA.
    ttft_deadline_ms: float | None = None
    # Max acceptable per-output-token latency, in ms. None == no TPOT SLA.
    tpot_budget_ms: float | None = None
    # Wall-clock (scheduler clock) timestamp at enqueue. Set by add_request.
    admitted_at: float = 0.0

    # --- internal bookkeeping (not part of the public SLA spec) -------------
    # Clock time the request emitted its first token (for TTFT measurement).
    first_token_at: float | None = field(default=None, repr=False)
    # Guard so a TTFT deadline miss fires the callback at most once.
    ttft_miss_fired: bool = field(default=False, repr=False)
    # Tokens emitted in prior life-cycles, before recompute preemptions folded
    # them into the prompt. Keeps the cumulative output count honest.
    emitted_before_preempt: int = field(default=0, repr=False)
    # How many times this request has been preempted (metrics / debugging).
    preempt_count: int = field(default=0, repr=False)

    @property
    def total_emitted(self) -> int:
        """Tokens emitted across all life-cycles (survives preemption folds)."""
        return self.emitted_before_preempt + len(self.generated_token_ids)

    @property
    def first_token_emitted(self) -> bool:
        return self.first_token_at is not None or self.total_emitted > 0


# ---------------------------------------------------------------------------
# Pure scheduling policy (no model, no pool -- unit-testable in isolation).
# ---------------------------------------------------------------------------


def abs_ttft_deadline(req: SLARequest) -> float:
    """Absolute clock time by which the first token must arrive.

    enqueue time + the TTFT budget. No deadline -> +inf (sorts last under EDF).
    """
    if req.ttft_deadline_ms is None:
        return float("inf")
    return req.admitted_at + req.ttft_deadline_ms / 1000.0


def sla_ordering_key(req: SLARequest) -> tuple:
    """Sort key implementing policy rules 1-2: priority first, then EDF.

    (priority.value ascending == INTERACTIVE before BATCH before BACKGROUND;
     within a class, earliest absolute deadline first.)
    """
    return (int(req.priority), abs_ttft_deadline(req))


def order_requests(reqs) -> list:
    """Stable-sort requests into scheduling order (priority, then EDF)."""
    return sorted(reqs, key=sla_ordering_key)


def ttft_miss(req: SLARequest, now: float) -> tuple[bool, float]:
    """Has this request blown its TTFT deadline as of `now`?

    Returns (missed, missed_by_ms). A miss requires a deadline, no first token
    yet, and the deadline already in the past.
    """
    if req.ttft_deadline_ms is None or req.first_token_emitted:
        return (False, 0.0)
    deadline = abs_ttft_deadline(req)
    if now > deadline:
        return (True, (now - deadline) * 1000.0)
    return (False, 0.0)


def preemption_candidates(active) -> list:
    """Active requests eligible to be preempted, most-preemptable first.

    Only BATCH/BACKGROUND with remaining work are candidates. Ordered lowest
    priority first (BACKGROUND before BATCH), and within a class the LEAST
    urgent (latest deadline) first -- we sacrifice the request that hurts least.
    """
    cands = [
        r for r in active
        if isinstance(r, SLARequest)
        and r.priority in (RequestPriority.BATCH, RequestPriority.BACKGROUND)
        and (r.max_new_tokens - len(r.generated_token_ids)) > 0
    ]
    # Highest priority.value (lowest priority) first; then latest deadline first.
    cands.sort(key=lambda r: (-int(r.priority), -abs_ttft_deadline(r)))
    return cands


# ---------------------------------------------------------------------------
# SLAScheduler
# ---------------------------------------------------------------------------


class SLAScheduler(ContinuousBatchScheduler):
    """ContinuousBatchScheduler with priority + deadline aware admission."""

    def __init__(
        self,
        *args,
        deadline_miss_callback: Callable[[str, float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
        **kwargs,
    ) -> None:
        """
        Extra args (everything else is forwarded to the base scheduler):
            deadline_miss_callback: optional `(request_id, missed_by_ms) -> None`
                fired once when a request misses its TTFT deadline.
            time_fn: clock source (defaults to time.monotonic). Injectable so
                deadline behaviour is deterministically testable without sleeps.
        """
        super().__init__(*args, **kwargs)
        self.deadline_miss_callback = deadline_miss_callback
        self._time_fn = time_fn or time.monotonic

        # Cumulative count of requests admitted (WAITING -> active) per class.
        self.scheduled_by_priority: dict[RequestPriority, int] = {
            p: 0 for p in RequestPriority
        }
        # Same, but just for the most recent step.
        self.last_scheduled_by_priority: dict[RequestPriority, int] = {
            p: 0 for p in RequestPriority
        }
        self.deadline_misses: int = 0
        self.preemptions: int = 0

    def _now(self) -> float:
        return self._time_fn()

    # ---- enqueue ---------------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        prompt_text: str | None = None,
        adapter_id: str | None = None,
        priority: RequestPriority = RequestPriority.INTERACTIVE,
        ttft_deadline_ms: float | None = None,
        tpot_budget_ms: float | None = None,
    ) -> None:
        """Enqueue an SLA-annotated request. Mirrors the base validation/device
        move, but builds an SLARequest and stamps its enqueue time."""
        if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(
                f"prompt_ids must be shape (1, S); got {tuple(prompt_ids.shape)}"
            )
        prompt_ids = prompt_ids.to(self.device)
        self.waiting.append(SLARequest(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            prompt_text=prompt_text,
            adapter_id=adapter_id,
            priority=priority,
            ttft_deadline_ms=ttft_deadline_ms,
            tpot_budget_ms=tpot_budget_ms,
            admitted_at=self._now(),
        ))

    # ---- policy application ---------------------------------------------

    def _sla_sort_waiting(self) -> None:
        """Reorder the waiting queue: priority first, EDF within a priority."""
        self.waiting = deque(order_requests(self.waiting))

    def _maybe_preempt(self) -> None:
        """If INTERACTIVE work is waiting and the batch is full, free slots by
        preempting the least-important active requests.

        'Batch is full' is the compute-budget constraint (max_batch_size). We
        preempt at most as many victims as there are waiting interactive
        requests with no free slot to absorb them.
        """
        waiting_interactive = sum(
            1 for r in self.waiting
            if isinstance(r, SLARequest) and r.priority is RequestPriority.INTERACTIVE
        )
        if waiting_interactive == 0:
            return
        free_slots = self.max_batch_size - len(self.active)
        need = waiting_interactive - max(0, free_slots)
        if need <= 0:
            return  # the batch has room; the base loop will admit them
        victims = preemption_candidates(self.active)
        for victim in victims[:need]:
            self._preempt(victim)

    def _preempt(self, r: SLARequest) -> None:
        """Recompute-preempt `r`: fold its generated tokens into the prompt,
        free its KV blocks, and requeue it with state preserved."""
        if r.generated_token_ids:
            gen = torch.tensor(
                [r.generated_token_ids], dtype=torch.long, device=self.device
            )
            r.prompt_ids = torch.cat([r.prompt_ids, gen], dim=1)
            n = len(r.generated_token_ids)
            r.emitted_before_preempt += n
            # Remaining budget shrinks by what we've already produced.
            r.max_new_tokens = max(1, r.max_new_tokens - n)
            r.generated_token_ids = []
        # Release the KV cache; recompute will rebuild it on re-admission.
        if r.request_id in self.pool._blocks:           # noqa: SLF001
            self.pool.free_request(r.request_id)
        r.cache = None
        r.chunk_state = None
        r.prefill_hit_boundary = 0
        r.status = RequestStatus.WAITING
        r.preempt_count += 1
        self.active.remove(r)
        self.waiting.append(r)   # _sla_sort_waiting will place it correctly
        self.preemptions += 1
        self._emit(events.request_waiting(
            request_id=r.request_id,
            reason=f"preempted by interactive request (recompute, "
                   f"{r.total_emitted} tokens preserved)",
        ))

    def _partition_background(self) -> list:
        """Hide BACKGROUND requests from the base admission loop when ANY higher
        priority work is waiting -- enforcing 'BACKGROUND only when the queue is
        otherwise empty'. Returns the hidden requests to restore afterwards.
        """
        has_higher = any(
            isinstance(r, SLARequest) and r.priority is not RequestPriority.BACKGROUND
            for r in self.waiting
        )
        if not has_higher:
            return []
        hidden = [
            r for r in self.waiting
            if isinstance(r, SLARequest) and r.priority is RequestPriority.BACKGROUND
        ]
        self.waiting = deque(
            r for r in self.waiting
            if not (isinstance(r, SLARequest) and r.priority is RequestPriority.BACKGROUND)
        )
        return hidden

    # ---- step ------------------------------------------------------------

    def step(self) -> list[tuple[str, int]]:
        now = self._now()

        # TTFT deadline check BEFORE doing work, so a request that has already
        # blown its deadline while queued is reported even if it never runs.
        self._check_ttft_deadlines(now)

        # 1) Preempt low-priority work to make room for waiting interactive.
        self._maybe_preempt()
        # 2) Order the queue by policy (priority, then EDF).
        self._sla_sort_waiting()
        # 3) Hide BACKGROUND while higher-priority work is queued.
        hidden_background = self._partition_background()

        # Snapshot which requests are active so we can attribute admissions.
        active_ids_before = {id(r) for r in self.active}

        # 4) Run the unmodified base iteration over the SLA-shaped queue.
        emitted = super().step()

        # 5) Restore hidden BACKGROUND requests to the back of the queue.
        for r in hidden_background:
            self.waiting.append(r)

        # 6) Bookkeeping: admissions by priority, first-token times, deadlines.
        self._record_admissions(active_ids_before)
        self._record_first_tokens(now)
        self._check_ttft_deadlines(self._now())
        return emitted

    # ---- bookkeeping -----------------------------------------------------

    def _record_admissions(self, active_ids_before: set) -> None:
        """Count requests that became active this step, by priority."""
        for p in RequestPriority:
            self.last_scheduled_by_priority[p] = 0
        for r in self.active:
            if id(r) not in active_ids_before and isinstance(r, SLARequest):
                self.scheduled_by_priority[r.priority] += 1
                self.last_scheduled_by_priority[r.priority] += 1

    def _record_first_tokens(self, now: float) -> None:
        """Stamp first_token_at the first step a request has produced output."""
        for r in self.active:
            if isinstance(r, SLARequest) and r.first_token_at is None \
                    and len(r.generated_token_ids) > 0:
                r.first_token_at = now

    def _check_ttft_deadlines(self, now: float) -> None:
        """Fire deadline_miss_callback once per request that blew its TTFT SLA."""
        # Both queued and running requests can miss (a slow prefill misses too).
        for r in list(self.waiting) + list(self.active):
            if not isinstance(r, SLARequest) or r.ttft_miss_fired:
                continue
            missed, by_ms = ttft_miss(r, now)
            if missed:
                r.ttft_miss_fired = True
                self.deadline_misses += 1
                if self.deadline_miss_callback is not None:
                    self.deadline_miss_callback(r.request_id, by_ms)


def sla_scheduler_enabled() -> bool:
    """True iff the ENABLE_SLA_SCHEDULER env var is set to a truthy value."""
    return os.environ.get("ENABLE_SLA_SCHEDULER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def parse_priority(name: str | None) -> RequestPriority:
    """Map an API priority string to the enum (default INTERACTIVE)."""
    if not name:
        return RequestPriority.INTERACTIVE
    try:
        return RequestPriority[name.strip().upper()]
    except KeyError:
        return RequestPriority.INTERACTIVE
