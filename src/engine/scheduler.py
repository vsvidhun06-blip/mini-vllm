"""
Continuous batching scheduler -- the centerpiece of the inference engine.

What problem this solves:

  Naive batching takes a fixed batch of N prompts, runs prefill on all of
  them, then runs decode in lockstep until every request is done. Two
  failure modes:

    1. Head-of-line blocking. One long request keeps every shorter request
       waiting for the batch to drain.
    2. Idle slots. As short requests finish, their batch slots sit empty
       until the longest one finishes.

  Continuous batching (a.k.a. iteration-level scheduling, originated by
  Orca, popularized by vLLM) processes one DECODE STEP at a time as the
  scheduling unit. After every step:
    * Finished requests are evicted from the batch immediately.
    * New requests can join the batch immediately.
  Batch composition is fluid across steps; throughput stays high under
  diverse request lengths.

Day 7 addition: paged KV cache + admission control.

  The scheduler owns a PagedKVCache pool of fixed total capacity. A request
  cannot be admitted unless the pool has enough free blocks for its
  worst-case footprint (prompt + max_new_tokens). Requests stuck behind
  insufficient capacity stay in the waiting queue until other requests
  finish and return their blocks.

  This is the production-grade admission story: under load, requests
  queue rather than thrash memory.

Day 6 simplification still in force: mixed prefill + decode batching.

  Prefill requests are processed sequentially -- one forward pass per
  prefill -- and the decode requests are batched into ONE forward pass.
  Per step we do (n_prefill + 1) forward passes instead of 1. Steady
  state most steps are decode-only and run one forward pass.

Public surface:

  scheduler.add_request(id, prompt_ids, max_new_tokens, eos_token_id)
  scheduler.step() -> list[(request_id, token_id)]
  scheduler.has_work() -> bool
  scheduler.get_finished() -> list[Request]
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from src.engine import events
from src.engine.kv_cache import PagedKVCache, PagedRequestCache

if TYPE_CHECKING:
    from src.engine.events import EventBus
    from src.engine.model import LlamaModel


from enum import Enum, auto


class RequestStatus(Enum):
    """Where a request is in its lifecycle.

    Transitions:
        WAITING --admit (capacity AND block budget)--> PREFILL
        PREFILL --run prefill, emit 1 token--> DECODE (or DONE if EOS / cap)
        DECODE  --decode step, emit 1 token--> DECODE (loops)
        DECODE  --EOS or cap--> DONE
    """
    WAITING = auto()
    PREFILL = auto()
    DECODE  = auto()
    DONE    = auto()


@dataclass
class Request:
    """All state for one in-flight request."""
    request_id: str
    prompt_ids: torch.Tensor    # (1, S_prompt) int64
    max_new_tokens: int
    eos_token_id: int | None
    # Prompt text is carried only so the request_admitted event can echo
    # it back to the visualiser. The model never sees it; tokenisation
    # happened before we got here. None when the caller didn't supply it.
    prompt_text: str | None = None

    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    cache: PagedRequestCache | None = None      # allocated at admission

    @property
    def last_token_id(self) -> int:
        if self.generated_token_ids:
            return self.generated_token_ids[-1]
        return int(self.prompt_ids[0, -1])

    def is_finished(self) -> bool:
        if not self.generated_token_ids:
            return False
        last = self.generated_token_ids[-1]
        if self.eos_token_id is not None and last == self.eos_token_id:
            return True
        if len(self.generated_token_ids) >= self.max_new_tokens:
            return True
        return False


class ContinuousBatchScheduler:
    def __init__(
        self,
        model: "LlamaModel",
        max_batch_size: int,
        num_blocks: int,
        block_size: int = 16,
        event_bus: "EventBus | None" = None,
        token_decoder: Callable[[int], str] | None = None,
        token_emitter: Callable[[str, int, str, int], None] | None = None,
    ) -> None:
        """
        Args:
            model: the LlamaModel.
            max_batch_size: max concurrent requests in the active batch.
                This is the COMPUTE budget (rows in one forward pass).
            num_blocks: total physical blocks in the paged KV pool.
                This is the MEMORY budget. Each block holds `block_size`
                tokens of K and V at every layer.
            block_size: tokens per block. 16 matches vLLM's default and
                is a good fit for typical prompt + decode lengths.
            event_bus: optional. When provided, the scheduler emits
                structured events at every state transition (admission,
                prefill, decode, eviction) and forwards it to the KV pool
                so block_allocated / block_freed also surface. When None
                the scheduler is silent and existing engine tests run
                byte-identically.
            token_decoder: optional callable `(token_id) -> token_str`.
                Only used for the decode_step event's human-readable
                token text. None means the event omits the string and
                the consumer can decode on its own.
            token_emitter: optional callable
                `(request_id, token_id, token_str, step_idx) -> None`.
                Fired once per generated token for EACH request, in both
                prefill (the first generated token) and decode phases.
                Used by the SSE `/generate/stream` plumbing to push
                tokens onto per-request asyncio queues from this
                (synchronous, scheduler-thread) call site. The scheduler
                stays asyncio-unaware; bridging happens in the callback
                via `loop.call_soon_threadsafe`. None means no
                per-token push (existing engine tests are unaffected).

        Two independent budgets:
            * max_batch_size: how many requests can run in one forward pass.
            * num_blocks: how much total cache space the engine has.
        A request might pass the batch check but fail the block check and
        stay queued; vice versa.
        """
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.event_bus = event_bus
        self.token_decoder = token_decoder
        self.token_emitter = token_emitter

        # The KV pool. One big tensor shared across all requests. The pool
        # also gets the event_bus so block_allocated / block_freed fire
        # at the source without us having to introspect.
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        # Stash the model's device so add_request and step() can put
        # prompt + decode inputs on it without re-reading parameters() each
        # call. The pool's K_pool.device is the same value; keeping it
        # here makes the dependency explicit at the scheduler level.
        self.device = device
        self.pool = PagedKVCache(
            num_layers=self.num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            event_bus=event_bus,
        )

        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.finished: list[Request] = []
        # Monotonic step counter. Exposed via the decode_step event so the
        # visualiser can align decode bursts to a timeline.
        self._step_idx: int = 0

    # ---- Helpers (event emission lives here so the core loop stays clean) ----

    def _emit(self, event: events.Event) -> None:
        """Fire an event if a bus is attached; otherwise no-op."""
        if self.event_bus is not None:
            self.event_bus.emit(event)

    def _decode_token_str(self, token_id: int) -> str:
        """Resolve a token id to display text. Empty string if no decoder."""
        if self.token_decoder is None:
            return ""
        try:
            return self.token_decoder(token_id)
        except Exception:
            return ""

    # ---- Public API ---------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        prompt_text: str | None = None,
    ) -> None:
        """Enqueue a new request. Admission happens at the next step() if
        BOTH the batch has room AND the pool has enough free blocks for
        this request's worst-case footprint.

        prompt_text is optional and only used by the request_admitted
        event payload for visualiser display.
        """
        if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(
                f"prompt_ids must be shape (1, S); got {tuple(prompt_ids.shape)}"
            )
        # Move once at submission so step() / model.forward don't see a
        # device mismatch later. Callers may hand us CPU tensors from the
        # tokenizer; the model and KV pool live on self.device.
        prompt_ids = prompt_ids.to(self.device)
        self.waiting.append(Request(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            prompt_text=prompt_text,
        ))

    def has_work(self) -> bool:
        return bool(self.waiting or self.active)

    def get_finished(self) -> list[Request]:
        out = self.finished
        self.finished = []
        return out

    # ---- Internals ----------------------------------------------------

    def _blocks_needed(self, r: Request) -> tuple[int, int]:
        """How many blocks does a request need?

        Returns:
            (prefill_blocks, total_blocks). prefill_blocks is what we
            allocate immediately at admission; total_blocks is the
            worst-case lifetime footprint and is what admission control
            checks against the free pool.
        """
        bs = self.block_size
        prompt_len = r.prompt_ids.shape[1]
        prefill = (prompt_len + bs - 1) // bs
        total = (prompt_len + r.max_new_tokens + bs - 1) // bs
        return prefill, total

    # ---- Core loop ----------------------------------------------------

    def step(self) -> list[tuple[str, int]]:
        """One engine iteration. Returns every token emitted this step.

        Phases:
          1. Admission: WAITING -> PREFILL while batch and pool have room.
          2. Prefill:   one forward pass per PREFILL request.
          3. Decode:    one batched forward pass over all DECODE requests.
          4. Eviction:  return finished requests' blocks to the pool.
          5. Pool state event: emit a snapshot so the visualiser can paint
             the memory bar after every step.
        """
        emitted: list[tuple[str, int]] = []
        self._step_idx += 1

        # --- 1. Admission --------------------------------------------------
        # Promote requests one at a time, checking BOTH constraints:
        # batch size and pool capacity. We scan the waiting queue from
        # the front; if a request doesn't fit, we don't skip it (FIFO
        # fairness -- avoid starvation of large requests behind small ones).
        # Real schedulers do more sophisticated bin packing here.
        while (
            self.waiting
            and len(self.active) < self.max_batch_size
        ):
            r = self.waiting[0]
            prefill_blocks, total_blocks = self._blocks_needed(r)
            if not self.pool.can_admit(total_blocks):
                # Not enough memory right now; surface why and stop scanning.
                # FIFO means we don't try later items in the queue this step.
                self._emit(events.request_waiting(
                    request_id=r.request_id,
                    reason=(
                        f"no free blocks: need {total_blocks}, "
                        f"{self.pool.num_free_blocks()} available"
                    ),
                ))
                break

            self.waiting.popleft()
            self.pool.admit_request(
                request_id=r.request_id,
                prefill_blocks_needed=prefill_blocks,
                total_blocks_needed=total_blocks,
            )
            r.cache = PagedRequestCache(
                pool=self.pool,
                request_id=r.request_id,
                num_layers=self.num_layers,
            )
            r.status = RequestStatus.PREFILL
            self.active.append(r)
            self._emit(events.request_admitted(
                request_id=r.request_id,
                prompt=r.prompt_text or "",
                prompt_len=int(r.prompt_ids.shape[1]),
            ))

        # --- 2. Prefill ----------------------------------------------------
        prefill_reqs = [r for r in self.active if r.status is RequestStatus.PREFILL]
        for r in prefill_reqs:
            prompt_len = int(r.prompt_ids.shape[1])
            self._emit(events.prefill_started(
                request_id=r.request_id,
                num_tokens=prompt_len,
            ))
            with torch.no_grad():
                logits = self.model(r.prompt_ids, kv_cache=r.cache)  # (1, S_prompt, V)
            next_id = int(torch.argmax(logits[0, -1, :]))
            r.generated_token_ids.append(next_id)
            emitted.append((r.request_id, next_id))
            # Per-token streaming hook. Prefill produces ONE token (the
            # first generated one) and that token must reach SSE
            # subscribers just like decode tokens do, so fire here too.
            if self.token_emitter is not None:
                self.token_emitter(
                    r.request_id,
                    next_id,
                    self._decode_token_str(next_id),
                    self._step_idx,
                )
            self._emit(events.prefill_done(
                request_id=r.request_id,
                # All prefill blocks are physically allocated at admit-time;
                # the block table length is the count.
                blocks_allocated=len(self.pool.get_block_table(r.request_id)),
            ))
            r.status = RequestStatus.DONE if r.is_finished() else RequestStatus.DECODE

        # --- 3. Decode (batched) ------------------------------------------
        decode_reqs = [r for r in self.active if r.status is RequestStatus.DECODE]
        if decode_reqs:
            input_ids = torch.tensor(
                [[r.last_token_id] for r in decode_reqs],
                dtype=torch.long,
                device=self.device,
            )
            caches = [r.cache for r in decode_reqs]
            with torch.no_grad():
                logits = self.model(input_ids, kv_cache=caches)  # (B, 1, V)

            # Collect the per-row results first so we can emit ONE
            # decode_step event covering the whole batch.
            batch_for_event: list[tuple[str, int, str]] = []
            for i, r in enumerate(decode_reqs):
                next_id = int(torch.argmax(logits[i, -1, :]))
                r.generated_token_ids.append(next_id)
                emitted.append((r.request_id, next_id))
                token_str = self._decode_token_str(next_id)
                batch_for_event.append((r.request_id, next_id, token_str))
                # Per-token streaming hook (see prefill emit for rationale).
                # Fired BEFORE the decode_step bus event so an SSE
                # consumer sees the token at roughly the same wall-clock
                # moment as the visualiser sees the batch event.
                if self.token_emitter is not None:
                    self.token_emitter(
                        r.request_id, next_id, token_str, self._step_idx,
                    )
                if r.is_finished():
                    r.status = RequestStatus.DONE
            self._emit(events.decode_step(
                step_idx=self._step_idx,
                batch=batch_for_event,
            ))

        # --- 4. Eviction --------------------------------------------------
        # Done requests release their blocks to the pool. This is what lets
        # waiting requests get admitted on a future step.
        still_active: list[Request] = []
        for r in self.active:
            if r.status is RequestStatus.DONE:
                # Capture stats BEFORE free_request clears the block table.
                total_tokens = len(r.generated_token_ids)
                reason = "eos" if (
                    r.eos_token_id is not None
                    and total_tokens > 0
                    and r.generated_token_ids[-1] == r.eos_token_id
                ) else "max_new_tokens"
                self.pool.free_request(r.request_id)
                self.finished.append(r)
                self._emit(events.request_finished(
                    request_id=r.request_id,
                    reason=reason,
                    total_tokens=total_tokens,
                    total_steps=self._step_idx,
                ))
            else:
                still_active.append(r)
        self.active = still_active

        # --- 5. Pool state snapshot ---------------------------------------
        # One event per step, AFTER eviction has freed blocks. The
        # visualiser polls this for the memory bar.
        free = len(self.pool._free_blocks)  # noqa: SLF001 -- friendly access
        self._emit(events.pool_state(
            free_blocks=free,
            used_blocks=self.pool.num_blocks - free,
            total_blocks=self.pool.num_blocks,
        ))

        return emitted
