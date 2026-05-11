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

Day 6 simplification: mixed prefill + decode batching.

  A real production scheduler decides per-step whether to do prefill, decode,
  or a mixed-shape forward. Mixing prefill (large q_len) and decode (q_len=1)
  in one forward pass needs an attention kernel that handles variable q_len
  per row, which is non-trivial without paged attention (Day 7).

  Our compromise here: prefill requests are processed sequentially -- one
  forward pass per prefill request -- and the decode requests are batched
  into ONE forward pass. So per step we do (n_prefill + 1) forward passes
  instead of 1. This is correct, simple, and still pays the speedup off
  the batched decode (which is the steady state once requests pile up).

Day 7 will replace SimpleKVCache here with a paged pool + admission control.

Public surface:

  scheduler.add_request(id, prompt_ids, max_new_tokens, eos_token_id)
  scheduler.step() -> list[(request_id, token_id)]
  scheduler.has_work() -> bool
  scheduler.get_finished() -> list[Request]
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from src.engine.kv_cache import SimpleKVCache

if TYPE_CHECKING:
    from src.engine.model import LlamaModel


from enum import Enum, auto


class RequestStatus(Enum):
    """Where a request is in its lifecycle.

    Transitions:
        WAITING --admit (batch has room)--> PREFILL
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

    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    cache: SimpleKVCache | None = None      # allocated at admission

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
    def __init__(self, model: "LlamaModel", max_batch_size: int) -> None:
        """
        Args:
            model: the LlamaModel.
            max_batch_size: max concurrent requests in the active batch.
                This is the COMPUTE budget (rows in one forward pass).

        Day 6: only compute budget. Day 7 adds a memory budget on top
        (paged KV pool with explicit admission control).
        """
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.max_batch_size = max_batch_size

        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.finished: list[Request] = []

    # ---- Public API ---------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> None:
        """Enqueue a new request. Admission happens at the next step() if
        the batch has room."""
        if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(
                f"prompt_ids must be shape (1, S); got {tuple(prompt_ids.shape)}"
            )
        self.waiting.append(Request(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ))

    def has_work(self) -> bool:
        return bool(self.waiting or self.active)

    def get_finished(self) -> list[Request]:
        out = self.finished
        self.finished = []
        return out

    # ---- Core loop ----------------------------------------------------

    def step(self) -> list[tuple[str, int]]:
        """One engine iteration. Returns every token emitted this step.

        Phases:
          1. Admission: WAITING -> PREFILL while batch has room.
          2. Prefill:   one forward pass per PREFILL request.
          3. Decode:    one batched forward pass over all DECODE requests.
          4. Eviction:  drop finished requests from the active list.
        """
        emitted: list[tuple[str, int]] = []

        # --- 1. Admission --------------------------------------------------
        # Promote waiting requests into the active batch until we hit the
        # compute budget. FIFO order -- first request queued is first admitted.
        while self.waiting and len(self.active) < self.max_batch_size:
            r = self.waiting.popleft()
            r.cache = SimpleKVCache(num_layers=self.num_layers)
            r.status = RequestStatus.PREFILL
            self.active.append(r)

        # --- 2. Prefill ----------------------------------------------------
        prefill_reqs = [r for r in self.active if r.status is RequestStatus.PREFILL]
        for r in prefill_reqs:
            with torch.no_grad():
                logits = self.model(r.prompt_ids, kv_cache=r.cache)  # (1, S_prompt, V)
            next_id = int(torch.argmax(logits[0, -1, :]))
            r.generated_token_ids.append(next_id)
            emitted.append((r.request_id, next_id))
            r.status = RequestStatus.DONE if r.is_finished() else RequestStatus.DECODE

        # --- 3. Decode (batched) ------------------------------------------
        decode_reqs = [r for r in self.active if r.status is RequestStatus.DECODE]
        if decode_reqs:
            input_ids = torch.tensor(
                [[r.last_token_id] for r in decode_reqs],
                dtype=torch.long,
            )
            caches = [r.cache for r in decode_reqs]
            with torch.no_grad():
                logits = self.model(input_ids, kv_cache=caches)  # (B, 1, V)

            for i, r in enumerate(decode_reqs):
                next_id = int(torch.argmax(logits[i, -1, :]))
                r.generated_token_ids.append(next_id)
                emitted.append((r.request_id, next_id))
                if r.is_finished():
                    r.status = RequestStatus.DONE

        # --- 4. Eviction --------------------------------------------------
        # Drop done requests from the active list. They keep their cache
        # objects (just for the finished bucket); GC reclaims when the
        # scheduler hands them back via get_finished().
        still_active: list[Request] = []
        for r in self.active:
            if r.status is RequestStatus.DONE:
                self.finished.append(r)
            else:
                still_active.append(r)
        self.active = still_active

        return emitted
