"""
Structured event stream from the inference engine.

Why this exists:

  The scheduler makes a lot of decisions every step -- admit a request,
  refuse a request because the pool is full, allocate a block, finish a
  request, etc. Internally they're just attribute mutations; from outside
  they're invisible. To build a visualiser (Day 9) or a debugger, or to
  log behavior in production, we need those decisions surfaced as
  STRUCTURED data, not formatted log strings.

  This module owns:
    * Event       -- a single dataclass shape: (timestamp, event_type, payload).
    * Factory functions per event type -- they construct an Event with
      the right `event_type` string and a `payload` dict shaped for that
      type. Keeping the wire shape uniform makes JSON serialisation and
      the WebSocket consumer trivial.
    * EventBus    -- a sync, thread-safe pub/sub. emit() runs every
      subscriber's callback inline. Subscribers are sync callables; the
      bus does NOT know about asyncio. Bridging to async (WebSocket
      sends) is done at the subscriber level via
      `loop.call_soon_threadsafe(queue.put_nowait, evt)`.

Threading model:

  emit() is called from whichever thread the scheduler is running in
  (in our server it's a FastAPI threadpool worker). Subscribers may be
  added or removed from any thread. A single Lock guards the subscriber
  list; iteration in emit() runs under the lock, so callbacks themselves
  must be fast (they should hand off the event and return). A callback
  raising an exception is logged-and-swallowed so a sick subscriber
  cannot take down the engine.

  We deliberately do NOT spawn worker threads inside the bus. Subscribers
  that need to bridge to async event loops do so themselves; that's the
  WebSocket /events endpoint's job.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The wire-shape event.
# ---------------------------------------------------------------------------
#
# One dataclass for every event; the `event_type` string discriminates.
# Two reasons not to subclass per type:
#   1. Subclasses can't be JSON-serialised polymorphically without custom
#      encoder logic. A flat (type, payload) dict round-trips trivially.
#   2. The WebSocket consumer is a JS visualiser; it doesn't have access
#      to Python class identity anyway. The type tag is what it switches on.
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """A single event. Uniform shape across all event types.

    Attributes:
        event_type: A short string tag. Consumers switch on this.
        payload: Type-specific data. Must be JSON-serialisable (dicts,
            lists, ints, strings, floats, bools, None).
        timestamp: Wall-clock time when the event was constructed.
    """
    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view. The WebSocket /events serialiser calls this."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Factory functions per event type.
# ---------------------------------------------------------------------------
#
# These exist so callers don't have to memorise payload-dict shapes. The
# scheduler emits, e.g., `bus.emit(request_admitted(rid, prompt, plen))`
# and the right Event falls out. If a payload field shape changes, you
# change it here in one place.
# ---------------------------------------------------------------------------


def request_admitted(
    request_id: str,
    prompt: str,
    prompt_len: int,
    cached_blocks: int = 0,
    total_prefill_blocks: int = 0,
) -> Event:
    """Request entered the active batch.

    cached_blocks / total_prefill_blocks expose the prefix-cache hit
    rate per-request: consumers compute `cached_blocks /
    total_prefill_blocks` for a per-request rate, or aggregate across
    admissions for an engine-wide rate. The fields default to 0 so
    callers that don't care (and pre-Day-12 tests) keep working.
    """
    return Event(
        event_type="request_admitted",
        payload={
            "request_id": request_id,
            "prompt": prompt,
            "prompt_len": prompt_len,
            "cached_blocks": cached_blocks,
            "total_prefill_blocks": total_prefill_blocks,
        },
    )


def request_waiting(request_id: str, reason: str) -> Event:
    return Event(
        event_type="request_waiting",
        payload={"request_id": request_id, "reason": reason},
    )


def prefill_started(request_id: str, num_tokens: int) -> Event:
    return Event(
        event_type="prefill_started",
        payload={"request_id": request_id, "num_tokens": num_tokens},
    )


def prefill_done(request_id: str, blocks_allocated: int) -> Event:
    return Event(
        event_type="prefill_done",
        payload={"request_id": request_id, "blocks_allocated": blocks_allocated},
    )


def prefill_chunk_start(
    request_id: str,
    chunk_index: int,
    tokens_in_chunk: int,
    prefilled_so_far: int = 0,
    prompt_len: int = 0,
) -> Event:
    """A single chunk of a chunked prefill is about to run.

    Chunked prefill splits a long prompt across several scheduler iterations
    so a big prompt can't monopolise the GPU and starve decode requests. Each
    chunk emits this on entry.

    Fields:
        chunk_index: 0-based index of this chunk within the request's prefill.
        tokens_in_chunk: how many prompt tokens this chunk processes.
        prefilled_so_far: prompt tokens already in the KV cache BEFORE this
            chunk (so a consumer can paint an X/total progress bar; this chunk
            will advance it to prefilled_so_far + tokens_in_chunk).
        prompt_len: total prompt length, the denominator of the progress bar.
    """
    return Event(
        event_type="prefill_chunk_start",
        payload={
            "request_id": request_id,
            "chunk_index": chunk_index,
            "tokens_in_chunk": tokens_in_chunk,
            "prefilled_so_far": prefilled_so_far,
            "prompt_len": prompt_len,
        },
    )


def prefill_chunk_done(
    request_id: str,
    chunk_index: int,
    tokens_in_chunk: int,
    prefilled_so_far: int = 0,
    prompt_len: int = 0,
) -> Event:
    """A chunk of a chunked prefill just finished.

    ``prefilled_so_far`` here is the count AFTER this chunk's tokens landed in
    the cache (== the previous prefilled_so_far + tokens_in_chunk). When it
    equals ``prompt_len`` the prefill is complete and the request transitions
    to DECODE (a ``prefill_done`` event follows).
    """
    return Event(
        event_type="prefill_chunk_done",
        payload={
            "request_id": request_id,
            "chunk_index": chunk_index,
            "tokens_in_chunk": tokens_in_chunk,
            "prefilled_so_far": prefilled_so_far,
            "prompt_len": prompt_len,
        },
    )


def decode_step(step_idx: int, batch: list[tuple[str, int, str]]) -> Event:
    """One batched decode iteration. `batch` is [(request_id, token_id, token_str)]."""
    return Event(
        event_type="decode_step",
        payload={
            "step_idx": step_idx,
            "batch": [
                {"request_id": rid, "token_id": tid, "token_str": tstr}
                for rid, tid, tstr in batch
            ],
        },
    )


def block_allocated(
    request_id: str,
    physical_block_idx: int,
    logical_idx: int,
    shared: bool = False,
) -> Event:
    """A logical block at `logical_idx` was bound to a physical block.

    `shared=True` means this binding is a prefix-cache hit -- the
    physical block already held K/V from a prior request and we
    incremented its refcount instead of pulling from the free pool.
    Default False so existing call sites (decode-time JIT growth) keep
    their previous semantics without naming the new field.
    """
    return Event(
        event_type="block_allocated",
        payload={
            "request_id": request_id,
            "physical_block_idx": physical_block_idx,
            "logical_idx": logical_idx,
            "shared": shared,
        },
    )


def block_freed(request_id: str, physical_block_idx: int) -> Event:
    return Event(
        event_type="block_freed",
        payload={
            "request_id": request_id,
            "physical_block_idx": physical_block_idx,
        },
    )


def request_finished(
    request_id: str, reason: str, total_tokens: int, total_steps: int
) -> Event:
    return Event(
        event_type="request_finished",
        payload={
            "request_id": request_id,
            "reason": reason,
            "total_tokens": total_tokens,
            "total_steps": total_steps,
        },
    )


def pool_state(
    free_blocks: int,
    used_blocks: int,
    total_blocks: int,
    cached_blocks: int = 0,
    waiting: int = 0,
) -> Event:
    """Snapshot of the KV-cache pool, emitted once per scheduler step.

    cached_blocks counts physically SHARED blocks (ref_count >= 2) --
    the blocks the prefix cache is actively deduplicating. It is a
    subset of used_blocks (used_blocks is everything not free), so a
    consumer wanting "uniquely owned" blocks computes
    used_blocks - cached_blocks. Defaults to 0 so pre-Day-13 callers
    and tests still construct valid events.

    waiting is the number of requests queued for admission (not yet in
    the active batch) at the moment of the snapshot -- the engine's
    backlog. Added for the observability stack's queue_depth gauge;
    defaults to 0 so older callers and hand-built events stay valid.
    """
    return Event(
        event_type="pool_state",
        payload={
            "free_blocks": free_blocks,
            "used_blocks": used_blocks,
            "total_blocks": total_blocks,
            "cached_blocks": cached_blocks,
            "waiting": waiting,
        },
    )


# ---------------------------------------------------------------------------
# The pub/sub bus.
# ---------------------------------------------------------------------------
#
# A subscriber is a sync callable `(Event) -> None`. The bus calls each
# subscriber inline from emit(). If a subscriber needs to deliver to an
# async loop (the WebSocket case), it captures the loop at subscribe time
# and uses `loop.call_soon_threadsafe(queue.put_nowait, evt)` inside the
# callback. The bus stays sync; the bridge happens at the edge.
# ---------------------------------------------------------------------------


Subscriber = Callable[[Event], None]


class EventBus:
    """Sync, thread-safe pub/sub. One instance per running engine."""

    def __init__(self) -> None:
        # Lock guards the subscriber list against concurrent
        # subscribe / unsubscribe / emit.
        self._lock = threading.Lock()
        self._subscribers: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> None:
        """Register `callback` to receive every subsequent event.

        Adding the same callback twice will deliver each event twice;
        callers should track their own subscription identity if that
        matters.
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        """Remove `callback`. Silent no-op if not present (idempotent disconnect)."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def emit(self, event: Event) -> None:
        """Fan out `event` to every subscriber.

        Subscribers run under the bus's lock. Each one should be a fast,
        non-blocking handoff (e.g. push onto an asyncio queue). A
        subscriber raising is logged and swallowed so one bad consumer
        cannot break the engine.
        """
        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                # Don't let a subscriber crash kill the engine. The
                # subscriber owns its own error reporting.
                log.exception("EventBus subscriber raised on %s", event.event_type)
