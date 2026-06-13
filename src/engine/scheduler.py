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

    # Optional LoRA adapter to serve this request under. None == base model.
    # Routed to the model right before each forward when the model supports
    # adapter switching (LoRALlamaModel); ignored by a plain LlamaModel.
    adapter_id: str | None = None

    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    cache: PagedRequestCache | None = None      # allocated at admission
    # How many tokens of the prompt are already in the cache from a
    # prefix-cache hit at admit time. The prefill forward pass skips
    # those tokens (slices prompt_ids[:, hit_boundary:]) and the
    # per-layer seq_len is pre-seeded to this value. 0 means no hit
    # (or prefix caching disabled); behaves identically to pre-Day-12.
    prefill_hit_boundary: int = 0
    # Chunked-prefill progress. Allocated at admission alongside the cache.
    # While the request is in PREFILL this tracks how much of the prompt has
    # been pushed through the model across successive scheduler iterations.
    chunk_state: "ChunkedPrefillRequest | None" = None

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


@dataclass
class ChunkedPrefillRequest:
    """Chunked-prefill bookkeeping for one request.

    A prompt longer than the scheduler's ``chunk_size`` is prefilled across
    several iterations instead of in one monopolising forward pass. This holds
    the resume point so each step knows which slice of the prompt to run next.

    ``prefilled_so_far`` is an ABSOLUTE token count (it starts at the prefix-
    cache hit boundary, not 0, when admit-time sharing already seeded part of
    the cache). It equals the per-layer ``cache.seq_len`` between iterations,
    which is exactly the RoPE position offset the model reads for the next
    chunk -- so chunking is mathematically identical to a single-shot prefill.
    """
    request: "Request"
    prefilled_so_far: int = 0   # prompt tokens already written to the KV cache
    chunk_index: int = 0        # number of chunks run == index of the NEXT chunk

    @property
    def prompt_len(self) -> int:
        return int(self.request.prompt_ids.shape[1])

    @property
    def tokens_remaining(self) -> int:
        return self.prompt_len - self.prefilled_so_far

    @property
    def is_complete(self) -> bool:
        return self.prefilled_so_far >= self.prompt_len


class ContinuousBatchScheduler:
    def __init__(
        self,
        model: "LlamaModel",
        max_batch_size: int,
        num_blocks: int,
        block_size: int = 16,
        chunk_size: int = 256,
        event_bus: "EventBus | None" = None,
        token_decoder: Callable[[int], str] | None = None,
        token_emitter: Callable[[str, int, str, int], None] | None = None,
        enable_prefix_cache: bool = True,
        enable_spec_decode: bool = False,
        spec_decode_k: int = 4,
        spec_decode_exit_layer: int = 8,
        spec_decode_observer: Callable[[int, int], None] | None = None,
        use_cuda_graphs: bool = True,
        cuda_graph_observer: Callable[[bool], None] | None = None,
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
            chunk_size: the per-iteration TOKEN budget for chunked prefill
                (vLLM v2 / SGLang style). Default 256. Each step, decode
                requests are scheduled first (one token each, latency
                priority); whatever budget remains is spent prefilling the
                next waiting prompt, up to `chunk_size - num_decode_tokens`
                tokens. A prompt longer than the remaining budget is split
                across multiple iterations so it can't monopolise the GPU and
                stall decode. A prompt that fits the budget prefills in one
                chunk -- byte-identical to the pre-chunking full-prefill path,
                which is why existing parity tests are unaffected.
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
            enable_prefix_cache: when True (default), the prefill path
                hashes each full prompt block and asks the pool to
                share with any previously-cached identical block.
                Tokens covered by hits skip the prefill forward
                compute. When False, every prefill block is allocated
                fresh -- byte-identical to pre-Day-12 behaviour. The
                parity tests construct schedulers with both settings
                and assert identical outputs.
            enable_spec_decode: when True, the DECODE phase uses
                self-speculative decoding (draft K tokens via early-exit
                forward, then verify with a single full forward) for
                requests that are running ALONE in the decode batch. Under
                greedy sampling this produces byte-identical output to
                vanilla decode -- the parity test enforces that. v0.3
                limitation: when 2+ requests are simultaneously in DECODE,
                the scheduler falls back to vanilla batched decode for
                that step. Default False so existing tests run untouched.
            spec_decode_k: number of draft tokens per round when
                speculative decoding is enabled. K=4 is the vLLM default
                and a good fit for our acceptance-rate regime.
            spec_decode_observer: optional callable
                `(accepted_count, k) -> None` fired once per spec_decode
                round. The metrics layer uses this to populate the
                acceptance-rate histogram. None disables observation; the
                spec decode path runs identically either way.
            use_cuda_graphs: when True (default), the batched-decode forward is
                routed through a CUDAGraphRunner whenever a graph matching the
                current decode batch (size, bound caches, seq_len) has been
                captured -- collapsing the step's hundreds of kernel launches
                into one graph replay. When no graph matches, or on CPU, the
                step runs eagerly and is byte-identical to the pre-graph path
                (so every existing CPU test is unaffected). A runner is attached
                via `self._graph_runner`; see src/engine/cuda_graph.py for why
                live growing-length decode is served eagerly and the win is
                demonstrated in the fixed-length benchmark.

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
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = chunk_size
        self.event_bus = event_bus
        self.token_decoder = token_decoder
        self.token_emitter = token_emitter
        self.enable_prefix_cache = enable_prefix_cache
        # Speculative-decoding configuration. The flags are inert unless
        # the DECODE phase finds exactly one request in DECODE status,
        # because batched spec decode is a v0.4 problem (per-row K/V
        # truncation in a batched cache layout).
        self.enable_spec_decode = enable_spec_decode
        self.spec_decode_k = spec_decode_k
        # Depth of the early-exit draft path. 8 is the Day-15 default; the
        # v0.4 probing (Day 16) tries deeper layers to see if a higher
        # acceptance rate can offset the extra draft cost. The breakeven
        # rule is alpha > exit_layer/total_layers, so going deeper only
        # helps if acceptance scales faster than depth.
        self.spec_decode_exit_layer = spec_decode_exit_layer
        self.spec_decode_observer = spec_decode_observer

        # CUDA-graph decode acceleration. The flag gates routing; the runner is
        # attached lazily/externally (e.g. by a benchmark or a serving setup
        # that pre-captures graphs). It stays None during ordinary scheduler
        # use, so the decode path is the eager one and existing tests see no
        # behavioural change. See _decode_forward.
        self.use_cuda_graphs = use_cuda_graphs
        self._graph_runner = None  # type: ignore[assignment]
        # Optional observability hook: called once per batched-decode forward
        # with True if a captured CUDA graph was replayed, False if we fell back
        # to eager. Stays None during ordinary use (no metrics dependency in the
        # engine); the server wires metrics.observe_cuda_graph into it.
        self._cuda_graph_observer = cuda_graph_observer

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
            enable_prefix_cache=enable_prefix_cache,
        )

        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.finished: list[Request] = []
        # Monotonic step counter. Exposed via the decode_step event so the
        # visualiser can align decode bursts to a timeline.
        self._step_idx: int = 0
        # Optional StepProfiler. When attached, step() marks the prefill /
        # decode / kv_alloc / overhead phase boundaries so the auto-tuner can
        # see where time goes. None (the default) makes every _prof_* call a
        # no-op, so existing tests run byte-identically.
        self.profiler = None

    # ---- Helpers (event emission lives here so the core loop stays clean) ----

    def _emit(self, event: events.Event) -> None:
        """Fire an event if a bus is attached; otherwise no-op."""
        if self.event_bus is not None:
            self.event_bus.emit(event)

    # ---- profiler marks (no-ops unless a StepProfiler is attached) -------

    def _prof_begin(self) -> None:
        if self.profiler is not None:
            self.profiler.begin_step()

    def _prof_lap(self, label: str) -> None:
        if self.profiler is not None:
            self.profiler.lap(label)

    def _prof_end(self) -> None:
        if self.profiler is not None:
            self.profiler.end_step()

    def _decode_token_str(self, token_id: int) -> str:
        """Resolve a token id to display text. Empty string if no decoder."""
        if self.token_decoder is None:
            return ""
        try:
            return self.token_decoder(token_id)
        except Exception:
            return ""

    def _route_adapters(self, reqs: list) -> None:
        """Set per-row LoRA adapters for the upcoming forward, in `reqs` order.

        No-op unless the model exposes `set_batch_adapters` (i.e. it is a
        LoRALlamaModel). The list aligns 1:1 with the batch rows the caller is
        about to feed the model, so row i is served under reqs[i].adapter_id
        (None == base). A plain LlamaModel has no such method, so this is a
        guarded no-op and the non-LoRA path is completely unchanged.
        """
        route = getattr(self.model, "set_batch_adapters", None)
        if route is not None:
            route([r.adapter_id for r in reqs])

    def _decode_forward(
        self,
        input_ids: torch.Tensor,
        caches: list,
    ) -> torch.Tensor:
        """One batched-decode forward, graph-accelerated when possible.

        Routes through the attached CUDAGraphRunner iff cuda graphs are enabled,
        we're on CUDA, and the runner has a graph captured for EXACTLY this
        decode batch (size + these cache objects + matching seq_len). Otherwise
        runs the model eagerly. The eager branch is byte-identical to the
        pre-graph code path, so on CPU (and whenever no graph matches) behaviour
        is unchanged.
        """
        if (
            self.use_cuda_graphs
            and self.device.type == "cuda"
            and self._graph_runner is not None
            and self._graph_runner.can_replay(len(caches), caches)
        ):
            if self._cuda_graph_observer is not None:
                self._cuda_graph_observer(True)   # graph hit
            # The graph runs under its own captured no_grad context.
            return self._graph_runner.replay(input_ids, caches)
        if self._cuda_graph_observer is not None:
            self._cuda_graph_observer(False)      # eager fallback
        with torch.no_grad():
            return self.model(input_ids, kv_cache=caches)

    # ---- Public API ---------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        prompt_text: str | None = None,
        adapter_id: str | None = None,
    ) -> None:
        """Enqueue a new request. Admission happens at the next step() if
        BOTH the batch has room AND the pool has enough free blocks for
        this request's worst-case footprint.

        prompt_text is optional and only used by the request_admitted
        event payload for visualiser display.

        adapter_id optionally selects a LoRA adapter to serve this request
        under; it is routed to the model per forward when the model is a
        LoRALlamaModel, and ignored otherwise.
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
            adapter_id=adapter_id,
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

    def _compute_block_hashes(
        self,
        prompt_ids: torch.Tensor,
        n_prefill_blocks: int,
    ) -> list[int | None]:
        """Build the per-prefill-block hash list for prefix caching.

        Each entry is either an int (this block is eligible for sharing
        and will be looked up in the pool's hash_to_block) or None
        (force fresh allocation -- partial-tail block, or the last
        full block of an exact-multiple prompt where prefill needs
        a writable slot).

        Hash chain (matches the design we agreed on):
            block_hash_i = hash((
                prev_block_hash,    # 0 for block 0; the chain
                tuple(tokens),      # the 16 tokens in this block
                start_position,     # explicit position, redundant
                                    # given the chain but kept for
                                    # belt-and-suspenders safety
            ))

        Shareability rule:
            n_full = prompt_len // block_size
            max_shareable = n_full - (1 if prompt_len % bs == 0 else 0)
            max_shareable = max(0, max_shareable)
        We force the last full block to be fresh when the prompt has no
        partial tail, otherwise prefill forward has no token to run on
        (no Q -> no next-token logits) AND no fresh block to write its
        own K/V into.

        When prefix caching is disabled at the scheduler level, this
        returns all-None and the pool's allocation path takes the
        pre-Day-12 fast lane.
        """
        if not self.enable_prefix_cache:
            return [None] * n_prefill_blocks

        bs = self.block_size
        prompt_len = int(prompt_ids.shape[1])
        n_full = prompt_len // bs
        has_partial = (prompt_len % bs) != 0
        max_shareable = n_full - (0 if has_partial else 1)
        max_shareable = max(0, max_shareable)

        # Pull the tokens once on the CPU side. tolist() is the cheap way
        # to get hashable Python ints from a torch tensor (we don't want
        # to hash the tensor itself -- equality semantics differ).
        ids_list: list[int] = prompt_ids[0].detach().cpu().tolist()

        hashes: list[int | None] = []
        prev_hash = 0  # sentinel: block-0 chains from this
        for i in range(n_prefill_blocks):
            if i < max_shareable:
                start = i * bs
                chunk = tuple(ids_list[start:start + bs])
                prev_hash = hash((prev_hash, chunk, start))
                hashes.append(prev_hash)
            else:
                # Either the partial tail (last entry of a non-aligned
                # prompt) or the forced-fresh last full block. No hash
                # recorded, no sharing.
                hashes.append(None)
        return hashes

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
        self._prof_begin()

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
            # Compute per-prefill-block hashes. With the chained hash,
            # once a block is a "fresh" slot (None) or the lookup misses
            # the cache, every subsequent block is also a miss (its
            # chain history is unique). So the run of hits is always a
            # prefix of the prefill block list -- which is exactly what
            # we need for slicing the prefill forward pass below.
            prefill_block_hashes = self._compute_block_hashes(
                r.prompt_ids, prefill_blocks,
            )
            cached_blocks = self.pool.admit_request(
                request_id=r.request_id,
                prefill_blocks_needed=prefill_blocks,
                total_blocks_needed=total_blocks,
                prefill_block_hashes=prefill_block_hashes,
            )
            # hit_boundary: how many tokens of the prompt are already
            # cached. Because hits form a contiguous prefix of the
            # prefill block list (chained-hash property), the boundary
            # is just cached_blocks * block_size. Prefill forward will
            # run on prompt_ids[:, hit_boundary:] and the model's RoPE
            # offset is read from the per-request cache's seq_len.
            hit_boundary = cached_blocks * self.block_size
            r.cache = PagedRequestCache(
                pool=self.pool,
                request_id=r.request_id,
                num_layers=self.num_layers,
            )
            if hit_boundary > 0:
                # Pre-seed the per-layer seq_lens so that the prefill
                # forward sees the right RoPE offset AND appends K/V
                # only for the suffix. The K/V for positions
                # [0, hit_boundary) was written to these same physical
                # blocks by an earlier request and is still in the
                # pool tensor -- cache.get() at SDPA time picks it up
                # via the block table.
                for layer_idx in range(self.num_layers):
                    r.cache._seq_lens[layer_idx] = hit_boundary
            # Stash the boundary on the request so the prefill phase
            # can slice without recomputing it.
            r.prefill_hit_boundary = hit_boundary
            # Initialise chunked-prefill progress. We start from the prefix-
            # cache hit boundary: those tokens are already in the cache, so
            # chunking only ever runs over the uncached suffix.
            r.chunk_state = ChunkedPrefillRequest(
                request=r,
                prefilled_so_far=hit_boundary,
                chunk_index=0,
            )
            r.status = RequestStatus.PREFILL
            self.active.append(r)
            self._emit(events.request_admitted(
                request_id=r.request_id,
                prompt=r.prompt_text or "",
                prompt_len=int(r.prompt_ids.shape[1]),
                cached_blocks=cached_blocks,
                total_prefill_blocks=prefill_blocks,
            ))

        self._prof_lap("kv_alloc")  # admission/block-allocation = the memory phase

        # --- 2. Chunked prefill --------------------------------------------
        # vLLM-v2 / SGLang scheduling: decode requests get latency priority,
        # so they spend the iteration's token budget FIRST. Whatever is left
        # of `chunk_size` is used to prefill the next waiting prompt(s), a
        # chunk at a time. A long prompt is split across iterations instead of
        # hogging one giant forward pass -- that's the whole point: decode
        # requests interleave between a long prompt's chunks instead of
        # stalling behind it.
        #
        # `num_decode_tokens` is the count of requests ALREADY in DECODE at
        # step entry (one new token each). A request that finishes prefill
        # THIS step also decodes this step (section 3 re-scans), but that
        # bonus token isn't charged here -- it can't be known until its last
        # chunk runs. The budget is a soft cap, not an exact ledger.
        num_decode_tokens = sum(
            1 for r in self.active if r.status is RequestStatus.DECODE
        )
        remaining_budget = self.chunk_size - num_decode_tokens

        prefill_reqs = [r for r in self.active if r.status is RequestStatus.PREFILL]
        for r in prefill_reqs:
            if remaining_budget <= 0:
                # Budget spent by decode (+ earlier prefill chunks this step).
                # This request stays PREFILL and resumes next iteration --
                # exactly the stall-prevention behaviour we want.
                break

            cs = r.chunk_state
            prompt_len = cs.prompt_len
            tokens_left = cs.tokens_remaining
            if tokens_left <= 0:
                # Defensive: an already-complete prefill shouldn't be here.
                continue

            chunk_tokens = min(remaining_budget, tokens_left)
            start = cs.prefilled_so_far
            is_first_chunk = cs.chunk_index == 0
            completes = (start + chunk_tokens) >= prompt_len

            # Keep the legacy bracketing events: prefill_started on the FIRST
            # chunk, prefill_done on the LAST. Consumers (metrics TTFT, the
            # events test) see the same start/done pair as before; for a
            # prompt that fits one chunk this is byte-identical to the old
            # single-shot prefill.
            if is_first_chunk:
                self._emit(events.prefill_started(
                    request_id=r.request_id,
                    num_tokens=prompt_len,
                ))
            self._emit(events.prefill_chunk_start(
                request_id=r.request_id,
                chunk_index=cs.chunk_index,
                tokens_in_chunk=chunk_tokens,
                prefilled_so_far=start,
                prompt_len=prompt_len,
            ))

            # Run the forward on just this slice. The KV cache's per-layer
            # seq_len is already `start` (seeded to the prefix-cache boundary
            # at admit, then advanced by each prior chunk's append), so RoPE
            # positions and the causal mask come out exactly as a single-shot
            # prefill would -- the model's sliced-prefill path (S>1,
            # pos_offset>0) handles it. Intermediate chunks' logits are
            # discarded; only the final chunk's last position yields a token.
            forward_input = r.prompt_ids[:, start:start + chunk_tokens]
            # Route this single request's adapter (batch row 0) before prefill.
            self._route_adapters([r])
            with torch.no_grad():
                logits = self.model(forward_input, kv_cache=r.cache)  # (1, chunk, V)

            cs.prefilled_so_far += chunk_tokens
            cs.chunk_index += 1
            remaining_budget -= chunk_tokens

            self._emit(events.prefill_chunk_done(
                request_id=r.request_id,
                chunk_index=cs.chunk_index - 1,
                tokens_in_chunk=chunk_tokens,
                prefilled_so_far=cs.prefilled_so_far,
                prompt_len=prompt_len,
            ))

            if not completes:
                # More chunks to come; stays PREFILL, resumes next iteration.
                continue

            # Last chunk: the prompt is fully in the cache. Sample the first
            # generated token off the final position's logits -- this is the
            # request's first emitted token, exactly as before.
            next_id = int(torch.argmax(logits[0, -1, :]))
            r.generated_token_ids.append(next_id)
            emitted.append((r.request_id, next_id))
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

        self._prof_lap("prefill")

        # --- 3. Decode -----------------------------------------------------
        # Two code paths share this slot:
        #   * Vanilla batched decode: one forward over the whole decode
        #     batch. Always correct, used when speculative decoding is
        #     disabled OR when 2+ requests are in DECODE this step.
        #   * Speculative decode (single-request only in v0.3): early-exit
        #     drafts K tokens, then a verify forward accepts a prefix and
        #     emits 1..K+1 tokens per step. Byte-identical to vanilla
        #     greedy by construction; the parity test enforces it.
        decode_reqs = [r for r in self.active if r.status is RequestStatus.DECODE]
        if decode_reqs:
            use_spec_decode = (
                self.enable_spec_decode and len(decode_reqs) == 1
            )
            if use_spec_decode:
                # Lazy import to keep the scheduler module load-light when
                # spec decode is disabled (the import pulls in nothing
                # exotic, but the lazy form documents the conditional
                # dependency).
                from src.engine.spec_decode import spec_decode_step

                r = decode_reqs[0]
                # Budget = how many more tokens this request is allowed to
                # emit before hitting its cap. spec_decode_step uses this
                # to truncate its emit list so we never overshoot.
                remaining = r.max_new_tokens - len(r.generated_token_ids)
                emit_list, accepted = spec_decode_step(
                    model=self.model,
                    request_cache=r.cache,
                    last_token_id=r.last_token_id,
                    k=self.spec_decode_k,
                    eos_token_id=r.eos_token_id,
                    max_emit=remaining,
                    n_draft_layers=self.spec_decode_exit_layer,
                )
                # Observer hook (metrics). Fires once per round with the
                # raw acceptance count -- truncation by EOS or budget
                # would deflate the speedup signal, so we observe the
                # untruncated `accepted` (out of self.spec_decode_k).
                if self.spec_decode_observer is not None:
                    self.spec_decode_observer(accepted, self.spec_decode_k)

                batch_for_event: list[tuple[str, int, str]] = []
                for next_id in emit_list:
                    r.generated_token_ids.append(next_id)
                    emitted.append((r.request_id, next_id))
                    token_str = self._decode_token_str(next_id)
                    batch_for_event.append((r.request_id, next_id, token_str))
                    # Per-token streaming hook fires per emitted token, so
                    # SSE consumers see the same number of token events as
                    # tokens emitted -- consistent with the vanilla path
                    # even when one step yields multiple tokens.
                    if self.token_emitter is not None:
                        self.token_emitter(
                            r.request_id, next_id, token_str, self._step_idx,
                        )
                # Status update after appending all spec-emitted tokens.
                # is_finished() now reflects the last token in the list
                # (which is either the truncating EOS or the budget cap,
                # if either applied -- spec_decode_step truncates).
                if r.is_finished():
                    r.status = RequestStatus.DONE
                # One decode_step event for the whole spec round, listing
                # every emitted token. The visualiser already handles
                # multiple entries per request in the batch field.
                self._emit(events.decode_step(
                    step_idx=self._step_idx,
                    batch=batch_for_event,
                ))
            else:
                # --- Vanilla batched decode (the pre-Day-15 path) -------
                input_ids = torch.tensor(
                    [[r.last_token_id] for r in decode_reqs],
                    dtype=torch.long,
                    device=self.device,
                )
                caches = [r.cache for r in decode_reqs]
                # Mixed-adapter batching: route each row to its request's adapter
                # (in decode_reqs order, matching input_ids/caches) before the
                # batched forward. No-op on a plain LlamaModel.
                self._route_adapters(decode_reqs)
                logits = self._decode_forward(input_ids, caches)  # (B, 1, V)

                # Collect the per-row results first so we can emit ONE
                # decode_step event covering the whole batch.
                batch_for_event = []
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

        self._prof_lap("decode")

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
        # visualiser polls this for the memory bar; the metrics layer
        # turns it into the POOL_BLOCKS_* gauges. cached_blocks is the
        # count of physically shared blocks (ref_count >= 2) -- the
        # prefix cache's live footprint.
        free = len(self.pool._free_blocks)  # noqa: SLF001 -- friendly access
        self._emit(events.pool_state(
            free_blocks=free,
            used_blocks=self.pool.num_blocks - free,
            total_blocks=self.pool.num_blocks,
            cached_blocks=self.pool.num_shared_blocks(),
            waiting=len(self.waiting),
        ))

        # Eviction + pool-state emission counts as overhead (the trailing
        # interval before end_step closes the step).
        self._prof_end()
        return emitted
