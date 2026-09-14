"""
Disaggregated prefill/decode -- a single-process simulation of the Mooncake
(2024) architecture.

WHY DISAGGREGATE
----------------
Prefill and decode have opposite hardware profiles:

  * PREFILL processes the whole prompt in one forward. It is COMPUTE-bound: big
    matmuls over many tokens, the GPU's FLOPs are the limit.
  * DECODE emits one token at a time. It is MEMORY-BANDWIDTH-bound: tiny matmuls,
    the limit is hauling the weights + KV cache through memory each step.

In a UNIFIED worker the two fight each other. A long prompt's prefill forward
monopolises the GPU, so the decode requests sharing that worker stall (you can
see this directly in benchmark_chunked_prefill.py: "max decode stall"). Chunked
prefill softens it, but the contention is fundamental.

Mooncake's answer: run them on SEPARATE workers. A PREFILL worker (throughput-
tuned, batches prompts) computes the prompt KV; a DECODE worker (latency-tuned,
continuous batching) takes that KV and generates. The KV cache is TRANSFERRED
from prefill to decode over the interconnect. Now neither interferes with the
other: decode never waits behind a prefill forward.

WHAT THIS FILE IS (and is not)
------------------------------
A faithful SINGLE-PROCESS simulation of that control flow:

  PrefillWorker      -- runs prompt forwards, emits a transferable KV bundle.
  DecodeWorker       -- ingests a KV bundle and runs continuous-batch decode.
  DisaggregatedEngine-- routes requests prefill -> (asyncio Queue) -> decode,
                        and exposes a LlamaModel-style generate().

Honest about the simulation boundary: a real deployment puts the two workers on
different GPUs/nodes with replicated weights and an RDMA KV transfer. Here both
workers share ONE model instance (so output is bit-identical to the unified
engine -- that's the parity test) and the "transfer" is an in-process
asyncio.Queue carrying tensors. The architectural property we DO reproduce: a
decode step's forward never carries prefill work, so decode latency is decoupled
from prompt length. True wall-clock overlap needs real separate hardware.

KV TRANSFER CORRECTNESS
-----------------------
The cache stores K already RoPE-rotated. The prefill worker gathers each layer's
(K, V) and ships them; the decode worker `append`s them into a fresh cache (the
append contract already expects post-RoPE K), so the decode cache is identical
to what a unified prefill would have produced. The decode worker's seq_len then
equals the prompt length, so the first decode token is rotated at the right
absolute position -- generation continues seamlessly.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from src.engine import events
from src.engine.kv_cache import PagedKVCache, PagedRequestCache

if TYPE_CHECKING:
    from src.engine.events import EventBus
    from src.engine.model import LlamaModel


# Sentinel marking "no more prefills will arrive" on the transfer queue.
_PREFILL_DONE = object()


@dataclass
class _Request:
    """A submitted generation request, before it has been prefilled."""
    request_id: str
    prompt_ids: torch.Tensor      # (1, P) int64
    max_new_tokens: int
    eos_token_id: int | None = None


@dataclass
class KVTransfer:
    """The bundle migrated from prefill worker to decode worker.

    `layers_kv[l]` is the (K, V) for layer l, each shape (1, P, NKV, D), already
    RoPE-rotated (K) -- exactly what PagedRequestCache.get() returns and what
    append() expects. This is the thing that crosses the "interconnect".
    """
    request_id: str
    layers_kv: list[tuple[torch.Tensor, torch.Tensor]]
    seq_len: int                  # prompt length == cached entries
    first_token: int              # the token prefill already produced
    max_new_tokens: int
    eos_token_id: int | None
    prompt_ids: torch.Tensor


@dataclass
class _DecodeState:
    """A request mid-decode on the decode worker."""
    cache: PagedRequestCache
    generated: list[int]          # starts with the prefill's first_token
    max_new_tokens: int
    eos_token_id: int | None

    def is_finished(self) -> bool:
        if self.eos_token_id is not None and self.generated[-1] == self.eos_token_id:
            return True
        return len(self.generated) >= self.max_new_tokens


# ---------------------------------------------------------------------------
# Prefill worker
# ---------------------------------------------------------------------------


class PrefillWorker:
    """Throughput-oriented worker: runs prompt forwards and emits KV bundles.

    Owns a model instance. Each prefill uses a fresh pool sized to the prompt
    (the worker holds no long-lived KV -- it hands the KV off and forgets). A
    batch is processed as one throughput unit; see process_prefill_batch.
    """

    def __init__(self, model: "LlamaModel", block_size: int = 16) -> None:
        self.model = model
        self.block_size = block_size
        self.num_layers = model.config.num_hidden_layers
        self._head_dim = model.config.hidden_size // model.config.num_attention_heads
        self._dtype = next(model.parameters()).dtype
        self._device = next(model.parameters()).device
        self.num_prefilled = 0
        self.prompt_tokens_processed = 0

    @torch.no_grad()
    def process_prefill(self, request: _Request) -> KVTransfer:
        """Run prefill for one request and return its transferable KV bundle."""
        prompt_ids = request.prompt_ids.to(self._device)
        P = int(prompt_ids.shape[1])
        n_blocks = (P + self.block_size - 1) // self.block_size

        # Fresh single-request pool -- decode happens on the other worker.
        pool = PagedKVCache(
            num_layers=self.num_layers,
            num_blocks=n_blocks,
            block_size=self.block_size,
            num_kv_heads=self.model.config.num_key_value_heads,
            head_dim=self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        pool.admit_request("prefill", prefill_blocks_needed=n_blocks, total_blocks_needed=n_blocks)
        cache = PagedRequestCache(pool, "prefill", num_layers=self.num_layers)

        logits = self.model(prompt_ids, kv_cache=cache)        # (1, P, V)
        first_token = int(logits[0, -1].argmax())

        # Snapshot each layer's K/V for transfer (clone so the bundle owns its
        # memory once this per-prefill pool is dropped).
        layers_kv = [
            (k.clone(), v.clone())
            for (k, v) in (cache.get(l) for l in range(self.num_layers))
        ]

        self.num_prefilled += 1
        self.prompt_tokens_processed += P
        return KVTransfer(
            request_id=request.request_id,
            layers_kv=layers_kv,
            seq_len=P,
            first_token=first_token,
            max_new_tokens=request.max_new_tokens,
            eos_token_id=request.eos_token_id,
            prompt_ids=prompt_ids,
        )

    def process_prefill_batch(self, requests: list[_Request]) -> list[KVTransfer]:
        """Process a group of prefills as one throughput unit.

        Note: the engine's attention does single-sequence prefill (variable-
        length batched prefill needs padding+masking we don't implement), so
        within a batch the prompts run sequentially. The "batch" is the
        scheduling unit -- a real prefill worker would pack these into one
        padded forward; the throughput accounting is the same.
        """
        return [self.process_prefill(r) for r in requests]


# ---------------------------------------------------------------------------
# Decode worker
# ---------------------------------------------------------------------------


class DecodeWorker:
    """Latency-oriented worker: ingests KV bundles and runs continuous-batch
    decode. Owns a model instance and a shared KV pool across active requests.
    """

    def __init__(
        self,
        model: "LlamaModel",
        num_blocks: int = 256,
        block_size: int = 16,
        event_bus: "EventBus | None" = None,
        token_observer: Callable[[str, int], None] | None = None,
    ) -> None:
        self.model = model
        self.block_size = block_size
        self.num_layers = model.config.num_hidden_layers
        self.event_bus = event_bus
        self.token_observer = token_observer
        self._device = next(model.parameters()).device

        head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.pool = PagedKVCache(
            num_layers=self.num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=next(model.parameters()).dtype,
            device=self._device,
        )
        self.active: dict[str, _DecodeState] = {}
        self.finished: dict[str, list[int]] = {}
        self.decode_tokens = 0     # total tokens emitted by decode steps
        self._step_idx = 0

    def receive_kv(self, transfer: KVTransfer) -> None:
        """Accept a migrated KV bundle and register the request for decode.

        Rebuilds the prompt's KV in this worker's pool (identical to a unified
        prefill's cache), seeded with the prefill's already-produced first
        token so decode continues from there.
        """
        P = transfer.seq_len
        bs = self.block_size
        prefill_blocks = (P + bs - 1) // bs
        total_blocks = (P + transfer.max_new_tokens + bs - 1) // bs
        self.pool.admit_request(
            request_id=transfer.request_id,
            prefill_blocks_needed=prefill_blocks,
            total_blocks_needed=total_blocks,
        )
        cache = PagedRequestCache(self.pool, transfer.request_id, num_layers=self.num_layers)
        # Write the transferred (post-RoPE) K/V; append advances seq_len to P.
        for layer_idx, (k, v) in enumerate(transfer.layers_kv):
            cache.append(layer_idx, k.to(self._device), v.to(self._device))

        self.active[transfer.request_id] = _DecodeState(
            cache=cache,
            generated=[transfer.first_token],
            max_new_tokens=transfer.max_new_tokens,
            eos_token_id=transfer.eos_token_id,
        )
        # The prefill's first token counts as an emitted token for this request.
        if self.token_observer is not None:
            self.token_observer(transfer.request_id, transfer.first_token)

    def has_work(self) -> bool:
        return bool(self.active)

    def prune_finished(self) -> list[tuple[str, list[int]]]:
        """Move finished requests out of `active`, freeing their blocks. Returns
        (request_id, generated_tokens) for each one finished this call."""
        done: list[tuple[str, list[int]]] = []
        for rid in list(self.active.keys()):
            st = self.active[rid]
            if st.is_finished():
                self.pool.free_request(rid)
                self.finished[rid] = st.generated
                done.append((rid, st.generated))
                if self.event_bus is not None:
                    self.event_bus.emit(events.request_finished(
                        request_id=rid,
                        reason="eos" if (st.eos_token_id is not None
                                         and st.generated[-1] == st.eos_token_id)
                               else "max_new_tokens",
                        total_tokens=len(st.generated),
                        total_steps=self._step_idx,
                    ))
                del self.active[rid]
        return done

    @torch.no_grad()
    def step(self) -> None:
        """One PURE-decode forward over every active request (no prefill work).

        Caller must `prune_finished()` first so we never over-generate a request
        that has already hit its cap.
        """
        if not self.active:
            return
        self._step_idx += 1
        rids = list(self.active.keys())
        states = [self.active[r] for r in rids]
        input_ids = torch.tensor(
            [[st.generated[-1]] for st in states],
            dtype=torch.long, device=self._device,
        )
        caches = [st.cache for st in states]
        logits = self.model(input_ids, kv_cache=caches)        # (B, 1, V)

        batch_event: list[tuple[str, int, str]] = []
        for i, st in enumerate(states):
            nxt = int(logits[i, -1].argmax())
            st.generated.append(nxt)
            self.decode_tokens += 1
            batch_event.append((rids[i], nxt, ""))
            if self.token_observer is not None:
                self.token_observer(rids[i], nxt)
        if self.event_bus is not None:
            self.event_bus.emit(events.decode_step(step_idx=self._step_idx, batch=batch_event))


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class DisaggregatedEngine:
    """Routes requests prefill -> transfer queue -> decode, with a LlamaModel-
    style generate(). The transfer queue is an asyncio.Queue standing in for the
    KV migration link.
    """

    def __init__(
        self,
        model: "LlamaModel",
        decode_blocks: int = 256,
        block_size: int = 16,
        event_bus: "EventBus | None" = None,
        prefill_model: "LlamaModel | None" = None,
        token_observer: Callable[[str, int], None] | None = None,
        prefill_observer: Callable[[str, int], None] | None = None,
    ) -> None:
        # Single-process sim: by default both workers share one model instance,
        # which is what makes output bit-identical to the unified engine. Pass a
        # distinct `prefill_model` (e.g. a replica) to mirror real separation.
        self.model = model
        self.event_bus = event_bus
        self.prefill_observer = prefill_observer
        self.prefill_worker = PrefillWorker(prefill_model or model, block_size=block_size)
        self.decode_worker = DecodeWorker(
            model, num_blocks=decode_blocks, block_size=block_size,
            event_bus=event_bus, token_observer=token_observer,
        )
        self.device = next(model.parameters()).device

        # Inspectable coordination state (read by /stats).
        self.prefill_queue: deque[_Request] = deque()   # submitted, not yet prefilled
        self.transfer_queue: asyncio.Queue = asyncio.Queue()  # KV bundles in flight
        self._lock = asyncio.Lock()                     # serialise agenerate calls

    # ---- stats --------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Queue depths for the /stats endpoint."""
        return {
            "prefill_queue_depth": len(self.prefill_queue),
            "decode_queue_depth": len(self.decode_worker.active),
            "transfer_buffer_size": self.transfer_queue.qsize(),
        }

    # ---- transfer -----------------------------------------------------------

    async def transfer_kv(self, request_id: str, kv_transfer: KVTransfer) -> None:
        """Simulate KV migration: hand the bundle to the decode side."""
        await self.transfer_queue.put(kv_transfer)

    # ---- orchestration ------------------------------------------------------

    async def _run_requests(self, requests: list[_Request]) -> dict[str, list[int]]:
        """Drive a set of requests through prefill -> transfer -> decode.

        A `prefiller` coroutine drains the prefill queue (pushing KV bundles +
        a final sentinel); a `decoder` coroutine ingests bundles as they land
        and runs pure-decode steps. They interleave cooperatively, so decode
        starts as soon as the first prompt's KV is ready -- pipelining prefill
        and decode the way the real architecture overlaps them across workers.
        """
        for r in requests:
            self.prefill_queue.append(r)
        results: dict[str, list[int]] = {}

        async def prefiller() -> None:
            while self.prefill_queue:
                r = self.prefill_queue.popleft()
                if self.event_bus is not None:
                    self.event_bus.emit(events.prefill_started(
                        request_id=r.request_id, num_tokens=int(r.prompt_ids.shape[1])))
                transfer = self.prefill_worker.process_prefill(r)
                if self.prefill_observer is not None:
                    self.prefill_observer(r.request_id, transfer.seq_len)
                if self.event_bus is not None:
                    self.event_bus.emit(events.prefill_done(
                        request_id=r.request_id, blocks_allocated=0))
                await self.transfer_kv(r.request_id, transfer)
                await asyncio.sleep(0)
            await self.transfer_queue.put(_PREFILL_DONE)

        async def decoder() -> None:
            prefills_done = False
            while True:
                # Ingest every bundle that has arrived (non-blocking).
                while not self.transfer_queue.empty():
                    item = self.transfer_queue.get_nowait()
                    if item is _PREFILL_DONE:
                        prefills_done = True
                    else:
                        self.decode_worker.receive_kv(item)
                # Retire finished requests (also catches max_new_tokens == 1).
                for rid, toks in self.decode_worker.prune_finished():
                    results[rid] = toks
                if self.decode_worker.has_work():
                    self.decode_worker.step()
                elif prefills_done and self.transfer_queue.empty():
                    break
                await asyncio.sleep(0)

        await asyncio.gather(prefiller(), decoder())
        return results

    async def run_batch(self, requests: list[_Request]) -> dict[str, list[int]]:
        """Public async entry to process a batch of requests (used by the
        benchmark's mixed workload). Returns {request_id: generated_tokens}."""
        async with self._lock:
            return await self._run_requests(requests)

    # ---- LlamaModel-compatible interface ------------------------------------

    async def agenerate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Async generate for one prompt; returns (1, P + n_generated)."""
        input_ids = input_ids.to(self.device)
        rid = "gen"
        async with self._lock:
            results = await self._run_requests(
                [_Request(rid, input_ids, max_new_tokens, eos_token_id)]
            )
        gen = results.get(rid, [])
        gen_t = torch.tensor([gen], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, gen_t], dim=1)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Synchronous generate matching LlamaModel.generate's signature.

        Spins a private event loop to run the prefill->transfer->decode pipeline
        to completion. `use_cache` is accepted for interface compatibility (the
        disaggregated path is always cached).
        """
        return asyncio.run(self.agenerate(input_ids, max_new_tokens, eos_token_id))
