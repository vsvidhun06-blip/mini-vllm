"""
CUDA graph capture + replay for the decode step.

WHY CUDA GRAPHS
---------------
A decode step is tiny per-kernel work (one new token per request) but LOTS of
kernels: 22 layers x (q/k/v/o projections + RoPE + attention + 3 MLP matmuls +
2 RMSNorms) ~= a few hundred kernel launches. Each launch is a separate
CPU->GPU submission. At decode batch sizes the GPU finishes each kernel faster
than Python can queue the next one, so the step becomes *launch-bound*: the GPU
sits idle waiting for the CPU. This is the classic "CPU overhead dominates
small-batch decode" problem.

A CUDA graph records the entire sequence of kernel launches ONCE (capture) and
then re-submits the whole thing with a single `cudaGraphLaunch` (replay). The
per-launch Python/driver overhead collapses from "hundreds of submissions" to
"one". For launch-bound decode that is a real latency win (benchmark in
scripts/benchmark_cuda_graphs.py).

THE HARD PART: CUDA GRAPHS FREEZE EVERYTHING
--------------------------------------------
Capture records kernels bound to *specific memory addresses*, and bakes in
every host-side value that controlled control flow at capture time. Replay
re-runs the exact same kernels on the exact same addresses. Two consequences
shape this whole module:

  1. STATIC INPUT BUFFERS. The new token ids can't be a fresh tensor each step
     (different address). We pre-allocate ONE `static_input` tensor and
     `copy_` the new tokens into it before every replay. The graph always reads
     that same address.

  2. NO HOST->DEVICE COPIES INSIDE THE CAPTURED REGION. `torch.tensor(py_list,
     device="cuda")` copies from pageable host memory, which forces a sync and
     is illegal mid-capture. The paged KV read path used to do exactly that
     (block table + per-row positions). We made PagedRequestCache cache those
     as persistent device tensors (see kv_cache.py: `_bt_tensor`,
     `seq_len_tensor`) so the captured decode forward has none.

  3. FROZEN SEQUENCE LENGTH. A graph captured with the KV cache at length L only
     ever attends over L+1 keys -- the slice bounds are baked in. Real decode
     grows L every token, so ONE graph cannot serve a growing sequence. We key
     graphs by (batch_size, seq_len). The microbenchmark fixes seq_len, which is
     the standard "decode-step latency" measurement. Serving a genuinely
     growing sequence from a single graph is the vLLM generalization: static
     MAX-shape block tables + an on-device `context_lens` tensor that a custom
     paged-attention kernel reads at runtime. That is a much larger change to
     attention.py and is the documented next step, not implemented here.

POOL OF GRAPHS
--------------
Decode batch size varies as requests join/leave. vLLM captures a graph for each
of a handful of common batch sizes and pads up to the nearest. We capture for
batch sizes [1, 2, 4, 8]; a batch size not in the pool (or any (batch_size,
seq_len) we never captured) simply falls back to eager execution -- always
correct, just without the launch-overhead win.

BINDING
-------
A captured graph is bound to the *exact* PagedRequestCache objects it saw at
capture (their pool tensor addresses are baked in). `replay` therefore only
works against those same cache objects -- `can_replay` enforces this by
identity. The scheduler routes to a graph only when the live decode batch
matches a captured (batch_size, caches, seq_len); otherwise it runs eager. In
practice live autoregressive decode keeps growing seq_len, so the win is
demonstrated in the fixed-length benchmark; the scheduler hook is the correct
integration point for the static-buffer generalization above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from src.engine.kv_cache import PagedKVCache, PagedRequestCache

if TYPE_CHECKING:
    from src.engine.model import LlamaModel


@dataclass
class _CapturedGraph:
    """Everything needed to replay one captured decode step.

    The graph is bound to `caches` (their `pool` tensor addresses are baked into
    the recorded kernels) and to the `static_input` / `static_logits` buffers.
    """
    graph: "torch.cuda.CUDAGraph"
    static_input: torch.Tensor      # (B, 1) int64 -- copied into before replay
    static_logits: torch.Tensor     # (B, 1, vocab) -- the captured output buffer
    caches: list[PagedRequestCache]  # bound per-request caches (B of them)
    pool: PagedKVCache               # the dedicated KV pool backing `caches`
    seq_len: int                    # KV context length this graph decodes at


class CUDAGraphRunner:
    """Captures and replays the batched-decode forward for a pool of batch sizes.

    Typical use (benchmark / test):
        runner = CUDAGraphRunner(model)
        runner.capture(model, batch_size=2, seq_len=256)
        caches = runner.caches_for(2)
        logits = runner.replay(input_ids, caches)   # input_ids: (2, 1) int64

    The scheduler holds a runner and routes decode steps through `replay` when
    `can_replay` says a matching graph exists, else runs the model eagerly.
    """

    # The batch sizes we are willing to capture graphs for (vLLM's common set).
    POOL_SIZES: tuple[int, ...] = (1, 2, 4, 8)
    # Warmup forwards before capture. The FIRST CUDA forward also JIT-compiles
    # (and autotunes) the Triton RoPE/FA2 kernels; that MUST happen here, not
    # mid-capture (compilation/autotuning launches its own timed work + syncs).
    WARMUP_ITERS: int = 3
    # block_size for the dedicated capture pools. Matches the engine default.
    BLOCK_SIZE: int = 16

    def __init__(self, model: "LlamaModel") -> None:
        self.model = model
        self.graphs: dict[int, _CapturedGraph] = {}

    # ---- capture ------------------------------------------------------------

    def capture(self, model: "LlamaModel", batch_size: int, seq_len: int) -> _CapturedGraph:
        """Warm up, then capture a decode step for (batch_size, seq_len).

        Builds a dedicated KV pool seeded so each of the `batch_size` requests
        already has `seq_len` cached tokens; the captured step decodes the
        (seq_len+1)-th token for each. Returns the _CapturedGraph (also stored
        in self.graphs[batch_size]).
        """
        if batch_size not in self.POOL_SIZES:
            raise ValueError(
                f"batch_size {batch_size} not in graph pool {self.POOL_SIZES}"
            )
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("CUDA graphs require a CUDA model/device")

        pool, caches = self._build_caches(model, batch_size, seq_len)

        # Static input buffer: the only thing we copy fresh data into per replay.
        static_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        # Always drive the BATCHED-decode path (a list of caches), even for
        # batch_size == 1 -- that's the path the scheduler uses for every
        # decode step, so the graph matches production exactly.
        kv_arg = caches

        # ---- Warmup on a side stream (required by the capture protocol) ----
        # Each warmup iteration re-seeds the caches to `seq_len` so every pass
        # is the IDENTICAL decode step. This stabilises the cached block-table
        # and seq-len device tensors (kv_cache.py) so that during capture the
        # forward issues NO host->device copies.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.WARMUP_ITERS):
                self._seed(caches, seq_len)
                with torch.no_grad():
                    model(static_input, kv_cache=kv_arg)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # ---- Capture ----
        self._seed(caches, seq_len)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_logits = model(static_input, kv_cache=kv_arg)

        cg = _CapturedGraph(
            graph=graph,
            static_input=static_input,
            static_logits=static_logits,
            caches=caches,
            pool=pool,
            seq_len=seq_len,
        )
        self.graphs[batch_size] = cg
        return cg

    # ---- replay -------------------------------------------------------------

    def can_replay(
        self,
        batch_size: int,
        caches: list[PagedRequestCache] | None = None,
    ) -> bool:
        """True iff a captured graph exists for `batch_size` and (when given)
        the provided caches are EXACTLY the ones it was captured against.

        The identity check matters: the graph baked in the pool tensor
        addresses of its bound caches. Replaying against different cache objects
        would read the wrong memory.
        """
        cg = self.graphs.get(batch_size)
        if cg is None:
            return False
        if caches is None:
            return True
        if len(caches) != len(cg.caches):
            return False
        return all(a is b for a, b in zip(caches, cg.caches))

    def caches_for(self, batch_size: int) -> list[PagedRequestCache]:
        """The bound caches for a captured batch size (for eager-vs-graph
        comparisons and benchmarking against the same KV state)."""
        return self.graphs[batch_size].caches

    def reset(self, batch_size: int) -> None:
        """Re-seed a captured graph's bound caches to its capture-time seq_len.

        Eager forwards (and the graph's own capture pass) advance the host-side
        seq_len by one; call this between measured steps so each step decodes
        the same fixed-context token.
        """
        cg = self.graphs[batch_size]
        self._seed(cg.caches, cg.seq_len)

    def replay(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[PagedRequestCache] | None = None,
    ) -> torch.Tensor:
        """Replay the captured decode step with new token ids.

        Copies `input_ids` into the static input buffer, replays the graph, and
        returns a fresh clone of the captured logits (B, 1, vocab). `kv_caches`,
        when provided, must be the caches this graph was captured against
        (validated by identity) -- the graph reads their pool at baked addresses.
        """
        bs = int(input_ids.shape[0])
        cg = self.graphs.get(bs)
        if cg is None:
            raise KeyError(
                f"no captured graph for batch_size={bs}; call capture() first "
                f"or fall back to eager execution"
            )
        if kv_caches is not None and not self.can_replay(bs, kv_caches):
            raise ValueError(
                "replay kv_caches are not the caches this graph was captured "
                "against; a CUDA graph bakes in its KV pool addresses at capture "
                "time and cannot be repointed to different cache objects"
            )
        cg.static_input.copy_(input_ids)
        # Keep host-side seq-len metadata aligned with the frozen graph. The
        # graph ignores host seq_len on replay (the slice bounds are baked), but
        # re-seeding keeps the caches in a sane, comparable state for any eager
        # forward the caller runs next.
        self._seed(cg.caches, cg.seq_len)
        cg.graph.replay()
        return cg.static_logits.clone()

    # ---- internals ----------------------------------------------------------

    def _build_caches(
        self,
        model: "LlamaModel",
        batch_size: int,
        seq_len: int,
    ) -> tuple[PagedKVCache, list[PagedRequestCache]]:
        """Build a dedicated KV pool + `batch_size` request caches, each seeded
        with `seq_len` cached tokens.

        We size each request for ceil((seq_len+1)/block_size) blocks and
        allocate them all at admit time. That guarantees the captured decode
        step (which appends ONE token at position seq_len) writes into an
        ALREADY-ALLOCATED block -- so it never triggers `allocate_block`, which
        would grow the block table and force a host->device rebuild of the
        cached block-table tensor mid-capture.
        """
        cfg = model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        bs = self.BLOCK_SIZE

        blocks_per_req = (seq_len + 1 + bs - 1) // bs
        pool = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            # One pool's worth for every request, plus a small slack margin.
            num_blocks=blocks_per_req * batch_size + batch_size,
            block_size=bs,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        caches: list[PagedRequestCache] = []
        for i in range(batch_size):
            rid = f"graph_b{batch_size}_r{i}"
            pool.admit_request(
                request_id=rid,
                prefill_blocks_needed=blocks_per_req,
                total_blocks_needed=blocks_per_req,
            )
            caches.append(
                PagedRequestCache(pool, rid, num_layers=cfg.num_hidden_layers)
            )
        self._seed(caches, seq_len)
        return pool, caches

    @staticmethod
    def _seed(caches: list[PagedRequestCache], seq_len: int) -> None:
        """Set every layer's seq_len to `seq_len` for every cache.

        Writing `_seq_lens` directly is the same private access the scheduler
        uses when seeding a prefix-cache hit boundary; it touches only host-side
        bookkeeping (no GPU work), so it is safe to call between graph replays.
        """
        for c in caches:
            for layer_idx in range(len(c._seq_lens)):  # noqa: SLF001
                c._seq_lens[layer_idx] = seq_len
