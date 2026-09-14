"""
B1 investigation: CUDA-graph decode replay x per-request LoRA routing.

THE QUESTION
------------
`ContinuousBatchScheduler._decode_forward` can serve a decode step either
eagerly or by replaying a captured CUDA graph. Adapter routing, meanwhile, is
carried as PYTHON state: `LoRALinear._active` is read by a Python `if` inside
`LoRALinear.forward`, which selects between the base path, `_apply_single` and
`_apply_per_row`. A CUDA graph records the kernels launched during capture and
replays that recording; it does not re-execute Python. Those two facts are in
tension, and this module establishes experimentally whether the tension is real
and what it costs.

HOW THE EVIDENCE IS SPLIT (following test_live_graph.py's convention)
---------------------------------------------------------------------
  CPU (runs everywhere)
    * That LoRA routing is expressed as a DIFFERENT SEQUENCE OF TENSOR
      OPERATIONS, not as data flowing through a fixed sequence. This is the
      mechanism, and it is fully observable without a GPU.
    * That the static-shape decode path is exact for a base workload, and
      exact for a LoRA workload when replay re-reads the live routing. These
      are the controls that isolate routing as the cause of the divergence
      below rather than the staging path.
    * That a replay which executes the CAPTURE-TIME routing branch produces
      wrong tokens for every request in a mixed-adapter workload.

  CUDA (skipped on CPU-only hosts -- must be run on the GPU box)
    * Whether a real `torch.cuda.CUDAGraph` capture under an active per-row
      plan even succeeds, and whether a real replay honours live routing.

NOTHING HERE FABRICATES A CUDA RESULT. `_FrozenRoutingRunner` is a MODEL of one
specific, documented CUDA-graph property -- that the Python branch resolved at
capture is the branch whose kernels replay executes -- and is labelled as such
everywhere it appears. It is not evidence about CUDA execution; the CUDA-gated
tests are the only ones that can be.
"""
from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.overrides import TorchFunctionMode

from src.engine.kv_cache import PagedKVCache, PagedRequestCache
from src.engine.live_graph import (
    DecodeGraphRouter,
    LiveDecodeGraphRunner,
    StaticDecodeBatch,
)
from src.engine.lora import (
    LoRALinear,
    LoRAManager,
    normalize_routing_plan,
)
from src.engine.lora_model import (
    LoRALlamaModel,
    model_routing_plan,
    random_adapter_weights,
)
from src.engine.model import LlamaConfig, LlamaModel
from src.engine.scheduler import ContinuousBatchScheduler, RequestStatus

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real CUDA graph capture/replay can only be verified on a GPU host; "
           "asserting it from CPU would be fabricating the result",
)

BLOCK_SIZE = 16
RANK = 8
ALPHA = 16.0
# See test_lora_scheduler_integration.py: random_adapter_weights' default 0.02
# is too gentle to flip a greedy argmax on a 256-token vocab, which would make
# every assertion here vacuous. 0.1 separates every adapter from base and from
# every other adapter without saturating. The negative-control test enforces it.
SCALE_INIT = 0.1

_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _tiny_model() -> LlamaModel:
    """head_dim = 128/8 = 16, a power of two, so the CUDA FA2 decode kernel
    accepts this model too -- the CUDA-gated tests reuse it unchanged."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=4096, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


def _lora_model(device="cpu") -> LoRALlamaModel:
    """Wrapped tiny model with adapters "a" and "b" resident."""
    model = LoRALlamaModel(_tiny_model().to(device), LoRAManager())
    for adapter_id, seed in (("a", 1), ("b", 2)):
        model.manager.load_adapter(
            adapter_id, rank=RANK, alpha=ALPHA,
            weights_dict=random_adapter_weights(
                model, rank=RANK, seed=seed, scale_init=SCALE_INIT),
        )
    return model


def _projections(model: LoRALlamaModel):
    for block in model.layers:
        for proj in _PROJECTIONS:
            yield getattr(block.attn, proj)


# ---------------------------------------------------------------------------
# CPU 1. The mechanism: routing selects an op SEQUENCE, not a data value.
# ---------------------------------------------------------------------------


class _OpRecorder(TorchFunctionMode):
    """Records every torch-level operation executed inside the context."""

    def __init__(self):
        self.ops: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.ops.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


def _decode_fixture(model: LoRALlamaModel, batch: int = 2, prompt_len: int = 8):
    """A pool + prefilled caches + a decode-shaped input, ready to forward."""
    cfg = model.config
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_blocks=64, block_size=BLOCK_SIZE,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=torch.float32, device="cpu", enable_prefix_cache=False,
    )
    caches = []
    g = torch.Generator().manual_seed(5)
    for i in range(batch):
        rid = f"fx{i}"
        pool.admit_request(request_id=rid, prefill_blocks_needed=4,
                           total_blocks_needed=4)
        cache = PagedRequestCache(pool, rid, num_layers=cfg.num_hidden_layers)
        with torch.no_grad():
            model.model(torch.randint(0, 256, (1, prompt_len), generator=g),
                        kv_cache=cache)
        caches.append(cache)
    return pool, caches, torch.randint(0, 256, (batch, 1), generator=g)


def _record_ops(model, caches, input_ids, plan) -> list[str]:
    """Ops executed by one decode forward under `plan`, cache state restored."""
    model.set_batch_adapters(plan) if isinstance(plan, list) else model.set_adapter(plan)
    saved = [list(c._seq_lens) for c in caches]            # noqa: SLF001
    recorder = _OpRecorder()
    with torch.no_grad(), recorder:
        model(input_ids, kv_cache=caches)
    for cache, seq_lens in zip(caches, saved):
        cache._seq_lens[:] = seq_lens                      # noqa: SLF001
        cache._bt_tensor = None                            # noqa: SLF001
        cache._seqlen_tensors.clear()                      # noqa: SLF001
        cache._seqlen_vals.clear()                         # noqa: SLF001
    return recorder.ops


def test_routing_selects_a_different_operation_sequence():
    """LoRA routing is CONTROL FLOW, not data -- the core of the B1 hazard.

    A mechanism that records operations once and replays the recording (which
    is exactly what a CUDA graph is) can only ever execute the sequence chosen
    at record time. This test shows the sequence genuinely differs, so such a
    mechanism cannot follow a routing change. It makes no claim about CUDA.
    """
    model = _lora_model()
    _pool, caches, input_ids = _decode_fixture(model)

    base_ops = _record_ops(model, caches, input_ids, [None, None])
    routed_ops = _record_ops(model, caches, input_ids, ["a", "b"])

    # The delta ops that only a routed forward performs.
    extra = Counter(routed_ops) - Counter(base_ops)
    assert extra["index_add"] > 0 and extra["index_select"] > 0, (
        f"expected per-row grouping ops to appear only under routing; "
        f"diff was {dict(extra)}"
    )
    assert extra["matmul"] > 0, "expected the LoRA delta GEMMs to be extra ops"

    # And the base-routed sequence contains NONE of them: a recording taken
    # while nothing is routed has no adapter kernels to replay, at all.
    assert not any("index_add" in op for op in base_ops), (
        "a base-routed forward unexpectedly contains grouping ops"
    )
    assert len(routed_ops) > len(base_ops), (
        f"routed forward should execute strictly more ops "
        f"({len(routed_ops)} vs {len(base_ops)})"
    )


def test_per_row_routing_builds_no_host_tensors_inside_forward():
    """A prepared per-row plan constructs NOTHING from host data in forward.

    THIS IS THE REGRESSION TEST FOR THE ROOT CAUSE. `_apply_per_row` used to
    build its row-index tensor with `torch.tensor(rows, device=...)` once per
    wrapped projection per step. On CUDA that is a pageable host->device copy,
    and a T4 rejected every routed capture with

        RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
        graph capture unless the CPU tensor is pinned.

    for plans [a,b], [a], [a,b,a], [a,b,b] -- so no routed graph existed and
    the routed arm silently ran eager. `set_active` now prepares those indices
    on the projection's device, ahead of capture, and forward only reads them.

    `_record_ops` installs the plan BEFORE entering the recorder, exactly as
    the scheduler installs it before the decode forward, so what this counts is
    the forward alone. attention.py:540-545 removed the same pattern from the
    decode path for the same reason; this keeps it removed from the LoRA path.
    """
    model = _lora_model()

    # ---- the projection in isolation: the exact site of the H2D copy -------
    # No KV cache in the picture at all, so a single `torch.tensor` here is
    # unambiguously _apply_per_row's.
    proj = next(_projections(model))
    x = torch.randn(2, 128, generator=torch.Generator().manual_seed(9))
    model.set_batch_adapters(["a", "b"])
    recorder = _OpRecorder()
    with torch.no_grad(), recorder:
        proj(x)
    assert recorder.ops.count("tensor") == 0, (
        f"_apply_per_row constructed a tensor from host data; on CUDA that is "
        f"the pageable H2D copy graph capture rejects. Routing indices must be "
        f"prepared in set_active(), not built in forward. ops={recorder.ops}"
    )
    # The delta arithmetic is still there -- this is a prepared-metadata fix,
    # not a disabled-routing fix.
    assert recorder.ops.count("index_select") == 2   # one gather per adapter
    assert recorder.ops.count("index_add") == 2      # one scatter per adapter

    # ---- and the whole decode forward: routing adds no host tensors --------
    # CALIBRATION MATTERS HERE. `_record_ops` restores the caches by dropping
    # their cached block tables, so the first recording after prefill reuses
    # them and every later one rebuilds -- two `torch.tensor(block_table)`
    # calls that belong to the KV cache, not to LoRA. The previous version of
    # this test compared an un-warmed base recording (0) against a routed one
    # (2) and credited the difference to `_apply_per_row`; it would have passed
    # just the same with the routing indices already fixed. Warm up first so
    # both arms are measured from the same cache state, then the comparison is
    # about routing and nothing else.
    model.clear_adapters()          # the fixture prefills one row at a time
    _pool, caches, input_ids = _decode_fixture(model)
    _record_ops(model, caches, input_ids, [None, None])          # warm-up, discarded
    base_ops = _record_ops(model, caches, input_ids, [None, None])
    routed_ops = _record_ops(model, caches, input_ids, ["a", "b"])

    assert routed_ops.count("tensor") == base_ops.count("tensor"), (
        f"a routed decode forward built "
        f"{routed_ops.count('tensor') - base_ops.count('tensor')} more host "
        f"tensors than the same batch running on base; on CUDA each one is a "
        f"pageable H2D copy that stream capture rejects"
    )
    assert routed_ops.count("index_select") > 0 and routed_ops.count("index_add") > 0


def test_set_active_prepares_row_indices_on_the_weight_device():
    """The prepared groups are the correct rows, on the projections' device."""
    model = _lora_model()
    model.set_batch_adapters(["a", None, "b", "a"])

    for proj in _projections(model):
        assert proj._routing_plan == ("a", None, "b", "a")     # noqa: SLF001
        groups = dict(proj._routing_groups)                    # noqa: SLF001
        assert set(groups) == {"a", "b"}, (
            "base (None) rows must not get a group; they are base-only"
        )
        assert groups["a"].tolist() == [0, 3]
        assert groups["b"].tolist() == [2]
        for idx in groups.values():
            assert idx.dtype == torch.long
            assert idx.device == proj.base.weight.device, (
                "row indices must live where the wrapped projection's weights "
                "do, or forward would need an H2D copy to use them"
            )

    # Whole-batch routing needs no indices at all: no allocation on the
    # single-adapter path and none on the zero-overhead base path.
    model.set_adapter("a")
    for proj in _projections(model):
        assert proj._routing_plan is None and proj._routing_groups == []   # noqa: SLF001
    model.clear_adapters()
    for proj in _projections(model):
        assert proj._routing_plan is None and proj._routing_groups == []   # noqa: SLF001


def test_changing_the_plan_refreshes_the_prepared_indices():
    """A new plan replaces the live groups; a repeat plan REUSES the tensors.

    Reuse is a correctness requirement, not an optimisation. A graph captured
    under plan P baked in the ADDRESS of P's index tensors, so re-installing P
    -- which the scheduler does on every step of a steady mixed batch -- must
    hand the same storage back, or the replay would gather from freed memory.
    """
    model = _lora_model()
    proj = next(_projections(model))

    model.set_batch_adapters(["a", "b"])
    first_ptrs = {k: v.data_ptr() for k, v in proj._routing_groups}   # noqa: SLF001

    model.set_batch_adapters(["b", "a", "a"])
    changed = dict(proj._routing_groups)                       # noqa: SLF001
    assert proj._routing_plan == ("b", "a", "a")               # noqa: SLF001
    assert changed["a"].tolist() == [1, 2] and changed["b"].tolist() == [0]

    # Re-install the ORIGINAL plan, from a fresh list object.
    model.set_batch_adapters(["a", "b"])
    assert proj._routing_plan == ("a", "b")                    # noqa: SLF001
    again_ptrs = {k: v.data_ptr() for k, v in proj._routing_groups}  # noqa: SLF001
    assert again_ptrs == first_ptrs, (
        "re-installing a plan reallocated its row-index tensors; a CUDA graph "
        "captured under that plan would replay against the old addresses"
    )


def test_unprepared_plan_still_computes_the_right_answer_eagerly():
    """A plan that reaches forward without set_active is still correct.

    The prepared path is what capture requires; it must not become what
    CORRECTNESS requires. A layer driven directly (a mutated list, a hand-set
    `_active`) has to fall back to preparing on the spot and produce exactly
    the same output as the routed model.
    """
    model = _lora_model()
    x = torch.randn(3, 128, generator=torch.Generator().manual_seed(11))
    plan = ["a", None, "b"]

    projections = list(_projections(model))
    model.set_batch_adapters(plan)
    prepared = projections[0](x)

    other = projections[0]
    other._routing_plan = None                                 # noqa: SLF001
    other._routing_groups = []                                 # noqa: SLF001
    other._routing_cache.clear()                               # noqa: SLF001
    other._active = list(plan)                                 # noqa: SLF001
    assert torch.equal(other(x), prepared)
    assert other._routing_plan == ("a", None, "b")             # noqa: SLF001


def test_per_row_output_row_ordering_is_unchanged():
    """Per-row routing composes as base + that row's adapter delta, in order.

    Row i of the routed output must equal row i of a single-adapter forward
    under active[i] (or the base forward where active[i] is None). This is the
    property the grouping/scatter is an optimisation OF, and it is what pins
    the output row order across the prepared-index rewrite.
    """
    model = _lora_model()
    proj = next(_projections(model))
    x = torch.randn(4, 128, generator=torch.Generator().manual_seed(12))

    model.clear_adapters()
    base = proj(x)
    per_adapter = {}
    for adapter_id in ("a", "b"):
        model.set_adapter(adapter_id)
        per_adapter[adapter_id] = proj(x)

    plan = ["b", None, "a", "b"]
    model.set_batch_adapters(plan)
    routed = proj(x)

    for i, adapter_id in enumerate(plan):
        expected = base[i] if adapter_id is None else per_adapter[adapter_id][i]
        assert torch.allclose(routed[i], expected, atol=1e-6), (
            f"row {i} of the mixed-adapter output is not row {i} under "
            f"adapter {adapter_id!r}"
        )


def test_plan_length_mismatch_still_raises_before_any_preparation():
    """The batch-size guard is checked first and still names both lengths."""
    model = _lora_model()
    model.set_batch_adapters(["a", "b"])
    proj = next(_projections(model))
    with pytest.raises(ValueError, match=r"length 2 != batch size 3"):
        proj(torch.randn(3, 128))


# ---------------------------------------------------------------------------
# CPU 2. Controls + the reproducer.
#
# `_StaticReplayRunner` runs the SAME StaticDecodeBatch staging a captured
# graph replays, through the real DecodeGraphRouter, exactly as the repo's
# existing `_CpuStaticRunner` (test_live_graph.py) does. The single added knob
# is `freeze_routing`:
#
#   freeze_routing=False -- replay re-reads live `_active`. This is what the
#                           existing CPU stand-in does, and it is why that
#                           stand-in cannot observe B1.
#   freeze_routing=True  -- replay executes the branch each LoRALinear took at
#                           CAPTURE time. This models the one CUDA-graph
#                           property under investigation and nothing else.
#
# Holding everything else identical between the two makes routing the only
# variable, which is what isolates it as the cause.
# ---------------------------------------------------------------------------


_UNSET = object()   # "caller passed no routing plan", distinct from plan None


class _StaticReplayRunner(DecodeGraphRouter):
    """CPU stand-in for a captured decode graph. NOT a CUDA result.

    `guard_routing=False` reproduces the PRE-FIX runner: `_resolve` ignores the
    routing plan, exactly as it did before the B1 guard existed. It is how the
    hazard characterisation below can still demonstrate what a stale replay
    does now that production refuses to perform one.
    """

    def __init__(self, model, pool, *, freeze_routing: bool,
                 guard_routing: bool = True, **kwargs):
        super().__init__(pool, **kwargs)
        self.model = model
        self.freeze_routing = freeze_routing
        self.guard_routing = guard_routing
        self.replay_calls = 0
        # Every routing plan replay() was actually CALLED with, so a test can
        # assert the scheduler threaded the live plan through (B3) rather
        # than replay() having recomputed it.
        self.replay_plans: list = []
        self._captured_plans: dict[int, object] = {}
        self.graph_plans: dict[tuple[int, int], object] = {}
        for batch_size in self.batch_sizes:
            for n_blocks in self.buckets:
                self.graphs[(batch_size, n_blocks)] = (
                    StaticDecodeBatch(pool, batch_size, n_blocks),)

    def capture(self) -> None:
        """Freeze the routing branch every projection would take right now, and
        register the plan in `graph_plans` the way `LiveDecodeGraphRunner.
        _capture` does.

        Mirrors `capture_all()`, which the scheduler docstring instructs callers
        to run BEFORE admitting any request -- i.e. while nothing is routed.
        Setting a plan on the model before calling this captures under that plan
        instead, which is how the "a routed graph cannot serve a different plan"
        test is built.
        """
        for layer in _projections(self.model):
            if isinstance(layer, LoRALinear):                   # plain model: skip
                self._captured_plans[id(layer)] = layer._active  # noqa: SLF001
        plan = model_routing_plan(self.model)
        for key in self.graphs:
            self.graph_plans[key] = plan

    def _resolve(self, batch_size, caches, routing_plan=None):
        if not self.guard_routing:
            # Pre-fix behaviour: whatever the recording holds is accepted.
            routing_plan = next(iter(self.graph_plans.values()), None)
        return super()._resolve(batch_size, caches, routing_plan)

    def replay(self, input_ids, caches, routing_plan=_UNSET):
        # Resolve with the plan the CALLER passed, as
        # LiveDecodeGraphRunner.replay now does (B3). The scheduler always
        # supplies it; `_UNSET` marks a direct call from a test that did not,
        # and only then do we fall back to reading it off the model.
        if routing_plan is _UNSET:
            routing_plan = model_routing_plan(self.model)
        self.replay_plans.append(routing_plan)
        key = self._resolve(len(caches), caches, routing_plan)
        assert key is not None, "replay() called without a routable key"
        self.replay_calls += 1
        (state,) = self.graphs[key]
        state.stage(input_ids, caches)
        if self.freeze_routing:
            out = self._run_with_captured_routing(state)
        else:
            out = state.run_eager(self.model)
        state.commit(caches)
        return out

    def _run_with_captured_routing(self, state):
        original = LoRALinear.forward
        captured = self._captured_plans

        def frozen_forward(layer, x, adapter_id=None):
            live = layer._active                                # noqa: SLF001
            layer._active = captured.get(id(layer))             # noqa: SLF001
            try:
                return original(layer, x, adapter_id)
            finally:
                layer._active = live                            # noqa: SLF001

        LoRALinear.forward = frozen_forward
        try:
            return state.run_eager(self.model)
        finally:
            LoRALinear.forward = original


# (request_id, adapter_id, max_new_tokens, admit_on_step).
#
# Tuned to produce BOTH kinds of composition change a frozen recording can fail
# on, because they are detectable by different failure modes:
#
#   * batch size changes (2 -> 1 -> 3): B retires early, C/D join mid-flight.
#   * batch size CONSTANT while the adapter at a row index changes:
#     D ("a") retires exactly as E ("b") joins, taking the batch from
#     ('a','b','a') to ('a','b','b') at size 3.
#
# The second is the only situation in which a plan that lags the request order
# by one step is observable -- without it, mutation M3 is undetectable because
# consecutive same-size steps always carry the identical plan. The reproducer
# asserts both properties hold so a future re-timing cannot silently drop them.
_WORKLOAD = [
    ("A", "a", 10, 0),
    ("B", "b",  3, 0),
    ("C", "b",  6, 4),
    ("D", "a",  3, 4),
    ("E", "b",  4, 6),
]


def _serve(model, *, adapters, runner_mode, ledger=None, guard_routing=True,
           capture_plan=None, out=None):
    """Drive the real scheduler over _WORKLOAD; return {request_id: tokens}.

    `runner_mode` is None (no runner -> eager), False (static replay that
    re-reads routing) or True (static replay frozen at capture time).
    `adapters` maps the workload's adapter column, so the same schedule can be
    served as an all-base control.
    """
    model.clear_adapters()
    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = None
    if runner_mode is not None:
        runner = _StaticReplayRunner(
            model, sched.pool, freeze_routing=runner_mode,
            guard_routing=guard_routing,
            max_batch_size=4, max_context_tokens=64,
        )
        if capture_plan is not None:
            model.set_batch_adapters(capture_plan)   # capture UNDER this plan
        runner.capture()          # before any request, per the documented usage
        model.clear_adapters()
        sched._graph_runner = runner
    if out is not None:
        out["runner"] = runner
        out["scheduler"] = sched

    original_decode = ContinuousBatchScheduler._decode_forward

    def spy(self, input_ids, caches):
        decode_reqs = [r for r in self.active if r.status is RequestStatus.DECODE]
        active = model._lora_layers[0]._active                  # noqa: SLF001
        replayed = (
            runner is not None and self.use_cuda_graphs
            and runner.can_replay(len(caches), caches)
        )
        if ledger is not None:
            ledger.append({
                "step": self._step_idx,
                "order": [r.request_id for r in decode_reqs],
                "adapters": [r.adapter_id for r in decode_reqs],
                "plan": list(active) if isinstance(active, list) else active,
                "batch_size": int(input_ids.shape[0]),
                "caches_aligned": [r.cache for r in decode_reqs] == caches,
                "path": "replay" if replayed else "eager",
            })
        return original_decode(self, input_ids, caches)

    tokens: dict[str, list[int]] = {}
    try:
        ContinuousBatchScheduler._decode_forward = spy
        g = torch.Generator().manual_seed(7)
        prompts = {rid: torch.randint(0, 256, (1, 6), generator=g)
                   for rid, _, _, _ in _WORKLOAD}
        for step in range(60):
            for rid, _adapter, max_new, admit_step in _WORKLOAD:
                if step == admit_step:
                    sched.add_request(
                        rid, prompts[rid], max_new_tokens=max_new,
                        eos_token_id=None, adapter_id=adapters[rid],
                    )
            last_admit = max(w[3] for w in _WORKLOAD)
            if not sched.has_work() and step > last_admit:
                break
            for rid, token_id in sched.step():
                tokens.setdefault(rid, []).append(token_id)
    finally:
        ContinuousBatchScheduler._decode_forward = original_decode
    return tokens


_LORA_ADAPTERS = {rid: adapter for rid, adapter, _, _ in _WORKLOAD}
_BASE_ADAPTERS = {rid: None for rid, _, _, _ in _WORKLOAD}


def test_negative_control_adapters_are_distinguishable():
    """base != a, base != b, a != b on this workload -- else all else is vacuous."""
    model = _lora_model()
    prompt = torch.randint(0, 256, (1, 6), generator=torch.Generator().manual_seed(7))
    outs = {}
    for adapter_id in (None, "a", "b"):
        out = model.generate(prompt, max_new_tokens=8, eos_token_id=None,
                             use_cache=True, adapter_id=adapter_id)
        outs[adapter_id] = tuple(out[0, prompt.shape[1]:].tolist())
    assert len(set(outs.values())) == 3, (
        f"adapters are not distinguishable on this input, so every routing "
        f"assertion in this module would pass vacuously: {outs}"
    )
    for adapter_id, seq in outs.items():
        assert len(set(seq)) > 1, (
            f"adapter {adapter_id!r} saturated to a single repeated token {seq}; "
            f"SCALE_INIT is too large"
        )


def test_control_static_replay_is_exact_for_a_base_workload():
    """CONTROL 1: with no adapters anywhere, frozen replay == eager exactly.

    Establishes that the StaticDecodeBatch staging path introduces no error of
    its own, so any divergence in the LoRA workload cannot be blamed on it.
    """
    model = _lora_model()
    eager = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=None)
    frozen = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=True)
    assert eager == frozen, (
        "the static-shape decode path is not exact even without LoRA; the B1 "
        "experiment below would be uninterpretable"
    )


def test_control_replay_that_rereads_routing_matches_eager():
    """CONTROL 2: same LoRA workload, same staging, routing re-read at replay.

    This is what the repo's existing `_CpuStaticRunner` does -- and it passes,
    which is precisely why that stand-in cannot detect B1. Together with the
    experiment below it isolates capture-time routing as the sole cause.
    """
    model = _lora_model()
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)
    # guard_routing=False so the batch actually reaches replay: with the B1
    # guard on, a routed batch is refused and this control would prove nothing.
    rereading = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=False,
                       guard_routing=False)
    assert eager == rereading, (
        "static replay diverged from eager even while re-reading live routing; "
        "the cause would then not be capture-time freezing"
    )


def test_replay_frozen_at_capture_time_routing_corrupts_every_request():
    """THE REPRODUCER: freezing the routing branch at capture corrupts output.

    Not a claim about CUDA execution -- a claim about the property CUDA graphs
    have. It stays true and worth pinning after any fix: it is the statement of
    WHY replay must not freeze routing.

    The scheduler's own plan is correct on every step (asserted below); the
    corruption is entirely downstream of it, which is why Phase 1's integration
    tests pass while the served tokens are wrong.
    """
    model = _lora_model()
    ledger: list[dict] = []
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)
    # guard_routing=False reproduces the PRE-FIX runner, in which _resolve
    # ignored the routing plan. Production now refuses this replay; the point
    # of this test is to keep on record what it would do if it did not.
    frozen = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True,
                    ledger=ledger, guard_routing=False)

    # The scheduler did its job: plan == live decode order on every forward.
    assert ledger, "no decode forward was recorded"
    for row in ledger:
        assert row["plan"] == row["adapters"], (
            f"scheduler routing itself drifted, which is a different bug: {row}"
        )
        assert row["caches_aligned"], f"KV-cache ordering drifted: {row}"
    assert all(row["path"] == "replay" for row in ledger), (
        f"not every decode step went through replay: "
        f"{[r['path'] for r in ledger]}"
    )
    # The composition really did change under the frozen recording -- both ways.
    assert len({row["batch_size"] for row in ledger}) >= 2, (
        f"batch size never changed: {[r['batch_size'] for r in ledger]}"
    )
    same_size_reorder = [
        (a, b) for a, b in zip(ledger, ledger[1:])
        if a["batch_size"] == b["batch_size"] and a["adapters"] != b["adapters"]
    ]
    assert same_size_reorder, (
        "no two consecutive decode steps kept the batch size while changing the "
        "adapter at a row index; the workload no longer exercises a plan that "
        "lags the request order (mutation M3 would be undetectable). Observed: "
        f"{[(r['batch_size'], r['adapters']) for r in ledger]}"
    )

    corrupted = [rid for rid in eager if eager[rid] != frozen.get(rid)]
    assert corrupted, (
        "freezing capture-time routing did not change any output; the model "
        "of CUDA-graph behaviour is not exercising the routing branch"
    )
    assert set(corrupted) == set(eager), (
        f"expected every request to be corrupted, only {corrupted} were"
    )


def test_scheduler_must_not_serve_a_routed_batch_from_a_stale_recording():
    """THE B1 INVARIANT.

    A graph-accelerated scheduler must produce exactly the eager tokens for a
    LoRA workload -- either by making replay follow the live plan, or by
    refusing to replay a batch whose routing the recording does not match. The
    implemented fix takes the second route; this test is agnostic and passes
    under either.
    """
    model = _lora_model()
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)
    graph_served = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True)
    assert eager == graph_served, (
        "graph-accelerated decode served different tokens than eager decode "
        "for a mixed-adapter workload"
    )


def test_routed_batch_is_refused_and_the_reason_is_recorded():
    """The guard REFUSES the replay -- not "the output happened to be right".

    Asserts on the mechanism rather than the symptom: replay() is never entered
    for a routed batch, and the refusal is attributable through the existing
    fallback accounting instead of being silent.
    """
    model = _lora_model()
    out: dict = {}
    _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True, out=out)
    runner, sched = out["runner"], out["scheduler"]

    assert runner.replay_calls == 0, (
        f"a routed batch reached graph replay {runner.replay_calls} times; the "
        f"recording was captured unrouted, so those replays served the wrong "
        f"adapters"
    )
    diagnostics = sched.graph_diagnostics()
    assert diagnostics["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) > 0, (
        f"the routing refusal was not recorded in the fallback accounting: "
        f"{diagnostics}"
    )
    assert diagnostics["reasons"].get("graph_hit", 0) == 0, (
        f"a graph hit was counted for a routed workload: {diagnostics}"
    )
    assert diagnostics["reasons"].get("runner_rejected", 0) > 0, (
        f"the scheduler did not record the runner as the refusing party: "
        f"{diagnostics}"
    )


def test_all_base_batch_still_takes_the_graph_fast_path():
    """The guard must not cost base traffic its graph acceleration.

    An all-base batch's plan is [None, ...] and an unrouted recording's is
    None; those canonicalise to the same thing, so the fast path stays open.
    If this fails, the fix has over-reached into a performance regression for
    every non-LoRA request.
    """
    model = _lora_model()
    out: dict = {}
    served = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=True, out=out)
    runner, sched = out["runner"], out["scheduler"]
    eager = _serve(model, adapters=_BASE_ADAPTERS, runner_mode=None)

    diagnostics = sched.graph_diagnostics()
    assert runner.replay_calls > 0, "an all-base batch never reached replay"
    assert diagnostics["reasons"].get("graph_hit", 0) > 0, (
        f"an all-base batch lost the graph fast path: {diagnostics}"
    )
    assert diagnostics["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) == 0, (
        f"an all-base batch was refused as a routing mismatch: {diagnostics}"
    )
    assert served == eager, "the graph fast path changed base-workload output"


def test_routed_graph_cannot_serve_a_different_routing_plan():
    """Plan equality is exact -- a routed recording is not a wildcard.

    Captures under ["b", "a"] and then serves a workload whose size-2 batches
    are ["a", "b"] -- same batch size, same bucket, same caches, different
    adapter order. Every decode step must be refused: plan equality is exact,
    and a recording taken under one adapter mix must not serve another.
    """
    model = _lora_model()
    out: dict = {}
    served = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=True,
                    capture_plan=["b", "a"], out=out)
    runner, sched = out["runner"], out["scheduler"]
    eager = _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None)

    assert runner.graph_plans[(2, runner.buckets[0])] == ("b", "a"), (
        "the runner did not record the plan it captured under"
    )
    assert runner.replay_calls == 0, (
        f"a recording captured under a routed plan served "
        f"{runner.replay_calls} batches routed differently"
    )
    assert sched.graph_diagnostics()["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) > 0
    assert served == eager


def test_non_lora_model_is_unaffected_by_the_routing_guard():
    """A plain LlamaModel produces no plan, so the guard is inert for it."""
    plain = _tiny_model()
    assert not hasattr(plain, "get_batch_adapters")
    assert model_routing_plan(plain) is None

    sched = ContinuousBatchScheduler(
        plain, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = _StaticReplayRunner(
        plain, sched.pool, freeze_routing=True,
        max_batch_size=4, max_context_tokens=64,
    )
    runner.capture()
    sched._graph_runner = runner
    assert runner.graph_plans[(1, runner.buckets[0])] is None

    g = torch.Generator().manual_seed(7)
    for i in range(2):
        sched.add_request(f"p{i}", torch.randint(0, 256, (1, 6), generator=g),
                          max_new_tokens=5, eos_token_id=None)
    while sched.has_work():
        sched.step()

    diagnostics = sched.graph_diagnostics()
    assert runner.replay_calls > 0, "the non-LoRA path lost graph replay"
    assert diagnostics["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) == 0, (
        f"the routing guard fired for a model that has no routing: {diagnostics}"
    )


# ---------------------------------------------------------------------------
# CUDA-gated. These are the only tests here that can say anything about real
# CUDA graph execution. They are unverified on a CPU-only host -- that is a
# reported gap, not a passing result.
# ---------------------------------------------------------------------------


def _cuda_live_decode_batch(model, runner, batch: int, prompt_len: int = 6):
    """`batch` real prefilled caches on the runner's pool, prefilled under base.

    A routed capture can only be driven from a LIVE decode batch of the RIGHT
    SIZE: a per-row plan has one entry per batch row, so an n-entry plan is
    only meaningful against n rows. Requests are prefilled unrouted because the
    plan under test is a property of the decode batch, which is where the
    scheduler installs it.
    """
    pool = runner.pool
    model.clear_adapters()
    blocks = (prompt_len + 2 + pool.block_size - 1) // pool.block_size + 1
    g = torch.Generator().manual_seed(31)
    rids, caches = [], []
    for i in range(batch):
        rid = f"__routed_capture_{i}__"
        pool.admit_request(request_id=rid, prefill_blocks_needed=blocks,
                           total_blocks_needed=blocks)
        rids.append(rid)
        cache = PagedRequestCache(pool, rid, num_layers=pool.num_layers)
        ids = torch.randint(0, 256, (1, prompt_len), generator=g).cuda()
        with torch.no_grad():
            model(ids, kv_cache=cache)
        caches.append(cache)
    return rids, caches


@cuda_only
def test_cuda_capture_under_active_per_row_routing():
    """Does a real capture even SUCCEED while a per-row plan is active?

    THE HAZARD. `_apply_per_row` used to build its row-index tensor from a
    Python list inside forward, which on CUDA is a pageable H2D copy. A T4
    settled it: stream capture rejects that outright --

        RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
        graph capture unless the CPU tensor is pinned.

    -- for every routed plan it was tried with ([a,b], [a], [a,b,a], [a,b,b]).
    `LoRALinear.set_active` now prepares those indices on the device ahead of
    capture. This test is the hardware check that the preparation is sufficient:
    if capture ever stops succeeding under an active plan, the on-demand
    capture policy needs revisiting and this must go red.

    WHY IT IS DRIVEN FROM A LIVE BATCH AND NOT FROM `capture_all()`. An earlier
    version of this test routed the model to a two-entry plan and then called
    `capture_all()`, which walks EVERY batch size in the grid -- including 1.
    A two-entry plan against a one-row batch is not a CUDA question at all; it
    is the plan-length guard doing its job, and it made this test fail with

        ValueError: per-row adapter list length 2 != batch size 1

    while saying nothing about capture. A per-row plan sizes its batch, so the
    routed capture is exercised at the batch size the plan describes, through
    the same on-demand path production uses (`can_replay` -> `_capture_on_demand`).
    """
    plan = ["a", "b"]

    model = _lora_model(device="cuda")
    sched = ContinuousBatchScheduler(
        model, max_batch_size=len(plan), num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = LiveDecodeGraphRunner(
        model, sched.pool, max_batch_size=len(plan), max_context_tokens=64,
    )
    # The unrouted grid first, at the documented point: before any request is
    # admitted, so nothing is routed yet and the pool is empty.
    runner.capture_all(require_all=True)
    assert runner.report()["capture_failures"] == []

    rids, caches = _cuda_live_decode_batch(model, runner, batch=len(plan))
    try:
        model.set_batch_adapters(plan)
        live_plan = model_routing_plan(model)
        assert live_plan == ("a", "b"), live_plan
        servable = runner.can_replay(len(caches), caches, live_plan)
        report = runner.report()
        print(f"\n[B1/cuda] routed capture under {live_plan}: "
              f"servable={servable} failures={report['routed_capture_failures']}")

        # ORDER MATTERS: name the CUDA exception before the symptom. A refused
        # `can_replay` and a thrown capture look identical from the boolean.
        assert report["routed_capture_failures"] == [], (
            f"capture under an active per-row routing plan FAILED: "
            f"{report['first_routed_capture_error']}"
        )
        assert servable, (
            f"nothing captured for the live routed batch: {report} "
            f"reasons={dict(runner.fallback_reasons)}"
        )
        # The capture really produced a graph, and it is keyed by the plan it
        # ran under -- B2: a batch routed differently cannot reach it.
        assert any(len(k) == 3 and k[2] == ("a", "b") for k in runner.graphs), (
            f"no graph is keyed by the active per-row plan: {sorted(runner.graphs)}"
        )
        assert report["routed_plans_captured"] == ["[a,b]"], (
            f"capture did not record the active per-row plan: {report}")
        assert "[a,b]" in report["routing_plans_captured"], report
        assert report["graphs_captured"] > 0
    finally:
        for rid in rids:
            sched.pool.free_request(rid)


@cuda_only
def test_cuda_replay_honours_live_lora_routing():
    """The real thing: capture unrouted, then serve a mixed-adapter batch.

    HISTORY. On the T4 this reported `graph_hits: 0` with
    `lora_routing_mismatch: 9`. That was the guard working and the capture
    policy missing: routed batches were correctly refused, but nothing ever
    captured a graph for the plans they used, so there was nothing to hit. The
    refusal was right; the zero was not acceptable.

    With B2 (plan in the graph key) and B3 (the live plan threaded into
    `replay`), `capture_all()` still captures only the unrouted grid -- it runs
    before any request exists -- and each routed plan is captured on demand the
    first time the scheduler actually produces it. So those same 9 steps must
    now be graph hits, and the tokens must still match eager exactly.

    Both halves matter. Hits without token equality would mean a graph is
    replaying the wrong adapter arithmetic, which is the failure this whole
    module exists to prevent.
    """
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _lora_model(device="cuda")

    def serve(attach_runner):
        model.clear_adapters()
        sched = ContinuousBatchScheduler(
            model, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
            use_cuda_graphs=True,
        )
        if attach_runner:
            runner = LiveDecodeGraphRunner(
                model, sched.pool, max_batch_size=4, max_context_tokens=64,
            )
            runner.capture_all()          # unrouted, per documented usage
            sched._graph_runner = runner
        g = torch.Generator().manual_seed(7)
        tokens: dict[str, list[int]] = {}
        prompts = {rid: torch.randint(0, 256, (1, 6), generator=g).cuda()
                   for rid, _, _, _ in _WORKLOAD}
        for step in range(60):
            for rid, _a, max_new, admit_step in _WORKLOAD:
                if step == admit_step:
                    sched.add_request(rid, prompts[rid], max_new_tokens=max_new,
                                      eos_token_id=None,
                                      adapter_id=_LORA_ADAPTERS[rid])
            if not sched.has_work() and step > max(w[3] for w in _WORKLOAD):
                break
            for rid, token_id in sched.step():
                tokens.setdefault(rid, []).append(token_id)
        return tokens, sched

    eager, _ = serve(False)
    graphed, sched = serve(True)
    diagnostics = sched.graph_diagnostics()
    hits = diagnostics["reasons"].get("graph_hit", 0)
    report = diagnostics["runner_report"]

    # ORDER MATTERS. Assert the capture exception FIRST. A routed capture that
    # throws every step yields hits == 0, which is also what a genuine routing
    # refusal looks like -- and asserting `hits > 0` first reports the symptom
    # while the cause sits unread in the report. That is precisely how a T4 run
    # came back blaming `lora_routing_mismatch` for a CUDA capture failure.
    assert report["routed_capture_failures"] == [], (
        f"on-demand routed capture FAILED -- this, not the routing guard, is "
        f"why there are no graph hits. "
        f"first error: {report['first_routed_capture_error']} | "
        f"all failures: {report['routed_capture_failures']}"
    )
    assert hits > 0, (
        f"no graph replay happened, so this test proved nothing: {diagnostics}"
    )
    # The point of B2/B3: the hits must come from ROUTED graphs, not merely
    # from whatever base-only steps the workload happened to contain.
    routed = [p for p in report["routing_plans_captured"] if p != "base"]
    assert routed, (
        f"only the unrouted grid was ever captured, so the mixed-adapter "
        f"batches all fell back: {diagnostics}"
    )
    assert eager == graphed, (
        f"CUDA graph replay served different tokens than eager decode for a "
        f"mixed-adapter workload ({hits} replays, routed plans {routed})"
    )


# ---------------------------------------------------------------------------
# B2 + B3. Routing plan as part of GRAPH IDENTITY.
#
# The doubles above use the LEGACY key layout -- `graphs[(batch, blocks)]` plus
# a `graph_plans` side table -- which production no longer writes and which
# these tests deliberately keep exercised (backwards compatibility). The runner
# below uses the layout `LiveDecodeGraphRunner._capture` now writes:
#
#     graphs[(batch_size, n_blocks, canonical_routing_plan)]
#
# so a plan mismatch is a MISSING KEY rather than a comparison some future edit
# could forget to make. Still CPU-only and still not a CUDA result: it swaps
# `graph.replay()` for the same `StaticDecodeBatch.run_eager` the other doubles
# use. What it pins is the routing/identity logic, which is pure host code and
# byte-identical on both.
# ---------------------------------------------------------------------------


class _RoutedStaticRunner(DecodeGraphRouter):
    """CPU stand-in keyed the way production keys routed graphs."""

    def __init__(self, model, pool, **kwargs):
        super().__init__(pool, **kwargs)
        self.model = model
        self.replay_calls = 0
        self.replay_plans: list = []
        self.captured_plans: list = []

    def capture_plan(self, plan) -> None:
        """Capture the whole (batch x bucket) grid under one routing plan.

        `plan` is canonicalised exactly as `_capture` does, so `[None, None]`
        registers as base and cannot masquerade as a distinct routed plan.
        """
        canonical = normalize_routing_plan(plan)
        self.captured_plans.append(canonical)
        for batch_size in self.batch_sizes:
            for n_blocks in self.buckets:
                self.graphs[(batch_size, n_blocks, canonical)] = (
                    StaticDecodeBatch(self.pool, batch_size, n_blocks),)

    def replay(self, input_ids, caches, routing_plan=None):
        self.replay_plans.append(routing_plan)
        key = self._resolve(len(caches), caches, routing_plan)
        assert key is not None, "replay() called without a routable key"
        self.replay_calls += 1
        (state,) = self.graphs[key]
        state.stage(input_ids, caches)
        out = state.run_eager(self.model)
        state.commit(caches)
        return out


def _drive(sched, adapters):
    """Run _WORKLOAD through an already-configured scheduler."""
    g = torch.Generator().manual_seed(7)
    prompts = {rid: torch.randint(0, 256, (1, 6), generator=g)
               for rid, _, _, _ in _WORKLOAD}
    tokens: dict[str, list[int]] = {}
    for step in range(60):
        for rid, _a, max_new, admit_step in _WORKLOAD:
            if step == admit_step:
                sched.add_request(rid, prompts[rid], max_new_tokens=max_new,
                                  eos_token_id=None, adapter_id=adapters[rid])
        if not sched.has_work() and step > max(w[3] for w in _WORKLOAD):
            break
        for rid, token_id in sched.step():
            tokens.setdefault(rid, []).append(token_id)
    return tokens


def _new_sched(model):
    model.clear_adapters()
    return ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )


def _serve_routed(model, *, capture_plans, adapters, out=None):
    """Serve _WORKLOAD with a routed-key runner pre-captured under each plan."""
    sched = _new_sched(model)
    runner = _RoutedStaticRunner(model, sched.pool, max_batch_size=4,
                                max_context_tokens=64)
    for plan in capture_plans:
        runner.capture_plan(plan)
    sched._graph_runner = runner
    if out is not None:
        out["runner"], out["scheduler"] = runner, sched
    return _drive(sched, adapters)


def _eager_reference(model, adapters):
    """Same workload, no runner attached -- the ground-truth token stream."""
    return _drive(_new_sched(model), adapters)


def test_invariant_1_unrouted_capture_refuses_routed_batches():
    """Capture under base only + live routed plan => every replay refused."""
    model = _lora_model()
    out: dict = {}
    served = _serve_routed(model, capture_plans=[None],
                           adapters=_LORA_ADAPTERS, out=out)
    runner, sched = out["runner"], out["scheduler"]

    routed = [p for p in runner.replay_plans if p is not None]
    assert not routed, f"a base-only recording served {len(routed)} routed batches"
    assert sched.graph_diagnostics()["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) > 0, (
        "routed batches were refused for some reason OTHER than the routing "
        f"guard: {sched.graph_diagnostics()}")
    assert served == _eager_reference(model, _LORA_ADAPTERS)


def test_invariant_2_matching_routed_plan_is_allowed():
    """Capture under P + live P => replay ALLOWED, and output still exact.

    This is the invariant that could not hold before B2: production recorded no
    plan at all, so `captured_plan` was always None and every routed batch was
    refused. A zero hit rate under LoRA was structural, not a policy choice.
    """
    model = _lora_model()
    ledger: list = []
    _serve(model, adapters=_LORA_ADAPTERS, runner_mode=None, ledger=ledger)
    plans = {tuple(e["plan"]) if isinstance(e["plan"], list) else e["plan"]
             for e in ledger}

    out: dict = {}
    served = _serve_routed(model, capture_plans=sorted(plans, key=str),
                           adapters=_LORA_ADAPTERS, out=out)
    runner, sched = out["runner"], out["scheduler"]

    assert runner.replay_calls > 0, (
        f"no batch was served from a matching recording: "
        f"{sched.graph_diagnostics()}")
    assert any(p is not None for p in runner.replay_plans), (
        "only base batches replayed; the routed path proved nothing")
    assert sched.graph_diagnostics()["reasons"].get("graph_hit", 0) > 0
    assert served == _eager_reference(model, _LORA_ADAPTERS), (
        "replaying a graph captured under the live plan changed the output")


def test_invariant_3_different_routed_plan_is_refused():
    """Capture under P + live Q => refused, even though shape and bucket match."""
    model = _lora_model()
    foreign = ("b", "a", "b", "a")
    out: dict = {}
    served = _serve_routed(model, capture_plans=[foreign],
                           adapters=_LORA_ADAPTERS, out=out)
    runner, sched = out["runner"], out["scheduler"]

    for plan in runner.replay_plans:
        assert plan == foreign, (
            f"a recording captured under {foreign} served plan {plan!r}")
    assert sched.graph_diagnostics()["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) > 0
    assert served == _eager_reference(model, _LORA_ADAPTERS)


def test_invariant_4_base_only_workload_still_replays():
    """The routing guard must not cost the base workload its graph hits."""
    model = _lora_model()
    out: dict = {}
    served = _serve_routed(model, capture_plans=[None],
                           adapters=_BASE_ADAPTERS, out=out)
    runner, sched = out["runner"], out["scheduler"]

    assert runner.replay_calls > 0, "base workload lost graph replay"
    assert all(p is None for p in runner.replay_plans)
    assert sched.graph_diagnostics()["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) == 0, (
        "an all-base batch was refused as a routing mismatch")
    assert served == _eager_reference(model, _BASE_ADAPTERS)


def test_invariant_5_plain_model_routing_machinery_is_inert():
    """A plain LlamaModel yields plan None everywhere; nothing changes."""
    plain = _tiny_model()
    assert model_routing_plan(plain) is None

    sched = ContinuousBatchScheduler(
        plain, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
    )
    runner = _RoutedStaticRunner(plain, sched.pool, max_batch_size=4,
                                 max_context_tokens=64)
    runner.capture_plan(None)
    sched._graph_runner = runner

    g = torch.Generator().manual_seed(7)
    for i in range(2):
        sched.add_request(f"p{i}", torch.randint(0, 256, (1, 6), generator=g),
                          max_new_tokens=5, eos_token_id=None)
    while sched.has_work():
        sched.step()

    assert runner.replay_calls > 0
    assert all(p is None for p in runner.replay_plans)
    assert sched.graph_diagnostics()["runner_fallback_reasons"].get(
        "lora_routing_mismatch", 0) == 0


def test_invariant_6_replay_receives_the_schedulers_live_plan():
    """B3: the plan `_decode_forward` computed is the plan `replay` resolves.

    Pins the threading itself, not just its effect. Every plan `replay` was
    called with must equal the model's live plan at that moment, which the spy
    on `_decode_forward` records independently.
    """
    model = _lora_model()
    ledger: list = []
    sched = _new_sched(model)
    runner = _RoutedStaticRunner(model, sched.pool, max_batch_size=4,
                                 max_context_tokens=64)
    runner.capture_plan(None)
    sched._graph_runner = runner

    original = ContinuousBatchScheduler._decode_forward

    def spy(self, input_ids, caches):
        before = len(runner.replay_plans)
        live = model_routing_plan(model)
        result = original(self, input_ids, caches)
        if len(runner.replay_plans) > before:          # this step replayed
            ledger.append((live, runner.replay_plans[-1]))
        return result

    ContinuousBatchScheduler._decode_forward = spy
    try:
        _drive(sched, _BASE_ADAPTERS)
    finally:
        ContinuousBatchScheduler._decode_forward = original

    assert ledger, "no replay happened, so the threading was never exercised"
    for live, passed in ledger:
        assert live == passed, (
            f"replay() resolved with {passed!r} while the scheduler's live "
            f"plan was {live!r}; the plan did not reach replay unchanged")


def _bare_pool(model):
    return PagedKVCache(
        num_layers=model.config.num_hidden_layers, num_blocks=64,
        block_size=BLOCK_SIZE, num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=torch.float32, device="cpu", enable_prefix_cache=False,
    )


def test_routed_keys_and_legacy_keys_both_resolve():
    """Backwards compatibility: the 2-tuple + graph_plans layout still works.

    `_StaticReplayRunner` (used by every test above this section) writes the
    legacy layout. This asserts the two coexist in one router and that neither
    can serve the other's plan.
    """
    r = DecodeGraphRouter(_bare_pool(_lora_model()), max_batch_size=2,
                          max_context_tokens=64)
    nb = r.buckets[0]
    r.graphs[(1, nb, None)] = ("routed-base",)
    r.graphs[(1, nb, ("a",))] = ("routed-a",)
    r.graphs[(2, nb)] = ("legacy",)
    r.graph_plans = {(2, nb): ["b", "b"]}

    assert r._lookup(1, nb, None) == (1, nb, None)
    assert r._lookup(1, nb, ("a",)) == (1, nb, ("a",))
    assert r._lookup(1, nb, ("b",)) is None
    # Legacy: the side table's plan is canonicalised before comparison.
    assert r._lookup(2, nb, ("b", "b")) == (2, nb)
    assert r._lookup(2, nb, None) is None
    assert r._lookup(2, nb, ("a", "b")) is None


def test_all_none_plan_is_canonically_base():
    """[None, None] is base -- it must NOT become a distinct routed graph."""
    model = _lora_model()
    r = _RoutedStaticRunner(model, _bare_pool(model), max_batch_size=2,
                            max_context_tokens=64)
    r.capture_plan([None, None])
    assert r.captured_plans == [None]
    nb = r.buckets[0]
    assert r._lookup(2, nb, None) == (2, nb, None)
    assert len(r.graphs) == len(r.batch_sizes) * len(r.buckets)


# ---------------------------------------------------------------------------
# The ON-DEMAND capture policy, exercised on CPU.
#
# `LiveDecodeGraphRunner.can_replay` is the method that decides whether to
# capture a graph for a newly-seen routing plan, and it is the only part of the
# B2/B3 change that lives on the CUDA-only class. Left untested it would first
# execute on the T4, which is the exact situation this repository has twice paid
# for. So the routing/budget/bookkeeping half is driven here through the REAL
# method, with only `_capture` -- the part that genuinely needs a GPU --
# replaced by a recorder.
#
# This is NOT a claim that CUDA capture works. It pins the policy: which plans
# get captured, how often, what happens at the budget, and that a capture
# failure degrades to eager instead of propagating.
# ---------------------------------------------------------------------------


class _FakeCudaRunner(LiveDecodeGraphRunner):
    """Real `can_replay` / `_capture_on_demand`; `_capture` records instead."""

    def __init__(self, pool, *, max_routed_graphs=16, fail_plans=()):
        DecodeGraphRouter.__init__(self, pool, max_batch_size=4,
                                   max_context_tokens=64)
        self.model = None
        self.device = pool.K_pool.device
        self._mempool = None
        self.max_routed_graphs = max_routed_graphs
        self.routed_plans = []
        self.routed_capture_failures = []
        self.captured: list = []
        self._fail_plans = set(fail_plans)

    def _capture(self, batch_size, n_blocks, plan=None, caches=None):
        if plan in self._fail_plans:
            raise RuntimeError("simulated capture failure (e.g. OOM)")
        assert caches is not None, (
            "an on-demand capture must stage the LIVE batch, not placeholders")
        self.captured.append((batch_size, n_blocks, plan))
        self.graphs[(batch_size, n_blocks, plan)] = ("recorded",)


def _cuda_sync_noop(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def _fixture_caches(model, batch=2):
    pool, caches, _ids = _decode_fixture(model, batch=batch)
    return pool, caches


def test_on_demand_captures_a_new_routed_plan_once(monkeypatch):
    """First sighting of a plan captures; later sightings reuse."""
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool)

    plan = ("a", "b")
    assert r.can_replay(2, caches, plan) is True
    assert len(r.captured) == 1, r.captured
    assert r.routed_plans == [plan]

    for _ in range(3):
        assert r.can_replay(2, caches, plan) is True
    assert len(r.captured) == 1, "a cached routed plan was re-captured"
    assert r.fallback_reasons == {}, (
        f"a step that ended in a hit also recorded a fallback: "
        f"{r.fallback_reasons}")


def test_on_demand_does_not_capture_the_unrouted_plan(monkeypatch):
    """Base is capture_all()'s job; a base miss is a real capture failure."""
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool)

    assert r.can_replay(2, caches, None) is False
    assert r.captured == [], "on-demand capture papered over a missing base grid"
    assert r.fallback_reasons.get("bucket_capture_failed", 0) == 1


def test_on_demand_respects_the_routed_graph_budget(monkeypatch):
    """Past the cap, new plans fall back and say so; existing ones still hit."""
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool, max_routed_graphs=2)

    assert r.can_replay(2, caches, ("a", "b")) is True
    assert r.can_replay(2, caches, ("b", "a")) is True
    # Third distinct plan exceeds the budget.
    assert r.can_replay(2, caches, ("a", "a")) is False
    assert r.fallback_reasons.get("routed_graph_budget_exhausted", 0) == 1
    assert len(r.routed_plans) == 2
    # A plan captured before the cap still replays.
    assert r.can_replay(2, caches, ("a", "b")) is True


def test_on_demand_capture_failure_degrades_to_eager(monkeypatch):
    """A failed capture is counted and falls back; it never propagates."""
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    plan = ("a", "b")
    r = _FakeCudaRunner(pool, fail_plans={plan})

    assert r.can_replay(2, caches, plan) is False        # no exception escapes
    assert r.fallback_reasons.get("routed_capture_failed", 0) == 1
    assert len(r.routed_capture_failures) == 1
    assert "simulated capture failure" in r.routed_capture_failures[0]["error"]
    assert r.report()["routed_capture_failures"], (
        "a routed capture failure must be visible in the runner report")


def test_on_demand_never_serves_a_plan_it_did_not_capture(monkeypatch):
    """The guard still holds: capturing P must not make Q replayable."""
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool, max_routed_graphs=1)

    assert r.can_replay(2, caches, ("a", "b")) is True
    assert r.can_replay(2, caches, ("b", "a")) is False   # budget exhausted
    # The captured key is P's, and Q resolves to nothing.
    assert r._lookup(2, r.buckets[0], ("a", "b")) is not None
    assert r._lookup(2, r.buckets[0], ("b", "a")) is None


def test_report_plan_strings_are_json_safe(monkeypatch):
    """Graph keys mix None/str/tuple; the report must still sort and serialise."""
    import json
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool)
    r.graphs[(1, r.buckets[0], None)] = ("base",)
    r.can_replay(2, caches, ("a", "b"))
    r.can_replay(2, caches, "a")

    report = r.report()
    json.dumps(report)                       # must not raise
    assert "base" in report["routing_plans_captured"]
    assert "[a,b]" in report["routing_plans_captured"]
    assert "a" in report["routing_plans_captured"]
    assert sum(report["graphs_per_routing_plan"].values()) == len(r.graphs)


def test_failed_routed_capture_is_not_reported_as_a_routing_mismatch(monkeypatch):
    """A CUDA capture that throws must NOT be labelled `lora_routing_mismatch`.

    THE BUG THIS PINS. `lora_routing_mismatch` means "a graph exists for this
    shape under a DIFFERENT plan and we refused to reuse it" -- the guard doing
    its job. When on-demand capture threw, `can_replay` fell through to
    `_resolve`, which found the base grid at this shape and recorded exactly
    that reason. So a T4 run whose every routed capture raised reported a
    routing bug that did not exist, while the real exception sat unread in
    `routed_capture_failures`. Two reasons were recorded for one decode
    forward, which also breaks the module's one-reason-per-forward contract.
    """
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    plan = ("a", "b")
    r = _FakeCudaRunner(pool, fail_plans={plan})
    # A base grid at this shape, exactly as capture_all() leaves it -- this is
    # what `_resolve` used to latch onto and misreport.
    for nb in r.buckets:
        r.graphs[(2, nb, None)] = ("base",)

    assert r.can_replay(2, caches, plan) is False

    assert r.fallback_reasons.get("lora_routing_mismatch", 0) == 0, (
        f"a capture failure was reported as a routing mismatch: "
        f"{dict(r.fallback_reasons)}")
    assert r.fallback_reasons.get("routed_capture_failed", 0) == 1
    assert sum(r.fallback_reasons.values()) == 1, (
        f"one decode forward recorded {sum(r.fallback_reasons.values())} "
        f"reasons: {dict(r.fallback_reasons)}")
    assert r.report()["first_routed_capture_error"], (
        "the capture exception must be surfaced at the top of the report")
    assert "simulated capture failure" in r.report()["first_routed_capture_error"]


def test_budget_exhaustion_is_not_reported_as_a_routing_mismatch(monkeypatch):
    """Same contract for the other early exit: one reason, and the right one."""
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool, max_routed_graphs=0)
    for nb in r.buckets:
        r.graphs[(2, nb, None)] = ("base",)

    assert r.can_replay(2, caches, ("a", "b")) is False
    assert r.fallback_reasons == {"routed_graph_budget_exhausted": 1}, (
        f"expected exactly one budget reason, got {dict(r.fallback_reasons)}")


def test_genuine_routing_mismatch_still_reports_itself(monkeypatch):
    """The guard must still be able to say `lora_routing_mismatch`.

    The fix above must not make that reason unreachable: when a graph really
    does exist under another plan and capture is not attempted (budget spent on
    that plan already), the refusal is the guard's, and it must say so.
    """
    _cuda_sync_noop(monkeypatch)
    model = _lora_model()
    pool, caches = _fixture_caches(model)
    r = _FakeCudaRunner(pool)
    nb = r.buckets[0]
    # A graph captured under a routed plan, and a live batch routed differently.
    r.graphs[(2, nb, ("b", "a"))] = ("routed-other",)
    # Take the plan out of contention for on-demand capture by exhausting the
    # budget with it already recorded.
    r.max_routed_graphs = 0

    assert r.can_replay(2, caches, ("a", "b")) is False
    # Budget is the proximate reason here; the point is that _resolve still
    # reports a mismatch when it is reached with no capture attempt pending.
    r2 = _FakeCudaRunner(pool)
    r2.graphs[(2, nb, ("b", "a"))] = ("routed-other",)
    assert r2._lookup(2, nb, ("a", "b")) is None
    assert r2._resolve(2, caches, ("a", "b")) is None
    assert r2.fallback_reasons.get("lora_routing_mismatch", 0) == 1
