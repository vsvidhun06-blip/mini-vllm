"""
Live-decode CUDA-graph contract tests (src/engine/live_graph.py).

WHAT THESE PIN, AND WHERE
-------------------------
The first T4 smoke run of scripts/eval/repair/batch_intervention.py produced two
"CUDA graph" rows with `cuda_graph_hits == 0`: they were eager runs wearing a
graph label. These tests exist so that cannot recur silently. They split along
what a machine can honestly verify:

  CPU (runs everywhere, no GPU needed)
    * The STATIC-SHAPE decode path -- padded block-table gather, additive -inf
      bias, on-device K/V scatter, host-side seq_len commit -- produces
      logits IDENTICAL to the ordinary eager decode step. This is the part of
      the repair that could silently corrupt the model, and it is fully
      testable without CUDA.
    * The ROUTING/VALIDATION contract: a scheduler with no runner attached
      records only eager fallbacks, and the experiment's accounting refuses to
      call such a run a CUDA-graph arm.
    * `LiveDecodeGraphRunner` REFUSES to construct on CPU rather than degrading
      to something that reports graph hits it never had.

  CUDA (skipped on CPU-only hosts -- the T4 is where these must pass)
    * Capture actually succeeds for the (batch size x context bucket) grid.
    * Replay of a captured graph equals eager decode for the same state.
    * A live scheduler with a runner attached records hits > 0, and repeated
      steps keep hitting as the context GROWS (the property the old
      cuda_graph.CUDAGraphRunner could never have, since its graphs froze
      seq_len and bound themselves to their own cache objects).
    * An uncaptured batch size falls back and is counted as a fallback, never
      as a hit.

NOTHING HERE FABRICATES A CUDA RESULT. The CPU tests make no claim about graph
execution; the GPU tests are the only ones that assert replays happened.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.engine.kv_cache import PagedKVCache, PagedRequestCache  # noqa: E402
from src.engine.live_graph import DecodeGraphRouter, StaticDecodeBatch  # noqa: E402
from src.engine.model import LlamaConfig, LlamaModel  # noqa: E402
from src.engine.scheduler import ContinuousBatchScheduler  # noqa: E402

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA graphs are a CUDA-only feature; capture/replay must be "
           "verified on the GPU box, never asserted from CPU",
)

BLOCK_SIZE = 16


def _tiny_model(device="cpu"):
    """head_dim = 128/8 = 16, a power of two, so the from-scratch FA2 kernel
    (the CUDA decode path) accepts it."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=4096, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval().to(device)


def _prefilled_pool(model, prompt_lens, device="cpu"):
    """A pool with len(prompt_lens) requests prefilled to their prompt length.

    Returns (pool, caches). Uses the real prefill forward so the KV contents,
    block tables and per-layer seq_lens are exactly what a served request has
    when it enters DECODE.
    """
    cfg = model.config
    pool = PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_blocks=64, block_size=BLOCK_SIZE,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        dtype=next(model.parameters()).dtype, device=device,
        enable_prefix_cache=False,
    )
    caches = []
    for i, L in enumerate(prompt_lens):
        rid = f"r{i}"
        blocks = (L + 32 + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool.admit_request(request_id=rid, prefill_blocks_needed=blocks,
                           total_blocks_needed=blocks)
        c = PagedRequestCache(pool, rid, num_layers=cfg.num_hidden_layers)
        prompt = torch.randint(0, cfg.vocab_size, (1, L), device=device)
        with torch.no_grad():
            model(prompt, kv_cache=c)
        caches.append(c)
    return pool, caches


def _snapshot(pool, caches):
    return {
        "K": pool.K_pool.clone(),
        "V": pool.V_pool.clone(),
        "seq_lens": [list(c._seq_lens) for c in caches],
        "blocks": {k: list(v) for k, v in pool._blocks.items()},
        "reserved": dict(pool._reserved),
        "free": set(pool._free_blocks),
    }


def _restore(pool, caches, snap):
    pool.K_pool.copy_(snap["K"])
    pool.V_pool.copy_(snap["V"])
    for c, sl in zip(caches, snap["seq_lens"]):
        c._seq_lens[:] = list(sl)
        c._bt_tensor = None
        c._seqlen_tensors.clear()
        c._seqlen_vals.clear()
    pool._blocks = {k: list(v) for k, v in snap["blocks"].items()}
    pool._reserved = dict(snap["reserved"])
    pool._free_blocks = set(snap["free"])


# ---------------------------------------------------------------------------
# CPU: the static-shape decode path must be numerically identical to eager.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_lens", [[20], [20, 33], [17, 40, 9, 64]])
def test_static_shape_decode_matches_eager(prompt_lens):
    """The padded gather + additive -inf bias + on-device scatter must give the
    same logits as the ordinary trimmed-gather eager decode step.

    TOLERANCE, AND WHY IT IS NOT EXACT HERE. Every masked column contributes
    exp(-inf - m) == 0, so the padding adds nothing in exact arithmetic. On the
    CUDA path that is also true in FLOATING-POINT arithmetic -- the FA2 kernel's
    online softmax folds a fully-masked KV tile in as `p == 0, alpha == 1`,
    leaving the accumulator untouched -- and the GPU test below asserts bitwise
    equality. On CPU the attention is `F.scaled_dot_product_attention`, which
    picks its own blocking for a 96-key masked call versus a 21-key trimmed
    call; feeding it identical q/k/v both ways already disagrees at ~1e-7
    before any of this module's code is involved. So the CPU claim is a tight
    tolerance plus EXACT agreement on the emitted token, which is the only thing
    the scheduler reads out of these logits.
    """
    model = _tiny_model()
    B = len(prompt_lens)
    pool, caches = _prefilled_pool(model, prompt_lens)
    snap = _snapshot(pool, caches)

    input_ids = torch.randint(0, model.config.vocab_size, (B, 1))

    # --- eager reference: the pre-existing decode path -------------------
    with torch.no_grad():
        eager = model(input_ids, kv_cache=caches)
    eager_seq_lens = [list(c._seq_lens) for c in caches]
    eager_K = pool.K_pool.clone()
    eager_V = pool.V_pool.clone()

    # --- static-shape path from the identical starting state -------------
    _restore(pool, caches, snap)
    max_ctx = max(prompt_lens)
    n_blocks = (max_ctx + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE + 2  # deliberate slack
    state = StaticDecodeBatch(pool, B, n_blocks)
    state.stage(input_ids, caches)
    static = state.run_eager(model)
    state.commit(caches)

    assert static.shape == eager.shape
    assert torch.allclose(static, eager, rtol=0.0, atol=1e-5), (
        "static-shape decode diverged from eager decode by more than SDPA's own "
        f"masked-vs-trimmed noise: max|d| = {(static - eager).abs().max():.3e}")
    assert torch.equal(static[:, -1].argmax(-1), eager[:, -1].argmax(-1)), (
        "static-shape decode emitted a different token than eager decode")
    # The KV writes must land in the same physical slots. (Same tolerance
    # story: layer 0's K/V is bit-identical, later layers inherit the attention
    # noise above.)
    assert torch.allclose(pool.K_pool, eager_K, rtol=0.0, atol=1e-5), (
        "graph-path K/V scatter wrote to different slots than eager append")
    assert torch.allclose(pool.V_pool, eager_V, rtol=0.0, atol=1e-5), (
        "graph-path K/V scatter wrote to different slots than eager append")
    assert torch.equal(pool.K_pool[0], eager_K[0]), (
        "layer 0 K is attention-independent, so the scatter must be exact")
    # ...and the host-side bookkeeping must advance identically.
    assert [list(c._seq_lens) for c in caches] == eager_seq_lens


def test_static_shape_decode_matches_eager_across_a_block_boundary():
    """Padding bounds are easiest to get wrong when a decode step crosses into a
    fresh block. Drive several consecutive steps starting one token short of a
    block boundary and require token-for-token agreement with eager."""
    model = _tiny_model()
    prompt_lens = [BLOCK_SIZE * 2 - 1, BLOCK_SIZE + 3]
    B = len(prompt_lens)

    pool, caches = _prefilled_pool(model, prompt_lens)
    snap = _snapshot(pool, caches)
    ids = [torch.randint(0, model.config.vocab_size, (B, 1)) for _ in range(6)]

    eager_tokens = []
    for step_ids in ids:
        with torch.no_grad():
            out = model(step_ids, kv_cache=caches)
        eager_tokens.append(out[:, -1].argmax(-1).tolist())

    _restore(pool, caches, snap)
    n_blocks = 8
    state = StaticDecodeBatch(pool, B, n_blocks)
    graph_tokens = []
    for step_ids in ids:
        state.stage(step_ids, caches)
        out = state.run_eager(model)
        state.commit(caches)
        graph_tokens.append(out[:, -1].argmax(-1).tolist())

    assert graph_tokens == eager_tokens


def test_static_batch_refuses_a_context_that_overflows_its_buckets():
    """A row whose context needs more blocks than the state holds must RAISE,
    not silently truncate the attention window. Silent truncation would be the
    'captured one shape, claimed all shapes' failure mode."""
    model = _tiny_model()
    pool, caches = _prefilled_pool(model, [40])
    state = StaticDecodeBatch(pool, 1, n_blocks=1)   # 16 tokens; needs 3
    with pytest.raises(ValueError, match="needs 3 blocks"):
        state.stage(torch.zeros((1, 1), dtype=torch.long), caches)


def test_static_row_cache_has_no_host_seq_len():
    """The host-side int is deliberately unavailable: a captured graph reads the
    context from device memory, so a Python value could only ever be stale."""
    model = _tiny_model()
    pool, _ = _prefilled_pool(model, [20])
    state = StaticDecodeBatch(pool, 1, 4)
    with pytest.raises(RuntimeError, match="no host-side seq_len"):
        state.rows[0].seq_len(0)


def test_eager_decode_is_unchanged_by_the_bias_hook():
    """An ordinary PagedRequestCache has no `decode_attn_bias`, so the eager
    decode path must be byte-identical to what it was before the hook."""
    model = _tiny_model()
    pool, caches = _prefilled_pool(model, [12, 30])
    for c in caches:
        assert not hasattr(c, "decode_attn_bias")
    ids = torch.randint(0, model.config.vocab_size, (2, 1))
    snap = _snapshot(pool, caches)
    with torch.no_grad():
        a = model(ids, kv_cache=caches)
    _restore(pool, caches, snap)
    with torch.no_grad():
        b = model(ids, kv_cache=caches)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# CPU: the routing / validation contract.
# ---------------------------------------------------------------------------


def test_runner_refuses_to_construct_on_cpu():
    """No CPU stand-in. A runner that 'worked' on CPU could report hits that
    were never CUDA graph replays -- exactly the mislabelling being repaired."""
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model()
    pool, _ = _prefilled_pool(model, [16])
    with pytest.raises(RuntimeError, match="requires a CUDA pool"):
        LiveDecodeGraphRunner(model, pool, max_batch_size=2, max_context_tokens=64)


def test_scheduler_with_no_runner_records_only_eager_fallbacks():
    """The pre-repair state, pinned as a test: `use_cuda_graphs=True` with no
    runner attached must record ZERO hits and one fallback per decode forward.
    This is the observation the first T4 smoke made; the accounting must keep
    making it rather than reporting the requested flag as fact."""
    model = _tiny_model()
    seen = {"hit": 0, "eager": 0}
    sched = ContinuousBatchScheduler(
        model, max_batch_size=2, num_blocks=64, block_size=BLOCK_SIZE,
        use_cuda_graphs=True,
        cuda_graph_observer=lambda hit: seen.__setitem__(
            "hit" if hit else "eager", seen["hit" if hit else "eager"] + 1),
    )
    assert sched._graph_runner is None
    sched.add_request("r", torch.randint(0, 256, (1, 6)), max_new_tokens=8,
                      eos_token_id=None)
    while sched.has_work():
        sched.step()

    assert seen["hit"] == 0
    assert seen["eager"] > 0


def test_graph_accounting_contract():
    """The experiment's row-level self-verification.

    Four properties, one per line of the Phase 16 contract:
      * requested + replays  -> arm valid, labelled as graph execution
      * requested + none     -> arm INVALID, labelled as eager execution
      * eager arm            -> always valid (it makes no graph claim)
      * fallbacks            -> never counted as hits
    """
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "eval", "repair"))
    from batch_intervention import GRAPH_CLEAN_HIT_RATE, graph_accounting

    hit = graph_accounting(requested=True, hits=600, fallbacks=0)
    assert hit["cuda_graph_arm_valid"] is True
    assert hit["cuda_graph_arm_clean"] is True
    assert hit["use_cuda_graphs"] is True
    assert hit["cuda_graph_hits"] == 600
    assert hit["cuda_graph_hit_rate"] == 1.0

    # The exact shape of the first T4 smoke's graph rows.
    dead = graph_accounting(requested=True, hits=0, fallbacks=2350)
    assert dead["cuda_graph_arm_valid"] is False
    assert dead["cuda_graph_arm_clean"] is False
    assert dead["use_cuda_graphs"] is False, (
        "an arm that never replayed a graph must not be labelled graph execution")
    assert dead["cuda_graph_hits"] == 0
    assert dead["cuda_graph_eager_fallbacks"] == 2350
    assert dead["cuda_graph_hit_rate"] == 0.0

    eager = graph_accounting(requested=False, hits=0, fallbacks=2350)
    assert eager["cuda_graph_arm_valid"] is True
    assert eager["use_cuda_graphs"] is False

    # A mixture is valid (graphs did run) but not clean, so the slope
    # comparison excludes it.
    mixed = graph_accounting(requested=True, hits=10, fallbacks=90)
    assert mixed["cuda_graph_arm_valid"] is True
    assert mixed["cuda_graph_arm_clean"] is False
    assert mixed["cuda_graph_hit_rate"] == 0.1
    assert GRAPH_CLEAN_HIT_RATE > 0.5


def _fake_row(batch, requested, hits, fallbacks, seed=42):
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "eval", "repair"))
    from batch_intervention import graph_accounting

    row = {
        "max_batch_size_forced": batch, "seed": seed, "arrival": "burst",
        "rate_rps": 0.0, "cap_was_binding": True, "steps": 100,
        "realised_mean_batch": float(batch),
        "step_time_mean_ms": 10.0 + 2.0 * batch,
        "cuda_graph_fallback_reasons": {},
    }
    row.update(graph_accounting(requested, hits, fallbacks))
    return row


def test_build_analysis_excludes_mislabelled_graph_rows():
    """`build_analysis` must never fit the HardwareProfile against an arm that
    did not actually run graphs, and must say so out loud.

    Tested on CPU because the alternative is discovering a KeyError in the
    summary block after a multi-hour GPU sweep -- the exact way this repository
    already lost two result files.
    """
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "eval", "repair"))
    from batch_intervention import build_analysis

    # The first T4 smoke, exactly: two eager rows, two requested-graph rows
    # with zero replays.
    dead = build_analysis([
        _fake_row(1, False, 0, 2350), _fake_row(4, False, 0, 627),
        _fake_row(1, True, 0, 2350), _fake_row(4, True, 0, 627),
    ])
    assert dead["graph_arm_usable"] is False
    assert dead["n_clean_graph_rows"] == 0
    assert len(dead["invalid_graph_rows"]) == 2
    assert dead["hardware_profile_fit_cuda_graph"]["fitted"] is False
    assert dead["hardware_profile_fit_eager"]["fitted"] is True

    # A repaired sweep: the graph rows genuinely replayed.
    good = build_analysis([
        _fake_row(1, False, 0, 2350), _fake_row(4, False, 0, 627),
        _fake_row(1, True, 2350, 0), _fake_row(4, True, 620, 7),
    ])
    assert good["graph_arm_usable"] is True
    assert good["n_clean_graph_rows"] == 2
    assert good["invalid_graph_rows"] == []
    assert good["hardware_profile_fit_cuda_graph"]["fitted"] is True

    # One bad row poisons the sweep: partial credit is not on offer.
    partial = build_analysis([
        _fake_row(1, False, 0, 2350), _fake_row(4, False, 0, 627),
        _fake_row(1, True, 2350, 0), _fake_row(4, True, 0, 627),
    ])
    assert partial["graph_arm_usable"] is False
    assert partial["n_clean_graph_rows"] == 1


class _CpuStaticRunner(DecodeGraphRouter):
    """CPU stand-in that runs the SAME `StaticDecodeBatch` a captured graph
    would replay -- eagerly, with no CUDA graph anywhere.

    This is NOT a claim about CUDA graph execution and must never be read as
    one. Its purpose is to put the parts of the repair that are pure host-side
    logic -- scheduler routing, bucket resolution, block pre-allocation,
    metadata staging, the host-side seq_len commit -- under test on a machine
    with no GPU, so the T4 run is not the first time they execute. On the T4,
    `LiveDecodeGraphRunner` swaps `run_eager` for `graph.replay()` and
    everything around it is this same code (`DecodeGraphRouter`).
    """

    def __init__(self, model, pool, **kw):
        super().__init__(pool, **kw)
        self.model = model
        for b in self.batch_sizes:
            for nb in self.buckets:
                self.graphs[(b, nb)] = (StaticDecodeBatch(pool, b, nb),)

    def replay(self, input_ids, caches, routing_plan=None):
        # Accepts the plan the scheduler now threads through (B3). This runner
        # is driven by a plain LlamaModel, so the plan is always None -- but it
        # must resolve WITH it, so that the key checked here is the same key
        # `can_replay` authorised.
        key = self._resolve(len(caches), caches, routing_plan)
        assert key is not None, "replay() called without a routable key"
        (state,) = self.graphs[key]
        state.stage(input_ids, caches)
        out = state.run_eager(self.model)
        state.commit(caches)
        return out


def _serve(model, runner_factory, n_requests=4, max_batch=3, max_new=20):
    """Serve a fixed workload and return (tokens, hit/fallback counts)."""
    seen = {"hit": 0, "eager": 0}
    sched = ContinuousBatchScheduler(
        model, max_batch_size=max_batch, num_blocks=256, block_size=BLOCK_SIZE,
        chunk_size=256, enable_spec_decode=False, use_cuda_graphs=True,
        cuda_graph_observer=lambda hit: seen.__setitem__(
            "hit" if hit else "eager", seen["hit" if hit else "eager"] + 1),
    )
    if runner_factory is not None:
        sched._graph_runner = runner_factory(sched.pool)

    torch.manual_seed(99)
    prompts = [torch.randint(0, model.config.vocab_size, (1, 9 + 7 * i))
               for i in range(n_requests)]
    for i, p in enumerate(prompts):
        sched.add_request(f"s{i}", p, max_new_tokens=max_new, eos_token_id=None)

    tokens = []
    while sched.has_work():
        tokens.extend(sched.step())
    return tokens, seen


def test_scheduler_routes_through_the_static_path_and_preserves_output():
    """End-to-end on CPU: a scheduler with a router attached must actually take
    the static-shape branch on real decode steps, AND emit exactly the tokens
    the eager scheduler emits.

    This is the integration the first T4 smoke never performed even once --
    `_graph_runner` was never assigned, so `_decode_forward`'s non-eager branch
    was unreachable and every 'graph' step fell through to eager.
    """
    model = _tiny_model()

    eager_tokens, eager_seen = _serve(model, None)
    assert eager_seen["hit"] == 0 and eager_seen["eager"] > 0

    def factory(pool):
        return _CpuStaticRunner(model, pool, max_batch_size=3,
                                max_context_tokens=64, num_seq_buckets=3)

    static_tokens, static_seen = _serve(model, factory)

    assert static_seen["hit"] > 0, "the static-shape branch was never taken"
    assert static_tokens == eager_tokens, (
        "routing decode through the static-shape path changed the served output")


def test_router_falls_back_when_context_outgrows_the_largest_bucket():
    """A context past the largest bucket must be COUNTED as a fallback and run
    eager -- never squeezed into a smaller captured window, which would silently
    truncate attention and still be reported as a graph hit."""
    model = _tiny_model()

    def factory(pool):
        # 24 tokens of context ceiling against requests that reach ~50.
        return _CpuStaticRunner(model, pool, max_batch_size=3,
                                max_context_tokens=24, num_seq_buckets=1)

    tokens, seen = _serve(model, factory, max_new=24)
    eager_tokens, _ = _serve(model, None, max_new=24)

    assert seen["eager"] > 0
    assert tokens == eager_tokens
    runner = _CpuStaticRunner(model, ContinuousBatchScheduler(
        model, max_batch_size=1, num_blocks=32, block_size=BLOCK_SIZE).pool,
        max_batch_size=1, max_context_tokens=24, num_seq_buckets=1)
    assert runner.buckets == (2,)   # 24 tokens -> 2 blocks of 16


# ---------------------------------------------------------------------------
# CPU: the fallback-reason diagnostics.
#
# Two T4 smokes both reported `cuda_graph_hits == 0` with `fallbacks == decode
# steps`, and that signature is IDENTICAL whether the flag is off, the runner
# was never attached, or the runner is attached and refusing -- three failures
# with three different fixes. These tests pin that each is now distinguishable
# from the artifact alone, without a rerun.
# ---------------------------------------------------------------------------


def test_diagnostics_name_graph_runner_missing():
    """The signature of running code that predates the runner attachment --
    which is what actually produced both zero-hit smoke runs."""
    model = _tiny_model()
    tokens, seen = _serve(model, None)
    sched = ContinuousBatchScheduler(
        model, max_batch_size=2, num_blocks=64, block_size=BLOCK_SIZE,
        use_cuda_graphs=True)
    sched.add_request("r", torch.randint(0, 256, (1, 6)), max_new_tokens=6,
                      eos_token_id=None)
    while sched.has_work():
        sched.step()

    diag = sched.graph_diagnostics()
    assert diag["graph_runner_attached"] is False
    assert diag["graph_runner_class"] is None
    assert diag["runner_fallback_reasons"] is None
    assert diag["reasons"].get("graph_runner_missing", 0) > 0
    assert diag["reasons"].get("graph_hit", 0) == 0
    assert "runner_rejected" not in diag["reasons"], (
        "a missing runner must not be reported as a refusing runner")


def test_diagnostics_name_graphs_disabled():
    """use_cuda_graphs=False is a third, distinct signature."""
    model = _tiny_model()
    sched = ContinuousBatchScheduler(
        model, max_batch_size=2, num_blocks=64, block_size=BLOCK_SIZE,
        use_cuda_graphs=False)
    sched.add_request("r", torch.randint(0, 256, (1, 6)), max_new_tokens=6,
                      eos_token_id=None)
    while sched.has_work():
        sched.step()

    diag = sched.graph_diagnostics()
    assert diag["reasons"].get("graphs_disabled", 0) > 0
    assert diag["reasons"].get("graph_runner_missing", 0) == 0


def test_diagnostics_name_runner_rejected_with_sub_reason():
    """An ATTACHED runner that refuses must be distinguishable from an absent
    one, and must say WHY it refused."""
    model = _tiny_model()

    def factory(pool):
        # Context ceiling well below what the requests reach.
        return _CpuStaticRunner(model, pool, max_batch_size=3,
                                max_context_tokens=16, num_seq_buckets=1)

    tokens, seen = _serve(model, factory, max_new=24)
    eager_tokens, _ = _serve(model, None, max_new=24)
    assert tokens == eager_tokens

    # Re-run holding the scheduler so we can read its diagnostics.
    sched = ContinuousBatchScheduler(
        model, max_batch_size=3, num_blocks=256, block_size=BLOCK_SIZE,
        chunk_size=256, use_cuda_graphs=True)
    sched._graph_runner = factory(sched.pool)
    torch.manual_seed(99)
    for i in range(3):
        sched.add_request(f"z{i}", torch.randint(0, 256, (1, 9 + 7 * i)),
                          max_new_tokens=24, eos_token_id=None)
    while sched.has_work():
        sched.step()

    diag = sched.graph_diagnostics()
    assert diag["graph_runner_attached"] is True
    assert "_CpuStaticRunner" in diag["graph_runner_class"]
    assert diag["reasons"].get("runner_rejected", 0) > 0
    assert diag["reasons"].get("graph_runner_missing", 0) == 0
    assert diag["runner_fallback_reasons"].get(
        "context_exceeds_largest_bucket", 0) > 0, (
        f"expected a context-bucket refusal, got "
        f"{diag['runner_fallback_reasons']}")


def test_reason_counts_cover_every_decode_forward():
    """The counters must be exhaustive: one entry per batched-decode forward,
    so `sum(reasons.values())` equals what the observer saw. A reason bucket
    that silently drops cases is worse than no diagnostic at all."""
    model = _tiny_model()
    seen = {"n": 0}
    sched = ContinuousBatchScheduler(
        model, max_batch_size=3, num_blocks=256, block_size=BLOCK_SIZE,
        chunk_size=256, use_cuda_graphs=True,
        cuda_graph_observer=lambda hit: seen.__setitem__("n", seen["n"] + 1),
    )
    sched._graph_runner = _CpuStaticRunner(
        model, sched.pool, max_batch_size=3, max_context_tokens=64,
        num_seq_buckets=3)
    torch.manual_seed(99)
    for i in range(3):
        sched.add_request(f"w{i}", torch.randint(0, 256, (1, 9 + 7 * i)),
                          max_new_tokens=20, eos_token_id=None)
    while sched.has_work():
        sched.step()

    diag = sched.graph_diagnostics()
    assert sum(diag["reasons"].values()) == seen["n"] > 0
    assert diag["reasons"].get("graph_hit", 0) > 0


# ---------------------------------------------------------------------------
# CPU: provenance -- the guard against the failure that actually occurred.
# ---------------------------------------------------------------------------


def test_provenance_fingerprints_the_graph_implementation():
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "eval", "repair"))
    from batch_intervention import graph_repair_provenance, require_graph_repair

    prov = graph_repair_provenance()
    assert prov["live_graph_present"] is True
    assert prov["scheduler_has_reason_counts"] is True
    assert prov["live_graph_sha256"] and len(prov["live_graph_sha256"]) == 16
    # In a repaired tree the guard passes and returns the same fingerprint.
    assert require_graph_repair()["live_graph_sha256"] == prov["live_graph_sha256"]


def test_require_graph_repair_refuses_a_stale_checkout(monkeypatch):
    """THE regression test for the second failure.

    The GPU box ran a bundle built before the repair: no `live_graph.py`, a
    pre-repair scheduler, no runner attachment. It produced a full table of
    eager numbers under a "graphs" label and nothing flagged it. Simulate that
    tree and require a hard refusal.
    """
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "eval", "repair"))
    import batch_intervention as bi

    monkeypatch.setattr(bi, "graph_repair_provenance", lambda: {
        "live_graph_present": False, "live_graph_sha256": None,
        "live_graph_bytes": None, "scheduler_has_reason_counts": False,
        "git_describe": "deadbee"})
    with pytest.raises(SystemExit) as exc:
        bi.require_graph_repair()
    msg = str(exc.value)
    assert "predates the graph repair" in msg
    assert "src/engine/live_graph.py is absent" in msg
    assert "--no-cuda-graph-arm" in msg, (
        "the refusal must offer the eager-only escape hatch")


# ---------------------------------------------------------------------------
# CUDA: the parts only a GPU can honestly verify.
# ---------------------------------------------------------------------------


@cuda_only
def test_capture_grid_succeeds():
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    pool, _ = _prefilled_pool(model, [], device="cuda")
    runner = LiveDecodeGraphRunner(model, pool, max_batch_size=3,
                                   max_context_tokens=128, num_seq_buckets=2)
    report = runner.capture_all()

    assert report["capture_failures"] == []
    assert report["graphs_captured"] == len(runner.batch_sizes) * len(runner.buckets)
    assert runner.batch_sizes == (1, 2, 3)


@cuda_only
def test_capture_refuses_a_live_pool():
    """Capture writes scratch K/V into block 0; on a pool with live requests
    that is silent corruption, so it must be refused outright."""
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    pool, _ = _prefilled_pool(model, [20], device="cuda")
    with pytest.raises(RuntimeError, match="before any request is admitted"):
        LiveDecodeGraphRunner(model, pool, max_batch_size=2, max_context_tokens=128)


@cuda_only
@pytest.mark.parametrize("prompt_lens", [[20], [20, 33]])
def test_replay_matches_eager(prompt_lens):
    """Graph replay must reproduce eager decode for the same KV state."""
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    B = len(prompt_lens)
    pool, _ = _prefilled_pool(model, [], device="cuda")
    runner = LiveDecodeGraphRunner(model, pool, max_batch_size=B,
                                   max_context_tokens=160, num_seq_buckets=2)
    runner.capture_all()
    assert runner.capture_failures == []

    caches = []
    cfg = model.config
    for i, L in enumerate(prompt_lens):
        rid = f"live{i}"
        blocks = (L + 32 + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool.admit_request(request_id=rid, prefill_blocks_needed=blocks,
                           total_blocks_needed=blocks)
        c = PagedRequestCache(pool, rid, num_layers=cfg.num_hidden_layers)
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (1, L), device="cuda"), kv_cache=c)
        caches.append(c)

    snap = _snapshot(pool, caches)
    ids = torch.randint(0, cfg.vocab_size, (B, 1), device="cuda")

    with torch.no_grad():
        eager = model(ids, kv_cache=caches)

    _restore(pool, caches, snap)
    assert runner.can_replay(B, caches) is True
    graphed = runner.replay(ids, caches)

    assert torch.equal(eager, graphed), "graph replay diverged from eager decode"


@cuda_only
def test_live_scheduler_records_graph_hits_as_context_grows():
    """THE test this whole repair exists for: a scheduler serving real requests
    must actually REPLAY graphs, and must keep replaying them step after step
    while every request's context grows.

    The old cuda_graph.CUDAGraphRunner could not do this by construction (its
    graphs froze seq_len and bound themselves to their own cache objects), which
    is why hits stayed at zero.
    """
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    seen = {"hit": 0, "eager": 0}
    sched = ContinuousBatchScheduler(
        model, max_batch_size=2, num_blocks=128, block_size=BLOCK_SIZE,
        chunk_size=256, enable_spec_decode=False, use_cuda_graphs=True,
        cuda_graph_observer=lambda hit: seen.__setitem__(
            "hit" if hit else "eager", seen["hit" if hit else "eager"] + 1),
    )
    runner = LiveDecodeGraphRunner(model, sched.pool, max_batch_size=2,
                                   max_context_tokens=96, num_seq_buckets=3)
    runner.capture_all()
    assert runner.capture_failures == []
    sched._graph_runner = runner

    for i in range(2):
        sched.add_request(f"q{i}",
                          torch.randint(0, model.config.vocab_size, (1, 12)),
                          max_new_tokens=24, eos_token_id=None)
    emitted = 0
    while sched.has_work():
        emitted += len(sched.step())

    assert emitted == 48
    assert seen["hit"] > 0, (
        f"no graph replays in a live run: fallbacks={runner.fallback_reasons}")
    # A context that grew from 12 to 36 tokens crossed at least one bucket, so
    # the hits cannot all come from one frozen shape.
    assert seen["hit"] >= 20, (
        f"hits={seen['hit']} eager={seen['eager']} "
        f"reasons={runner.fallback_reasons}")


@cuda_only
def test_capture_failure_raises_instead_of_degrading_to_eager():
    """A failed capture must be a hard error, not a silent eager downgrade.

    The earlier version recorded the exception and carried on, so a grid in
    which EVERY capture failed produced a graph arm with zero hits -- the same
    class of silent mislabelling the module exists to prevent, one level down.
    """
    from src.engine.live_graph import GraphCaptureError, LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    pool, _ = _prefilled_pool(model, [], device="cuda")
    runner = LiveDecodeGraphRunner(model, pool, max_batch_size=2,
                                   max_context_tokens=64, num_seq_buckets=1)

    def boom(*a, **k):
        raise RuntimeError("simulated capture failure")

    runner._capture = boom
    with pytest.raises(GraphCaptureError, match="Refusing to continue"):
        runner.capture_all()

    # The opt-out still records the failure and leaves the grid empty, which
    # itself raises rather than returning an empty runner.
    runner2 = LiveDecodeGraphRunner(model, pool, max_batch_size=2,
                                    max_context_tokens=64, num_seq_buckets=1)
    runner2._capture = boom
    with pytest.raises(GraphCaptureError, match="no CUDA graphs were captured"):
        runner2.capture_all(require_all=False)
    assert runner2.capture_failures, "the failure must still be recorded"


@cuda_only
def test_self_test_proves_replay_before_measurement():
    """`self_test` must demonstrate a real replay against live request caches
    and agree with eager -- the proof-of-life that was missing when the smoke
    came back at zero hits with no intermediate signal."""
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    pool, _ = _prefilled_pool(model, [], device="cuda")
    runner = LiveDecodeGraphRunner(model, pool, max_batch_size=2,
                                   max_context_tokens=96, num_seq_buckets=2)
    runner.capture_all()

    free_before = pool.num_free_blocks()
    report = runner.self_test()
    assert report["checked_batch_sizes"]
    assert report["max_abs_logit_delta"] == 0.0
    # It must leave the pool exactly as it found it -- the measured run starts
    # immediately afterwards.
    assert pool.num_free_blocks() == free_before
    assert not pool._blocks


@cuda_only
def test_self_test_raises_when_the_runner_cannot_replay():
    """A runner that captured nothing usable must fail the self-test loudly,
    naming its own refusal reason, rather than passing and producing an
    all-eager 'graph' arm."""
    from src.engine.live_graph import GraphSelfTestError, LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    pool, _ = _prefilled_pool(model, [], device="cuda")
    runner = LiveDecodeGraphRunner(model, pool, max_batch_size=2,
                                   max_context_tokens=96, num_seq_buckets=2)
    runner.capture_all()
    runner.graphs.clear()          # simulate "capture never really happened"
    with pytest.raises(GraphSelfTestError, match="can_replay\\(\\) refused"):
        runner.self_test()


@cuda_only
def test_uncaptured_batch_size_falls_back_and_is_not_a_hit():
    """A decode batch no graph covers must route to eager and be COUNTED as a
    fallback -- never rounded up into a captured shape and called a hit."""
    from src.engine.live_graph import LiveDecodeGraphRunner

    model = _tiny_model("cuda")
    pool, _ = _prefilled_pool(model, [], device="cuda")
    runner = LiveDecodeGraphRunner(model, pool, max_batch_size=1,
                                   max_context_tokens=64, num_seq_buckets=1)
    runner.capture_all()

    assert runner.can_replay(1, None) is True
    assert runner.can_replay(2, []) is False
    assert runner.fallback_reasons.get("batch_size_not_captured", 0) >= 1
    with pytest.raises(RuntimeError, match="no captured graph can"):
        runner.replay(torch.zeros((2, 1), dtype=torch.long, device="cuda"), [])
