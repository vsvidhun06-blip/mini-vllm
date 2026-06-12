"""
CUDA graph capture/replay tests.

A captured graph records the whole decode forward as a static graph and replays
it with one launch instead of hundreds. These tests pin the properties that
matter for correctness (speed is the benchmark's job):

  1. Capture succeeds for the pooled batch sizes (1, 2, 4).
  2. Replay produces output IDENTICAL to eager execution of the same decode
     step (the graph must not change the math).
  3. A batch size that was never captured (or isn't in the pool) falls back --
     `can_replay` is False, and the scheduler routes such steps to eager.

Everything here needs a real GPU (graphs are a CUDA feature and the decode
forward goes through the Triton kernels), so the whole module skips cleanly on
CPU-only hosts -- exactly like test_flash_attention.py.
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs are a CUDA-only feature"
)

SEQ_LEN = 48  # multiple of block_size(16); the +1 decode token stays in-block


def _tiny_model():
    """Small random-weight LlamaModel on CUDA. head_dim=16 is a power of 2 so
    the from-scratch FA2 kernel (used on the CUDA decode path) is happy."""
    from src.engine.model import LlamaConfig, LlamaModel

    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,       # head_dim = 128 / 8 = 16
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaModel(config).eval().to("cuda")
    return model


def _input_ids(batch_size: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(7)
    return torch.randint(0, 256, (batch_size, 1), device="cuda", generator=g)


@cuda_only
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_capture_succeeds(batch_size):
    from src.engine.cuda_graph import CUDAGraphRunner

    model = _tiny_model()
    runner = CUDAGraphRunner(model)
    cg = runner.capture(model, batch_size, SEQ_LEN)

    assert batch_size in runner.graphs
    assert runner.can_replay(batch_size)
    # The graph is bound to `batch_size` caches at `SEQ_LEN`.
    assert len(cg.caches) == batch_size
    assert cg.seq_len == SEQ_LEN
    # Static output shape is (B, 1, vocab).
    assert tuple(cg.static_logits.shape) == (batch_size, 1, model.config.vocab_size)


@cuda_only
@pytest.mark.parametrize("batch_size", [1, 2])
def test_replay_matches_eager(batch_size):
    """Graph replay must reproduce eager decode bit-for-bit for the same KV
    state and input -- both run the identical kernels, so they should agree."""
    from src.engine.cuda_graph import CUDAGraphRunner

    model = _tiny_model()
    runner = CUDAGraphRunner(model)
    runner.capture(model, batch_size, SEQ_LEN)
    caches = runner.caches_for(batch_size)
    input_ids = _input_ids(batch_size)

    # Eager: re-seed the bound caches to the capture-time length, then run the
    # model exactly as the scheduler's eager decode path would.
    runner.reset(batch_size)
    with torch.no_grad():
        eager = model(input_ids, kv_cache=caches)

    # Graph: re-seed and replay the SAME (input, KV state).
    runner.reset(batch_size)
    graphed = runner.replay(input_ids, caches)

    assert eager.shape == graphed.shape == (batch_size, 1, model.config.vocab_size)
    assert torch.equal(eager, graphed), (
        "graph replay diverged from eager decode for an identical step"
    )
    # The argmax token (what the scheduler actually emits) must agree too.
    assert torch.equal(eager[:, -1].argmax(-1), graphed[:, -1].argmax(-1))


@cuda_only
def test_fallback_for_batch_size_not_in_pool():
    """A batch size we never captured (here 3, also not a pool size) cannot be
    replayed -- the caller must fall back to eager."""
    from src.engine.cuda_graph import CUDAGraphRunner

    model = _tiny_model()
    runner = CUDAGraphRunner(model)
    runner.capture(model, 2, SEQ_LEN)            # capture only batch size 2

    assert runner.can_replay(2) is True
    assert runner.can_replay(3) is False         # never captured
    assert runner.can_replay(8) is False         # pool size, but not captured
    with pytest.raises(KeyError):
        runner.replay(_input_ids(3), None)

    # And capturing a non-pool size is rejected outright.
    with pytest.raises(ValueError):
        runner.capture(model, 3, SEQ_LEN)


@cuda_only
def test_replay_rejects_foreign_caches():
    """Replay must refuse caches it wasn't captured against -- the graph baked
    in its bound caches' pool addresses."""
    from src.engine.cuda_graph import CUDAGraphRunner

    model = _tiny_model()
    runner = CUDAGraphRunner(model)
    runner.capture(model, 2, SEQ_LEN)
    bound = runner.caches_for(2)
    foreign = [bound[0]]  # wrong length / not the bound set
    with pytest.raises(ValueError):
        runner.replay(_input_ids(2), foreign)


def test_scheduler_use_cuda_graphs_flag_is_safe_on_cpu():
    """The use_cuda_graphs flag must not change behaviour when there's no GPU /
    no captured graph: decode falls back to eager and still produces tokens.

    Runs on CPU (no skip) -- this is the guard that the new flag/route doesn't
    regress the existing engine.
    """
    from src.engine.model import LlamaConfig, LlamaModel
    from src.engine.scheduler import ContinuousBatchScheduler

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaModel(config).eval()
    # ONE fixed prompt shared by both runs. Generating it inside run() would
    # advance the RNG between the two calls, so run(True) and run(False) would
    # see DIFFERENT prompts and their greedy outputs would legitimately differ --
    # the assertion would fail for a reason unrelated to the cuda-graphs flag.
    prompt = torch.randint(0, 256, (1, 6))

    def run(use_cuda_graphs):
        sched = ContinuousBatchScheduler(
            model, max_batch_size=2, num_blocks=64, use_cuda_graphs=use_cuda_graphs,
        )
        sched.add_request("r", prompt.clone(), max_new_tokens=8, eos_token_id=None)
        out = []
        while sched.has_work():
            for _rid, tok in sched.step():
                out.append(tok)
        return out

    on = run(True)
    off = run(False)
    assert len(on) == 8
    assert on == off, "the use_cuda_graphs flag changed CPU decode output"
