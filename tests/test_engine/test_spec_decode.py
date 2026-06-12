"""
True draft/target speculative-decoding tests (Leviathan et al. 2023).

All CPU-compatible: the acceptance-rejection math is tested with hand-built
probability tensors, and the end-to-end paths use a small random-weight
LlamaModel plus RandomDraftModel -- no TinyLlama download, no GPU.

What we pin:

  1. The acceptance-rejection rule is DISTRIBUTION-CORRECT: the first emitted
     token is distributed exactly as the target's p_1 (Leviathan Theorem 1),
     regardless of what the draft proposed. This is the whole correctness
     guarantee of the algorithm.
  2. GUARANTEED PROGRESS: decode_step always emits at least one token, even
     when every draft token is rejected.
  3. acceptance_rate tracking: when draft == target the accept ratio is
     identically 1, so every draft token is accepted and the rate is 1.0.
  4. RandomDraftModel drives a real SpeculativeDecoder end to end.
  5. SelfSpecDraftModel adapts the early-exit draft to the DraftModel contract.
"""
from __future__ import annotations

import torch

from src.engine.draft_model import RandomDraftModel
from src.engine.model import LlamaConfig, LlamaModel
from src.engine.spec_decode import (
    DraftModel,
    FullModelTarget,
    SelfSpecDraftModel,
    SpeculativeDecoder,
    TargetModel,
    TinyDraftModel,
    speculative_sample,
)


def _tiny_model(num_layers: int = 2, vocab: int = 64) -> LlamaModel:
    """A small random-weight LlamaModel. head_dim = 32/4 = 8."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


# ---------------------------------------------------------------------------
# 1. Acceptance-rejection math: emitted distribution == target p_1.
# ---------------------------------------------------------------------------


def test_acceptance_rejection_matches_target_distribution():
    """Over many trials, the FIRST emitted token must be distributed as p_1.

    This is the Leviathan guarantee: whatever the (arbitrary, even adversarial)
    draft proposes, the accept/resample correction makes the marginal of the
    emitted token equal the target's distribution. We use a single drafted slot
    (K=1) so the first emission is governed purely by p_1 vs q_1.
    """
    V = 4
    # An arbitrary, peaked target distribution and a DIFFERENT draft.
    p = torch.tensor([[0.5, 0.3, 0.15, 0.05],   # p_1 (the slot we check)
                      [0.25, 0.25, 0.25, 0.25]])  # p_2 (bonus slot, K+1)
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4]])      # q_1 (deliberately mismatched)

    gen = torch.Generator().manual_seed(0)
    counts = torch.zeros(V)
    trials = 40000
    for _ in range(trials):
        # Draft token sampled from q_1 each trial (as a real draft would).
        x1 = int(torch.multinomial(q[0], 1, generator=gen))
        draft_tokens = torch.tensor([x1])
        emitted, _ = speculative_sample(draft_tokens, q, p, generator=gen)
        counts[emitted[0]] += 1

    empirical = counts / trials
    # Monte-Carlo tolerance: ~40k trials over 4 bins -> a couple % is plenty.
    assert torch.allclose(empirical, p[0], atol=0.02), (
        f"first emitted token distribution {empirical.tolist()} != target "
        f"p_1 {p[0].tolist()}"
    )


def test_speculative_sample_all_accepted_appends_bonus():
    """If every accept check passes, output length is K+1 and n_accepted == K."""
    V = 3
    # q == p and we force acceptance by making the ratio exactly 1 everywhere:
    # with q == p, accept_prob = min(1, p/q) = 1, and rand() < 1 always.
    p = torch.tensor([[0.2, 0.3, 0.5],
                      [0.2, 0.3, 0.5],
                      [0.1, 0.1, 0.8]])   # rows 0,1 = p_1,p_2 ; row 2 = bonus
    q = torch.tensor([[0.2, 0.3, 0.5],
                      [0.2, 0.3, 0.5]])
    draft_tokens = torch.tensor([2, 2])
    gen = torch.Generator().manual_seed(7)
    emitted, n_accepted = speculative_sample(draft_tokens, q, p, generator=gen)
    assert n_accepted == 2
    assert len(emitted) == 3                     # K accepted + 1 bonus
    assert emitted[:2] == [2, 2]                 # the two accepted drafts


# ---------------------------------------------------------------------------
# 2. Guaranteed progress.
# ---------------------------------------------------------------------------


def test_guaranteed_progress_even_on_total_rejection():
    """Even when the draft is maximally wrong, at least one token is emitted."""
    V = 5
    # Target puts ALL mass on token 0; draft proposes token 4 with prob 1.
    # The accept ratio for token 4 is p(4)/q(4) = 0/1 = 0 -> always rejected,
    # so we must fall through to the residual resample and still emit a token.
    p = torch.zeros(2, V)
    p[:, 0] = 1.0
    q = torch.zeros(1, V)
    q[0, 4] = 1.0
    draft_tokens = torch.tensor([4])

    gen = torch.Generator().manual_seed(1)
    for _ in range(100):
        emitted, n_accepted = speculative_sample(draft_tokens, q, p, generator=gen)
        assert len(emitted) >= 1, "decode must always make progress"
        assert n_accepted == 0, "the deterministically-wrong draft cannot be accepted"
        assert emitted[0] == 0, "residual resample must land on the target's only mass"


# ---------------------------------------------------------------------------
# 3. acceptance_rate tracking (draft == target -> rate 1.0).
# ---------------------------------------------------------------------------


def test_acceptance_rate_is_one_when_draft_equals_target():
    """Draft and target are the SAME model at the same temperature, so the draft
    distribution equals the target distribution at every slot and the accept
    ratio is identically 1 -- every draft token is accepted, every step."""
    model = _tiny_model()
    draft = TinyDraftModel(model)
    target = FullModelTarget(model)
    gen = torch.Generator().manual_seed(3)
    dec = SpeculativeDecoder(draft, target, k=4, generator=gen)

    ids = torch.randint(0, model.config.vocab_size, (1, 5), generator=torch.Generator().manual_seed(9))
    for _ in range(5):
        emitted = dec.decode_step(ids)
        assert dec.acceptance_rate == 1.0
        assert len(emitted) == dec.k + 1          # all K accepted + bonus
        # Feed the emitted tokens back so the next round has a longer context.
        ids = torch.cat([ids, torch.tensor([emitted], dtype=torch.long)], dim=1)

    assert dec.mean_acceptance_rate == 1.0


# ---------------------------------------------------------------------------
# 4. RandomDraftModel end-to-end.
# ---------------------------------------------------------------------------


def test_random_draft_model_end_to_end():
    """A full SpeculativeDecoder round with a random draft + real tiny target.

    Random proposals against a peaked target accept rarely, but the decoder must
    still run cleanly, emit valid in-vocab tokens, and report a sane rate."""
    model = _tiny_model()
    V = model.config.vocab_size
    draft = RandomDraftModel(vocab_size=V, seed=42)
    target = FullModelTarget(model)
    gen = torch.Generator().manual_seed(5)
    dec = SpeculativeDecoder(draft, target, k=4, generator=gen)

    ids = torch.randint(0, V, (1, 6), generator=torch.Generator().manual_seed(2))
    total = 0
    for _ in range(8):
        emitted = dec.decode_step(ids)
        assert 1 <= len(emitted) <= dec.k + 1
        assert all(0 <= t < V for t in emitted), "emitted token out of vocab range"
        assert 0.0 <= dec.acceptance_rate <= 1.0
        total += len(emitted)
        ids = torch.cat([ids, torch.tensor([emitted], dtype=torch.long)], dim=1)

    assert total >= 8, "should emit at least one token per step (guaranteed progress)"
    assert 0.0 <= dec.mean_acceptance_rate <= 1.0


def test_random_draft_satisfies_protocol():
    """RandomDraftModel / FullModelTarget structurally satisfy the protocols."""
    model = _tiny_model()
    assert isinstance(RandomDraftModel(vocab_size=model.config.vocab_size), DraftModel)
    assert isinstance(FullModelTarget(model), TargetModel)


# ---------------------------------------------------------------------------
# 5. SelfSpecDraftModel adapter.
# ---------------------------------------------------------------------------


def test_self_spec_draft_model_wraps_early_exit():
    """SelfSpecDraftModel.propose returns (K,) tokens and (K, V) probs that are
    valid categorical distributions, using the early-exit path of the target."""
    model = _tiny_model(num_layers=4)
    V = model.config.vocab_size
    # Exit after 2 of the 4 layers -- a genuinely shallower draft forward.
    draft = SelfSpecDraftModel(model, n_layers=2)
    ids = torch.randint(0, V, (1, 5), generator=torch.Generator().manual_seed(8))

    k = 3
    token_ids, draft_probs = draft.propose(ids, k)
    assert token_ids.shape == (k,)
    assert draft_probs.shape == (k, V)
    assert all(0 <= int(t) < V for t in token_ids)
    # Each row is a proper distribution.
    row_sums = draft_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(k), atol=1e-5)

    # And it drops straight into a SpeculativeDecoder against the full target.
    target = FullModelTarget(model)
    dec = SpeculativeDecoder(draft, target, k=k, generator=torch.Generator().manual_seed(0))
    emitted = dec.decode_step(ids)
    assert 1 <= len(emitted) <= k + 1
