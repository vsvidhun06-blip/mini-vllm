"""
Multi-LoRA adapter serving tests.

LoRA adds a low-rank delta to a linear layer: y = Wx + (alpha/r) * B(A(x)). These
tests pin the properties that make multi-adapter serving correct:

  1. Numerics -- LoRALinear's output equals the manual base + scaled B@A@x delta.
  2. Switching -- different adapters give different outputs; base (no adapter) is
     distinct from any adapter and is recovered exactly when the adapter clears.
  3. Mixed-adapter batch -- one batched forward with per-row routing produces, for
     each row, exactly what a single-adapter forward of that row produces. This
     is the property that lets one base model serve several fine-tunes at once.
  4. Zero overhead -- with no adapter active, LoRALinear is byte-identical to the
     wrapped nn.Linear, and LoRALlamaModel is byte-identical to the base model.
  5. Registry -- the LoRAManager caps residency and evicts LRU.

Everything runs on a tiny random-weight model on CPU with synthetic adapters --
no GPU, no HF download.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.engine.lora import LoRALinear, LoRAManager
from src.engine.lora_model import LoRALlamaModel, layer_name, random_adapter_weights
from src.engine.model import LlamaConfig, LlamaModel


def _tiny_model() -> LlamaModel:
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,       # head_dim = 16
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


# ---------------------------------------------------------------------------
# 1. LoRALinear numerics match the manual base + delta.
# ---------------------------------------------------------------------------


def test_lora_linear_matches_manual_delta():
    torch.manual_seed(0)
    in_f, out_f, rank = 32, 48, 4
    base = nn.Linear(in_f, out_f, bias=False)
    mgr = LoRAManager()
    lname = "layers.0.attn.q_proj"
    lora = LoRALinear(base, mgr, lname)

    A = torch.randn(rank, in_f) * 0.05      # (r, in)
    B = torch.randn(out_f, rank) * 0.05     # (out, r)
    alpha = 8.0
    mgr.load_adapter("adp", rank=rank, alpha=alpha, weights_dict={lname: (A, B)})

    x = torch.randn(3, 5, in_f)
    scale = alpha / rank
    expected = base(x) + scale * (x @ A.transpose(0, 1)) @ B.transpose(0, 1)

    got = lora(x, adapter_id="adp")
    assert torch.allclose(got, expected, atol=1e-5), "LoRA delta does not match manual computation"

    # The same result via externally-set active state (the model's path).
    lora.set_active("adp")
    assert torch.allclose(lora(x), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Switching adapters changes the output; clearing recovers base.
# ---------------------------------------------------------------------------


def test_adapter_switching_changes_output():
    torch.manual_seed(1)
    in_f, out_f, rank = 32, 32, 4
    base = nn.Linear(in_f, out_f, bias=False)
    mgr = LoRAManager()
    lname = "layers.0.attn.q_proj"
    lora = LoRALinear(base, mgr, lname)

    for aid, seed in [("a", 10), ("b", 20)]:
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(rank, in_f, generator=g) * 0.1
        B = torch.randn(out_f, rank, generator=g) * 0.1
        mgr.load_adapter(aid, rank=rank, alpha=8.0, weights_dict={lname: (A, B)})

    x = torch.randn(2, 4, in_f)
    out_base = lora(x)                 # no adapter
    out_a = lora(x, adapter_id="a")
    out_b = lora(x, adapter_id="b")

    assert not torch.allclose(out_a, out_b), "different adapters gave identical output"
    assert not torch.allclose(out_a, out_base), "adapter a did not change the base output"
    assert not torch.allclose(out_b, out_base), "adapter b did not change the base output"


# ---------------------------------------------------------------------------
# 3. Mixed-adapter batch: per-row routing == per-row single-adapter forward.
# ---------------------------------------------------------------------------


def test_mixed_adapter_batch_routes_per_row():
    model = _tiny_model()
    mgr = LoRAManager()
    lora_model = LoRALlamaModel(model, mgr)

    # Two distinct adapters spanning every wrapped projection.
    for aid, seed in [("a", 1), ("b", 2)]:
        mgr.load_adapter(aid, rank=8, alpha=16.0,
                         weights_dict=random_adapter_weights(lora_model, rank=8, seed=seed))

    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (2, 6), generator=g)   # 2 requests, len-6 prompts

    # One batched forward, row 0 under "a", row 1 under "b".
    mixed = lora_model.forward(ids, adapter_ids=["a", "b"])         # (2, 6, V)

    # Reference: each row run on its own under its own adapter.
    ref_a = lora_model.forward(ids[0:1], adapter_ids="a")          # (1, 6, V)
    ref_b = lora_model.forward(ids[1:2], adapter_ids="b")

    assert torch.allclose(mixed[0], ref_a[0], atol=1e-5), "row 0 not routed to adapter a"
    assert torch.allclose(mixed[1], ref_b[0], atol=1e-5), "row 1 not routed to adapter b"

    # And mixing actually mattered: swapping the routing changes the rows.
    swapped = lora_model.forward(ids, adapter_ids=["b", "a"])
    assert not torch.allclose(mixed[0], swapped[0]), "routing order had no effect"


def test_mixed_batch_with_one_base_row():
    """A None entry in the per-row plan means that row uses the base path."""
    model = _tiny_model()
    mgr = LoRAManager()
    lora_model = LoRALlamaModel(model, mgr)
    mgr.load_adapter("a", rank=8, alpha=16.0,
                     weights_dict=random_adapter_weights(lora_model, rank=8, seed=3))

    g = torch.Generator().manual_seed(9)
    ids = torch.randint(0, 256, (2, 5), generator=g)

    mixed = lora_model.forward(ids, adapter_ids=["a", None])
    ref_a = lora_model.forward(ids[0:1], adapter_ids="a")
    ref_base = lora_model.forward(ids[1:2], adapter_ids=None)

    assert torch.allclose(mixed[0], ref_a[0], atol=1e-5)
    assert torch.allclose(mixed[1], ref_base[0], atol=1e-5)


# ---------------------------------------------------------------------------
# 3b. Routing-state contract: omitted vs. explicit None.
#
# Routing state on a LoRALlamaModel is STICKY -- the scheduler and generate()
# both set a plan once and then run several forwards under it. So forward() has
# to tell apart two different caller intents, and the tests below pin both:
#
#   forward(ids)                    -- argument omitted  -> keep the current plan
#   forward(ids, adapter_ids=None)  -- explicitly None    -> route to base
#
# Collapsing those two (a plain `None` default) silently serves the PREVIOUS
# adapter to a request that asked for the base model -- a cross-tenant
# correctness bug, not just an API wart.
# ---------------------------------------------------------------------------


def _model_with_two_adapters():
    """A wrapped tiny model with adapters "a" and "b" loaded, plus a pristine
    unwrapped twin of the same weights to compare base numerics against."""
    lora_model = LoRALlamaModel(_tiny_model(), LoRAManager())
    for aid, seed in [("a", 11), ("b", 22)]:
        lora_model.manager.load_adapter(
            aid, rank=8, alpha=16.0,
            weights_dict=random_adapter_weights(lora_model, rank=8, seed=seed),
        )
    return lora_model, _tiny_model()      # _tiny_model() is seeded -> same weights


def test_explicit_none_routes_to_base_after_single_adapter():
    """C: explicit adapter_ids=None must not inherit a previously-set adapter."""
    lora_model, base_model = _model_with_two_adapters()
    g = torch.Generator().manual_seed(31)
    ids = torch.randint(0, 256, (2, 6), generator=g)

    with_a = lora_model.forward(ids, adapter_ids="a")
    assert not torch.allclose(with_a, base_model(ids), atol=1e-5),         "adapter a had no effect -- test cannot detect stale routing"

    # Same call, now explicitly asking for the base path.
    got_base = lora_model.forward(ids, adapter_ids=None)
    assert torch.equal(got_base, base_model(ids)),         "explicit adapter_ids=None inherited the previously-active adapter"


def test_explicit_none_routes_to_base_after_per_row_plan():
    """C: the same, but clearing a per-row (mixed-batch) plan rather than a str."""
    lora_model, base_model = _model_with_two_adapters()
    g = torch.Generator().manual_seed(32)
    ids = torch.randint(0, 256, (2, 6), generator=g)

    lora_model.forward(ids, adapter_ids=["a", "b"])
    got_base = lora_model.forward(ids, adapter_ids=None)
    assert torch.equal(got_base, base_model(ids)),         "explicit adapter_ids=None inherited the previously-active per-row plan"


def test_omitted_adapter_ids_preserves_routing_state():
    """The sticky half of the contract: no argument => keep the current plan.

    This is what the scheduler (set_batch_adapters then forward) and generate()
    (set_adapter then a prefill + N decode forwards) depend on.
    """
    lora_model, base_model = _model_with_two_adapters()
    g = torch.Generator().manual_seed(33)
    ids = torch.randint(0, 256, (2, 6), generator=g)

    ref_a = lora_model.forward(ids, adapter_ids="a")

    # Via set_adapter, then a forward with the argument omitted.
    lora_model.set_adapter("a")
    assert torch.equal(lora_model.forward(ids), ref_a),         "omitted adapter_ids dropped the state set by set_adapter"

    # Via set_batch_adapters, then a forward with the argument omitted.
    lora_model.set_batch_adapters(["a", "a"])
    assert torch.allclose(lora_model.forward(ids), ref_a, atol=1e-5),         "omitted adapter_ids dropped the state set by set_batch_adapters"

    # And a plain forward() straight after another forward() keeps that plan --
    # the prefill -> decode inheritance the KV-cache loop relies on.
    assert torch.equal(lora_model.forward(ids), ref_a)


def test_all_none_per_row_plan_is_the_base_path():
    """D (batched form): a per-row plan of all Nones is exactly the base model.

    The scheduler builds its plan as [r.adapter_id for r in reqs], so an
    all-base batch arrives here as [None, None] -- it must cost nothing and
    change nothing.
    """
    lora_model, base_model = _model_with_two_adapters()
    g = torch.Generator().manual_seed(34)
    ids = torch.randint(0, 256, (2, 6), generator=g)

    lora_model.forward(ids, adapter_ids="a")                 # dirty the state
    got = lora_model.forward(ids, adapter_ids=[None, None])
    assert torch.equal(got, base_model(ids)),         "an all-None per-row plan did not reduce to the base path"


def test_clear_adapters_restores_base_exactly():
    """C: the explicit clear_adapters()/set_batch_adapters(None) escape hatch."""
    lora_model, base_model = _model_with_two_adapters()
    g = torch.Generator().manual_seed(35)
    ids = torch.randint(0, 256, (1, 6), generator=g)

    lora_model.set_adapter("b")
    lora_model.clear_adapters()
    assert torch.equal(lora_model.forward(ids), base_model(ids)),         "clear_adapters() left stale routing behind"


def test_single_adapter_forward_is_unaffected_by_prior_state():
    """E: adapter_ids="a" always means adapter a, whatever ran before it."""
    lora_model, _ = _model_with_two_adapters()
    g = torch.Generator().manual_seed(36)
    ids = torch.randint(0, 256, (2, 6), generator=g)

    from_clean = lora_model.forward(ids, adapter_ids="a")
    lora_model.forward(ids, adapter_ids="b")
    lora_model.forward(ids, adapter_ids=["b", None])
    lora_model.forward(ids, adapter_ids=None)
    assert torch.equal(lora_model.forward(ids, adapter_ids="a"), from_clean),         "single-adapter routing depended on the preceding call"


def test_generate_base_after_adapter_matches_base_model():
    """generate(adapter_id=None) is base, even after a generation under "a"."""
    lora_model, base_model = _model_with_two_adapters()
    g = torch.Generator().manual_seed(37)
    ids = torch.randint(0, 256, (1, 5), generator=g)

    lora_model.generate(ids, max_new_tokens=4, adapter_id="a")
    got = lora_model.generate(ids, max_new_tokens=4, adapter_id=None)
    with torch.no_grad():
        expected = base_model.generate(ids, 4, use_cache=True)
    assert torch.equal(got, expected),         "generate(adapter_id=None) inherited the previous adapter"


def test_forward_signature_default_is_the_unset_sentinel():
    """The sentinel is load-bearing: guard against a refactor to `=None`."""
    import inspect

    from src.engine.lora_model import UNSET

    default = inspect.signature(LoRALlamaModel.forward).parameters["adapter_ids"].default
    assert default is UNSET, (
        "forward()'s adapter_ids default must stay the UNSET sentinel -- a None "
        "default cannot distinguish 'omitted' from 'route to base'"
    )
    assert UNSET is not None


# ---------------------------------------------------------------------------
# 4. Zero-overhead path: no adapter => identical to base.
# ---------------------------------------------------------------------------


def test_lora_linear_zero_overhead_is_exact():
    torch.manual_seed(2)
    base = nn.Linear(16, 24, bias=False)
    lora = LoRALinear(base, LoRAManager(), "layers.0.attn.q_proj")
    x = torch.randn(4, 3, 16)
    # No adapter set, none passed -> byte-identical to the wrapped Linear.
    assert torch.equal(lora(x), base(x))


def test_lora_model_no_adapter_matches_base_exactly():
    model = _tiny_model()
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 256, (1, 7), generator=g)

    base_logits = model(ids)                       # plain LlamaModel forward

    lora_model = LoRALlamaModel(model, LoRAManager())
    lora_logits = lora_model.forward(ids)          # no adapter active

    # Wrapping must not perturb the base numerics at all.
    assert torch.equal(base_logits, lora_logits)


# ---------------------------------------------------------------------------
# 5. Registry residency cap + LRU eviction.
# ---------------------------------------------------------------------------


def test_manager_caps_residency_and_evicts_lru():
    mgr = LoRAManager(max_adapters=8)
    lname = "layers.0.attn.q_proj"

    def _w():
        return {lname: (torch.zeros(2, 4), torch.zeros(4, 2))}

    for i in range(8):
        mgr.load_adapter(f"a{i}", rank=2, alpha=4.0, weights_dict=_w())
    assert len(mgr) == 8

    # Touch a0 so it is most-recently-used; a1 is now the LRU.
    mgr.get_adapter("a0")
    # Loading a 9th evicts the LRU (a1), not a0.
    mgr.load_adapter("a8", rank=2, alpha=4.0, weights_dict=_w())
    assert len(mgr) == 8
    assert "a1" not in mgr
    assert "a0" in mgr and "a8" in mgr
    assert mgr.num_evicted == 1


def test_get_unknown_adapter_raises():
    mgr = LoRAManager()
    with pytest.raises(KeyError):
        mgr.get_adapter("nope")
