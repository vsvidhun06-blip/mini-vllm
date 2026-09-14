"""
ContinuousBatchScheduler <-> LoRALlamaModel integration tests.

WHY THIS FILE EXISTS
--------------------
LoRALinear and LoRALlamaModel are covered by test_lora.py; the scheduler is
covered by test_scheduler_parity.py, test_chunked_prefill.py and friends. Until
now NOTHING exercised the two together, so `ContinuousBatchScheduler.
_route_adapters` -- the entire integration surface between them -- had zero
coverage. These tests close that gap.

The property under test is the one that makes multi-tenant LoRA serving
correct: a request served in a mixed batch must get EXACTLY the tokens it would
have got running alone under its own adapter. Row i of the batched decode
forward must carry request i's adapter, request i's KV cache, and request i's
token -- and must keep carrying them as other requests finish and new ones join.

WHAT WOULD SLIP PAST A NAIVE VERSION OF THESE TESTS
---------------------------------------------------
"Scheduler output == solo output" passes trivially if routing is broken in the
direction of doing NOTHING: if every request silently ran on base weights, both
sides of the comparison would be base and the test would be green. So every
equivalence assertion here is paired with a NEGATIVE CONTROL -- the same prompt
generated under the wrong adapter, asserted to differ. A test that cannot fail
when routing is disabled is not a routing test.

Everything runs on a tiny random-weight LlamaModel on CPU with synthetic
adapters -- no HF download, no GPU, no sleeps, no threads. Weights and prompts
are seeded, so token sequences are reproducible run to run.
"""
from __future__ import annotations

import pytest
import torch

from src.engine.lora import LoRALinear, LoRAManager
from src.engine.lora_model import LoRALlamaModel, random_adapter_weights
from src.engine.model import LlamaConfig, LlamaModel
from src.engine.scheduler import ContinuousBatchScheduler, RequestStatus

BLOCK_SIZE = 16
RANK = 8
ALPHA = 16.0

# Delta magnitude for the synthetic adapters. random_adapter_weights defaults to
# 0.02, which is a realistically gentle perturbation -- and at that size the
# delta usually does NOT flip a greedy argmax on a 256-token vocab, so every
# adapter produces the same token sequence as base and the equivalence tests
# below become vacuous. (Measured across the seeds used here: 0.02 and 0.05 both
# leave adapters indistinguishable from base on several prompts; 0.2 saturates
# into a single repeated token.) 0.1 makes every adapter separate from base and
# from every other adapter on every prompt in this file while keeping the output
# non-degenerate. The negative-control assertions in each test enforce this
# property rather than trusting the constant -- if a future change breaks the
# separation, the tests fail loudly instead of passing for the wrong reason.
SCALE_INIT = 0.1

# The four attention projections LoRALlamaModel wraps. Spelled out here rather
# than imported so a silent change to the wrap set shows up as a test failure.
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _tiny_model() -> LlamaModel:
    """A small, random-weight LlamaModel on CPU. Same shape as the one in
    test_chunked_prefill.py: head_dim=16 (a power of 2, so the CUDA flash path
    would accept it too) and 3 layers to keep forwards cheap."""
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,       # head_dim = 16
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    return LlamaModel(config).eval()


def _lora_model(adapter_seeds: dict[str, int]) -> LoRALlamaModel:
    """A wrapped tiny model with one synthetic adapter per (id, seed) entry."""
    model = LoRALlamaModel(_tiny_model(), LoRAManager())
    for adapter_id, seed in adapter_seeds.items():
        model.manager.load_adapter(
            adapter_id, rank=RANK, alpha=ALPHA,
            weights_dict=random_adapter_weights(
                model, rank=RANK, seed=seed, scale_init=SCALE_INIT),
        )
    return model


def _prompt(seed: int, length: int = 6) -> torch.Tensor:
    """A (1, length) prompt of deterministic random token ids."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, length), generator=g)


def _solo(model: LoRALlamaModel, prompt_ids, max_new, adapter_id) -> list[int]:
    """Generated tokens (prompt excluded) for one prompt run on its own under
    `adapter_id`. This is the reference the scheduler must reproduce."""
    out = model.generate(
        prompt_ids, max_new_tokens=max_new, eos_token_id=None,
        use_cache=True, adapter_id=adapter_id,
    )
    return out[0, prompt_ids.shape[1]:].tolist()


def _drain(sched: ContinuousBatchScheduler, request_ids) -> dict[str, list[int]]:
    """Run the scheduler to completion, collecting each request's tokens."""
    emitted: dict[str, list[int]] = {rid: [] for rid in request_ids}
    guard = 0
    while sched.has_work():
        for rid, token_id in sched.step():
            emitted[rid].append(token_id)
        guard += 1
        assert guard < 500, "scheduler did not drain; possible infinite loop"
    return emitted


def _resident_plan(model: LoRALlamaModel):
    """The routing plan currently held by every wrapped projection.

    Walks the PUBLIC module tree rather than the `_lora_layers` bookkeeping
    list, so a projection that got wrapped but never registered (and therefore
    never routed) shows up as a disagreement instead of being invisible.
    Returns the single plan they all agree on; raises if they don't.
    """
    plans = []
    for block in model.layers:
        for proj in _PROJECTIONS:
            layer = getattr(block.attn, proj)
            assert isinstance(layer, LoRALinear), (
                f"{proj} is not LoRA-wrapped; the integration under test is absent"
            )
            active = layer._active                       # noqa: SLF001
            plans.append(tuple(active) if isinstance(active, list) else active)
    distinct = set(plans)
    assert len(distinct) == 1, (
        f"wrapped projections disagree on the active routing plan: {distinct}"
    )
    return plans[0]


# ---------------------------------------------------------------------------
# 1. Mixed-adapter scheduler equivalence.
#
# The headline property: one batched forward serving several fine-tunes gives
# every request byte-identical tokens to running that request alone under its
# own adapter. Greedy decode is deterministic and each request owns its KV
# cache, so this is exact equality -- the same contract test_scheduler_parity
# pins for the non-LoRA path.
# ---------------------------------------------------------------------------


def test_scheduler_mixed_adapters_match_solo_generation():
    """adapter "a", base (None) and adapter "b" in one batch, each exact."""
    model = _lora_model({"a": 1, "b": 2})
    max_new = 8
    plan = {"r-a": "a", "r-base": None, "r-b": "b"}
    prompts = {"r-a": _prompt(11), "r-base": _prompt(22), "r-b": _prompt(33)}

    # Reference: each prompt alone, under its own adapter.
    solo = {
        rid: _solo(model, prompts[rid], max_new, plan[rid])
        for rid in plan
    }

    # NEGATIVE CONTROL. If routing were a no-op (every request silently served
    # by the base weights) the equivalence assertions below would still pass,
    # because both sides would be base. Pin that the adapters actually move the
    # output first, so a green run below means something.
    for rid, adapter_id in plan.items():
        if adapter_id is None:
            continue
        base_ref = _solo(model, prompts[rid], max_new, None)
        assert solo[rid] != base_ref, (
            f"adapter {adapter_id!r} produced the same tokens as base for {rid}; "
            f"this test cannot detect a routing failure"
        )

    # Batched run through the real scheduler.
    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=128, block_size=BLOCK_SIZE,
    )
    for rid, prompt_ids in prompts.items():
        sched.add_request(
            request_id=rid, prompt_ids=prompt_ids, max_new_tokens=max_new,
            eos_token_id=None, adapter_id=plan[rid],
        )
    batched = _drain(sched, prompts)

    failures = []
    for rid in plan:
        if batched[rid] != solo[rid]:
            failures.append(
                f"\nRequest {rid} (adapter={plan[rid]!r})"
                f"\n  solo:    {solo[rid]}"
                f"\n  batched: {batched[rid]}"
            )
    if failures:
        raise AssertionError(
            "mixed-adapter batched generation diverged from solo generation:"
            + "".join(failures)
        )


def test_scheduler_routes_identical_prompts_to_different_adapters():
    """Same prompt, three different adapters, one batch -- three outputs.

    Sharper than the test above: with the prompt held constant, EVERY
    difference between the three requests is attributable to routing. A batch
    that collapsed to one adapter (or to base) yields identical rows and fails
    immediately, which is exactly the CUDA-graph / stale-state failure shape.
    """
    model = _lora_model({"a": 3, "b": 4})
    max_new = 8
    shared_prompt = _prompt(44)
    plan = {"same-a": "a", "same-base": None, "same-b": "b"}

    solo = {
        rid: _solo(model, shared_prompt, max_new, adapter_id)
        for rid, adapter_id in plan.items()
    }
    # The three references must be mutually distinct, else the assertions below
    # are vacuous.
    assert len({tuple(v) for v in solo.values()}) == 3, (
        f"the three adapters do not separate on this prompt: {solo}"
    )

    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=128, block_size=BLOCK_SIZE,
    )
    for rid, adapter_id in plan.items():
        sched.add_request(
            request_id=rid, prompt_ids=shared_prompt, max_new_tokens=max_new,
            eos_token_id=None, adapter_id=adapter_id,
        )
    batched = _drain(sched, plan)

    for rid, adapter_id in plan.items():
        assert batched[rid] == solo[rid], (
            f"{rid} (adapter={adapter_id!r}) got {batched[rid]}, "
            f"expected {solo[rid]}"
        )
    # And the rows really did diverge inside the one batch.
    assert len({tuple(v) for v in batched.values()}) == 3, (
        "all three rows produced the same tokens; the batch was served under a "
        "single adapter"
    )


def test_scheduler_base_requests_unaffected_by_co_batched_adapters():
    """A base request batched alongside adapter traffic stays on base weights.

    The leak this catches runs the other way from the tests above: an adapter
    row bleeding into the base row (a mis-sized per-row plan, an off-by-one in
    the grouping, a stale whole-batch str plan).
    """
    model = _lora_model({"a": 5, "b": 6})
    max_new = 8
    base_prompt = _prompt(55)

    # Reference: the base request completely alone, no adapter ever loaded into
    # the routing state.
    model.clear_adapters()
    solo_base = _solo(model, base_prompt, max_new, None)

    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=128, block_size=BLOCK_SIZE,
    )
    sched.add_request("noisy-a", _prompt(66), max_new_tokens=max_new,
                      eos_token_id=None, adapter_id="a")
    sched.add_request("quiet-base", base_prompt, max_new_tokens=max_new,
                      eos_token_id=None, adapter_id=None)
    sched.add_request("noisy-b", _prompt(77), max_new_tokens=max_new,
                      eos_token_id=None, adapter_id="b")
    batched = _drain(sched, ["noisy-a", "quiet-base", "noisy-b"])

    assert batched["quiet-base"] == solo_base, (
        f"the base request was perturbed by co-batched adapter traffic:\n"
        f"  alone:   {solo_base}\n"
        f"  batched: {batched['quiet-base']}"
    )


# ---------------------------------------------------------------------------
# 2. Churn / alignment regression.
#
# Batch composition is fluid: requests finish and are evicted, new ones are
# admitted, and the decode batch is rebuilt every step. The three row-indexed
# structures the decode forward consumes -- input_ids, caches, and the adapter
# plan -- must stay in lockstep across all of that.
#
# The spy below observes the EXISTING production boundary (set_batch_adapters,
# _decode_forward) via monkeypatch. It changes no semantics: both wrappers
# record and delegate.
# ---------------------------------------------------------------------------


class _DecodeObservation:
    """What was true at one batched-decode forward."""

    def __init__(self, expected, resident, caches_aligned, batch_size, request_ids):
        self.expected = expected                # [r.adapter_id for r in decode_reqs]
        self.resident = resident                # plan actually held by the layers
        self.caches_aligned = caches_aligned    # caches arg == decode_reqs' caches
        self.batch_size = batch_size            # input_ids.shape[0]
        self.request_ids = request_ids

    @property
    def is_per_row_plan(self) -> bool:
        """True iff a per-row plan is resident (_resident_plan returns a tuple
        for a list plan, and the bare value for None / a whole-batch str).

        A decode forward with anything else is already wrong: the scheduler
        pushes a list for every decode batch, so None means no routing reached
        the model at all and a str means a whole-batch plan leaked in. Checking
        this separately keeps a dropped-routing regression reporting as a clean
        assertion instead of a TypeError inside len().
        """
        return isinstance(self.resident, tuple)

    def is_aligned(self) -> bool:
        return (
            self.is_per_row_plan
            and list(self.resident) == self.expected
            and len(self.resident) == self.batch_size
            and self.caches_aligned
        )

    def __repr__(self) -> str:                  # shown verbatim on failure
        note = "" if self.is_per_row_plan else "  <-- NOT a per-row plan"
        return (
            f"<decode reqs={self.request_ids} expected={self.expected} "
            f"resident={self.resident!r} caches_aligned={self.caches_aligned} "
            f"batch_size={self.batch_size}>{note}"
        )


def _install_spies(monkeypatch, model: LoRALlamaModel):
    """Record every routing plan pushed, and the state at every decode forward.

    Returns (pushed_plans, decode_observations). Both wrappers delegate to the
    originals unchanged; monkeypatch undoes them at teardown.
    """
    pushed: list = []
    observed: list[_DecodeObservation] = []

    original_route = LoRALlamaModel.set_batch_adapters

    def spy_set_batch_adapters(self, adapter_ids):
        pushed.append(list(adapter_ids) if isinstance(adapter_ids, list) else adapter_ids)
        return original_route(self, adapter_ids)

    original_decode = ContinuousBatchScheduler._decode_forward

    def spy_decode_forward(self, input_ids, caches):
        decode_reqs = [r for r in self.active if r.status is RequestStatus.DECODE]
        observed.append(_DecodeObservation(
            expected=[r.adapter_id for r in decode_reqs],
            resident=_resident_plan(model),
            caches_aligned=[r.cache for r in decode_reqs] == caches,
            batch_size=int(input_ids.shape[0]),
            request_ids=[r.request_id for r in decode_reqs],
        ))
        return original_decode(self, input_ids, caches)

    monkeypatch.setattr(
        LoRALlamaModel, "set_batch_adapters", spy_set_batch_adapters)
    monkeypatch.setattr(
        ContinuousBatchScheduler, "_decode_forward", spy_decode_forward)
    return pushed, observed


def test_routing_stays_aligned_under_admission_and_completion_churn(monkeypatch):
    """Staggered arrivals + staggered completions, three adapters and a base row.

    Requests join on different steps and finish on different steps, so the
    decode batch grows and shrinks and every row index is reused by a different
    request over the run. On EVERY batched decode forward the resident plan must
    equal the live decode order, and the caches handed to the model must be that
    same order's caches.
    """
    model = _lora_model({"a": 7, "b": 8, "c": 9})
    pushed, observed = _install_spies(monkeypatch, model)

    sched = ContinuousBatchScheduler(
        model, max_batch_size=4, num_blocks=256, block_size=BLOCK_SIZE,
    )
    # (request_id, adapter, max_new_tokens, admit_on_step). Staggered arrivals
    # AND deliberately unequal budgets so completions stagger too.
    #
    # The budgets are chosen so "churn-a" is still decoding when "churn-base"
    # is admitted behind it: active order is admission order, so this puts the
    # base row at index 1 with an adapter row ahead of it -- the configuration
    # that catches a plan which forward-fills None from the preceding row. The
    # short-lived "churn-b" is what makes the batch shrink mid-run.
    schedule = [
        ("churn-a",    "a", 12, 0),
        ("churn-base", None, 9, 2),
        ("churn-b",    "b",  4, 4),
        ("churn-c",    "c", 11, 6),
    ]
    emitted: dict[str, list[int]] = {rid: [] for rid, _, _, _ in schedule}

    for step_idx in range(60):
        for rid, adapter_id, max_new, admit_step in schedule:
            if step_idx == admit_step:
                sched.add_request(
                    request_id=rid, prompt_ids=_prompt(100 + admit_step),
                    max_new_tokens=max_new, eos_token_id=None,
                    adapter_id=adapter_id,
                )
        if not sched.has_work() and step_idx > max(s[3] for s in schedule):
            break
        for rid, token_id in sched.step():
            emitted[rid].append(token_id)

    # ---- the churn actually happened -------------------------------------
    assert len(observed) >= 10, (
        f"only {len(observed)} batched decode forwards; the run was too short "
        f"to exercise churn"
    )
    batch_sizes = {o.batch_size for o in observed}
    assert len(batch_sizes) >= 3, (
        f"decode batch size never varied enough to test churn: {sorted(batch_sizes)}"
    )
    assert all(len(v) > 0 for v in emitted.values()), (
        f"some request never emitted a token: "
        f"{ {k: len(v) for k, v in emitted.items()} }"
    )

    # ---- alignment held on every single forward ---------------------------
    misaligned = [o for o in observed if not o.is_aligned()]
    if misaligned:
        raise AssertionError(
            "adapter routing / KV-cache ordering drifted out of alignment with "
            "the decode batch on "
            f"{len(misaligned)} of {len(observed)} decode forwards:\n  "
            + "\n  ".join(repr(o) for o in misaligned)
        )

    # ---- a base row really was present in a mixed batch -------------------
    assert any(None in o.expected and len(set(o.expected)) > 1 for o in observed), (
        "no forward mixed a base row with adapter rows; the interesting case "
        "was never exercised"
    )
    # ...and specifically NOT only at row 0. A base row sitting after an adapter
    # row is what catches a plan that forward-fills None with the preceding
    # adapter; if every base row were at index 0 that bug would be invisible
    # here (the equivalence tests above still catch it, but this test should
    # not silently stop covering it if the schedule is ever re-timed).
    assert any(
        None in o.expected[1:] and len(set(o.expected)) > 1 for o in observed
    ), (
        "every mixed batch had its base row at index 0; the schedule no longer "
        "exercises a base row preceded by an adapter row"
    )

    # ---- the scheduler pushed a plan for every decode forward -------------
    # (prefill pushes length-1 plans too, so pushes >= decode forwards.)
    assert len(pushed) >= len(observed), (
        f"{len(observed)} decode forwards but only {len(pushed)} routing pushes; "
        f"some forward ran on an inherited plan"
    )


def test_decode_plan_length_always_matches_batch_size(monkeypatch):
    """Every plan the scheduler pushes for a decode forward sizes the batch.

    A plan shorter or longer than the batch is the failure mode LoRALinear.
    _apply_per_row raises on; catching it here names the scheduler as the
    culprit instead of surfacing it as a mid-forward ValueError.
    """
    model = _lora_model({"a": 12, "b": 13})
    _pushed, observed = _install_spies(monkeypatch, model)

    sched = ContinuousBatchScheduler(
        model, max_batch_size=3, num_blocks=128, block_size=BLOCK_SIZE,
    )
    # Unequal prompt lengths as well as unequal budgets, so prefill and decode
    # interleave differently for each request.
    sched.add_request("len-a", _prompt(201, length=4), max_new_tokens=6,
                      eos_token_id=None, adapter_id="a")
    sched.add_request("len-b", _prompt(202, length=9), max_new_tokens=4,
                      eos_token_id=None, adapter_id="b")
    sched.add_request("len-base", _prompt(203, length=7), max_new_tokens=8,
                      eos_token_id=None, adapter_id=None)
    _drain(sched, ["len-a", "len-b", "len-base"])

    assert observed, "no batched decode forward was recorded"
    for o in observed:
        assert o.is_per_row_plan, (
            f"no per-row routing plan was resident for this decode forward: {o!r}"
        )
        assert len(o.resident) == o.batch_size == len(o.request_ids), (
            f"routing plan does not size the decode batch: {o!r}"
        )


def test_scheduler_leaves_no_routing_state_for_a_plain_model(monkeypatch):
    """A plain LlamaModel is never routed -- the non-LoRA path is untouched.

    `_route_adapters` is guarded by getattr(model, "set_batch_adapters"), so a
    plain model must produce zero pushes even when requests carry adapter_ids.
    """
    pushed: list = []
    original_route = LoRALlamaModel.set_batch_adapters

    def spy(self, adapter_ids):
        pushed.append(adapter_ids)
        return original_route(self, adapter_ids)

    monkeypatch.setattr(LoRALlamaModel, "set_batch_adapters", spy)

    plain = _tiny_model()
    assert not hasattr(plain, "set_batch_adapters")
    sched = ContinuousBatchScheduler(
        plain, max_batch_size=4, num_blocks=128, block_size=BLOCK_SIZE,
    )
    # adapter_id is accepted by add_request and must simply be ignored here.
    sched.add_request("plain-1", _prompt(301), max_new_tokens=4,
                      eos_token_id=None, adapter_id="not-registered")
    sched.add_request("plain-2", _prompt(302), max_new_tokens=4,
                      eos_token_id=None)
    emitted = _drain(sched, ["plain-1", "plain-2"])

    assert pushed == [], f"a plain LlamaModel was routed: {pushed}"
    assert all(len(v) == 4 for v in emitted.values()), (
        f"plain-model generation was disturbed: "
        f"{ {k: len(v) for k, v in emitted.items()} }"
    )
