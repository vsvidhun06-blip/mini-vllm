"""
Tests for the routing policy layer.

All CPU-compatible and model-free: the router is pure policy + bookkeeping, so
we drive it with the real RuleBasedClassifier and the default small/large fleet
without ever loading TinyLlama. record_outcome is fed synthetic latencies
directly, exactly as serving.py would after a real generation.

Coverage:
  * Simple queries route to the small (cheap) model.
  * Complex queries route to the large (capable) model.
  * cost_savings_pct is positive once any request avoids the largest model.
  * EMA latency tracking seeds then blends with alpha=0.1.
  * routing_stats() shape and the all-busy / over-complex fallback.
"""
from __future__ import annotations

import pytest

from src.router.classifier import RequestComplexity, RuleBasedClassifier
from src.router.router import LLMRouter, ModelConfig, default_model_configs


@pytest.fixture
def router() -> LLMRouter:
    return LLMRouter(default_model_configs(), RuleBasedClassifier())


# ---------------------------------------------------------------------------
# Routing decisions.
# ---------------------------------------------------------------------------


def test_simple_query_routes_to_small(router):
    # A short factual lookup -> SIMPLE -> cheapest capable model is "small".
    assert router.route("What is the capital of France?").name == "small"


def test_complex_query_routes_to_large(router):
    # Code request -> COMPLEX -> only "large" is capable.
    assert router.route("Write a Python function that implements quicksort").name == "large"


def test_moderate_query_routes_to_small(router):
    # MODERATE is still within "small"'s max_complexity, so it stays cheap.
    cfg = router.route("Explain the difference between TCP and UDP with examples")
    assert cfg.name == "small"


def test_cheapest_capable_is_chosen():
    # Three models all capable of SIMPLE: the cheapest must win.
    configs = [
        ModelConfig("a", "p", cost_per_token=3.0, max_complexity=RequestComplexity.COMPLEX),
        ModelConfig("b", "p", cost_per_token=1.0, max_complexity=RequestComplexity.COMPLEX),
        ModelConfig("c", "p", cost_per_token=2.0, max_complexity=RequestComplexity.COMPLEX),
    ]
    router = LLMRouter(configs, RuleBasedClassifier())
    assert router.route("Who wrote Hamlet?").name == "b"


def test_fallback_to_most_capable_when_none_capable():
    # Only SIMPLE-rated models exist, but the prompt is COMPLEX (code). With no
    # capable model the router escalates to the most capable one rather than
    # dropping the request.
    configs = [
        ModelConfig("tiny", "p", cost_per_token=1.0, max_complexity=RequestComplexity.SIMPLE),
        ModelConfig("small", "p", cost_per_token=2.0, max_complexity=RequestComplexity.SIMPLE),
    ]
    router = LLMRouter(configs, RuleBasedClassifier())
    chosen = router.route("def f(): return 1")
    # max_complexity ties at SIMPLE, so the cost tie-break picks the pricier one
    # as "most capable".
    assert chosen.name == "small"


# ---------------------------------------------------------------------------
# Cost savings.
# ---------------------------------------------------------------------------


def test_cost_savings_positive(router):
    # A workload with some cheap traffic must show positive savings vs always
    # using the large model.
    prompts = [
        "What is the capital of France?",          # SIMPLE -> small
        "Who wrote Hamlet?",                        # SIMPLE -> small
        "Write a Python function to sort a list",   # COMPLEX -> large
    ]
    for p in prompts:
        router.route(p)
    stats = router.routing_stats()
    assert stats["cost_savings_pct"] > 0.0


def test_cost_savings_zero_when_all_to_largest(router):
    # If every request is COMPLEX they all go to the large model == the
    # baseline, so there are no savings.
    for _ in range(5):
        router.route("Write a Python function that implements quicksort")
    assert router.routing_stats()["cost_savings_pct"] == pytest.approx(0.0)


def test_cost_savings_value(router):
    # 2 small (cost 1 each) + 1 large (cost 5): actual = 7, baseline = 3*5 = 15,
    # savings = (15-7)/15 = 53.33%.
    router.route("Who wrote Hamlet?")
    router.route("What is the capital of France?")
    router.route("Write a Python function to sort a list")
    stats = router.routing_stats()
    assert stats["cost_savings_pct"] == pytest.approx((15 - 7) / 15 * 100.0)


# ---------------------------------------------------------------------------
# EMA latency tracking.
# ---------------------------------------------------------------------------


def test_ema_seeds_on_first_sample(router):
    # The first outcome sets avg_latency_ms directly (no blend against the 0.0
    # "never measured" sentinel).
    router.record_outcome("small", 100.0, tokens_generated=10)
    assert router._by_name["small"].avg_latency_ms == pytest.approx(100.0)


def test_ema_blends_subsequent_samples(router):
    # alpha=0.1: new = 0.1*sample + 0.9*prev.
    router.record_outcome("small", 100.0, 10)       # seed -> 100
    router.record_outcome("small", 200.0, 10)       # 0.1*200 + 0.9*100 = 110
    assert router._by_name["small"].avg_latency_ms == pytest.approx(110.0)
    router.record_outcome("small", 200.0, 10)       # 0.1*200 + 0.9*110 = 119
    assert router._by_name["small"].avg_latency_ms == pytest.approx(119.0)


def test_ema_is_per_model(router):
    # Each model tracks its own latency independently.
    router.record_outcome("small", 50.0, 10)
    router.record_outcome("large", 500.0, 10)
    assert router._by_name["small"].avg_latency_ms == pytest.approx(50.0)
    assert router._by_name["large"].avg_latency_ms == pytest.approx(500.0)


def test_record_outcome_unknown_model_raises(router):
    with pytest.raises(KeyError):
        router.record_outcome("nonexistent", 100.0, 10)


# ---------------------------------------------------------------------------
# routing_stats() shape.
# ---------------------------------------------------------------------------


def test_routing_stats_shape(router):
    router.route("Who wrote Hamlet?")
    router.route("Write a Python function to sort a list")
    router.record_outcome("small", 80.0, 12)
    stats = router.routing_stats()

    assert set(stats) == {
        "requests_per_model",
        "avg_latency_per_model",
        "cost_savings_pct",
        "complexity_distribution",
    }
    # Every configured model appears in both per-model maps.
    assert set(stats["requests_per_model"]) == {"small", "large"}
    assert set(stats["avg_latency_per_model"]) == {"small", "large"}
    # Complexity distribution is keyed by enum name and counts every routed req.
    assert stats["complexity_distribution"]["SIMPLE"] == 1
    assert stats["complexity_distribution"]["COMPLEX"] == 1
    assert sum(stats["complexity_distribution"].values()) == 2


def test_routing_stats_counts_match_routes(router):
    for _ in range(3):
        router.route("Who wrote Hamlet?")                 # SIMPLE -> small
    for _ in range(2):
        router.route("Write a Python function to sort")    # COMPLEX -> large
    stats = router.routing_stats()
    assert stats["requests_per_model"]["small"] == 3
    assert stats["requests_per_model"]["large"] == 2


def test_empty_router_rejected():
    with pytest.raises(ValueError):
        LLMRouter([], RuleBasedClassifier())
