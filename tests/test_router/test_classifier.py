"""
Tests for the request-complexity classifier.

Everything here is pure CPU and loads no model -- the classifier is plain Python
over lexical features, which is the whole point (the router must decide where to
send a request without paying a model's worth of compute to do it). That makes
these tests fast and dependency-free.

Coverage:
  * RuleBasedClassifier over 10 prompts spanning all three complexity levels.
  * The three canonical prompts from the spec.
  * FeatureExtractor output correctness (each feature, exercised directly).
  * EmbeddingClassifier's fallback contract.
"""
from __future__ import annotations

import pytest

from src.router.classifier import (
    EmbeddingClassifier,
    FeatureExtractor,
    RequestComplexity,
    RuleBasedClassifier,
)


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture
def classifier() -> RuleBasedClassifier:
    return RuleBasedClassifier()


def _classify(extractor, classifier, prompt: str) -> RequestComplexity:
    return classifier.classify(extractor.extract(prompt))


# ---------------------------------------------------------------------------
# The three canonical prompts from the spec.
# ---------------------------------------------------------------------------


def test_canonical_simple(extractor, classifier):
    # Short + factual -> rule 1 fires before the has_math rule can promote it.
    assert _classify(extractor, classifier, "What is 2+2?") is RequestComplexity.SIMPLE


def test_canonical_moderate(extractor, classifier):
    # Not factual, no code/math, but "explain"+"difference"+"examples" push
    # question_word_count >= 2 -> MODERATE.
    prompt = "Explain the difference between TCP and UDP with examples"
    assert _classify(extractor, classifier, prompt) is RequestComplexity.MODERATE


def test_canonical_complex(extractor, classifier):
    # "function" is a code keyword -> has_code -> COMPLEX.
    prompt = "Write a Python function that implements quicksort"
    assert _classify(extractor, classifier, prompt) is RequestComplexity.COMPLEX


# ---------------------------------------------------------------------------
# 10 prompts spanning all three levels.
# ---------------------------------------------------------------------------


# (prompt, expected) -- 4 SIMPLE, 3 MODERATE, 3 COMPLEX.
_TEN_PROMPTS = [
    # SIMPLE: short factual lookups.
    ("What is the capital of France?", RequestComplexity.SIMPLE),
    ("Who wrote Hamlet?", RequestComplexity.SIMPLE),
    ("When did World War 2 end?", RequestComplexity.SIMPLE),
    ("Where is Mount Everest?", RequestComplexity.SIMPLE),
    # MODERATE: multi-part analytical asks (>=2 question words), no code/math.
    ("Explain the difference between TCP and UDP with examples",
     RequestComplexity.MODERATE),
    ("Compare and contrast supervised versus unsupervised learning",
     RequestComplexity.MODERATE),
    ("Describe how and why caching improves throughput",
     RequestComplexity.MODERATE),
    # COMPLEX: code or maths.
    ("Write a Python function that implements quicksort",
     RequestComplexity.COMPLEX),
    ("def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)",
     RequestComplexity.COMPLEX),
    ("Solve the equation 3*x = 12 and show the derivative",
     RequestComplexity.COMPLEX),
]


@pytest.mark.parametrize("prompt,expected", _TEN_PROMPTS)
def test_ten_prompts_all_levels(extractor, classifier, prompt, expected):
    assert _classify(extractor, classifier, prompt) is expected


def test_ten_prompts_cover_every_level(extractor, classifier):
    """Sanity: the suite above actually exercises all three complexity levels."""
    seen = {_classify(extractor, classifier, p) for p, _ in _TEN_PROMPTS}
    assert seen == {
        RequestComplexity.SIMPLE,
        RequestComplexity.MODERATE,
        RequestComplexity.COMPLEX,
    }


# ---------------------------------------------------------------------------
# Rule-ordering edge cases.
# ---------------------------------------------------------------------------


def test_short_factual_beats_math(extractor, classifier):
    # "What is 2+2?" has_math is True, but the short-factual rule is checked
    # first, so it must NOT be promoted to COMPLEX.
    feats = extractor.extract("What is 2+2?")
    assert feats["has_math"] is True
    assert feats["is_factual"] is True
    assert classifier.classify(feats) is RequestComplexity.SIMPLE


def test_long_factual_is_not_simple_via_rule1(extractor, classifier):
    # A factual opener but > 20 tokens skips rule 1; with two question words it
    # lands MODERATE rather than SIMPLE.
    prompt = (
        "What are the main reasons that explain and compare why distributed "
        "systems are so much harder to reason about than single machines when "
        "you scale them up"
    )
    feats = extractor.extract(prompt)
    assert feats["token_count"] >= 20
    assert classifier.classify(feats) is RequestComplexity.MODERATE


def test_empty_prompt_defaults_simple(extractor, classifier):
    # Degenerate input must not crash and should fall through to the cheap
    # default rather than raising.
    assert _classify(extractor, classifier, "") is RequestComplexity.SIMPLE


# ---------------------------------------------------------------------------
# FeatureExtractor correctness.
# ---------------------------------------------------------------------------


def test_feature_token_count(extractor):
    assert extractor.extract("one two three four")["token_count"] == 4


def test_feature_question_word_count(extractor):
    feats = extractor.extract("Explain the difference and compare these examples")
    # explain, difference, compare, examples -> 4
    assert feats["question_word_count"] == 4


def test_feature_has_code_keyword(extractor):
    assert extractor.extract("write a function to do x")["has_code"] is True
    assert extractor.extract("class Foo: pass")["has_code"] is True


def test_feature_has_code_fence(extractor):
    assert extractor.extract("here is code:\n```\nx=1\n```")["has_code"] is True


def test_feature_no_code(extractor):
    assert extractor.extract("tell me a story about a cat")["has_code"] is False


def test_feature_has_math_operator(extractor):
    assert extractor.extract("compute 3 + 4")["has_math"] is True


def test_feature_has_math_word(extractor):
    assert extractor.extract("find the derivative of this")["has_math"] is True


def test_feature_no_math(extractor):
    assert extractor.extract("hello there friend")["has_math"] is False


def test_feature_sentence_count(extractor):
    assert extractor.extract("One. Two! Three?")["sentence_count"] == 3
    # No terminal punctuation still counts as a single sentence.
    assert extractor.extract("just a fragment")["sentence_count"] == 1


def test_feature_avg_word_length(extractor):
    feats = extractor.extract("ab cde")            # lengths 2 and 3 -> mean 2.5
    assert feats["avg_word_length"] == pytest.approx(2.5)


def test_feature_avg_word_length_empty(extractor):
    assert extractor.extract("")["avg_word_length"] == 0.0


def test_feature_is_factual(extractor):
    assert extractor.extract("Who is the president?")["is_factual"] is True
    assert extractor.extract("Where is Rome?")["is_factual"] is True
    assert extractor.extract("Generate a poem")["is_factual"] is False


def test_feature_keys_complete(extractor):
    feats = extractor.extract("any prompt here")
    assert set(feats) == {
        "token_count",
        "question_word_count",
        "has_code",
        "has_math",
        "sentence_count",
        "avg_word_length",
        "is_factual",
    }


# ---------------------------------------------------------------------------
# EmbeddingClassifier fallback contract.
# ---------------------------------------------------------------------------


def test_embedding_classifier_returns_complexity_and_confidence():
    # With sentence-transformers absent (the case in this project), the wrapper
    # must fall back to rules and still honour its (complexity, confidence)
    # contract. The classification must match the rule-based result.
    ec = EmbeddingClassifier()
    complexity, confidence = ec.classify("What is 2+2?")
    assert complexity is RequestComplexity.SIMPLE
    assert 0.0 <= confidence <= 1.0


def test_embedding_classifier_fallback_agrees_with_rules():
    ec = EmbeddingClassifier()
    rb = RuleBasedClassifier()
    fx = FeatureExtractor()
    if ec.embeddings_available:
        pytest.skip("sentence-transformers installed; fallback path not exercised")
    for prompt, _ in _TEN_PROMPTS:
        emb_complexity, _conf = ec.classify(prompt)
        rule_complexity = rb.classify(fx.extract(prompt))
        assert emb_complexity is rule_complexity
