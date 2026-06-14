"""
Request complexity classification.

The router's first job is to look at an incoming prompt and decide how hard it
is: a trivial factual lookup ("What is 2+2?"), a moderate explanation ("Explain
the difference between TCP and UDP"), or a genuinely complex generation task
("Write a Python function that implements quicksort"). That label then drives
the routing decision in router.py -- simple goes to a small/cheap model, complex
goes to the large/capable one.

Two classifiers live here, in increasing order of sophistication and cost:

  * RuleBasedClassifier  -- a handful of hand-written rules over cheap lexical
                            features. Deterministic, ~microseconds, zero
                            dependencies. This is the production default: the
                            classifier itself must never become the bottleneck
                            it's trying to avoid.
  * EmbeddingClassifier  -- an OPTIONAL wrapper that, *if* sentence-transformers
                            is installed, embeds the prompt and scores it against
                            per-class prototype sentences. When the dependency
                            isn't present it transparently falls back to the
                            rule-based classifier, so importing this module never
                            forces a heavy ML dependency on the caller.

Everything here is pure Python -- no torch, no tokenizer, no model. That's
deliberate: the whole point of a router is to spend as little as possible
deciding where to send a request, and it keeps the routing logic unit-testable
on CI without a GPU or a multi-GB checkpoint.
"""
from __future__ import annotations

import re
from enum import IntEnum


# ---------------------------------------------------------------------------
# RequestComplexity
# ---------------------------------------------------------------------------
#
# Why IntEnum and not a plain Enum?
#
#   The routing rule is "pick the cheapest model whose max_complexity is >= the
#   request's complexity". That comparison (`model.max_complexity >= complexity`)
#   only works if the members order. IntEnum gives us free, well-defined ordering
#   (SIMPLE < MODERATE < COMPLEX) while still printing as `RequestComplexity.SIMPLE`
#   and carrying a `.name` we can use as a JSON-friendly dict key in stats.
# ---------------------------------------------------------------------------


class RequestComplexity(IntEnum):
    SIMPLE = 0
    MODERATE = 1
    COMPLEX = 2


# ---------------------------------------------------------------------------
# Lexical vocabularies driving feature extraction.
# ---------------------------------------------------------------------------
#
# These are intentionally small, hand-curated word lists rather than anything
# learned. The router trades a little classification accuracy for speed and
# determinism; the embedding classifier exists for callers who want to pay more
# for better accuracy.

# Words that signal an analytical / multi-part request: the user is asking for
# reasoning, comparison, or explanation rather than a single fact. Two or more of
# these is a strong "this is at least MODERATE" signal. Note "what/who/when/where"
# are NOT here -- those are *factual* markers (see _FACTUAL_RE) and pull the other
# direction (toward SIMPLE).
_QUESTION_WORDS = frozenset({
    "how", "why", "explain", "compare", "comparison", "analyse", "analyze",
    "analysis", "describe", "discuss", "evaluate", "difference", "differences",
    "differ", "versus", "vs", "contrast", "elaborate", "examples", "example",
    "summarise", "summarize", "derive", "prove", "justify",
})

# Keywords/structures that betray a code request. Whole-word matches plus the
# fenced-code-block marker. "function" is here so "Write a Python function ..."
# trips has_code even without a fence.
_CODE_KEYWORDS = frozenset({
    "def", "class", "function", "return", "import", "lambda", "async", "await",
    "public", "private", "void", "const",
})

# Factual-question openers. A prompt matching one of these is asking for a
# discrete fact (who/what/when/where/which), which -- when short -- is the
# canonical SIMPLE request.
_FACTUAL_RE = re.compile(r"\b(who|what|when|where|which)\b", re.IGNORECASE)

# A number, an arithmetic/relational operator, then another number: "2+2",
# "3 * 4", "x = 5". Captures the most common "has math" signal cheaply.
_MATH_PAIR_RE = re.compile(r"\d+\s*[\+\-\*/=^%]\s*\d+")

# Standalone math symbols / notation that imply a maths task even without a
# digit-operator-digit pair.
_MATH_SYMBOL_RE = re.compile(r"[=≤≥≠∑∫√π∞]|\\(?:frac|sum|int|sqrt)")

# Words that strongly imply a maths/quantitative task in prose form.
_MATH_WORDS = frozenset({
    "equation", "derivative", "integral", "matrix", "matrices", "polynomial",
    "theorem", "calculate", "compute", "probability", "factorial",
})

# Tokenisation for feature extraction: split into word-ish runs. We strip
# surrounding punctuation so "examples" and "examples." count the same.
_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------
#
# Turns a raw prompt string into a flat dict of cheap lexical features. The
# RuleBasedClassifier consumes this dict; keeping extraction separate means the
# features are inspectable (useful in tests and in the /route/stats surface) and
# reusable by any future classifier.
#
# token_count is a *word* count, not a real BPE token count. We deliberately do
# NOT load the model's tokenizer here: that would pull torch + a multi-MB merges
# table into the hot routing path just to approximate length. Whitespace/word
# count correlates well enough with true token count for a length threshold, and
# costs nothing. The thresholds in RuleBasedClassifier are tuned against this
# proxy, not against true tokens.
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Extract cheap lexical features from a prompt. Stateless and pure."""

    def extract(self, prompt: str) -> dict:
        """Return a feature dict for `prompt`.

        Keys:
            token_count          -- number of word tokens (BPE-token proxy).
            question_word_count  -- count of analytical words (how/why/explain...).
            has_code             -- fenced block or code keyword present.
            has_math             -- digit-operator-digit, math symbol, or math word.
            sentence_count       -- number of sentences (>= 1).
            avg_word_length      -- mean characters per word token (0.0 if empty).
            is_factual           -- a who/what/when/where/which question.
        """
        # Lowercased word tokens, punctuation stripped. The single pass here
        # feeds token_count, question_word_count and avg_word_length.
        words = _WORD_RE.findall(prompt)
        lowered = [w.lower() for w in words]

        token_count = len(words)

        # Count every occurrence (not distinct) -- a prompt that says "compare"
        # and "contrast" is more analytical than one that just says "compare".
        question_word_count = sum(1 for w in lowered if w in _QUESTION_WORDS)

        has_code = ("```" in prompt) or any(w in _CODE_KEYWORDS for w in lowered)

        has_math = bool(
            _MATH_PAIR_RE.search(prompt)
            or _MATH_SYMBOL_RE.search(prompt)
            or any(w in _MATH_WORDS for w in lowered)
        )

        # Sentence count: split on terminal punctuation runs, drop empty
        # fragments, and floor at 1 so a fragment with no '.'/'!'/'?' still
        # counts as one sentence.
        sentences = [s for s in re.split(r"[.!?]+", prompt) if s.strip()]
        sentence_count = max(1, len(sentences))

        avg_word_length = (
            sum(len(w) for w in words) / token_count if token_count else 0.0
        )

        is_factual = bool(_FACTUAL_RE.search(prompt))

        return {
            "token_count": token_count,
            "question_word_count": question_word_count,
            "has_code": has_code,
            "has_math": has_math,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "is_factual": is_factual,
        }


# ---------------------------------------------------------------------------
# RuleBasedClassifier
# ---------------------------------------------------------------------------
#
# The rules are applied IN ORDER and the first match wins. Order matters because
# the conditions overlap -- e.g. "What is 2+2?" is both factual-and-short AND
# contains math. We want the short factual lookup to win (route cheap), so the
# factual rule is checked before the code/math rule.
#
#   1. token_count < 20 AND is_factual            -> SIMPLE
#        Short, fact-shaped questions. A small model nails these; sending them
#        to a big model is pure waste. Checked first so "What is 2+2?" lands
#        SIMPLE despite the arithmetic.
#   2. has_code OR has_math                        -> COMPLEX
#        Code generation and real maths are where small models fall down most
#        visibly. Route straight to the capable model.
#   3. question_word_count >= 2 OR token_count>100 -> MODERATE
#        Multi-part analytical asks ("explain ... and compare ...") or just long
#        prompts. A mid-tier capable model, not the biggest one.
#   4. else                                        -> SIMPLE
#        Default to cheap. The router's fallback guarantees correctness if this
#        under-estimates; over-estimating would just burn money.
# ---------------------------------------------------------------------------


class RuleBasedClassifier:
    """Deterministic complexity classifier over FeatureExtractor output."""

    def classify(self, features: dict) -> RequestComplexity:
        # Rule 1: short factual lookups are the canonical SIMPLE case. Checked
        # before code/math so a tiny arithmetic fact ("What is 2+2?") doesn't get
        # promoted to COMPLEX by the has_math signal.
        if features["token_count"] < 20 and features["is_factual"]:
            return RequestComplexity.SIMPLE

        # Rule 2: code or maths -> the capable model. These are the tasks where a
        # small model's failure is most expensive (wrong code, wrong arithmetic).
        if features["has_code"] or features["has_math"]:
            return RequestComplexity.COMPLEX

        # Rule 3: multi-part analytical asks or very long prompts -> mid tier.
        if features["question_word_count"] >= 2 or features["token_count"] > 100:
            return RequestComplexity.MODERATE

        # Rule 4: everything else is cheap by default.
        return RequestComplexity.SIMPLE


# ---------------------------------------------------------------------------
# EmbeddingClassifier (optional)
# ---------------------------------------------------------------------------
#
# A semantic upgrade over keyword rules: embed the prompt with a small
# sentence-transformers model and pick the complexity class whose prototype
# sentences it's most similar to. This catches phrasings the rules miss (a
# complex request worded without any code keyword, say).
#
# Crucially it is OPTIONAL. sentence-transformers is a heavy dependency
# (transformers + a model download); this project does not require it. So:
#
#   * If the import succeeds, we build the encoder lazily on first use and run
#     the cosine-similarity classification, returning a real confidence.
#   * If the import fails (the common case here), we fall back to the
#     RuleBasedClassifier and report a fixed confidence reflecting how decisive
#     the matched rule was.
#
# Either way the public contract is identical: classify(prompt) -> (complexity,
# confidence). The router can hold one of these without caring which path ran.
# ---------------------------------------------------------------------------


# Prototype sentences for each complexity class. The embedding path scores a
# prompt against these and takes the argmax class. Hand-written to span the
# typical phrasing of each tier.
_PROTOTYPES: dict[RequestComplexity, list[str]] = {
    RequestComplexity.SIMPLE: [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "When did World War 2 end?",
        "Define photosynthesis.",
    ],
    RequestComplexity.MODERATE: [
        "Explain the difference between TCP and UDP with examples.",
        "Compare and contrast supervised and unsupervised learning.",
        "Summarise the causes of the French Revolution.",
        "Why does inflation rise when interest rates fall?",
    ],
    RequestComplexity.COMPLEX: [
        "Write a Python function that implements quicksort.",
        "Derive the gradient of the softmax cross-entropy loss.",
        "Implement a thread-safe LRU cache in C++.",
        "Prove that the halting problem is undecidable.",
    ],
}


class EmbeddingClassifier:
    """Embedding-based classifier with a transparent rule-based fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._extractor = FeatureExtractor()
        self._fallback = RuleBasedClassifier()
        # Lazily constructed on first classify() if sentence-transformers is
        # importable. Stays None forever in the fallback case.
        self._encoder = None
        self._proto_matrix = None        # (n_protos, dim) embeddings
        self._proto_labels: list[RequestComplexity] = []
        self._tried_load = False

    # -- availability / lazy load -----------------------------------------

    @property
    def embeddings_available(self) -> bool:
        """True iff the embedding path is usable (after attempting a load)."""
        self._ensure_encoder()
        return self._encoder is not None

    def _ensure_encoder(self) -> None:
        """Try to build the encoder + prototype matrix exactly once.

        Swallows ImportError (sentence-transformers not installed) and any
        load-time error (no network for the model download), leaving
        `self._encoder` as None so callers fall back gracefully.
        """
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            # Most common path in this project: dependency absent. Fall back.
            return
        try:
            encoder = SentenceTransformer(self.model_name)
            protos: list[str] = []
            labels: list[RequestComplexity] = []
            for complexity, sentences in _PROTOTYPES.items():
                for s in sentences:
                    protos.append(s)
                    labels.append(complexity)
            # normalize so a dot product IS cosine similarity.
            matrix = encoder.encode(protos, normalize_embeddings=True)
            self._encoder = encoder
            self._proto_matrix = matrix
            self._proto_labels = labels
        except Exception:
            # Import worked but the model couldn't load (offline, etc.).
            self._encoder = None

    # -- classification ----------------------------------------------------

    def classify(self, prompt: str) -> tuple[RequestComplexity, float]:
        """Classify `prompt`, returning (complexity, confidence in [0, 1])."""
        self._ensure_encoder()
        if self._encoder is not None:
            return self._classify_embedding(prompt)
        return self._classify_fallback(prompt)

    def _classify_embedding(self, prompt: str) -> tuple[RequestComplexity, float]:
        """Cosine-similarity classification against the prototype set."""
        # encode returns a (1, dim) array when given a single string in a list;
        # normalize so dot product == cosine similarity.
        vec = self._encoder.encode([prompt], normalize_embeddings=True)[0]
        # Similarity to every prototype, then take, per class, the best-matching
        # prototype's score (max-pool within class).
        per_class: dict[RequestComplexity, float] = {}
        for label, proto in zip(self._proto_labels, self._proto_matrix):
            sim = float(sum(a * b for a, b in zip(vec, proto)))
            if label not in per_class or sim > per_class[label]:
                per_class[label] = sim
        # argmax class; confidence = softmax of the per-class best similarities,
        # which keeps it in [0, 1] and reflects how separated the winner is.
        best = max(per_class, key=per_class.get)
        confidence = _softmax_confidence(per_class, best)
        return best, confidence

    def _classify_fallback(self, prompt: str) -> tuple[RequestComplexity, float]:
        """Rule-based result, with a confidence reflecting rule decisiveness."""
        features = self._extractor.extract(prompt)
        complexity = self._fallback.classify(features)
        # A decisive rule (short+factual, or code/math) gets high confidence;
        # the catch-all default gets a low one. This lets a caller threshold on
        # confidence even without the embedding model.
        if (features["token_count"] < 20 and features["is_factual"]) \
                or features["has_code"] or features["has_math"]:
            confidence = 0.9
        elif features["question_word_count"] >= 2 or features["token_count"] > 100:
            confidence = 0.7
        else:
            confidence = 0.5
        return complexity, confidence


def _softmax_confidence(
    per_class: dict[RequestComplexity, float], winner: RequestComplexity
) -> float:
    """Softmax over per-class similarities; return the winner's probability."""
    import math

    scores = list(per_class.values())
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]                # shift for stability
    total = sum(exps)
    winner_exp = math.exp(per_class[winner] - m)
    return winner_exp / total if total else 0.0
