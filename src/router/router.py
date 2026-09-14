"""
The routing policy layer.

Given a classified request complexity and a fleet of models of different
size/cost, decide which model should serve the request. The policy is the same
one production inference stacks (OpenAI, Anthropic, Together) use to manage the
cost/latency/quality frontier: send each request to the *cheapest* model that
is still capable of handling it, and reserve the big expensive model for the
work that actually needs it.

This module is pure policy + bookkeeping -- it never touches torch or loads a
model. It takes a `classifier` (RuleBasedClassifier or EmbeddingClassifier from
classifier.py) and a list of ModelConfigs, and produces a routing decision plus
running statistics. serving.py is what actually executes generation on the
ModelConfig this layer picks.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.router.classifier import (
    EmbeddingClassifier,
    FeatureExtractor,
    RequestComplexity,
)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------
#
# One entry per servable model. `cost_per_token` is a RELATIVE weight, not a
# dollar figure -- what matters to the router is the ratio between models (the
# small model might be 1.0, the large 5.0), which is what drives both the
# "pick the cheapest capable model" decision and the cost-savings accounting.
#
# `max_complexity` is the most complex request this model is trusted to serve.
# The routing invariant is: a model may serve a request iff
# `model.max_complexity >= request_complexity`. So a model with
# max_complexity=MODERATE handles SIMPLE and MODERATE but not COMPLEX.
#
# `avg_latency_ms` is mutable, runtime-tracked state, not a static config value.
# It starts at 0.0 ("never measured") and record_outcome() folds in each real
# latency via an exponential moving average. It lives on the config object so a
# single ModelConfig instance carries both the static policy inputs and the
# live measured behaviour -- the router holds references to these same objects.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    model_path: str
    cost_per_token: float
    max_complexity: RequestComplexity
    # Runtime-tracked. 0.0 means "no outcome recorded yet". field(default=...)
    # because it is measured, not supplied at construction.
    avg_latency_ms: float = field(default=0.0)


# ---------------------------------------------------------------------------
# Default two-model fleet.
# ---------------------------------------------------------------------------
#
# A small/cheap model that handles everything up to MODERATE, and a large/
# expensive model that additionally handles COMPLEX. With these two:
#
#   SIMPLE   -> small (cheapest capable)
#   MODERATE -> small (cheapest capable; small.max_complexity == MODERATE)
#   COMPLEX  -> large (small can't, so the cheapest *capable* model is large)
#
# The 5x cost ratio is what makes the savings real: every SIMPLE/MODERATE
# request that avoids the large model saves 4 cost units.
#
# Both names ("small"/"large") are the keys serving.MultiModelServer uses to
# select generation parameters, so the two modules agree on the fleet by name.
# ---------------------------------------------------------------------------


def default_model_configs() -> list[ModelConfig]:
    """The standard small+large fleet used by the server and benchmark."""
    return [
        ModelConfig(
            name="small",
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cost_per_token=1.0,
            max_complexity=RequestComplexity.MODERATE,
        ),
        ModelConfig(
            name="large",
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cost_per_token=5.0,
            max_complexity=RequestComplexity.COMPLEX,
        ),
    ]


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------
#
# Holds the fleet + a classifier and turns prompts into model choices, while
# accumulating the statistics an operator needs to see the policy is paying off:
# how traffic split across models, measured per-model latency, and the headline
# number -- how much cheaper this is than naively sending everything to the big
# model.
# ---------------------------------------------------------------------------


# EMA smoothing factor for latency. alpha=0.1 weights the newest sample at 10%
# and the running estimate at 90% -- a ~10-sample memory. Small enough to ride
# out a single slow request, responsive enough to track a real regime change.
_LATENCY_EMA_ALPHA = 0.1


class LLMRouter:
    def __init__(self, configs: list[ModelConfig], classifier) -> None:
        if not configs:
            raise ValueError("LLMRouter needs at least one ModelConfig")
        self.configs = configs
        self.classifier = classifier

        # ---- CARL live-tunable knobs (advisory) -------------------------
        # These two attributes exist so the CARL controller has explicit, named
        # router parameters to drive at runtime (its _apply only writes knobs a
        # component actually declares). They are deliberately additive and do
        # NOT alter the default complexity-based routing policy below:
        #   * routing_threshold   -- a complexity-score cutoff a future
        #     score-based routing path can consult; recorded here so CARL can
        #     adapt it per regime without the router needing a code change.
        #   * cache_affinity_weight -- how strongly a cache-affinity-aware
        #     variant should bias toward a model whose KV the prefix cache
        #     already holds. 0.0 == ignore affinity (the current behaviour).
        # Both start at neutral values, so the existing routing tests are
        # unaffected; CARL overwrites them live.
        self.routing_threshold = 0.5
        self.cache_affinity_weight = 0.0
        # FeatureExtractor for the rule-based classifier path. The embedding
        # classifier takes the raw prompt and does its own extraction, so this
        # is only used when the classifier consumes a feature dict.
        self.extractor = FeatureExtractor()

        # By-name lookup so record_outcome can find a config in O(1).
        self._by_name = {c.name: c for c in configs}

        # The single most capable model: the one we'd use if we ALWAYS used the
        # biggest model. Tie-break by cost so "most capable" means the genuinely
        # largest/priciest. This is both the cost-savings baseline and the
        # fallback target when no model is nominally capable of a request.
        self._largest = max(configs, key=lambda c: (c.max_complexity, c.cost_per_token))

        # Running stats.
        self._requests_per_model: dict[str, int] = defaultdict(int)
        self._complexity_counts: dict[RequestComplexity, int] = defaultdict(int)
        self._total_requests = 0
        # EMA needs to seed on the first sample rather than blend against the
        # 0.0 sentinel; track which models have at least one recorded outcome.
        self._latency_seen: set[str] = set()

    # -- classification helper --------------------------------------------

    def _classify(self, prompt: str) -> RequestComplexity:
        """Run whichever classifier we hold and return a RequestComplexity.

        Supports both classifier shapes uniformly:
          * EmbeddingClassifier.classify(prompt) -> (complexity, confidence)
          * RuleBasedClassifier.classify(features) -> complexity
        """
        if isinstance(self.classifier, EmbeddingClassifier):
            complexity, _confidence = self.classifier.classify(prompt)
            return complexity
        # Rule-based (or any duck-typed classifier consuming a feature dict).
        features = self.extractor.extract(prompt)
        return self.classifier.classify(features)

    # -- routing -----------------------------------------------------------

    def route(self, prompt: str) -> ModelConfig:
        """Classify `prompt` and return the ModelConfig that should serve it.

        Policy:
          1. Classify the prompt.
          2. Among models with max_complexity >= the request complexity, pick the
             cheapest (lowest cost_per_token).
          3. If NO model is nominally capable, fall back to the most capable
             model -- better to over-serve than to drop the request. (This is
             also the "all busy" fallback hook: a capacity-aware extension would
             treat a saturated cheap model as ineligible and land here.)
        """
        complexity = self._classify(prompt)

        capable = [c for c in self.configs if c.max_complexity >= complexity]
        if capable:
            # Cheapest capable model. Tie-break by max_complexity so equal-cost
            # models prefer the less-capable (more specialised) one.
            chosen = min(capable, key=lambda c: (c.cost_per_token, c.max_complexity))
        else:
            # Nothing is rated for this complexity -> escalate to the biggest
            # model we have rather than fail.
            chosen = self._largest

        # Bookkeeping for routing_stats().
        self._requests_per_model[chosen.name] += 1
        self._complexity_counts[complexity] += 1
        self._total_requests += 1
        return chosen

    # -- outcome tracking --------------------------------------------------

    def record_outcome(
        self, model_name: str, latency_ms: float, tokens_generated: int
    ) -> None:
        """Fold a realised generation outcome into the model's latency estimate.

        Uses an exponential moving average (alpha=0.1) so avg_latency_ms tracks
        recent behaviour without being whipsawed by any single request. The
        FIRST sample for a model seeds the average directly -- blending against
        the 0.0 "never measured" sentinel would otherwise drag the first
        estimate to a tenth of the true value.

        `tokens_generated` is accepted as part of the outcome contract (a
        latency-per-token extension would use it); the EMA here is over total
        request latency.
        """
        config = self._by_name.get(model_name)
        if config is None:
            raise KeyError(f"unknown model {model_name!r}")

        if model_name not in self._latency_seen:
            config.avg_latency_ms = latency_ms
            self._latency_seen.add(model_name)
        else:
            a = _LATENCY_EMA_ALPHA
            config.avg_latency_ms = a * latency_ms + (1.0 - a) * config.avg_latency_ms

    # -- statistics --------------------------------------------------------

    def routing_stats(self) -> dict:
        """Snapshot of routing behaviour and the cost win.

        Returns:
            requests_per_model      -- {name: count} routing decisions so far.
            avg_latency_per_model   -- {name: EMA latency ms} (0.0 if unmeasured).
            cost_savings_pct        -- % cheaper than always using the largest
                                       model, on a per-request cost-weight basis.
            complexity_distribution -- {complexity_name: count}.
        """
        requests_per_model = {c.name: self._requests_per_model[c.name] for c in self.configs}
        avg_latency_per_model = {c.name: c.avg_latency_ms for c in self.configs}

        # Cost accounting on a per-request basis using cost_per_token as the
        # relative weight. actual = what we spent routing each request to its
        # chosen model; baseline = what we'd have spent sending every request to
        # the largest model. Savings is the gap as a percentage of baseline.
        actual_cost = sum(
            self._requests_per_model[c.name] * c.cost_per_token for c in self.configs
        )
        baseline_cost = self._total_requests * self._largest.cost_per_token
        if baseline_cost > 0:
            cost_savings_pct = (baseline_cost - actual_cost) / baseline_cost * 100.0
        else:
            cost_savings_pct = 0.0

        # Use .name keys so the distribution is JSON-serialisable and stable
        # across processes (enum identity isn't).
        complexity_distribution = {
            level.name: self._complexity_counts[level] for level in RequestComplexity
        }

        return {
            "requests_per_model": requests_per_model,
            "avg_latency_per_model": avg_latency_per_model,
            "cost_savings_pct": cost_savings_pct,
            "complexity_distribution": complexity_distribution,
        }
