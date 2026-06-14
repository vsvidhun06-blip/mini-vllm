"""
Benchmark: the LLM router on a 1000-prompt workload.

Run:
    python scripts/benchmark_router.py                 # synthetic (default)
    python scripts/benchmark_router.py --synthetic
    python scripts/benchmark_router.py --real          # real LMSYS-Chat-1M prompts
    python scripts/benchmark_router.py --real --limit 1000

Two workload sources, identical metrics:

  --synthetic  1000 template-generated prompts (500 SIMPLE + 300 MODERATE +
               200 COMPLEX) whose intended level is known, so we can also report
               an intended-vs-classified match rate. Deterministic, offline,
               and what CI runs.
  --real       The first N rows of LMSYS-Chat-1M (https://huggingface.co/
               datasets/lmsys/lmsys-chat-1m), a corpus of real user/assistant
               conversations. We take the first USER turn of each conversation
               as the prompt. This shows how the router behaves on genuine
               traffic rather than tidy templates -- real prompts are messier,
               multilingual, and not pre-labelled, so there's no match rate to
               report, only the routing behaviour itself.

This benchmark measures the ROUTING layer only -- no model is loaded. The
router's value proposition is "spend microseconds deciding so you can save
dollars serving", so the three things worth measuring (for either source) are:

  1. Where does traffic land?       (routing decision distribution, % per model)
  2. How much cheaper is it?         (cost savings vs always using the large model)
  3. What does the decision cost?    (classification + selection: P50/P95/P99)

LMSYS-Chat-1M is a *gated* dataset: you must accept its conditions on the
Hugging Face page and authenticate (set HF_TOKEN, or run `huggingface-cli
login`) before --real can stream it. We stream rather than download the full
multi-GB corpus, so only the first shard is fetched.
"""
from __future__ import annotations

import argparse
import time

from src.router.classifier import RequestComplexity, RuleBasedClassifier
from src.router.router import LLMRouter, default_model_configs

# Synthetic workload mix: (intended level, count). Weighted toward easy traffic
# -- the regime where routing pays off, and representative of real assistant
# traffic (most requests are easy).
N_SIMPLE = 500
N_MODERATE = 300
N_COMPLEX = 200

# Real-workload source.
LMSYS_DATASET = "lmsys/lmsys-chat-1m"
DEFAULT_LIMIT = 1000


# ---------------------------------------------------------------------------
# Synthetic prompt generation.
# ---------------------------------------------------------------------------
#
# Each template is crafted to classify to its intended complexity level under
# the rule-based classifier (short+factual -> SIMPLE, multi-question-word ->
# MODERATE, code/math -> COMPLEX). We cycle through fillers per template so the
# 1000 prompts aren't 1000 identical strings.

_SIMPLE_TEMPLATES = [
    "What is the capital of {x}?",
    "Who wrote {x}?",
    "When did {x} happen?",
    "Where is {x} located?",
    "Which year was {x} founded?",
]
_SIMPLE_FILL = ["France", "Hamlet", "the moon landing", "Mount Everest",
                "the company", "Japan", "Rome", "the treaty"]

_MODERATE_TEMPLATES = [
    "Explain the difference between {a} and {b} with examples",
    "Compare and contrast {a} versus {b}",
    "Describe how and why {a} differs from {b}",
    "Analyse and explain the trade-offs between {a} and {b}",
]
_MODERATE_FILL = [("TCP", "UDP"), ("threads", "processes"), ("SQL", "NoSQL"),
                  ("REST", "GraphQL"), ("stacks", "queues"), ("HTTP", "HTTPS")]

_COMPLEX_TEMPLATES = [
    "Write a Python function that implements {x}",
    "Implement a class that performs {x}",
    "def solve(): return {x}  # complete this code",
    "Derive the equation for {x} and compute 3 * x = 12",
]
_COMPLEX_FILL = ["quicksort", "a binary search tree", "an LRU cache",
                 "matrix multiplication", "Dijkstra's algorithm", "a hash map"]


def build_synthetic_workload() -> list[tuple[str, RequestComplexity | None]]:
    """Materialise the 1000-prompt synthetic workload as (prompt, intended_level)."""
    prompts: list[tuple[str, RequestComplexity | None]] = []

    for i in range(N_SIMPLE):
        t = _SIMPLE_TEMPLATES[i % len(_SIMPLE_TEMPLATES)]
        prompts.append((t.format(x=_SIMPLE_FILL[i % len(_SIMPLE_FILL)]),
                        RequestComplexity.SIMPLE))

    for i in range(N_MODERATE):
        t = _MODERATE_TEMPLATES[i % len(_MODERATE_TEMPLATES)]
        a, b = _MODERATE_FILL[i % len(_MODERATE_FILL)]
        prompts.append((t.format(a=a, b=b), RequestComplexity.MODERATE))

    for i in range(N_COMPLEX):
        t = _COMPLEX_TEMPLATES[i % len(_COMPLEX_TEMPLATES)]
        prompts.append((t.format(x=_COMPLEX_FILL[i % len(_COMPLEX_FILL)]),
                        RequestComplexity.COMPLEX))

    return prompts


# ---------------------------------------------------------------------------
# Real prompts: LMSYS-Chat-1M.
# ---------------------------------------------------------------------------
#
# We stream the dataset (streaming=True) so only the first shard is fetched, not
# the whole multi-GB corpus, and pull the first USER turn from each of the first
# `limit` conversations. The dataset is gated, so a missing/invalid token
# surfaces as a clear, actionable error rather than a stack trace.


def load_real_prompts(limit: int = DEFAULT_LIMIT) -> list[tuple[str, None]]:
    """Stream the first `limit` user prompts from LMSYS-Chat-1M.

    Returns (prompt, None) pairs -- None because real traffic has no intended
    complexity label. Raises a RuntimeError with setup guidance if the dataset
    can't be loaded (missing `datasets`, or gated access without a token).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the `datasets` package is required for --real. "
            "Install it with `pip install datasets`."
        ) from exc

    try:
        # streaming=True avoids materialising the full dataset; we only touch
        # the first shard to read `limit` rows.
        stream = load_dataset(LMSYS_DATASET, split="train", streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"could not open {LMSYS_DATASET}. It is a GATED dataset: accept its "
            f"conditions at https://huggingface.co/datasets/{LMSYS_DATASET} and "
            "authenticate (set the HF_TOKEN env var or run `huggingface-cli "
            f"login`) before using --real.\nUnderlying error: {exc}"
        ) from exc

    prompts: list[tuple[str, None]] = []
    for row in stream:
        if len(prompts) >= limit:
            break
        text = _first_user_turn(row)
        if text:
            prompts.append((text, None))

    if not prompts:
        raise RuntimeError(
            f"streamed {LMSYS_DATASET} but extracted no user prompts -- the "
            "schema may have changed (expected a 'conversation' list of "
            "role/content turns)."
        )
    return prompts


def _first_user_turn(row: dict) -> str | None:
    """Pull the first user message from an LMSYS-Chat-1M conversation row.

    Each row carries a `conversation` list of {"role", "content"} turns. We take
    the first turn whose role is "user" (a.k.a. "human" in some exports).
    """
    conversation = row.get("conversation")
    if not isinstance(conversation, list):
        return None
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") in ("user", "human"):
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (matches the style used in the other benchmarks)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def run_benchmark(
    workload: list[tuple[str, RequestComplexity | None]], source: str
) -> None:
    """Route every prompt, then print routing distribution, savings, and latency."""
    router = LLMRouter(default_model_configs(), RuleBasedClassifier())
    total = len(workload)
    print(f"Routing {total} prompts  [source: {source}]\n")

    # Route every prompt, timing ONLY the route() call (classify + select).
    overheads_us: list[float] = []
    labelled = 0          # how many prompts carry an intended label (synthetic)
    matches = 0           # of those, how many the classifier agreed with
    for prompt, intended in workload:
        t0 = time.perf_counter()
        router.route(prompt)
        overheads_us.append((time.perf_counter() - t0) * 1e6)
        if intended is not None:
            labelled += 1
            # route() doesn't return the complexity, so re-derive it cheaply.
            if router._classify(prompt) is intended:
                matches += 1

    stats = router.routing_stats()

    # -- Routing decision distribution ------------------------------------
    print("Routing decisions (% of traffic per model):")
    print(f"  {'model':>8} | {'requests':>9} | {'share':>7} | {'cost/tok':>8} | {'max cx':>8}")
    print("  " + "-" * 52)
    for cfg in router.configs:
        n = stats["requests_per_model"][cfg.name]
        print(f"  {cfg.name:>8} | {n:>9} | {n / total:>6.1%} | "
              f"{cfg.cost_per_token:>8.1f} | {cfg.max_complexity.name:>8}")

    # -- Complexity distribution ------------------------------------------
    print("\nComplexity distribution (classifier output):")
    for level, n in stats["complexity_distribution"].items():
        print(f"  {level:>8}: {n:>4} ({n / total:>5.1%})")
    if labelled:
        print(f"  intended-vs-classified match rate: {matches / labelled:.1%}")

    # -- Cost savings -----------------------------------------------------
    print("\nCost (relative weight; baseline = always-large):")
    print(f"  cost savings vs always-large model: {stats['cost_savings_pct']:.1f}%")

    # -- Routing overhead -------------------------------------------------
    print("\nClassification + selection latency (per request):")
    print(f"  P50: {_percentile(overheads_us, 50):>7.2f} us")
    print(f"  P95: {_percentile(overheads_us, 95):>7.2f} us")
    print(f"  P99: {_percentile(overheads_us, 99):>7.2f} us")
    print(f"  mean: {sum(overheads_us) / len(overheads_us):>6.2f} us")

    print("\nTakeaway: the small model absorbs the easy majority, the large "
          "model is\nreserved for COMPLEX work, and the routing decision costs "
          "microseconds --\norders of magnitude below any model's per-token "
          "latency.")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the LLM router.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--synthetic", action="store_true",
        help="use template-generated prompts with known labels (default)",
    )
    mode.add_argument(
        "--real", action="store_true",
        help="use real user prompts from LMSYS-Chat-1M (gated; needs HF auth)",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"number of real prompts to stream (default {DEFAULT_LIMIT}; --real only)",
    )
    args = parser.parse_args()

    if args.real:
        workload = load_real_prompts(limit=args.limit)
        source = f"LMSYS-Chat-1M (first {len(workload)} user prompts)"
    else:
        workload = build_synthetic_workload()
        source = "synthetic templates"

    run_benchmark(workload, source)


if __name__ == "__main__":
    main()
