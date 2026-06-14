"""
LLM Router: classify incoming requests and route them to the cheapest model
capable of serving them.

The package mirrors the layering production inference stacks use:

  * classifier.py -- turn a raw prompt into a RequestComplexity. Pure Python,
                     no torch, no model -- so it runs in microseconds and is
                     trivially unit-testable on CPU/CI.
  * router.py     -- the policy layer. Given a complexity + a fleet of
                     ModelConfigs, pick the cheapest model that can handle the
                     request, track per-model latency, and report cost savings.
  * serving.py    -- the execution layer. Holds the actual LlamaModel
                     instances, lazy-loads them, and runs generation under the
                     router's decision.
  * api.py        -- the HTTP surface (FastAPI): /route/generate, /route/stats,
                     /route/models, /route/generate/stream.

The split exists so the *decision* (classify + route) is fully testable
without ever loading a 1.1 B-parameter model -- exactly how you'd want to
unit-test routing logic in production.
"""
