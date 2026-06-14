"""
CARL: Coordinated Adaptive Runtime for LLM serving.

The research question this package answers in code: can a single online
controller that JOINTLY adapts every serving knob (batch size, chunk size,
speculative-decode depth, routing threshold, KV eviction) beat the existing
per-component AutoTuner under non-stationary workloads?

Layering (each module depends only on the ones above it):

  state.py       RuntimeState feature observer + WorkloadRegime classifier.
                 Reads the live engine (scheduler / spec decoder / router /
                 KV cache) DEFENSIVELY -- it never crashes if a component is
                 missing an attribute, so it works against stubs in tests and
                 the real engine in the server.
  config.py      CARLConfig: the joint configuration space (one dataclass that
                 spans all five subsystems) + hand-tuned DEFAULT_CONFIGS per
                 regime + the discrete arm sets the bandit chooses among.
  bandit.py      LinUCB / Thompson contextual bandits over CARLConfig arms,
                 plus the utility() reward. Pure numpy -- no torch.
  controller.py  CARLController: ties it together. Every `observe_interval`
                 scheduler steps it observes -> classifies -> selects a config
                 -> applies it live (thread-safe) -> rewards the PREVIOUS
                 choice. Plugs into src/server/api.py via an optional param.
  api.py         FastAPI routes (/carl/state, /stats, /log, /config, /reset).

The whole package is import-time torch-free so it (and its tests + the
benchmark) run on a CPU-only box; the only third-party dependency is numpy for
the bandit's small per-arm linear algebra.
"""
