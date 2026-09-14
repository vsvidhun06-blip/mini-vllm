"""
The joint configuration space.

A serving engine has knobs scattered across five subsystems. Tuned
independently (one knob per bottleneck, as the AutoTuner does) you can never
express a JOINT setting like "small batch AND deep speculation AND gentle
eviction" that is only good as a combination. CARLConfig is the unified vector:
ONE dataclass that names every knob CARL controls, so a single bandit decision
sets a coherent operating point across the whole engine at once.

Two things live here:

  * CARLConfig          -- the dataclass + range clamping. Every field documents
                           its valid range; clamp() projects any candidate back
                           into the box so a hand-typed or perturbed config can
                           never drive an out-of-range parameter into the engine.
  * DEFAULT_CONFIGS     -- a hand-tuned starting config per regime. These encode
                           domain knowledge (what a human would pick for each
                           regime) and serve as the bandit's warm start: arm 0
                           for every regime is its default, so CARL is never
                           worse than the hand-tuned baseline before it learns.
  * config_arms()       -- the 5-6 DISCRETE configs the per-regime bandit chooses
                           among. A contextual bandit needs a finite arm set;
                           we build it by perturbing the regime's default along
                           the one or two knobs that matter most for that regime.

Torch-free: pure dataclasses + arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from src.carl.state import WorkloadRegime


# ---------------------------------------------------------------------------
# Parameter ranges (single source of truth for clamping AND documentation).
# ---------------------------------------------------------------------------
#
# (low, high) inclusive bounds per numeric knob. clamp() reads this so the box
# is defined in exactly one place; if a range changes, clamping follows.
# ---------------------------------------------------------------------------

_RANGES = {
    "max_batch_size": (1, 32),
    "chunk_size": (64, 512),
    "spec_k": (0, 8),
    "routing_threshold": (0.0, 1.0),
    "cache_affinity_weight": (0.0, 1.0),
    "eviction_threshold": (0.5, 0.95),
    "eviction_window": (16, 64),
}


def _clamp(value, lo, hi):
    """Clamp a scalar into [lo, hi], preserving int-ness of integer knobs."""
    if value < lo:
        value = lo
    elif value > hi:
        value = hi
    return value


@dataclass
class CARLConfig:
    """One coherent operating point spanning all five subsystems.

    Fields map 1:1 onto live engine attributes the controller mutates:
      max_batch_size, chunk_size, preemption_enabled  -> scheduler
      spec_k                                           -> spec decoder / scheduler
      routing_threshold, cache_affinity_weight         -> router
      eviction_threshold, eviction_window              -> KV cache
      use_cuda_graphs                                  -> scheduler
    """

    # -- Scheduler --
    max_batch_size: int = 8         # [1, 32]   rows per forward pass
    chunk_size: int = 256           # [64, 512] chunked-prefill token budget
    preemption_enabled: bool = True

    # -- Speculative decoding --
    spec_k: int = 2                 # [0, 8]    draft length; 0 disables spec decode

    # -- Router --
    routing_threshold: float = 0.5      # [0, 1] complexity cutoff small->large
    cache_affinity_weight: float = 0.2  # [0, 1] weight of cache-hit bonus in routing

    # -- KV cache --
    eviction_threshold: float = 0.8     # [0.5, 0.95] occupancy that triggers eviction
    eviction_window: int = 32           # [16, 64]    H2O recency window

    # -- CUDA graphs --
    use_cuda_graphs: bool = True

    def clamp(self) -> "CARLConfig":
        """Return a copy with every numeric field projected into its valid range.

        Defensive boundary: configs reach the engine from three sources -- the
        hand-tuned defaults (already valid), the bandit's discrete arms (built
        valid), and the /carl/config override endpoint (arbitrary user JSON).
        Clamping here means a bad override can never push, say, max_batch_size=999
        or a negative spec_k into the live scheduler.
        """
        return CARLConfig(
            max_batch_size=int(_clamp(self.max_batch_size, *_RANGES["max_batch_size"])),
            chunk_size=int(_clamp(self.chunk_size, *_RANGES["chunk_size"])),
            preemption_enabled=bool(self.preemption_enabled),
            spec_k=int(_clamp(self.spec_k, *_RANGES["spec_k"])),
            routing_threshold=float(_clamp(self.routing_threshold, *_RANGES["routing_threshold"])),
            cache_affinity_weight=float(
                _clamp(self.cache_affinity_weight, *_RANGES["cache_affinity_weight"])
            ),
            eviction_threshold=float(
                _clamp(self.eviction_threshold, *_RANGES["eviction_threshold"])
            ),
            eviction_window=int(_clamp(self.eviction_window, *_RANGES["eviction_window"])),
            use_cuda_graphs=bool(self.use_cuda_graphs),
        )

    def as_dict(self) -> dict:
        """JSON-friendly view for /carl/config and the controller log."""
        return {
            "max_batch_size": self.max_batch_size,
            "chunk_size": self.chunk_size,
            "preemption_enabled": self.preemption_enabled,
            "spec_k": self.spec_k,
            "routing_threshold": self.routing_threshold,
            "cache_affinity_weight": self.cache_affinity_weight,
            "eviction_threshold": self.eviction_threshold,
            "eviction_window": self.eviction_window,
            "use_cuda_graphs": self.use_cuda_graphs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CARLConfig":
        """Build a (clamped) CARLConfig from a partial dict; unknown keys ignored.

        Used by POST /carl/config. Missing keys fall back to the dataclass
        defaults, so a caller can override a single knob for an ablation.
        """
        fields = {
            k: d[k] for k in (
                "max_batch_size", "chunk_size", "preemption_enabled", "spec_k",
                "routing_threshold", "cache_affinity_weight",
                "eviction_threshold", "eviction_window", "use_cuda_graphs",
            ) if k in d
        }
        return cls(**fields).clamp()


# ---------------------------------------------------------------------------
# DEFAULT_CONFIGS: the hand-tuned warm start, one per regime.
# ---------------------------------------------------------------------------
#
# These are the "regime oracle" a human expert would pick, and double as arm 0
# of each regime's bandit. The rationale per regime:
#
#   INTERACTIVE  -- latency first. Small batch keeps per-step decode latency low;
#                   light speculation (k=2) can shave TPOT when acceptance helps;
#                   gentle eviction (0.6) avoids compaction stalls mid-request;
#                   CUDA graphs on (launch overhead dominates tiny batches).
#   BATCH        -- throughput first. Max batch packs the GPU; k=4 amortises the
#                   verify pass over a long generation; aggressive eviction (0.9)
#                   keeps memory dense so admission never stalls; graphs on.
#   BURST        -- drain the backlog. Medium batch + aggressive eviction clears
#                   the queue; k=0 because a surge has many requests in flight,
#                   where batched spec decode falls back to vanilla anyway, so
#                   drafting is wasted work.
#   CACHE_HEAVY  -- exploit shared prefixes. High cache_affinity_weight biases
#                   routing toward the model whose KV the prefix cache already
#                   holds; medium batch; modest speculation.
#   LONG_CONTEXT -- respect memory. Small batch (long KV per request), large
#                   chunk to push the big prefill efficiently, gentle eviction
#                   (0.6) + wide recency window (64) so quality survives, no
#                   speculation (deep contexts make drafts miss).
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[WorkloadRegime, CARLConfig] = {
    WorkloadRegime.INTERACTIVE: CARLConfig(
        max_batch_size=4, chunk_size=128, preemption_enabled=True, spec_k=2,
        routing_threshold=0.5, cache_affinity_weight=0.2,
        eviction_threshold=0.6, eviction_window=32, use_cuda_graphs=True,
    ),
    WorkloadRegime.BATCH: CARLConfig(
        max_batch_size=32, chunk_size=512, preemption_enabled=False, spec_k=4,
        routing_threshold=0.6, cache_affinity_weight=0.2,
        eviction_threshold=0.9, eviction_window=16, use_cuda_graphs=True,
    ),
    WorkloadRegime.BURST: CARLConfig(
        max_batch_size=16, chunk_size=256, preemption_enabled=True, spec_k=0,
        routing_threshold=0.5, cache_affinity_weight=0.2,
        eviction_threshold=0.9, eviction_window=16, use_cuda_graphs=True,
    ),
    WorkloadRegime.CACHE_HEAVY: CARLConfig(
        max_batch_size=16, chunk_size=256, preemption_enabled=True, spec_k=2,
        routing_threshold=0.5, cache_affinity_weight=0.8,
        eviction_threshold=0.8, eviction_window=32, use_cuda_graphs=True,
    ),
    WorkloadRegime.LONG_CONTEXT: CARLConfig(
        max_batch_size=4, chunk_size=512, preemption_enabled=True, spec_k=0,
        routing_threshold=0.5, cache_affinity_weight=0.3,
        eviction_threshold=0.6, eviction_window=64, use_cuda_graphs=True,
    ),
}


def default_config_for(regime: WorkloadRegime) -> CARLConfig:
    """The hand-tuned default for a regime (the regime-oracle baseline uses this)."""
    return DEFAULT_CONFIGS[regime]


# ---------------------------------------------------------------------------
# config_arms: the discrete arm set per regime.
# ---------------------------------------------------------------------------
#
# A contextual bandit picks among a FINITE set of arms. We build 5-6 arms per
# regime: arm 0 is the hand-tuned default (warm start), and the rest perturb the
# one or two knobs that matter most for that regime, so the bandit explores the
# locally relevant axis rather than the whole 9-D space (which would need far
# more samples than a short workload provides). Every arm is clamped on the way
# out, so perturbations can't escape the valid box.
# ---------------------------------------------------------------------------


def config_arms(regime: WorkloadRegime) -> list[CARLConfig]:
    """Return the discrete CARLConfig arms the bandit chooses among for `regime`.

    The arm set always leads with DEFAULT_CONFIGS[regime] (arm 0), then varies
    the regime-salient knobs. Length is 5-6 per the design.
    """
    base = DEFAULT_CONFIGS[regime]
    arms: list[CARLConfig] = [base]

    if regime is WorkloadRegime.INTERACTIVE:
        # Latency regime: explore batch size and speculation depth.
        arms += [
            replace(base, max_batch_size=2),                 # even smaller batch
            replace(base, max_batch_size=8),                 # a bit larger
            replace(base, spec_k=0),                         # speculation off
            replace(base, spec_k=4),                         # deeper speculation
            replace(base, max_batch_size=8, spec_k=4),       # combined
        ]
    elif regime is WorkloadRegime.BATCH:
        # Throughput regime: explore batch size and eviction aggressiveness.
        arms += [
            replace(base, max_batch_size=16),
            replace(base, max_batch_size=24, eviction_threshold=0.85),
            replace(base, spec_k=2),                         # lighter speculation
            replace(base, chunk_size=384),
            replace(base, eviction_threshold=0.95),          # most aggressive
        ]
    elif regime is WorkloadRegime.BURST:
        # Drain regime: explore batch size and eviction; keep speculation off.
        arms += [
            replace(base, max_batch_size=24),
            replace(base, max_batch_size=32),
            replace(base, eviction_threshold=0.85),
            replace(base, eviction_threshold=0.95, eviction_window=16),
            replace(base, chunk_size=512),                   # fatter prefill chunks
        ]
    elif regime is WorkloadRegime.CACHE_HEAVY:
        # Prefix regime: explore the cache-affinity weight and batch size.
        arms += [
            replace(base, cache_affinity_weight=0.6),
            replace(base, cache_affinity_weight=1.0),
            replace(base, max_batch_size=8),
            replace(base, max_batch_size=24),
            replace(base, spec_k=0),
        ]
    else:  # LONG_CONTEXT
        # Memory regime: explore eviction threshold/window and batch size.
        arms += [
            replace(base, eviction_threshold=0.5),
            replace(base, eviction_threshold=0.7),
            replace(base, eviction_window=48),
            replace(base, max_batch_size=2),
            replace(base, max_batch_size=8, chunk_size=384),
        ]

    return [a.clamp() for a in arms]


def all_arm_sets() -> dict[WorkloadRegime, list[CARLConfig]]:
    """The arm set for every regime -- what PerRegimeBandit is built from."""
    return {regime: config_arms(regime) for regime in WorkloadRegime}
