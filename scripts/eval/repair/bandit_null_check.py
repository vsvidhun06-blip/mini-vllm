"""PHASE 1 -- Is the LinUCB bandit doing anything at all?

THE QUESTION
------------
The hostile cold review alleged that CARL's contextual bandit is inert: that at
the paper's own alpha=0.5 it never leaves arm 0, that arm 0 is exactly
DEFAULT_CONFIGS[regime], and that CARL is therefore behaviourally identical to a
zero-learning rule-based controller:

    RULE_ONLY(state) = DEFAULT_CONFIGS[classify_regime(state)]

If true, every "online learning" claim in the paper is a claim about a five-branch
if/elif chain, and the LinUCB machinery is decoration.

This script does not inspect source code and reason about it. It EXECUTES the
shipped harness and records what actually happens, decision by decision.

WHAT IT MEASURES
----------------
For each alpha in a sweep, over the existing NON-STATIONARY workload and several
seeds, it drives the real CarlAgent through the real _harness.run_once and logs:

  * every selected arm, per regime
  * per-arm selection counts
  * the reward at every cycle
  * the context vector and its L2 norm
  * the exploit term (theta^T x) and the explore bonus (alpha*sqrt(x^T A^-1 x))
    for EVERY arm at EVERY decision -- so we can see precisely why argmax lands
    where it does
  * final throughput

and then runs, over the SAME seeds and the SAME workload, three references:

  * RULE_ONLY  -- classify, look up DEFAULT_CONFIGS, no learning, no state
  * Oracle     -- DEFAULT_CONFIGS[TRUE regime] (perfect regime knowledge)
  * Static-Best-- the harness's own single-best fixed config

and reports exact-equality checks between them.

READ THE OUTPUT THIS WAY
------------------------
If carl_equals_rule_only is true at the paper's alpha, the bandit contributes
nothing on this substrate and the paper's central mechanism is unsupported.
That is a finding, not a bug to be tuned away.

CPU-only, seconds to run, no GPU and no model weights.

Run:
  python scripts/eval/repair/bandit_null_check.py
  python scripts/eval/repair/bandit_null_check.py --alphas 0.5 --seeds 42
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# --- path bootstrap ---------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))            # scripts/eval/repair
_EVAL = os.path.dirname(_THIS)                                 # scripts/eval
_SCRIPTS = os.path.dirname(_EVAL)                              # scripts
_ROOT = os.path.dirname(_SCRIPTS)                              # repo root
for _p in (_ROOT, _SCRIPTS, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_carl as bc  # noqa: E402
from _harness import (  # noqa: E402
    best_static_config, make_agent, nonstationary, run_once, slo_ttft_only,
)
from src.carl.bandit import LinUCBBandit  # noqa: E402
from src.carl.config import DEFAULT_CONFIGS  # noqa: E402
from src.carl.state import classify_regime  # noqa: E402
from src.eval.provenance import write_result  # noqa: E402

OUT_PATH = os.path.join(_ROOT, "docs", "eval", "raw", "repair",
                        "bandit_null_check.json")

DEFAULT_ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
PAPER_ALPHA = 0.5


# ---------------------------------------------------------------------------
# RULE_ONLY: the null hypothesis made executable.
# ---------------------------------------------------------------------------


class RuleOnlyAgent:
    """Classify the observed state, look up the hand-tuned config. That's it.

    No bandit, no arm indices, no learned statistics, no exploration. This is
    exactly what CARL degenerates to if LinUCB always selects arm 0, because
    arm 0 of every regime IS DEFAULT_CONFIGS[regime] (see config.config_arms).

    It observes the SAME state CARL observes -- the noisy synthesised state, not
    the true regime label -- so it is subject to identical classifier error.
    That matters: any gap between RULE_ONLY and Oracle is classifier error, and
    any gap between CARL and RULE_ONLY is what learning actually bought.
    """

    name = "RuleOnly"

    def __init__(self) -> None:
        self.adaptations = 0
        self._last = None
        self.regimes_seen: list[str] = []

    def choose(self, true_regime, state):
        regime = classify_regime(state)
        self.regimes_seen.append(regime.value)
        cfg = DEFAULT_CONFIGS[regime]
        if self._last is not None and cfg != self._last:
            self.adaptations += 1
        self._last = cfg
        return cfg

    def note_realised(self, metrics: dict) -> None:
        # Stateless by construction: it has nothing to learn from.
        pass


# ---------------------------------------------------------------------------
# Instrumented CARL: capture the UCB algebra at every decision.
# ---------------------------------------------------------------------------


class _TracingLinUCB(LinUCBBandit):
    """LinUCB that records exploit / explore / total per arm on every select().

    Pure observation: it calls the parent implementation for the actual choice,
    so the traced run is behaviourally identical to an untraced one.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.trace: list[dict] = []

    def select(self, context) -> int:
        x = self._context(context)
        exploit, explore = [], []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            exploit.append(float(theta @ x))
            explore.append(self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x))))
        chosen = super().select(context)
        self.trace.append({
            "chosen_arm": chosen,
            "context_l2": float(np.linalg.norm(x)),
            "exploit": [round(v, 9) for v in exploit],
            "explore": [round(v, 9) for v in explore],
            "ucb": [round(e + x_, 9) for e, x_ in zip(exploit, explore)],
        })
        return chosen


def _carl_with_tracing(alpha: float, slo):
    """A CarlAgent whose per-regime bandits are all _TracingLinUCB."""
    agent = bc.CarlAgent(_TracingLinUCB, "CARL-Full", slo, alpha=alpha)
    return agent


def _collect(agent) -> dict:
    """Pull selection counts, decision log and UCB traces off a CarlAgent."""
    bandit = agent.controller.bandit
    log = agent.controller.controller_log
    traces = {}
    for regime, b in bandit.bandits.items():
        if getattr(b, "trace", None):
            traces[regime.value] = b.trace
    return {
        "selection_counts": bandit.selection_counts(),
        "total_adaptations": agent.controller._total_adaptations,
        "config_distribution": agent.controller.stats()["config_distribution"],
        "rewards": [round(e.reward, 9) for e in log],
        "regimes": [e.regime.value for e in log],
        "context_l2": [round(float(np.linalg.norm(e.state_features)), 6) for e in log],
        "ucb_trace_by_regime": traces,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--requests", type=int, default=40,
                    help="total requests; split half INTERACTIVE / half BATCH")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.replace(",", " ").split()]
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    half = args.requests // 2

    slo = slo_ttft_only()
    regimes = nonstationary(half, args.requests - half)
    static_cfg = best_static_config(regimes, slo, seed=0)

    print(f"Workload: NON-STATIONARY {half} INTERACTIVE -> {args.requests - half} BATCH")
    print(f"Static-Best resolved to: {static_cfg.as_dict()}")
    print(f"Static-Best is a DEFAULT_CONFIGS entry: "
          f"{static_cfg in list(DEFAULT_CONFIGS.values())}\n")

    per_seed: dict[str, dict] = {}
    for seed in seeds:
        entry: dict = {}

        # Reference agents (identical workload, identical seed).
        rule = RuleOnlyAgent()
        m_rule = run_once(rule, regimes, slo, seed)
        m_oracle = run_once(make_agent("Oracle", slo), regimes, slo, seed)
        m_static = run_once(make_agent("Static-Best", slo, static_best_cfg=static_cfg),
                            regimes, slo, seed)
        entry["rule_only"] = {
            "throughput": m_rule["throughput"], "slo_sat": m_rule["slo_sat"],
            "ttft_p99": m_rule["ttft_p99"], "adaptations": rule.adaptations,
        }
        entry["oracle"] = {"throughput": m_oracle["throughput"],
                           "slo_sat": m_oracle["slo_sat"]}
        entry["static_best"] = {"throughput": m_static["throughput"],
                                "slo_sat": m_static["slo_sat"],
                                "ttft_p99": m_static["ttft_p99"]}

        entry["carl_by_alpha"] = {}
        for alpha in alphas:
            agent = _carl_with_tracing(alpha, slo)
            m = run_once(agent, regimes, slo, seed)
            collected = _collect(agent)
            counts = collected["selection_counts"]
            visited = {r: c for r, c in counts.items() if sum(c) > 0}
            only_arm0 = all(
                c[0] == sum(c) for c in visited.values()
            ) if visited else None
            entry["carl_by_alpha"][str(alpha)] = {
                "throughput": m["throughput"],
                "slo_sat": m["slo_sat"],
                "ttft_p99": m["ttft_p99"],
                # -- the decisive equality checks --
                "equals_rule_only_exact":
                    m["throughput"] == m_rule["throughput"],
                "equals_oracle_exact":
                    m["throughput"] == m_oracle["throughput"],
                "delta_vs_rule_only": m["throughput"] - m_rule["throughput"],
                "delta_vs_static_best": m["throughput"] - m_static["throughput"],
                "never_left_arm0": only_arm0,
                "n_distinct_arms_selected": sum(
                    1 for c in visited.values() for v in c if v > 0
                ),
                "reward_distinct_values": sorted(set(collected["rewards"])),
                "reward_variance": float(np.var(collected["rewards"])) if collected["rewards"] else None,
                **collected,
            }
            print(f"  seed {seed} alpha={alpha:<5} tps={m['throughput']:9.4f}  "
                  f"rule_only={m_rule['throughput']:9.4f}  "
                  f"equal={m['throughput'] == m_rule['throughput']}  "
                  f"arm0_only={only_arm0}")
        per_seed[str(seed)] = entry

    # ---- verdict at the paper's own alpha ---------------------------------
    pa = str(PAPER_ALPHA)
    equal_flags = [per_seed[str(s)]["carl_by_alpha"][pa]["equals_rule_only_exact"]
                   for s in seeds if pa in per_seed[str(s)]["carl_by_alpha"]]
    arm0_flags = [per_seed[str(s)]["carl_by_alpha"][pa]["never_left_arm0"]
                  for s in seeds if pa in per_seed[str(s)]["carl_by_alpha"]]

    verdict = {
        "paper_alpha": PAPER_ALPHA,
        "n_seeds": len(equal_flags),
        "carl_equals_rule_only_all_seeds": bool(equal_flags) and all(equal_flags),
        "carl_never_left_arm0_all_seeds": bool(arm0_flags) and all(arm0_flags),
        "static_best_is_a_default_config": static_cfg in list(DEFAULT_CONFIGS.values()),
        "interpretation": (
            "If carl_equals_rule_only_all_seeds is true, the LinUCB policy is "
            "behaviourally inert on this substrate at the paper's alpha: CARL's "
            "output is bit-identical to a stateless lookup of "
            "DEFAULT_CONFIGS[classify_regime(state)]. Every claim about online "
            "learning, convergence and regret is then a claim about the "
            "rule-based classifier, not about the bandit."
        ),
    }

    payload = {
        "description": (
            "PHASE 1 repair check: does LinUCB ever deviate from arm 0, and is "
            "CARL distinguishable from a zero-learning rule-only controller?"
        ),
        "substrate": "CONTROL-LOOP SIMULATION (scripts/benchmark_carl.WorkloadModel)",
        "workload": f"NON-STATIONARY {half} INTERACTIVE -> {args.requests - half} BATCH",
        "static_best_config": static_cfg.as_dict(),
        "alphas": alphas,
        "seeds": seeds,
        "verdict": verdict,
        "per_seed": per_seed,
    }
    path = write_result(args.out, payload, "bandit_null_check",
                        script="scripts/eval/repair/bandit_null_check.py",
                        extra={"alphas": alphas, "seeds": seeds,
                               "substrate": "simulation"})

    print("\n=== VERDICT (alpha = %.2f) ===" % PAPER_ALPHA)
    for k, v in verdict.items():
        if k != "interpretation":
            print(f"  {k}: {v}")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
