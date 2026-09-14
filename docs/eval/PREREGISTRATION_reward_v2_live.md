# Pre-registration — reward-v2 live experiment

**Written before the run. Not amended after.**

Experiment: `scripts/eval/repair/reward_v2_live.py`
Artifact it will produce: `docs/eval/reward_v2_live_results.json`
Registered: 2026-09-02, against repo state described in the run's `_provenance`.

---

## Why this experiment exists

The 2026-09-02 T4 run (`ablation_live_results.json`) recorded, in its
`dynoracle.per_arm_mean_reward` block:

```json
{"0": 0.29999999999999993, "1": 0.29999999999999993, "2": 0.29999999999999993, ...}
```

Every arm, in every regime, scored **exactly 0.3**. That is reward v1 fully
saturated:

```
0.3 = 0.3·(1.0)      throughput 96 tok/s clipped at the 50 tok/s reference
    + 0.3·(1 − 1.0)  TTFT p50 ≈ 21 s against a 200 ms SLO: 100% violated
    + 0.2·(1 − 1.0)  TPOT ≈ 275 ms against a 50 ms SLO: 100% violated
    + 0.2·(0.0)      no prefix reuse in this workload
```

Root cause: `src/carl/controller.py` imported `utility` from `src.carl.bandit`
(reward v1). `utility_v2` was imported only by `scripts/eval/repair/*`. The
Phase 3 reward repair reached the simulation harness and never reached the live
controller.

**Consequence.** In that run every learner was blind. The static-vs-adaptive
result stands — `Static-Best` is a fixed configuration and consumes no reward —
but the claim the paper needs, *"we repaired the learner and it still ties"*,
was **not tested on hardware**. This experiment tests it.

---

## What is held fixed, and what changes

| | Run A (have) | Run B (this) |
|---|---|---|
| artifact | `docs/eval/ablation_live_results.json` | `docs/eval/reward_v2_live_results.json` |
| **reward** | **v1 — `bandit.utility`, saturating** | **v2 — `reward.utility_v2`, frozen calibrated scales** |
| hardware | Tesla T4 | Tesla T4 |
| model | TinyLlama-1.1B-Chat-v1.0, fp16 | identical |
| seeds | 42–51 | identical |
| requests | 200/run | identical |
| workload | `ablation_live._build_workload`, NON-STATIONARY, flip at `n//2`, burst | identical (same function) |
| cadence | `OBSERVE_INTERVAL = 10` | identical |
| arm sets | restricted / expanded | identical (same functions) |
| learner | `RepairedLinUCBBandit`, α = 0.5 | identical |
| static search | LHS, 16 candidates, wide space, seed 999 | identical (same function) |
| context | `FEATURE_DIM = 10` | identical (unchanged) |

Run B calls Run A's own harness functions rather than reimplementing them, so
"identical" is a property of the code path, not of a promise.

---

## Hypotheses

### H1 — adaptation will not pay at this operating point

> CARL-Repaired will **not** materially outperform Static-Best at this saturated
> operating point, because the calibrated substrate predicts
> `max_batch_size = 32` as the throughput optimum and the Static-Best search
> selects it.

- **Comparison:** `CARL-Repaired vs Static-Best` (primary)
- **Prediction:** throughput delta ≤ 0, or within noise of 0.
- **Falsified if:** CARL-Repaired exceeds Static-Best by more than 2% with a
  paired 95% CI excluding 0.
- **Basis:** B7 (`load_sweep_b7.json`) finds the throughput-optimal
  `max_batch_size` pinned at 32 for all λ at and above saturation in 4/4
  shape×arm cells. Run A's static search independently selected 32.

### H2 — a non-degenerate reward should make the learner *do* something

> With a non-degenerate reward, CARL-Repaired should differ **measurably** from
> RuleOnly if the learner extracts useful information. Under reward v1 the two
> were behaviourally indistinguishable because the reward was constant.

- **Comparison:** `CARL-Repaired vs RuleOnly` (learning isolation)
- **Prediction:** arm-change count and played-config histogram differ from
  RuleOnly's. The throughput delta may be positive, zero **or negative** —
  H2 is about whether learning *does* anything, not whether it *helps*.
- **Falsified if:** CARL-Repaired is bit-identical to RuleOnly on every seed,
  which would mean the repaired learner is still inert on hardware.
- **Note:** `all_diffs_zero` is reported explicitly so a bit-identical
  *identity* is never mistaken for a statistical *tie*. This is the exact
  distinction the reward-v1 run could not make.

### H3 — a wider action space will not pay for itself

> CARL-Expanded will not recover enough benefit to offset its increased
> exploration cost.

- **Comparison:** `CARL-Expanded vs CARL-Repaired` (action-space isolation)
- **Prediction:** throughput delta ≤ 0.
- **Falsified if:** CARL-Expanded exceeds CARL-Repaired with a paired 95% CI
  excluding 0.
- **Basis:** the calibrated simulation found wider arms worse on 11/11
  scenarios (−0.0073 to −0.0537 reward units); Run A found −4.06% on hardware
  under a blind reward. H3 asks whether that survives an informative reward.

**H1 and H3 predict null/negative results; H2 predicts a behavioural difference
of unspecified sign. Registering them before the run is what makes a null result
evidence rather than an absence of evidence.**

---

## Reward-scale calibration — held out, then frozen

Scales are **not** derived from the evaluation seeds. Deriving `t_half` from the
test runs' own throughput would place the reward's 0.5 mid-point by
construction, manufacturing discrimination rather than measuring it.

Procedure, executed before any evaluation run:

1. Sweep `max_batch_size` over the validated axis `{2,4,8,12,16,24,32}` on
   **held-out seed 999**, at the **same 200 requests** used for evaluation.
2. `RewardScales.from_measurements(...)` → medians of the observed throughput,
   TTFT p99 and TPOT p99.
3. **Freeze.** Written verbatim into the artifact under
   `reward_calibration.frozen_scales`, together with every observation they were
   computed from, so a reader can recompute them.

Seed 999 is documented in `ablation_live.py` as *"held out: never used for an
eval run"*. Calibrating at the same `n` as evaluation matters because under a
burst arrival the TTFT p99 scales with the number of queued requests.

---

## Abort gates

Both **abort**; neither warns. Thresholds and their provenance are in
`src/carl/live_reward.py`.

| Gate | When | Condition | Threshold |
|---|---|---|---|
| candidate-arm spread | after calibration, **before** any eval run | reward must separate the calibration configs | `MIN_ARM_REWARD_SPREAD = 0.02` |
| live reward stream | after **every** controller-driven run | one run's reward stream must vary | `MIN_LIVE_REWARD_SPREAD = 0.01`, `MIN_LIVE_DISTINCT = 3` |

Thresholds are read off the per-arm spreads measured on the repaired calibrated
substrate (`docs/eval/reward_diagnostics.md`, 8 workloads × 30 arms, spreads
**0.0219–0.1838**). `MIN_ARM_REWARD_SPREAD` sits just below the observed floor
of 0.0219, so any substrate on which v2 has been shown to discriminate passes,
and the observed v1 failure (spread ≈ 1e-16) fails by many orders of magnitude.
`MIN_LIVE_DISTINCT = 3` is chosen so that *both* historical failures fail it:
reward v1 produced two distinct values in `decisions_042.csv` and one in the
2026-09-02 run.

The live gate aborts for **every** controller-driven config, RuleOnly included.
RuleOnly ignores the reward, so degeneracy would not corrupt its policy — but
the reward stream is a property of the substrate and the frozen scales, not of
the policy, so degeneracy anywhere means the scales do not discriminate at this
operating point, which invalidates the learners' runs too.

---

## What this experiment does **not** establish

One operating point: **burst arrivals at ρ ≫ 1**, a single Tesla T4, one 1.1B
model, one arrival process. It resolves the reward-version confound in Run A. It
does **not** establish a load envelope, and no result from it may be reported as
one.

---

## Configs, and why only four

| Config | Policy |
|---|---|
| `Static-Best` | tuned fixed configuration, no controller |
| `RuleOnly` | `classify → DEFAULT_CONFIGS[regime]`; no learning |
| `CARL-Repaired` | `RepairedLinUCBBandit` + restricted arms |
| `CARL-Expanded` | `RepairedLinUCBBandit` + expanded arms |

Not re-run, with reasons recorded in the artifact's `not_run` block:
`CARL-Full` (as-published learner is provably inert — a better reward cannot
help an inert learner), `CARL-NoSpec/NoCache/NoRouter` (measure CARL-Full by
design in this harness), `CARL-NoSched/NoChunk` (knob-freeze ablations, not a
reward question), `AutoTuner` (not reward-driven), `DynOracle` (meaningless
under v1, and required by none of the three comparisons).
