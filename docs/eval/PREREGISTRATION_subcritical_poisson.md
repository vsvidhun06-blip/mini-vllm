# Pre-registration — Sub-critical Poisson operating point

**Script:** `scripts/eval/repair/subcritical_poisson.py`
**Artifact:** `docs/eval/subcritical_poisson_results.json`
**Raw:** `docs/eval/raw/subcritical_poisson/`
**Registered:** before the GPU run. The artifact embeds a structured summary of
it (the script's `PRE_REGISTRATION` dict, under `pre_registration`) — not this
document's text. This document is identified by its SHA256, recorded as
`preregistration_document_sha256` and as a resume-fingerprint key.
**Does not supersede anything.** `final_hardening` is **frozen evidence**: its
artifact is neither read nor written by this experiment, and none of its
numbers are recomputed here.

---

## Why this experiment exists

The final hardening measured exactly one operating point: a **saturated,
non-stationary burst**. Every request is dumped at t=0, the queue never drains,
and the engine — not the arrival process — sets the pace. At that point, on a
Tesla T4 with TinyLlama-1.1B fp16, seeds 42–51, 200 requests per run, it found:

| comparison | metric | paired Δ | 95% CI | Cohen's d |
|---|---|---:|---|---:|
| CARL-Repaired − Static-Tput | `throughput_tps` | −3.3678 | [−3.6633, −3.0723] | −7.06395 |
| CARL-Repaired − Static-Utility | end-of-run `utility_v2` | −0.0245 | [−0.0263, −0.0227] | −8.44215 |
| CARL-Repaired − RuleOnly | `throughput_tps` | +3.7633 | [+3.2932, +4.2335] | +4.96133 |
| CARL-Expanded − CARL-Repaired | `throughput_tps` | −1.7956 | [−2.2754, −1.3158] | — |

> At the saturated T4 operating point, adaptive reconfiguration is useful but
> insufficient: CARL-Repaired improves throughput and end-of-run utility over
> RuleOnly, but remains below objective-matched static configurations. Expanded
> live configuration coverage further reduces both metrics.

A saturated burst is precisely the regime in which `max_batch_size` binds
hardest, so it is the regime in which a **well-chosen static configuration has
the most to offer**. The obvious rival explanation for the saturated result is
therefore not "adaptation does not work" but "there was little left for
adaptation to win, and a fixed configuration could take all of it". That is a
statement about the *operating point*, and it is testable by moving the
operating point and re-measuring.

This experiment moves exactly one factor: **arrivals**.

---

## The operating point is derived, not chosen

"We lowered the arrival rate" is not an experimental condition, because ρ is
left unstated. So λ is derived mechanically:

```
lambda_phase = TARGET_RHO * mu_phase        TARGET_RHO = 0.40
```

### Source of `mu`

`docs/eval/raw/repair/load_sweep_b7.json`, block `saturation`, field
`mu_max_req_per_s`, rows `interactive@eager` and `batch@eager`.

| phase | row | μ (req/s) | λ = 0.40·μ (rps) | 1/λ (s) |
|---|---|---:|---:|---:|
| interactive | `interactive@eager` | 2.9613881136988742 | 1.1845552454795498 | 0.8441987014249932 |
| batch | `batch@eager` | 1.4135291672527908 | 0.5654116669011163 | 1.7686228610752879 |

Both rows are `method: "burst dump at max_batch=32"`, 3 seeds, 300 requests.

### Why the **eager** rows and not the CUDA-graph rows

The artifact reports both arms — `interactive@graph` 5.5638, `batch@graph`
2.5649 — and `CARLConfig.use_cuda_graphs` **defaults to `True`**. A reader who
took that config field at face value would set λ roughly 1.9× too high, so a
nominal ρ = 0.40 would in fact run at ρ ≈ 0.75 and "sub-critical" would be a
label rather than a fact.

It would be wrong because the execution path does not read the flag alone:

* `ablation_live._new_scheduler` — the scheduler this harness serves on —
  never assigns `_graph_runner`;
* `ContinuousBatchScheduler.__init__` sets `self._graph_runner = None` and
  `_decode_forward` requires `_graph_runner is not None` before it will replay
  a graph.

Every decode therefore falls back to eager with reason `graph_runner_missing`.
This is the same trap that shipped two "CUDA graph" arms with
`cuda_graph_hits == 0` on an earlier T4 smoke, and it is why
`final_hardening.probe_live_dimensions` exists.

Accordingly the arm is read from **the live scheduler object**
(`execution_arm_for(sched)`), never from a config field, and the derivation
**aborts** (`AmbiguousServiceRateError`) if the artifact has no unambiguous row
for the arm that will actually execute.

### Why these μ values describe the real non-graph path

The `hardware_profile` block that produced the saturation rows carries

```
decode_per_row_s         0.009448482794434556
decode_fixed_per_step_s  0.025461779480659390
```

and those are **bit-identical** to `analysis.hardware_profile_fit_eager` in
`docs/eval/raw/repair/batch_intervention_phase15_16_s42_s43_s44.json` — a real
**Tesla T4 / TinyLlama-1.1B-Chat-v1.0 / fp16** measurement (54 eager rows,
R² = 0.9985, CUDA 12.8, torch 2.11.0+cu128). The CUDA-graph parameters
(0.0051007379055177885 / 0.010862193551853631, R² = 0.9927) are a **separate**
regression from the same run, which is what makes the two arms distinguishable
at all rather than by assumption.

`assert_arm_evidence()` re-checks both facts at run time — that the two cost
models differ, and that the eager parameters still equal the T4 eager fit — and
raises rather than proceeding if either stops holding.

### Stated limitation of the derivation

μ is measured **on the calibrated mechanistic substrate**
(`src/eval/engine_model`), whose decode parameters are fitted to the T4 run
above; it is not an end-to-end T4 measurement under this harness's own prompt
windows. The shapes are close but not identical:

| | load-sweep probe shape | this harness (`_SCENARIO_SPEC`) |
|---|---|---|
| interactive | prompt mean 32 (σ 8), output mean 32 (σ 8) | prompt U[16, 32], `max_new` 32 |
| batch | prompt mean 220 (σ 40), output mean 64 (σ 16) | prompt U[128, 256], `max_new` 64 |

ρ = 0.40 is therefore a **derived target**, not a measured outcome.

### Two quantities, two names — only one of them is ρ

**`offered_load.per_phase[p].empirical_rho`** is the realised offered load, and
is the only thing here entitled to be called ρ:

```
empirical_rho = empirical_lambda_rps / mu_req_per_s
```

computed per phase from the **scheduled** arrival offsets, so it is a property
of the seeded draw and is known before a second of GPU time is spent. Recorded
per phase, per seed: `target_rho`, `mu_req_per_s`, `requested_lambda_rps`,
`empirical_lambda_rps`, `empirical_rho`, `empirical_rho_minus_target`,
`empirical_mean_interarrival_s`, `empirical_median_interarrival_s`,
`scheduled_arrival_window_s`, and — separately — the observed drain timing
(`actual_first_finish_s`, `actual_last_finish_s`, `drain_overrun_s`).

**`drain_overrun.arrival_window_overrun_ratio` = `wall_s / scheduled_arrival_window_s`**
is a wall-clock observation about the **engine**. It is **not** a load, a
utilisation or a ρ, and is deliberately not named as though it were. Under a
genuinely sub-critical open-loop stream the run ends shortly after its last
arrival, so a ratio near 1.0 corroborates the target and a value well above 1.0
says the engine set the pace, which bounds what the run can claim.
**Reported, never used to re-tune λ after the fact.**

### No seed is rejected for its realised Poisson rate

Poisson sampling makes `empirical_rho` scatter around 0.40. That scatter **is**
the arrival process, not an error in it.

> **No seed will be rejected, re-drawn, re-run or excluded because its
> realised Poisson rate came out above or below the ρ = 0.40 target. Every
> seed's first draw is the evidence.**

Discarding draws that missed the target would turn an arrival process into a
hand-picked schedule and make the stated ρ a selection effect rather than a
condition. The per-seed values are reported as a spread (`mean`, `min`, `max`
per phase) precisely so the scatter is visible instead of being tidied away.

---

## What moves and what is held

**Moves — exactly one thing:** the arrival process. Burst bulk dump → open-loop
Poisson, with a **separate λ per phase**, each at ρ = 0.40 of *its own* measured
capacity. The two phases differ in capacity by more than 2× (2.96 vs 1.41
req/s), so a single λ would put one phase at roughly twice the other's offered
load and the sub-critical claim would hold for only one of them.

The semantic phase transition is preserved exactly: requests 1–100 use the
INTERACTIVE shape, 101–200 the BATCH shape.

### The splice is an invariant, not an expectation

Two different λ values are only meaningful if each governs its own,
non-overlapping stretch of the stream. `assert_phase_splice` therefore **raises**
(`PhaseSpliceError`) rather than recording a problem, and every run records:

`first_interactive_arrival_s` · `last_interactive_arrival_s` ·
`first_batch_arrival_s` · `last_batch_arrival_s` · `phase_boundary_gap_s`

The invariant is

```
first_batch_arrival_s >= last_interactive_arrival_s
```

plus: every request index before `n // 2` is INTERACTIVE, and every index at or
after it is BATCH. `build_arrivals` places the first phase-1 arrival exactly at
`t_offset`, so the gap is `0.0` by construction and the inequality holds with
equality. A violation of any of the three would mean neither phase ran at its
stated rate, so it aborts the run.

This is the **open-loop** splice. The historical closed-loop dump — phase 1
released on engine progress — is not restored and remains off by default.

One additive harness change is required. `ablation_live._serve` historically
dumps every phase-1 request the instant half of phase 0 has finished — a
**closed-loop trigger** that makes the BATCH phase's arrival pattern a function
of how fast the engine drained phase 0, under which a phase-specific arrival
*rate* cannot be expressed. The new keyword `phase1_open_loop` (default
`False`, so the historical path is bit-for-bit unchanged) releases both phases
from their own `arrival_offset_s` and records the regime shift when the first
BATCH request actually arrives.

**Held:** Tesla T4, TinyLlama-1.1B fp16, seeds 42–51, 200 requests, held-out
seed 999, `OBSERVE_INTERVAL = 10`, the repaired `LiveRewardV2` with
`require_valid_metrics=True`, the restricted/expanded arm-set definitions,
`RepairedLinUCBBandit` at α = 0.5, `FEATURE_DIM = 10`, the reward gates and
their frozen thresholds (`MIN_ARM_REWARD_SPREAD = 0.02`,
`MIN_LIVE_REWARD_SPREAD = 0.01`, `MIN_LIVE_DISTINCT = 3`), and
`final_hardening.convergence_diagnostic` / `CONVERGENCE_RULE` — **imported, not
copied**, so the rule cannot drift between the two experiments.

**Re-derived, not reused:** the reward scales and both static baselines. The
saturated selections (Static-Tput `mb=32, cs=512`; Static-Utility `mb=8,
cs=256`) and the saturated scales are properties of a saturated queue. Carrying
them here would hand this experiment a baseline tuned for a condition it is not
running — the objective-mismatch defect in a new costume. So the full
7 × 4 = 28-configuration live plane is swept **again** on seed 999 under the
Poisson workload, and the three consumers are fed in the same order:

```
held-out MEASUREMENTS -> frozen RewardScales -> utility_v2 -> Static-Utility
```

Static-Utility therefore cannot influence the yardstick it is ranked under.

### Static search

| axis | values |
|---|---|
| `max_batch_size` | 2, 4, 8, 12, 16, 24, 32 |
| `chunk_size` | 64, 128, 256, 512 |

28 live configurations, exhaustive, held-out seed 999, at the evaluation's own
request count (200).

* **Static-Tput** = argmax `throughput_tps`
* **Static-Utility** = argmax end-of-run `utility_v2` under the frozen scales

Both selections may only be computed once **all 28** grid points exist
(`assert_sweep_complete`). An argmax over a partial plane is an argmax over a
different search space.

### Flatness diagnostics — descriptive, and selection-free

> **At sub-critical offered load, end-to-end throughput may be
> arrival-limited. Static-Tput remains the pre-registered argmax, but near-ties
> and the size of the near-optimal set will be reported so a noise-level argmax
> is not over-interpreted.**

For each objective (throughput and `utility_v2`) the artifact records, over the
28-point held-out surface: `best`, `second_best`, `best_minus_second_best`, its
relative %, `min`, `max`, `range_absolute`, `range_relative_pct`, the count of
configurations **within 0.5%** and **within 1.0%** of the best, and the
identities of those near-best configurations.

**No post-hoc tie-break is introduced and the winner is never changed after
evaluation.** `flatness_diagnostics` runs *after* the winners are fixed, returns
no configuration anything acts on, and is consumed by nothing.

`probe_live_dimensions` runs first and **aborts** if any dimension declared
inert (`spec_k`, `routing_threshold`, `cache_affinity_weight`,
`eviction_threshold`, `eviction_window`, `preemption_enabled`,
`use_cuda_graphs`) turns out to be live against the real scheduler object. A
baseline searched over the wrong space is not a baseline.

### Treatments

Exactly five, no more:

1. **Static-Tput** — fixed config, argmax throughput on seed 999
2. **Static-Utility** — fixed config, argmax `utility_v2` on seed 999
3. **RuleOnly** — classify → `DEFAULT_CONFIGS[regime]`, no learning
4. **CARL-Repaired** — `RepairedLinUCBBandit` + restricted arms
5. **CARL-Expanded** — `RepairedLinUCBBandit` + expanded arms

No DynOracle (withdrawn: it ranks arms by rewards collected in a particular
cycle *order*, so it is not an oracle). No inactive ablations — speculation,
routing and KV eviction cannot execute here, so freezing them measures nothing.

---

## Research questions

**Q1.** How much flatter is the live configuration landscape under ρ ≈ 0.40
Poisson arrivals than under the frozen saturated burst? Reported from the
held-out grid: min/max/absolute/relative throughput range, min/max/absolute
utility range, and the identities of Static-Tput and Static-Utility. Compared
**descriptively** with the saturated sweep. *No significance test is applied to
28 deterministic search points and none would mean anything.*

**Q2.** Does CARL-Repaired beat Static-Tput on throughput?
**Q3.** Does CARL-Repaired beat Static-Utility on end-of-run `utility_v2`?
**Q4.** Does CARL-Repaired improve over RuleOnly?
**Q5.** Does Expanded help or hurt relative to Repaired?
**Q6.** Does CARL satisfy the **same** pre-registered convergence rule under
sub-critical load?
**Q7.** Does the relationship between knob leverage and adaptation headroom
change relative to the saturated operating point?

---

## The mechanistic hypothesis

> **The sub-critical Poisson operating point will expose less
> configuration-dependent throughput headroom than the saturated burst
> operating point. Therefore the value available to online adaptation should be
> smaller relative to its finite-horizon search and reconfiguration cost.**

This predicts a shrinking **effect size**. It does **not** predict a winner.

### H1 — the landscape flattens

* **Statement:** the *relative* throughput range across the 28-configuration
  held-out grid will be smaller at ρ ≈ 0.40 than at the saturated point,
  because below saturation the realised batch is arrival-bounded rather than
  config-bounded and `max_batch_size` does not bind.
* **Metric:** `landscape.throughput_tps.range_relative_pct`.
* **Prediction:** strictly smaller than the saturated sweep's.
* **Falsified if:** equal to or larger than the saturated sweep's.
* **Note:** descriptive comparison of two deterministic 28-point grids.

### H2 — adaptation headroom shrinks

* **Statement:** because less configuration-dependent headroom is available,
  the *magnitude* of every paired difference between treatments will be
  smaller, in relative terms, than at the saturated point.
* **Metric (amended 2026-10-22, see corrections log):**
  `|mean paired throughput difference| / mean comparator throughput`, i.e.
  `abs(mean_s(a_s − b_s)) / mean_s(b_s)` on `throughput_tps`, where the
  comparator `b` is **Static-Tput** for A, **RuleOnly** for C and
  **CARL-Repaired** for D. The identical definition applies at both operating
  points. Comparison B is excluded: its outcome is `utility_v2`, which has no
  natural zero, and no relative-utility hypothesis is created in its place.
* **Prediction:** smaller relative paired differences than the saturated run's.
* **Falsified if:** any of A, C or D shows a *larger* relative paired
  difference than at the saturated point.
* **Unresolved if** the frozen saturated artifact's comparator means and
  unrounded paired differences are unavailable — then H2 is **neither
  supported nor falsified**. They are never reconstructed from the rounded
  paired differences transcribed in the script. The same applies to H1, whose
  saturated side needs the saturated grid's throughput min and max.
* **Saturated side (recorded 2026-10-22, after the operational T4 smoke was
  inspected and before the full run).** The frozen artifact was recovered and
  is now read, hash-verified, by `scripts/eval/repair/saturated_reference.py`
  (see the corrections log). The definitions above are unchanged.
  * H1's saturated side is 100 · (96.00104467760788 − 44.97500716097522) /
    96.00104467760788 = 53.1515440149522 %, over the 28-point held-out grid.
  * H2's saturated sides are A = 0.0350242571445, C = 0.0422731152326 and
    D = 0.0193515168703. They come from the full-precision per-seed
    throughputs, aligned through the raw seed files.
  * A verdict is given only for the registered published run. If the
    reference fails verification, both hypotheses stay unresolved and a
    published run refuses to start.

### H3 — the direction is deliberately not pre-registered

No winner is registered for A, B, C or D. All three outcomes below were written
down before the GPU ran and **all three are scientifically informative**.

| case | outcome | interpretation |
|---|---|---|
| **A** | CARL still loses to objective-matched static tuning | the static advantage — the general limitation of finite-horizon online adaptation in this setting — **survives across offered-load regimes**, which *widens* the scope of the saturated result rather than confining it |
| **B** | CARL becomes competitive with or exceeds an objective-matched static baseline | offered load is a **boundary condition**: the saturated finding is then scoped to ρ ≫ 1 and must be reported as such |
| **C** | CARL improves over RuleOnly but not over static | adaptive behaviour is useful, but **finite-horizon adaptation economics remain unfavourable**; the saturated conclusion holds in shape across regimes even if its magnitudes change |

**Hypotheses will not be rewritten after the GPU results are seen.**

### H4 — cold start (positive control, carried over)

* **Statement:** every adaptive run shows a non-empty warm-up — at least one
  leading control cycle whose reward is withheld for want of a measured
  TTFT/TPOT.
* **Metric:** `reward_validity.warmup_cycles`.
* **Prediction:** `warmup_cycles >= 1` for every controller run.
* **Falsified if:** any controller run reports `warmup_cycles == 0`, which
  would mean the validity gate never engaged.
* **Note:** under Poisson arrivals the warm-up may be **longer** than at the
  saturated point, because fewer requests are in flight early and the first
  completion arrives later. A longer warm-up is an observation, not a failure.

### H5 — the arrival process is the one described

*Amended 2026-10-22, before any GPU run; the superseded wording and the
prior-exposure disclosure are in the corrections log.*

* **Statement:** per phase, the empirical mean interarrival time is consistent
  with 1/λ for the derived λ.
* **Metric:** `arrival_validation.<phase>.empirical_mean_interarrival_s`, the
  mean of the phase's k consecutive gaps.
* **Reference distribution:** for k independent Exp(λ) gaps the **sum** is
  Gamma(shape = k, rate = λ) and the **mean** is Gamma(shape = k, rate = kλ).
  The mean's distribution is the one used. At n = 200 each phase has 100
  requests and **k = 99** gaps (the first BATCH arrival sits exactly on the last
  INTERACTIVE arrival, so no gap spans the boundary).
* **Family:** 20 cells — evaluation seeds 42–51 × {INTERACTIVE, BATCH}. Each
  cell is tested against **exact two-sided Gamma bounds at α = 0.05/20**
  (Bonferroni). That keeps the familywise error at ≤ 0.05 **without** requiring
  the cells to be independent. In units of 1/λ the bounds are
  **[0.7233298293, 1.3314135661]**: INTERACTIVE [0.610634, 1.123978] s around
  0.844199 s; BATCH [1.279298, 2.354768] s around 1.768623 s.
* **Held-out seed 999** is **not** in the family. It is reported separately
  against its own exact 95% bounds, [0.8127506551, 1.2063698009] × 1/λ.
* **Bounds are computed from λ and n alone,** before any measurement
  (`h5_reference`, stored in the artifact at the arrival-plan stage). They were
  fixed analytically, not chosen by looking at which draws pass.
* **Descriptive only:** the per-cell normal-approximation 95% interval
  (`mean_within_ci95`, unchanged), and empirical ρ.
* **Prediction:** every family cell lies inside its Bonferroni bounds.
* **A family rejection** flags statistical inconsistency of the realised draws
  with Exp(λ), and it requires investigation. It does **not** by itself:
  * prove a generator defect;
  * automatically invalidate every conclusion;
  * authorise changing seeds, λ or the workload.

  Every cell, rejected or not, is preserved and reported. No seed is rejected,
  replaced or retried.
* **Generator correctness is checked separately, and not by H5.** Separate
  checks cover:
  * deterministic regeneration of every unit's offsets from the named stream;
  * positive, finite gaps;
  * the splice and the counts;
  * a large-sample test of the sampler (`random.expovariate` inside
    `build_arrivals`) on a namespace no evaluation uses: mean, coefficient of
    variation and median of Exp(λ).

  Reproducibility alone is not treated as proof of exponential sampling.

---

## Statistics

For every paired 10-seed comparison the artifact reports:

* mean paired difference
* **95% paired CI, Student-t (primary; amended 2026-10-22).** The interval is
  `mean ± t(0.975, n−1) · sd / √n`, where `sd` is the sample SD of the paired
  differences.
  * t is computed for the actual n: 2.2621571628 at n = 10. A ten-seed value
    is never reused for another n.
  * With n < 2 no interval is reported, and the status says so.
  * With zero variance the interval collapses to a point, and the record says
    it is degenerate rather than certain.
* **95% normal-approximation CI (secondary).** The interval is
  `mean ± 1.96 · sd / √n`, labelled as such. It is kept only for comparison with
  the frozen saturated analysis, which used it. The saturated artifacts and
  their reported intervals are not altered.
* **Interval limitations.** The interval assumes the paired differences are
  roughly normal, which ten seeds can barely check. Skew or outliers, for
  example in drain tails, degrade coverage. It covers seed-to-seed variation
  on one GPU, not hardware variation. These are **per-comparison** intervals:
  there is no familywise adjustment across the four comparisons or the two
  metrics.
* Cohen's *d* (paired)
* the **sign of the paired difference on every seed**, plus
  `n_positive` / `n_negative` / `n_zero` and `all_diffs_zero`
* an **exact two-sided Wilcoxon signed-rank test** as robustness, by
  enumerating all 2ⁿ sign assignments over the observed midranks. It is
  **declined and says so** when any difference is exactly zero: a bit-identical
  pair is an *identity*, not a tie to be broken.

**p-values are never the headline.** The paired difference, its CI, the effect
size and the per-seed signs are the result.

### Main comparisons

| label | A − B | pre-registered metric |
|---|---|---|
| A | CARL-Repaired − Static-Tput | `throughput_tps` |
| B | CARL-Repaired − Static-Utility | end-of-run `utility_v2` |
| C | CARL-Repaired − RuleOnly | `throughput_tps` **and** end-of-run `utility_v2` |
| D | CARL-Expanded − CARL-Repaired | `throughput_tps` **and** end-of-run `utility_v2` |

Both metrics are reported for every comparison; `metric` names the
pre-registered primary so nothing is chosen after the fact.

---

## Two utility estimators, kept distinct

These are **not** interchangeable and neither is ever substituted for the other:

* **`run_utility_v2`** — `utility_v2` of a run's **end-of-run aggregates** under
  the frozen scales. Defined for every treatment including the statics, and the
  basis of comparison B.
* **`mean_valid_cycle_utility`** — the mean over a controller's **scored control
  cycles**; what the learner actually consumed. Reported for controllers only,
  and never compared against the statics, which have no control cycles.

**Permitted wording for the static-utility baseline:**

> Static-Utility is selected using the same `utility_v2` function, weights, and
> frozen scales used by CARL, and both systems are compared on end-of-run
> `utility_v2`.

**Not permitted:** any claim that Static-Utility optimises the exact sequence of
per-cycle rewards CARL consumed.

---

## Arrival-process validation

Logged per phase, per seed:

requested λ · target ρ · the μ λ was derived from · number of requests · number
of interarrival gaps · empirical interarrival mean · empirical interarrival
median · empirical interarrival stdev · empirical arrival rate · phase arrival
duration · **expected mean interarrival = 1/λ** · the 95% interval around it ·
whether the empirical mean falls inside it.

Raw arrival offsets are retained per seed in
`docs/eval/raw/subcritical_poisson/<config>_seed<NNN>.json` under
`arrival_offsets_s`.

The generator is deterministic in the seed, and arrivals are drawn from a
**named sub-stream** (`random.Random(f"subcritical_poisson/arrivals:{seed}")`)
independent of the prompt-length draws, so a change to shape sampling cannot
silently perturb the arrival process — the one factor this experiment varies.

The amended H5 family (`h5`) and its bounds (`h5_reference`) are reported
alongside. `h5.arrivals_identical_across_treatments` checks, rather than
assumes, that every treatment on a seed saw the same offsets.

### Descriptive phase / regime reporting (added 2026-10-22)

`phase_regime_per_seed`, per treatment and seed, is **descriptive only**: never
an outcome and never a filter. It reports:

* **Per-phase drain timing:** each scheduled phase's last scheduled arrival,
  first submit, last finish, and the drain after its last arrival.
* **The mixed-execution window:** from the first BATCH submit to the last
  INTERACTIVE finish. The arrival windows are sequential by construction, but
  **execution is not**: INTERACTIVE requests can still be in flight after the
  first BATCH arrival.
* **For controllers, each control cycle cross-tabulated** by:
  * its **scheduled arrival window** (before or after the scheduled first BATCH
    arrival);
  * its **in-flight composition** (`idle`, `interactive_only`, `batch_only` or
    `mixed`: requests with `submit_s ≤ t < finish_s`);
  * the classifier's **inferred** regime.

  It also gives the share classified `batch` among cycles in the BATCH window,
  among BATCH-only cycles and among mixed cycles.

**How the timing is joined, verified in `ablation_live._serve`:**
* `step_log[*].t`, every request's `stage_timestamps` and the scheduled
  offsets are seconds since the same `t0`.
* A cycle is joined to time **only through its scheduler-step identifier**: the
  controller is called with `sched._step_idx` straight after the step that
  `step_log` recorded under the same index.
* The cycle's own `t_monotonic_s` has a different origin and is never compared
  with them.
* An unjoinable cycle, a duplicated step identifier, or a request without finite
  timestamps is listed in `problems`, and `attribution_reliable` is then false.
  Nothing is guessed.

The regime is the classifier's inference, never ground truth. The predicted
misclassification of BATCH-phase cycles remains a hypothesis until measured.

**Tokens per busy-second is NOT reported.** `step_log[*].dt` includes idle
sleeps, request submission and the previous controller step, so the existing
timestamps do not support an exact busy-time measure.

---

## Reward, cold start, and non-degeneracy

Preserved from `final_hardening`, unchanged:

* missing TTFT/TPOT is **missing**, never zero-as-good;
* `reward_valid = False` until the required latency measurements exist;
* an invalid reward performs **no** learner update and substitutes no fill
  value;
* `rewarded_arm = -1` on a withheld cycle;
* `warmup_cycles`, `first_valid_reward_cycle_index`,
  `first_valid_reward_scheduler_step`, `warmup_duration_s`,
  `valid_reward_updates` and `withheld_after_first_valid` are all recorded;
* `withheld_after_first_valid` must remain **0**.

**Pre-run reward-spread gate.** `gate_candidate_arm_spread` over the 28 held-out
candidates at the existing frozen threshold `MIN_ARM_REWARD_SPREAD = 0.02`.
Logged: candidate reward spread, threshold, PASS/FAIL. Thresholds are **not**
changed. If the sub-critical plane is so flat that the reward cannot separate 28
configurations, that is a **result about this operating point** and is reported
as one — the run aborts and writes its reason, and the threshold is not lowered
to get past it.

**Per-run gate.** `gate_live_reward_stream` over the **valid rewards only** at
`MIN_LIVE_REWARD_SPREAD = 0.01` / `MIN_LIVE_DISTINCT = 3`. Padding a withheld
cycle with a fill value would make the gate pass or fail for a reason unrelated
to whether the reward discriminates.

For each adaptive run the artifact also reports the mean valid cycle reward,
the reward spread, and the number of distinct valid reward values.

---

## Convergence

`final_hardening.convergence_diagnostic`, imported — the same function object,
the same `CONVERGENCE_RULE`:

> at least 6 decisions in the regime; at least 2 distinct arms in the first
> half; a second-half modal-arm share of at least 0.75; and a modal share that
> **strictly increased** from the first half to the second.

Reported supported/unsupported per regime, per seed, and as an aggregate count
out of 10 seeds. **No more favourable definition is introduced.** For reference,
the saturated run reported 0/10 for RuleOnly, CARL-Repaired and CARL-Expanded
alike.

---

## Action-space reporting

Both counts, per regime, for the restricted and expanded spaces: nominal arm
count, **unique effective live configurations**, the reachable
`max_batch_size` values, the reachable `chunk_size` values, and whether nominal
arms collapse onto identical effective configurations.

Because some nominal arms differ only in inert fields, the finding is worded as
**"expanded live configuration coverage"**, not "10 arms vs 6 arms".

---

## Provenance

Local only. **Nothing contacts GitHub.** The artifact embeds the
`source_provenance/2` sidecar (local git HEAD, dirty state, captured
`git status`, `git diff HEAD` SHA256, working-tree source manifest, the archive's
SHA256 under `archive.sha256`, and the archive-member manifest under
`archive_manifest`), the start-up `source_verification` report, the
experiment timestamp, `argv`, GPU, CUDA, torch, Python, model, dtype,
seeds, n, workload version, target ρ, the derived μ and λ, the static search
definition, the arm-space definition, the reward version, the frozen reward
scales, and the convergence rule and its source.

The Colab archive contains no `.git`; that is expected, and the sidecar is what
replaces the runtime's null `git_sha`. An absent sidecar is recorded as an
explicit absence, never as a bare null.

### The provenance workflow, in order

The source repository with `.git` lives on **Windows**. Provenance is captured
there, never reconstructed in Colab and never from a remote.

1. Vidhu builds the immutable source ZIP locally, in PowerShell.
2. Vidhu runs `source_provenance.py --archive <zip>` **locally**, against that
   ZIP. It refuses an archive that would not extract identically on Linux (a
   member name containing `\`, an absolute or `..` path, a case-insensitive
   collision) or that lacks a required runtime data file (the μ sweep, the T4
   eager-fit artifact, this document), and records the SHA256 of every source
   and required data **member** (raw bytes) as `archive_manifest`.
3. That produces `docs/eval/SOURCE_PROVENANCE.json` (written atomically).
4. Vidhu uploads **both** the ZIP and the provenance sidecar to Colab.
5. Colab extracts the ZIP.
6. The sidecar is copied into the extracted repository.
7. At start-up, before the model loads, `subcritical_poisson.py` recomputes the
   raw-byte SHA256 of every source and required data file in the extracted tree
   — pure Python, no git — and compares it with `archive_manifest`. A
   **published** run refuses to start unless every file matches with nothing
   missing and nothing extra; a smoke run records the result. This check is
   mandatory for a published run.
8. Optionally, and recommended: `--verify-archive <uploaded zip>`.
   * **With it,** the ZIP itself is hashed and compared with `archive.sha256`,
     and the artifact records `archive_sha256_status:
     verified_against_uploaded_zip`.
   * **Without it,** the ZIP digest is only `asserted_by_sidecar`.

   The two statuses are never conflated. A mismatch refuses a published run.

`archive.sha256` and `archive_manifest.manifest_sha256` must be hexadecimal
SHA256 digests (64 lowercase hex characters), not merely 64-character strings.

**What enters the resume fingerprint:** `archive.sha256`, the archive-member
manifest digest, the working-tree diff SHA256, and the manifest digest
**recomputed over the executing tree** — not the sidecar's own claim. So a
resume across a differently-built archive, or across an edit made on Colab after
extraction, aborts even if the sidecar file is unchanged.

**Line endings.** The archive is built from a `core.autocrlf=true` Windows
checkout, so most `.py` members carry CRLF. ZIP extraction does not translate
line endings, so verification compares raw bytes and matches exactly. Each
manifest entry also records `sha256_lf` (after CRLF → LF) so a member can be
compared with the LF blob git stores; it is informational and never used to
verify.

---

## Checkpointing

The artifact is written after the arrival derivation, after **every** held-out
grid configuration, after the static selections, after **every**
(treatment, seed) pair, after every completed treatment, and on every abort
path **that occurs after the run has taken ownership of the checkpoint**. An
interruption of the *process* loses at most the unit of work in flight — but
only while the storage holding the output survives. On Colab that is not the
same thing; see **Colab: runtime resets and persistence** below.

Every write — the results JSON, each sweep unit, each evaluation unit, the
calibration record, terminal markers and failed evidence — is serialised
completely in memory, written to a temporary file in the destination
directory, flushed and fsync'd, and moved into place with `os.replace`. Three
separate statements:

* **Process interruption** (an exception, Ctrl-C, a killed process) on the
  local filesystems the test suite exercises: the destination is the previous
  complete file or the new complete file, never a partial one.
* **Atomic replacement:** `os.replace` is a single rename on POSIX and
  `MoveFileEx` on Windows, both atomic within one local volume.
* **Not guaranteed:**
  * **Power loss or a kernel crash.** The directory entry is not fsync'd, so
    which version is visible afterwards is not promised, and it is not claimed
    that the previous file necessarily survives.
  * **Network or FUSE filesystems such as Google Drive.** Their rename and
    fsync semantics are not those of a local disk, and nothing is claimed for
    them.

**The checkpoint carries its identity before the run owns it.** These are
already in the in-memory checkpoint before a resumed session may write it:
* the fingerprint;
* the ledger exactly as the previous session left it;
* every verified row;
* the checkpoint's calibration binding.

So a failure anywhere in the restore, inside a `supersede` move, inside the
calibration-record write, or at the first checkpoint write, still leaves a
resumable checkpoint. Fault-injection tests cover each of these. An abort
record that lacks the fingerprint never replaces a checkpoint; it is written
aside.

Each held-out sweep point is written to **its own file**
(`raw/subcritical_poisson/sweep/mb<B>_cs<C>.json`) before it is checkpointed, and
the checkpoint always carries **every** verified row in grid order — never only
the rows a resumed session has walked past.

A failure **before** ownership — model or tokenizer load, an invalid path, the
liveness probe, the service-rate derivation, unverifiable source, a refused
resume, an interrupt during start-up — never writes the checkpoint. It is
recorded in its own file under `subcritical_poisson_preflight_errors/`.

**Outcomes have three statuses.**

**`operational_interruption`** (a disconnect, a crash) may be resumed.

**`terminal_scientific_gate_failure`** (the pre-run spread gate or a per-run
reward gate fired) may not.
* **Recording order:**
  1. the verdict is recorded **in memory** (results and session);
  2. a small **terminal marker** is written to `raw/subcritical_poisson/failed/`;
  3. the failed unit's measurements, controller trace and gate evidence are
     written beside it.
* **If writing the marker or the evidence fails,** the failure is recorded
  next to the verdict, and the original scientific exception still propagates.
* **The run is classified terminal by its recorded state,** not only by the
  final exception type.
* **`--resume` refuses** on a terminal checkpoint, an abort record, a terminal
  run-history entry, **or any marker or failed-unit file**. That covers a crash
  between the marker and the checkpoint update. Re-measuring until timing noise
  lets the gate pass would turn the gate into a retry loop.
* **Limitation:** if *every* write fails (marker, evidence and checkpoint),
  nothing on disk can tell a later session that the gate fired. The run prints
  that it must not be resumed, and that is the only record.

**`invalid_fresh_evidence`** may not be resumed either. Every freshly measured
unit is validated **before it counts as complete**, by the **same** rules a
resume applies, on its JSON round-tripped form.
* Invalid fresh evidence stops the run.
* It is preserved under `raw/subcritical_poisson/invalid/` and is never
  written as a completed unit.
* `--resume` refuses until it is investigated, so it is never silently
  regenerated.
* The in-memory unit used for aggregation is the round-tripped one, so fresh
  and resumed units aggregate identically.

**Published and smoke runs never share a destination.**

A published run must:
* run on a **Tesla T4**, checked both on the default device and on the device
  that actually holds the model's weights;
* use exactly the registered seeds, n, grid, ρ and μ artifact;
* pass source verification;
* establish the model's **immutable identity**: the HF cache snapshot commit
  plus SHA256 of the config, weights and tokenizer files, read offline;
* write only to `docs/eval`.

**Anything else must be `--smoke`** (or `--allow-cpu-smoke`).
* A smoke run may run on other hardware. It records its actual GPU, library
  stack and model identity, and is never a result.
* It writes under `smoke_runs/subcritical_poisson/` or an explicit
  `--output-dir`.
* Output locations are compared **after resolving symlinks and junctions**,
  and with `samefile` for existing paths. A smoke run whose output, results,
  raw, sweep, failed, invalid, superseded or calibration location resolves
  onto a published one is refused.

A fresh run refuses to start over an existing results file or existing unit
evidence.

---

## Engine warm-up (`engine_warmup/1`; added 2026-10-22 after the T4 smoke)

**Why.** The T4 smoke charged a one-off ~8 s start-up stall to the first
measured unit only; see the Corrections log. Its precise cause was not
measured.

**What it is.** It is a fixed, declared, bounded sequence of disposable
schedulers. Each is built with the same `ablation_live._new_scheduler` and
`_apply_sched` as the measured units, and driven with `add_request` and
`step()`. The stages are:

* **prefill:** one request at a time at the largest chunk size. Prompt
  lengths are 16, 17, 32, 33, …, 240, 241, 256: every 16-token boundary of the
  workload's prompt range and one past it, so prefill widths that are and are
  not multiples of 16 both run. Because the prompts share a tiled prefix,
  later requests also take the prefix-cache (sliced, masked) prefill path.
* **chunked_prefill:** one stage for each chunk size below 256, with prompts
  of c+1, c+17 and 256 tokens. These force continuation chunks through the
  position-offset, masked prefill path.
* **batched_decode:** one stage for each batch size b from 1 to the largest
  batch any unit can execute. Each runs b requests with 30-token prompts and 4
  new tokens, so the decode context passes 31–34, crossing 32.

**Inputs and coverage.**
* Chunk sizes and the largest batch are the union of the sweep grid and every
  arm either controller can select, after the same clamp. The smoke grid
  therefore still covers chunk sizes 384 and 512, and batch sizes up to 32.
* Prompts come from `src.carl.live._make_prompt`, the workload's own
  construction. No RNG is drawn, and `eos_token_id=None` makes every request
  emit exactly its declared tokens.
* The protocol is a function of the grid and the arm space only, never of an
  observation, ranking or reward.

**Why this coverage.**
* Triton specialises compiled kernels on `constexpr` arguments: causal or not,
  mask or not, block sizes. It also specialises integer arguments on
  divisibility by 16 and on equality to 1. (That behaviour depends on the
  Triton version and was not checked on Colab's Triton.)
* cuBLAS chooses kernels by GEMM shape.
* Decode GEMMs have M equal to the decode batch (1–32 here). Prefill GEMMs
  have M equal to the chunk width, and a single prefill forward never exceeds
  256 tokens in this workload.

**Limits, stated honestly.**
* A step's prefill budget is `chunk_size` minus the number of decoding
  requests, so measured prefill widths can take any value. Not every (width,
  context) pair is executed.
* A kernel or cuBLAS variant first reached by an uncovered shape can still
  cost time inside a measured unit. That residue is not measured.

**When it runs.**
* Once per process, after the resume fingerprint and evidence checks, and
  **before the run takes ownership of the checkpoint**.
* Only if at least one unit will be measured. That decision applies the same
  validation the loops apply, read-only.
* A fully completed resume performs no inference.
* A measured unit about to run without a completed warm-up aborts the run
  (fail closed).

**Isolation.**
* Each stage uses a new scheduler and KV pool, which is drained and discarded.
* No `CARLController`, `MetricsTracker`, `LiveRewardV2`, `_serve` or arrival
  clock is involved.
* Python, NumPy, torch CPU and CUDA RNG states are snapshotted and restored,
  and their restoration is checked.
* The device is synchronised before the warm-up clock starts and after the
  last stage, before any measured unit.
* The CUDA caching allocator is not emptied, as is also the case between
  measured units.
* The warm-up's time, tokens and requests never enter a measured metric, an
  arrival timestamp, the reward calibration or the convergence rule.

**Termination.**
* Each stage has a declared step bound; a stage that exceeds it, or does not
  emit exactly its declared tokens, aborts the run.
* The warm-up is never repeated, and nothing about it depends on how fast it
  ran.

**What is recorded.**
* `engine_warmup_protocol`: the declaration, which is also a fingerprint key.
* `engine_warmup`: the per-session record — status, per-stage steps, tokens
  and elapsed time, synchronisation, RNG restoration, and the forward shapes
  actually observed.
* Each `run_history` entry carries its own session's record, so earlier
  sessions' warm-ups survive a resume.
* Measured warm-up duration is never part of fingerprint equality.

**Failure.** A warm-up that fails or is interrupted happens before ownership.
The checkpoint is left byte-for-byte as it was, and the failure, with the
warm-up record, goes to `subcritical_poisson_preflight_errors/`. `--resume`
then continues normally.

## Colab: runtime resets and persistence

**A browser disconnect and a runtime reset are different events.**
* **Browser disconnect:** the VM and its `/content` disk keep running, and the
  process may finish or be resumed in place.
* **Runtime reset:** idle timeout, the session limit, "disconnect and delete
  runtime", or a crash of the VM. This **deletes `/content`**: the checkpoint,
  the calibration record and every raw file are gone.

The atomic writes above protect files only while the disk holding them exists.

Run **on the local `/content` disk**, not directly on a Google Drive FUSE mount:
local disk is where the atomic-write statements above hold. Persistence is a
separate step: **copy a complete, verified snapshot** to persistent storage.

**Snapshot, whenever the process has exited.** After an abort, after
completion, or by stopping the run deliberately. A copy taken while the run is
writing is not guaranteed to be a consistent set.

```
cd /content/mini-vllm
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar -cf /content/sp_$STAMP.tar \
    docs/eval/subcritical_poisson_results.json \
    docs/eval/raw/subcritical_poisson \
    docs/eval/subcritical_poisson_preflight_errors 2>/dev/null
sha256sum /content/sp_$STAMP.tar > /content/sp_$STAMP.tar.sha256
cp /content/sp_$STAMP.tar /content/sp_$STAMP.tar.sha256 /content/drive/MyDrive/sp_snapshots/
( cd /content/drive/MyDrive/sp_snapshots && sha256sum -c sp_$STAMP.tar.sha256 )
```

**What the snapshot carries:**
* the checkpoint, with its `run_history` / audit trail;
* `raw/subcritical_poisson/calibration.json`;
* `raw/subcritical_poisson/sweep/` (sweep units);
* the evaluation units;
* `raw/subcritical_poisson/failed/` (terminal markers and failed evidence);
* `raw/subcritical_poisson/invalid/`;
* `raw/subcritical_poisson/superseded/`;
* the preflight error records.

The `sha256sum -c` re-reads the copy on Drive. That checks the bytes Drive
returns; it does not make Drive behave like a local disk.

**Recover after a runtime reset:**
1. Re-extract the **same** ZIP.
2. Copy in the **same** sidecar.
3. Verify the snapshot: `sha256sum -c` against its `.sha256`.
4. Extract it into the repository root:
   `tar -xf sp_<STAMP>.tar -C /content/mini-vllm`.
5. Run the resume command.

The fingerprint then checks:
* the source;
* the GPU and library stack;
* the model identity, which needs the same HF snapshot: a re-download that
  resolves a different commit is refused;
* everything else.

The evidence validation and the terminal / invalid refusals apply exactly as
before.

**The HF model cache is wiped by a reset too.** It lives under
`/root/.cache/huggingface`. To keep the same commit available, copy
`hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0` to Drive with the snapshot,
and restore it before resuming. Otherwise a re-download that resolves a
different commit makes the resume refuse. That is the intended, fail-closed
outcome.

**Only snapshots are protected.** Anything measured after the last snapshot is
lost by a reset and is re-measured on resume. Re-measuring follows the rules
above and is not a scientific choice.

**Unresolved operational requirement.** There is no automated in-run
snapshotting to persistent storage. Doing that safely needs a storage
integration with verified semantics, which is beyond this pass. Snapshots are
manual, between sessions.

---

## Stopping rule

After this experiment the experimental campaign **stops**, unless a genuine
validity defect is discovered.

---

## Cost, and why it is what it is

Under an open-loop sub-critical stream the wall clock is set by the **arrival
window**, not by the engine — that is the whole point of the condition, and it
is the price of a stated ρ. Per 200-request run:

```
99 gaps / 1.1846 rps  +  99 gaps / 0.5654 rps  =  83.6 s + 175.1 s  ≈  259 s
```

Each phase has 100 requests and therefore 99 exponential gaps: the first BATCH
arrival is placed exactly on the last INTERACTIVE arrival (`t_offset`), so the
splice itself adds no gap.

78 runs (28 held-out sweep + 5 treatments × 10 seeds) × ~259 s ≈ **5.6 h floor**,
and ~**6–6.5 h** once per-run drain and setup are included. Compare the saturated
run, whose wall clock was engine-bound and therefore far shorter.

This is not reducible without changing a pre-registered quantity (ρ, n, seeds or
treatments), so it is not reduced. It is instead why the artifact is
checkpointed at every boundary and why `--resume` exists.

### `--resume` is an execution feature only

It changes **no scientific quantity**. It decides only whether a unit of work is
re-measured or read back.

**Unit order**, fixed and deterministic (`unit_plan`): the 28 held-out sweep
points in grid order, then every treatment in `CONFIGS` order crossed with the
seeds in the order given. A resume walks this same list and skips; it never
reorders.

**Fingerprint.** Before skipping a single unit, the stored fingerprint must match
the current one on **every** key: experiment and version, target ρ, execution
arm, μ and λ for both phases, the μ source artifact, seeds, held-out seed, n,
both grid axes, model, dtype, arrival mode and RNG namespace, workload version,
`phase1_open_loop`, reward version and thresholds, `OBSERVE_INTERVAL`, α,
`FEATURE_DIM`, a SHA256 of the convergence rule, the treatment list, a SHA256 of
the arm space, the run mode (published / smoke), the source archive SHA256
(`archive.sha256`), the archive-member manifest digest, the manifest digest
recomputed over the executing tree, the working-tree diff SHA256, plus a SHA256
of this pre-registration document.

It also includes, since 2026-10-22:
* the **runtime identity**: device type and index, GPU name / compute
  capability / memory, Python, torch, torch-CUDA, cuDNN, transformers,
  huggingface_hub and safetensors versions;
* the **model identity**: the HF snapshot commit and SHA256 of every model and
  tokenizer file.

Every key except the execution arm and the model identity is checked **before
the model is loaded**. Both are re-checked, with every other key, once it is
loaded. Any mismatch raises `ResumeMismatchError`, names the
differing keys, and **writes nothing to the checkpoint or the evidence** —
overwriting the artifact with an abort record would destroy the checkpoint
being resumed.

**A checkpoint entry is not sufficient, and neither is a file that parses.** A
unit counts as complete only if its ledger entry exists **and** its durable
evidence validates: schema and experiment version, the experiment fingerprint,
(for evaluation units) the calibration digest, the unit / treatment / seed
identity, n, finite metrics, the raw arrival offsets equal to the seeded draw,
every nested arrival / offered-load / drain / phase-boundary block the
aggregation reads, the selected static configuration (statics), and — for
controllers — a cycle trace from which `decisions`, `arm_changes`,
`arm_change_rate`, `arm_histogram`, the valid-reward statistics and the imported
convergence diagnostic all reproduce exactly. Otherwise the unit is **re-run**,
the untrusted file is moved to `raw/subcritical_poisson/superseded/` (never
deleted), and the reason is recorded in `checkpoint.units_rerun_on_resume` and
`checkpoint.recovery_audit`. A half-written file is re-measured, never patched.
The **same** validation runs on every freshly measured unit before it counts;
fresh evidence that fails it stops the run (`invalid_fresh_evidence`, above)
rather than being re-measured.

**The selection cannot move.** Measurements → frozen scales → pre-run gate →
both argmaxes, by the same imported functions, produce a **calibration digest**
binding the full held-out measurements, the frozen scales, the reward weights and
both selected configurations. It is written to
`raw/subcritical_poisson/calibration.json` before the first evaluation, and every
evaluation unit records it. Before evaluation has started, missing sweep points
are simply measured. Once it has started, the held-out rows are restored from the
calibration record and must reproduce its digest; if that record is lost, it is
recovered only from a complete, verified sweep that reproduces the digest every
evaluation unit carries **and the digest the checkpoint recorded** — otherwise
the resume **refuses**, rather than mixing a replacement calibration with
evaluations bound to the original.
* If neither the checkpoint nor any evaluation unit records a digest, the
  recovery is refused. A recomputed calibration could then only be shown to be
  self-consistent, not to be the original.
* A calibration record that disagrees with the checkpoint's digest is refused
  too.

## Commands

Provenance is captured **locally, in PowerShell, before the archive is
uploaded** — see the workflow above. Nothing below regenerates it in Colab.

### T4 smoke (real CUDA; wiring only, never a result)

```
python scripts/eval/repair/subcritical_poisson.py --smoke --seeds 42 --limit 40 --grid-batches 4,8 --grid-chunks 128,256 --rho 0.40 --verify-archive /content/<uploaded>.zip
```

A smoke records its GPU, library stack and model identity. It is allowed on
other hardware and with unverified source, and records both. Once it is done,
interrupting it once and re-running the same line with `--resume` exercises
the resume validators on real CUDA output.

Under `/4`, review these before scaling:
* the `[warm-up]` log lines and `engine_warmup` (status, per-stage
  `elapsed_s`, `forward_shapes_observed`);
* the first sweep point's first-token time and maximum submission lag,
  compared with the other three;
* the resumed session's own `engine_warmup` entry in `run_history`.

Reviewing these means comparing them. Nothing in the run is tuned to them.

Exercises the phase splice, the fresh static sweep, both statics, RuleOnly,
CARL-Repaired, CARL-Expanded, the valid-reward warm-up, the arrival and ρ
reporting, and the embedded provenance. `--smoke` (not `--allow-cpu-smoke`): on a
T4 this is a real CUDA run, but it is still marked `run_mode: smoke`, is never a
result, and writes under `smoke_runs/subcritical_poisson/`, never `docs/eval`.

A smoke at a *very* small `--limit` can legitimately trip the per-run reward
gate — with only a handful of control cycles the reward stream has almost no
spread. That is a property of the smoke size, not of the experiment. **Do not
lower the threshold to get past it**; raise `--limit`, or read the abort record,
which says exactly this. `--limit 40` is chosen to be past that point.

### Final published run (Tesla T4)

**NOT READY as of 2026-10-22.**
* The frozen saturated artifact was recovered and integrated (2026-10-22, see
  the corrections log). That makes H1/H2 *resolvable*; it does not make the
  run ready, and the cross-regime comparison is descriptive (comparability
  limits below).
* The engine warm-up amendment (`subcritical_poisson/4`) needs a NEW immutable
  ZIP and a locally captured sidecar.
* The T4 smoke that ran under `/3` must be repeated under `/4` and reviewed.

Every pre-registered quantity is stated explicitly. Nothing relies on a default.

```
python scripts/eval/repair/subcritical_poisson.py --seeds 42,43,44,45,46,47,48,49,50,51 --limit 200 --grid-batches 2,4,8,12,16,24,32 --grid-chunks 64,128,256,512 --rho 0.40 --mu-artifact docs/eval/raw/repair/load_sweep_b7.json --verify-archive /content/<uploaded>.zip
```

### Resume, after an interruption (or after restoring a snapshot)

The same line with `--resume` appended. It refuses unless the stored
fingerprint matches on every key: source, GPU and library stack, model
identity, and the rest. It also refuses outright if a reward gate has fired,
or if fresh evidence was invalid.

```
python scripts/eval/repair/subcritical_poisson.py --seeds 42,43,44,45,46,47,48,49,50,51 --limit 200 --grid-batches 2,4,8,12,16,24,32 --grid-chunks 64,128,256,512 --rho 0.40 --mu-artifact docs/eval/raw/repair/load_sweep_b7.json --verify-archive /content/<uploaded>.zip --resume
```

---

## Corrections log

**2026-10-21 — factual and execution corrections, before any GPU run.** No
hypothesis, prediction, falsification criterion, statistic, threshold, seed,
request count, grid, treatment, workload shape, ρ, μ or λ was changed, and no
generated arrival sequence moved (pinned by test). Corrected:

* the claim that this document is "copied verbatim" into the artifact (it is a
  structured summary plus this document's SHA256);
* the H5 metric's field name (`empirical_mean_interarrival_s`, as the code
  records it);
* the per-run gap count in the cost estimate (99 + 99, not 99 + 100);
* the provenance section (the archive digest is `archive.sha256`; Colab now
  verifies the extracted tree against an archive-member manifest; the resume
  fingerprint uses the recomputed digest);
* the checkpointing / resume guarantees, to describe what is now implemented:
  atomic writes, per-unit sweep evidence, the calibration digest, validated raw
  evidence, start-up failures that never touch the checkpoint, terminal
  reward-gate failures, and separated smoke / published destinations;
* the T4 smoke command, which now needs `--smoke`.

This document's SHA256 is a resume-fingerprint key, so these edits make any
checkpoint written before them non-resumable. That is intended.

*Status as of 2026-10-21 (superseded by the entry below):* "Pending, NOT
applied. Proposed amendments to H5 (multiplicity), to H1/H2 (missing saturated
reference values and H2's denominator), to the paired CI (Student-t with
n = 10), and additional interpretation reporting, await the author's
approval."

**2026-10-22 — authorised statistical amendments, before any GPU run.**

The author authorised these amendments explicitly. No hypothesis direction,
seed, n, grid, treatment, workload shape, ρ, μ, λ, arrival RNG, controller
cadence, arm definition, reward function, weight, threshold or convergence rule
changed. No generated arrival sequence moved; they are pinned by test.

**A. H5.**
* **Why:** 20 unadjusted 95% checks fail at least once with probability
  1 − 0.95²⁰ ≈ 0.64 for a correct generator. The old rule would then have
  voided every load-regime conclusion. The 2026-10-21 proposal also named the
  wrong distribution, Gamma(n−1, λ), which is the distribution of the **sum**;
  the mean is Gamma(k, kλ).
* **Superseded wording, verbatim:**
  * *Prediction:* "within the reported 95% interval around 1/λ, for every phase
    and seed."
  * *Falsified if:* "outside it — which would mean the run does not sit at the
    stated ρ and **no load-regime conclusion may be drawn from it**."

**B. H2's metric.**
* **Why:** "mean level" did not say whose level.
* **Superseded wording, verbatim:** "|mean paired difference| / mean level,
  comparisons A, C, D." It is now the comparator's mean throughput, identical
  at both operating points, and B is excluded.
* **H1/H2 status:** unresolved while the frozen saturated artifact is missing.
  *(Superseded 2026-10-22: the artifact was recovered; see the final entry of
  this log. This entry is kept as written.)*

**C. Paired CI.**
* **Superseded wording:** "95% paired CI", computed as mean ± 1.96 · sd / √n.
* The primary interval is now Student-t at the actual n. The z interval is
  retained as a labelled secondary, for comparison with the frozen saturated
  analysis.

**D. Descriptive phase / regime reporting** added: scheduled phase, inferred
regime, drain timing and mixed execution. Tokens per busy-second is
deliberately **not** added.

**Prior exposure to the seeded arrival sequences — disclosed, not minimised.**
These amendments do **not** predate all inspection of the draws.
* **What was generated before the amendments:**
  * the evaluation and held-out arrival sequences (seeds 42–51 and 999 at
    n = 200), generated repeatedly during development and testing;
  * their SHA256 digests, computed and pinned in the test suite;
  * earlier tests, which build these workloads and their arrival reports.
* **What was done for the amendments:** the thresholds (α = 0.05/20 per cell,
  exact Gamma quantiles) were fixed analytically from λ and n. No n = 200 H5
  cell statistic was computed or inspected to choose them.
* **What the tests observe:** the new tests compute family reports on short
  n = 20 draws for seeds 42/43, which are different, shorter sequences. They
  observe that those cells fall inside their bounds.
* **What cannot be established:** local evidence cannot show whether any
  earlier session viewed the realised n = 200 means.

**2026-10-22 (later) — engine warm-up: an operational amendment made AFTER
inspecting the T4 wiring smoke.**

This amendment follows inspection of smoke data, and it is disclosed as such.
The smoke ran `subcritical_poisson/3` with seed 42, 40 requests and a 2×2
held-out sweep (batch sizes 4 and 8, chunk sizes 128 and 256). Its evidence is
preserved read-only and unchanged in `subcritical_poisson_smoke_evidence.zip`.

* **What was observed.** Sweep point `mb4_cs128` was the first measured unit
  of the process.
  * The first request was submitted at 0.0000179 s; its first token came at
    8.100 s.
  * The second request was scheduled for 0.189 s but submitted at 8.100 s.
  * The maximum submission lag was 7.912 s, and TTFT p99 was 10 858 ms.
  * The other three sweep points, and all five evaluation units that followed
    in the same process, had their first token at about 0.066–0.072 s and a
    maximum lag of about 0.046–0.088 s.
* **What the code confirms.**
  * `load_model_and_tokenizer` loads weights but executes no forward pass.
  * The liveness probe reads scheduler attributes only.
  * `ablation_live._serve` starts its arrival clock before the first
    `sched.step()`, and releases arrivals from the same synchronous loop that
    steps the engine, so one slow step also delays every submission behind it.
  * On CUDA the attention path lazily imports two `@triton.jit` kernels (fused
    RoPE and FA2), which compile on first use.
  * cuBLAS kernels may load lazily per GEMM shape.
* **What is not known.** The 8 s was never broken down into Triton
  compilation, CUDA or cuBLAS initialisation, or anything else. This amendment
  does not claim to know which.
* **What changes.** A declared engine warm-up, `engine_warmup/1`, now runs
  before the first measured unit of each process (see **Engine warm-up**
  above). `EXPERIMENT_VERSION` becomes `subcritical_poisson/4`, and the
  warm-up protocol is a fingerprint key.
  * Every checkpoint and every unit written under `/3` is refused. The smoke's
    measurements are not migrated or reused, and are not results.
* **What does not change.** ρ, μ, λ, seeds, n, the grid, treatments, the
  workload and arrival RNG, controller cadence, the reward, scales, gates,
  arms, static-selection objectives and the convergence rule are unchanged.
  So is the controller's reward-validity cold start (`warmup_cycles`), which
  is a separate concept from engine warm-up. No hypothesis was rewritten.
* **What it does not claim.**
  * It does not make the single-threaded arrival loop precise. Submission lag
    from ordinary step durations remains, and is still reported.
  * It does not cover every prefill width or context length (see the coverage
    limits above), so some first-use cost may remain inside a measured unit.
  * Neither of these residues has been measured. CPU tests show only ordering,
    isolation and accounting; they show no T4 timing improvement.
* **Comparability.**
  * The frozen saturated final-hardening runs had no engine warm-up. Whether
    their first measured unit absorbed a similar start-up cost was not
    verified. A first-unit latency compared across the two operating points
    therefore carries an unquantified asymmetry. This changes nothing in the
    frozen evidence or the manuscript.
  * Within this experiment, every process now warms up before measuring, so
    the first unit of a fresh run and the first unit after a resume start
    under the same declared conditions.

**2026-10-22 (later still) — the frozen saturated reference was recovered and
integrated. Recorded AFTER the operational T4 smoke results had been inspected
and BEFORE the full sub-critical run.**

This timing is disclosed, not minimised. The smoke was operational (seed 42,
40 requests, a 2×2 sweep), its results are not scientific evidence, and none
of them is used here.

* **What was recovered.**
  * **Source:** `D:\final_hardening_FINAL.zip`, sha256 `39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae`.
  * **Member:** `docs/eval/final_hardening_results.json`, raw-byte sha256
    `f1c1c95e2740abc4ee9b45c65938e24f1144a04357e0d85b377496b591431ecd`. It was copied into the tree byte-for-byte and was never
    reserialised.
  * **Raw seed files:** 40 of the archive's 50 files under
    `docs/eval/raw/final_hardening/`, for Static-Tput, RuleOnly, CARL-Repaired
    and CARL-Expanded at seeds 042–051. Each is pinned by its own raw-byte
    sha256.
    * They are needed because the results JSON stores per-seed throughput as
      unlabelled lists; only the raw files bind each value to its seed.
    * Static-Utility's raw files are not read, because B is outside H2.
  * **Line endings:** `.gitattributes` marks these files `-text`, so
    `core.autocrlf` cannot rewrite their bytes.
  * **Not touched:** the ZIP itself, the existing smoke evidence, source
    bundles and sidecars were not modified.
* **What is verified before H1/H2 read it** (`saturated_reference.py`,
  validator `saturated_reference/1`):
  * **Hashes:** the pinned raw-byte hashes, and strict JSON.
  * **Identity:** `final_hardening`, Tesla T4,
    `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `torch.float16`, CUDA; seeds 42–51,
    n = 200, held-out seed 999, and the 7 × 4 axes.
  * **Grid:** exactly 28 unique configurations, identical in both tables that
    hold them, with min / max / spread equal to the rows'.
  * **Selections:** Static-Tput is mb 32 / cs 512 and is the grid's argmax;
    Static-Utility is mb 8 / cs 256.
  * **Treatments:** all five are present, with finite measurements.
  * **Seed alignment:** every raw file against the per-seed lists.
  * **Aggregates:** the stored means, SDs and the A–D paired differences
    reproduce from the per-seed values.
  * **Tolerance:** relative 1e-12, for summation order only. On the
    recovered artifact every value reproduces exactly.
  * **Failure:** any failure means no fallback. In particular, nothing falls
    back to the rounded transcription.
* **Recovered quantities (full precision).**
  * **Grid throughput:** min 44.97500716097522, max 96.00104467760788.
  * **Mean throughput:**
    * Static-Tput: 96.15569365964514
    * RuleOnly: 89.02456617457298
    * CARL-Repaired: 92.78791191900132
    * CARL-Expanded: 90.99232507613624
  * **H1, saturated:** 53.1515440149522 %.
  * **H2, saturated:** A 0.0350242571445, C 0.0422731152326,
    D 0.0193515168703.
  * **Historical statistics:** the frozen run's own paired statistics
    (z-intervals, Cohen's d) are carried verbatim and are not recomputed or
    rewritten.
* **What did not change.** No hypothesis, prediction or falsification
  criterion changed. H1 still uses its relative-range definition with the grid
  maximum as denominator. H2 still uses its amended definition, with
  comparators A Static-Tput, C RuleOnly and D CARL-Repaired, and B excluded.
  No significance test is applied to the 28 search points. Unchanged also: ρ,
  μ, λ, seeds, n, grid, treatments, arrival RNG, reward, scales, gates,
  static-selection objectives, controller behaviour, the warm-up protocol and
  the convergence rule.
* **Binding.**
  * The reference's identity is a resume-fingerprint key,
    `saturated_reference_identity`: validator version, status, artifact sha256
    and raw-file manifest digest. Changed or unverified evidence therefore
    never resumes an older checkpoint silently.
  * The reference files are required data files in the archive manifest, so
    the runtime tree is verified against the packaged bytes.
  * A published run refuses to start if the reference does not verify.
* **Comparability limits — explicit.**
  * **Runtime stack:** the frozen saturated run recorded torch 2.11.0+cu128 /
    CUDA 12.8. The current T4 smoke recorded torch 2.11.0+cu130 / CUDA 13.0.
    The current runtime is recorded as-is and is not downgraded.
  * **Engine warm-up:** the saturated harness had no declared engine warm-up;
    sub-critical `/4` runs `engine_warmup/1`.
  * **Model weights:** the historical model revision was not recorded, so
    matching the repository name does not prove identical weights.
  * **Interpretation:** cross-regime comparisons are descriptive. They cannot
    cleanly isolate offered load from every implementation and runtime
    difference between the two runs.
