# CARL Repair Status — engineering ledger

**Local working document. Not committed. Not pushed.**

Started 2026-08-20 from `dfd3820` (branch `main`, working tree dirty: `CLAUDE.md`,
`docs/screenshots/hero_v2.png`, `paper/main.tex` deleted; `notes.md` untracked).

Every row below is either **VERIFIED** (I executed something and it reproduced),
**DISPROVED** (I executed something and the allegation was wrong), **CODE-READ**
(established by reading source, not yet executed), or **BLOCKED** (needs GPU,
which this host does not have — `torch 2.12.0+cpu`, `cuda_available=False`).

Numbers here were measured on this machine unless a row says otherwise. No GPU
number in this document was produced by me; GPU rows are marked BLOCKED and carry
a hand-off command instead.

---

## Phase 1 — Is the bandit inert?

### R1. LinUCB never leaves arm 0 at the paper's α=0.5 — **VERIFIED**

- **Artifact:** `docs/eval/raw/repair/bandit_null_check.json`
- **Script:** `scripts/eval/repair/bandit_null_check.py`
- **Measured:** 5 seeds (42–46) × 6 α values, NON-STATIONARY 20→20, simulation.
  At α ∈ {0.1, 0.25, 0.5}: `never_left_arm0 = True` on every seed, and CARL's
  throughput is **exactly equal** to `RuleOnly` (float equality, not "within
  noise") on every seed. `carl_equals_rule_only_all_seeds: true`.
- **Also measured:** `static_best_is_a_default_config: true` — the harness's
  Static-Best resolves to `DEFAULT_CONFIGS[BATCH]`, i.e. CARL's own arm 0 for
  that regime.
- **Status:** the cold review's central allegation is correct on this substrate.

### R2. Why it never explores — **VERIFIED** (this is the actionable bug)

From the UCB trace at seed 42, α=0.5, INTERACTIVE regime:

| cycle | ‖x‖ | exploit(arm0) | explore(arm0) | exploit(untried) | explore(untried) |
|---|---|---|---|---|---|
| 0 | 0.139 | 0.000 | 0.069 | 0.000 | 0.069 |
| 2 | 0.534 | 0.014 | 0.267 | 0.000 | 0.267 |
| 5 | 0.517 | 0.339 | 0.193 | 0.000 | 0.259 |
| 19 | 0.508 | **0.609** | 0.108 | **0.000** | 0.254 |

An untried arm has `b = 0`, so `θ = A⁻¹b = 0`, so its exploit term is **0** —
and its total UCB is capped at `α·‖x‖ ≈ 0.25`. Meanwhile arm 0, having observed
rewards around 0.75, climbs to ≈ 0.61 + 0.11 = 0.72. The optimism term can never
close a 0.46 gap.

Root cause is the interaction of two choices:

1. **Rewards are strictly positive and unshifted** (`utility()` returns ≈[0,1],
   observed mean ≈ 0.78). LinUCB's zero-initialised `b` therefore encodes the
   *pessimistic* belief that an untried arm is worth 0, not the optimistic
   belief that it is worth `r_max`.
2. **Feature normalisation drives ‖x‖ ≈ 0.5** (`state._FEATURE_SCALES` divides
   by generous characteristic scales; `gpu_utilization` reads 0.0 in simulation,
   several features sit near 0.03). The confidence radius scales with ‖x‖, so
   the optimism budget is ≈ 0.5α.

To make exploration possible at all you need `α·‖x‖ ≳ r̄`, i.e. **α ≳ 1.5** at
the current scaling. The paper uses 0.5.

- **Standard fixes** (both are textbook LinUCB practice, neither is tuning to win):
  centre the reward on a running baseline so "untried = 0" is neutral rather than
  pessimistic; and add an intercept feature so ‖x‖ ≥ 1.

### R3. Exploration, when it happens, *loses* — **VERIFIED**

α=1.0/2.0/5.0 do explore, and throughput falls monotonically away from RuleOnly
on all 5 seeds (e.g. seed 42: 62.62 → 60.43 → 60.83 → 60.01). This is **not**
evidence that exploration is bad in general; it is evidence that the simulator's
optimum is CARL's initialisation, so any deviation is pure loss. See R4.

### R4. The simulation cost model is circular — **VERIFIED (code-read + executed)**

`scripts/benchmark_carl.py`:

```
throughput = _REGIME_BASE[regime].tps * (0.5 + 0.5 * m)
m          = 1 - ‖config − DEFAULT_CONFIGS[regime]‖ / √6      # _match_score
```

- Oracle (`OracleAgent`) returns `DEFAULT_CONFIGS[true_regime]`.
- Arm 0 of every regime (`config_arms`) **is** `DEFAULT_CONFIGS[regime]`.
- `best_static_config` searches `{CARLConfig()} ∪ {DEFAULT_CONFIGS[r] ∀r}` — six
  candidates, five of which are the answer key.

So the reward landscape's global maximum is, by construction, the point the
bandit is initialised at and the point the oracle returns. Confirmed by
execution: `carl_linucb` throughput in `docs/eval/oracle_results.json` is
bit-identical to `oracle` in phases 1–2 (39.71338147278326 both).

**Consequence:** every simulation-derived number in the paper (ablation,
workload suite, oracle gap, α sweep, trace replay, 100k stability, and the
headline `d=23.17 / p=2.3e-41` statistics) is downstream of a model that cannot,
even in principle, show learning to be useful.

### R5. Reward degeneracy is TWO different failures — **VERIFIED (sim) / CODE-READ (GPU)**

- **In simulation the reward is *not* degenerate:** variance 0.0027 across 40
  distinct values at seed 42. The sim failure is R2 (optimism scaling), not a
  flat reward.
- **On GPU the reward *is* degenerate.** `docs/eval/raw/adaptation/decisions_042.csv`
  contains exactly two reward values ever: `0.8` (cycles 0–2) then `0.3` for all
  remaining cycles. Decomposing against `utility()` weights (0.3/0.3/0.2/0.2):

  `0.3 = 0.3·(1.0) + 0.3·(1−1) + 0.2·(1−1) + 0.2·(0)`

  i.e. `throughput_norm` saturated at 1.0 (`throughput_ref=50` vs ≈85 tok/s
  measured), TTFT violated 100% (200 ms SLO vs ≈2200 ms measured), TPOT violated
  100% (50 ms SLO), `cache_hit_rate` = 0. All four terms pinned.

  These two failure modes need different fixes and must not be conflated.

### R6. The three adaptation "seeds" are byte-identical — **VERIFIED**

`diff docs/eval/raw/adaptation/decisions_042.csv decisions_043.csv` → no output.
Same for 044. `total_cumulative_regret` is `2.583333333333333` in all three to 15
decimals. This follows mechanically from R5: with a constant reward the
controller is deterministic, so the seed cannot influence it.

### R7. CARL converged to a non-optimal arm — **VERIFIED**

`docs/eval/adaptation_results.json`: `final_arm_per_regime.interactive = 3`,
while `oracle_arms_per_regime.interactive.best_arm = 0`. The paper's intro says
CARL "converges to the per-regime optimum". It converged to arm 3.

Additionally the "oracle" ranking is confounded by cycle ordering: arm 0's mean
reward of 0.633 comes from cycles 1–3 (before the reward collapse at R5); arms
1–3 were only ever tried after it. The ranking reflects *when* an arm ran.

---

## Phase 6 — Provenance audit of committed results

### R8. Flagship GPU tables were not produced by their scripts — **VERIFIED**

`docs/eval/ablation_live_results.json` carries `"status": "RECONSTRUCTED"` and
its schema does not match `ablation_live.py::_finalize()`:

| field | script writes | committed JSON has |
|---|---|---|
| `subsystem_contributions` | flat `{name: delta}` | `{definition, ranked:[…]}` |
| oracle gap | `oracle_gap_pct` (float) | `oracle_gap` (object) |
| `linucb_vs_thompson` | `carl_full_linucb_tput`, … | `linucb_tps`, `thompson_tps`, … |
| `static_best_selection` | written | **absent** |
| `live_effective_configs` | written | **absent** |
| `scope_note` | written | **absent** |
| `carl_overhead`, `dynoracle` | written | **absent** |
| `slo_violations` | **never written anywhere in repo** | present per config |

`docs/eval/raw/ablation/` — the directory `_save_raw()` writes per-seed data to —
**does not exist**. `failure_cases_results.json` is likewise reconstructed
("from verified Colab GPU run output (PDF artifact)") with
`environment.captured: false, gpu: null`.

**Handling:** preserved, not deleted. Moved to `docs/eval/legacy/` with a README
recording provenance. Must be regenerated on GPU before any citation.

### R9. Eleven experiments are code-only — **VERIFIED**

No results file exists for: `knob_attribution`, `knob_recovery`, `arrival_probe`,
`distribution_shift`, `static_slo`, `feature_importance`, `feature_noise`,
`global_noise`, `classifier_robustness`, `cross_model_extended`,
`static_best_selection`. `docs/eval/raw/distribution_shift/` contains only
`.gitkeep`.

### R10. `slo_violations: 0` alongside p99 = 61 s — **VERIFIED, contradictory**

`ablation_live_results.json` AutoTuner: `ttft_p99_mean: 61386.1`,
`slo_violations: 0`, against `slo_ttft_ms: 200.0`.
`failure_cases_results.json` `memory_pressure`: `ttft_p99_mean: 43232.0`,
`slo_rate_mean: 0.0`; while `single_queue` reports `slo_rate_mean: 100.0`. Units
and polarity are inconsistent between the two files and within one of them.

Since `slo_violations` is written by no script (R8), the field is an artefact of
hand-transcription and carries no measurement.

### R11. A named intro claim is false against the artifact — **VERIFIED**

`paper/sections/01-intro.tex` claims that on "short prompts, long prompts,
interactive-only, batch-only … the mean throughput difference is exactly zero,
with Cohen's d = 0 and p = 1.0". `docs/eval/statistical_validation_results.json`:

| workload | d | p |
|---|---|---|
| short_prompts | 0.0 | 1.0 |
| interactive_only | 0.0 | 1.0 |
| **long_prompts** | **−0.352** | **0.693** |
| **batch_only** | **−0.352** | **0.693** |
| long_context (not mentioned) | −1.516 | 0.167 |

Two of the four explicitly named workloads do not have d = 0.

Separately: where d = 0 *does* hold it is an identity, not a statistical result.
`best_static_config` returns `DEFAULT_CONFIGS[regime]` on a stationary workload,
which is CARL's arm 0, so both agents run the identical config on the identical
seed and produce bit-identical output (verified: `40.035801` for both on
stationary INTERACTIVE).

---

## Environment constraint

This host has **no GPU** (`torch 2.12.0+cpu`, `torch.cuda.is_available() == False`).
Every GPU-dependent phase is therefore implemented and unit-tested here but
**cannot be executed here**. Those phases produce code + a hand-off command, and
are marked BLOCKED. No GPU number in this repair pass was invented.

---

## Status board

| # | Issue | Status | Fix |
|---|---|---|---|
| R1 | LinUCB inert at α≤0.5 | VERIFIED | Phase 3 reward + optimism repair |
| R2 | Optimism term too small (‖x‖≈0.5, unshifted reward) | VERIFIED | intercept feature + reward centring |
| R3 | Exploration loses because sim optimum = init | VERIFIED | Phase 2 simulator rebuild |
| R4 | Simulator circular (oracle == arm 0 == static-best) | VERIFIED | Phase 2 |
| R5 | GPU reward fully saturated (4/4 terms pinned) | VERIFIED (from committed CSV) | Phase 3 |
| R6 | Adaptation seeds byte-identical | VERIFIED | consequence of R5 |
| R7 | Converged to arm 3, oracle says arm 0 | VERIFIED | claim retraction |
| R8 | Flagship JSONs hand-authored | VERIFIED | quarantine + GPU regen |
| R9 | 11 experiments code-only | VERIFIED | run what CPU allows; hand off GPU |
| R10 | SLO accounting contradictory | VERIFIED | Phase 12 |
| R11 | Intro claim contradicted by artifact | VERIFIED | claim ledger |

---

# Repair outcomes (end of pass)

## Phase 2 — simulation substrate: REPLACED

`src/eval/engine_model.py`. Throughput/latency emerge from
`step_time = host_overhead + kernel_floor + b·c_dec`; nothing in the module knows
what a good configuration is, and a test asserts it imports nothing from
`src.carl`. Oracle is found by coordinate ascent **against the whole-episode
objective**; Static-Best is searched over a 210-point grid strictly wider than
the bandit's arm set.

Two solver bugs found and fixed during the pass, both worth recording because
both would have produced misleading results:

- **Oracle solved per-phase in isolation lost to Static-Best on 4/10 scenarios.**
  Cause: phases inherit the queue the previous phase left behind, and an
  isolated solver cannot see that carry-over cost. An oracle that loses is not
  an oracle. Fixed by solving in episode context.
- **`Static-Best-SLO` scored 0.3417 while `Static-Best-Tput` scored 0.4851 on the
  same utility** — the selector was picking a config its own objective rated
  worse, for the same isolation reason. Fixed identically.

## Phase 3 — reward: REPLACED (v1 retained)

`src/carl/reward.py`. `tps/(tps+t_half)` and `1/(1+(x/target)^k)`: strictly
monotone, never clipped, `target` still the 0.5 crossing so SLOs keep their
meaning. Scales derived from the observed operating range, not chosen. 14 tests.

## Phase 4 — reward diagnostics: ADDED

`docs/eval/reward_diagnostics.md` + `check_non_degenerate`, which raises rather
than letting a flat reward through. Measured per-arm spread on the repaired
substrate: 0.030–0.189, no degeneracy on any workload.

## Phase 5 — the null ablation: RUN. CARL TIES RULEONLY.

11 scenarios × 10 seeds. Largest |Δ| 0.0013 reward units; CARL behind on 5 of 11;
bit-identical on 3. Full table in `docs/paper/RESULTS_DECISION.md`.

**The deeper result:** the per-phase **Oracle** equals a single tuned static
config on 9 of 11 scenarios. Perfect regime knowledge is worth ≈0, so no learner
can be worth more.

## Phase 9 — AutoTuner: REPAIRED

`ClosedLoopAutoTunerController` feeds the profiler the real phase breakdown its
own configuration produced. Verified: two configs → two different observed
profiles. It now loses by 2–10% (a plausible hill-climber exploration cost)
rather than the 2.4× catastrophe the open-loop version produced.

The legacy `benchmark_carl.AutoTunerAgent` is left in place and pinned by a test
that asserts it still keys on `_BOTTLENECK[true_regime]`, so the defect cannot be
silently "fixed" without updating the claim ledger.

## Phase 10 — arrivals: IMPLEMENTED, NOT RUN (no GPU)

`build_arrivals()` (burst / poisson / deterministic) in `src/carl/live.py`;
`ablation_live.py` releases requests when due and records per-stage timestamps.
Bulk dump retained as the `burst` stress mode.

**Superseded in status by Phase 24**, which puts these arrivals to work at a
*derived* ρ = 0.40 and adds the one thing missing here: `_serve` could release
phase-0 arrivals on a schedule but still dumped phase 1 on a closed-loop
trigger, so a phase-specific arrival rate could not be expressed. See
`phase1_open_loop`.

## Phase 12 — SLO accounting: UNIFIED

`src/carl/slo.py`. Request is the statistical unit; `_rate` is always a fraction,
`_pct` always a percentage; `<=` is satisfaction; unfinished requests are
reported separately, never absorbed. `check_consistency` detects the exact
contradiction in the quarantined file (0 violations against a 61 s p99). 10 tests.

## Phase 14 — per-knob attribution: NORMALISATION REMOVED

Raw deltas only, now reported with `delta_pooled_std`, `delta_exceeds_1std`,
`delta_exceeds_2std`. The rescale-to-28.51 logic is gone.

## Phases 15/16 — intervention: IMPLEMENTED, NOT RUN (no GPU)

`scripts/eval/repair/batch_intervention.py`. Refuses to run without CUDA; writes
its artifact incrementally after every row. Simulation preview of the Phase 16
question: batching benefit 1.84× eager → 1.37× with CUDA graphs.

## Phase 22 — statistics: seed path verified live

`seeds_distinct` is asserted for every method in every scenario and is `True`
throughout; a test fails if five seeds ever produce five identical episodes.
Paired comparisons report `all_diffs_zero`, which is how a d=0/p=1.0 identity is
distinguished from a statistical tie.

---

## Not done, and why

| Phase | Status | Reason |
|---|---|---|
| 11 static-SLO on GPU | not run | no GPU on this host |
| 13 spec decoding | audited only | cannot be evaluated; claim scoped in ledger (B4) |
| 17 classifier robustness | not run | `classifier_robustness.py` untouched; lower priority than 15/16 |
| 18 distribution shift | not run | needs GPU for the version that matters |
| 19 long-horizon | not rebuilt | pointless until the substrate question (15/16) is settled |
| 20 discounted/sliding LinUCB | not built | premature: the repaired learner already ties, and headroom is ≈0 |
| 21 cross-model | not re-run | no GPU |

Phase 20 deserves an explicit note: the brief says to measure before replacing
LinUCB. Measured. There is no adaptation failure to fix — there is no headroom
to adapt into. Adding a forgetting mechanism now would be solving a problem the
data says does not exist.

---

## Phase 23 — final hardening on a T4: **EXECUTED, FROZEN**

The first decisive result in this document that was **not** produced on this
host. Run on Colab; the artifact
(`docs/eval/final_hardening_results.json`, raw in
`docs/eval/raw/final_hardening/`) is **not in this working tree**, so the numbers
below are transcribed from the recorded result and are not recomputed here.

*Recovery note (2026-10-22):* the artifact has since been recovered and placed
in the tree byte-for-byte.
* **Archive:** `final_hardening_FINAL.zip`, sha256 `39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae`.
* **Member:** sha256 `f1c1c95e2740abc4ee9b45c65938e24f1144a04357e0d85b377496b591431ecd`.
* **Raw seed files:** the 40 files H2 needs.

The frozen numbers below are unchanged, and remain the rounded transcription.
`saturated_reference.py` verifies the artifact at full precision (see the
sub-critical section).

**Status: FROZEN.** Not to be re-run, recomputed, softened or strengthened. A
later experiment may *scope* it by measuring a different operating point; it may
not overwrite it.

- **Script:** `scripts/eval/repair/final_hardening.py`
- **Pre-registration:** `docs/eval/PREREGISTRATION_final_hardening.md`, before the run
- **Measured on:** Tesla T4, TinyLlama-1.1B-Chat-v1.0, fp16
- **Design:** seeds 42–51 (n = 10, paired), 200 requests/run, held-out
  calibration/selection seed 999, saturated / non-stationary burst workload
- **Live plane:** `max_batch_size × chunk_size` = 7 × 4 = **28 configs,
  exhaustive** — verified live by `probe_live_dimensions` against the real
  scheduler object
- **Verified inactive in this harness:** speculative decoding, router, KV
  eviction, preemption, CUDA graphs (no `_graph_runner` attached)
- **Static-Tput:** `max_batch_size=32`, `chunk_size=512`
- **Static-Utility:** `max_batch_size=8`, `chunk_size=256`

| comparison | metric | paired Δ | 95% CI | paired *d* |
|---|---|---:|---|---:|
| CARL-Repaired − Static-Tput | `throughput_tps` | −3.3678 | [−3.6633, −3.0723] | −7.06395 |
| CARL-Repaired − Static-Utility | end-of-run `utility_v2` | −0.0245 | [−0.0263, −0.0227] | −8.44215 |
| CARL-Repaired − RuleOnly | `throughput_tps` | +3.7633 | [+3.2932, +4.2335] | +4.96133 |
| CARL-Repaired − RuleOnly | end-of-run `utility_v2` | +0.0363 | [+0.0347, +0.0380] | — |
| CARL-Expanded − CARL-Repaired | `throughput_tps` | −1.7956 | [−2.2754, −1.3158] | — |
| CARL-Expanded − CARL-Repaired | end-of-run `utility_v2` | −0.0277 | [−0.0295, −0.0260] | — |

**Cold start (defect 2's repair, as a positive control):** every adaptive run
withheld exactly **3** invalid-reward cycles; no learner update on a withheld
reward; `rewarded_arm = −1` on those cycles; `withheld_after_first_valid = 0`
everywhere; reward validity monotone after warm-up. Pre-registered H5 passed.

**Convergence:** 0/10 seeds for RuleOnly, 0/10 for CARL-Repaired, 0/10 for
CARL-Expanded, under the rule fixed before the run (≥ 6 decisions; ≥ 2 distinct
arms in the first half; second-half modal share ≥ 0.75; modal share strictly
increased).

**What changed against Phase 1's verdict.** Phase 1 and the simulation
concluded that CARL ties RuleOnly to four decimal places. On real hardware at a
saturated operating point it does **not** tie — it wins by +3.7633 tok/s with a
CI excluding zero. Learning does something; it does not do enough to reach an
objective-matched static configuration. Both halves of that sentence are the
result.

**Allowed wording (the only permitted summary of this point):**

> At the saturated T4 operating point, adaptive reconfiguration is useful but
> insufficient: CARL-Repaired improves throughput and end-of-run utility over
> RuleOnly, but remains below objective-matched static configurations. Expanded
> live configuration coverage further reduces both metrics.

**Forbidden.** "Static tuning always wins" (over-generalises one operating
point); "online adaptation is useless" (contradicted by the RuleOnly
comparison); "CARL converges" (0/10); "Expanded loses because it switches more"
(a mechanism this run did not test); and substituting *mean valid cycle reward*
for *end-of-run `utility_v2`* in either direction. Per-claim detail is in
`docs/paper/CLAIM_LEDGER.md` section D.

## Phase 24 — sub-critical Poisson operating point: **RUN, COMPLETE (78/78 units); CLOSED with reporting limitations**

- **Script:** `scripts/eval/repair/subcritical_poisson.py`
- **Pre-registration:** `docs/eval/PREREGISTRATION_subcritical_poisson.md`
- **Tests:** `tests/test_eval/test_subcritical_poisson.py`
- **Artifact:** `docs/eval/subcritical_poisson_results.json`,
  raw in `docs/eval/raw/subcritical_poisson/` — held in the evidence archives
  listed under *Run record and closeout* below, not in this working tree.
- **Status:** ~~BLOCKED on GPU — this host has no CUDA.~~ *Superseded
  2026-10-25: run on a Colab T4 on 2026-10-23 and closed out. See the end of
  this section.* The script still refuses to run without CUDA unless
  `--allow-cpu-smoke` is passed, and a CPU artifact is stamped `cpu_smoke` and
  is never a result.
- **Campaign history (added 2026-10-26).** This design was executed to
  completion **three** times, not once: **S1** (2026-09-18, September
  pre-registration `b455cc2`, OOM + resume), **S2** (2026-09-19, same protocol,
  uninterrupted) and **O** (2026-10-23, the October pre-registration, OOM +
  resume). S1 and S2 results were viewed and discussed before the October
  amendments. The text below was written when only O was known locally and is
  kept as written; where it says "the T4 run" or "the run" it means **O**. The
  dated correction record is `docs/eval/DISCLOSURE_2026-10-26_subcritical_campaign_history.md`; the
  three-campaign results are under *Three-campaign addendum* at the end of this
  section.

**Not a repair experiment.** Nothing in Phase 23 is alleged to be broken. This
is a **scope** experiment: it asks whether the saturated finding is a property of
adaptation or of the operating point it was measured at. Saturation is precisely
the regime in which `max_batch_size` binds hardest, so it is the regime in which
a fixed configuration has the most to offer.

**Exactly one factor moves: arrivals.** Burst bulk dump → open-loop Poisson,
with a **phase-specific** λ, each at ρ = 0.40 of *its own* measured capacity:

| phase | source row in `docs/eval/raw/repair/load_sweep_b7.json` | μ (req/s) | λ = 0.40·μ (rps) | 1/λ (s) |
|---|---|---:|---:|---:|
| interactive | `saturation["interactive@eager"].mu_max_req_per_s` | 2.9613881136988742 | 1.1845552454795498 | 0.8442 |
| batch | `saturation["batch@eager"].mu_max_req_per_s` | 1.4135291672527908 | 0.5654116669011163 | 1.7686 |

**Why the eager rows — CODE-READ, then cross-checked at run time.**
`CARLConfig.use_cuda_graphs` defaults `True`, and the artifact also carries
`interactive@graph` 5.5638 / `batch@graph` 2.5649. Taking the config field at
face value would set λ ≈ 1.9× too high, so a nominal ρ = 0.40 would in fact run
at ρ ≈ 0.75. But `ablation_live._new_scheduler` never assigns `_graph_runner`,
`ContinuousBatchScheduler.__init__` sets it to `None`, and `_decode_forward`
requires it non-`None` before replaying — every decode is eager. So the arm is
read from the **live scheduler object** (`execution_arm_for`), never from a
config field.

That the eager rows describe the *measured* non-graph path is checked, not
asserted: the `hardware_profile` behind them carries
`decode_per_row_s = 0.009448482794434556` and
`decode_fixed_per_step_s = 0.02546177948065939`, **bit-identical** to
`analysis.hardware_profile_fit_eager` in
`docs/eval/raw/repair/batch_intervention_phase15_16_s42_s43_s44.json` — a real
Tesla T4 / TinyLlama / fp16 run, 54 eager rows, R² = 0.9985. The CUDA-graph
parameters are a separate regression from the same run (R² = 0.9927), which is
what makes the two arms distinguishable at all. `assert_arm_evidence()` re-checks
both facts and raises `AmbiguousServiceRateError` if either stops holding.

**Stated limitation.** μ comes from the calibrated mechanistic substrate, whose
decode parameters are fitted to that T4 run — it is not an end-to-end T4
measurement under this harness's own prompt windows, which are close but not
identical to the load-sweep probe shapes (interactive: probe prompt mean 32 /
output 32 vs harness U[16,32] / `max_new` 32; batch: probe 220 / 64 vs harness
U[128,256] / `max_new` 64). ρ = 0.40 is therefore a **derived target**, so every
run reports `realised_load` — wall clock against the scheduled arrival window —
and a ratio well above 1.0 says the engine set the pace and the realised ρ
exceeded the target. Reported, never fed back into λ.

**One additive harness change.** `ablation_live._serve` gained
`phase1_open_loop` (default `False`, so the historical path is unchanged and
every earlier artifact stays reproducible). The historical harness dumps every
phase-1 request the instant half of phase 0 has finished — a *closed-loop*
trigger under which a phase-specific arrival *rate* cannot be expressed, because
the pattern becomes a function of engine speed. With the flag on, both phases are
released from their own `arrival_offset_s` and the regime shift is recorded when
the first BATCH request actually arrives.

**Re-derived, not reused:** the reward scales and both static baselines. The
saturated selections are properties of a saturated queue; carrying them here
would be the objective-mismatch defect in a new costume. The 28-configuration
plane is swept again on seed 999 under the Poisson workload, feeding the same
three consumers in the same order — MEASUREMENTS → frozen scales → utility →
Static-Utility.

**Held fixed, by import rather than by retyping:** `probe_live_dimensions`,
`freeze_scales`, `select_static_baselines`, `run_utility`,
`convergence_diagnostic`, `CONVERGENCE_RULE`, `_cycle_records`, `_build_bandit`
and `load_source_provenance` all come from `final_hardening`. The frozen
artifact itself is neither read nor written.

**Statistics added:** per-seed sign of every paired difference, and an **exact**
two-sided Wilcoxon signed-rank test by enumerating all 2ⁿ sign assignments over
the observed midranks (no scipy on this host, and n = 10 makes exact
enumeration trivial). It declines and says so when any difference is exactly
zero — a bit-identical pair is an *identity*, not a tie. p-values are never the
headline.

**Expected cost.** Under an open-loop sub-critical stream the wall clock is set
by the **arrival window**, not by the engine: 99/λ_i + 99/λ_b ≈ 84 s + 175 s
≈ **259 s per run** (100 requests per phase give 99 gaps each; the first BATCH
arrival sits on the last INTERACTIVE one), times 28 held-out sweep configs + 50 evaluation runs ≈
**5.5–6 hours on a T4**. That is the price of a stated ρ, and it is why the
artifact is checkpointed after the derivation, after every grid config, after
the static selections, after every (treatment, seed) pair, after every completed
treatment, and on every abort path — and why `--resume` exists.

**Hardening pass (before the T4 run).** Four narrow changes, no redesign:

1. **ρ is now ρ.** The wall-clock ratio was previously mis-named a "realised
   load"; it is now `drain_overrun.arrival_window_overrun_ratio`, explicitly
   flagged as *not* a load. The actual offered load is computed per phase as
   `empirical_rho = empirical_lambda_rps / mu_req_per_s` from the scheduled
   offsets, and reported with its delta from target, its interarrival mean and
   median, the scheduled window, and — separately — the observed per-phase drain
   timing. **No seed is ever rejected or re-run for its realised Poisson rate**;
   that policy is stated in the pre-registration, in the artifact, and pinned by
   a test that greps the driver for seed-filtering paths.
2. **The splice is an invariant.** `assert_phase_splice` raises
   `PhaseSpliceError` unless `first_batch_arrival_s >= last_interactive_arrival_s`
   and every index before `n // 2` is INTERACTIVE and every index after is
   BATCH. Recorded per run: both phases' first/last arrival and
   `phase_boundary_gap_s` (0.0 by construction). The closed-loop dump is not
   restored.
3. **Flatness diagnostics.** Per objective, over the 28-point held-out surface:
   best, second-best, their absolute and relative gap, min/max, absolute and
   relative range, and the count and identities of configurations within 0.5%
   and 1.0% of the best. Purely descriptive — computed *after* the winners are
   fixed, consumed by nothing, and introducing **no** post-hoc tie-break.
4. **`--resume`.** Execution only. A wide fingerprint (operating point, seeds,
   grid, model, dtype, reward, convergence-rule hash, arm-space hash, local
   archive/manifest/diff SHA256s, pre-registration SHA256) must match on every
   key or the resume aborts *without writing*, so the checkpoint it was asked to
   resume survives. A unit counts as done only if its ledger entry **and** its
   durable artifact are present, parseable and complete; otherwise it is
   re-measured. The frozen scales and both static baselines are recomputed from
   the full grid every invocation, so a resume cannot move them — verified end
   to end on CPU: a resumed pass reproduced `t_half=14.3663`,
   `ttft_target=2014.9285`, `tpot_target=272.0139` and the same
   Static-Tput/Static-Utility exactly.

**Second hardening pass (2026-10-21, before the T4 run).** *(2026-10-26: before the **October** T4 run only — S1 and S2 had already run on these seeds; see the disclosure.)* An independent
read-only review returned NO-GO. Each claim was re-checked against the source
before anything changed. What was confirmed, and repaired:

* **A failed `--resume` could destroy its own checkpoint.** `main` wrote
  whatever was in the empty `partial` dict on any error raised before `run_all`
  filled it (model/tokenizer load, liveness probe, a bad `--mu-artifact`, an
  interrupt). Now the whole fingerprint except the execution arm is checked
  before the model loads, the run takes **ownership** of the checkpoint only
  after every refusal has had its chance, and pre-ownership failures are
  recorded in `subcritical_poisson_preflight_errors/` without touching evidence.
* **Writes were not atomic** (`open(path, "w")`). All writes now go through a
  serialise-first, same-directory temp file, fsync, `os.replace`.
* **Sweep rows lived only in the results JSON**, and a resume checkpointed only
  the rows it had walked past. Each sweep point now has its own file, and the
  checkpoint always holds every verified row in grid order.
* **Nothing bound evaluations to the calibration they were measured under.** A
  calibration digest (held-out measurements, frozen scales, weights, both
  selected configurations) is written before the first evaluation and recorded
  in every evaluation unit. After evaluation has started, the sweep is restored
  from that record; a resume that cannot reproduce it refuses.
* **Raw files were trusted once they parsed with the right keys.** They are now
  validated for schema, version, fingerprint, calibration digest, identity,
  seed, n, finite metrics, the exact seeded arrival offsets, nested evidence,
  and (controllers) internal consistency with their own cycle trace.
  Untrusted files are moved to `superseded/`, never deleted, and every rerun or
  recovery is audited.
* **The archive digest in the fingerprint was always null** — the consumer read
  `archive_sha256`, the producer writes `archive.sha256`, and the test fixture
  used the consumer's wrong shape. Fixed and tested against the producer's real
  output. In addition `source_provenance.py --archive` now records an
  archive-member manifest (raw bytes; `sha256_lf` informational), refuses
  non-portable or incomplete ZIPs (checked on `orig_filename`, since Python's
  `zipfile` rewrites `\` to `/` on Windows), and Colab verifies the extracted
  tree against it without git. A published run refuses unverified source; the
  fingerprint uses the digest **recomputed** over the executing tree. **The
  existing `SOURCE_PROVENANCE.json` predates this and cannot clear a published
  run: rebuild the ZIP and re-capture the sidecar locally.**
* **A reward-gate failure lost its evidence and could be retried until it
  passed.** The failed unit's measurements, controller trace and gate evidence
  are written to `failed/` before the error propagates; `--resume` refuses a
  terminal gate failure (including old-format abort records).
* **Smoke and published runs shared paths, and only CPU runs were marked
  smoke.** `--smoke` / `--output-dir` now separate them; a published run must be
  CUDA + the exact registered parameters + verified source, and a fresh run
  refuses to overwrite existing evidence.
* **Text errors:** "uniform prompt 16-64" (the code is 16–32), the 99 + 100 gap
  count (it is 99 + 99), the H5 field name, and the "copied verbatim" claim.
* **The eager-fit cross-check was skipped silently if its file was missing**; it
  now aborts (the file is also a required archive member).

Tests now drive the real `main()` / `run_all()` on CPU with a fake scheduler,
model loader and harness clock, and pin the seeded workload of every seed
byte-for-byte to the pre-pass code.

**Not repaired, because they are scientific changes awaiting the author's
approval** (proposals in the review report; nothing applied): H5's per-seed
95% rule fails by chance for a correct generator (1 − 0.95²⁰ ≈ 64%); H1/H2
cannot be evaluated because the saturated grid ranges and per-treatment mean
levels were never recorded and `final_hardening_results.json` is not available
locally, and H2's denominator is undefined; the paired CI uses z = 1.96 at
n = 10; arrival-limited throughput and probable BATCH-phase misclassification
constrain interpretation. **The T4 run is not cleared while these are open.**

**Third (final) hardening pass (2026-10-22, before the T4 run).** *(2026-10-26: before the **October** T4 run only; see the disclosure.)* A second
read-only review found these; each was re-verified against the code before
fixing.

* **Any CUDA GPU could produce a "published" run.** Fixed in four parts:
  * A published run now requires a Tesla T4, checked on the default device and
    on the device that actually holds the model's weights.
  * The fingerprint binds the GPU, Python / torch / CUDA / cuDNN /
    transformers / huggingface_hub / safetensors versions.
  * It also binds the model's immutable identity: the HF cache snapshot commit
    plus SHA256 of the config, weights and tokenizer files, read offline.
  * An unresolvable model identity fails closed for published runs. Smoke runs
    record whatever they ran on.
* **An abort inside `apply_restore` could overwrite a valid checkpoint with
  one lacking its fingerprint.** Fixed in three parts:
  * The fingerprint, the inherited ledger, every verified row and the
    checkpoint's calibration binding now enter the in-memory checkpoint
    **before** ownership.
  * `main` never writes an abort record that lacks a fingerprint.
  * Fault injection covers `supersede` (OSError and Ctrl-C), the
    calibration-record write, the ledger and the first checkpoint write.
* **A failed evidence write could turn a fired gate into an operational
  interruption.** Fixed in four parts:
  * The verdict is now recorded in memory first, then as a small terminal
    marker, then as full evidence.
  * Write failures are recorded and the original exception still propagates.
  * The handler classifies by recorded state.
  * `--resume` refuses on any marker or failed-unit file.

  If every write fails, the run says it must not be resumed. Nothing on disk
  can then record the verdict.
* **Fresh evidence was never validated; only resumed evidence was.**
  * Every new unit now passes the resume validator on its JSON round-trip
    before it counts.
  * Invalid fresh evidence stops the run, is preserved in `invalid/`, and
    makes `--resume` refuse.
* **Calibration recovery ignored the checkpoint's own digest.** It is now a
  binding digest. Recovery with no binding digest at all is refused, as is a
  calibration record that disagrees with the checkpoint.
* **Path isolation could be bypassed through aliases.** Paths are now resolved
  with `realpath` and `samefile`, and links and junctions are tested.
* **Provenance and durability:**
  * Digests must be hexadecimal SHA256.
  * `--verify-archive` hashes the uploaded ZIP, so a *verified* ZIP digest is
    distinguished from a sidecar-*asserted* one.
  * The durability text no longer claims the old file survives power loss, and
    separates process interruption, atomic replacement, and the unverified
    power-loss / FUSE cases.
  * A Colab persistence and recovery workflow is documented: runtime reset ≠
    browser disconnect, plus manual verified snapshots. No automated in-run
    persistence exists; that is an open operational requirement.

**Authorised statistical amendments applied** (dated in the
pre-registration's corrections log, with the superseded wording and the
prior-exposure disclosure):

* **H5:** exact Gamma(k, kλ) bounds for the mean of k = 99 gaps over a
  20-cell family at α = 0.05/20, with seed 999 separate. A rejection flags
  inconsistency for investigation and voids nothing automatically. Generator
  correctness is checked separately, including a large-sample sampler test.
* **H2:** the denominator is the comparator's mean throughput (A Static-Tput,
  C RuleOnly, D CARL-Repaired), and B is excluded.
* **Paired CI:** Student-t at the actual n is primary; the z interval is a
  labelled secondary.
* **Reporting:** descriptive phase / regime / drain / mixed-execution
  reporting, joined only through scheduler-step identifiers.

**Still open:**
* ~~**H1 and H2 are UNRESOLVED.** `final_hardening_results.json` is not in this
  tree, in any git ref, or in any adjacent project ZIP, so the saturated side
  cannot be filled in.~~ *Superseded 2026-10-22: the artifact was recovered
  from a separate archive and integrated. See the saturated-reference note
  below.*
* **The ZIP and v2 sidecar are not rebuilt.**
* **The T4 smoke ran under `/3` only.** It must be repeated under `/4`, which adds the engine warm-up.
* **No registered model revision exists** for the saturated run, so the model
  identity binds the run only to itself.

The isolated T4 smoke may proceed once the ZIP and sidecar are rebuilt. **The
published run remains NOT READY.**

**Engine warm-up amendment (2026-10-22, after the T4 wiring smoke).**

A T4 smoke has since run under `subcritical_poisson/3`, with seed 42, 40
requests and a 2×2 sweep. Its evidence is preserved read-only.
* **What it showed:** the process's first measured unit, sweep point
  `mb4_cs128`, absorbed an ~8.1 s start-up stall. Its first token came at
  8.10 s, the maximum submission lag was 7.91 s and TTFT p99 was 10.9 s. Every
  later unit in the same process had its first token at about 0.07 s.
* **What was not measured:** the cause — Triton JIT, CUDA or cuBLAS
  initialisation, or something else.
* **What changed:**
  * A declared, bounded engine warm-up, `engine_warmup/1`, now runs once per
    process before the first measured unit and before the run takes ownership
    of the checkpoint.
  * It uses disposable schedulers on the real prefill, chunked-prefill and
    batched-decode paths, restores all RNG states and synchronises the device.
  * It runs only when some unit will be measured.
  * Its protocol is a fingerprint key, and the experiment version is now `/4`,
    so `/3` checkpoints and units, including the smoke's, are refused rather
    than reused.
* **What did not change:** no scientific quantity, and not the controller's
  reward-validity cold start.
* **Comparability:** the saturated runs had no engine warm-up, so a
  first-unit latency compared across the two operating points carries an
  unquantified asymmetry.
* **What is still unproven on hardware:** the T4 effect of the warm-up is
  untested, and residual first-use cost from shapes it does not cover is not
  measured.
* **Before the next smoke:** a NEW immutable ZIP and a locally captured
  provenance sidecar are required.

**Saturated reference recovered and integrated (2026-10-22, after the T4
wiring smoke was inspected and before the full run).**
* **What was recovered.** `final_hardening_results.json` was recovered from
  `D:\final_hardening_FINAL.zip` (sha256 `39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae`). Its member sha256 is
  `f1c1c95e2740abc4ee9b45c65938e24f1144a04357e0d85b377496b591431ecd`, and it was copied byte-for-byte. With it came the 40 raw seed files
  for the four treatments H2 reads.
* **How it is verified.** `scripts/eval/repair/saturated_reference.py` checks
  the hashes, the identity (T4, TinyLlama fp16, seeds 42–51, n = 200, seed
  999), the 28 unique grid points, both selections, all treatments, per-seed
  alignment against the raw files, and aggregate consistency. There is no
  fallback to the rounded transcription.
* **Saturated sides of the hypotheses.**
  * H1: 53.1515440149522 % (100 · (max − min) / max).
  * H2: A 0.0350242571445, C 0.0422731152326, D 0.0193515168703.
  * Both use the existing definitions, unchanged.
* **Binding.** The reference identity is a resume-fingerprint key, and the
  reference files are required data in the archive manifest. A published run
  refuses an unverified reference.
* **Comparability (descriptive only).**
  * Torch 2.11.0+cu128 / CUDA 12.8 (saturated) against 2.11.0+cu130 / CUDA
    13.0 (current T4 smoke).
  * No engine warm-up in the saturated harness.
  * No historical model revision, and a matching repository name does not
    prove identical weights.
  * Cross-regime differences cannot be attributed cleanly to offered load.
* **Before the next smoke.** A NEW immutable ZIP and a locally captured
  sidecar are required. The existing ZIP and sidecar predate both the warm-up
  and this integration.

**Stopping rule.** After this experiment the campaign stops unless a genuine
validity defect is found.

### Run record and closeout (2026-10-25)

**Outcome.** No demonstrated validity defect. The campaign is **closed with the
reporting limitations below**. No further GPU work is justified. Per-claim
wording is in `docs/paper/CLAIM_LEDGER.md` section E. The two read-only reviews
behind this closeout were written in **one session by the same reviewer**, the
second after seeing the first, so they are not independent reviews. Every
number below was re-derived from the archives.

**Evidence** (all SHA256 verified; both ZIPs pass `testzip`):

| item | path | SHA256 |
|---|---|---|
| final checkpoint | `D:\checkpoint_1792796860834369199_f793587b2717.zip` (+ `.verified.json`) | `cb95067ba0bf8f183d78f3c49cfccb74f8ac2eeebd9898393fcd564aa8083417` |
| final results member | `docs/eval/subcritical_poisson_results.json` in that ZIP | `f793587b2717d9b64e3b5335eeb38aef6b70c73a0934f40c2e49db9b0c784433` |
| pre-resume checkpoint | `D:\checkpoint_1792793381218572067_d125f3c55192.zip` (+ `.verified.json`) | `fcf6c2b5579ed7cfd585446c03498c39bb13a753e63d49ca1361cd8672844ae9` |
| pre-resume results member | same member name in that ZIP | `d125f3c5519240e5e0dd49cf5993cfa109aec1627f2e5dc6dce39f324c66eaa8` |
| executed source | `mini-vllm-bundles/subcritical-reference-20261023-171950/mini-vllm-subcritical-reference.zip` | `d78a24ed55bd90aa346f76d15a5ee5b98cdea216322883bd71b0acd9fe75efed` |
| source sidecar | `SOURCE_PROVENANCE.json` beside it (`source_provenance/2`) | `44bcd27d213d1792f8738850ac2f1d605993f74461443da11cd89ea1924eb9be` |
| saturated reference | `D:\final_hardening_FINAL.zip` | `39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae` |

* **Source manifests.**
  * The archive manifest recomputes to `64aeffc5…fe` over 209 files, matching
    the results' `source_verification`.
  * The source manifest recomputes to `094aef84…bc` over 165 `.py` files.
  * The archive's 80 unmanifested members are docs and data, not code.
* **Identity.**
  * All 78 units carry fingerprint `894cc422…3766`.
  * All 50 evaluation units carry calibration `f7b8593f…0e64`.
* **Completeness.** The run holds 28 held-out sweep units on seed 999 and 50
  evaluation units (5 treatments × seeds 42–51 × 200 requests).
  * Arrival offsets regenerate from the seeds.
  * There are no duplicates or omissions, and every metric is finite.

**Interruption and resume.** Session 0 started at 17:13:32Z and wrote 66 units.
Its last write was `eval:CARL-Repaired:seed049` at 22:09:39.083Z. It then hit a
CUDA OOM at 22:09:39.312Z.
* **Where the OOM happened** (traceback in the operational log
  `.codex/attachments/c48d93c1-…/Pasted text.txt`):
  1. `run_one` (`subcritical_poisson.py:2087`)
  2. `_new_scheduler` (`ablation_live.py:266`)
  3. `PagedKVCache(...)` (`scheduler.py:331`)
  4. `V_pool` allocation of 176 MiB (`kv_cache.py:133`)

  This was the construction of Repaired seed 050, before any of its
  measurement. **Cause: unresolved.** No memory telemetry exists, and
  allocation arithmetic alone does not show a leak.
* **Session 1** (`--resume`, log `.codex/attachments/c5dafb08-…/Pasted text.txt`)
  started at 22:16:13Z.
  * It recorded the same hostname, `4cc7b9406f20`.
  * It reran the engine warm-up and restored the sweep from the calibration
    record.
  * It measured exactly 12 units: Repaired seeds 050–051 and Expanded
    seeds 042–051.
  * `units_rerun_on_resume = []`, and nothing was superseded.
* **Preservation, verified byte for byte.** All 66 pre-resume raw units and
  `calibration.json` are identical in the final archive.
  * The final archive adds exactly the 12 units above.
  * The pre-resume ledger is an exact prefix of the final one, and session 0's
    `run_history` entry is identical in both.
  * This proves consistency of provenance and content. It does **not** prove
    identical physical hardware, because no GPU UUID is recorded.

*(2026-10-26: everything in this closeout is campaign **O** only. "Registered"
below means "specified in the October pre-registration", which was frozen after
S1/S2 results existed; it is not a protocol fixed before any result was seen.
See the addendum and the disclosure.)*

**Registered results.** Paired over seeds 42–51 (n = 10), a − b. The 95% CI is
Student-t, the registered primary. *d* is the paired d_z (mean / SD of the
differences). p-values are robustness checks only and are not shown.

| | comparison (a − b) | metric | mean Δ | 95% t CI | d_z | seeds +/− |
|---|---|---|---:|---|---:|---|
| A | Repaired − Static-Tput | `throughput_tps` (primary) | −0.02864 tok/s | [−0.04595, −0.01133] | −1.18 | 2/8 |
| A | | end-of-run `utility_v2` | −0.01775 | [−0.02376, −0.01174] | −2.11 | 0/10 |
| B | Repaired − Static-Utility | end-of-run `utility_v2` (primary) | −0.13961 | [−0.15925, −0.11997] | −5.08 | 0/10 |
| B | | `throughput_tps` | −0.04386 tok/s | [−0.07672, −0.01100] | −0.95 | 0/10 |
| C | Repaired − RuleOnly | `throughput_tps` | −0.03247 tok/s | [−0.05829, −0.00665] | −0.90 | 2/8 |
| C | | end-of-run `utility_v2` | −0.01957 | [−0.02522, −0.01393] | −2.48 | 0/10 |
| D | Expanded − Repaired | `throughput_tps` | +0.00312 tok/s | [−0.01616, +0.02240] | +0.12 | 4/6 |
| D | | end-of-run `utility_v2` | −0.00925 | [−0.01714, −0.00135] | −0.84 | 1/9 |

* **B in words:** the mean paired end-of-run `utility_v2` difference was
  approximately −0.140, with negative differences on all ten seeds.
* **No percentages for `utility_v2`:** it has no meaningful zero.
* **D's throughput CI spans zero**, which is not evidence of equivalence.
* **Selections and frozen scales.**
  * Static-Tput is (4, 512), with a best-to-second gap of 0.0030%. Static-Utility
    is (12, 512). The two objectives disagree.
  * Frozen scales: `t_half` 33.2225, `ttft_target` 150.569 ms, `tpot_target`
    62.163 ms.
* **Landscape:** throughput spans 33.136–33.242 tok/s, with all 28
  configurations within 0.5% of the best.
* **Hypotheses (descriptive cross-regime comparisons):**
  * H1: 0.317% vs a saturated 53.152%.
  * H2: A 0.000754 vs 0.0350; C 0.000855 vs 0.0423; D 0.0000822 vs 0.0194.
* **H5:** 0 of 20 cells outside their Bonferroni bounds.
* **H4:** every learner withheld exactly 3 leading cycles, and none after the
  first valid reward.
* **Convergence**, under the unchanged rule: 0/10 for RuleOnly, CARL-Repaired
  and CARL-Expanded.
* **Mean valid cycle reward vs end-of-run `utility_v2`** (never substituted for
  each other):

  | | mean valid cycle reward | end-of-run `utility_v2` |
  |---|---:|---:|
  | RuleOnly | 0.2953 | 0.2673 |
  | Repaired | 0.2762 | 0.2478 |
  | Expanded | 0.2725 | 0.2385 |

**Reporting limitations.** These are observations, not causal explanations.

1. **Throughput mostly reflects the finish time of the last request.**
   * Every seed emits exactly 9,600 tokens under every treatment (32 per
     interactive request, 64 per batch request), so `throughput_tps` =
     9,600 / `wall_s` (`ablation_live.py:477`).
   * Every paired wall difference equals the paired last-finish difference:
     A +0.189 s, B +0.283 s, C +0.216 s and D −0.023 s, on runs of about 250 s.
   * Wall time ran a median 2.85 s past the last scheduled arrival.
2. **ρ = 0.40 is a target relative to modelled eager burst capacity, not
   measured GPU utilisation.**
   * μ comes from `load_sweep_b7.json` `saturation["*@eager"]`. It was produced
     on the CPU mechanistic substrate, whose eager decode fit comes from a real
     T4 run (R² 0.9985).
   * Realised per-seed offered load is 0.377–0.457.
   * *Descriptive, not registered:* at least one request was in flight for
     78–81% of wall time, with mean concurrency 1.8–2.0.
3. **BATCH-only cycles were predominantly classified as interactive.**
   * Share classified `batch` among BATCH-only cycles: RuleOnly 14/3170,
     Repaired 15/3040, Expanded 16/3016.
   * Mixed cycles: 0/30 for each controller.
   * Source: `configs.<ctrl>.phase_regime_per_seed.*.controller_attribution`.
   * Consequence: in the BATCH phase the controllers **predominantly used the
     interactive arm table** (chunk size 128). Batch-table configurations
     appeared in 14–16 cycles per controller.
4. **TTFT starts at actual submission** (`ablation_live.py:431`).
   * It excludes the gap between intended arrival and submission. In evaluation
     units that gap had a median of 15.7 ms, a p99 of 72.5 ms (linear
     interpolation) and a maximum of 1.030 s.
   * SLO rate, `utility_v2` and the controller reward inherit this origin.
   * `queue_wait_ms` equals `ttft_ms`; it is not a separate queue measurement.
   * Two engine stalls had no established cause:
     * one directly observed 2.401 s step (`RuleOnly_seed044`, step 1547);
     * one inferred step of at least 2.206 s in `sweep/mb2_cs64`, which keeps
       no step log.
5. **Expanded vs Repaired is confounded with the restart.** All ten Expanded
   units ran in session 1.
   * Session 1's warm-up stages were much faster: prefill 12.02 → 3.71 s.
   * D cannot be separated from session effects.
   * Treatment order was fixed.
6. **Nominal arm changes overstate live changes.**
   * Live `(max_batch_size, chunk_size)` changes vs nominal arm changes:
     Repaired 1693 vs 2271; Expanded 2283 vs 2517.
   * RuleOnly made 0 nominal arm changes and 6 live changes on regime flips.
   * Restricted interactive arms alias 6 nominal arms onto 3 live
     configurations.
7. **A configuration decrease limits new admissions; requests already running
   drain** (`scheduler.py:651`).
   * More requests were active than the `max_batch_size` in force on
     1822/45628 Repaired steps and 1371/45355 Expanded steps.
   * The active count never grew on those steps.
8. **Cross-regime comparisons are descriptive only.**
   * The saturated run used CUDA 12.8 (this run used 13.0) and had no warm-up.
   * The saturated model revision was not recorded, and its arrivals and
     harness differ.

**Exploratory sensitivity — not a registered result, replaces nothing.**
Re-basing TTFT to intended arrival (TTFT + lag):
* leaves A, C and D essentially unchanged;
* moves B from −0.1396 to −0.1003 (CI [−0.121, −0.080], 10/10 seeds negative);
* would change the held-out Static-Utility argmax to (24, 512).

The registered estimators, scales and selections stand.

**Corrections to earlier review text (chat reports, not documents).**
* An earlier read-only report said that no OOM traceback and no pre-resume
  snapshot existed. Both exist: the traceback is in the operational log, and
  the snapshot is the pre-resume checkpoint above.
* It also expressed utility differences as percentages. That was invalid.
* It said only the final session's hostname was recorded. Session 0's is
  recorded too, and it is the same.

### Three-campaign addendum (2026-10-26)

Recorded with `docs/eval/DISCLOSURE_2026-10-26_subcritical_campaign_history.md`, which holds the full
hashes, environments, chronology and the "before any GPU run" corrections. The
closeout above is preserved; this addendum scopes it.

**Three complete campaigns, one design.** Same T4/TinyLlama setup, seeds
42–51, 200 requests, held-out seed 999, 7 × 4 grid, ρ = 0.40, μ/λ. Arrival
offsets and request sequences are byte-identical across all three (50/50
units). Engine, controller and learner code are identical up to line endings;
only the experiment driver and provenance code differ.

| | S1 | S2 | O |
|---|---|---|---|
| date (UTC) | 2026-09-18 | 2026-09-19 | 2026-10-23 |
| protocol | Sept. prereg (`b455cc2`), `/1` | same as S1 | Oct. prereg (`e3776082…`), `/4` |
| recorded primary CI | normal (z) | normal (z) | Student-t |
| session | OOM + resume, no warm-up | uninterrupted, no warm-up | OOM + resume, warm-up |
| evidence | `D:\subcritical_poisson_FINAL.zip` | `D:\CARL_September19_recovery\subcritical_poisson_FINAL (1).zip` | `D:\checkpoint_1792796860834369199_f793587b2717.zip` |
| Static-Tput / Static-Utility | (16, 256) / (16, 512) | (32, 256) / (12, 512) | (4, 512) / (12, 512) |
| scales `t_half` / `ttft_target` / `tpot_target` | 33.1630 / 248.775 / 82.361 | 33.1765 / 200.100 / 105.403 | 33.2225 / 150.569 / 62.163 |
| sweep throughput range | 0.39% | 0.75% | 0.32% |

Paired results, recorded primary interval, seeds +/−:

| | metric | S1 (z) | S2 (z) | O (t) |
|---|---|---|---|---|
| A | throughput | −0.0764 [−0.1379, −0.0148], 2/8 | −0.0520 [−0.1179, +0.0140], 3/7 | −0.0286 [−0.0459, −0.0113], 2/8 |
| B | end-of-run `utility_v2` | −0.1925 [−0.2156, −0.1693], 0/10 | −0.1580 [−0.1836, −0.1323], 0/10 | −0.1396 [−0.1593, −0.1200], 0/10 |
| C | throughput | −0.0427 [−0.1009, +0.0154], 3/7 | −0.0144 [−0.0560, +0.0273], 3/7 | −0.0325 [−0.0583, −0.0066], 2/8 |
| C | end-of-run `utility_v2` | −0.0229 [−0.0289, −0.0170], 0/10 | −0.0163 [−0.0217, −0.0109], 0/10 | −0.0196 [−0.0252, −0.0139], 0/10 |
| D | throughput | −0.0018 [−0.0357, +0.0320], 5/5 | +0.0334 [−0.0266, +0.0935], 5/5 | +0.0031 [−0.0162, +0.0224], 4/6 |
| D | end-of-run `utility_v2` | +0.0009 [−0.0070, +0.0088], 4/6 | +0.0012 [−0.0051, +0.0075], 5/5 | −0.0092 [−0.0171, −0.0014], 1/9 |

**What holds in all three:** B and C utility negative on 10/10 seeds; A and C
throughput means negative; throughput landscape within 0.75%; convergence 0/10;
3 withheld leading cycles and none after the first valid reward; every
evaluation unit 200/200 completed; BATCH-only cycles predominantly classified
interactive (11–16 `batch` labels per controller per campaign out of roughly
4,100–4,800 cycles).

**What does not:** whether the A and C throughput intervals exclude zero; the
sign of D (throughput −/+/+, utility +/+/−). Limitation 5 above (Expanded
confounded with the restart) is specific to O; in S1 four of ten Expanded units
ran after the restart, and S2 had no restart. Cross-campaign differences are not
attributed to any cause, and utility magnitudes are not compared across
campaigns (each froze its own scales and baselines).

**Limits specific to S1/S2.** No per-sweep-point files (the 28 sweep rows exist
only as summary rows); the run copied its source-manifest digest from the
sidecar and recorded no archive digest; no model revision recorded; no
warm-up — S1's first post-resume unit (Expanded seed 48) has a 1,926 ms
request-0 TTFT.

**Correction to an earlier chat report (not a document).** A read-only audit
reply described O's sweep files as row summaries plus arrivals; each of O's 28
sweep files also holds 200 per-request records.

**Decision unchanged.** No validity defect has been demonstrated in any
campaign, and no new GPU work is started without one. The reason O was run, the
influence of S1/S2 on the October amendments, and whether other attempts
occurred remain unestablished.
