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
