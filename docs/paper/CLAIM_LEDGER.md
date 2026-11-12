# Claim ledger

**Source of truth for the eventual paper.** Every claim that appears in the
abstract, introduction or contributions list must reach **VERIFIED** before it
may be written. A claim at any other status is not permitted in prose.

Statuses: `VERIFIED` · `PARTIALLY VERIFIED` · `NOT VERIFIED` · `CONTRADICTED` ·
`NEEDS NEW EXPERIMENT`.

`CONTRADICTED` means the repository's own artifacts disagree with the claim. Such
a claim must be **retracted**, not softened.

---

## A. Claims currently in `paper/main.tex` and `paper/sections/01-intro.tex`

| # | Claim (as written) | Evidence | Status | Allowed wording |
|---|---|---|---|---|
| A1 | "adaptive online scheduling, rather than speculative decoding, is the primary driver of performance improvements under changing LLM-serving workloads" | Ablation whose `CARL-NoSpec` arm is flagged `live_effective=false`; speculation was pinned off in every run | **CONTRADICTED** | None. You cannot ablate a disabled subsystem. Retract; replace with the scoped self-speculation claim (B4). |
| A2 | "CARL reaches 108.2 ± 3.4% of a per-model-tuned static best across two model families" | `cross_model_results.json`, real T4, n=3 seeds × 2 models | **PARTIALLY VERIFIED** | "108.2% averaged over two models (TinyLlama-1.1B, Qwen2-0.5B), n=3 seeds each; ±3.4% is the spread of two model means, not a confidence interval." Third model (`gemma-2b-it`) failed to load — gated repo. |
| A3 | "+10.6% TinyLlama, +5.8% Qwen2" | `cross_model_results.json` per-run ratios | **PARTIALLY VERIFIED** | Numerically supported. Must disclose that Static-Best was selected on **throughput alone** and that the workload is a bulk dump (see A5). |
| A4 | "static-best TTFT p99 inflates ~8×, 2,199 ms → 18,058 ms" | `cross_model_results.json` | **PARTIALLY VERIFIED** | Numerically real, but the baseline was tuned on throughput and then judged on latency, and 25 requests were dumped at once. Needs `static_slo.py` + Poisson arrivals before it can be stated as a serving result. |
| A5 | Non-stationary workload representative of production traffic | `live.py::_build_workload` — all phase-0 at t=0; phase-1 dumped when half of phase 0 finished; prompts are `"The quick brown fox…"` tiled | **NOT VERIFIED** | Call it a **burst stress workload**. Arrival-process support now exists (`build_arrivals`); re-measure before claiming serving realism. |
| A6 | "LinUCB policy closes the per-regime oracle gap to 0.45% versus 16.2% for static" | `oracle_results.json`: `carl_linucb` throughput is **bit-identical** to `oracle` in phases 1–2 (39.71338147278326) | **CONTRADICTED** | Retract. The residual is rule-based-classifier error; the bandit contributes nothing. |
| A7 | "on every stationary single-regime workload CARL is statistically identical to static best (d = 0, p = 1.0)" | `statistical_validation_results.json`: `long_prompts` d = −0.352 / p = 0.693; `batch_only` d = −0.352 / p = 0.693; `long_context` d = −1.516 (unmentioned) | **CONTRADICTED** | Retract. Two of the four named workloads do not have d = 0. Where d = 0 does hold it is an *identity* — both agents run the same config on the same seed — not a statistical finding. |
| A8 | "NON-STATIONARY d = 23.2, p = 2.3×10⁻⁴¹, n = 30" | `stats_results.json` | **PARTIALLY VERIFIED** | Arithmetically correct; scientifically uninformative. The statistical unit is a draw from `1 + N(0, 0.05)` in a deterministic simulator, so the effect size is set by `WorkloadModel._noise`. Do not report as evidence of robustness. |
| A9 | "deterministic convergence at cycle 9 across all seeds, cumulative regret 2.583" | `decisions_042.csv` / `_043` / `_044` are **byte-identical**; reward takes two values (0.8, then 0.3 forever) | **CONTRADICTED** | Retract. One run reported three times; the seed cannot influence a controller whose reward is constant. |
| A10 | "CARL converges to the per-regime optimum" | `adaptation_results.json`: `final_arm_per_regime.interactive = 3`; `oracle_arms_per_regime.interactive.best_arm = 0` | **CONTRADICTED** | Retract. It converged to a non-optimal arm, and the "oracle" ranking is confounded by cycle ordering. |
| A11 | "Sched +28.51 tok/s; Router +0.02; Spec −0.19; Chunk −0.37; Cache −0.65" | `ablation_live_results.RECONSTRUCTED.json` — hand-authored, schema mismatch, no raw data | **NOT VERIFIED** | Unusable until regenerated. Also: `+28.51` compares against `max_batch_size` pinned to 4, not against Static-Best (85.2 vs 81.1 = +5.1%). The four sub-tok/s deltas are inside σ = 1.2–1.9. |
| A12 | "CARL ties or loses on all five stress workloads, up to −24.2%" | `failure_cases_results.RECONSTRUCTED.json` — hand-authored, environment not captured, n=3, σ=7.8 on the headline | **NOT VERIFIED** | Regenerate at n≥10. The *direction* is plausible and is the paper's most interesting material. |
| A13 | "AutoTuner leaves an 18.9% gap — wider than static's 16.2%" | `benchmark_carl.AutoTunerAgent.choose` refills the profiler from `_BOTTLENECK[true_regime]`, ignoring its own config | **CONTRADICTED** | Retract. The baseline is open-loop; the result describes a defect. A closed-loop replacement now exists (`ClosedLoopAutoTunerController`). |
| A14 | Controller overhead ≈ 79 µs mean / 102 µs p99, 76% bandit | `overhead_results.json` + `raw/overhead/component_breakdown.json` | **VERIFIED** | Real CPU measurement with raw data. Usable as written. |

---

## B. Claims the repository *can* support

| # | Claim | Evidence | Status | Notes |
|---|---|---|---|---|
| B1 | LinUCB as shipped is behaviourally inert at α=0.5: it never leaves arm 0, and CARL is bit-identical to a stateless `DEFAULT_CONFIGS[classify_regime(state)]` lookup | `raw/repair/bandit_null_check.json`, 5/5 seeds, `carl_equals_rule_only_all_seeds: true` | **VERIFIED** | Diagnosable: untried arms get exploit = 0 against strictly-positive rewards, and ‖x‖ ≈ 0.5 caps the optimism budget at ≈ 0.5α. Exploration needs α ≳ 1.5. |
| B2 | The defect is in the bandit, not only the simulator | `tests/test_eval/test_controllers.py::test_as_published_linucb_locks_on_arm_zero` — 200/200 on arm 0 on a synthetic problem where arm 2 is provably best | **VERIFIED** | Two textbook repairs (intercept feature, reward centring) fix it; disabling both reproduces the lock-in exactly. |
| B3 | Reward v1 is fully saturated on real GPU: all four terms pinned, reward constant at 0.3 | `raw/adaptation/decisions_042.csv`, decomposed against `utility()` weights | **VERIFIED** | This is a genuinely novel and useful systems-ML finding about reward design for serving control. |
| B4 | Self-speculative (early-exit) decoding is below break-even at every (L, K) tested on TinyLlama/T4: acceptance peaks at 5.1% against a 42.5% requirement | `spec_breakeven_results.json` | **VERIFIED** | Single seed, disclosed. The gap is ~an order of magnitude. **Scope it to self-speculation on this model/hardware** — it does not license A1. |
| B5 | The prior simulation substrate is circular: oracle == bandit arm 0 == 5/6 static-best candidates == `DEFAULT_CONFIGS` | `benchmark_carl._match_score`, `_harness.best_static_config`; confirmed by bit-identical `carl_linucb`/`oracle` output | **VERIFIED** | Replaced by `src/eval/engine_model`, which imports nothing from `src.carl` (asserted by test). |
| B6 | `max_batch_size` does not bind below saturation: realised batch is arrival-bounded, so the knob has no leverage | `tests/test_eval/test_engine_model.py::test_batch_cap_does_not_bind_below_saturation` | **VERIFIED (simulation)** | Needs GPU confirmation via `batch_intervention.py` (`cap_was_binding`). |
| B7 | The reward-optimal `max_batch_size` is driven by **load**, not prompt length: light load wants ~16, heavy load wants ~2 | load sweep in `rule_only_ablation.py` | **VERIFIED (simulation)** | Undercuts the paper's regime taxonomy, which keys on prompt length. |
| B8 | CARL's INTERACTIVE and LONG_CONTEXT arm sets cap at `max_batch_size = 8`, while the measured optimum is 16 — CARL cannot express the answer | `arm_set_coverage`, `tests/…::test_arm_set_coverage_is_reported_per_regime` | **VERIFIED** | Action-space failure, distinct from a learning failure. Both must be reported. |
| B9 | Roughly half the batching benefit is Python-dispatch amortisation (1.84× eager → 1.37× with CUDA graphs) | simulation preview | **NEEDS NEW EXPERIMENT** | GPU arm of `batch_intervention.py` decides this. Directly governs external validity. |
| B10 | On the mechanistic substrate, a single well-tuned static config matches or beats every adaptive policy, and the per-phase **oracle** itself barely beats it | `raw/repair/rule_only_ablation.json` | **VERIFIED (simulation)** | The strongest honest result available today. It says the *premise* — that regimes have nearly disjoint optima — is false for this engine. |

---

## C. What must happen before the paper is written

1. Run `batch_intervention.py` on a GPU (B9, B6). Calibrate `HardwareProfile`,
   re-run every CPU experiment against it.
2. Regenerate A11 and A12 with provenance and raw per-seed data.
3. Run `static_slo.py` so A4 is a fair comparison.
4. Re-measure A2–A5 under a Poisson arrival process at a stated ρ.
   *Registered as section E* (`subcritical_poisson.py`, ρ = 0.40 derived from a
   measured service rate). Not yet run; no claim may be drawn from it until the
   artifact exists.
5. Delete A1, A6, A7, A9, A10, A13 from the paper. Do not soften them.

Sections **D** (frozen, saturated T4) and **E** (registered, sub-critical
Poisson) are below. D supersedes nothing in A–C; it is a new operating point
measured on real hardware, and every claim drawn from it must carry its scope.

Every claim in section A that is `CONTRADICTED` is contradicted **by this
repository's own committed artifacts**, not by an outside standard. A reviewer
with the artifact will find each one in minutes.

---

## D. Final hardening — the saturated T4 operating point (FROZEN)

**Status of this section: FROZEN EVIDENCE.** The rows below are the recorded
result of `scripts/eval/repair/final_hardening.py`. They are not to be
recomputed, re-run, softened or strengthened. A later experiment may *scope*
them by measuring a different operating point; it may not overwrite them.

### D0. Shared experimental facts (apply to every row D1–D6)

| field | value |
|---|---|
| Artifact | `docs/eval/final_hardening_results.json`, raw per-seed in `docs/eval/raw/final_hardening/` |
| Artifact present in this working tree | **No** — produced on Colab; the values below are the recorded result, transcribed, not recomputed |
| Script | `scripts/eval/repair/final_hardening.py` |
| Pre-registration | `docs/eval/PREREGISTRATION_final_hardening.md`, registered before the run |
| Hardware / simulation | **Real hardware.** Tesla T4, TinyLlama-1.1B-Chat-v1.0, fp16. Not a simulation. |
| Evaluation seeds | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 |
| n (paired units) | 10 seeds, paired by seed |
| Requests per run | 200 |
| Held-out calibration/selection seed | 999 (never an evaluation seed) |
| Workload | Two-phase NON-STATIONARY, **saturated / non-stationary burst** (bulk dump), ρ ≫ 1 |
| Live search plane | `max_batch_size × chunk_size` = 7 × 4 = **28 configurations, exhaustive** |
| Verified live dimensions | `max_batch_size`, `chunk_size` (probed against the real scheduler object) |
| Verified inactive | speculative decoding, router, KV eviction, preemption, CUDA graphs (no graph runner attached) |
| Static-Tput | `max_batch_size=32`, `chunk_size=512` |
| Static-Utility | `max_batch_size=8`, `chunk_size=256` |
| Statistic | paired mean difference over matched seeds, 95% CI, paired Cohen's *d* |

### D1. Adaptation vs the throughput-matched static baseline — **VERIFIED**

| field | value |
|---|---|
| Comparison | CARL-Repaired − Static-Tput |
| Metric | `throughput_tps` |
| Paired difference | **−3.3678 tok/s** |
| 95% CI | **[−3.6633, −3.0723]** |
| Cohen's *d* (paired) | **−7.06395** |

**Allowed wording.** "At the saturated T4 operating point, CARL-Repaired
delivers lower throughput than a throughput-selected static configuration
(−3.3678 tok/s, 95% CI [−3.6633, −3.0723], paired *d* = −7.06, 10 seeds)."

**Forbidden stronger wording.** Any claim that adaptation matches or beats
static tuning on throughput. Any generalisation beyond this operating point,
this model, this hardware, or these two knobs — the result is measured at ρ ≫ 1
under burst arrivals and says nothing yet about other offered loads. Do not
present the effect size as evidence of robustness: *d* is large because the
paired differences are consistent across seeds, not because the phenomenon has
been shown to generalise.

### D2. Adaptation vs the utility-matched static baseline — **VERIFIED**

| field | value |
|---|---|
| Comparison | CARL-Repaired − Static-Utility |
| Metric | **end-of-run `utility_v2`** (`run_utility_v2`) |
| Paired difference | **−0.0245** |
| 95% CI | **[−0.0263, −0.0227]** |
| Cohen's *d* (paired) | **−8.44215** |

**Allowed wording.** "CARL-Repaired also remains below an *objective-matched*
static configuration on CARL's own objective (−0.0245 end-of-run `utility_v2`,
95% CI [−0.0263, −0.0227], paired *d* = −8.44). Static-Utility is selected using
the same `utility_v2` function, weights, and frozen scales used by CARL, and
both systems are compared on end-of-run `utility_v2`."

**Forbidden stronger wording.**

* Do **not** claim Static-Utility optimises the exact sequence of per-cycle
  rewards CARL consumed. It does not, and it is not selected to.
* Do **not** substitute *mean valid cycle reward* for *end-of-run
  `utility_v2`*. See D6.
* Do **not** write "CARL loses even on its own metric" without stating that the
  metric compared is the end-of-run aggregate.

### D3. What learning bought — **VERIFIED**

| field | value |
|---|---|
| Comparison | CARL-Repaired − RuleOnly |
| Primary metric | `throughput_tps` |
| Paired difference | **+3.7633 tok/s** |
| 95% CI | **[+3.2932, +4.2335]** |
| Cohen's *d* (paired) | **+4.96133** |
| Secondary metric | end-of-run `utility_v2` |
| Paired difference (secondary) | **+0.0363** |
| 95% CI (secondary) | **[+0.0347, +0.0380]** |

**Allowed wording.** "Holding the classifier and the arm set fixed, learning is
worth +3.7633 tok/s (95% CI [+3.2932, +4.2335], paired *d* = +4.96) and +0.0363
end-of-run `utility_v2` (95% CI [+0.0347, +0.0380]) over a non-learning rule."

**Forbidden stronger wording.** This is **not** convergence (see D5) and **not**
optimality. Do not write that CARL "learns the optimal configuration" or "closes
the gap to the best static": it improves on a fixed rule while remaining behind
both static baselines.

### D4. What a wider action space bought — **VERIFIED**

| field | value |
|---|---|
| Comparison | CARL-Expanded − CARL-Repaired |
| Primary metric | `throughput_tps` |
| Paired difference | **−1.7956 tok/s** |
| 95% CI | **[−2.2754, −1.3158]** |
| Secondary metric | end-of-run `utility_v2` |
| Paired difference (secondary) | **−0.0277** |
| 95% CI (secondary) | **[−0.0295, −0.0260]** |

**Allowed wording.** "Expanded live configuration coverage *reduces* both
throughput (−1.7956 tok/s, 95% CI [−2.2754, −1.3158]) and end-of-run
`utility_v2` (−0.0277, 95% CI [−0.0295, −0.0260]) relative to the restricted
coverage, with the learner held fixed."

**Forbidden stronger wording.** Do **not** assert the mechanism — "Expanded
loses *because* it switches more" is a hypothesis this experiment did not test.
Report arm-change rate alongside the result and leave the attribution open. Also
avoid "10 arms vs 6 arms" wherever nominal arms collapse onto identical
effective configurations; the correct phrase is **expanded live configuration
coverage**.

### D5. Convergence — **NOT SUPPORTED** (a measured negative)

| treatment | seeds supporting convergence |
|---|---|
| RuleOnly | **0 / 10** |
| CARL-Repaired | **0 / 10** |
| CARL-Expanded | **0 / 10** |

Rule, fixed before the run (`final_hardening.CONVERGENCE_RULE`): at least 6
decisions in the regime; at least 2 distinct arms in the first half; a
second-half modal-arm share ≥ 0.75; and a modal share that **strictly increased**
from the first half to the second.

**Allowed wording.** "Under a pre-registered rule requiring exploration followed
by exploitation, no run in any treatment showed convergence (0/10 seeds for each
of RuleOnly, CARL-Repaired and CARL-Expanded)."

**Forbidden stronger wording.** "CARL converges", "deterministic convergence at
cycle 9", or any convergence claim derived from a high modal share alone. A
learner that never moved has modal share 1.0 and has converged to nothing.

### D6. Cold start — **VERIFIED** (positive control held)

| field | value |
|---|---|
| Warm-up withheld per adaptive run | exactly **3** invalid-reward cycles, every run |
| Learner updates on invalid reward | **none** |
| `rewarded_arm` on a withheld cycle | **−1** |
| `withheld_after_first_valid` | **0** everywhere |
| Reward validity after warm-up | valid throughout |
| Pre-registered H5 | **passed** |

**Allowed wording.** "Rewards computed from unmeasured TTFT/TPOT are withheld
rather than fabricated; every adaptive run withheld exactly three leading
cycles, performed no learner update on them, and reported zero withheld cycles
after the first valid reward."

**Two estimators, not interchangeable.** The ledger records both and forbids
substitution:

* **mean valid cycle reward** — the mean over a controller's *scored control
  cycles*; what the learner actually consumed. Defined only for controllers.
* **end-of-run `utility_v2`** — `utility_v2` of a run's end-of-run aggregates
  under the frozen scales. Defined for every treatment including the statics,
  and the metric of D2.

Neither may stand in for the other, and the statics have no control cycles at
all, so no cycle-level quantity may be compared against them.

### D7. The central saturated-result wording

> At the saturated T4 operating point, adaptive reconfiguration is useful but
> insufficient: CARL-Repaired improves throughput and end-of-run utility over
> RuleOnly, but remains below objective-matched static configurations. Expanded
> live configuration coverage further reduces both metrics.

### D8. Retracted claims that may not be resurrected

None of the following may re-enter any document, under any phrasing:

`108.2%` · `+10.6% / +5.8%` adaptation gains · `d = 23.2` · `p = 2.3×10⁻⁴¹` ·
`0.45%` oracle gap · cycle-9 convergence · `+28.51` scheduler contribution ·
`93–101%` shift recovery · DynOracle described as a true oracle · "CARL
converges" · "CARL learns the optimal configuration" · "static tuning always
wins" · "online adaptation is useless" · "Expanded loses because it switches
more".

The last two deserve naming: D1–D4 are **one operating point**. "Static tuning
always wins" over-generalises them, and "online adaptation is useless"
contradicts D3.

---

## E. Sub-critical Poisson operating point — **RUN, COMPLETE; CLOSED WITH REPORTING LIMITATIONS**

| field | value |
|---|---|
| Script | `scripts/eval/repair/subcritical_poisson.py` |
| Pre-registration | `docs/eval/PREREGISTRATION_subcritical_poisson.md` |
| Artifact | `docs/eval/subcritical_poisson_results.json` (sha256 `f793587b…784433`), inside `D:\checkpoint_1792796860834369199_f793587b2717.zip` |
| Status | **Complete.** 28 held-out units and 50 evaluation units, run on a T4 on 2026-10-23, with one OOM interruption and a verified resume. Evidence, hashes and limitations are in `docs/eval/CARL_REPAIR_STATUS.md` → Phase 24 → *Run record and closeout*. |
| Campaign history (2026-10-26) | **Three complete campaigns, not one.** S1 (2026-09-18, `D:\subcritical_poisson_FINAL.zip`), S2 (2026-09-19, `D:\CARL_September19_recovery\subcritical_poisson_FINAL (1).zip`) under the September pre-registration (`b455cc2`), and O (2026-10-23, the row above) under the October one. S1/S2 results were viewed and discussed before the October amendments. The rows above describe O only and are kept as written. Record: `docs/eval/DISCLOSURE_2026-10-26_subcritical_campaign_history.md`. |

**What it was designed to decide** *(text written before the run)*. Whether section D is a property of *adaptation* or of
the *operating point*. D was measured at ρ ≫ 1 under burst arrivals — precisely
the regime in which `max_batch_size` binds hardest and a fixed configuration has
the most to offer. E moves exactly one factor, arrivals, to an open-loop Poisson
stream at a **derived** ρ = 0.40, with a phase-specific λ = 0.40·μ:

| phase | μ source row (`docs/eval/raw/repair/load_sweep_b7.json`) | μ (req/s) | λ (rps) |
|---|---|---:|---:|
| interactive | `saturation["interactive@eager"].mu_max_req_per_s` | 2.9613881136988742 | 1.1845552454795498 |
| batch | `saturation["batch@eager"].mu_max_req_per_s` | 1.4135291672527908 | 0.5654116669011163 |

The **eager** rows are used because `ablation_live._new_scheduler` attaches no
`_graph_runner` and `ContinuousBatchScheduler._decode_forward` will not replay a
graph without one — execution reality, not the `use_cuda_graphs` config label.
The eager decode parameters behind those rows are bit-identical to the eager
least-squares fit in
`docs/eval/raw/repair/batch_intervention_phase15_16_s42_s43_s44.json` (real
Tesla T4, 54 rows, R² = 0.9985).

**Section D must still be written with its scope stated:** one saturated
operating point, burst arrivals, one T4, one 1.1B model, two live knobs. E adds
a second operating point. It is a boundary check, not a load envelope. That
scope sentence is not optional.

### E0. Three-campaign rules (added 2026-10-26; govern every E claim)

* **Report all three campaigns** (S1, S2, O), each with its own recorded
  primary interval (z for S1/S2, Student-t for O). Label any common-method
  interval computed afterwards as retrospective.
* **Do not pool** the campaigns or count 30 runs: they are three measurements of
  the same ten workloads (identical arrivals and requests).
* **Do not pick a headline campaign by its results.** O may carry more method
  detail because its provenance is stronger; it may not be presented as the
  only campaign.
* **Do not compare utility magnitudes across campaigns**: each froze its own
  scales and selected its own static baselines.
* A CI that excludes zero in one campaign and not another is **not** evidence
  that the effects differ.
* Withdrawn for this point: unqualified "pre-registered", "registered before
  the run", "confirmatory", and "before any GPU run". Allowed: "specified in the
  September pre-registration" (design, seeds, n, grid, ρ, μ/λ, treatments,
  reward, convergence rule, no registered winner) versus "amended in October,
  after September results existed" (Student-t primary, H2 denominator, H5
  family rule, descriptive phase reporting, engine warm-up).
* Forbidden labels: "pilot" for S1/S2; "independent confirmation" or
  "replication" for O.

**Three-campaign wording (allowed):**

| # | allowed wording |
|---|---|
| E0.1 | "Across three complete campaigns on the same seeds, CARL-Repaired's end-of-run `utility_v2` was below Static-Utility and below RuleOnly on all ten seeds in each campaign." |
| E0.2 | "Mean throughput differences against Static-Tput and RuleOnly were negative in all three campaigns and below 0.08 tok/s; whether the 95% interval excluded zero differed between campaigns (A: S1 and O yes, S2 no; C: O yes, S1 and S2 no)." |
| E0.3 | "Expanded − Repaired changed sign across campaigns for both metrics; only O's utility interval excluded zero, and in O every Expanded unit ran after the restart." |
| E0.4 | "The held-out throughput range was 0.39%, 0.75% and 0.32% of the best configuration in S1, S2 and O; the three campaigns selected three different Static-Tput configurations." |

### E results — allowed and forbidden wording

*(2026-10-26: E1–E8 below are **campaign O** wording, kept as written. Use
them only alongside E0; "pre-registered" in E6 means "under the convergence
rule fixed for the saturated study", which is unchanged across all three.)*

All comparisons are paired over seeds 42–51 (n = 10), a − b, with the 95%
Student-t CI. Full table: `CARL_REPAIR_STATUS.md` Phase 24. p-values are never
the headline.

| # | claim | status | allowed wording | forbidden |
|---|---|---|---|---|
| E1 | Flatter throughput landscape (H1/H2) | **SUPPORTED, descriptive** | "The relative throughput range over the 28-point held-out grid was 0.32%, against 53.15% at the saturated point. Relative paired throughput differences were smaller for A, C and D. These compare two harness and runtime conditions descriptively." | "Offered load causes the flattening"; "generalises across load regimes" |
| E2 | Repaired vs Static-Tput (A) | **SUPPORTED, small** | "Throughput was lower by 0.0286 tok/s (95% CI [−0.0459, −0.0113]; 8/10 seeds negative). Under fixed output counts this is a mean final completion about 0.19 s later on runs of about 250 s." | "CARL substantially reduces throughput" |
| E3 | Repaired vs Static-Utility (B) | **SUPPORTED** | "The mean paired end-of-run `utility_v2` difference was approximately −0.140 (95% CI [−0.159, −0.120]), with negative differences on all ten seeds." | Any percentage of `utility_v2`; "static always wins" |
| E4 | Repaired vs RuleOnly (C) | **SUPPORTED, small** | "Throughput was lower by 0.0325 tok/s (95% CI [−0.0583, −0.0066]; 8/10 seeds negative). End-of-run `utility_v2` was lower by 0.0196 (95% CI [−0.0252, −0.0139]; 10/10 seeds negative)." | "Learning hurts" as a general claim; "online adaptation is useless" |
| E5 | Expanded vs Repaired (D) | **QUALIFIED** | "No resolvable throughput difference (+0.0031 tok/s, 95% CI [−0.0162, +0.0224]). End-of-run `utility_v2` was lower by 0.0092 (95% CI [−0.0171, −0.0014]; 9/10 seeds negative). All Expanded units ran after the restart, so this comparison is confounded with session." | "Equivalent"; "Expanded loses because it switches more" |
| E6 | Convergence | **NOT SUPPORTED** | "0/10 seeds for each controller under the unchanged pre-registered rule." | "CARL converges"; "CARL learns the optimum" |
| E7 | Regime-aware adaptation | **NOT SUPPORTED as run** | "BATCH-only cycles were predominantly classified as interactive (classified `batch`: 14/3170, 15/3040, 16/3016), so the controllers predominantly used the interactive arm table during the BATCH phase." | Any causal account of the classifier; "CARL adapts to the batch regime" |
| E8 | Operating point | **QUALIFIED** | "Target ρ = 0.40 relative to modelled eager burst capacity. Realised offered load was 0.38–0.46 per seed." | "40% GPU utilisation"; "requests were submitted at their exact Poisson times" |

**Every E claim carries these qualifications.**
* TTFT starts at actual submission and excludes the gap between intended
  arrival and submission (median 15.7 ms).
* Arm-change rates count nominal arms, which overstate live configuration
  changes.
* Configuration decreases bind at admission while requests already running
  drain.
* Mean valid cycle reward and end-of-run `utility_v2` are never substituted for
  each other.
* Mechanistic explanations involving adaptation cost, switching or classifier
  behaviour are **not tested** and may not be written as findings.

**Exploratory only (not registered).** Re-basing TTFT to intended arrival moves
B to −0.100 and leaves A, C and D essentially unchanged. It may be reported only
as a labelled sensitivity analysis, never in place of the registered result.
