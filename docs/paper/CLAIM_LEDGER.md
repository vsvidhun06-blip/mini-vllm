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
5. Delete A1, A6, A7, A9, A10, A13 from the paper. Do not soften them.

Every claim in section A that is `CONTRADICTED` is contradicted **by this
repository's own committed artifacts**, not by an outside standard. A reviewer
with the artifact will find each one in minutes.
