# Results decision

What the repaired experiments actually show, and what the paper may therefore
claim. Written before any paper rewrite, deliberately.

**Scope warning.** Every quantitative statement below is from the **repaired
simulation** (`src/eval/engine_model`) on a CPU host. Items 4–6 require a GPU and
are genuinely open. Nothing here is a hardware measurement.

---

## 1. Does CARL beat RuleOnly?

**No. It ties, to four decimal places, on every scenario tested.**

`docs/eval/raw/repair/rule_only_ablation.json`, 11 scenarios × 10 seeds, paired.

| scenario | Δreward (CARL-Repaired − RuleOnly) | all seed diffs zero | Cohen's d |
|---|---:|---|---:|
| stationary_interactive | +0.00000 | **yes** | — |
| stationary_batch | +0.00006 | no | 0.22 |
| stationary_long_context | −0.00128 | no | −0.31 |
| nonstationary_i_b | +0.00009 | no | 0.36 |
| nonstationary_i_b_l | +0.00009 | no | 0.36 |
| oscillating | +0.00005 | no | 0.36 |
| load_shift_interactive | +0.00000 | **yes** | — |
| load_shift_batch | −0.00016 | no | −0.65 |
| load_oscillating | +0.00000 | **yes** | — |
| load_and_regime_shift | −0.00012 | no | −0.65 |
| stress_burst | −0.00017 | no | −0.65 |

Largest effect in either direction: 0.0013 reward units (~0.4% relative). CARL is
*behind* on 5 of 11. This is **outcome C** from the brief, shading into **D**.

For the as-published bandit the result is stronger still: it is **bit-identical**
to RuleOnly on 9 of 11 scenarios, because at α=0.5 it never leaves arm 0
(`bandit_null_check.json`, 5/5 seeds).

## 2. Does learning add value?

**No, and the reason is not that learning is broken.** Both were tested
separately:

- **As-published LinUCB genuinely cannot explore.** On a synthetic problem where
  arm 2 is provably best, it plays arm 0 200/200 times. Untried arms have `b=0`,
  so exploit = 0 against strictly-positive rewards (~0.78), while the optimism
  budget is capped at `α·‖x‖ ≈ 0.25`. Exploration requires α ≳ 1.5.
- **Repaired LinUCB explores properly** (intercept feature + reward centring):
  152/200 on the best arm, converged. Disabling both repairs reproduces the
  lock-in exactly.

So the repaired learner works — and *still* ties RuleOnly. The reason is item 3.

## 3. Where does the gain come from?

**There is almost nothing to gain, because the paper's premise is false on this
engine.**

The decisive number is the **Oracle** column. A per-phase oracle, given the true
phase label and solved by coordinate ascent against the whole-episode objective,
**equals a single tuned static config on 9 of 11 scenarios**:

| scenario | Static-Best-Tput | Oracle | oracle advantage |
|---|---:|---:|---:|
| oscillating | 0.5413 | 0.5413 | 0.0000 |
| load_shift_batch | 0.4851 | 0.4851 | 0.0000 |
| load_oscillating | 0.5121 | 0.5121 | 0.0000 |
| load_and_regime_shift | 0.4504 | 0.4504 | 0.0000 |
| nonstationary_i_b | 0.5469 | 0.5467 | −0.0002 |

If perfect per-phase knowledge is worth ~0, no learner can be worth more. The
ceiling on adaptation is the finding.

Three mechanisms explain it, each independently verified:

1. **The knob does not bind below saturation.** Realised batch is set by
   arrivals, not by `max_batch_size`; mb=8, 12 and 16 produce *identical*
   output at ρ well under 1.
2. **The optimum is driven by load, not prompt length.** Light load wants
   `max_batch ≈ 16`, heavy load wants ≈ 2 — regardless of whether prompts are
   interactive-length or batch-length. The regime taxonomy keys on prompt
   length, so it is largely orthogonal to the variable that matters.
3. **CARL cannot express the answer anyway.** INTERACTIVE and LONG_CONTEXT arm
   sets cap at `max_batch_size = 8` while the measured optimum is 16. This is an
   action-space failure, separate from a learning failure — and giving the
   learner a wider arm set made it *worse* (0.5048 vs 0.5585), because
   exploration over a wider space costs more than it recovers.

The honest one-line mechanism is therefore: **at realistic load a single
operating point is near-optimal across regimes, so online re-selection has
almost no headroom — and what headroom exists is smaller than the cost of
finding it.**

## 4. Does the gain survive realistic arrivals? — **OPEN (needs GPU)**

*(Superseded 2026-10-26, kept as written: Poisson arrivals were run on a T4 in
three complete campaigns — see the Addendum below and `docs/eval/DISCLOSURE_2026-10-26_subcritical_campaign_history.md`.)*

Arrival-process support is implemented (`build_arrivals`: burst / Poisson /
deterministic) and `ablation_live.py` now honours per-request offsets and records
per-stage timestamps. Not yet run on hardware.

In simulation the direction is clear: a bulk dump produces far worse tail TTFT
than a paced stream at the same rate (asserted as a test). The paper's
"~8× TTFT p99" headline is therefore likely to be substantially a queueing
artifact of dumping 25 requests at once behind a concurrency cap. **Expect it to
shrink.**

## 5. Does the gain survive CUDA graphs? — **OPEN (needs GPU)**

Simulation preview, heavy interactive load, throughput at mb=32 ÷ mb=2:

| | ratio |
|---|---:|
| eager (host overhead 14 ms/step) | **1.84×** |
| CUDA graphs (2 ms/step) | **1.37×** |

Roughly **half the batching benefit is Python-dispatch amortisation**. If the GPU
confirms it, that is a hard boundary on external validity and must be stated:
part of the measured effect is a property of mini-vLLM, not of LLM serving.
`batch_intervention.py` decides this.

## 6. Does the gain survive different models? — **NOT RE-TESTED**

`cross_model_results.json` (TinyLlama 1.1B, Qwen2 0.5B, n=3) is a real GPU
measurement and stands, with caveats: gemma-2b failed to load (gated repo), and
"±3.4%" is the spread of two model means. Both models are ≤1.1B on one T4. No
claim about production scale is supported.

## 7. Distribution shift

`distribution_shift.py` exists; no results were ever committed
(`raw/distribution_shift/` holds only `.gitkeep`). On the repaired substrate the
`load_and_regime_shift` scenario is the closest analogue, and CARL loses to
static there by the largest margin of any scenario (−0.0144). **NEEDS RUN.**

## 8. Rapid oscillation

`oscillating` and `load_oscillating`: CARL ties RuleOnly and both lose slightly to
static. The repaired bandit changes arms 20.9 and 20.6 times per episode
respectively and gains nothing for it — consistent with the original −24.2%
direction, though the original artifact is quarantined (n=3, σ=7.8, hand-authored).

## 9. Is the reward itself a limiting factor?

**It was, decisively, and this is the most publishable positive finding here.**

On real GPU, reward v1 collapsed to a constant: `decisions_042.csv` contains
exactly two values ever — 0.8 for three cycles, then 0.3 forever. All four terms
pinned simultaneously (`throughput_ref = 50` against ~85 tok/s measured; 200 ms
TTFT SLO against ~2200 ms; 50 ms TPOT SLO; cache hit rate 0):

```
0.3 = 0.3·(1.0) + 0.3·(1−1) + 0.2·(1−1) + 0.2·(0)
```

A bandit cannot learn from a constant. This also explains why the three
"independent seeds" are byte-identical, and why the reported convergence and
regret numbers are one run reported three times.

Reward v2 (operating-range normalisation, smooth SLO margins) restores
discrimination: per-arm spread 0.03–0.19 across workloads, no degeneracy on any.
`check_non_degenerate` now fails loudly rather than letting a flat reward pass.

## 10. Strongest scientifically defensible claim today

> **On a single-GPU serving engine at realistic offered load, online
> re-selection of the serving configuration has almost no headroom: a per-phase
> oracle with perfect regime knowledge matches a single well-tuned static
> configuration on 9 of 11 non-stationary workloads. The apparent gains reported
> for adaptive serving in this setting are attributable to three separable
> artifacts — a saturated reward that made all configurations indistinguishable,
> an evaluation substrate whose optimum was the controller's own initialisation,
> and a bulk-arrival workload that turned queueing delay into apparent tail
> latency. We characterise when configuration adaptivity *can* matter (the knob
> must bind, which requires ρ near saturation) and show that the controllable
> variable is offered load, not prompt-length regime.**

Supporting, separately defensible:

- **Reward saturation as a systems-ML failure mode.** A four-term serving utility
  with hard clips and SLO indicators pins completely at a realistic operating
  point; the controller goes blind while aggregate statistics look healthy.
  Not, to our knowledge, written down before.
- **Self-speculation break-even on small models** (`spec_breakeven_results.json`):
  acceptance peaks at 5.1% against a 42.5% requirement, every one of 25
  configurations a net slowdown. Real, well-scoped, single-seed, disclosed.
- **A negative-result methodology**: the null ablation (freeze *learning*, not a
  knob), the circularity test (does the substrate's optimum equal the
  controller's initialisation?), and the arm-set coverage check (can the action
  space even express the answer?). All three are cheap, general, and would have
  caught this project's problems on day one.

### What this is not

It is not the paper in `paper/main.tex`. The title claim (A1) is unsupported;
six further claims are contradicted by the repository's own artifacts (see
`CLAIM_LEDGER.md`). Those must be retracted, not softened.

### What could still change the verdict

Items 4 and 5. If the GPU shows that (a) the batch cap *does* bind at realistic ρ
on real hardware, and (b) the throughput-vs-batch slope survives CUDA-graph
capture, then adaptation has real headroom and the negative result is confined to
the simulation. Both are one script and a few GPU-hours away.

**Run `batch_intervention.py` before writing anything.**

---

# Addendum — the final hardening, on real hardware

Everything above this line is the **repaired simulation** on a CPU host. This
addendum is not. It records the first decisive result measured on a **Tesla T4**,
and what may and may not be said about it.

**Frozen.** `final_hardening` is finished evidence. Nothing below is to be
recomputed, re-run, softened or strengthened. A later experiment may *scope* it
by measuring a different operating point; it may not overwrite it.

Full per-claim detail — allowed wording, forbidden wording, effect sizes — is in
`docs/paper/CLAIM_LEDGER.md` section **D**. This page records the *decision*.

## The experiment

| field | value |
|---|---|
| Artifact | `docs/eval/final_hardening_results.json` (produced on Colab; not in this working tree) |
| Script | `scripts/eval/repair/final_hardening.py` |
| Pre-registration | `docs/eval/PREREGISTRATION_final_hardening.md`, before the run |
| Hardware | **Tesla T4**, TinyLlama-1.1B-Chat-v1.0, fp16 — not a simulation |
| Seeds | 42–51 (n = 10, paired by seed), 200 requests per run |
| Held-out seed | 999, used for the reward scales and both static baselines |
| Workload | two-phase NON-STATIONARY, **saturated burst** (bulk dump), ρ ≫ 1 |
| Live plane | `max_batch_size × chunk_size` = 7 × 4 = 28, exhaustive |
| Live dimensions (probed) | `max_batch_size`, `chunk_size` |
| Inactive (probed) | speculation, router, KV eviction, preemption, CUDA graphs |

Two static baselines, selected from **one** held-out sweep by two different
ranking functions, so every comparison has a baseline tuned for the metric it is
judged on:

* **Static-Tput** — `max_batch_size=32`, `chunk_size=512` (argmax throughput)
* **Static-Utility** — `max_batch_size=8`, `chunk_size=256` (argmax `utility_v2`
  under the frozen scales)

## What it found

| comparison | metric | paired Δ | 95% CI | paired *d* |
|---|---|---:|---|---:|
| CARL-Repaired − Static-Tput | `throughput_tps` | −3.3678 | [−3.6633, −3.0723] | −7.06395 |
| CARL-Repaired − Static-Utility | end-of-run `utility_v2` | −0.0245 | [−0.0263, −0.0227] | −8.44215 |
| CARL-Repaired − RuleOnly | `throughput_tps` | +3.7633 | [+3.2932, +4.2335] | +4.96133 |
| CARL-Repaired − RuleOnly | end-of-run `utility_v2` | +0.0363 | [+0.0347, +0.0380] | — |
| CARL-Expanded − CARL-Repaired | `throughput_tps` | −1.7956 | [−2.2754, −1.3158] | — |
| CARL-Expanded − CARL-Repaired | end-of-run `utility_v2` | −0.0277 | [−0.0295, −0.0260] | — |

Convergence, under the rule fixed before the run: **0/10 seeds** for RuleOnly,
**0/10** for CARL-Repaired, **0/10** for CARL-Expanded.

Cold start: every adaptive run withheld exactly **3** invalid-reward cycles, no
learner update occurred on a withheld reward, `rewarded_arm = −1` on those
cycles, and `withheld_after_first_valid = 0` everywhere. Pre-registered H5
passed.

## The decision

**The central wording, and the only one permitted for the saturated point:**

> At the saturated T4 operating point, adaptive reconfiguration is useful but
> insufficient: CARL-Repaired improves throughput and end-of-run utility over
> RuleOnly, but remains below objective-matched static configurations. Expanded
> live configuration coverage further reduces both metrics.

This **changes the verdict recorded above** in one specific way, and only that
way. Section 1 of this document concluded from the simulation that CARL "ties
RuleOnly to four decimal places". On real hardware, at a saturated operating
point, it does not tie — it **wins** over RuleOnly by +3.7633 tok/s with a CI
excluding zero. Learning does something. It simply does not do enough to reach
an objective-matched static configuration.

That is a materially better result for the paper than the simulation's, and a
materially worse one than the original manuscript's. Both statements must be
made together.

### Two things the numbers do not say

**1. Mean valid cycle reward is not end-of-run `utility_v2`.** The two are
different estimators and neither may stand in for the other:

* *mean valid cycle reward* — the mean over a controller's scored control
  cycles; what the learner actually consumed. Defined only for controllers, and
  never comparable against a static configuration, which has no control cycles.
* *end-of-run `utility_v2`* — `utility_v2` of a run's end-of-run aggregates
  under the frozen scales. Defined for every treatment, and the metric of the
  Static-Utility comparison.

The permitted phrasing for the utility baseline is exactly:

> Static-Utility is selected using the same `utility_v2` function, weights, and
> frozen scales used by CARL, and both systems are compared on end-of-run
> `utility_v2`.

It is **not** claimed to optimise the exact sequence of per-cycle rewards CARL
consumed, and no document may say so.

**2. This is one operating point.** Saturated burst arrivals at ρ ≫ 1, one T4,
one 1.1B model, two live knobs. Saturation is precisely the regime in which
`max_batch_size` binds hardest and a fixed configuration therefore has the most
to offer. "Static tuning always wins" over-generalises the result; "online
adaptation is useless" contradicts it. Neither may be written.

## What could still change the verdict — and what could not

**Could not.** Re-running the saturated point. It is finished.

**Could.** The load axis. If the saturated result is an artefact of the regime
in which the knob binds hardest, then the same comparisons at a comfortably
sub-critical offered load should look materially different — not necessarily in
sign, but in magnitude. That is registered as
`docs/eval/PREREGISTRATION_subcritical_poisson.md` and implemented in
`scripts/eval/repair/subcritical_poisson.py`.

The operating point there is **derived, not chosen**: λ = 0.40·μ per phase, with
μ read from `docs/eval/raw/repair/load_sweep_b7.json` for the execution path
that actually runs (eager — no CUDA-graph runner is attached by
`ablation_live._new_scheduler`):

| phase | μ (req/s) | λ (rps) | 1/λ (s) |
|---|---:|---:|---:|
| interactive | 2.9613881136988742 | 1.1845552454795498 | 0.8442 |
| batch | 1.4135291672527908 | 0.5654116669011163 | 1.7686 |

The mechanistic hypothesis, registered before the run *(2026-10-26: before S1,
in the September pre-registration; unchanged in October)*:

> The sub-critical Poisson operating point will expose less
> configuration-dependent throughput headroom than the saturated burst operating
> point. Therefore the value available to online adaptation should be smaller
> relative to its finite-horizon search and reconfiguration cost.

That predicts a shrinking **effect size**, not a winner. Three outcomes were
written down in advance and all three are informative:

| case | outcome | what it would mean |
|---|---|---|
| A | CARL still loses to objective-matched static tuning | the limitation survives across offered-load regimes — the saturated result *widens* rather than narrows |
| B | CARL becomes competitive with or beats an objective-matched static baseline | offered load is a **boundary condition**; the saturated result must be scoped to ρ ≫ 1 |
| C | CARL beats RuleOnly but not static | adaptive behaviour is useful, but finite-horizon adaptation economics remain unfavourable |

**Campaign history (2026-10-26).** The paragraph below was written when only
the October campaign was known locally and is kept as written. The design was in
fact executed to completion three times on the same seeds and arrivals: **S1**
(2026-09-18) and **S2** (2026-09-19) under the September pre-registration, and
**O** (2026-10-23) under the October amendments, which were made after S1/S2
results had been viewed. Record: `docs/eval/DISCLOSURE_2026-10-26_subcritical_campaign_history.md`.

**Outcome (2026-10-25).** The experiment ran to completion: 28 held-out units
and 50 evaluation units on a T4, with one OOM interruption and a byte-verified
resume. It is closed with reporting limitations, and no validity defect was
demonstrated.
* Observed case: **A** — CARL-Repaired remained below both objective-matched
  static baselines.
* The absolute throughput differences were small: −0.029 tok/s against
  Static-Tput, roughly a 0.2 s later final completion.
* H1 and H2 were supported as descriptive cross-regime comparisons.
* Convergence was 0/10 for every controller.

Numbers, evidence hashes and limitations: `docs/eval/CARL_REPAIR_STATUS.md`
Phase 24 → *Run record and closeout*. Wording: `CLAIM_LEDGER.md` section E.

**Outcome across the three campaigns (2026-10-26).** Case **A** in every
campaign: CARL-Repaired's end-of-run `utility_v2` was below Static-Utility on
10/10 seeds in S1, S2 and O, and its mean throughput was below Static-Tput in
all three (−0.0764, −0.0520, −0.0286 tok/s; the S2 z interval includes zero).
Against RuleOnly, utility was lower on 10/10 seeds in each campaign and mean
throughput was lower in each (only O's interval excludes zero) — so case C
("beats RuleOnly but not static") was **not** observed at this point.
Expanded − Repaired changed sign across campaigns. Convergence 0/10 everywhere.
Unresolved: why O was run, whether S1/S2 influenced the October amendments, and
whether other attempts occurred.

**After that experiment, the campaign stops** unless a genuine validity defect
is found. *(2026-10-26: still the decision, now covering all three campaigns. No
validity defect has been demonstrated in S1, S2 or O.)* Two operating points on a load axis is a boundary check, and a
boundary check is what the scope sentence needs. It is not a load envelope, and
no third point is required to say so honestly.

**Do not write the paper from this addendum alone.** Read `CLAIM_LEDGER.md`
sections D and E for the per-claim allowed and forbidden wording.
