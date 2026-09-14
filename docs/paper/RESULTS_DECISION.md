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
