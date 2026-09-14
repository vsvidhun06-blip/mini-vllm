# Pre-registration — Final hardening experiment

**Script:** `scripts/eval/repair/final_hardening.py`
**Artifact:** `docs/eval/final_hardening_results.json`
**Registered:** before the run. Copied verbatim into the artifact under
`pre_registration`.
**Supersedes (but does not replace):** `reward_v2_live`. That experiment's
artifact is preserved unchanged — it is the evidence for both defects below.

---

## Why this experiment exists

Per-cycle forensic inspection of the reward-v2 artifact found two defects. Both
are errors in *what was being compared*, not coding slips, and either one alone
is enough to make the headline comparison uninterpretable.

### Defect 1 — objective mismatch

CARL maximises `utility_v2`. Its `Static-Best` baseline was chosen by
`ablation_live.select_static_best`, which ranks candidates on **throughput
alone**. The comparison therefore scored an agent optimising one objective
against a baseline tuned for a different one.

That would be harmless if the two objectives agreed. They do not, and the
experiment's own calibration sweep is the evidence: `max_batch_size=32`
maximises throughput while `max_batch_size=12` maximises the held-out
utility axis. Under a mismatch, both "CARL ties on throughput" and "CARL wins on
utility" are artefacts of which yardstick each side was tuned to.

Additionally, that static search sampled 16 Latin-hypercube points from a 5-D
space of which **three dimensions cannot change execution in this harness** —
`spec_k`, `routing_threshold`, `eviction_threshold`. It looked wide and was
narrow.

### Defect 2 — cold-start missing-metric reward

In **every** seed, the first three control cycles logged

```
ttft_p99_ms = 0     tpot_p99_ms = 0
```

not because latency was zero but because no request had completed and
`MetricsTracker`'s windows were empty; `state._percentile` returns a NaN-free
`0.0`. `reward.latency_term` maps 0 ms to **1.0 — perfect latency** — so those
cycles collected the full TTFT + TPOT weight (0.5 of the maximum) for latencies
that had never been measured, and handed the result to LinUCB. LinUCB credits a
reward to the *previous* cycle's arm, and the first arm played in every regime is
arm 0 by construction, so the bias is directional: it inflates the hand-tuned
default before any evidence exists.

---

## What changed

| | reward-v2 run | this run |
|---|---|---|
| static baseline | `Static-Best`, argmax throughput, LHS ×16 over 5 dims (3 inert) | `Static-Tput` **and** `Static-Utility`, exhaustive over the 2 live dims |
| unmeasured TTFT/TPOT | scored as 0 ms = perfect | reward withheld (`None`), `reward_valid=false` |
| learner update on such a cycle | performed | **not performed**; no fill value substituted |
| warm-up | not recorded | recorded per seed: length, first valid cycle, duration |
| source provenance | `git_sha: null`, `git_dirty: null` | sidecar captured off-runtime and embedded |

**Held fixed:** Tesla T4 · TinyLlama-1.1B-Chat-v1.0 fp16 · seeds 42–51 · 200
requests · two-phase `NON-STATIONARY` workload · `OBSERVE_INTERVAL = 10` ·
held-out calibration seed 999 · restricted/expanded arm-set definitions ·
`RepairedLinUCBBandit`, α = 0.5 · `FEATURE_DIM = 10`.

---

## Search space: only what executes

`probe_live_dimensions()` reads the **live scheduler object** at run time and
aborts if anything it declares inert would in fact change execution. The
declared-inert set and the reason each is inert:

| dimension | why it cannot change execution here |
|---|---|
| `spec_k` | `_apply_sched` pins `enable_spec_decode=False`; `_serve` re-pins it after every controller step |
| `routing_threshold`, `cache_affinity_weight` | no router wired; `CARLController._set` drops the write |
| `eviction_threshold`, `eviction_window` | no KV cache wired; same |
| `preemption_enabled` | `ContinuousBatchScheduler` declares no such attribute |
| `use_cuda_graphs` | flag exists, but `_decode_forward` also needs `_graph_runner is not None` and `_new_scheduler` never attaches one — every decode falls back to eager (`graph_runner_missing`) |

What remains is `max_batch_size × chunk_size`, swept **exhaustively**:
7 × 4 = 28 configurations, on held-out seed 999, at the evaluation's own request
count. One sweep, three uses — frozen scales, `Static-Tput`, `Static-Utility` —
with the scales frozen from the *measurements* before any utility is computed,
so the utility baseline cannot have influenced the yardstick it is ranked by.

---

## Configurations

| name | policy |
|---|---|
| `Static-Tput` | fixed config; argmax throughput on seed 999 |
| `Static-Utility` | fixed config; argmax `utility_v2` (CARL's own objective, frozen scales) on seed 999 |
| `RuleOnly` | classify → `DEFAULT_CONFIGS[regime]`; no learning |
| `CARL-Repaired` | `RepairedLinUCBBandit` + restricted arms |
| `CARL-Expanded` | `RepairedLinUCBBandit` + expanded arms |

---

## Two utility estimators, never interchanged

- **`run_utility_v2`** — `utility_v2` of a run's end-of-run aggregates under the
  frozen scales. Defined for *every* config, including the statics; the basis of
  comparison B. For a config constant across the whole run, this simply *is* its
  utility.
- **`mean_valid_cycle_utility`** — mean over a controller's **scored** control
  cycles; what the learner actually consumed. Reported for controllers only and
  **not** compared against the statics, which have no control cycles.

---

## Hypotheses

### H1 — adaptation vs the throughput baseline (comparison A)
CARL-Repaired will not materially outperform `Static-Tput` on **throughput**.
The substrate is saturated at this operating point and the exhaustive held-out
sweep is expected to select the largest batch the axis offers.
- **Prediction:** paired mean difference ≤ 0, or a 95 % CI containing 0.
- **Falsified if:** CARL-Repaired exceeds `Static-Tput` throughput by more than
  2 % with a paired 95 % CI excluding 0.

### H2 — adaptation vs the utility baseline (comparison B)
CARL-Repaired will not materially outperform `Static-Utility` on **utility**
either. This is the hypothesis the previous experiment *could not test*, because
its only static baseline was tuned for a different objective. If CARL's apparent
utility advantage was an artefact of that mismatch, it disappears here.
- **Prediction:** paired mean difference ≤ 0, or a 95 % CI containing 0.
- **Falsified if:** CARL-Repaired exceeds `Static-Utility` run utility with a
  paired 95 % CI excluding 0.

### H3 — what learning bought (comparison C)
With cold-start rewards withheld, CARL-Repaired remains **behaviourally**
distinct from `RuleOnly` — it still moves between arms — but the distinction does
not convert into a throughput gain. H3 is about whether learning *does*
anything, not whether it *helps*.
- **Prediction:** arm-change rate strictly greater than `RuleOnly`'s 0;
  throughput difference of unspecified sign.
- **Falsified if:** CARL-Repaired is bit-identical to `RuleOnly` on every seed —
  which would mean the learner is inert once the cold-start rewards it had been
  consuming are withheld.

### H4 — what a wider action space bought (comparison D)
CARL-Expanded will not recover enough benefit to offset the extra exploration a
larger arm set costs.
- **Prediction:** paired mean difference ≤ 0.
- **Falsified if:** CARL-Expanded exceeds CARL-Repaired with a paired 95 % CI
  excluding 0.

### H5 — cold-start positive control
Every seed will show a non-empty warm-up: at least one leading control cycle
whose reward is withheld for want of a measured TTFT/TPOT.
- **Why it is registered:** a warm-up of zero everywhere would mean the validity
  gate never engaged, and the run would say nothing about the defect it was built
  to fix. This is the control that distinguishes "repaired" from "the repair was
  never exercised".
- **Prediction:** `warmup_cycles ≥ 1` for every controller run.
- **Falsified if:** any controller run reports `warmup_cycles == 0`.

H1, H2 and H4 predict null or negative results; H3 predicts a behavioural
difference of unspecified sign; H5 predicts the repair engages. Registering them
before the run is what makes a null result evidence rather than an absence of
evidence.

---

## Convergence rule (fixed before the run)

Convergence is **not** inferred from a mean reward. A regime's trace shows
convergence only if it **explored and then exploited**:

1. at least 6 decisions in that regime;
2. at least 2 distinct arms in the first half — there was exploration;
3. second-half modal-arm share ≥ 0.75;
4. modal share strictly **increased** from the first half to the second.

A high modal share with no exploration is a learner that never moved. That is
precisely the failure the as-published LinUCB already exhibited (200/200 on
arm 0), and it must not be reported as convergence. Where the rule is not met,
the artifact records `supported: false` and the reason.

---

## Gates (abort, not warn)

1. **Pre-run**, on the exhaustive held-out sweep: if the frozen scales do not
   separate candidates by at least `MIN_ARM_REWARD_SPREAD`, abort before
   spending GPU time.
2. **Per-run**, on each controller's stream of **valid** rewards. Withheld cycles
   carry no number and are excluded rather than padded — a stream padded with a
   constant would pass or fail for a reason unrelated to whether the reward
   discriminates.

A firing gate is a **result**, not a crash: the partial artifact is written with
the reason. It must not be re-run with a lowered threshold.

---

## Provenance

The reward-v2 runtime artifact carries `git_sha: null` and `git_dirty: null`
because the Colab archive excludes `.git`, so every `git` call on the runtime
fails and `provenance._run_git` returns `None` — indistinguishable from "clean".

`scripts/eval/repair/source_provenance.py` is run **on the machine that has the
repository**, before the archive is built, and captures: git HEAD (+ branch,
subject, commit time), the dirty flag, the full `git status --porcelain`, the
SHA256 of `git diff HEAD` (raw bytes), a per-file SHA256 manifest of all source
under `src/`, `scripts/`, `tests/` with a single manifest digest, the SHA256 of
the uploaded archive, and a creation timestamp.

The diff digest and the manifest cover disjoint failure modes: the diff covers
edits to *tracked* files and is empty when the tree is clean; the manifest covers
what is actually on disk, including the untracked source files this experiment
depends on. A run is reproducible if HEAD matches **and** both digests match.

`final_hardening.py` embeds that block verbatim under
`_provenance.source_provenance`. If the sidecar is absent it records the absence
and the remedy rather than emitting a null.

---

## Scope

One operating point: burst arrivals at ρ ≫ 1 on a single T4 with one 1.1 B
model, over the two live knobs this harness exposes. It does **not** establish a
load envelope, and it says nothing about speculation, routing or KV eviction,
which are inactive here.
