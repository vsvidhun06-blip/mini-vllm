# CARL Evaluation Suite

Rigorous evaluation of **CARL** (the Coordinated Adaptive Runtime Learner): an
ablation study, a workload-diversity sweep, an oracle-gap analysis, a sensitivity
analysis, and a statistical-significance test. The scripts live in
[`scripts/eval/`](../../scripts/eval) and write their results here as JSON.

---

## ⚠️ Read this first: these are SIMULATION results

Every script in this suite is a **control-loop simulation**, not a GPU benchmark.
It drives the **real** CARL machinery — the real `CARLController`, the real
LinUCB / Thompson bandits, the real `AutoTuner`, the real workload-regime
classifier — but the serving metrics (throughput, TTFT, TPOT, cache/spec rates)
come from an **analytical cost model** (`scripts/benchmark_carl.py`'s
`WorkloadModel`), **not** from running TinyLlama on a GPU.

**Why a simulation.** The contribution under test is the *controller*: does a
unified online bandit that jointly adapts every knob converge to the per-regime
optimum and beat independent tuning under non-stationary load? Answering that
needs hundreds of regime-varying control cycles across many seeds — and, for the
ablations, the ability to switch individual *subsystems* (scheduler / spec /
cache / router / chunking) on and off as adaptive knobs. The real-inference
harness (`src/carl/live.py`) only ever wires the controller to the **scheduler**,
so it physically cannot express the spec/cache/router ablations. A transparent
CPU simulation is therefore the honest, reproducible substrate for the policy
questions, and it runs anywhere in under a second per script.

**What this means for the numbers.**

* **Comparisons between methods are the result; absolute tok/s are illustrative.**
* The **regime oracle is near-optimal *by construction*** — the cost model
  encodes the same domain knowledge as the hand-tuned `DEFAULT_CONFIGS`. CARL's
  job is to *learn online* to approach that known-good target without being told
  the regime. We do **not** claim a measured wall-clock speedup on hardware.
* For a **measured** real-inference check (TinyLlama on a GPU, CARL adaptive vs a
  fixed baseline), see the complementary live harness:
  `python scripts/benchmark_carl.py --live` and **cell 6c** of
  [`docs/run_benchmarks.ipynb`](../run_benchmarks.ipynb).

---

## Hardware & environment

| Aspect | This eval suite | Real-inference validation (cell 6c) |
|---|---|---|
| Execution | **CPU**, torch-free simulation | GPU forward passes |
| Hardware | any machine; no GPU needed | **Tesla T4** (Colab) |
| Model | none (analytical cost model) | **TinyLlama-1.1B** |
| Deps | `numpy`; `tqdm` optional; `datasets` optional (LMSYS) | `torch`, `transformers`, `triton` |
| Runtime | < 1 s per script (defaults) | minutes (real generation) |

The suite needs only the Python standard library plus **numpy** (used by the
bandits). `tqdm` is used for progress bars if installed and silently skipped
otherwise. The t-test in `statistical_validation.py` is computed from scratch
(regularised incomplete beta) — **no scipy required**.

---

## How to reproduce

Each script is runnable standalone from the repo root and self-bootstraps its
import path:

```bash
python scripts/eval/ablation.py                 # ablation study (simulation)
python scripts/eval/workload_suite.py           # workload diversity
python scripts/eval/oracle_comparison.py        # oracle gap
python scripts/eval/sensitivity.py              # sensitivity sweeps
python scripts/eval/statistical_validation.py   # paired t-test
python scripts/eval/ablation_live.py            # ablation on REAL TinyLlama (GPU)
```

`ablation_live.py` is the only **measured** (non-simulation) script: it reuses
`src/carl/live.py`'s harness to run the ablation through actual TinyLlama
inference on a GPU. See "Live ablation" below for its important scope limitation.

Useful flags (every script has `--out` to redirect its JSON):

```bash
python scripts/eval/ablation.py --runs 5 --requests 30      # defaults
python scripts/eval/ablation.py --runs 3 --requests 20      # fast preview (= notebook cell 6d)
python scripts/eval/workload_suite.py --real                # try real LMSYS-Chat-1M
python scripts/eval/workload_suite.py --skip long_context   # skip a workload
python scripts/eval/statistical_validation.py --runs 30     # N independent paired runs
```

On Colab, **cell 6d** of `docs/run_benchmarks.ipynb` runs the ablation at reduced
settings (3 runs × 20 requests) as a fast preview; the full suite is meant to be
run locally (it is CPU-only and finishes in seconds).

### Random seeds

All scripts use **seeds `0 … N-1`** where `N` is the run count (`--runs`).
Multi-seed runs report **mean ± sample std** across those seeds. For a fixed
workload the regime *sequence* is held constant across seeds; the seed varies
only the cost-model noise draw, so method differences are never confounded by a
different request stream. `statistical_validation.py` is **paired**: CARL and
Static-Best see the same seed on each run, so the t-test runs on the per-seed
difference.

---

## The configurations

| Name | What adapts | Role |
|---|---|---|
| `CARL-Full` | all 5 subsystems (LinUCB) | the proposed controller |
| `CARL-NoSched` | all but `max_batch_size`+`chunk_size` (pinned to defaults) | ablation |
| `CARL-NoSpec` | all but speculation (`spec_k`=0) | ablation |
| `CARL-NoCache` | all but eviction (`eviction_threshold`=0.8) | ablation |
| `CARL-NoRouter` | all but routing (`routing_threshold`=0.5) | ablation |
| `CARL-NoChunk` | all but `chunk_size` (=256) | ablation |
| `Static-Best` | nothing — best single fixed config for the workload | baseline |
| `AutoTuner` | per-component, bottleneck-reactive (existing) | baseline |
| `Oracle` | perfect regime knowledge → `DEFAULT_CONFIGS[regime]` | upper bound |

---

## How to interpret each result

### Oracle gap (`oracle_comparison.py`)
The Oracle is *told* the true regime and applies its hand-tuned optimal config,
so no online method can beat it. The **gap** is therefore an upper-bound metric:

```
throughput_gap% = (oracle_throughput - method_throughput) / oracle_throughput × 100
slo_gap%        =  oracle_slo_rate    - method_slo_rate
```

A gap near **0%** means the method has effectively reached perfect-knowledge
performance. In our runs `carl_linucb` sits within ~0–2% of the oracle while
`autotuner`/`static_default` trail 11–25%. **Because the oracle is near-optimal
by construction, the gap measures online-learning quality in simulation — it is
not a claim that CARL would be within 2% of optimal on real hardware.**
*Adaptation lag* is the number of requests after a regime transition until CARL's
detected regime tracks the new one (≈ 0–1 request here).

### Ablation (`ablation.py`)
Per-subsystem contribution is `delta = CARL-Full − CARL-NoX` (throughput): how
much removing that subsystem costs. Positive = the subsystem pulls weight.
**Note:** `CARL-NoRouter`'s delta is ≈ 0 because `routing_threshold` is **not a
modelled performance lever** in this analytical cost model (the cost model's
match score covers batch/chunk/spec/eviction/cache-affinity, not the routing
threshold). That is a property of the simulation, reported honestly, not a bug.

### Workload diversity (`workload_suite.py`)
CARL **ties** Static-Best on homogeneous single-regime workloads — when the
regime never changes, one fixed config is already optimal, so there is nothing to
adapt to (and CARL is never *worse*). CARL **wins** under heterogeneity (mixed
lengths, bursty arrivals, the LMSYS mix), which is exactly where joint online
adaptation is supposed to help. `lmsys_sample` uses real LMSYS-Chat-1M prompts
when `--real` is set and HF auth is available, and falls back to a synthetic
length mix otherwise.

### Sensitivity (`sensitivity.py`)
Request rate and context length are **not** direct levers in the cost model, so
they are mapped through the real regime classifier (rate → queue backlog →
BATCH/BURST; length → `classify_regime`). This mapping is documented at the top
of the script. Read these sweeps as *"CARL stays effective across the range"*:
it ties Static-Best where the setting stays in the baseline regime and gains
~8–11% throughput once the setting induces a contrasting regime.

### Live ablation (`ablation_live.py`) — MEASURED, not simulated
This is the hardware counterpart to `ablation.py`: the same eight configs, but run
through real TinyLlama inference (`src/carl/live.py`'s `ContinuousBatchScheduler`)
over NON-STATIONARY (25 INTERACTIVE → 25 BATCH), 3 runs, reporting measured
throughput / TTFT / TPOT. **Scope limitation (read before interpreting):** the
live harness is single-model and wires CARL to the **scheduler only**, with
speculation pinned off (TinyLlama self-spec is below break-even), no router, and
KV eviction inactive at these sizes. Therefore **only `CARL-NoSched` and
`CARL-NoChunk` (plus `Static-Best`/`Oracle`) differ from `CARL-Full`** —
`CARL-NoSpec`/`CARL-NoCache`/`CARL-NoRouter` measure identically to `CARL-Full`
and are marked `no*` in the table's `live?` column. That is the honest result: it
shows the *scheduler* is what moves real single-GPU TinyLlama metrics, while the
simulation ablation is the one that can vary all five subsystems. Needs a GPU
(runs on CPU as a slow smoke test).

### Statistical validation (`statistical_validation.py`)
A **paired t-test** (H0: equal mean throughput) over N=30 seeds. We report the
effect size, 95% CIs and the two-sided p-value. **Caveat:** the cost model is
near-deterministic given a seed, so seed-to-seed variance is tiny and the p-value
is astronomically small (and the t-statistic huge). **Trust the effect size**
(≈ +10% throughput, ~+6 tok/s), not the raw p-value — a low-noise simulation's
p-value is not comparable to one measured on hardware.

---

## Output files

| Script | Output |
|---|---|
| `ablation.py` | `docs/eval/ablation_results.json` |
| `workload_suite.py` | `docs/eval/workload_results.json` |
| `oracle_comparison.py` | `docs/eval/oracle_results.json` |
| `sensitivity.py` | `docs/eval/sensitivity_results.json` |
| `statistical_validation.py` | `docs/eval/stats_results.json` |
| `ablation_live.py` (GPU) | `docs/eval/ablation_live_results.json` |

All tables are printed as GitHub-flavoured pipe tables, so the notebook's
`to_md_table()` renders them straight into `docs/benchmarks.md`.
