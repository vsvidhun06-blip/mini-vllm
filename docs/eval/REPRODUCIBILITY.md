# Reproducibility

**Status: partially reproducible. GPU results are NOT currently reproducible
because no committed GPU artifact has verifiable provenance.** See
`docs/eval/legacy/README.md`.

## The rule

A result file is admissible only if it carries a `_provenance` block with
`generated_by: "script"`, written by `src/eval/provenance.write_result`. This is
enforced by `tests/test_eval/test_provenance_and_schema.py`, which also fails if
any file containing the string `RECONSTRUCTED` sits in the live results
directory.

Provenance travels *inside* the result. The previous scheme wrote
`docs/eval/environment.json`, a single host-specific file that every script
overwrote and that `.gitignore` excluded — so no committed result had a
committed environment record, and a stale CPU capture was once shipped alongside
GPU numbers.

Each block records: git SHA + dirty flag, UTC timestamp, hostname, Python,
platform, numpy, torch, CUDA, GPU name, plus `reward_version` and
`workload_version`. The last two matter because both the reward and the workload
generator changed during the repair pass; without them, two disagreeing result
files are indistinguishable from two files measuring different things.

| version | meaning |
|---|---|
| `reward v1` | `min(1, tps/50)` + fixed 200 ms / 50 ms SLO indicators. **Saturates.** |
| `reward v2` | operating-range normalisation + smooth SLO margins. Current. |
| `workload v1` | bulk dump — all phase-0 at t=0, phase-1 dumped mid-run. |
| `workload v2` | configurable arrival process; v1 retained as the `burst` stress mode. |

## What reproduces on CPU, today

No GPU required. Total runtime a few minutes.

```bash
python -m pytest tests/test_carl tests/test_eval -q       # 45 tests
python scripts/eval/repair/bandit_null_check.py           # Phase 1
python scripts/eval/repair/rule_only_ablation.py          # Phase 4+5
```

Outputs, all provenance-stamped:

- `docs/eval/raw/repair/bandit_null_check.json`
- `docs/eval/raw/repair/rule_only_ablation.json`
- `docs/eval/reward_diagnostics.md`

## What requires a GPU, and how to run it

This development host is CPU-only (`torch 2.12.0+cpu`,
`torch.cuda.is_available() == False`). The scripts below are implemented and
import-tested here but **have not been run**; no number in this repository's
repair artifacts came from them.

`batch_intervention.py` **refuses to run without CUDA** rather than emitting
placeholder numbers, and writes its artifact **incrementally after every row** —
both are direct mitigations for the failure that produced the quarantined files
(a Colab VM torn down before the artifact was downloaded).

### 1. Batch-size intervention + host-overhead check (Phases 15, 16) — highest priority

**Run the smoke first, and check the graph arm before scaling up.** The first T4
smoke returned four rows in which `cuda_graph_hits == 0` on *both*
`graphs_requested=True` rows: `use_cuda_graphs=True` only gates a branch that
also requires `scheduler._graph_runner`, and nothing ever assigned it, so the
"CUDA graph" arm was eager execution wearing a graph label. That is repaired
(`src/engine/live_graph.py`), but the repair is unverified on hardware until this
smoke passes.

#### STEP 0 — sync the repair to the GPU box (do not skip)

The **second** smoke returned byte-identical fallback counts (2350 / 627) and
zero hits again. The cause was not code logic: the GPU box was running a bundle
built *before* the repair, so `src/engine/live_graph.py` was not present and no
runner could be attached. Two runs of different code were indistinguishable in
the results, because nothing recorded which implementation produced them.

Use `mini-vllm-repair-v2.zip` (the pre-repair `mini-vllm-repair.zip` is kept as
the record of what the failed runs actually executed). On the GPU box, confirm
before running:

```bash
python -c "import sys; sys.path.insert(0,'scripts/eval/repair'); \
from batch_intervention import graph_repair_provenance as p; print(p())"
# live_graph_present must be True and live_graph_sha256 must be 189b7a249a40699d
```

The graph arm now **refuses to run** from a checkout without the repair, and
every artifact carries a `code_provenance` block. Cross-check
`code_provenance.live_graph_sha256` in the JSON against the local tree before
believing any graph row.

```bash
# STEP 1 -- smoke. Must pass the gate below before the full sweep is run.
python scripts/eval/repair/batch_intervention.py \
    --batches 1,4 \
    --seeds 42 \
    --rates burst
```

Gate — for **every** row with `use_cuda_graphs_requested == true`:

| field | required |
|---|---|
| `cuda_graph_arm_valid` | `true` |
| `cuda_graph_hits` | `> 0` |
| `cuda_graph_hit_rate` | `> 0` (expect ≈ 1.0) |
| `cuda_graph_capture.capture_failures` | `[]` (a failure now raises) |
| `cuda_graph_selftest.max_abs_logit_delta` | `0.0` (replay == eager) |
| `cuda_graph_diagnostics.reasons` | contains `graph_hit`, no `graph_runner_missing` |
| `code_provenance.live_graph_sha256` | matches the local tree |

and `analysis.graph_arm_usable == true`. If any row fails, **stop** and read
`cuda_graph_diagnostics.reasons`, which holds one entry per batched-decode
forward and names the cause without a rerun:

| reason | meaning | fix |
|---|---|---|
| `graph_runner_missing` | runner never attached — **stale code on the box** | re-sync (STEP 0) |
| `graphs_disabled` | `use_cuda_graphs` was False | check the arm wiring |
| `runner_rejected` | attached but refusing | see `runner_fallback_reasons` |
| `graph_hit` | replayed a captured graph | — |

`runner_fallback_reasons` then distinguishes `batch_size_not_captured`,
`context_exceeds_largest_bucket`, `foreign_kv_pool`, `ragged_layer_seq_lens`,
`bucket_capture_failed`. Do **not** compare eager against graph rows, and do
**not** fit `HardwareProfile`, from a run that fails this gate.

If capture OOMs (the grid is `1..max_batch × --graph-seq-buckets` graphs, so 128
at `--batches 32`), lower `--graph-seq-buckets` or `--graph-max-batch` rather
than accepting failed captures — a failed capture becomes an eager fallback and
correctly invalidates the arm.

```bash
# STEP 2 -- full sweep, only once STEP 1 passes the gate.
python scripts/eval/repair/batch_intervention.py \
    --batches 1,2,4,8,12,16,24,32 \
    --seeds 42,43,44 \
    --rates burst,2.0,8.0
```

Produces `docs/eval/raw/repair/batch_intervention.json`. Answers three things:

1. Whether forcing `max_batch_size` actually changes the **realised** batch
   (`cap_was_binding`). The simulation predicts it does **not** below saturation.
2. The measured `throughput(batch)` curve — the mechanism the paper asserts and
   has never measured.
3. Whether that curve survives CUDA-graph capture. The simulation predicts the
   batching benefit shrinks from **1.84× to 1.37×** when host overhead is
   removed, i.e. roughly half the effect is Python dispatch amortisation. If the
   GPU agrees, that is a boundary condition on external validity.

It also emits a fitted `HardwareProfile` (`analysis.hardware_profile_fit_*`),
which replaces the anchored guesses in `src/eval/engine_model.py`. Re-run the
CPU experiments afterwards with the calibrated profile.

### 2. Regenerate the quarantined flagship tables

```bash
python scripts/eval/ablation_live.py --seeds 42,43,44,45,46,47,48,49,50,51
python scripts/eval/failure_cases.py --seeds 42,43,44,45,46,47,48,49,50,51
```

Before citing either result, confirm:

- `docs/eval/raw/ablation/` exists and holds per-seed files;
- the result carries `_provenance` with `generated_by: "script"`;
- `tests/test_eval/test_provenance_and_schema.py` passes;
- the SLO block passes `src.carl.slo.check_consistency` (the old file reported
  `slo_violations: 0` with a 61-second TTFT p99).

**Download the artifacts before the VM is torn down.**

### 3. Per-knob attribution (Phase 14)

```bash
python scripts/eval/knob_attribution.py --seeds 42,43,44,45,46,47,48,49,50,51
```

Now reports **raw** deltas with `delta_pooled_std`, `delta_exceeds_1std` and
`delta_exceeds_2std`. The previous version rescaled every delta so the set summed
to the headline +28.51 tok/s, which turned noise into a tidy percentage
decomposition — and normalised against a number whose source artifact is now
quarantined.

### 4. Arrival-process re-measurement (Phase 10)

`src/carl/live.py` now exposes `build_arrivals()` with `burst` / `poisson` /
`deterministic`, and `scripts/eval/ablation_live.py` honours per-request arrival
offsets and records per-stage timestamps (`arrival_target_s`, `submit_s`,
`queue_entry_s`, `first_token_s`, `finish_s`) so TTFT can be decomposed into
queue wait versus prefill. Under the legacy bulk dump nearly all of TTFT was
queue wait, which is why an 18-second TTFT p99 was reported for a 1.1B model on a
T4.

Calibrate λ against a measured service rate with
`scripts/eval/arrival_probe.py` so offered load ρ is a stated condition rather
than an accident.

## Known reproducibility defects, not yet fixed

| Defect | Impact | Fix |
|---|---|---|
| `requirements.txt` is UTF-16LE with BOM (`pip freeze` under PowerShell) | `pip install -r` cannot parse it | re-export as UTF-8 |
| No lockfile, Dockerfile, or `reproduce.sh` | environment not pinned | add |
| No LICENSE / CITATION | artifact evaluation blocker | add |
| `docs/eval/environment.json` is gitignored | superseded by embedded provenance, but the ignore rule remains | remove the rule once all results carry `_provenance` |
| `paper/figures/` is empty; 5 of 7 paper sections are stubs | not an eval issue, but blocks submission | — |
| Legacy sim results (`ablation_results.json`, `oracle_results.json`, `stats_results.json`, `trace_replay_results.json`, `stability_100k_results.json`, …) were produced by the circular cost model | numbers are not wrong *as simulation output*, but the substrate cannot demonstrate learning | re-run on `src/eval/engine_model` |

## Test baseline

Before this pass: **207 passed, 1 failed, 22 skipped**. The pre-existing failure
is `tests/test_engine/test_lora.py::test_mixed_batch_with_one_base_row` and is
unrelated to CARL — it was failing at `dfd3820` before any change here.
