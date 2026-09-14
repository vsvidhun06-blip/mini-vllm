# Legacy / quarantined result artifacts

**Nothing in this directory may be cited by the paper.**

These files are preserved, not deleted, because deleting them would destroy the
evidence trail for how the project's headline numbers came to exist. They are
quarantined because their provenance does not survive inspection.

## `ablation_live_results.RECONSTRUCTED.json`

Source of the paper's flagship subsystem ablation: **Sched +28.51 tok/s, Router
+0.02, Spec −0.19, Chunk −0.37, Cache −0.65**.

**Why quarantined.** The file declares its own status:

```json
"status": "RECONSTRUCTED",
"generation_note": "Reconstructed from verified Colab GPU terminal output.
                    VM reset before JSON download. Numbers are real GPU
                    measurements."
```

It was therefore typed by hand, not emitted by `scripts/eval/ablation_live.py`.
That is independently confirmed by a schema comparison against
`ablation_live.py::_finalize()`:

| field | what the script writes | what this file contains |
|---|---|---|
| `subsystem_contributions` | flat `{name: delta}` | `{definition, ranked: [...]}` |
| oracle gap | `oracle_gap_pct` (float) | `oracle_gap` (object) |
| `linucb_vs_thompson` | `carl_full_linucb_tput`, `carl_thompson_tput`, `linucb_minus_thompson` | `linucb_tps`, `thompson_tps`, `delta_tps`, `note` |
| `static_best_selection` | written | **absent** |
| `live_effective_configs` | written | **absent** |
| `scope_note` | written | **absent** |
| `carl_overhead`, `dynoracle` | written | **absent** |
| `runs`, `requests`, `validation_seed` | written | **absent** |
| `slo_violations` (per config) | **written nowhere in the repository** | present |

`docs/eval/raw/ablation/` — the directory `_save_raw()` writes per-seed data to —
does not exist, so there is no raw data behind any of these numbers.

**Additional internal contradiction.** AutoTuner is reported with
`ttft_p99_mean: 61386.1` and `slo_violations: 0`, against `slo_ttft_ms: 200.0`.
A p99 of 61 seconds cannot coexist with zero violations of a 200 ms SLO. Since
`slo_violations` is written by no script, the field is an artifact of
transcription and carries no measurement.

**Also note** the reported `oracle_gap.gap_pct: -2.8` — CARL-Full (85.2) exceeds
DynOracle (82.9). An "oracle" that the policy beats is not an upper bound; the
DynOracle maximises *recorded reward*, and the reward on that run was constant
(see below), so its ranking is meaningless.

## `failure_cases_results.RECONSTRUCTED.json`

Source of the paper's failure-case claims: `rapid_oscillation −24.2%`,
`memory_pressure −15.7%`, "CARL loses 5/5 stress workloads".

**Why quarantined.**

```json
"generation_note": "Reconstructed from verified Colab GPU run output (PDF
                    artifact). Colab VM disconnected before JSON download."
"environment": { "gpu": null, "captured": false }
```

No environment was captured. Every value is rounded to one decimal place
(`78.0`, `1.0`, `2027.0`, `11.0`), and several fields are `null`
(`ttft_p50_mean`, `slo_rate_std`), both consistent with hand transcription from a
rendered table rather than serialisation from a run.

**Internal contradiction.** `memory_pressure` reports `ttft_p99_mean: 43232.0`
with `slo_rate_mean: 0.0`, while `single_queue` reports `slo_rate_mean: 100.0`.
The unit and polarity of `slo_rate_mean` are inconsistent between rows of the
same file.

**Statistical weakness even if the numbers are accurate.** The headline
`rapid_oscillation −24.2%` rests on n=3 seeds with CARL σ=7.8 against Static
σ=1.4.

## What has to happen before either can be cited

1. Re-run `scripts/eval/ablation_live.py` and `scripts/eval/failure_cases.py`
   end to end on a GPU.
2. Download the artifacts **before** the VM is torn down. (Both quarantined
   files exist because this step failed.)
3. Confirm `docs/eval/raw/ablation/` and the per-seed raw files are populated.
4. Confirm each result carries a `_provenance` block with `generated_by:
   "script"` — enforced by `tests/test_eval/test_provenance_and_schema.py`.

Hand-off commands are in `docs/eval/REPRODUCIBILITY.md`.

## Related: the reward was degenerate on the runs these files came from

`docs/eval/raw/adaptation/decisions_042.csv` — which *is* script-generated —
shows the controller's reward taking exactly two values across an entire GPU run:
`0.8` for three cycles, then `0.3` for every remaining cycle. All four reward
terms were saturated simultaneously (`throughput_ref=50` vs ~85 tok/s measured;
TTFT 200 ms SLO vs ~2200 ms measured; TPOT 50 ms SLO; cache hit rate 0).

The three per-seed traces `decisions_042.csv`, `decisions_043.csv` and
`decisions_044.csv` are **byte-identical**, which follows mechanically: with a
constant reward the controller is deterministic and the seed cannot influence it.

Consequently the paper's convergence and regret claims ("deterministic
convergence at cycle 9 across all seeds, cumulative regret 2.583") describe one
run reported three times. Those traces are retained in place under
`docs/eval/raw/adaptation/` as evidence of the defect.
