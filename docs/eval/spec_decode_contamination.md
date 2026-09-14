# Speculative-decoding regression: eval contamination record

**Status:** spec-decode regression fixed in `2e11321`. Affected eval artifacts
were never persisted and are unrecoverable; no numerical before/after is
possible. Regeneration must happen in a GPU environment (see Feasibility).

## Regression window

| | Commit | Date | Meaning |
|---|---|---|---|
| Last good | `5e28446` | 2026-06-11 17:32 | parity passes (bisect) |
| **First bad** | **`9a5a4c9`** | **2026-06-11 18:17** | "Add CUDA graph capture/replay for the decode step" — introduces the bug |
| Fix | `2e11321` | 2026-06-18 | invalidate stale `_bt_tensor` on spec-decode rewind |

Confirmed correct at Day 16 (`ee34e93`, 2026-05-16); regression bisected to
`9a5a4c9`; first bad commit verified with `git bisect run` over
`test_spec_decode_matches_greedy`.

## Root cause

`9a5a4c9` added a per-request cached block-table device tensor
(`PagedRequestCache._bt_tensor`) keyed only on `len(block_table)`, assuming the
block table only grows and existing entries never move. Speculative decoding's
`rewind_cache` violates that: it frees tail blocks after partial rejection, and
the next `allocate_block` can hand the same logical tail slot a *different*
physical block at the *same* length. The length check then passes and the stale
tensor gathers K/V from the wrong block, corrupting decode just past the first
`block_size`-token boundary. The H2O eviction path (`kv_eviction.py`, after
`trim_request_blocks`) already invalidated `_bt_tensor` for the same reason; the
spec-decode rewind path did not. Fix mirrors that invalidation at both
spec-decode rewind sites.

Observed symptoms (HEAD before fix): greedy parity diverged at generated token
~11 (≈ absolute position 17); draft acceptance 0.0% (0 of 196). After fix:
byte-identical greedy parity; depth-8 acceptance restored to 1.1% (matches the
documented pre-regression value). 10/10 targeted spec-decode tests pass.

## Affected artifacts and validity

These three artifacts were generated 2026-06-16 (timestamp 14:20:09) from a
checkout whose HEAD was `10876ad` (2026-06-15 20:20) — which has `9a5a4c9` as
an ancestor (linear history, 0 merges; the eval scripts themselves postdate the
regression by 4 days, so no checkout that contains them can omit it):

- `docs/eval/failure_cases_results.json`
- `docs/eval/ablation_live_results.json`
- `docs/eval/cross_model_results.json`

**Any `spec_k > 0` result in these files is INVALID** and must be regenerated
after the fix. `spec_k = 0` configurations are unaffected (the bug only triggers
with `enable_spec_decode=True`, a single decode request, crossing a block
boundary).

## Pre-fix artifacts are unrecoverable

No persisted pre-fix copy of any of the three files exists. Searched and found
nothing in: working tree, all `*.json` under the repo, `.gitignore`d paths
(`git status --ignored`), every commit on every branch (`git log --all
--diff-filter=A`), tags, stashes, the reflog, dangling objects (`git fsck`), and
the notebook (`docs/run_carl_evals.ipynb` has no executed output cells). The
only references to these filenames are output-path strings in the eval scripts
and commit-message text. This matches `overhead_reconciliation.md`, which
already noted the live-eval JSONs "were never committed." **Therefore no
numerical before/after comparison is possible — only a fresh post-fix baseline.**

## Feasibility of regeneration in this environment

This machine is CPU-only (`torch.cuda.is_available()` is False; torch is
`2.x+cpu`). `failure_cases.py` itself prints
`WARNING: no CUDA -- CPU smoke only; run on a Colab GPU for real numbers`, so a
CPU run does not produce a trustworthy baseline regardless of runtime.

Measured per-request cost (real code path, this CPU): ultra_short ~0.9 s/req,
memory_pressure ~11.5 s/req. `failure_cases.py` original config (seeds 42/43/44,
n=30) is 16 LHS-validation runs + 6 measurement runs per scenario × 5 scenarios
(~2,100 request-generations, ~420 of them in the memory_pressure scenario).
Estimated wall time ≈ 2.5–5 h for `failure_cases.py` alone, dominated by
memory_pressure. `ablation_live.py` (10 configs × N=10 seeds + validation) is
heavier; `cross_model.py` loads multiple non-TinyLlama checkpoints that are
likely not cached (download/availability blocker).

**Conclusion:** a trustworthy post-fix baseline cannot be produced here.
Regenerate on GPU (e.g. Colab) with the original seeds (42/43/44) and unchanged
scenario definitions, then compare `spec_k > 0` rows against this record's
notes (winner, margin_pct, throughput, TTFT, SLO rate, acceptance).

## Post-fix regeneration status (commit `2e11321`)

A post-fix GPU regeneration was run on Colab, but the VM disconnected before the
detailed JSON artifacts could be downloaded; only a PDF artifact of the run
output was preserved. Current state of the three committed files:

- `failure_cases_results.json` — **reconstructed** from the verified Colab GPU
  PDF artifact after the VM disconnected before download. Values are real GPU
  measurements from `2e11321`; metrics absent from the PDF (TTFT p50, SLO-rate
  std, `static_best_config`) are left `null` and were not inferred or
  backfilled. See its `regeneration_metadata` block.
- `ablation_live_results.json` and `cross_model_results.json` — **schema-compliant
  placeholders** (`status: "PLACEHOLDER"`, `rerun_required: true`) pending a
  future GPU rerun, because the detailed regenerated outputs for these two were
  not recoverable from the captured artifact. Rerun `scripts/eval/ablation_live.py`
  and `scripts/eval/cross_model.py` on GPU at `2e11321` to populate them.
