# Stakeholder summary — speculative-decoding bug & eval impact

**For:** Marko **Date:** 2026-06-18 **Status:** fix landed; affected evals pending GPU re-run

## What happened
A correctness bug in **speculative decoding** was found. It was introduced on
2026-06-11 by the CUDA-graph commit (`9a5a4c9`), which cached a block-table
tensor that went stale when speculative decoding frees/reuses cache blocks. With
speculation on, decode read the wrong cache blocks — producing garbage output
and **0% draft acceptance** (vs the intended ~1%).

## Contamination — confirmed
Our CARL evaluation artifacts — `failure_cases`, `ablation_live`, `cross_model`
— were generated on **2026-06-16**, while the bug was live. Any result using
**speculative decoding (`spec_k > 0`)** in those files is therefore **invalid**.
Results with speculation off (`spec_k = 0`) are unaffected.

Note: the original result files were never committed and cannot be recovered, so
there is no saved baseline to diff against — we must regenerate from scratch.

## Fix — validated
Root cause fixed in commit **`2e11321`** (mirrors a cache-invalidation we already
had on the eviction path). Verification on the fixed code:
* speculative output is now **byte-identical** to standard greedy decoding;
* draft acceptance restored to **~1.1%** at the default depth;
* **10/10** speculative-decoding tests pass.

## What's needed next
The original evals were **GPU** runs; this validation machine is CPU-only, and
the eval code itself flags CPU numbers as smoke-tests only. So the affected evals
must be **re-run on GPU**. A ready-to-run Colab package is prepared
(`scripts/eval/regen/`) that regenerates pre-fix and post-fix numbers and
auto-flags any winner flips or ranking changes. Estimated runtime: a few GPU-hours.

## Bottom line
* The bug was real, is understood, and is fixed.
* **All `spec_k > 0` evaluation results — and in particular the failure-case
  conclusions — should be treated as PROVISIONAL until the GPU regeneration is
  complete.**
* `spec_k = 0` results and all non-speculative conclusions stand.

Detail: `docs/eval/spec_decode_contamination.md`.
