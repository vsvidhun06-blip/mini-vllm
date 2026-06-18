# Spec-decode fix: eval regeneration package

Regenerates the three CARL eval artifacts on GPU and produces a pre-fix vs
post-fix comparison, after the speculative-decoding regression
(`9a5a4c9`, fixed in `2e11321`). Background: `docs/eval/spec_decode_contamination.md`.

## Why "regenerate both sides"

The original artifacts are unrecoverable (never committed; absent from git
history, reflog, fsck, notebook outputs). There is therefore **no baseline to
diff against**. Instead this package regenerates *both* sides on the same GPU:

* `before/` = pre-fix commit **`f23d6ff`** (contains the bug)
* `after/`  = post-fix commit **`2e11321`**

so every delta is cleanly attributable to the fix. Do **not** run on CPU: the
eval scripts print `CPU smoke only ... run on a Colab GPU for real numbers`.

## Contents

| file | role |
|---|---|
| `Regenerate_spec_fix_evals.ipynb` | single ready-to-run Colab workflow (start here) |
| `run_phase.sh` | regenerate the 3 artifacts at one commit; collect into an out dir |
| `compare_results.py` | pure-stdlib diff of before/ vs after/; flags spec_k>0 impact |

## Quick start (Colab)

1. Open `Regenerate_spec_fix_evals.ipynb` in Colab, set runtime to **GPU**.
2. Run cells top to bottom. It clones the repo, installs deps, runs the pre-fix
   then post-fix phase, and renders the comparison report.

**Prerequisite:** commits `f23d6ff` and `2e11321` and this `regen/` folder must
be pushed to `origin` first.

## Manual / CLI equivalent

```bash
# from the repo root, with GPU + deps installed
cp -r scripts/eval/regen /tmp/regen_tools           # survive git checkout
bash /tmp/regen_tools/run_phase.sh f23d6ff out/before
bash /tmp/regen_tools/run_phase.sh 2e11321 out/after
python /tmp/regen_tools/compare_results.py \
    --before out/before --after out/after --out out/comparison_report.md
```

## What the comparison reports

For every `spec_k > 0` row (and all CARL-Full rows, which control spec_k
dynamically): throughput, TTFT p50/p99, and SLO-rate deltas. For
`failure_cases`: winner and `margin_pct` per scenario, with **winner-flip**
flags. For `ablation_live`: per-config deltas (spec_k tagged from `config.spec_k`)
and subsystem-contribution **ranking-change** flags. Acceptance rate is captured
separately per phase via `probe_spec_acceptance.py` (the eval JSONs do not
persist it); the headline signal is depth-8 acceptance **~0% (pre-fix) -> ~1.1%
(post-fix)**.

A `TOP-LEVEL FLAGS` section lists every winner flip and ranking change — these
are the conclusions that need correction in the paper.

## Seeds / configuration caveat

`run_phase.sh` and the notebook default to each script's **native** seeds, which
equal the original runs:

* `failure_cases.py`, `cross_model.py`: seeds **42, 43, 44**
* `ablation_live.py`: seeds **42–51** (N=10)

The task requested "seeds 42–44" uniformly. You can force that with
`SEEDS='42,43,44'`, but for `ablation_live` that **deviates from its original
N=10** configuration — a revalidation-vs-original choice for the operator. The
only hard invariant is that `before/` and `after/` use **identical** seeds.

## Note on `spec_k` identification

* `ablation_live`: `spec_k` is an explicit per-config field — tagged exactly.
* `failure_cases` / `cross_model`: CARL-Full sets `spec_k` *dynamically*, so
  there is no fixed per-row value. All CARL-Full rows are treated as
  "spec-capable"; the before/after delta reveals whether speculation was active
  and corrupted. Static-Best rows are tagged `spec_k>0` only if their selected
  config carries it.
