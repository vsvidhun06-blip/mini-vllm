#!/usr/bin/env bash
#
# run_phase.sh -- regenerate the three CARL eval artifacts at ONE commit and
# collect them into an output directory. Run once for the pre-fix commit and
# once for the post-fix commit; compare_results.py then diffs the two.
#
# Because we git-checkout different commits, this script and compare_results.py
# must live OUTSIDE the repo working tree (the Colab notebook copies the regen/
# folder to /content/regen_tools first). It is invoked from the repo root.
#
# Usage:
#   bash run_phase.sh <commit> <out_dir> [seeds_csv] [limit]
#
#   <commit>    e.g. f23d6ff (pre-fix, contains the bug) or 2e11321 (post-fix)
#   <out_dir>   where the 3 result JSONs + acceptance probe are copied
#   seeds_csv   optional, e.g. "42,43,44". Empty => each script's native default
#               (the ORIGINAL config: failure_cases/cross_model=42-44,
#               ablation_live=42-51). Keep before/after identical!
#   limit       optional requests-per-run override (default: script's own)
#
set -euo pipefail

COMMIT="${1:?need a commit}"
OUT="${2:?need an output dir}"
SEEDS="${3:-}"
LIMIT="${4:-}"

mkdir -p "$OUT"
echo "================================================================"
echo "PHASE  commit=$COMMIT  out=$OUT  seeds='${SEEDS:-native}'  limit='${LIMIT:-native}'"
echo "================================================================"

git checkout -q "$COMMIT"
echo "HEAD now: $(git rev-parse --short HEAD) -- $(git log -1 --format=%s)"

ARGS=""
[ -n "$SEEDS" ] && ARGS="$ARGS --seeds $SEEDS"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

export PYTHONPATH=.
PY="${PYTHON:-python}"

for script in failure_cases ablation_live cross_model; do
  echo; echo "--- running $script.py $ARGS ---"
  if ! $PY "scripts/eval/$script.py" $ARGS; then
    echo "ERROR: $script.py failed at $COMMIT" >&2
    exit 1
  fi
done

# Acceptance rate is NOT persisted by the eval scripts; capture it explicitly so
# the before/after acceptance delta (the headline fix signal: ~0% -> ~1.1% at
# depth 8) is recorded per phase.
echo; echo "--- acceptance probe (depth sweep) ---"
$PY scripts/probe_spec_acceptance.py > "$OUT/acceptance_probe.txt" 2>&1 || \
  echo "WARN: acceptance probe failed (non-fatal)"

# Collect the artifacts the eval scripts wrote to docs/eval/.
for j in failure_cases_results ablation_live_results cross_model_results; do
  if [ -f "docs/eval/$j.json" ]; then
    cp -f "docs/eval/$j.json" "$OUT/"
    echo "collected $j.json"
  else
    echo "WARN: docs/eval/$j.json missing after run" >&2
  fi
done

echo "PHASE DONE: $OUT"
