#!/usr/bin/env bash
# Batch 3 regression A/B: the workspace's own modified files are temporarily
# restored to HEAD, the suite is run, then the batch-3 versions are put back and
# the suite is run again.  Failures are diffed by pytest NODE ID, so "same count,
# different nodes" cannot hide a regression.
#
# Usage:  bash evidence/_batch3_test_ab.sh
set -uo pipefail

REPO=/home/sunhaiwei/quant_projects
cd "$REPO" || exit 1

FILES=(
  "factor_engine/cleaned_operators/price_volume/polars_liquidity_v2.py"
  "factor_engine/cleaned_operators/technical/polars_tech_misc.py"
  "factor_engine/cleaned_operators/fundamental/polars_fundamental.py"
  "factor_engine/cleaned_operators/intraday/polars_next_stage.py"
  "factor_engine/cleaned_operators/intraday/polars_intraday_full.py"
  "factor_engine/cleaned_operators/common/polars_ops.py"
  "factor_engine/cleaned_operators/common/polars_daily_native.py"
  "factor_engine/cleaned_operators/common/polars_auto.py"
  "evidence/_verify_batch_specs.py"
)

BAK=/tmp/b3_backup.$$
mkdir -p "$BAK"

restore_batch3() {
  for f in "${FILES[@]}"; do
    if [ -f "$BAK/$(echo "$f" | tr '/' '_')" ]; then
      cp "$BAK/$(echo "$f" | tr '/' '_')" "$REPO/$f"
    fi
  done
  echo "== batch-3 versions restored =="
}
trap restore_batch3 EXIT

# 1) snapshot the batch-3 versions
for f in "${FILES[@]}"; do
  cp "$REPO/$f" "$BAK/$(echo "$f" | tr '/' '_')"
done
echo "== batch-3 snapshot taken, sha256 =="
for f in "${FILES[@]}"; do
  shasum -a 256 "$REPO/$f" | awk '{print $1"  '"$f"'"}'
done

run_suite() {
  local label="$1"
  echo "== running suite: $label  ($(date -Is)) =="
  # Scope matches the batch-2 A/B (`/tmp/b2_test_ab.sh`) exactly, so the two
  # batches are directly comparable: the two directories that carry the backend
  # parity and operator-contract surface.
  # NOTE: do NOT export ASHARE_PARQUET_ROOT / DATA_ACCESS_SKIP_COS_MIRROR here.
  # With real data wired in, a handful of modules scan the whole parquet tree and
  # the session stalls (observed: no output growth for >10 minutes at 75%).
  # --continue-on-collection-errors keeps the session alive for the pre-existing
  # collection failures in factor_engine/tests/tools/.
  PYTHONPATH="$REPO" \
    python3 -m pytest factor_engine/tests/backend_parity factor_engine/tests/operator_contracts \
    -q -p no:cacheprovider --tb=no --continue-on-collection-errors -rfE \
    > "$BAK/$label.out" 2>&1
  tail -3 "$BAK/$label.out"
  grep -E '^(FAILED|ERROR) ' "$BAK/$label.out" \
    | sed 's/^FAILED //; s/^ERROR //' | awk '{print $1}' | sort -u > "$BAK/$label.nodes"
  grep -c . "$BAK/$label.nodes" | sed "s/^/unique failing\/erroring nodes ($label): /"
  grep -c '^FAILED' "$BAK/$label.out" | sed "s/^/FAILED lines ($label): /"
}

# 2) baseline = HEAD versions of exactly the files this batch touched
for f in "${FILES[@]}"; do
  git show "HEAD:$f" > "$REPO/$f"
done
run_suite baseline

# 3) after = batch-3 versions
restore_batch3
run_suite after

echo
echo "--- NEW failures introduced by batch-3 (in after, not in baseline) ---"
comm -13 "$BAK/baseline.nodes" "$BAK/after.nodes"
echo "--- failures FIXED by batch-3 (in baseline, not in after) ---"
comm -23 "$BAK/baseline.nodes" "$BAK/after.nodes"
echo
echo "--- summary lines ---"
tail -1 "$BAK/baseline.out"
tail -1 "$BAK/after.out"
echo
echo "--- sha256 verification (must equal the snapshot above) ---"
for f in "${FILES[@]}"; do
  shasum -a 256 "$REPO/$f" | awk '{print $1"  '"$f"'"}'
done
echo "artifacts kept in $BAK"
