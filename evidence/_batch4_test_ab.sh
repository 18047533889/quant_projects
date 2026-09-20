#!/usr/bin/env bash
# Batch 4 regression A/B: temporarily restore the 18 touched files to HEAD, run
# the parity/contract suite, put the batch-4 versions back, run again, diff by
# pytest NODE ID.
set -uo pipefail

REPO=/home/sunhaiwei/quant_projects
cd "$REPO" || exit 1

FILES=(
  "factor_engine/cleaned_operators/price_volume/polars_liquidity_v2.py"
  "factor_engine/cleaned_operators/technical/polars_tech_misc.py"
  "factor_engine/cleaned_operators/fundamental/polars_fundamental.py"
  "factor_engine/cleaned_operators/intraday/polars_next_stage.py"
  "factor_engine/cleaned_operators/intraday/polars_intraday_full.py"
  "factor_engine/cleaned_operators/common/polars_limit_misc.py"
  "factor_engine/cleaned_operators/fundamental/polars_fiscal_v2.py"
  "factor_engine/cleaned_operators/price_volume/polars_candle.py"
  "factor_engine/cleaned_operators/technical/polars_indicators_v2.py"
  "factor_engine/cleaned_operators/technical/polars_misc_v2.py"
  "factor_engine/cleaned_operators/microstructure/polars_flow_impact.py"
  "factor_engine/cleaned_operators/polars_chip_tail.py"
  "factor_engine/cleaned_operators/common/polars_ops.py"
  "factor_engine/cleaned_operators/common/polars_math_extended.py"
  "factor_engine/cleaned_operators/common/polars_extended.py"
  "factor_engine/cleaned_operators/common/polars_auto.py"
  "factor_engine/cleaned_operators/index_listing/polars_ops_v2.py"
  "factor_engine/cleaned_operators/overhaul/base.py"
)

BAK=/tmp/b4_backup.$$
mkdir -p "$BAK"

restore_batch4() {
  for f in "${FILES[@]}"; do
    if [ -f "$BAK/$(echo "$f" | tr '/' '_')" ]; then
      cp "$BAK/$(echo "$f" | tr '/' '_')" "$REPO/$f"
    fi
  done
  echo "== batch-4 versions restored =="
}
trap restore_batch4 EXIT

for f in "${FILES[@]}"; do
  cp "$REPO/$f" "$BAK/$(echo "$f" | tr '/' '_')"
done
echo "== batch-4 snapshot taken, sha256 =="
for f in "${FILES[@]}"; do
  shasum -a 256 "$REPO/$f" | awk '{print $1"  '"$f"'"}'
done

run_suite() {
  local label="$1"
  echo "== running suite: $label  ($(date -Is)) =="
  PYTHONPATH="$REPO" \
    python3 -m pytest factor_engine/tests/backend_parity factor_engine/tests/operator_contracts \
    -q -p no:cacheprovider --tb=no --continue-on-collection-errors -rfE \
    > "$BAK/$label.out" 2>&1
  tail -3 "$BAK/$label.out"
  grep -E '^(FAILED|ERROR) ' "$BAK/$label.out" \
    | sed 's/^FAILED //; s/^ERROR //' | awk '{print $1}' | sort -u > "$BAK/$label.nodes"
  grep -c . "$BAK/$label.nodes" | sed "s/^/unique failing\/erroring nodes ($label): /"
}

for f in "${FILES[@]}"; do
  git show "HEAD:$f" > "$REPO/$f"
done
run_suite baseline

restore_batch4
run_suite after

echo
echo "--- NEW failures introduced by batch-4 (in after, not in baseline) ---"
comm -13 "$BAK/baseline.nodes" "$BAK/after.nodes"
echo "--- failures FIXED by batch-4 (in baseline, not in after) ---"
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
