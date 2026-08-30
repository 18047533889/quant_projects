#!/usr/bin/env bash
# Orchestrate the full lightgbm_qs chain AFTER the LQTP full-market rebuild landed:
#   Step 1 (rank_ic + flip + dedup)  -> merge_flip_chunk.py x8 + merge_flip_combine.py
#   Step 2 (feature matrix)          -> build_features_flip.py
#   Step 3 (preprocess)              -> preprocess_features_flip.py
#   Step 4a (walk-forward manifest)  -> build_walkforward_manifest_flip.py
#   Step 4b (train + 12mo re-opt)    -> train_opt_adj_flip.py <0..3>
#   Step 5 (vectorbt_qs backtest)    -> backtest_final_vectorbt.py
#
# Run with the lightgbm_qs venv for steps 1-4 and the quantaalpha env for step 5.
set -euo pipefail
cd /home/sunhaiwei/quant_projects/lightgbm_qs
export PY=/home/sunhaiwei/quant_projects/.venv/bin/python
export PQ=/srv/quant/envs/quantaalpha/bin/python
export ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data
export DATA_ACCESS_COS_READ_MODE=mirror
export QUANT_SCHEMA_CHECK=warn

TS() { date +%H:%M:%S; }

echo "[$(TS)] STEP 1a: parallel rank_ic+flip (8 chunks)"
rm -f /tmp/flip_partial/chunk_*.csv
mkdir -p /tmp/flip_partial
for i in 0 1 2 3 4 5 6 7; do
  nohup $PY scripts/merge_flip_chunk.py $i 8 /tmp/flip_partial/chunk_$i.csv \
      > /tmp/flip_chunk_$i.log 2>&1 &
done
# poll until all 8 chunk CSVs appear
while :; do
  n=$(ls /tmp/flip_partial/chunk_*.csv 2>/dev/null | wc -l)
  if [ "$n" -ge 8 ]; then break; fi
  sleep 20
done
echo "[$(TS)] STEP 1b: combine + dedup"
$PY scripts/merge_flip_combine.py /tmp/flip_partial

echo "[$(TS)] STEP 2: feature matrix"
$PY scripts/build_features_flip.py

echo "[$(TS)] STEP 3: preprocess"
$PY scripts/preprocess_features_flip.py

echo "[$(TS)] STEP 4a: walk-forward selection manifest"
$PY scripts/build_walkforward_manifest_flip.py

echo "[$(TS)] STEP 4b: rolling train with 12-month re-opt (quantaalpha env: lightgbm+optuna)"
for sl in 0 1 2 3; do
  nohup $PQ scripts/train_opt_adj_flip.py $sl > /tmp/train_flip_slice_$sl.log 2>&1 &
done
# poll until 4 slices done (preds_adj_flip_slice_*.parquet all present)
while :; do
  n=$(ls data/build/preds_adj_flip_slice_*.parquet 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then break; fi
  sleep 30
done
$PQ scripts/train_opt_adj_flip.py   # merge slices

echo "[$(TS)] STEP 5: vectorbt_qs final backtest"
$PQ scripts/backtest_final_vectorbt.py --topk 30 --maxw 0.05 --turnover 0.30 --mode accurate

echo "[$(TS)] ALL DONE"
