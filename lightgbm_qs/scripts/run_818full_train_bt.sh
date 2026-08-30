#!/usr/bin/env bash
# #23 train stage ONLY (4 slices + merge). The merge/features/prep/manifest stages
# are already done. This script runs the rolling training with REAL Optuna re-opt
# (RE_OPT_EVERY=2) writing to preds_818full_slice_*.parquet, then merges into
# predictions_818full.parquet. Uses env-prefix var passing (correct form).
set -uo pipefail
cd /home/sunhaiwei/quant_projects/lightgbm_qs
export PQ=/srv/quant/envs/quantaalpha/bin/python
export ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data
export DATA_ACCESS_COS_READ_MODE=mirror
export QUANT_SCHEMA_CHECK=warn
export OMP_NUM_THREADS=31
export MKL_NUM_THREADS=31

PROG=/tmp/rerun_progress.log
TS() { date +%H:%M:%S; }
log() { echo "[$(TS)] $*" | tee -a "$PROG"; }

log "STEP 4b(retry): 4 slices with correct env prefix"
rm -f data/build/preds_818full_slice_*.parquet data/build/predictions_818full.parquet
PIDS=()
for sl in 0 1 2 3; do
  FEATURES_PARQUET=data/build/features_full3_adj_prep.parquet \
  SELECTION_MANIFEST=data/build/walkforward_selection_flip.json \
  BEST_JSON_DIR=data/build/best_params_flip \
  SLICE_PAT=data/build/preds_818full_slice_%d.parquet \
  nohup $PQ scripts/train_opt_adj_flip.py $sl > /tmp/rerun_train_slice_$sl.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
log "  4 slices done -> merge"
FEATURES_PARQUET=data/build/features_full3_adj_prep.parquet \
SELECTION_MANIFEST=data/build/walkforward_selection_flip.json \
BEST_JSON_DIR=data/build/best_params_flip \
SLICE_PAT=data/build/preds_818full_slice_%d.parquet \
$PQ scripts/train_opt_adj_flip.py
log "  predictions_818full.parquet 就绪"

log "STEP 5: vectorbt_qs backtest"
$PQ scripts/backtest_final_vectorbt.py --topk 30 --maxw 0.05 --turnover 0.30 \
    --mode accurate --benchmark 000300.SH \
    --pred data/build/predictions_818full.parquet \
    --weights-out /home/sunhaiwei/quant_projects/lightgbm_qs/outputs/run_20260830_818full_v1/weights_final_818full.parquet
log "===== 818full_v1 train+backtest DONE ====="
