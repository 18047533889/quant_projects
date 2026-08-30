#!/usr/bin/env bash
# #23 full rerun: 34 new factors merged into the full 10-pool chain (NO_DEDUP),
# then feature build -> preprocess -> walk-forward manifest -> rolling train
# (RE_OPT_EVERY=2, real Optuna refit) -> vectorbt_qs backtest.
# Baseline run_20260830_2000fac / run_20260830_818fac_v1 products are NOT touched;
# this run writes only data/build/*_818full* + preds_818full_slice_* +
# predictions_818full.parquet + outputs/run_20260830_818full_v1/.
set -uo pipefail
cd /home/sunhaiwei/quant_projects/lightgbm_qs
export PY=/home/sunhaiwei/quant_projects/.venv/bin/python
export PQ=/srv/quant/envs/quantaalpha/bin/python
export ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data
export DATA_ACCESS_COS_READ_MODE=mirror
export QUANT_SCHEMA_CHECK=warn
export NO_DEDUP=1
export OMP_NUM_THREADS=31
export MKL_NUM_THREADS=31

PROG=/tmp/rerun_progress.log
TS() { date +%H:%M:%S; }
log() { echo "[$(TS)] $*" | tee -a "$PROG"; }

log "===== #23 全量重跑开始 818full_v1 ====="

log "STEP 1: merge (10 pools, NO_DEDUP) + dedup(0) -> selected_factors_flipped.csv"
rm -rf /tmp/rerun_flip_partial
mkdir -p /tmp/rerun_flip_partial
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  $PY scripts/merge_flip_chunk.py $i 8 /tmp/rerun_flip_partial/chunk_$i.csv \
      > /tmp/rerun_flip_chunk_$i.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
log "  8 chunks done -> combine"
$PY scripts/merge_flip_combine.py /tmp/rerun_flip_partial
log "  merge done"
N_SEL=$(grep -c "" data/build/selected_factors_flipped.csv 2>/dev/null || echo 0)
log "  selected rows=$N_SEL (incl header)"
grep -c "new026:" data/build/selected_factors_flipped.csv || true

log "STEP 2: build_features_flip (全量特征矩阵)"
$PY scripts/build_features_flip.py
log "  features done"

log "STEP 3: preprocess_features_flip"
$PY scripts/preprocess_features_flip.py
log "  preprocess done"

log "STEP 4a: build_walkforward_manifest_flip"
$PY scripts/build_walkforward_manifest_flip.py
log "  manifest done"

log "STEP 4b: rolling train 4 slices (RE_OPT_EVERY=2, real Optuna)"
rm -f data/build/preds_818full_slice_*.parquet data/build/predictions_818full.parquet
PIDS=()
for sl in 0 1 2 3; do
  nohup $PQ scripts/train_opt_adj_flip.py $sl \
      FEATURES_PARQUET=data/build/features_full3_adj_prep.parquet \
      SELECTION_MANIFEST=data/build/walkforward_selection_flip.json \
      BEST_JSON_DIR=data/build/best_params_flip \
      SLICE_PAT=data/build/preds_818full_slice_%d.parquet \
      > /tmp/rerun_train_slice_$sl.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
# nohup + env vars: env vars were placed AFTER the command -> they were passed to
# train_opt_adj_flip.py as argv, not env. We must redo with correct env form.
log "  WARN: env 形式有误，改用 env 前缀重跑 4 slices"
rm -f data/build/preds_818full_slice_*.parquet
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
log "  4 slices done -> merge slices"
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
log "===== 818full_v1 ALL DONE ====="
