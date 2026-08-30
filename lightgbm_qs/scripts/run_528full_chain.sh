#!/usr/bin/env bash
# #29 528full: full combination rerun with DUAL GATES.
# Same verified chain as run_818full_chain.sh, ONLY difference:
#   - manifest step adds --min-coverage 0.6 --require-history-days 30
#   - 528full-specific output paths (manifest/preds/weights) so the production
#     manifest (walkforward_selection_flip.json) and the three baseline runs
#     (2000fac/818fac_v1/818full_v1) are NOT touched.
# merge -> build_features -> preprocess -> manifest(gated) -> train 4 slices
#   -> manual slice merge -> vectorbt_qs backtest.
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

PROG=/tmp/rerun528_progress.log
TS() { date +%H:%M:%S; }
log() { echo "[$(TS)] $*" | tee -a "$PROG"; }

log "===== 528full 全量重跑开始 (dual gates) ====="

log "STEP 1: merge (10 pools, NO_DEDUP) -> selected_factors_flipped.csv"
rm -rf /tmp/rerun528_flip_partial
mkdir -p /tmp/rerun528_flip_partial
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  $PY scripts/merge_flip_chunk.py $i 8 /tmp/rerun528_flip_partial/chunk_$i.csv \
      > /tmp/rerun528_flip_chunk_$i.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
log "  8 chunks done -> combine"
$PY scripts/merge_flip_combine.py /tmp/rerun528_flip_partial
log "  merge done"
N_SEL=$(grep -c "" data/build/selected_factors_flipped.csv 2>/dev/null || echo 0)
log "  selected rows=$N_SEL (incl header)"
grep -c "new026:" data/build/selected_factors_flipped.csv || true

log "STEP 2: build_features_flip (全量特征矩阵, incl new026)"
$PY scripts/build_features_flip.py
log "  features done"

log "STEP 3: preprocess_features_flip"
$PY scripts/preprocess_features_flip.py
log "  preprocess done"

log "STEP 4a: build_walkforward_manifest_flip WITH GATES -> 528full manifest"
$PY scripts/build_walkforward_manifest_flip.py --min-coverage 0.6 --require-history-days 30 \
    --out data/build/walkforward_selection_528full.json
log "  manifest (gated) done"

log "STEP 4b: rolling train 4 slices (RE_OPT_EVERY=2, real Optuna)"
rm -f data/build/preds_528full_slice_*.parquet data/build/predictions_528full.parquet
PIDS=()
for sl in 0 1 2 3; do
  FEATURES_PARQUET=data/build/features_full3_adj_prep.parquet \
  SELECTION_MANIFEST=data/build/walkforward_selection_528full.json \
  BEST_JSON_DIR=data/build/best_params_flip \
  SLICE_PAT=data/build/preds_528full_slice_%d.parquet \
  nohup $PQ scripts/train_opt_adj_flip.py $sl > /tmp/rerun528_train_slice_$sl.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
log "  4 slices done -> merge slices (manual, train merge glob is hardcoded)"
$PQ -c "
import pandas as pd, glob
parts=[pd.read_parquet(p) for p in sorted(glob.glob('data/build/preds_528full_slice_*.parquet'))]
pred=pd.concat(parts, ignore_index=True)
pred.to_parquet('data/build/predictions_528full.parquet')
print('merged', len(pred), pred.date.min(), pred.date.max())
"
log "  predictions_528full.parquet 就绪"

log "STEP 5: vectorbt_qs backtest"
mkdir -p outputs/run_20260830_528full_v1
$PQ scripts/backtest_final_vectorbt.py --topk 30 --maxw 0.05 --turnover 0.30 \
    --mode accurate --benchmark 000300.SH \
    --pred data/build/predictions_528full.parquet \
    --weights-out /home/sunhaiwei/quant_projects/lightgbm_qs/outputs/run_20260830_528full_v1/weights_final_528full.parquet
# copy shared backtest outputs into the 528full run dir
cp -f outputs/backtest_final_nav.parquet outputs/run_20260830_528full_v1/backtest_final_nav_528full.parquet 2>/dev/null || true
cp -f outputs/backtest_final_nav_curves.parquet outputs/run_20260830_528full_v1/backtest_final_nav_curves_528full.parquet 2>/dev/null || true
cp -f outputs/backtest_final_equity.png outputs/run_20260830_528full_v1/backtest_final_equity_528full.png 2>/dev/null || true
log "===== 528full ALL DONE ====="
