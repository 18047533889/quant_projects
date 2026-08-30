#!/usr/bin/env bash
# #23 stage 2+3+4a: feature build (with new026 fix) + preprocess + manifest.
# Env-prefix form for all vars. Runs to manifest DONE.
set -uo pipefail
cd /home/sunhaiwei/quant_projects/lightgbm_qs
export PY=/home/sunhaiwei/quant_projects/.venv/bin/python
export OMP_NUM_THREADS=31
export MKL_NUM_THREADS=31
PROG=/tmp/rerun_progress.log
TS() { date +%H:%M:%S; }
log() { echo "[$(TS)] $*" | tee -a "$PROG"; }

log "STAGE2+3+4a (with new026 fix in build_features_flip)"
$PY scripts/build_features_flip.py
log "  features done (incl new026 cols)"
$PY scripts/preprocess_features_flip.py
log "  preprocess done"
$PY scripts/build_walkforward_manifest_flip.py
log "  manifest done"
