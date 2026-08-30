#!/bin/bash
# Resume full-market LQTP back adjust factor recompute.
# Idempotent: the underlying rebuild_all_factors_from_new_cos.py skips files
# whose parquet has expected (3127) columns and mtime >= cut-off, so re-running
# after a crash only finishes the gaps.
cd /home/sunhaiwei/quant_projects
export ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data
export DATA_ACCESS_SKIP_COS_MIRROR=1
export DATA_ACCESS_RUN_MODE=interactive_research
nohup /home/sunhaiwei/quant_projects/.venv/bin/python \
  lightgbm_qs/scripts/rebuild_fullmarket_dispatch.py \
  --procs 8 --shreds 6 --workers 1 --norm-force \
  --start 2016-01-01 --end 2026-08-21 \
  > /tmp/rebuild_fullmarket_dispatch_resume.log 2>&1 &
echo "resume dispatch PID $!"
