#!/usr/bin/env bash
# 修复阶段：先"仅重评"（coverage，修复判定规则后不需要重落值），再重落值（resource/missing/dsl_fix）。
# fail-fast：任一步失败写 repair.status=failed 并退出。
set -u
ROOT=/home/sunhaiwei/quant_projects
WORK=$ROOT/work/newmining_20260919
PY=$ROOT/.venv/bin/python
LOG=$WORK/repair.log
STATUS=$WORK/repair.status

cd "$ROOT" || exit 1
echo "running" > "$STATUS"
echo "[$(date '+%F %T')] === repair start (pid $$) ===" >> "$LOG"

"$PY" jobs/new_mining_intake.py --repair-all --repair-rounds 3 \
  --batch-size 8 --threads 6 --root-workers 8 --window-years 3 \
  --memory-gib 40 --host-mem-floor-gib 6 --wave-time-limit-s 21600 \
  >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] repair exit=$rc" >> "$LOG"
if [ $rc -ne 0 ]; then
  echo "failed" > "$STATUS"
  exit $rc
fi
echo "done" > "$STATUS"
echo "[$(date '+%F %T')] === repair done ===" >> "$LOG"
