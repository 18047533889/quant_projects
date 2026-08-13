#!/usr/bin/env bash
# Ultra-safe: process at most MAX_N factors, one subprocess each, half-year FE chunks.
# Abort between factors if MemAvailable < MIN_AVAIL_GB. Never parallelize.
set -uo pipefail
ROOT=/home/shw/quant_projects
WORK=$ROOT/data/cogalpha_lqtp_production
LOG=$WORK/reports/weekly_dug_charts_onebyone.log
MAX_N=${MAX_N:-3}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-18}
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source .env
set +a
export PYTHONPATH="$ROOT:$ROOT/scripts/cogalpha_lqtp:$ROOT/ashare_lqtp_kit/protos:${PYTHONPATH:-}"
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

avail_gb() { awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo; }

echo "$(date -Is) SAFE_START max_n=$MAX_N min_avail=${MIN_AVAIL_GB}G avail=$(avail_gb)G" | tee -a "$LOG"

done_n=0
while [[ $done_n -lt $MAX_N ]]; do
  avail=$(avail_gb)
  if awk -v a="$avail" -v n="$MIN_AVAIL_GB" 'BEGIN{exit !(a<n)}'; then
    echo "$(date -Is) WAIT_MEM avail=${avail}G need>=${MIN_AVAIL_GB}G" | tee -a "$LOG"
    sleep 15
    continue
  fi

  NAME=$(python3 - <<'PY'
from pathlib import Path
import json
work=Path('/home/shw/quant_projects/data/cogalpha_lqtp_production')
w=json.loads((work/'reports/weekly_dug_neutral_rankic.json').read_text())
def has(t):
    return ('data:image/png;base64,' in t and 'alt="Daily RankIC"' in t and
            ('Decile cumulative' in t or 'Group PnL' in t) and
            ('Cumulative long-short' in t or 'Cumulative LS' in t))
for r in w['selected']:
    dn=str(r.get('display_name') or r.get('factor_id'))
    p=work/'reports_weekly_dug'/f'{dn}.html'
    if not (p.exists() and has(p.read_text(errors='ignore'))):
        print(dn); break
PY
)
  if [[ -z "${NAME:-}" ]]; then
    echo "$(date -Is) ALL_DONE" | tee -a "$LOG"
    exit 0
  fi

  echo "$(date -Is) START [$((done_n+1))/$MAX_N] $NAME avail=$(avail_gb)G" | tee -a "$LOG"
  set +e
  python3 scripts/cogalpha_lqtp/fill_one_weekly_chart.py \
    --name "$NAME" \
    --skip-cache \
    --min-avail-gb "$MIN_AVAIL_GB" \
    >>"$LOG" 2>&1
  rc=$?
  set -e
  echo "$(date -Is) END $NAME rc=$rc avail=$(avail_gb)G" | tee -a "$LOG"
  done_n=$((done_n+1))

  # cool-down so OS can reclaim child pages
  sleep 8
  python3 -c 'import gc; gc.collect()' 2>/dev/null || true

  if [[ $rc -ne 0 ]]; then
    echo "$(date -Is) STOP_ON_FAIL after $NAME" | tee -a "$LOG"
    exit $rc
  fi
done

echo "$(date -Is) SAFE_BATCH_DONE n=$done_n avail=$(avail_gb)G" | tee -a "$LOG"
