#!/usr/bin/env bash
# Fill charts for EXACTLY one missing weekly factor, then exit.
# Safe for ~25–32G / no-swap: half-year FE chunks in child processes.
set -euo pipefail
ROOT=/home/shw/quant_projects
WORK=$ROOT/data/cogalpha_lqtp_production
LOG=$WORK/reports/weekly_dug_charts_onebyone.log
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

# never leave a batch runner around
pkill -f 'run_fill_charts_lowmem' 2>/dev/null || true

avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
echo "$(date -Is) gate avail=${avail}G need>=${MIN_AVAIL_GB}G" | tee -a "$LOG"
awk -v a="$avail" -v n="$MIN_AVAIL_GB" 'BEGIN{exit !(a>=n)}' || {
  echo "$(date -Is) SKIP_LOW_MEM" | tee -a "$LOG"
  exit 8
}

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
        print(dn)
        break
else:
    print('')
PY
)

if [[ -z "$NAME" ]]; then
  echo "$(date -Is) ALL_DONE" | tee -a "$LOG"
  exit 0
fi

echo "$(date -Is) ONE $NAME" | tee -a "$LOG"
set +e
python3 scripts/cogalpha_lqtp/fill_one_weekly_chart.py \
  --name "$NAME" \
  --skip-cache \
  --min-avail-gb "$MIN_AVAIL_GB" \
  >>"$LOG" 2>&1
rc=$?
set -e
avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
if [[ $rc -eq 0 ]]; then
  echo "$(date -Is) OK $NAME avail=${avail}G" | tee -a "$LOG"
else
  echo "$(date -Is) FAIL rc=$rc $NAME avail=${avail}G" | tee -a "$LOG"
fi
exit $rc
