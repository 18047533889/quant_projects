#!/usr/bin/env bash
# One-factor-per-subprocess chart fill — child exits to release FE/DuckDB RAM.
set -euo pipefail
ROOT=/home/shw/quant_projects
WORK="$ROOT/data/cogalpha_lqtp_production"
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

MISS_FILE=${1:-/tmp/weekly_chart_miss.txt}
LOG="$WORK/reports/weekly_dug_charts_fill_safe.log"
MIN_AVAIL_GB=${MIN_AVAIL_GB:-14}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1200}

wait_mem() {
  local need=$1
  local waited=0
  while true; do
    local avail
    avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
    if awk -v a="$avail" -v n="$need" 'BEGIN{exit !(a>=n)}'; then
      echo "mem_ok avail=${avail}G need>=${need}G" | tee -a "$LOG"
      return 0
    fi
    echo "low_mem avail=${avail}G need>=${need}G wait..." | tee -a "$LOG"
    sleep 8
    waited=$((waited+8))
    if [[ $waited -ge $WAIT_TIMEOUT ]]; then
      echo "mem_timeout avail=${avail}G" | tee -a "$LOG"
      return 1
    fi
  done
}

# refresh miss list
python3 - <<'PY'
from pathlib import Path
import json
work=Path('/home/shw/quant_projects/data/cogalpha_lqtp_production')
w=json.loads((work/'reports/weekly_dug_neutral_rankic.json').read_text())
def has(t):
    return ('data:image/png;base64,' in t and 'alt="Daily RankIC"' in t and
            ('Decile cumulative' in t or 'Group PnL' in t) and
            ('Cumulative long-short' in t or 'Cumulative LS' in t))
miss=[]
for r in w['selected']:
    dn=r.get('display_name') or r.get('factor_id')
    p=work/'reports_weekly_dug'/f'{dn}.html'
    if not p.exists() or not has(p.read_text(errors='ignore')):
        miss.append(dn)
Path('/tmp/weekly_chart_miss.txt').write_text('\n'.join(miss))
print(f'refresh_miss n={len(miss)}')
PY

mapfile -t NAMES < "$MISS_FILE"
echo "safe_fill start n=${#NAMES[@]} $(date -Is)" | tee -a "$LOG"
ok=0
fail=0
i=0
for name in "${NAMES[@]}"; do
  [[ -z "$name" ]] && continue
  i=$((i+1))
  echo "[$i/${#NAMES[@]}] $name $(date -Is)" | tee -a "$LOG"
  wait_mem "$MIN_AVAIL_GB" || { echo "ABORT mem"; exit 2; }
  # skip extended extras; one name; force rebuild this one
  if python3 scripts/cogalpha_lqtp/fill_weekly_dug_charts.py \
      --force --no-extended --min-mem-gb 8 \
      --only "$name" >>"$LOG" 2>&1; then
    ok=$((ok+1))
    echo "  CHILD_OK $name" | tee -a "$LOG"
  else
    fail=$((fail+1))
    echo "  CHILD_FAIL $name" | tee -a "$LOG"
  fi
  # give kernel a moment to reclaim after child exit
  sleep 2
  sync || true
  python3 - <<'PY'
import gc
gc.collect()
try:
    from data_access import reset_store
    reset_store()
except Exception:
    pass
PY
done
echo "safe_fill done ok=$ok fail=$fail $(date -Is)" | tee -a "$LOG"
# final index patch already done per child; refresh miss count
python3 - <<'PY'
from pathlib import Path
import json
work=Path('/home/shw/quant_projects/data/cogalpha_lqtp_production')
w=json.loads((work/'reports/weekly_dug_neutral_rankic.json').read_text())
def has(t):
    return ('data:image/png;base64,' in t and 'alt="Daily RankIC"' in t and
            ('Decile cumulative' in t or 'Group PnL' in t) and
            ('Cumulative long-short' in t or 'Cumulative LS' in t))
ok=sum(1 for r in w['selected'] if (work/'reports_weekly_dug'/f"{r.get('display_name') or r.get('factor_id')}.html").exists() and has((work/'reports_weekly_dug'/f"{r.get('display_name') or r.get('factor_id')}.html").read_text(errors='ignore')))
print('final charts_ok', ok, 'of', len(w['selected']))
PY
