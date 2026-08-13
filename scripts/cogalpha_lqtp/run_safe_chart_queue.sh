#!/usr/bin/env bash
# Safe chart fill queue (no parallel FE):
#   Phase1: COS cache melt only (no FactorEngine) — up to MAX_N
#   Phase2: no-cs-rank DSL year chunks (FE child per year) — only if PHASE2=1
# Never runs cross-sectional rank() FE unless ALLOW_CS_RANK=1 (not recommended yet).
set -uo pipefail
ROOT=/home/shw/quant_projects
WORK=$ROOT/data/cogalpha_lqtp_production
LOG=$WORK/reports/weekly_dug_charts_safe.log
MAX_N=${MAX_N:-3}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-18}
PHASE2=${PHASE2:-0}
cd "$ROOT"
set -a; source .env; set +a
export PYTHONPATH="$ROOT:$ROOT/scripts/cogalpha_lqtp:$ROOT/ashare_lqtp_kit/protos:${PYTHONPATH:-}"
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

avail_gb() { awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo; }

# refresh class lists
python3 - <<'PY'
import json,re
from pathlib import Path
work=Path('/home/shw/quant_projects/data/cogalpha_lqtp_production')
w=json.loads((work/'reports/weekly_dug_neutral_rankic.json').read_text())
cache=work/'candidate_pool_neutral_cache'
def has(t):
    return ('data:image/png;base64,' in t and 'alt="Daily RankIC"' in t and
            ('Decile cumulative' in t or 'Group PnL' in t) and
            ('Cumulative long-short' in t or 'Cumulative LS' in t))
def cs_rank(d):
    d=re.sub(r'ts_rank\s*\(','',str(d).lower())
    return bool(re.search(r'(?<![a-z_])rank\s*\(', d))
def has_cache(dn):
    h=dn.split('_')[-1]
    if not cache.exists(): return False
    for src in cache.iterdir():
        if not src.is_dir(): continue
        for d in src.iterdir():
            if d.is_dir() and h in d.name and any(d.glob('*.parquet')): return True
    return False
weak=set()
wp=Path('/tmp/safe_cache_weak.txt')
if wp.exists():
    weak={x.strip() for x in wp.read_text().splitlines() if x.strip()}
g={'cache':[],'dsl_nocs':[],'dsl_cs':[]}
for r in w['selected']:
    dn=str(r.get('display_name') or r.get('factor_id'))
    p=work/'reports_weekly_dug'/f'{dn}.html'
    if p.exists() and has(p.read_text(errors='ignore')): continue
    d=str(r.get('lqtp_formula') or r.get('dsl') or '')
    if dn in weak:
        # previously CACHE_WEAK — do not retry melt; route by DSL class later
        if cs_rank(d): g['dsl_cs'].append(dn)
        else: g['dsl_nocs'].append(dn)
    elif has_cache(dn): g['cache'].append(dn)
    elif cs_rank(d): g['dsl_cs'].append(dn)
    else: g['dsl_nocs'].append(dn)
for k,v in g.items():
    Path(f'/tmp/safe_{k}.txt').write_text('\n'.join(v)+('\n' if v else ''))
    print(f'{k}={len(v)}')
PY

echo "$(date -Is) SAFE_Q start max_n=$MAX_N avail=$(avail_gb)G" | tee -a "$LOG"
n=0
run_one() {
  local name=$1 mode=$2
  local avail; avail=$(avail_gb)
  if awk -v a="$avail" -v m="$MIN_AVAIL_GB" 'BEGIN{exit !(a<m)}'; then
    echo "$(date -Is) SKIP_LOW $name avail=$avail" | tee -a "$LOG"
    return 8
  fi
  echo "$(date -Is) RUN $mode $name avail=$avail" | tee -a "$LOG"
  set +e
  if [[ "$mode" == cache ]]; then
    python3 scripts/cogalpha_lqtp/fill_one_weekly_chart.py --name "$name" --cache-only --min-avail-gb "$MIN_AVAIL_GB" >>"$LOG" 2>&1
  else
    python3 scripts/cogalpha_lqtp/fill_one_weekly_chart.py --name "$name" --skip-cache --min-avail-gb "$MIN_AVAIL_GB" >>"$LOG" 2>&1
  fi
  local rc=$?
  set -e
  echo "$(date -Is) END $name rc=$rc avail=$(avail_gb)G" | tee -a "$LOG"
  if [[ "$mode" == cache && $rc -eq 3 ]]; then
    echo "$name" >> /tmp/safe_cache_weak.txt
  fi
  # emergency reclaim pause
  sleep 5
  return $rc
}

while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  [[ $n -ge $MAX_N ]] && break
  run_one "$name" cache || true
  n=$((n+1))
  if awk -v a="$(avail_gb)" 'BEGIN{exit !(a<12)}'; then
    echo "$(date -Is) STOP_LOW_AFTER avail=$(avail_gb)G" | tee -a "$LOG"
    exit 9
  fi
done < /tmp/safe_cache.txt

if [[ "$PHASE2" == "1" && $n -lt $MAX_N ]]; then
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    [[ $n -ge $MAX_N ]] && break
    run_one "$name" dsl || { echo "$(date -Is) STOP_ON_DSL_FAIL $name" | tee -a "$LOG"; exit 1; }
    n=$((n+1))
  done < /tmp/safe_dsl_nocs.txt
fi

echo "$(date -Is) SAFE_Q done n=$n avail=$(avail_gb)G" | tee -a "$LOG"
