#!/usr/bin/env bash
set -u
LOG1=/home/shw/quant_projects/data/cogalpha_lqtp_production/reports/weekly_flip_retest.stdout.log
LOG2=/home/shw/quant_projects/data/cogalpha_lqtp_production/reports/weekly_flip_retest_extra.stdout.log
OUT=/home/shw/quant_projects/data/cogalpha_lqtp_production/reports/weekly_flip_finish.log
ROOT=/home/shw/quant_projects
cd "$ROOT"
echo "$(date -Is) finish_watcher_start" >> "$OUT"
for i in $(seq 1 500); do
  if grep -q '^DONE ok=' "$LOG1" 2>/dev/null && grep -q '^DONE ok=' "$LOG2" 2>/dev/null; then
    echo "$(date -Is) both DONE" >> "$OUT"
    break
  fi
  echo "$(date -Is) wait i=$i" >> "$OUT"
  sleep 60
done

NAMES=$(python3 - <<'PY'
import json
from pathlib import Path
note=json.loads(Path('/home/shw/quant_projects/data/cogalpha_lqtp_production/reports/weekly_formula_flip_note.json').read_text())
print(' '.join(note['names']))
PY
)

echo "$(date -Is) full_pack charts for flipped names" >> "$OUT"
python3 scripts/cogalpha_lqtp/complete_weekly_dug_full.py --skip-topk --min-avail-gb 3.5 --only $NAMES \
  >> /home/shw/quant_projects/data/cogalpha_lqtp_production/reports/weekly_flip_fullpack.stdout.log 2>&1
echo "$(date -Is) full_pack_exit=$?" >> "$OUT"

python3 scripts/cogalpha_lqtp/patch_weekly_factor_explanations.py >> "$OUT" 2>&1
echo "$(date -Is) FINISH_OK" >> "$OUT"
