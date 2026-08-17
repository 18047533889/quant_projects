#!/usr/bin/env bash
set -u
ROOT=/home/shw/quant_projects
WORK=$ROOT/data/cogalpha_lqtp_production
LOGDIR=$WORK/reports/flip_parallel
mkdir -p "$LOGDIR"
cd "$ROOT"

python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime
work = Path("/home/shw/quant_projects/data/cogalpha_lqtp_production")
names = json.loads((work / "reports/weekly_formula_flip_note.json").read_text())["names"]
pend = []
done = []
for n in names:
    lake = work / "factor_lake" / n / "values.parquet"
    ok = lake.exists() and datetime.fromtimestamp(lake.stat().st_mtime) >= datetime(2026, 8, 17, 1, 55)
    (done if ok else pend).append(n)
Path("/tmp/flip_pending.txt").write_text("\n".join(pend) + ("\n" if pend else ""))
print(f"done={len(done)} pend={len(pend)}")
print("PEND", " ".join(pend))
PY

export FACTOR_ENGINE_MAX_MEMORY_MB=7000
export FACTOR_ENGINE_RESERVE_GB=1
export DATA_ACCESS_SKIP_COS_MIRROR=1

echo "$(date -Is) parallel_start n=$(wc -l </tmp/flip_pending.txt)" | tee -a "$LOGDIR/master.log"

# 4 concurrent factor jobs
cat /tmp/flip_pending.txt | xargs -P 4 -I{} bash -c '
  name="$1"
  LOGDIR="'"$LOGDIR"'"
  ROOT="'"$ROOT"'"
  echo "$(date -Is) START $name" >> "$LOGDIR/master.log"
  python3 "$ROOT/scripts/cogalpha_lqtp/retest_weekly_recovered_dsl.py" \
    --only "$name" --min-avail-gb 2.0 \
    > "$LOGDIR/${name}.log" 2>&1
  ec=$?
  echo "$(date -Is) END $name ec=$ec" >> "$LOGDIR/master.log"
' _ {}

echo "$(date -Is) ALL_PARALLEL_DONE" | tee -a "$LOGDIR/master.log"

python3 - <<'PY'
import json, sys
from pathlib import Path
ROOT = Path("/home/shw/quant_projects")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]
from scripts.cogalpha_lqtp.complete_weekly_dug_full import _patch_index
work = ROOT / "data/cogalpha_lqtp_production"
w = json.loads((work / "reports/weekly_dug_neutral_rankic.json").read_text())
_patch_index(work, w)
print("homepage patched")
PY

python3 scripts/cogalpha_lqtp/patch_weekly_factor_explanations.py | tee -a "$LOGDIR/master.log"

NAMES=$(python3 -c "import json;print(' '.join(json.load(open('/home/shw/quant_projects/data/cogalpha_lqtp_production/reports/weekly_formula_flip_note.json'))['names']))")
python3 scripts/cogalpha_lqtp/complete_weekly_dug_full.py --skip-topk --min-avail-gb 2.0 --only $NAMES \
  >> "$LOGDIR/fullpack.log" 2>&1

echo "$(date -Is) FINISH_PARALLEL_OK" | tee -a "$LOGDIR/master.log"
