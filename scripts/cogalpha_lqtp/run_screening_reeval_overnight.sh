#!/usr/bin/env bash
# Overnight supervisor: keep screening re-eval running until all factors done.
# - Auto-resume on crash (checkpoint JSON)
# - LQTP token refresh every 25min (1500s) if LQTP creds are set
# - Regenerates index HTML when finished
set -u

ROOT="/home/shw/quant_projects"
WORK="$ROOT/data/cogalpha_lqtp_production"
LOG="$WORK/screening_reeval_overnight.log"
LOCK="$WORK/screening_reeval_overnight.pid"
TARGET="${TARGET_FACTOR_COUNT:-171}"
POLL_SEC="${POLL_SEC:-300}"

export PYTHONUNBUFFERED=1
export LQTP_TOKEN_REFRESH_SECONDS="${LQTP_TOKEN_REFRESH_SECONDS:-1500}"

cd "$ROOT" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

count_json() {
  local file="$1" key="$2"
  python3 - <<PY 2>/dev/null || echo 0
import json
from pathlib import Path
p=Path("$file")
if not p.exists():
    print(0)
else:
    print(len(json.loads(p.read_text()).get("$key",[])))
PY
}

batch_running() {
  pgrep -f "scripts.cogalpha_lqtp.run_screening_reeval_batch" >/dev/null 2>&1
}

start_batch() {
  nohup python3 -m scripts.cogalpha_lqtp.run_screening_reeval_batch \
    --workers 4 >> "$WORK/screening_reeval_batch.log" 2>&1 &
  log "started batch pid=$!"
}

log "overnight supervisor begin TARGET=$TARGET POLL=${POLL_SEC}s"

while true; do
  if [[ -f "$WORK/screening_reeval_DONE.json" ]]; then
    log "DONE file exists — exiting supervisor"
    break
  fi

  mat_done="$(count_json "$WORK/screening_reeval_materialize.json" completed)"
  eval_done="$(count_json "$WORK/screening_reeval_progress.json" completed)"
  mat_fail="$(count_json "$WORK/screening_reeval_materialize.json" failed)"
  eval_fail="$(python3 - <<PY 2>/dev/null || echo 0
import json
from pathlib import Path
p=Path("$WORK/screening_reeval_progress.json")
print(len(json.loads(p.read_text()).get("failed",{})) if p.exists() else 0)
PY
)"

  log "progress mat=${mat_done}/${TARGET} eval=${eval_done}/${TARGET} mat_fail=${mat_fail} eval_fail=${eval_fail}"

  if [[ "$mat_done" -ge "$TARGET" && "$eval_done" -ge "$TARGET" ]]; then
    log "all targets reached — rendering index"
    python3 -m scripts.cogalpha_lqtp.render_rankic_screening_index >> "$LOG" 2>&1
    python3 -m scripts.cogalpha_lqtp.run_screening_reeval_batch --eval-only --workers 4 >> "$LOG" 2>&1 || true
    log "supervisor finished"
    break
  fi

  if ! batch_running; then
    log "batch not running — launching/resuming"
    start_batch
  fi

  sleep "$POLL_SEC"
done
