#!/usr/bin/env bash
# Stop context bloat: scheduled loops, stale todos, huge session logs, paste-cache.
set -euo pipefail
ARCH="${HOME}/.claude/archives"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ARCH/tasks" "$ARCH/sessions" "$ARCH/paste-cache" "$ARCH/tmp-claude"

echo "== 1) Disable scheduled loop tasks =="
printf '%s\n' '{"tasks":[]}' > "${HOME}/.claude/scheduled_tasks.json"
rm -f "${HOME}/.claude/scheduled_tasks.lock"
echo "scheduled_tasks cleared"

echo "== 2) Archive Claude Code todo JSON (TaskCreate bloat) =="
for d in "${HOME}/.claude/tasks"/*/; do
  [ -d "$d" ] || continue
  sid=$(basename "$d")
  cnt=$(find "$d" -maxdepth 1 -name '*.json' | wc -l)
  [ "$cnt" -eq 0 ] && continue
  dest="$ARCH/tasks/${TS}-${sid}"
  mkdir -p "$dest"
  mv "$d"*.json "$dest/" 2>/dev/null || true
  echo "  archived $cnt tasks for $sid"
done

echo "== 3) Archive old session transcripts (>50MB, idle >2m) =="
find "${HOME}/.claude/projects/-home-shw" -maxdepth 1 -name '*.jsonl' -size +50M 2>/dev/null | while read -r f; do
  age=$(( $(date +%s) - $(stat -c %Y "$f") ))
  if [ "$age" -gt 120 ]; then
    mv "$f" "$ARCH/sessions/"
    echo "  archived $(basename "$f")"
  else
    echo "  keep hot $(basename "$f")"
  fi
done

echo "== 4) Rotate paste-cache =="
if [ -d "${HOME}/.claude/paste-cache" ]; then
  mv "${HOME}/.claude/paste-cache" "$ARCH/paste-cache/${TS}" 2>/dev/null || true
  mkdir -p "${HOME}/.claude/paste-cache"
fi

echo "== 5) Trim old telemetry =="
find "${HOME}/.claude/telemetry" -name '*.json' -mtime +3 -delete 2>/dev/null || true

echo "== 6) Collapse todo UI =="
python3 - <<'PY'
import json
p = __import__("pathlib").Path.home() / ".claude.json"
if p.exists():
    d = json.loads(p.read_text())
    d["showExpandedTodos"] = False
    p.write_text(json.dumps(d, indent=2) + "\n")
PY

echo "Done. Restart Claude Code or run /clear in active sessions."
