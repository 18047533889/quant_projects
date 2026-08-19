#!/usr/bin/env bash
# Sync compact Claude skills from working tree into stale agent worktrees.
set -euo pipefail
ROOT="${1:-/home/shw/quant_projects}"
SRC="$ROOT/factor_engine/.claude/skills/factor-engine-rules/SKILL.md"
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
count=0
while IFS= read -r dst; do
  cp "$SRC" "$dst"
  count=$((count + 1))
done < <(find "$ROOT/.claude/worktrees" -path '*/factor_engine/.claude/skills/factor-engine-rules/SKILL.md' 2>/dev/null)
echo "synced factor-engine-rules -> $count worktrees"
