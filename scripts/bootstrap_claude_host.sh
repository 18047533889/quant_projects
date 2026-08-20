#!/usr/bin/env bash
# Point this machine's Claude auto-load at the repo contract. No secrets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${HOME}/.claude/archives"

cat > "${HOME}/CLAUDE.md" <<EOF
# Local Claude pointer (generated)

Repo contract (always prefer this over leftover home notes):
- ${ROOT}/CLAUDE.md
- ${ROOT}/loop/care.md
- Loop: ${ROOT}/loop/README.md

No plan mode unless asked. Working tree is truth. No git checkout/restore/stash/clean/reset.
Never load ${ROOT}/archives/ or docs/R2_HISTORY_ARCHIVE.md wholesale.
If auto-compact / scheduled loop goes crazy: bash ${ROOT}/scripts/cleanup_claude_bloat.sh then /clear.
EOF

MEM="${HOME}/.claude/projects/-home-shw/memory"
if [ -d "${HOME}/.claude/projects" ]; then
  mkdir -p "$MEM"
  cat > "${MEM}/MEMORY.md" <<EOF
# Memory index (compact — auto-loaded)

Standing rules live in the **repo** so they survive a new server:

- ${ROOT}/loop/care.md
- ${ROOT}/CLAUDE.md
- Loop: ${ROOT}/loop/README.md

Never load archives/ or memory/archives/* wholesale.
EOF
fi

printf '%s\n' '{"tasks":[]}' > "${HOME}/.claude/scheduled_tasks.json"
rm -f "${HOME}/.claude/scheduled_tasks.lock"

echo "Wrote ${HOME}/CLAUDE.md → ${ROOT}"
echo "Copy .env yourself. Model/API keys stay in ~/.claude/settings.json (not in git)."
echo "Paste loop/start_platform.md or loop/start_fe.md after /clear."
