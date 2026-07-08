#!/usr/bin/env bash
# 推送 quant_projects 到 GitHub（读取 .env 中的 GITHUB_TOKEN）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.env"
fi

: "${GITHUB_TOKEN:?请设置 GITHUB_TOKEN（可在 $ROOT/.env 中配置）}"

BRANCH="${1:-$(git branch --show-current)}"
REMOTE_URL="https://${GITHUB_TOKEN}@github.com/18047533889/quant_projects.git"

echo "Pushing ${BRANCH} -> origin (github.com/18047533889/quant_projects.git)"
GIT_TERMINAL_PROMPT=0 git push "$REMOTE_URL" "$BRANCH"
