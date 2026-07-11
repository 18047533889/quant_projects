#!/usr/bin/env bash
# 将 data_access / factor_engine 等公共代码同步进 Git（替代仅 git add -u 的定时同步）。
#
# 用法:
#   bash scripts/git_sync_public_code.sh --dry-run          # 预览将 add 的内容
#   bash scripts/git_sync_public_code.sh                    # add + 显示 status
#   bash scripts/git_sync_public_code.sh --commit "msg"     # add + commit
#   bash scripts/git_sync_public_code.sh --commit "msg" --push
#   bash scripts/git_sync_public_code.sh --push             # 仅 push 当前分支
#
# 环境变量:
#   GIT_SYNC_REMOTE   默认 origin
#   GIT_SYNC_BRANCH   默认当前分支
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

REMOTE="${GIT_SYNC_REMOTE:-origin}"
BRANCH="${GIT_SYNC_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
DRY_RUN=0
DO_COMMIT=0
DO_PUSH=0
COMMIT_MSG=""

# 白名单：只 add 公共基础设施，不扫 data/ logs/ workspace
SYNC_PATHS=(
  data_access
  factor_engine
  factor_layer
  factor-pool-standard
  gtja191
  week2_pv_factors
  raw_data_layer
  scripts
  docs
  configs
  .github
  infra
  manifests
  README.md
  STRUCTURE.md
  env.sh
  requirements.txt
  requirements-dev.txt
  pytest.ini
  .gitignore
  .data_access_allowlist.yaml
  .pre-commit-config.yaml
)

usage() {
  sed -n '2,12p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --commit)
      DO_COMMIT=1
      COMMIT_MSG="${2:-}"
      [[ -n "${COMMIT_MSG}" ]] || usage
      shift 2
      ;;
    --push) DO_PUSH=1; shift ;;
    -h|--help) usage ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done

log() { echo "==> $*"; }

add_whitelist() {
  local existing=()
  for p in "${SYNC_PATHS[@]}"; do
    if [[ -e "${p}" || -d "${p}" ]]; then
      existing+=("${p}")
    fi
  done
  if [[ "${#existing[@]}" -eq 0 ]]; then
    echo "❌ 白名单路径均不存在，请确认在 quant_projects 根目录执行" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[dry-run] git add -A -- ${existing[*]}"
    git status --short -- "${existing[@]}" | head -100 || true
    return 0
  fi
  git add -A -- "${existing[@]}"
}

log "仓库: ${ROOT}"
log "远程: ${REMOTE}  分支: ${BRANCH}"

if [[ "${DO_PUSH}" -eq 1 && "${DO_COMMIT}" -eq 0 ]]; then
  log "推送到 ${REMOTE}/${BRANCH} ..."
  git push -u "${REMOTE}" "${BRANCH}"
  exit 0
fi

bash scripts/verify_repo_tracking.sh || true

log "按白名单 git add（含新文件，不仅 git add -u）..."
add_whitelist

echo ""
log "暂存区变更预览:"
git diff --cached --stat | tail -30 || true
echo ""
git status --short | head -80

if [[ "${DO_COMMIT}" -eq 0 ]]; then
  echo ""
  echo "下一步: bash scripts/git_sync_public_code.sh --commit \"你的提交说明\" [--push]"
  exit 0
fi

if git diff --cached --quiet; then
  log "暂存区无变更，跳过 commit"
else
  git commit -m "${COMMIT_MSG}"
  log "已提交: $(git log -1 --oneline)"
fi

if [[ "${DO_PUSH}" -eq 1 ]]; then
  log "推送到 ${REMOTE}/${BRANCH} ..."
  git push -u "${REMOTE}" "${BRANCH}"
  log "GitHub 可见 commit: $(git rev-parse HEAD)"
fi
