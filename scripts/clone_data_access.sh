#!/usr/bin/env bash
# 从私密仓库拉取 data_access（推荐改用 sync_quantsociety_backend.sh）
#
# 用法:
#   bash scripts/clone_data_access.sh
#   bash scripts/clone_data_access.sh --ssh
#   bash scripts/sync_quantsociety_backend.sh docs raw_data_layer factor_layer  # 全量推荐
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data_access"
REPO_SSH="git@github.com:18047533889/projects.git"
SPARSE_PATH="projects/quantsociety_backend_project/data_access"

GITHUB_TOKEN="${GITHUB_TOKEN:?请设置 GITHUB_TOKEN 环境变量（GitHub PAT）}"

use_ssh=false
if [[ "${1:-}" == "--ssh" ]]; then
  use_ssh=true
fi

if [[ -d "${DEST}" ]] && [[ -f "${DEST}/store.py" ]]; then
  echo "已存在: ${DEST}"
  exit 0
fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

if [[ "${use_ssh}" == true ]]; then
  REPO="${REPO_SSH}"
else
  REPO="https://${GITHUB_TOKEN}@github.com/18047533889/projects.git"
fi

echo "Sparse clone ${SPARSE_PATH} -> ${DEST}"
git clone --filter=blob:none --sparse "${REPO}" "${TMP}/repo"
git -C "${TMP}/repo" sparse-checkout set "${SPARSE_PATH}"
git -C "${TMP}/repo" checkout

rm -rf "${DEST}"
mv "${TMP}/repo/projects/quantsociety_backend_project/data_access" "${DEST}"

echo "完成: ${DEST}"
ls -la "${DEST}" | head -15
