#!/usr/bin/env bash
# 校验 data_access / factor_engine 等公共代码已被 Git 正式跟踪（非 git add -u 漏提）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

fail=0

require_tracked() {
  local path="$1"
  if ! git ls-files --error-unmatch "${path}" >/dev/null 2>&1; then
    echo "❌ 未跟踪（Git 仓库中不可见）: ${path}"
    fail=1
  fi
}

require_dir_tracked() {
  local dir="$1"
  local min="${2:-1}"
  local count
  count="$(git ls-files "${dir}" | wc -l | tr -d ' ')"
  if [[ "${count}" -lt "${min}" ]]; then
    echo "❌ 目录跟踪文件过少: ${dir}（仅 ${count} 个，期望 >= ${min}）"
    fail=1
  else
    echo "✓ ${dir}: ${count} 个文件已跟踪"
  fi
}

echo "==> 校验公共模块 Git 跟踪状态（${ROOT}）"

# 顶层目录必须存在且被跟踪
require_dir_tracked "data_access" 50
require_dir_tracked "factor_engine" 200

# P0/P1 关键文件（审计清单）
REQUIRED_FILES=(
  data_access/store.py
  data_access/read/read_contract.py
  data_access/read/scan_handle.py
  data_access/registry/params_validation.py
  data_access/write/publish_manifest.py
  data_access/read/key_policy.py
  data_access/cos/mirror.py
  data_access/config/datasets.yaml
  factor_engine/storage/data_scope.py
  factor_engine/storage/sources/data_access_source.py
  factor_engine/backend/polars_lazy.py
  factor_engine/runtime/lineage_service.py
  .data_access_allowlist.yaml
  scripts/check_data_access_allowlist.py
  .github/workflows/ci.yml
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ -e "${f}" ]]; then
    require_tracked "${f}"
  else
    echo "⚠ 本地缺失: ${f}"
    fail=1
  fi
done

echo ""
echo "==> 检查未跟踪的新文件（应在 git add 白名单内）"
untracked="$(git ls-files --others --exclude-standard data_access/ factor_engine/ .github/ scripts/ 2>/dev/null | head -50 || true)"
if [[ -n "${untracked}" ]]; then
  echo "⚠ 以下公共路径文件尚未 git add："
  echo "${untracked}"
  fail=1
else
  echo "✓ data_access / factor_engine / .github 无未跟踪新文件"
fi

echo ""
echo "==> 检查是否被 .gitignore 误伤"
for probe in data_access/store.py factor_engine/runtime/engine.py; do
  if [[ -e "${probe}" ]]; then
    ignored="$(git check-ignore -v "${probe}" 2>/dev/null || true)"
    if [[ -n "${ignored}" ]]; then
      echo "❌ 被 gitignore 忽略: ${probe}"
      echo "   ${ignored}"
      fail=1
    fi
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo ""
  echo "修复：bash scripts/git_sync_public_code.sh --dry-run"
  exit 1
fi

echo ""
echo "✅ 公共代码 Git 跟踪校验通过"
