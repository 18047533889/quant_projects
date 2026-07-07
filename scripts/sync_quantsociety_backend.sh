#!/usr/bin/env bash
# 从 GitHub projects/quantsociety_backend_project 按需同步到 quant_projects
#
# 用法:
#   bash scripts/sync_quantsociety_backend.sh              # 同步推荐组件
#   bash scripts/sync_quantsociety_backend.sh --list       # 列出可同步组件
#   bash scripts/sync_quantsociety_backend.sh docs raw_data_layer
#   bash scripts/sync_quantsociety_backend.sh --all        # 同步全部（不含 factor_engine/data_access）
#
# 环境变量:
#   GITHUB_TOKEN   GitHub PAT（可选，脚本内有默认）
#   QS_SYNC_DRY_RUN=1  只打印将复制的路径，不写入
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_SSH="git@github.com:18047533889/projects.git"
GITHUB_TOKEN="${GITHUB_TOKEN:?请设置 GITHUB_TOKEN 环境变量（GitHub PAT）}"
SPARSE_ROOT="projects/quantsociety_backend_project"

# 组件名 → 上游相对路径（相对于 SPARSE_ROOT）
declare -A COMPONENTS=(
  [docs]="docs"
  [raw_data_layer]="raw_data_layer"
  [factor_layer]="factor_layer"
  [scripts]="scripts"
  [streaming]="streaming"
  [backtest_layer]="backtest_layer"
  [strategy_layer]="strategy_layer"
  [demo]="demo"
  [config_root]="."  # 根目录配置文件，特殊处理
)

# 默认推荐同步（与当前 quant_projects 工作流相关、体积可控）
DEFAULT_COMPONENTS=(
  docs
  raw_data_layer
  factor_layer
  scripts
  config_root
)

# 永不覆盖的本地定制目录（即使 --all）
PROTECTED_PATHS=(
  data_access
  factor_engine
  gtja191
  week2_pv_factors
  data
  logs
  output
)

DRY_RUN="${QS_SYNC_DRY_RUN:-0}"
TMP=""
UP=""

cleanup() {
  [[ -n "${TMP}" && -d "${TMP}" ]] && rm -rf "${TMP}"
}
trap cleanup EXIT

log() { echo "==> $*"; }

is_protected() {
  local rel="$1"
  for p in "${PROTECTED_PATHS[@]}"; do
    if [[ "${rel}" == "${p}" || "${rel}" == ${p}/* ]]; then
      return 0
    fi
  done
  return 1
}

copy_tree() {
  local src="$1"
  local dest="$2"
  if [[ ! -e "${src}" ]]; then
    echo "跳过（上游不存在）: ${src}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] rsync ${src} -> ${dest}"
    return 0
  fi
  rsync -a --delete "${src}/" "${dest}/"
}

copy_file() {
  local src="$1"
  local dest="$2"
  if [[ ! -f "${src}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] cp ${src} -> ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}"
}

clone_upstream() {
  local paths=("$@")
  TMP="$(mktemp -d)"
  local repo_url="https://${GITHUB_TOKEN}@github.com/18047533889/projects.git"
  if [[ "${1:-}" == "--ssh" ]]; then
    repo_url="${REPO_SSH}"
    shift
    paths=("$@")
  fi

  log "Sparse clone ${SPARSE_ROOT} ..."
  git clone --filter=blob:none --depth 1 --sparse "${repo_url}" "${TMP}/repo"
  local checkout_paths=()
  for p in "${paths[@]}"; do
    if [[ "${p}" == "." ]]; then
      checkout_paths+=("${SPARSE_ROOT}")
    else
      checkout_paths+=("${SPARSE_ROOT}/${p}")
    fi
  done
  git -C "${TMP}/repo" sparse-checkout set "${checkout_paths[@]}"
  git -C "${TMP}/repo" checkout
  UP="${TMP}/repo/${SPARSE_ROOT}"
}

sync_config_root() {
  copy_file "${UP}/pytest.ini" "${ROOT}/pytest.ini"
  copy_file "${UP}/.pre-commit-config.yaml" "${ROOT}/.pre-commit-config.yaml"
  # requirements-dev：仅在上游不存在本地文件时覆盖（保留 quant_projects 定制）
  if [[ ! -f "${ROOT}/requirements-dev.txt" ]]; then
    copy_file "${UP}/requirements-dev.txt" "${ROOT}/requirements-dev.txt"
  else
    log "保留本地 requirements-dev.txt"
  fi
  # allowlist：保留本地版本（quant_projects 扁平布局有额外豁免）
  if [[ ! -f "${ROOT}/.data_access_allowlist.yaml" ]]; then
    copy_file "${UP}/.data_access_allowlist.yaml" "${ROOT}/.data_access_allowlist.yaml"
  else
    log "保留本地 .data_access_allowlist.yaml"
  fi
}

sync_component() {
  local name="$1"
  local rel="${COMPONENTS[$name]:-}"
  if [[ -z "${rel}" ]]; then
    echo "未知组件: ${name}" >&2
    return 1
  fi

  if is_protected "${name}"; then
    echo "拒绝同步受保护路径: ${name}" >&2
    return 1
  fi

  if [[ "${name}" == "config_root" ]]; then
    sync_config_root
    return 0
  fi

  if [[ "${name}" == "factor_layer" ]]; then
    # 只同步 factor_engine 以外的子目录（本地 factor_engine 在仓库根）
    # 跳过因子评估相关组件（本机只做落盘，不做 evaluation）
    local skip_dirs=(
      factor_evaluation
      factor_evaluation_alphapurify
      factor_indicators_lysj
    )
    local sub base skip
    for sub in "${UP}/factor_layer"/*; do
      [[ -d "${sub}" ]] || continue
      base="$(basename "${sub}")"
      if [[ "${base}" == "factor_engine" ]]; then
        log "跳过 factor_layer/factor_engine（本地使用 quant_projects/factor_engine）"
        continue
      fi
      for skip in "${skip_dirs[@]}"; do
        if [[ "${base}" == "${skip}" ]]; then
          log "跳过 factor_layer/${base}（本机不做因子评估）"
          continue 2
        fi
      done
      log "同步 factor_layer/${base}"
      copy_tree "${sub}" "${ROOT}/factor_layer/${base}"
    done
    copy_file "${UP}/factor_layer/README.md" "${ROOT}/factor_layer/README.md"
    return 0
  fi

  if [[ "${name}" == "scripts" ]]; then
    mkdir -p "${ROOT}/scripts"
    for f in "${UP}/scripts"/*; do
      [[ -f "${f}" ]] || continue
      local bn
      bn="$(basename "${f}")"
      # 保留本地已有定制脚本
      if [[ -f "${ROOT}/scripts/${bn}" && "${bn}" != "check_data_access_allowlist.py" ]]; then
        log "保留本地 scripts/${bn}（不覆盖）"
        continue
      fi
      log "同步 scripts/${bn}"
      copy_file "${f}" "${ROOT}/scripts/${bn}"
      chmod +x "${ROOT}/scripts/${bn}" 2>/dev/null || true
    done
    return 0
  fi

  log "同步 ${name}"
  copy_tree "${UP}/${rel}" "${ROOT}/${rel}"
}

list_components() {
  echo "可同步组件:"
  for k in "${!COMPONENTS[@]}"; do
    echo "  - ${k}  →  ${COMPONENTS[$k]}"
  done
  echo ""
  echo "默认推荐: ${DEFAULT_COMPONENTS[*]}"
  echo "受保护（永不覆盖）: ${PROTECTED_PATHS[*]}"
}

main() {
  local selected=()

  if [[ "${1:-}" == "--list" ]]; then
    list_components
    exit 0
  fi

  if [[ "${1:-}" == "--all" ]]; then
    selected=("${!COMPONENTS[@]}")
  elif [[ $# -eq 0 ]]; then
    selected=("${DEFAULT_COMPONENTS[@]}")
  else
    selected=("$@")
  fi

  # 计算需要 checkout 的上游路径
  local paths=()
  for c in "${selected[@]}"; do
    if [[ "${c}" == "config_root" ]]; then
      paths+=(".")
    elif [[ -n "${COMPONENTS[$c]:-}" ]]; then
      paths+=("${COMPONENTS[$c]}")
    fi
  done

  clone_upstream "${paths[@]}"

  for c in "${selected[@]}"; do
    sync_component "${c}"
  done

  # factor_layer 包根
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${ROOT}/factor_layer"
    touch "${ROOT}/factor_layer/__init__.py"
  fi

  log "完成。根目录: ${ROOT}"
}

main "$@"
