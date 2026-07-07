#!/usr/bin/env bash
# quant_projects 一键初始化（给他人分发后首次运行）
#
# 用法:
#   bash scripts/setup_quant_projects.sh
#   bash scripts/setup_quant_projects.sh --verify-only
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERIFY_ONLY=false
if [[ "${1:-}" == "--verify-only" ]]; then
  VERIFY_ONLY=true
fi

log() { echo "==> $*"; }

# shellcheck source=/dev/null
source "${ROOT}/env.sh"

if [[ "${VERIFY_ONLY}" != true ]]; then
  log "安装 Python 依赖"
  python3 -m pip install -q -r "${ROOT}/requirements-dev.txt"
  python3 -m pip install -q -e "${ROOT}/factor_engine" 2>/dev/null || true
fi

log "检查目录结构"
required_dirs=(
  data_access
  factor_engine
  factor-pool-standard
  scripts
  data/a_share/lqtp_data
  data/us_stock/massive_data
  data/factors/lake
)
for d in "${required_dirs[@]}"; do
  if [[ ! -e "${ROOT}/${d}" ]]; then
    echo "缺少目录: ${ROOT}/${d}" >&2
    exit 1
  fi
done

log "检查关键文件"
required_files=(
  data_access/store.py
  data_access/config/datasets.yaml
  factor_engine/runtime/engine.py
  factor-pool-standard/enums/canonical_data_fields.json
  scripts/check_data_access_allowlist.py
  .data_access_allowlist.yaml
)
for f in "${required_files[@]}"; do
  if [[ ! -f "${ROOT}/${f}" ]]; then
    echo "缺少文件: ${ROOT}/${f}" >&2
    exit 1
  fi
done

log "运行 data_access allowlist 检查"
python3 "${ROOT}/scripts/check_data_access_allowlist.py"

log "运行 data_access 单元测试"
python3 -m pytest "${ROOT}/data_access/tests/unit" -q

log "运行 factor_engine 核心测试"
cd "${ROOT}/factor_engine"
# 不 source env.sh 中的 FACTOR_LAKE_ROOT，避免覆盖 pytest monkeypatch
env -u FACTOR_LAKE_ROOT DATA_ACCESS_SKIP_COS_MIRROR=1 \
  PYTHONPATH="${ROOT}:${ROOT}/factor_engine" \
  python3 -m pytest tests/ -q --ignore=tests/test_real_data_factor_smoke.py 2>&1 | tail -5

log "smoke: 加载 data_access registry"
python3 - <<'PY'
import os
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
from data_access.registry import load_registry
reg = load_registry()
assert "ashare_stock_daily" in reg._datasets
assert "us_stock_daily" in reg._datasets
print("datasets:", len(reg._datasets))
PY

log "完成。请执行: source ${ROOT}/env.sh"
echo ""
echo "常用命令:"
echo "  source ${ROOT}/env.sh"
echo "  bash ${ROOT}/scripts/sync_ashare_lqtp_cos.sh StockDailyBar"
echo "  python3 ${ROOT}/scripts/materialize_gtja191_factors.py --skip-existing"
