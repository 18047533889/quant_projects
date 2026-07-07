#!/usr/bin/env bash
# 延迟启动 GTJA191 全量 force 重落盘（算子改完后用）
# 用法:
#   bash scripts/scheduled_gtja191_rematerialize.sh --delay 3600   # 1 小时后
#   bash scripts/scheduled_gtja191_rematerialize.sh --now         # 立即
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DELAY_SEC=3600
RUN_NOW=0
LOG="${ROOT}/logs/scheduled_gtja191_rematerialize.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delay) DELAY_SEC="$2"; shift 2 ;;
    --now) RUN_NOW=1; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${ROOT}/logs"

if [[ "${RUN_NOW}" -eq 0 && "${DELAY_SEC}" -gt 0 ]]; then
  echo "[$(date -Iseconds)] 等待 ${DELAY_SEC}s 后开始 GTJA191 force 重落盘..." | tee -a "${LOG}"
  sleep "${DELAY_SEC}"
fi

echo "[$(date -Iseconds)] 开始：检查 factor_engine 测试 + force 全量落盘" | tee -a "${LOG}"

cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/env.sh"
unset FACTOR_LAKE_ROOT

# 1) 快速 smoke：factor_engine 核心测试
if ! python3 -m pytest factor_engine/tests/ -q --tb=line -x >> "${LOG}" 2>&1; then
  echo "[$(date -Iseconds)] ABORT: factor_engine 测试未通过，请人工检查后再落盘" | tee -a "${LOG}"
  exit 1
fi

# 2) 全量 force 重落盘（算子变更后必须 --force，禁止 --skip-existing）
python3 scripts/materialize_gtja191_factors.py \
  --force \
  --batch-size 1 \
  >> "${LOG}" 2>&1

echo "[$(date -Iseconds)] 落盘完成，运行校验..." | tee -a "${LOG}"
python3 scripts/verify_gtja191_factor_lake.py --sample-size 25 >> "${LOG}" 2>&1 || true

echo "[$(date -Iseconds)] DONE" | tee -a "${LOG}"
