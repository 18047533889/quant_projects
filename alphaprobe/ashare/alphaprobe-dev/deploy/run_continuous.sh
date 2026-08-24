#!/usr/bin/env bash
# AlphaPROBE 7×24 连续挖掘启动脚本
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Python 环境（优先 rdagent4qlib：含 torch+cuda；可通过 ALPHAPROBE_PYTHON 覆盖）
PYTHON_BIN="${ALPHAPROBE_PYTHON:-/home/hsunbj/anaconda3/envs/rdagent4qlib/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[deploy] PYTHON not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

export PYTHONPATH="${PROJECT_ROOT}/src"
export ALPHAPROBE_DATA_BACKEND="${ALPHAPROBE_DATA_BACKEND:-factor_engine}"

CONFIG="${ALPHAPROBE_EXPERIMENT_CONFIG:-configs/alphaprobe/experiment_ashare_pv_7x24.yaml}"
CUDA_DEVICE="${ALPHAPROBE_CUDA:-0}"

mkdir -p "${PROJECT_ROOT}/data/logs/continuous"

echo "[deploy] project=${PROJECT_ROOT}"
echo "[deploy] python=${PYTHON_BIN}"
echo "[deploy] config=${CONFIG}"
echo "[deploy] cuda=${CUDA_DEVICE}"

exec "${PYTHON_BIN}" apps/alphaprobe/run_continuous.py \
  --experiment_config "${CONFIG}" \
  --cuda "${CUDA_DEVICE}"
