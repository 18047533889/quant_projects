#!/bin/bash

# ========================================
# 每日数据更新自动化脚本 (重构版)
# Daily Data Update Automation Script (Refactored)
# TASK-ENG-004: 修复前默认熔断 — 见 ../CRON_PAUSED
# ========================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# scripts -> data_daily_update -> raw_data_layer -> project root
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_RUNNER_SCRIPT="$PROJECT_ROOT/raw_data_layer/data_daily_update/production_update_runner.py"
LOG_DIR="$PROJECT_ROOT/raw_data_layer/data_daily_update/_meta/logs"
PYTHON_CMD="${PYTHON_CMD:-/home/yluel/anaconda3/bin/python3}"

# --- TASK-ENG-004 熔断 ---
# shellcheck source=_assert_cron_allowed.sh
source "$SCRIPT_DIR/_assert_cron_allowed.sh"

if [[ -z "${MASSIVE_API_KEY:-}" ]]; then
  echo "❌ MASSIVE_API_KEY 未设置（禁止在脚本内写死默认 key）"
  exit 1
fi

export QS_NODE_ID="${QS_NODE_ID:-$(hostname -s)}"

EST_TIME=$(TZ="America/New_York" date '+%Y-%m-%d %H:%M:%S %Z')
BEIJING_TIME=$(TZ="Asia/Shanghai" date '+%Y-%m-%d %H:%M:%S %Z')

echo "========================================"
echo "每日数据更新流程启动 (QS_NODE_ID=$QS_NODE_ID)"
echo "EST Time: $EST_TIME"
echo "Beijing Time: $BEIJING_TIME"
echo "========================================"

cd "$PROJECT_ROOT" || {
  echo "❌ 无法进入项目根目录: $PROJECT_ROOT"
  exit 1
}

mkdir -p "$LOG_DIR"

CURRENT_DATE=$(date '+%Y%m%d')
CURRENT_TIME=$(date '+%H%M%S')
LOG_FILE="$LOG_DIR/daily_update_${CURRENT_DATE}_${CURRENT_TIME}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "开始执行每日数据更新流程..."
log "项目根目录: $PROJECT_ROOT"
log "日志文件: $LOG_FILE"

if ! command -v "$PYTHON_CMD" &> /dev/null; then
  log "❌ Python解释器未找到: $PYTHON_CMD"
  exit 1
fi

if [[ ! -f "$PYTHON_RUNNER_SCRIPT" ]]; then
  log "❌ Python执行脚本不存在: $PYTHON_RUNNER_SCRIPT"
  exit 1
fi

log "✅ 环境检查通过"
log "正在执行增量数据更新..."
START_TIME=$(date '+%s')

export PYTHONPATH="$PROJECT_ROOT"

$PYTHON_CMD "$PYTHON_RUNNER_SCRIPT" >> "$LOG_FILE" 2>&1
UPDATE_EXIT_CODE=$?

END_TIME=$(date '+%s')
DURATION=$((END_TIME - START_TIME))

if [[ $UPDATE_EXIT_CODE -eq 0 ]]; then
  log "✅ 数据更新成功完成 (耗时: ${DURATION}秒)"
  find "$LOG_DIR" -name "daily_update_*.log" -type f -mtime +30 -delete 2>/dev/null || true
else
  log "❌ 数据更新失败 (退出码: $UPDATE_EXIT_CODE, 耗时: ${DURATION}秒)"
  log "请检查日志: $LOG_FILE"
  exit 1
fi

log "每日数据更新流程完成 (耗时: ${DURATION}秒)"
exit 0
