#!/bin/bash

# ========================================
# 分钟级数据更新自动化脚本
# Minute-level Data Update Automation Script
# TASK-ENG-004: 修复前默认熔断 — incremental_update_master.py 尚不存在
# ========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAILY_UPDATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$DAILY_UPDATE_DIR/_meta/logs"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# --- TASK-ENG-004 熔断 ---
# shellcheck source=_assert_cron_allowed.sh
source "$SCRIPT_DIR/_assert_cron_allowed.sh"

EST_TIME=$(TZ="America/New_York" date '+%Y-%m-%d %H:%M:%S %Z')
BEIJING_TIME=$(TZ="Asia/Shanghai" date '+%Y-%m-%d %H:%M:%S %Z')

echo "========================================"
echo "分钟级数据更新自动化脚本启动"
echo "EST时间: $EST_TIME"
echo "北京时间: $BEIJING_TIME"
echo "========================================"

mkdir -p "$LOG_DIR"

CURRENT_DATE=$(date '+%Y%m%d')
CURRENT_TIME=$(date '+%H%M%S')
LOG_FILE="$LOG_DIR/minute_scheduler_${CURRENT_DATE}_${CURRENT_TIME}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "❌ incremental_update_master.py 尚未实现；分钟 cron 在 FEAT-008 / TASK-ENG-004 完成前禁用。"
log "生产分钟/SIP 补数请用 download_history.py（见手册 §4 / 附录 I.2）。"
exit 1
