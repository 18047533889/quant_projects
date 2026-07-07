#!/bin/bash

# ========================================
# 自动化定时任务安装脚本
# Cron Job Installation Script
# TASK-ENG-004: CRON_PAUSED 存在时禁止安装
# ========================================

set -euo pipefail

SCRIPT_DIR="/home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts"
PAUSED_FILE="$SCRIPT_DIR/../CRON_PAUSED"
CRON_FILE="/tmp/data_update_cron"

echo "========================================"
echo "数据更新定时任务安装脚本"
echo "========================================"

if [[ -f "$PAUSED_FILE" ]]; then
  echo "❌ TASK-ENG-004: 增量 cron 已熔断，禁止安装定时任务。"
  echo "   见: $PAUSED_FILE"
  sed -n '1,14p' "$PAUSED_FILE"
  echo ""
  echo "解除后再安装，且须同时满足 FEAT-008 / TASK-ENG-001/002 验收。"
  exit 1
fi

if [[ "${MASSIVE_ALLOW_CRON_INSTALL:-}" != "1" ]]; then
  echo "❌ 安全门禁: 请显式设置 MASSIVE_ALLOW_CRON_INSTALL=1 后再安装。"
  echo "   并确认 output 已指向 massive_parquet/raw_massive_data（非 lfl_workspace）。"
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/daily_update_scheduler.sh" ]]; then
  echo "❌ 每日更新脚本不存在: $SCRIPT_DIR/daily_update_scheduler.sh"
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/minute_update_scheduler.sh" ]]; then
  echo "❌ 分钟级更新脚本不存在: $SCRIPT_DIR/minute_update_scheduler.sh"
  exit 1
fi

chmod +x "$SCRIPT_DIR/daily_update_scheduler.sh"
chmod +x "$SCRIPT_DIR/minute_update_scheduler.sh"
chmod +x "$SCRIPT_DIR/_assert_cron_allowed.sh"

echo "✅ 脚本权限设置完成"

cat > "$CRON_FILE" << EOF
# Massive 数据更新定时任务 — 输出须指向主库 raw_massive_data
# 每日数据更新 - 5:00 AM EST
0 5 * * * QS_NODE_ID=\$(hostname -s) $SCRIPT_DIR/daily_update_scheduler.sh >> $SCRIPT_DIR/cron_daily.log 2>&1

# 分钟级：FEAT-008 完成前保持注释
# */15 9-16 * * 1-5 $SCRIPT_DIR/minute_update_scheduler.sh >> $SCRIPT_DIR/cron_minute.log 2>&1
EOF

echo "生成的定时任务配置："
echo "----------------------------------------"
cat "$CRON_FILE"
echo "----------------------------------------"

echo "当前用户的定时任务："
crontab -l 2>/dev/null || echo "（无定时任务）"
echo "----------------------------------------"

echo "安装选项："
echo "1) 安装每日更新任务（推荐）"
echo "2) 查看说明"
echo "3) 退出"
echo "----------------------------------------"

read -p "请选择操作 [1-3]: " choice

case $choice in
  1)
    echo "正在安装每日更新任务..."
    (crontab -l 2>/dev/null | grep -v 'daily_update_scheduler.sh' || true
     echo "# Massive daily update (TASK-ENG-004+FEAT-008)"
     echo "0 5 * * * QS_NODE_ID=\$(hostname -s) $SCRIPT_DIR/daily_update_scheduler.sh >> $SCRIPT_DIR/cron_daily.log 2>&1") | crontab -
    echo "✅ 每日更新任务已安装"
    ;;
  2)
    echo "说明："
    echo "- 日更须写 massive_parquet/raw_massive_data"
    echo "- SIP 用 download_history.py，REST 用 download_all_history (max_pages=None)"
    echo "- 分钟 cron 待 incremental_update_master / FEAT-008"
    ;;
  3)
    rm -f "$CRON_FILE"
    exit 0
    ;;
  *)
    echo "❌ 无效选择"
    rm -f "$CRON_FILE"
    exit 1
    ;;
esac

rm -f "$CRON_FILE"
echo "更新后的定时任务："
crontab -l
echo "========================================"
