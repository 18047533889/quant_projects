#!/bin/bash
# ================================================================
# 从 COS 复制样本数据到本地 data/ 目录
#
# 用法:
#   bash copy_sample_data.sh
#
# 前提:
#   COS 已挂载到 /mnt/cos/ 或可访问
#   或者你有 COS CLI 工具 (coscmd)
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

COS_BASE="cos://qs-cold/clean_data"

# ================================================================
# 方式 1: COS 已挂载为本地文件系统
# ================================================================
# 取消注释下面几行，设置你的挂载路径
# COS_MOUNT="/mnt/cos"
#
# echo "从 COS 挂载点复制数据..."
#
# # A 股: 最近 60 个交易日
# echo "  A 股 StockDailyBar (最近 60 日)..."
# cp "$COS_MOUNT/qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/2025-0"* "$DATA_DIR/ashare/StockDailyBar/"
# cp "$COS_MOUNT/qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/2025-1"* "$DATA_DIR/ashare/StockDailyBar/"
#
# # A 股: Calendar
# echo "  A 股 Calendar..."
# cp "$COS_MOUNT/qs-cold/clean_data/ashare/lqtp_data/Calendar/full.parquet" "$DATA_DIR/ashare/Calendar/"
#
# # 美股: PanelDaily 最近 60 个交易日
# echo "  美股 PanelDaily (最近 60 日)..."
# cp "$COS_MOUNT/qs-cold/clean_data/us_stock/massive_data/PanelDaily/2025-0"* "$DATA_DIR/us_stock/PanelDaily/"
# cp "$COS_MOUNT/qs-cold/clean_data/us_stock/massive_data/PanelDaily/2025-1"* "$DATA_DIR/us_stock/PanelDaily/"


# ================================================================
# 方式 2: 使用 coscmd (腾讯云 COS CLI)
# ================================================================
# 取消注释下面几行，先 pip install coscmd 并配置

echo "使用 coscmd 下载数据..."
echo "请确保已安装并配置 coscmd: pip install coscmd && coscmd config -a ..."

COS_BUCKET="qs-cold"
COS_REGION=""  # 如 ap-guangzhou

# A 股 Calendar（单文件）
echo "  [1/4] A 股 Calendar..."
coscmd download "clean_data/ashare/lqtp_data/Calendar/full.parquet" \
    "$DATA_DIR/ashare/Calendar/full.parquet"

# A 股 StockDailyBar 最近 60 日
echo "  [2/4] A 股 StockDailyBar (2025-10 ~ 2025-12)..."
for month in 10 11 12; do
    coscmd download -r "clean_data/ashare/lqtp_data/StockDailyBar/2025-${month}-" \
        "$DATA_DIR/ashare/StockDailyBar/" || true
done

# 美股 PanelDaily 最近 60 日
echo "  [3/4] 美股 PanelDaily (2025-10 ~ 2025-12)..."
for month in 10 11 12; do
    coscmd download -r "clean_data/us_stock/massive_data/PanelDaily/2025-${month}-" \
        "$DATA_DIR/us_stock/PanelDaily/" || true
done


# ================================================================
# 方式 3: Python 脚本（使用 boto3 / cos-python-sdk-v5）
# ================================================================
# 也提供了 Python 版本: python copy_sample_data.py

echo ""
echo "========================================"
echo "  数据目录结构"
echo "========================================"
find "$DATA_DIR" -name "*.parquet" | head -20 | while read f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $size  $(basename "$f")"
done

echo ""
echo "文件总数: $(find "$DATA_DIR" -name '*.parquet' | wc -l)"
echo ""
echo "✅ 如果上面没有文件，请选择一种方式下载数据："
echo "   1. 挂载 COS + cp"
echo "   2. coscmd download"
echo "   3. Python cos-python-sdk-v5"
