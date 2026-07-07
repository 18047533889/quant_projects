#!/bin/bash

# 脚本将在遇到任何错误时立即退出
set -e

# --- 请在这里配置您的路径 ---
# 源目录 (注意末尾的斜杠，表示复制目录内容)
SOURCE_DIR="/home/yluel/share/projects/massive_parquet/"

# 目标目录
DEST_DIR="/home/yluel/share/project_data_backup/"
# --------------------------

# 确保目标目录存在
mkdir -p "$DEST_DIR"

echo "================================================="
echo "开始将数据从 '$SOURCE_DIR' 复制到 '$DEST_DIR'..."
echo "这是一个大任务，可能需要数小时或数天。"
echo "如果中断，只需重新运行此脚本即可恢复。"
echo "================================================="

# 执行 rsync 命令
rsync -avh --progress "$SOURCE_DIR" "$DEST_DIR"

echo "================================================="
echo "复制完成！"
echo "================================================="