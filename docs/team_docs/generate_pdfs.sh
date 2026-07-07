#!/bin/bash

echo "Starting Markdown to PDF conversion..."

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Define the launch options to bypass sandbox issues in some Linux environments
LAUNCH_OPTIONS='{"args": ["--no-sandbox", "--disable-setuid-sandbox"]}'

# Convert specific files
echo "Converting: 企业级量化因子评估与自动化入库系统需求文档.md"
npx md-to-pdf --launch-options "$LAUNCH_OPTIONS" "企业级量化因子评估与自动化入库系统需求文档.md"

echo "Converting: 企业级时序因子体系设计与实施总说明.md"
npx md-to-pdf --launch-options "$LAUNCH_OPTIONS" "企业级时序因子体系设计与实施总说明.md"

echo "Converting: 反馈意见_对模块间服务接口契约的改进建议.md"
npx md-to-pdf --launch-options "$LAUNCH_OPTIONS" "反馈意见_对模块间服务接口契约的改进建议.md"

echo "PDF generation complete! You can find the PDFs in $DIR"
