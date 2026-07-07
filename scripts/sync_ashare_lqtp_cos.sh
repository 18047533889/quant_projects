#!/usr/bin/env bash
# 从 COS 同步 A 股 lqtp 数据到本地镜像
#
# 用法:
#   bash scripts/sync_ashare_lqtp_cos.sh                    # 同步全部 20 张表
#   bash scripts/sync_ashare_lqtp_cos.sh StockDailyBar      # 只同步日线
#   bash scripts/sync_ashare_lqtp_cos.sh StockDailyBar IndexDailyBar
#
# 环境变量:
#   ASHARE_PARQUET_ROOT   本地根目录（默认 ~/quant_projects/data/a_share/lqtp_data）
#   ASHARE_LQTP_COS_PREFIX COS 前缀（默认 cos://qs-cold/clean_data/ashare/lqtp_data）
#   ASHARE_COS_CLI        命令（默认 clean-cos-ro）
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_ROOT="${ASHARE_PARQUET_ROOT:-${ROOT}/data/a_share/lqtp_data}"
COS_PREFIX="${ASHARE_LQTP_COS_PREFIX:-cos://qs-cold/clean_data/ashare/lqtp_data}"
COS_CLI="${DATA_ACCESS_COS_CLI:-clean-cos-ro}"

ALL_TABLES=(
  Calendar
  ETFDailyBar
  ETFList
  IndexConstituent
  IndexDailyBar
  IndexList
  StockBalance
  StockCapitalDaily
  StockCashFlow
  StockDailyBar
  StockDividend
  StockIncome
  StockIndicator
  StockIndustry
  StockList
  StockMinuteBar
  StockStatus
  StockTopTenFloatShareholder
  StockTopTenShareholder
  StockValuationDaily
)

sync_table() {
  local table="$1"
  local dest="${LOCAL_ROOT}/${table}"
  mkdir -p "${dest}"

  if [[ "${table}" == "Calendar" ]]; then
    echo "==> ${table} (single file)"
    "${COS_CLI}" cp "${COS_PREFIX}/${table}/full.parquet" "${dest}/full.parquet"
    return
  fi

  echo "==> ${table}"
  "${COS_CLI}" sync "${COS_PREFIX}/${table}/" "${dest}/"
}

if [[ $# -eq 0 ]]; then
  TABLES=("${ALL_TABLES[@]}")
else
  TABLES=("$@")
fi

for t in "${TABLES[@]}"; do
  sync_table "${t}"
done

echo "完成。本地镜像: ${LOCAL_ROOT}"
