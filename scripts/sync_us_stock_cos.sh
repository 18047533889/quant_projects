#!/usr/bin/env bash
# 从 COS 同步美股数据到本地镜像（massive_data + clean_data 衍生层）
#
# 用法:
#   bash scripts/sync_us_stock_cos.sh                  # 同步全部
#   bash scripts/sync_us_stock_cos.sh StockDailyBar    # 只同步日线
#   bash scripts/sync_us_stock_cos.sh --clean-only     # 只同步 clean_data 层
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MASSIVE_ROOT="${US_MASSIVE_ROOT:-${ROOT}/data/us_stock/massive_data}"
CLEAN_ROOT="${US_CLEAN_ROOT:-${ROOT}/data/us_stock/clean_data}"
MASSIVE_COS="${US_MASSIVE_COS_PREFIX:-cos://qs-cold/clean_data/us_stock/massive_data}"
CLEAN_COS="${US_CLEAN_COS_PREFIX:-cos://qs-cold/clean_data}"
COS_CLI="${DATA_ACCESS_COS_CLI:-clean-cos-ro}"

MASSIVE_TABLES=(
  Calendar
  ETFDailyBar
  ETFList
  SecurityMaster
  StockBalance
  StockCapitalDaily
  StockCashFlow
  StockDailyBar
  StockDividend
  StockIncome
  StockIndicator
  StockIndicesComponents
  StockIndustry
  StockList
  StockStatus
  StockValuationDaily
  TickerAlias
  TickerMap
)

sync_massive_table() {
  local table="$1"
  local dest="${MASSIVE_ROOT}/${table}"
  mkdir -p "${dest}"
  echo "==> massive_data/${table}"
  "${COS_CLI}" sync "${MASSIVE_COS}/${table}/" "${dest}/"
}

sync_clean_layer() {
  echo "==> universe_daily"
  mkdir -p "${CLEAN_ROOT}/universe_daily"
  "${COS_CLI}" sync "${CLEAN_COS}/universe_daily/" "${CLEAN_ROOT}/universe_daily/"

  echo "==> adj_factor"
  mkdir -p "${CLEAN_ROOT}/adj_factor"
  "${COS_CLI}" sync "${CLEAN_COS}/adj_factor/" "${CLEAN_ROOT}/adj_factor/"

  echo "==> is_adj_factor_clamped"
  mkdir -p "${CLEAN_ROOT}/is_adj_factor_clamped"
  "${COS_CLI}" sync "${CLEAN_COS}/is_adj_factor_clamped/" "${CLEAN_ROOT}/is_adj_factor_clamped/"

  echo "==> is_early_close"
  mkdir -p "${CLEAN_ROOT}/is_early_close"
  "${COS_CLI}" cp "${CLEAN_COS}/is_early_close/data.parquet" "${CLEAN_ROOT}/is_early_close/data.parquet"

  echo "==> is_ticker_halt/minute"
  mkdir -p "${CLEAN_ROOT}/is_ticker_halt/minute"
  "${COS_CLI}" sync "${CLEAN_COS}/is_ticker_halt/minute/" "${CLEAN_ROOT}/is_ticker_halt/minute/"
}

clean_only=false
tables=()

for arg in "$@"; do
  if [[ "${arg}" == "--clean-only" ]]; then
    clean_only=true
  else
    tables+=("${arg}")
  fi
done

if [[ "${clean_only}" == true ]]; then
  sync_clean_layer
elif [[ ${#tables[@]} -eq 0 ]]; then
  for t in "${MASSIVE_TABLES[@]}"; do
    sync_massive_table "${t}"
  done
  sync_clean_layer
else
  for t in "${tables[@]}"; do
    sync_massive_table "${t}"
  done
fi

echo "完成。massive: ${MASSIVE_ROOT}  clean: ${CLEAN_ROOT}"
