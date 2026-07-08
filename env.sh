# quant_projects 环境变量（使用前 source 本文件）
#   source ~/quant_projects/env.sh

_QUANT_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_QUANT_ENV_ROOT}/.env" ]]; then
  # shellcheck source=/dev/null
  source "${_QUANT_ENV_ROOT}/.env"
fi
unset _QUANT_ENV_ROOT

export QUANT_PROJECTS_ROOT="${QUANT_PROJECTS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export QUANTSOCIETY_WORKSPACE_DATA_ROOT="${QUANTSOCIETY_WORKSPACE_DATA_ROOT:-${QUANT_PROJECTS_ROOT}/data}"
# FACTOR_LAKE_ROOT 默认由 workspace_paths 从 QUANTSOCIETY_WORKSPACE_DATA_ROOT 推导；显式覆盖时才 export
# export FACTOR_LAKE_ROOT="${FACTOR_LAKE_ROOT:-${QUANTSOCIETY_WORKSPACE_DATA_ROOT}/factors/lake}"
export FACTOR_ENGINE_ROOT="${FACTOR_ENGINE_ROOT:-${QUANT_PROJECTS_ROOT}/factor_engine}"

# data_access 数据集本地镜像
export ASHARE_PARQUET_ROOT="${ASHARE_PARQUET_ROOT:-${QUANTSOCIETY_WORKSPACE_DATA_ROOT}/a_share/lqtp_data}"
export US_MASSIVE_ROOT="${US_MASSIVE_ROOT:-${QUANTSOCIETY_WORKSPACE_DATA_ROOT}/us_stock/massive_data}"
export US_CLEAN_ROOT="${US_CLEAN_ROOT:-${QUANTSOCIETY_WORKSPACE_DATA_ROOT}/us_stock/clean_data}"

# COS 前缀（读前按需镜像）
export ASHARE_LQTP_COS_PREFIX="${ASHARE_LQTP_COS_PREFIX:-cos://qs-cold/clean_data/ashare/lqtp_data}"
export US_MASSIVE_COS_PREFIX="${US_MASSIVE_COS_PREFIX:-cos://qs-cold/clean_data/us_stock/massive_data}"
export US_CLEAN_COS_PREFIX="${US_CLEAN_COS_PREFIX:-cos://qs-cold/clean_data}"

# Python 导入路径：data_access + factor_engine
export PYTHONPATH="${QUANT_PROJECTS_ROOT}:${FACTOR_ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# DuckDB（30G 机器可酌情调低）
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-12GB}"
export DUCKDB_THREADS="${DUCKDB_THREADS:-$(nproc 2>/dev/null || echo 4)}"

# data_access：首次读跳过 schema 校验（已登记数据集可信；调试可改为 warn/strict）
export QUANT_SCHEMA_CHECK="${QUANT_SCHEMA_CHECK:-off}"

# factor_engine 性能默认（最快路径；调试时可显式关闭）
# CSE + panel_native 默认开启（见 PerfConfig.from_env）
# export FACTOR_ENGINE_DISABLE_CSE=1
# export FACTOR_ENGINE_DISABLE_PANEL_NATIVE=1
export FACTOR_ENGINE_USE_NUMBA="${FACTOR_ENGINE_USE_NUMBA:-1}"

# 关闭读前 COS 自动拉取（离线包分发时建议开启）
# export DATA_ACCESS_SKIP_COS_MIRROR=1
