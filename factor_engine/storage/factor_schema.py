"""因子湖长表单点契约：Parquet / datasets.yaml / ClickHouse 共用列定义。

R32-P0-026: 本模块是 factor lake schema 的单一事实源。``FACTOR_LAKE_SCHEMA_VERSION``
版本化，materializer / readers / DataAccess / ClickHouse 共用同一列集合与语义。
"""

from __future__ import annotations

#: R32-P0-026: factor lake long 表 schema 版本。schema 变更必须升版本 + 提供
#: 受控迁移（reader 支持旧版本迁移，writer 只写当前版本）。
FACTOR_LAKE_SCHEMA_VERSION = 1

FACTOR_CORE_COLUMNS: tuple[str, ...] = ("datetime", "asset", "value")

#: 每行落盘的 metadata 列（materializer 写当前 schema；reader 兼容旧 schema
#: 缺失列用 None 回填）。R32-P0-026: 补上 materializer 实际已写但此前未声明的
#: ``resolved_snapshot_id`` / ``storage_precision_policy`` —— 否则 schema authority
#: 与落盘 metadata 漂移。
FACTOR_METADATA_COLUMNS: tuple[str, ...] = (
    "calc_time",
    "factor_version",
    "data_snapshot_id",
    "is_valid",
    "invalid_reason",
    "resolved_snapshot_id",
    "storage_precision_policy",
)

FACTOR_VALUE_COLUMNS: tuple[str, ...] = FACTOR_CORE_COLUMNS + FACTOR_METADATA_COLUMNS

# datasets.yaml schema 块（union_by_name=true 时元数据列可缺省）
FACTOR_LAKE_DATASET_SCHEMA: dict[str, str] = {
    "datetime": "timestamp",
    "asset": "string",
    "value": "double",
    "calc_time": "string",
    "factor_version": "string",
    "data_snapshot_id": "string",
    "is_valid": "int",
    "invalid_reason": "string",
    "resolved_snapshot_id": "string",
    "storage_precision_policy": "string",
}

#: R32-P0-052: 可读的最小 schema 版本（受控旧版本迁移的下限）。低于此版本需
#: 显式 migration 而非静默读取。
FACTOR_LAKE_MIN_READABLE_SCHEMA_VERSION = 1
