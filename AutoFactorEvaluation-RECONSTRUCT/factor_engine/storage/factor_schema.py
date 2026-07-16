"""因子湖长表单点契约：Parquet / datasets.yaml / ClickHouse 共用列定义。"""

from __future__ import annotations

FACTOR_CORE_COLUMNS: tuple[str, ...] = ("datetime", "asset", "value")

FACTOR_METADATA_COLUMNS: tuple[str, ...] = (
    "calc_time",
    "factor_version",
    "data_snapshot_id",
    "is_valid",
    "invalid_reason",
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
}
