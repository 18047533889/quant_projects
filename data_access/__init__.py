"""
data_access —— 团队统一数据读写入口（PR1：读；PR2：写；PR3：publish/upsert/sql）

读（PR1）：
    from data_access import get_store
    store = get_store()
    df = store.read_frame(
        "us_stocks_sip_day_aggs",
        columns=["align_time", "ticker", "close"],
        time_range=("2024-01-01", "2024-12-31"),
    )

写（PR2，只允许 namespaced/staging 数据集）：
    import pyarrow as pa
    store.write_arrow(
        "single_asset_backtest_runs",
        pa.Table.from_pandas(df),
        strategy_id="mom_3d", version="v1",
        mode="overwrite",                  # or "append"
        partition_by=["year"],             # 可选
    )

合并写入（PR3，按 upsert_on 去重的幂等合并）：
    store.upsert(
        "factor_lake_staging", tbl,
        factor_id="mom_3d",
        upsert_on=["datetime", "asset"],
        partition_by=["year"],
    )

发布（PR3，staging → published 原子晋升 + 旧版本归档）：
    store.publish_from_staging(
        "factor_lake_staging", "factor_lake",
        factor_id="mom_3d",                # 两边 params_schema 必须一致
    )
    # → {"target_path": "...", "archive_path": "..._archive/...", "rows": ...}

临时分析（PR3，有限 SQL 逃生口；只允许 SELECT + 预声明数据集）：
    tbl = store.sql(
        "SELECT asset, AVG(value) FROM factor_lake GROUP BY asset",
        read_datasets=["factor_lake"],
        read_params={"factor_lake": {"factor_id": "mom_3d"}},
    )

进阶：
    tbl = store.read_arrow(...)          # 零拷贝 Arrow Table，大表首选
    cols = store.load_columns(...)       # 批量多列 → dict[name, Series]

禁止：
    直接 import duckdb / pd.read_parquet / pq.read_table（除了 data_access 内部
    和 `.data_access_allowlist.yaml` 列出的例外）
    直接写 published 数据集 —— 永远必须走 staging → publish_from_staging

更多：
    - 团队规范：docs/data_access/01_团队使用规范.md
    - 快速上手：docs/data_access/02_快速上手.md
    - 数据集登记：data_access/config/datasets.yaml
    - 本模块设计：docs/data_access/10_架构设计.md
"""

from __future__ import annotations

from .engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from .exceptions import DataAccessError, DataError, EngineError, ValidationError
from .query_budget import QueryBudget
from .store import DataAccessStore, get_store, reset_store

__all__ = [
    "get_store",
    "reset_store",
    "get_shared_engine",
    "reset_shared_engine",
    "DataAccessStore",
    "DuckDBEngine",
    "DataAccessError",
    "ValidationError",
    "DataError",
    "EngineError",
    "QueryBudget",
]

__version__ = "0.1.0"
