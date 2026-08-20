# -*- coding: utf-8 -*-
"""SQL 下推后端适配器（DuckDB / ClickHouse 共享编译 IR，独立运行时边界）。

``SqlBackend`` 仍是整树 SQL + Python/Polars fallback 的通用引擎；
``DuckDBPushdownBackend`` 和 ``ClickHousePushdownBackend`` 继承它，
分别声明各自的执行身份与生产标注，使工厂、证书链与 telemetry 不再通过
动态属性注入来区分方言。

企业级别名 ``SqlPushdownBackend`` 保留为 DuckDB 默认，避免既有导入断裂。

MB-P0-002: Uses concrete backend identity 'duckdb_sql' / 'clickhouse_sql'.
MB-P1-018 (Section 21): DuckDB Region boundary prefers Arrow/Relation output.
"""
from __future__ import annotations

from .sql_backend import SqlBackend


class DuckDBPushdownBackend(SqlBackend):
    """DuckDB SQL 下推执行适配器。

    支持整树或部分子树 SQL 预计算，剩余节点委托 Polars auto 路径执行；
    算子后端固定为 ``auto``（优先 Polars，必要时 Pandas fallback）。

    MB-P0-002: Concrete backend identity (duckdb_sql) for cost/capability/telemetry.
    MB-P1-018: Prefers Arrow boundary over Pandas when downstream can consume it.
    """

    # MB-P0-002: Use concrete backend identity, not generic "sql"
    runtime_backend_label = "duckdb_sql"

    def __init__(self) -> None:
        """初始化 DuckDB SQL 下推后端，算子层使用 ``operator_backend='auto'``。"""
        super().__init__(operator_backend="auto")


class ClickHousePushdownBackend(SqlBackend):
    """ClickHouse SQL 下推执行适配器。

    与 ``DuckDBPushdownBackend`` 共享同一套 SQL 编译 IR / emitter /
    executor 链，但拥有独立的运行时边界、证书链字段与 telemetry 身份，
    使 ``build_backend('clickhouse_sql')`` 不再需要运行时注入私有属性。

    MB-P0-002: Concrete backend identity (clickhouse_sql) for cost/capability/telemetry.
    """

    runtime_backend_label = "clickhouse_sql"

    def __init__(self) -> None:
        """初始化 ClickHouse SQL 下推后端，算子层使用 ``operator_backend='auto'``。"""
        super().__init__(operator_backend="auto")


# 企业级别名：默认仍为 DuckDB（保持向后兼容）。
SqlPushdownBackend = DuckDBPushdownBackend
