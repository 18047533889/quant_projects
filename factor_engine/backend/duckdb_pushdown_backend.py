# -*- coding: utf-8 -*-
"""DuckDB / ClickHouse SQL 下推后端。

``DuckDBPushdownBackend`` 继承 ``SqlBackend``，将逻辑计划中的可下推子树
编译为 DuckDB 或 ClickHouse SQL 执行，不可下推部分回退到 Polars/Pandas 算子层。

用法：``build_backend('duckdb_sql')`` 或 ``build_backend('clickhouse_sql')``。
企业级别名 ``SqlPushdownBackend`` 按 ``data_source`` 自动选择方言。
"""
from __future__ import annotations

from .sql_backend import SqlBackend


class DuckDBPushdownBackend(SqlBackend):
    """DuckDB / ClickHouse SQL 下推执行后端。

    支持整树或部分子树 SQL 预计算，剩余节点委托 Polars auto 路径执行；
    算子后端固定为 ``auto``（优先 Polars，必要时 Pandas fallback）。
    """

    def __init__(self) -> None:
        """初始化 SQL 下推后端，算子层使用 ``operator_backend='auto'``。"""
        super().__init__(operator_backend="auto")


# 企业级别名：按 data_source 自动选 DuckDB / ClickHouse 方言
SqlPushdownBackend = DuckDBPushdownBackend
