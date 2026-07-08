# -*- coding: utf-8 -*-
"""DuckDB SQL 下推后端（委托 SqlBackend）。"""
from __future__ import annotations

from .sql_backend import SqlBackend


class DuckDBPushdownBackend(SqlBackend):
    """``build_backend('duckdb_sql')``：部分/整树 SQL + Python fallback。"""

    def __init__(self) -> None:
        super().__init__(operator_backend="pandas_numpy")
