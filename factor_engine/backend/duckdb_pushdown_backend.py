# -*- coding: utf-8 -*-
"""DuckDB SQL 下推后端：因子在数据库内计算，不支持则回退 PandasBackend。"""
from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .pandas_backend import PandasBackend
from .panel_native import finalize_panel_result
from .sql_pushdown.executor import try_execute_sql_pushdown


class DuckDBPushdownBackend(Backend):
    """``build_backend('duckdb_sql')``：优先 DuckDB SQL，失败则 Python 算子。"""

    def __init__(self) -> None:
        self._fallback = PandasBackend()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        pushed = try_execute_sql_pushdown(plan, ctx)
        if pushed is not None:
            return finalize_panel_result(pushed, ctx)
        return self._fallback.execute(plan, ctx)
