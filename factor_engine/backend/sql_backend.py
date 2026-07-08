# -*- coding: utf-8 -*-
"""SqlBackend：整树 SQL + 部分子树 SQL 预计算 + Python/Polars fallback。"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode
from planner.sql_lowerer import lower_to_physical_plan

from .base import Backend
from .context import ExecutionContext
from .panel_native import finalize_panel_result
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .sql_pushdown.executor import try_execute_sql_pushdown, try_execute_sql_pushdown_batch
from .sql_pushdown.sql_registry import is_sql_capable


class SqlBackend(Backend):
    """混合 SQL 执行后端（DuckDB / ClickHouse + pandas/polars）。"""

    def __init__(self, *, operator_backend: str = "auto") -> None:
        self._operator_backend = operator_backend
        self._python = PandasBackend()
        self._polars = PolarsBackend()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        physical = lower_to_physical_plan(plan)

        if physical.fully_sql:
            pushed = try_execute_sql_pushdown(physical.root, ctx)
            if pushed is not None:
                return finalize_panel_result(pushed, ctx)

        mat_cache = dict(getattr(ctx, "materialized_series", None) or {})
        pending = {
            sid: sub
            for sid, sub in physical.sql_subtrees.items()
            if sid not in mat_cache
        }
        if len(pending) > 1:
            batch = try_execute_sql_pushdown_batch(pending, ctx)
            if batch:
                mat_cache.update(batch)
                pending = {sid: sub for sid, sub in pending.items() if sid not in mat_cache}

        for sid, sub in pending.items():
            if sid in mat_cache:
                continue
            pushed = try_execute_sql_pushdown(sub, ctx)
            if pushed is not None:
                mat_cache[sid] = pushed
            else:
                mat_cache[sid] = self._eval_python(sub, ctx)

        ctx = replace(ctx, materialized_series=mat_cache)
        return self._eval_hybrid(physical.root, ctx)

    def _eval_hybrid(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        if self._operator_backend == "pandas_numpy":
            return self._python.execute(plan, ctx)
        return self._polars.execute(plan, ctx)

    def _eval_python(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        backend = self._python if self._operator_backend == "pandas_numpy" else self._polars
        if hasattr(backend, "_eval"):
            return backend._eval(plan, ctx)
        return backend.execute(plan, ctx)
