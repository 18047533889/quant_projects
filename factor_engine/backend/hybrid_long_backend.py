# -*- coding: utf-8
"""混合 long-table 后端：SQL 下推 → PolarsLong → Pandas fallback。"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode

from .context import ExecutionContext
from .polars_long_backend import PolarsLongBackend
from .sql_backend import SqlBackend


class HybridLongBackend(SqlBackend):
    """``build_backend('auto_long')`` / ``hybrid_long``：SQL 子树 + Polars long native。"""

    runtime_backend_label = "hybrid_long"

    prefers_native_scan = True
    supports_lazy_shared = True

    def __init__(self) -> None:
        super().__init__(operator_backend="auto")
        self._polars_long = PolarsLongBackend()

    def compile_lazy_shared(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        return self._polars_long.compile_lazy_shared(plan, ctx, sid=sid)

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        ctx = replace(ctx, materialize_sql_as_long_lazy=True)
        return super().execute(plan, ctx)

    def _eval_hybrid(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        if self._operator_backend == "pandas_numpy":
            return self._python.execute(plan, ctx)
        return self._polars_long.execute(plan, ctx)

    def _eval_python(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        if self._operator_backend == "pandas_numpy":
            return self._python.execute(plan, ctx)
        return self._polars_long.execute(plan, ctx)
