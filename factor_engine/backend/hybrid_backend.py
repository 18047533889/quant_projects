# -*- coding: utf-8 -*-
"""混合后端：SqlBackend（部分 SQL）→ Polars auto。"""
from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .sql_backend import SqlBackend


class HybridBackend(Backend):
    """``build_backend('auto')`` / ``hybrid``：SQL 子树 + Polars 算子 + Pandas fallback。"""

    def __init__(self) -> None:
        self._sql = SqlBackend(operator_backend="auto")

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        return self._sql.execute(plan, ctx)
