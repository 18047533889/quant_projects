"""Polars 后端：当前委托给 ``PandasBackend``（cleaned_operators 路径）。"""

from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .pandas_backend import PandasBackend


class PolarsBackend(Backend):
    """保留 ``build_backend(\"polars\")`` 入口；执行走 cleaned + pandas panel 桥。"""

    def __init__(self, use_lazy: bool | None = None) -> None:
        _ = use_lazy
        self._inner = PandasBackend()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        return self._inner.execute(plan, ctx)
