# -*- coding: utf-8 -*-
"""Polars 后端：算子层 auto 优先 polars，缺失时回退 pandas。"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .pandas_backend import PandasBackend
from runtime.perf_config import PerfConfig


class PolarsBackend(Backend):
    """``build_backend('polars')`` 入口；执行链与 PandasBackend 相同，算子优先 polars。"""

    def __init__(self, use_lazy: bool | None = None) -> None:
        _ = use_lazy  # lazy collect 预留
        self._inner = PandasBackend()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        ctx = self._with_polars_perf(ctx)
        return self._inner.execute(plan, ctx)

    @staticmethod
    def _with_polars_perf(ctx: ExecutionContext) -> ExecutionContext:
        perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
        if getattr(perf, "operator_backend", "auto") == "auto":
            perf = replace(perf, operator_backend="auto")
        return replace(ctx, perf=perf)
