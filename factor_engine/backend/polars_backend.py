# -*- coding: utf-8 -*-
"""Polars 后端：``operator_backend=auto`` + panel-native，优先 Polars kernel。"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode

from runtime.perf_config import PerfConfig

from .context import ExecutionContext
from .pandas_backend import PandasBackend
from .panel_native import finalize_panel_result


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class PolarsBackend(PandasBackend):
    """``build_backend('polars')``：继承 Pandas 计划遍历，算子层走 Polars auto 路径。

    与旧版「委托 PandasBackend 并强制 pandas_numpy」不同，本后端：
    - 保持 ``operator_backend=auto``，``OperatorRegistry.get_preferred`` 优先 polars
    - 默认 ``panel_native=True``，减少 stack/unstack
    - 可选 ``FACTOR_ENGINE_POLARS_LAZY``（预留 lazy collect）
    """

    def __init__(self, use_lazy: bool | None = None) -> None:
        super().__init__()
        self._use_lazy = use_lazy if use_lazy is not None else _env_flag(
            "FACTOR_ENGINE_POLARS_LAZY", False
        )

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        ctx = self._with_polars_perf(ctx)
        if ctx.panel_cache is None:
            ctx = replace(ctx, panel_cache={})
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["backend"] = "polars"
        ctx = replace(ctx, runtime_stats=runtime)
        return finalize_panel_result(self._eval(plan, ctx), ctx)

    @staticmethod
    def _with_polars_perf(ctx: ExecutionContext) -> ExecutionContext:
        perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
        backend_pref = getattr(perf, "operator_backend", "auto")
        if backend_pref in {"auto", "pandas_numpy"}:
            perf = replace(perf, operator_backend="auto")
        query_budget = ctx.query_budget
        if query_budget is None:
            query_budget = perf.build_query_budget()
        return replace(ctx, perf=perf, query_budget=query_budget)

    @property
    def use_lazy(self) -> bool:
        return bool(self._use_lazy)
