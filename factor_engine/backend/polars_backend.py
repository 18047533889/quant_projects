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
        root_ctx = ctx
        ctx = self._with_polars_perf(ctx)
        if self._use_lazy and ctx.data_source is not None:
            enable = getattr(ctx.data_source, "enable_lazy_scan", None)
            if callable(enable):
                enable(True)
            elif hasattr(ctx.data_source, "read_auto"):
                ctx.data_source.read_auto = True
        if ctx.panel_cache is None:
            ctx = replace(ctx, panel_cache={})
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["backend"] = "polars"
        if self._use_lazy:
            runtime["lazy_scan"] = True
        ctx = replace(ctx, runtime_stats=runtime, prefer_polars_panel=True)

        if _env_flag("FACTOR_ENGINE_POLARS_EXPR") and ctx.data_source is not None:
            from .polars_expr_backend import execute_polars_expr_plan, plan_is_polars_expr_capable

            if plan_is_polars_expr_capable(plan):
                try:
                    result = execute_polars_expr_plan(plan, ctx)
                    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
                    runtime["polars_expr"] = True
                    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
                    root_ctx.runtime_stats = runtime  # type: ignore[attr-defined]
                    result = finalize_panel_result(result, ctx)
                    from runtime.production_policy import assert_no_production_pandas_fallbacks

                    assert_no_production_pandas_fallbacks(ctx, context="polars_execute")
                    return result
                except Exception:
                    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
                    runtime["polars_expr_fallback"] = True
                    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
                    root_ctx.runtime_stats = runtime  # type: ignore[attr-defined]

        result = finalize_panel_result(self._eval(plan, ctx), ctx)
        from runtime.production_policy import assert_no_production_pandas_fallbacks

        assert_no_production_pandas_fallbacks(ctx, context="polars_execute")
        return result

    def execute_lazy(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """显式 lazy scan 入口（等价于 ``use_lazy=True`` 的 ``execute``）。"""
        prev = self._use_lazy
        self._use_lazy = True
        try:
            return self.execute(plan, ctx)
        finally:
            self._use_lazy = prev

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
