# -*- coding: utf-8 -*-
"""Polars 执行后端：继承 Pandas 计划遍历，算子层优先 Polars。

``PolarsBackend`` 是 ``build_backend('polars')`` 的入口，在 ``PandasBackend``
的递归求值框架上叠加 Polars 性能配置与可选表达式编译快路径。

特性
----
- 保持 ``operator_backend=auto``，``OperatorRegistry.get_preferred`` 优先 polars；
- 默认 ``panel_native=True``，减少 stack/unstack 往返；
- 可选 ``FACTOR_ENGINE_POLARS_LAZY`` 启用 lazy scan（预留 lazy collect）；
- 可选 ``FACTOR_ENGINE_POLARS_EXPR`` 尝试整树 Polars 表达式编译。
"""
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
    """读取环境变量并解析为布尔开关。

    参数
    ----
    name : str
        环境变量名。
    default : bool
        变量未设置或为空时的默认值。

    返回
    ----
    bool
        变量值为 ``1``/``true``/``yes``/``on``（不区分大小写）时为 ``True``，否则 ``False``。
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class PolarsBackend(PandasBackend):
    """Polars 优先执行后端。

    与旧版「委托 PandasBackend 并强制 pandas_numpy」不同，本后端：
    - 保持 ``operator_backend=auto``，``OperatorRegistry.get_preferred`` 优先 polars；
    - 默认 ``panel_native=True``，减少 stack/unstack；
    - 可选 ``FACTOR_ENGINE_POLARS_LAZY``（预留 lazy collect）；
    - 可选 ``FACTOR_ENGINE_POLARS_EXPR`` 整树表达式编译快路径。
    """

    def __init__(self, use_lazy: bool | None = None) -> None:
        """初始化 Polars 后端。

        参数
        ----
        use_lazy : bool | None
            是否启用 lazy scan；为 ``None`` 时读取环境变量
            ``FACTOR_ENGINE_POLARS_LAZY``（默认 ``False``）。
        """
        super().__init__()
        self._use_lazy = use_lazy if use_lazy is not None else _env_flag(
            "FACTOR_ENGINE_POLARS_LAZY", False
        )

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """执行逻辑计划，优先 Polars 算子与可选表达式编译路径。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文；会被注入 Polars 性能配置与运行时标签。

        返回
        ----
        Any
            经 ``finalize_panel_result`` 对齐后的因子结果。

        说明
        ----
        若 ``FACTOR_ENGINE_POLARS_EXPR=1`` 且计划具备 Polars 表达式能力，
        优先走 ``execute_polars_expr_plan``；失败时回退到逐节点 kernel 路径。
        """
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
        """显式 lazy scan 入口（等价于 ``use_lazy=True`` 的 ``execute``）。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文。

        返回
        ----
        Any
            因子计算结果，与 ``execute`` 相同。
        """
        prev = self._use_lazy
        self._use_lazy = True
        try:
            return self.execute(plan, ctx)
        finally:
            self._use_lazy = prev

    @staticmethod
    def _with_polars_perf(ctx: ExecutionContext) -> ExecutionContext:
        """为 Polars 后端注入性能配置与查询预算。

        参数
        ----
        ctx : ExecutionContext
            原始执行上下文。

        返回
        ----
        ExecutionContext
            ``operator_backend`` 保持 ``auto``，并确保 ``query_budget`` 已构建的副本。
        """
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
        """是否启用 lazy scan 模式。

        返回
        ----
        bool
            当前 lazy scan 开关状态。
        """
        return bool(self._use_lazy)
