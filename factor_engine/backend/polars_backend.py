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

线程安全（R13 P1-66/P1-67）
---------------------------
后端实例相对 lazy 模式是**无状态**的：``execute_lazy`` 不再改写
``self._use_lazy``，而是把 lazy 作为一次调用的参数传入 ``execute(..., lazy=True)``，
两个线程共享同一 backend 不会互相污染。``execute`` 启用数据源的 lazy 读
（``enable_lazy_scan`` / ``read_auto``）也会在 ``finally`` 中恢复，数据源不被
执行副作用修改。
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
from .polars_long_policy import UnsupportedCausalOperatorError
from .operator_capability import UnsupportedOperatorBackendError


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
            后端默认是否启用 lazy scan；为 ``None`` 时读取环境变量
            ``FACTOR_ENGINE_POLARS_LAZY``（默认 ``False``）。该值只是默认值，
            可被 ``execute(..., lazy=...)`` 参数覆盖，且永不被执行修改。
        """
        super().__init__()
        self._use_lazy = use_lazy if use_lazy is not None else _env_flag(
            "FACTOR_ENGINE_POLARS_LAZY", False
        )

    @staticmethod
    def _apply_lazy_scan(data_source: Any, *, restore: list) -> None:
        """启用数据源 lazy 读并注册恢复函数。

        审计 R13 P1-66：执行不得在数据源上留下可观察副作用。启用
        ``enable_lazy_scan(True)`` / ``read_auto=True`` 后，把改动前的
        ``_lazy_scan`` / ``read_auto`` 记入 ``restore``，由调用方在 ``finally``
        中还原。直接回写属性而非再次调用 ``enable_lazy_scan``，避免其
        ``clear_cache()`` 副作用。

        ``ExecutionContext`` 会用一个 LQTP 包装器包住真实数据源
        （``__getattr__`` 委托到 ``inner``）。只回写包装器会产生 shadow 属性而
        ``inner`` 仍保持 lazy——因此同时回写 ``inner``，保证后续 normal run 的
        读语义与全新后端一致。
        """
        enable = getattr(data_source, "enable_lazy_scan", None)
        if callable(enable):
            prev_lazy = getattr(data_source, "_lazy_scan", None)
            prev_auto = getattr(data_source, "read_auto", None)
            enable(True)
            inner = getattr(data_source, "inner", None) or getattr(
                data_source, "_inner", None
            )

            def _restore_lazy_source() -> None:
                if isinstance(prev_lazy, bool):
                    try:
                        data_source._lazy_scan = prev_lazy
                    except Exception:
                        pass
                    if inner is not None:
                        try:
                            inner._lazy_scan = prev_lazy
                        except Exception:
                            pass
                if isinstance(prev_auto, bool):
                    try:
                        data_source.read_auto = prev_auto
                    except Exception:
                        pass
                    if inner is not None:
                        try:
                            inner.read_auto = prev_auto
                        except Exception:
                            pass

            restore.append(_restore_lazy_source)
        elif hasattr(data_source, "read_auto"):
            prev_auto = getattr(data_source, "read_auto", None)
            data_source.read_auto = True
            if isinstance(prev_auto, bool):

                def _restore_read_auto() -> None:
                    data_source.read_auto = prev_auto

                restore.append(_restore_read_auto)

    def execute(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        lazy: bool | None = None,
    ) -> Any:
        """执行逻辑计划，优先 Polars 算子与可选表达式编译路径。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文；会被注入 Polars 性能配置与运行时标签。
        lazy : bool | None
            本次调用的 lazy scan 开关；``None`` 时使用后端默认
            ``self._use_lazy``。该参数是 per-call 的，不修改任何实例状态
            （R13 P1-67）。

        返回
        ----
        Any
            经 ``finalize_panel_result`` 对齐后的因子结果。

        说明
        ----
        若 ``FACTOR_ENGINE_POLARS_EXPR=1`` 且计划具备 Polars 表达式能力，
        优先走 ``execute_polars_expr_plan``；失败时回退到逐节点 kernel 路径。
        启用 lazy 读时对数据源的改动会在 ``finally`` 中恢复（R13 P1-66）。
        """
        root_ctx = ctx
        ctx = self._with_polars_perf(ctx)
        use_lazy = self._use_lazy if lazy is None else bool(lazy)
        _restore: list = []
        try:
            if use_lazy and ctx.data_source is not None:
                self._apply_lazy_scan(ctx.data_source, restore=_restore)
            if ctx.panel_cache is None:
                ctx = replace(ctx, panel_cache={})
            runtime = dict(getattr(ctx, "runtime_stats", None) or {})
            runtime["backend"] = "polars"
            if use_lazy:
                runtime["lazy_scan"] = True
            ctx = replace(ctx, runtime_stats=runtime, prefer_polars_panel=True)

            # R31-P1-034 (POLARS_NATIVE_AUTO_PLANNED)：Polars native 整树编译是
            # 默认自动路径，不再靠运维手工开关。env 仅用于 debug force-disable
            # （``FACTOR_ENGINE_POLARS_EXPR=0``）。能力不足/异常时按既有逻辑诚实
            # 回退逐节点。
            _polars_expr_auto = os.environ.get("FACTOR_ENGINE_POLARS_EXPR", "").strip().lower() not in {"0", "false", "off"}
            if _polars_expr_auto and ctx.data_source is not None:
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
                    except (UnsupportedCausalOperatorError, UnsupportedOperatorBackendError) as exc:
                        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
                        runtime["polars_expr_fallback"] = True
                        runtime["polars_expr_fallback_reason"] = str(exc)
                        ctx.runtime_stats = runtime  # type: ignore[attr-defined]
                        root_ctx.runtime_stats = runtime  # type: ignore[attr-defined]

            result = finalize_panel_result(self._eval(plan, ctx), ctx)
            from runtime.production_policy import assert_no_production_pandas_fallbacks

            assert_no_production_pandas_fallbacks(ctx, context="polars_execute")
            return result
        finally:
            for _fn in _restore:
                try:
                    _fn()
                except Exception:
                    pass

    def execute_lazy(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """显式 lazy scan 入口（等价于 ``lazy=True`` 的 ``execute``）。

        与旧实现不同，本方法**不修改** ``self._use_lazy``：lazy 模式作为
        per-call 参数传入 ``execute``，线程共享同一 backend 时互不干扰
        （R13 P1-67）。

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
        return self.execute(plan, ctx, lazy=True)

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
            ``operator_backend`` 为 ``auto``/``None`` 时保持 ``auto``；**显式**
            ``pandas_numpy`` / ``polars`` 偏好被保留（R13 P2-68），并确保
            ``query_budget`` 已构建的副本。
        """
        perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
        backend_pref = getattr(perf, "operator_backend", "auto")
        if backend_pref in {None, "auto"}:
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
            后端默认 lazy scan 开关状态（执行期可被 per-call 参数覆盖）。
        """
        return bool(self._use_lazy)
