"""Pandas 执行后端：逻辑计划 → MultiIndex Series 结果。

本后端是因子引擎的基准 Python 执行路径，按 ``PlanNode.op`` 分派到
``KernelRegistry`` 中注册的 kernel，递归求值整棵逻辑计划树。

注册策略
--------
- ``column`` / ``literal``：内建 kernel，从 ``ExecutionContext.data_source`` 取列或常量；
- **其余所有** ``PlanNode.op``：启动时由 ``list_cleaned_ops_for_backend`` 批量注册为
  ``make_cleaned_kernel``，100% 走 ``cleaned_operators``，无第二套逐算子 kernel 文件。

缓存
----
若 ``ctx.cache`` 存在，按 ``plan_cache_key(node)`` memoize 子计划结果，
避免同一子树在 DAG 中重复计算。

特殊节点
--------
- ``plan_ref``：从 ``shared_result_cache`` 读取 CSE 预计算结果；
- ``materialized_series``：从 SQL partial pushdown 物化缓存读取。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode
from planner.plan_hash import plan_cache_key

from runtime.perf_config import PerfConfig

from .base import Backend
from .cleaned_bridge import list_cleaned_ops_for_backend, make_cleaned_kernel
from .context import ExecutionContext
from .kernels import KernelRegistry
from .panel_native import finalize_panel_result, load_column_as_panel, panel_native_enabled


class PandasBackend(Backend):
    """Pandas/Numpy 基准执行后端。

    按 ``PlanNode.op`` 分派 kernel；除 ``column`` / ``literal`` / ``plan_ref`` /
    ``materialized_series`` 外，其余算子均通过 ``cleaned_operators`` 桥接执行。
    """

    def __init__(self) -> None:
        """初始化 kernel 注册表并批量注册内建与 cleaned 算子。"""
        self._registry = KernelRegistry()
        self._register_kernels()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """执行逻辑计划并返回最终因子结果。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文，含数据源、缓存与列名配置。

        返回
        ----
        Any
            经 ``finalize_panel_result`` 对齐后的因子值，
            通常为 ``(timestamp, instrument)`` MultiIndex Series。
        """
        ctx = self._with_pandas_perf(ctx)
        return finalize_panel_result(self._eval(plan, ctx), ctx)

    @staticmethod
    def _with_pandas_perf(ctx: ExecutionContext) -> ExecutionContext:
        """为 Pandas 后端注入性能配置，强制算子层走 ``pandas_numpy``。

        参数
        ----
        ctx : ExecutionContext
            原始执行上下文。

        返回
        ----
        ExecutionContext
            ``perf.operator_backend`` 设为 ``pandas_numpy`` 的新上下文副本。
        """
        perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
        if getattr(perf, "operator_backend", "auto") == "auto":
            perf = replace(perf, operator_backend="pandas_numpy")
        return replace(ctx, perf=perf)

    def _eval(self, node: PlanNode, ctx: ExecutionContext) -> Any:
        """递归求值单个计划节点（含缓存、CSE 与 SQL 物化快捷路径）。

        参数
        ----
        node : PlanNode
            当前待求值的逻辑计划节点。
        ctx : ExecutionContext
            执行期上下文。

        返回
        ----
        Any
            节点求值结果，类型取决于算子（Series、标量、DataFrame 等）。

        异常
        ----
        RuntimeError
            ``plan_ref`` 节点但 ``shared_result_cache`` 未配置。
        KeyError
            CSE 或 SQL 物化缓存中缺少对应 ``sid``。
        NotImplementedError
            算子名未在 ``KernelRegistry`` 中注册。
        """
        if node.op == "plan_ref":
            sc = getattr(ctx, "shared_result_cache", None)
            if sc is None:
                raise RuntimeError(
                    "plan_ref requires ExecutionContext.shared_result_cache; "
                    "use FactorEngine.run_many() after compile_many CSE."
                )
            sid = node.attrs["sid"]
            if sid not in sc:
                raise KeyError(f"missing shared subplan result for sid prefix={sid[:64]!r}...")
            return sc[sid]

        if node.op == "materialized_series":
            sid = str(node.attrs["sid"])
            mat = getattr(ctx, "materialized_series", None) or {}
            if sid not in mat:
                raise KeyError(f"missing SQL materialized_series for sid={sid[:64]!r}...")
            return mat[sid]

        cache = getattr(ctx, "cache", None)
        cache_key: str | None = None
        if cache is not None:
            cache_key = plan_cache_key(node)
            hit = cache.get(cache_key)
            if hit is not None:
                return hit

        try:
            kernel = self._registry.get(node.op)
        except KeyError as exc:
            from cleaned_operators.registry import OperatorRegistry
            try:
                resolved = "where" if node.op == "if_else" else OperatorRegistry.resolve_canonical(node.op)
                kernel = self._registry.get(resolved)
            except (KeyError, ValueError):
                raise NotImplementedError(f"Unsupported op: {node.op}") from exc

        out = kernel(node, ctx)
        if cache is not None and cache_key is not None:
            cache.set(cache_key, out)
        return out

    def _register_kernels(self) -> None:
        """向注册表填入 ``column``/``literal`` 内建 kernel 及全部 cleaned 算子。"""
        reg = self._registry.register
        reg("column", self._op_column)
        reg("literal", self._op_literal)
        for op in list_cleaned_ops_for_backend(set()):
            reg(op, make_cleaned_kernel(self._eval, op))
        # Compatibility aliases resolve to the now-registered canonical kernel.
        from cleaned_operators.registry import OperatorRegistry
        for alias, canonical in OperatorRegistry._aliases.items():
            if canonical in self._registry._kernels:
                reg(alias, self._registry._kernels[canonical])

    def _op_column(self, node: PlanNode, ctx: ExecutionContext):
        """``column`` 节点 kernel：从数据源加载指定列。

        参数
        ----
        node : PlanNode
            属性 ``name`` 为列名。
        ctx : ExecutionContext
            执行期上下文，提供 ``data_source``。

        返回
        ----
        Any
            列数据；panel-native 模式下为宽表 DataFrame，否则为 MultiIndex Series。
        """
        name = node.attrs["name"]
        if panel_native_enabled(ctx):
            return load_column_as_panel(ctx.data_source, name, ctx)
        return ctx.data_source.load_column(name)

    def _op_literal(self, node: PlanNode, ctx: ExecutionContext):
        """``literal`` 节点 kernel：返回节点属性中的常量值。

        参数
        ----
        node : PlanNode
            属性 ``value`` 为字面量。
        ctx : ExecutionContext
            未使用，保留以符合 kernel 签名。

        返回
        ----
        Any
            ``node.attrs['value']`` 中的常量。
        """
        _ = ctx
        return node.attrs["value"]
