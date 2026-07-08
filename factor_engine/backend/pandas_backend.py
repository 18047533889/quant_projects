"""Pandas 执行后端：逻辑计划 → MultiIndex Series 结果。

注册策略
--------
- ``column`` / ``literal``：内建 kernel，从 ``ExecutionContext.data_source`` 取列或常量；
- **其余所有** ``PlanNode.op``：启动时由 ``list_cleaned_ops_for_backend`` 批量注册为
  ``make_cleaned_kernel``，100% 走 ``cleaned_operators``，无第二套逐算子 kernel 文件。

缓存：若 ``ctx.cache`` 存在，按 ``plan_cache_key(node)``  memoize 子计划结果。
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
    """按 ``PlanNode.op`` 分派 kernel；除 ``column`` / ``literal`` / ``plan_ref`` 外均为 cleaned。"""

    def __init__(self) -> None:
        self._registry = KernelRegistry()
        self._register_kernels()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        ctx = self._with_pandas_perf(ctx)
        return finalize_panel_result(self._eval(plan, ctx), ctx)

    @staticmethod
    def _with_pandas_perf(ctx: ExecutionContext) -> ExecutionContext:
        perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
        if getattr(perf, "operator_backend", "auto") == "auto":
            perf = replace(perf, operator_backend="pandas_numpy")
        return replace(ctx, perf=perf)

    def _eval(self, node: PlanNode, ctx: ExecutionContext) -> Any:
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
            raise NotImplementedError(f"Unsupported op: {node.op}") from exc

        out = kernel(node, ctx)
        if cache is not None and cache_key is not None:
            cache.set(cache_key, out)
        return out

    def _register_kernels(self) -> None:
        reg = self._registry.register
        reg("column", self._op_column)
        reg("literal", self._op_literal)
        for op in list_cleaned_ops_for_backend(set()):
            reg(op, make_cleaned_kernel(self._eval, op))

    def _op_column(self, node: PlanNode, ctx: ExecutionContext):
        name = node.attrs["name"]
        if panel_native_enabled(ctx):
            return load_column_as_panel(ctx.data_source, name, ctx)
        return ctx.data_source.load_column(name)

    def _op_literal(self, node: PlanNode, ctx: ExecutionContext):
        _ = ctx
        return node.attrs["value"]
