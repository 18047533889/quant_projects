# -*- coding: utf-8
"""混合 long-table 执行后端：SQL 下推 + Polars long native。

``HybridLongBackend`` 是 ``build_backend('auto_long')`` / ``build_backend('hybrid_long')``
的入口，在 ``SqlBackend`` 的 SQL 预计算框架上，将剩余节点委托给
``PolarsLongBackend`` 以 long-table 原生路径执行，避免 wide panel 往返。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from planner.logical_plan import PlanNode

from .context import ExecutionContext
from .polars_long_backend import PolarsLongBackend
from .sql_backend import SqlBackend


class HybridLongBackend(SqlBackend):
    """混合 SQL + Polars long-table 原生执行后端。

    执行流程：逻辑计划物理化 → SQL 子树优先物化为 long LazyFrame →
    ``PolarsLongBackend`` 以 ``scan_polars_long`` + 表达式发射器一次 collect。
    支持 CSE 共享子树的 lazy 编译（``compile_lazy_shared``）。
    """

    runtime_backend_label = "hybrid_long"

    prefers_native_scan = True
    supports_lazy_shared = True

    def __init__(self) -> None:
        """初始化 SQL 下推后端与 Polars long 回退执行器。"""
        super().__init__(operator_backend="auto")
        self._polars_long = PolarsLongBackend()

    def compile_lazy_shared(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        """CSE 共享子树：编译 LazyFrame 写入 ``shared_long_lazy_cache``，不 collect。

        参数
        ----
        plan : PlanNode
            共享子树的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文，需提供 ``data_source.scan_polars_long``。
        sid : str
            共享子树标识符，用作 lazy 缓存键。

        返回
        ----
        bool
            编译成功写入缓存为 ``True``，不支持或失败为 ``False``。
        """
        return self._polars_long.compile_lazy_shared(plan, ctx, sid=sid)

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """执行逻辑计划：SQL 下推优先物化 long LazyFrame，再 Polars long 求值。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文；会设置 ``materialize_sql_as_long_lazy=True``。

        返回
        ----
        Any
            因子计算结果，通常为 MultiIndex Series。
        """
        ctx = replace(ctx, materialize_sql_as_long_lazy=True)
        return super().execute(plan, ctx)

    def _eval_hybrid(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """SQL 预计算完成后，以 Polars long 或 Pandas 路径求值根节点。

        参数
        ----
        plan : PlanNode
            物理化后的根计划节点。
        ctx : ExecutionContext
            已注入物化结果的执行上下文。

        返回
        ----
        Any
            根节点求值结果。
        """
        if self._operator_backend == "pandas_numpy":
            return self._python.execute(plan, ctx)
        return self._polars_long.execute(plan, ctx)

    def _eval_python(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """SQL 下推失败时，以 Polars long 或 Pandas 路径回退执行子树。

        参数
        ----
        plan : PlanNode
            无法 SQL 下推的子计划节点。
        ctx : ExecutionContext
            执行期上下文。

        返回
        ----
        Any
            子树求值结果。
        """
        if self._operator_backend == "pandas_numpy":
            return self._python.execute(plan, ctx)
        return self._polars_long.execute(plan, ctx)
