# -*- coding: utf-8 -*-
"""混合执行后端：SQL 下推 + Polars 算子层。

``HybridBackend`` 是 ``build_backend('auto')`` / ``build_backend('hybrid')`` 的入口，
内部委托 ``SqlBackend`` 完成 SQL 子树预计算后，以 Polars auto 路径执行剩余计划节点。
"""
from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .sql_backend import SqlBackend


class HybridBackend(Backend):
    """混合 SQL + Polars 执行后端。

    执行流程：逻辑计划物理化 → 可下推子树走 DuckDB/ClickHouse SQL →
    物化结果注入 ``ExecutionContext`` → Polars/Pandas 算子层完成根节点求值。
    """

    def __init__(self) -> None:
        """初始化内部 ``SqlBackend``，算子后端设为 ``auto``（优先 Polars）。"""
        self._sql = SqlBackend(operator_backend="auto")

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """执行逻辑计划：SQL 下推 + Polars 算子混合路径。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文，含数据源、缓存与运行时统计。

        返回
        ----
        Any
            因子计算结果，通常为 ``(timestamp, instrument)`` MultiIndex Series。
        """
        return self._sql.execute(plan, ctx)
