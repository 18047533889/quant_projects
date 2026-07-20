# -*- coding: utf-8 -*-
"""混合执行后端：SQL 下推 + Polars 算子层。

``HybridBackend`` 是 ``build_backend('auto')`` / ``build_backend('hybrid')`` 的入口，
内部委托 ``SqlBackend`` 完成 SQL 子树预计算后，以 Polars auto 路径执行剩余计划节点。

当数据源提供 ``scan_polars_long`` 时，``auto`` 升级为 long hybrid（与
``auto_long`` 相同）：SQL 子树物化为 long LazyFrame，剩余节点走原生 Polars long，
避免 wide-panel unpivot 往返。无 long scan 时保持宽表 hybrid。
"""
from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext
from .sql_backend import SqlBackend


def _supports_polars_long_scan(ctx: ExecutionContext) -> bool:
    ds = getattr(ctx, "data_source", None)
    scan = getattr(ds, "scan_polars_long", None)
    return callable(scan)


class HybridBackend(Backend):
    """混合 SQL + Polars 执行后端。

    执行流程：逻辑计划物理化 → 可下推子树走 DuckDB/ClickHouse SQL →
    物化结果注入 ``ExecutionContext`` → Polars/Pandas 算子层完成根节点求值。
    """

    runtime_backend_label = "hybrid"

    def __init__(self) -> None:
        """初始化内部 ``SqlBackend``，算子后端设为 ``auto``（优先 Polars）。"""
        self._sql = SqlBackend(operator_backend="auto")
        self._long: Backend | None = None

    def _long_backend(self) -> Backend:
        if self._long is None:
            from .hybrid_long_backend import HybridLongBackend

            self._long = HybridLongBackend()
        return self._long

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
        if _supports_polars_long_scan(ctx):
            return self._long_backend().execute(plan, ctx)
        return self._sql.execute(plan, ctx)
