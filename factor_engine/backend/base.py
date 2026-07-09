"""执行后端抽象：输入逻辑计划 + 上下文，输出因子值（通常为 MultiIndex Series）。"""

from abc import ABC, abstractmethod
from typing import Any

from planner.logical_plan import PlanNode

from .context import ExecutionContext


class Backend(ABC):
    """具体后端（如 Pandas）实现 ``execute``。"""

    #: 为 True 时 ``FactorEngine.run`` 跳过列 prefetch（由 ``scan_polars_long`` 等原生读）
    prefers_native_scan: bool = False
    #: 为 True 时 ``run_many`` 共享子树可仅编译 LazyFrame，defer collect
    supports_lazy_shared: bool = False

    @abstractmethod
    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        raise NotImplementedError

    def compile_lazy_shared(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        """CSE 共享子树：仅编译 long LazyFrame 写入 ``shared_long_lazy_cache``。"""
        return False
