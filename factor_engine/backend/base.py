"""执行后端抽象：输入逻辑计划 + 上下文，输出因子值（通常为 MultiIndex Series）。"""

from abc import ABC, abstractmethod
from typing import Any

from factor_engine.planner.logical_plan import PlanNode

from .context import ExecutionContext


class Backend(ABC):
    """所有执行后端的基类。

    子类实现 ``execute(plan, ctx)``，递归求值 ``PlanNode`` 树，
    从 ``ctx.data_source`` 拉列，最终返回 ``pd.Series``（MultiIndex: timestamp × instrument）。
    """

    #: 为 True 时 ``FactorEngine.run`` 跳过列 prefetch（后端用 scan_polars_long 等原生读）
    runtime_backend_label: str = "pandas_numpy"
    prefers_native_scan: bool = False
    #: 为 True 时 ``run_many`` 共享子树可仅编译 LazyFrame，defer collect
    supports_lazy_shared: bool = False

    @abstractmethod
    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """求值整棵计划树。

        参数：
            plan: 编译后的根 PlanNode（通常来自 planner.lowerer）
            ctx: 数据源、各级缓存、perf 提示

        返回：
            一般为 ``pd.Series``；long/polars 路径可能中间为 LazyFrame，由子类 collect。
        """
        raise NotImplementedError

    def compile_lazy_shared(
        self,
        plan: PlanNode,
        ctx: ExecutionContext,
        *,
        sid: str,
    ) -> bool:
        """CSE 共享子树：仅编译 long LazyFrame 写入 ``ctx.shared_long_lazy_cache``。

        参数：
            sid: CSE 分配的子树 id（plan_ref 节点 attrs 里）

        返回：
            True 表示已成功编译并缓存；False 表示本后端不支持，需 fallback 到 eager execute。
        """
        return False
