"""执行后端抽象：输入逻辑计划 + 上下文，输出因子值（通常为 MultiIndex Series）。"""

from abc import ABC, abstractmethod
from typing import Any

from planner.logical_plan import PlanNode

from .context import ExecutionContext


class Backend(ABC):
    """具体后端（如 Pandas）实现 ``execute``。"""

    @abstractmethod
    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        raise NotImplementedError
