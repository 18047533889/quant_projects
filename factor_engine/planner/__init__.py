"""因子引擎计划层公共导出：逻辑计划节点、IR 下降器与轻量优化器。"""

from .logical_plan import DuplicateLogicalNodeIdentityError, PlanNode
from .lowerer import Lowerer
from .optimizer import Optimizer
from .physical_lowerer import (
    compile_many_chunked,
    validate_expression_depth,
)

# 向后兼容：导出自适应常量（延迟求值）
# 注意：这些不再是常量，而是函数调用，但兼容 `from planner import MAX_DAG_WIDTH` 语法
def __getattr__(name: str):
    """延迟求值自适应常量。"""
    if name == "MAX_DAG_WIDTH":
        from .physical_lowerer import _get_adaptive_dag_width_limit
        return _get_adaptive_dag_width_limit()
    elif name == "MAX_EXPRESSION_DEPTH":
        from .physical_lowerer import _get_expression_depth_limit
        return _get_expression_depth_limit()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PlanNode",
    "DuplicateLogicalNodeIdentityError",
    "Lowerer",
    "Optimizer",
    "compile_many_chunked",
    "validate_expression_depth",
    "MAX_DAG_WIDTH",  # 延迟求值，自适应
    "MAX_EXPRESSION_DEPTH",  # 延迟求值，基于 sys.getrecursionlimit()
]
