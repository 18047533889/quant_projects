"""因子引擎计划层公共导出：逻辑计划节点、IR 下降器与轻量优化器。"""

from .logical_plan import PlanNode
from .lowerer import Lowerer
from .optimizer import Optimizer
from .physical_lowerer import (
    compile_many_chunked,
    validate_expression_depth,
    MAX_DAG_WIDTH,
    MAX_EXPRESSION_DEPTH,
)

__all__ = [
    "PlanNode",
    "Lowerer",
    "Optimizer",
    "compile_many_chunked",
    "validate_expression_depth",
    "MAX_DAG_WIDTH",
    "MAX_EXPRESSION_DEPTH",
]
