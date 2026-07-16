"""常量叶子节点：公式中的数字、布尔等字面量。"""

from dataclasses import dataclass

from .base import Expr


@dataclass(frozen=True)
class Literal(Expr):
    """不参与数据拉取，执行时直接作为常数值参与运算。"""

    value: object
