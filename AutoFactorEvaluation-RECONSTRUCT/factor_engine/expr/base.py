"""
因子表达式基类。

所有公式节点为 ``ColumnRef`` / ``Literal`` / ``CleanedCall``（来自 ``cleaned_operators``）。
四则运算重载为 ``add`` / ``subtract`` / ``multiply`` / ``divide`` 调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Expr:
    """表达式抽象基类。"""

    def children(self) -> Tuple["Expr", ...]:
        return ()

    def _binop(self, op: str, other: Any) -> "Expr":
        from .cleaned_call import CleanedCall

        return CleanedCall(op=op, args=(self, ensure_expr(other)))

    def __add__(self, other: Any) -> "Expr":
        return self._binop("add", other)

    def __radd__(self, other: Any) -> "Expr":
        return ensure_expr(other)._binop("add", self)

    def __sub__(self, other: Any) -> "Expr":
        return self._binop("subtract", other)

    def __rsub__(self, other: Any) -> "Expr":
        return ensure_expr(other)._binop("subtract", self)

    def __mul__(self, other: Any) -> "Expr":
        return self._binop("multiply", other)

    def __rmul__(self, other: Any) -> "Expr":
        return ensure_expr(other)._binop("multiply", self)

    def __truediv__(self, other: Any) -> "Expr":
        return self._binop("divide", other)

    def __rtruediv__(self, other: Any) -> "Expr":
        return ensure_expr(other)._binop("divide", self)

    def __neg__(self) -> "Expr":
        from .cleaned_call import CleanedCall

        return CleanedCall(op="neg", args=(self,))


def ensure_expr(x: Any) -> Expr:
    """标量或已是 Expr 时统一成 Expr。"""
    from .literal import Literal

    if isinstance(x, Expr):
        return x
    return Literal(x)
