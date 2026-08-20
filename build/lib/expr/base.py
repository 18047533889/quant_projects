"""
因子表达式基类。

所有公式节点为 ``ColumnRef`` / ``Literal`` / ``CleanedCall``（来自 ``cleaned_operators``）。
四则运算重载为 ``add`` / ``subtract`` / ``multiply`` / ``divide`` 调用。

设计说明
--------
- 不在 ``Expr`` 层做数值计算，只构建 AST；
- ``+`` ``-`` ``*`` ``/`` 统一落成 ``CleanedCall(op=...)``，与 DSL 算子同一套 lower 路径；
- ``ensure_expr`` 把 Python 标量自动包成 ``Literal``，方便 ``close + 1`` 写法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Expr:
    """表达式抽象基类；子类必须实现 ``children()`` 供遍历。"""

    def children(self) -> Tuple["Expr", ...]:
        """直接子节点（叶子节点返回空元组）。"""
        return ()

    def _binop(self, op: str, other: Any) -> "Expr":
        """构造二元 ``CleanedCall``；``other`` 会先经 ``ensure_expr`` 规范化。"""
        from .cleaned_call import CleanedCall

        return CleanedCall(op=op, args=(self, ensure_expr(other)))

    def __add__(self, other: Any) -> "Expr":
        """表达式加法 → ``CleanedCall(op='add')``。"""
        return self._binop("add", other)

    def __radd__(self, other: Any) -> "Expr":
        """右加法 ``scalar + expr``。"""
        return ensure_expr(other)._binop("add", self)

    def __sub__(self, other: Any) -> "Expr":
        """表达式减法。"""
        return self._binop("subtract", other)

    def __rsub__(self, other: Any) -> "Expr":
        """右减法 ``scalar - expr``。"""
        return ensure_expr(other)._binop("subtract", self)

    def __mul__(self, other: Any) -> "Expr":
        """表达式乘法。"""
        return self._binop("multiply", other)

    def __rmul__(self, other: Any) -> "Expr":
        """右乘法。"""
        return ensure_expr(other)._binop("multiply", self)

    def __truediv__(self, other: Any) -> "Expr":
        """表达式真除法。"""
        return self._binop("divide", other)

    def __rtruediv__(self, other: Any) -> "Expr":
        """右除法 ``scalar / expr``。"""
        return ensure_expr(other)._binop("divide", self)

    def __neg__(self) -> "Expr":
        """一元取负 → ``CleanedCall(op='neg')``。"""
        from .cleaned_call import CleanedCall

        return CleanedCall(op="neg", args=(self,))


def ensure_expr(x: Any) -> Expr:
    """把标量或已是 ``Expr`` 的对象统一成 ``Expr``。

    - ``Expr`` 实例：原样返回
    - ``int`` / ``float`` / 其他：包成 ``Literal(x)``
    """
    from .literal import Literal

    if isinstance(x, Expr):
        return x
    return Literal(x)
