"""``CleanedCall``：DSL 算子调用的通用 Expr 节点。

背景
----
历史上部分算子在 ``expr/*`` 有强类型节点（如 ``ColumnRef``）；绝大多数 cleaned 算子
只有 runtime 实现，解析阶段统一落成 ``CleanedCall(op, args, kwargs)``。

下游
----
- ``ir/analyzer``：遍历 ``CleanedCall``，推导 lookback、引用列、是否含 ts/cs 算子；
- ``planner/lowerer``：lower 为 ``PlanNode(op=..., inputs=..., attrs=kwargs)``；
- ``backend/cleaned_bridge``：按 ``op`` 查 ``OperatorRegistry`` 并 ``calculate``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .base import Expr


@dataclass(frozen=True)
class CleanedCall(Expr):
    """一次 cleaned 算子调用；``op`` 可为 canonical 或别名（执行前会再解析）。"""

    op: str
    args: Tuple[Expr, ...] = ()
    kwargs: Tuple[Tuple[str, Any], ...] = ()

    def children(self) -> Tuple[Expr, ...]:
        """子表达式列表（即 ``args``）。"""
        return self.args

    def kwargs_dict(self) -> Mapping[str, Any]:
        """把 ``kwargs`` 元组转为普通 dict，供 analyzer 读取。"""
        return dict(self.kwargs)
