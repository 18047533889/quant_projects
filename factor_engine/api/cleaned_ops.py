"""``CleanedCall`` 工厂：把 DSL 函数调用变成表达式节点。

``make_cleaned_call_factory('ts_mean')`` 返回的可调用对象 **不执行计算**，
仅构造 ``CleanedCall(op='ts_mean', args=..., kwargs=...)``。
真正求值发生在 ``PandasBackend`` → ``cleaned_bridge.make_cleaned_kernel`` →
``OperatorRegistry.get(...).calculate(panel)``。

参数规则
--------
- 子表达式（``col(...)``、嵌套算子）经 ``ensure_expr`` 包装为 ``Expr``。
- 字面量 ``int/float/str/bool/None`` 原样保留，供 planner 下推为 ``literal`` 节点。
"""

from __future__ import annotations

from typing import Any, Callable

from factor_engine.expr.base import ensure_expr
from factor_engine.expr.cleaned_call import CleanedCall


def make_cleaned_call_factory(op_name: str) -> Callable[..., CleanedCall]:
    """为算子名 ``op_name`` 生成 DSL 工厂，例如 ``SMA(col('close'), 20)``。

    Parameters
    ----------
    op_name : str
        cleaned 算子 canonical 名或别名（须在 ``OperatorRegistry`` 中已实现）。

    Returns
    -------
    Callable[..., CleanedCall]
        调用后返回 ``CleanedCall(op=op_name, ...)``，不执行数值计算。
    """
    def _factory(*args: Any, **kwargs: Any) -> CleanedCall:
        """构造 ``CleanedCall`` 节点；子表达式经 ``ensure_expr`` 包装。"""
        kw_pairs = tuple(
            (k, v if isinstance(v, (int, float, str, bool)) or v is None else ensure_expr(v))
            for k, v in kwargs.items()
        )
        return CleanedCall(
            op=op_name,
            args=tuple(ensure_expr(a) for a in args),
            kwargs=kw_pairs,
        )

    _factory.__name__ = op_name
    return _factory
