"""factor_engine 对外 DSL 入口（**只造表达式树，不算数**）。

本包与 ``cleaned_operators/`` 的分工
------------------------------------
- **api/**：挖掘侧写公式时用到的名字与工厂。
  ``col('close')``、``rank(x)``、``ts_mean(x, 20)`` 等调用返回 ``Expr`` / ``CleanedCall``，
  供 ``parse_expr`` 或 ``Factor(expr=...)`` 使用。
- **cleaned_operators/**：唯一 runtime；``Operator.calculate(panel_df)`` 在宽表上真正计算。
- **backend/cleaned_bridge.py**：执行时把 MultiIndex Series ↔ panel，再调 registry。

常用导入::

    from api import col, rank, ts_mean, Factor
    from api import parse_expr   # 见 dsl_parser

未在 ``__all__`` 里显式导出的算子名，可通过 ``from api import ts_corr`` 动态解析，
前提是该名已在 ``build_dsl_allowlist()`` 白名单内（canonical 或别名，且已有 runtime）。

维护约定：新增/修改算子实现请改 ``cleaned_operators/``；``api/`` 通常无需手写 kernel。
"""

from __future__ import annotations

from typing import Any, Callable

from .cleaned_ops import make_cleaned_call_factory
from .columns import col
from .factor import Factor

# 高频算子显式导出；其余算子通过 ``__getattr__`` 按白名单懒加载
rank = make_cleaned_call_factory("rank")
ts_mean = make_cleaned_call_factory("ts_mean")
ts_std = make_cleaned_call_factory("ts_std")
ts_std_dev = make_cleaned_call_factory("ts_std_dev")
zscore = make_cleaned_call_factory("zscore")
delay = make_cleaned_call_factory("delay")

__all__ = [
    "Factor",
    "col",
    "delay",
    "make_cleaned_call_factory",
    "rank",
    "ts_mean",
    "ts_std",
    "ts_std_dev",
    "zscore",
]


def __getattr__(name: str) -> Callable[..., Any]:
    """按名懒加载 cleaned 算子工厂（须在 ``OperatorRegistry`` 中已实现）。"""
    from backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    canon = OperatorRegistry._aliases.get(name, name)
    if OperatorRegistry.get(canon) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    factory = make_cleaned_call_factory(name)
    globals()[name] = factory
    return factory
