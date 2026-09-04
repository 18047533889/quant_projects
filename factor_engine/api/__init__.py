"""factor_engine 对外 DSL 入口（**只造表达式树，不算数**）。

本包与 ``cleaned_operators/`` 的分工
------------------------------------
- **api/**：挖掘侧写公式时用到的名字与工厂。
  ``col('close')``、``rank(x)``、``ts_mean(x, 20)`` 等调用返回 ``Expr`` / ``CleanedCall``，
  供 ``parse_expr`` 或 ``Factor(expr=...)`` 使用。
- **cleaned_operators/**：唯一 runtime；``Operator.calculate(panel_df)`` 在宽表上真正计算。
- **backend/cleaned_bridge.py**：执行时把 MultiIndex Series ↔ panel，再调 registry。

常用导入::

    from factor_engine.api import col, rank, ts_mean, Factor
    from factor_engine.api import parse_expr   # 见 dsl_parser

未在 ``__all__`` 里显式导出的算子名，可通过 ``from api import ts_corr`` 动态解析，
前提是该名已在 ``build_dsl_allowlist()`` 白名单内（canonical 或别名，且已有 runtime）。

维护约定：新增/修改算子实现请改 ``cleaned_operators/``；``api/`` 通常无需手写 kernel。
"""

from __future__ import annotations

from typing import Any, Callable

from .cleaned_ops import make_cleaned_call_factory
from .columns import col, field
from .factor import Factor

# 高频算子显式导出；其余算子通过 ``__getattr__`` 按白名单懒加载
rank = make_cleaned_call_factory("rank")
ts_mean = make_cleaned_call_factory("ts_mean")
ts_std = make_cleaned_call_factory("ts_std")
ts_std_dev = make_cleaned_call_factory("ts_std_dev")
zscore = make_cleaned_call_factory("zscore")
delay = make_cleaned_call_factory("delay")

#: ``ast_transform``（typed AST transform 公共 API）的懒导出映射。
#: 名字映射到 ast_transform 模块的函数；避免 api 包启动时加载 registry。
_AST_TRANSFORM_EXPORTS: dict[str, str] = {
    "ast_call": "call",
    "ast_crossover": "crossover",
    "ast_expr_kind": "expr_kind",
    "ast_infer_expr_kind": "infer_expr_kind",
    "ast_resolve_op_semantics": "resolve_op_semantics",
    "ast_scale_window": "scale_window",
    "ast_substitute_field": "substitute_field",
    "ast_to_dsl_text": "to_dsl_text",
}

__all__ = [
    "Factor",
    "col",
    "delay",
    "field",
    "make_cleaned_call_factory",
    "rank",
    "ts_mean",
    "ts_std",
    "ts_std_dev",
    "zscore",
]


def __getattr__(name: str) -> Callable[..., Any]:
    """按名懒加载 cleaned 算子工厂（须在 ``OperatorRegistry`` 中已实现）。

    首次访问 ``from api import ts_corr`` 等未在 ``__all__`` 显式导出的算子时，
    校验白名单并缓存工厂至模块 ``globals``。

    Parameters
    ----------
    name : str
        算子 canonical 名或别名。

    Returns
    -------
    Callable[..., CleanedCall]
        ``make_cleaned_call_factory(name)`` 返回的 DSL 工厂。

    Raises
    ------
    AttributeError
        算子未在 registry 中实现。
    """
    from factor_engine.api.operator_registry import build_dsl_allowlist

    if name in _AST_TRANSFORM_EXPORTS:
        import factor_engine.api.ast_transform as _at

        return getattr(_at, _AST_TRANSFORM_EXPORTS[name])
    if name not in build_dsl_allowlist():
        raise AttributeError(
            f"module {__name__!r} has no public daily-factor operator {name!r}"
        )
    factory = make_cleaned_call_factory(name)
    globals()[name] = factory
    return factory
