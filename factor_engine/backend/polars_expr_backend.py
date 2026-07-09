# -*- coding: utf-8
"""兼容层：旧 ``polars_expr_backend`` 模块名 → ``polars_expr_emitter``。"""
from __future__ import annotations

from .polars_expr_emitter import (  # noqa: F401
    POLARS_EXPR_CAPABLE,
    POLARS_LONG_CAPABLE,
    collect_columns,
    compile_plan_to_polars,
    execute_polars_expr_plan,
    execute_polars_long_plan,
    plan_is_polars_expr_capable,
    plan_is_polars_long_capable,
)

__all__ = [
    "POLARS_EXPR_CAPABLE",
    "POLARS_LONG_CAPABLE",
    "collect_columns",
    "compile_plan_to_polars",
    "execute_polars_expr_plan",
    "execute_polars_long_plan",
    "plan_is_polars_expr_capable",
    "plan_is_polars_long_capable",
]
