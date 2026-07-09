# -*- coding: utf-8
"""兼容层：旧 ``polars_expr_backend`` 模块名 → ``polars_expr_emitter``。"""
from __future__ import annotations

from .polars_expr_emitter import (  # noqa: F401
    POLARS_EXPR_CAPABLE,
    POLARS_LONG_CAPABLE,
    POLARS_LONG_COMPATIBLE,
    POLARS_LONG_MAP_GROUPS,
    POLARS_LONG_NATIVE,
    collect_columns,
    collect_plan_op_stats,
    compile_plan_to_polars,
    execute_polars_expr_plan,
    execute_polars_long_plan,
    get_polars_long_capable,
    plan_is_polars_expr_capable,
    plan_is_polars_long_capable,
)
from .polars_long_policy import strict_polars_long_fallback  # noqa: F401

__all__ = [
    "POLARS_EXPR_CAPABLE",
    "POLARS_LONG_CAPABLE",
    "POLARS_LONG_COMPATIBLE",
    "POLARS_LONG_MAP_GROUPS",
    "POLARS_LONG_NATIVE",
    "get_polars_long_capable",
    "collect_columns",
    "collect_plan_op_stats",
    "compile_plan_to_polars",
    "execute_polars_expr_plan",
    "execute_polars_long_plan",
    "plan_is_polars_expr_capable",
    "plan_is_polars_long_capable",
    "strict_polars_long_fallback",
]
