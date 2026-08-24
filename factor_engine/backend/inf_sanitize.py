# -*- coding: utf-8
"""按 ``NumericSemantics`` 对算子输出做 ±Inf 清洗（不作用于 is_finite/is_infinite 等检测算子）。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from factor_engine.backend.numeric_semantics import semantics_for

if TYPE_CHECKING:
    import polars as pl


def inf_policy_applies(canon: str) -> bool:
    """该算子是否应对输出 Inf 做 to_null/to_nan 清洗。"""
    if canon in {"is_finite", "is_infinite", "is_nan", "is_null", "is_not_null", "nan_to_num"}:
        return False
    sem = semantics_for(canon)
    return sem.output_inf_to_nan or sem.output_inf_to_null


def apply_inf_policy_polars_fast(expr: "pl.Expr", canon: str) -> "pl.Expr":
    """Polars 原生 expr：±Inf → NULL。"""
    import polars as pl

    if not inf_policy_applies(canon):
        return expr
    return pl.when(expr.is_infinite()).then(None).otherwise(expr)


def apply_inf_policy_sql(expr_sql: str, canon: str, *, dialect_is_clickhouse: bool = False) -> str:
    """SQL：±Inf → NULL。"""
    if not inf_policy_applies(canon):
        return expr_sql
    finite_fn = "isFinite" if dialect_is_clickhouse else "isfinite"
    return f"CASE WHEN {expr_sql} IS NULL OR NOT {finite_fn}({expr_sql}) THEN NULL ELSE {expr_sql} END"
