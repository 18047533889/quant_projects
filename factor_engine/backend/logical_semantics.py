# -*- coding: utf-8
"""逻辑真值与 is_null/is_nan 语义（Pandas / PolarsLong / DuckDB 共用）。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def truthy_inf_is_true() -> bool:
    """±Inf 在 and_/or_/not_/where 中视为 true（非零有限值规则）。"""
    return True


def truthy_nan_is_false() -> bool:
    """NaN 在逻辑算子中视为 false。"""
    return True


def truthy_null_is_false() -> bool:
    """SQL/Polars NULL 在逻辑算子中视为 false。"""
    return True


def is_nan_excludes_null() -> bool:
    """``is_nan(NULL)=0``；``is_null(NULL)=1``。"""
    return True


def truthy_polars_expr(expr: "pl.Expr") -> "pl.Expr":
    """Polars 浮点 truthy：非 NULL、非 NaN、非零。"""
    import polars as pl

    finite = expr.is_not_null() & ~expr.is_nan()
    if truthy_inf_is_true():
        return finite & (expr != 0)
    return finite & expr.is_finite() & (expr != 0)


def truthy_sql(value_col: str, *, dialect_is_clickhouse: bool = False) -> str:
    """DuckDB/ClickHouse 浮点 truthy SQL。"""
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    base = f"({value_col} IS NOT NULL AND NOT {isnan_fn}({value_col}) AND {value_col} <> 0)"
    if not truthy_inf_is_true():
        finite_fn = "isFinite" if dialect_is_clickhouse else "isfinite"
        return f"({value_col} IS NOT NULL AND {finite_fn}({value_col}) AND {value_col} <> 0)"
    return base


def is_null_polars_expr(col: str = "_v") -> "pl.Expr":
    """缺失检测：SQL NULL 与 IEEE NaN（与 Pandas ``isna`` 对齐）。"""
    import polars as pl

    c = pl.col(col)
    return (c.is_null() | c.is_nan()).cast(pl.Float64)


def is_not_null_polars_expr(col: str = "_v") -> "pl.Expr":
    """非缺失检测（与 Pandas ``notna`` 对齐）。"""
    import polars as pl

    return pl.when(is_null_polars_expr(col) > 0).then(0.0).otherwise(1.0)


def is_nan_polars_expr(col: str = "_v") -> "pl.Expr":
    """仅检测 IEEE NaN（NULL → 0）。"""
    import polars as pl

    return (
        pl.when(pl.col(col).is_null())
        .then(0.0)
        .otherwise(pl.col(col).is_nan().cast(pl.Float64))
    )


def is_null_sql(value_col: str = "_v", *, dialect_is_clickhouse: bool = False) -> str:
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    return (
        f"CASE WHEN {value_col} IS NULL OR {isnan_fn}({value_col}) "
        f"THEN 1.0 ELSE 0.0 END"
    )


def is_nan_sql(value_col: str = "_v", *, dialect_is_clickhouse: bool = False) -> str:
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    return f"CASE WHEN {value_col} IS NULL THEN 0.0 WHEN {isnan_fn}({value_col}) THEN 1.0 ELSE 0.0 END"
