# -*- coding: utf-8
"""统计有效值谓词：rank/zscore/quantile 等排除 NULL/NaN，诊断算子仍见原始 NaN。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from factor_engine.backend.sql_pushdown.emitter import SqlDialect


def rank_excludes_nan(canon: str) -> bool:
    """rank 族是否将 NaN 视为无效（不参与分母/排名）。"""
    from factor_engine.backend.rank_spec import rank_ignore_nan

    return rank_ignore_nan(canon)


def polars_rank_input(value_col: str, *, exclude_nan: bool) -> "pl.Expr":
    """rank 输入列：无效值置 NULL，不参与 rank/count。

    ``exclude_nan=True`` masks NaN and ±Inf (RankSpec.inf_policy="exclude",
    pandas reference ``cs_rank_01`` masks via ``np.isfinite`` — NEW-255).
    ``exclude_nan=False`` (NaN-only skip) keeps ±Inf as a rankable extreme so a
    cross-section with an Inf cell ranks over n values exactly like pandas
    ``x.rank(pct=True, axis=1)`` (cs_pct_rank / rank_pct inf_policy="participate").
    """
    import polars as pl

    c = pl.col(value_col)
    if exclude_nan:
        return pl.when(c.is_null() | c.is_nan() | c.is_infinite()).then(None).otherwise(c)
    return pl.when(c.is_null() | c.is_nan()).then(None).otherwise(c)


def polars_row_stat_invalid(value_col: str, *, exclude_nan: bool) -> "pl.Expr":
    """当前行是否统计无效（输出 NULL）。"""
    import polars as pl

    c = pl.col(value_col)
    if exclude_nan:
        return c.is_null() | c.is_nan() | c.is_infinite()
    return c.is_null() | c.is_nan()


def stat_valid_sql(col: str, *, dialect: "SqlDialect", exclude_nan: bool = True) -> str:
    """SQL 谓词：值可用于 rank/aggregate 统计。

    Aligns with pandas ``CrossSectionSampleMask`` / ``np.isfinite`` and
    ``polars_rank_input``: ±Inf is never a legal statistical sample.
    """
    from factor_engine.backend.sql_pushdown.emitter import SqlDialect

    if not exclude_nan:
        return f"{col} IS NOT NULL"
    isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
    isinf_fn = "isInfinite" if dialect == SqlDialect.CLICKHOUSE else "isinf"
    return (
        f"({col} IS NOT NULL AND NOT {isnan_fn}({col}) AND NOT {isinf_fn}({col}))"
    )


def row_stat_invalid_sql(col: str, *, dialect: "SqlDialect", exclude_nan: bool = True) -> str:
    """SQL 谓词：当前行统计无效 → 输出 NULL。

    ±Inf is invalid exactly like NaN (R19-027..029 finite sample mask).
    """
    from factor_engine.backend.sql_pushdown.emitter import SqlDialect

    if not exclude_nan:
        return f"{col} IS NULL"
    isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
    isinf_fn = "isInfinite" if dialect == SqlDialect.CLICKHOUSE else "isinf"
    return (
        f"({col} IS NULL OR {isnan_fn}({col}) OR {isinf_fn}({col}))"
    )
