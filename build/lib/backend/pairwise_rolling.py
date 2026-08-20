# -*- coding: utf-8
"""Pairwise rolling 辅助：ts_beta / vwap 等须共享有效样本集合。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

_TS = "ts"
_INST = "inst"


def polars_pairwise_valid(left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    """左右同时有效（非 NULL）。"""
    import polars as pl

    return left.is_not_null() & right.is_not_null()


def polars_pairwise_output_guard(left: "pl.Expr", right: "pl.Expr", value: "pl.Expr") -> "pl.Expr":
    """当前行任一侧无效 → 输出 NULL。"""
    import polars as pl

    return pl.when(~polars_pairwise_valid(left, right)).then(None).otherwise(value)


def polars_pairwise_mask(left: "pl.Expr", right: "pl.Expr") -> tuple["pl.Expr", "pl.Expr"]:
    """将非 pairwise 位置置 NULL，供 rolling_cov/var/sum 使用。"""
    import polars as pl

    valid = polars_pairwise_valid(left, right)
    return (
        pl.when(valid).then(left).otherwise(None),
        pl.when(valid).then(right).otherwise(None),
    )


def polars_ts_beta_expr(
    left: "pl.Expr",
    right: "pl.Expr",
    *,
    window: int,
    min_periods: int = 2,
    ddof: int = 1,
    inst_col: str = _INST,
    ts_col: str = _TS,
) -> "pl.Expr":
    """beta = cov(pair)/var(pair_right)；分母 var 与 cov 同样本集。"""
    import polars as pl

    pl_left, pl_right = polars_pairwise_mask(left, right)
    w = int(window)
    mp = max(int(min_periods), 2)
    over = inst_col, ts_col
    cov = pl.rolling_cov(pl_left, pl_right, window_size=w, min_samples=mp, ddof=ddof).over(
        inst_col, order_by=ts_col
    )
    var = pl_right.rolling_var(window_size=w, min_samples=mp, ddof=ddof).over(
        inst_col, order_by=ts_col
    )
    return pl.when(var.is_null() | (var == 0)).then(None).when(~polars_pairwise_valid(left, right)).then(None).otherwise(cov / var)


def polars_vwap_expr(
    price: "pl.Expr",
    volume: "pl.Expr",
    *,
    window: int,
    min_periods: int = 1,
    inst_col: str = _INST,
    ts_col: str = _TS,
) -> "pl.Expr":
    """VWAP = sum(pair_pv)/sum(pair_v)，price/volume 须同时有效。"""
    import polars as pl

    valid = polars_pairwise_valid(price, volume)
    pair_pv = pl.when(valid).then(price * volume).otherwise(None)
    pair_v = pl.when(valid).then(volume).otherwise(None)
    w = int(window)
    mp = int(min_periods)
    sum_pv = pair_pv.rolling_sum(window_size=w, min_samples=mp).over(inst_col, order_by=ts_col)
    sum_v = pair_v.rolling_sum(window_size=w, min_samples=mp).over(inst_col, order_by=ts_col)
    return pl.when(sum_v.is_null() | (sum_v == 0)).then(None).otherwise(sum_pv / sum_v)


def sql_pairwise_beta_expr(
    *,
    left_col: str,
    right_col: str,
    over: str,
    dialect_is_duckdb: bool = True,
    min_periods: int = 2,
) -> str:
    """DuckDB/CH pairwise ts_beta SQL 核心表达式。"""
    if dialect_is_duckdb:
        cov_fn, var_fn = "covar_samp", "var_samp"
    else:
        cov_fn, var_fn = "covarSamp", "varSamp"
    pair_mask = f"{left_col} IS NOT NULL AND {right_col} IS NOT NULL"
    pair_l = f"CASE WHEN {pair_mask} THEN {left_col} END"
    pair_r = f"CASE WHEN {pair_mask} THEN {right_col} END"
    cnt = f"COUNT(CASE WHEN {pair_mask} THEN 1 END) OVER ({over})"
    cov = f"{cov_fn}({pair_l}, {pair_r}) OVER ({over})"
    var = f"{var_fn}({pair_r}) OVER ({over})"
    body = f"({cov}) / NULLIF({var}, 0)"
    if min_periods <= 2:
        return body
    return f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {body} END"


def sql_pairwise_vwap_expr(
    *,
    price_col: str,
    volume_col: str,
    over: str,
    min_periods: int = 1,
) -> str:
    pair_mask = f"{price_col} IS NOT NULL AND {volume_col} IS NOT NULL"
    pair_pv = f"CASE WHEN {pair_mask} THEN {price_col} * {volume_col} END"
    pair_v = f"CASE WHEN {pair_mask} THEN {volume_col} END"
    cnt = f"COUNT(CASE WHEN {pair_mask} THEN 1 END) OVER ({over})"
    num = f"SUM({pair_pv}) OVER ({over})"
    den = f"SUM({pair_v}) OVER ({over})"
    body = f"{num} / NULLIF({den}, 0)"
    if min_periods <= 1:
        return body
    return f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {body} END"
