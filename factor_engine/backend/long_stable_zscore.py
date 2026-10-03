"""Standalone native-Polars grouped-long finite-anchor z-score math.

This module is an FE-owned reusable math helper, not an operator registration
or production admission. It accepts one grouped long table and preserves its
row order and non-value columns without pivoting.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import polars as pl


def _validate_parameters(ddof: Any, constant_value: Any) -> tuple[int | float, float]:
    if isinstance(ddof, bool) or not isinstance(ddof, Real):
        raise ValueError("ddof must be a finite non-negative real number")
    if isinstance(ddof, Integral):
        if ddof < 0:
            raise ValueError("ddof must be a finite non-negative real number")
        # Keep arbitrarily large integer ddof exact. The height shortcut runs
        # before any Float64 conversion, avoiding overflow for huge integers.
        normalized_ddof: int | float = int(ddof)
    else:
        try:
            normalized_ddof = float(ddof)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("ddof must be a finite non-negative real number") from exc
        if not (0.0 <= normalized_ddof < float("inf")):
            raise ValueError("ddof must be a finite non-negative real number")
    if isinstance(constant_value, bool) or not isinstance(constant_value, Real):
        raise ValueError("constant_value must be a finite real number")
    try:
        constant = float(constant_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("constant_value must be a finite real number") from exc
    if not (-float("inf") < constant < float("inf")):
        raise ValueError("constant_value must be a finite real number")
    return normalized_ddof, constant


def _fresh_name(existing: set[str], stem: str) -> str:
    candidate = stem
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    existing.add(candidate)
    return candidate


def finite_anchor_centered_zscore_long(
    frame: pl.DataFrame,
    *,
    value_col: str,
    group_col: str,
    ddof: float = 1,
    constant_value: float = 0.0,
) -> pl.DataFrame:
    """Compute stable cross-sectional z-scores within each long-table group.

    The input must be an eager Polars DataFrame with a real numeric value
    column. Values are converted to Float64 before arithmetic. Finite members
    alone determine group moments; null/NaN inputs become NaN outputs, while
    any infinity causes all non-missing values in that group to receive the
    constant fallback. The input row/column order and other columns are kept.

    Finite non-negative real ``ddof`` values are accepted, including
    fractional values. The expression plan stages group-level intermediates
    and never pivots to a wide panel.
    """
    ddof, constant = _validate_parameters(ddof, constant_value)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if value_col not in frame.columns:
        raise ValueError(f"value column {value_col!r} is missing")
    if group_col not in frame.columns:
        raise ValueError(f"group column {group_col!r} is missing")
    if value_col == group_col:
        raise ValueError("value_col and group_col must be distinct")
    value_dtype = frame.schema[value_col]
    if value_dtype == pl.Boolean or (
        not value_dtype.is_numeric() and value_dtype != pl.Null
    ):
        raise TypeError("value column must contain real numeric values")

    if ddof >= frame.height:
        value = pl.col(value_col).cast(pl.Float64, strict=False)
        return frame.with_columns(
            pl.when(value.is_null() | value.is_nan().fill_null(False))
            .then(float("nan"))
            .otherwise(constant)
            .alias(value_col)
        )

    work = frame.with_columns(
        pl.col(value_col).cast(pl.Float64, strict=False).alias(value_col)
    )
    names = set(work.columns)
    temp = {
        key: _fresh_name(names, f"__fe_fa_long_{key}")
        for key in (
            "missing",
            "finite",
            "has_inf",
            "low",
            "high",
            "anchor",
            "delta",
            "scale",
            "unit",
            "mean",
            "centered",
            "squared_sum",
            "count",
            "std",
        )
    }
    group = pl.col(group_col)
    value = pl.col(value_col)
    finite = value.is_finite().fill_null(False)
    missing = value.is_null() | value.is_nan().fill_null(False)

    work = work.with_columns(
        [
            missing.alias(temp["missing"]),
            finite.alias(temp["finite"]),
            value.is_infinite()
            .fill_null(False)
            .any()
            .over(group)
            .alias(temp["has_inf"]),
            pl.when(finite)
            .then(value)
            .otherwise(None)
            .min()
            .over(group)
            .alias(temp["low"]),
            pl.when(finite)
            .then(value)
            .otherwise(None)
            .max()
            .over(group)
            .alias(temp["high"]),
        ]
    )
    work = work.with_columns(
        (pl.col(temp["low"]) / 2.0 + pl.col(temp["high"]) / 2.0)
        .fill_null(0.0)
        .alias(temp["anchor"])
    )
    work = work.with_columns(
        pl.when(pl.col(temp["finite"]))
        .then(value - pl.col(temp["anchor"]))
        .otherwise(0.0)
        .alias(temp["delta"])
    )
    work = work.with_columns(
        pl.col(temp["delta"])
        .abs()
        .max()
        .over(group)
        .fill_null(0.0)
        .alias(temp["scale"])
    )
    work = work.with_columns(
        pl.when(pl.col(temp["scale"]) > 0.0)
        .then(pl.col(temp["delta"]) / pl.col(temp["scale"]))
        .otherwise(0.0)
        .alias(temp["unit"])
    )
    work = work.with_columns(
        pl.when(pl.col(temp["finite"]))
        .then(pl.col(temp["unit"]))
        .otherwise(None)
        .mean()
        .over(group)
        .fill_null(0.0)
        .alias(temp["mean"])
    )
    work = work.with_columns(
        pl.when(pl.col(temp["finite"]))
        .then(pl.col(temp["unit"]) - pl.col(temp["mean"]))
        .otherwise(0.0)
        .alias(temp["centered"])
    )
    work = work.with_columns(
        pl.when(pl.col(temp["finite"]))
        .then(pl.col(temp["centered"]).pow(2))
        .otherwise(0.0)
        .sum()
        .over(group)
        .alias(temp["squared_sum"]),
        pl.col(temp["finite"]).cast(pl.UInt64).sum().over(group).alias(temp["count"]),
    )
    count_float = pl.col(temp["count"]).cast(pl.Float64)
    denominator = count_float - float(ddof)
    work = work.with_columns(
        (pl.col(temp["squared_sum"]) / denominator).sqrt().alias(temp["std"])
    )

    degenerate = (
        (pl.col(temp["count"]) <= ddof)
        | (pl.col(temp["scale"]) == 0.0)
        | (pl.col(temp["squared_sum"]) == 0.0)
        | ~pl.col(temp["std"]).is_finite().fill_null(False)
        | (pl.col(temp["std"]) <= 0.0)
    )
    output = (
        pl.when(pl.col(temp["missing"]))
        .then(float("nan"))
        .when(pl.col(temp["has_inf"]) | degenerate)
        .then(constant)
        .otherwise(pl.col(temp["centered"]) / pl.col(temp["std"]))
        .alias(value_col)
    )
    return work.with_columns(output).drop(list(temp.values()))


__all__ = ["finite_anchor_centered_zscore_long"]
