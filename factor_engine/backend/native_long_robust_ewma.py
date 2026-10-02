"""Pure-Polars lagged robust EWMA for long panels.

This reusable numerical primitive is intentionally not registered as
``ts_robust_ema``: its lagged rolling mean/sample-standard-deviation
winsorization contract differs from that operator.

Input row order is the within-asset chronology. Assets may be interleaved, but
each asset's time values must be non-null and monotone in input order. The
returned frame retains every input row and column in the same order, with one
output column appended.
"""
from __future__ import annotations

import math
import numbers

import polars as pl

_WINDOW = 10
_ROLLING_MIN_SAMPLES = 2
_STD_FLOOR = 1e-12


def _halflife_alpha(halflife: float) -> float:
    if isinstance(halflife, bool) or not isinstance(halflife, numbers.Real):
        raise ValueError("halflife must be a finite positive real number")
    value = float(halflife)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("halflife must be a finite positive real number")
    alpha = -math.expm1(-math.log(2.0) / value)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("halflife is outside the supported binary64 EWMA domain")
    return alpha


def _winsor_width(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("winsor_std must be a finite real number")
    width = float(value)
    if not math.isfinite(width):
        raise ValueError("winsor_std must be a finite real number")
    # Preserve factor_preprocess's historical handling of non-positive widths.
    return width if width > 0.0 else _STD_FLOOR


def _min_periods(value: int) -> int:
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < 0):
        raise ValueError("min_periods must be a non-negative integer")
    return int(value)


def _temporary_name(columns: set[str], base: str) -> str:
    name = base
    suffix = 0
    while name in columns:
        suffix += 1
        name = f"{base}_{suffix}"
    columns.add(name)
    return name


def lagged_robust_ewma(
    frame: pl.DataFrame,
    *,
    halflife: float,
    winsor_std: float = 4.0,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
    output_col: str = "lagged_robust_ewma",
) -> pl.DataFrame:
    """Append a causal rolling-winsorized EWMA column to an eager long panel.

    Each row consumes the value lagged once within its asset. Clip bounds use
    the prior ten lagged rows with at least two finite observations, a rolling
    mean and sample standard deviation floored at ``1e-12``. Null or NaN
    bounds impose no clip constraint. Clipping occurs before remaining
    non-finite values become null. EWM uses ``adjust=False`` and
    ``ignore_nulls=False``; forward fill is per asset and gated by the number
    of valid EWM inputs.

    Equal timestamps are allowed and retain input order. The input is neither
    sorted nor mutated; interleaved assets and duplicate times retain row
    identity through an internal row ordinal.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be an eager Polars DataFrame")
    alpha = _halflife_alpha(halflife)
    width = _winsor_width(winsor_std)
    min_periods = _min_periods(min_periods)
    for name, value in (("asset_col", asset_col), ("time_col", time_col),
                        ("value_col", value_col), ("output_col", output_col)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty column name")
    if len({asset_col, time_col, value_col}) != 3:
        raise ValueError("asset_col, time_col, and value_col must be distinct")
    missing = {asset_col, time_col, value_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"long panel is missing columns: {sorted(missing)}")
    if not frame.schema[value_col].is_numeric():
        raise TypeError(f"{value_col} must have a numeric Polars dtype")
    if output_col in frame.columns:
        raise ValueError(f"output_col already exists: {output_col}")

    names = set(frame.columns)
    row = _temporary_name(names, "__lagged_robust_ewma_row")
    previous_time = _temporary_name(names, "__lagged_robust_ewma_previous_time")
    lag = _temporary_name(names, "__lagged_robust_ewma_lag")
    finite_lag = _temporary_name(names, "__lagged_robust_ewma_finite_lag")
    mean = _temporary_name(names, "__lagged_robust_ewma_mean")
    std = _temporary_name(names, "__lagged_robust_ewma_std")
    lower = _temporary_name(names, "__lagged_robust_ewma_lower")
    upper = _temporary_name(names, "__lagged_robust_ewma_upper")
    clipped = _temporary_name(names, "__lagged_robust_ewma_clipped")
    ewm_input = _temporary_name(names, "__lagged_robust_ewma_input")
    ewm_value = _temporary_name(names, "__lagged_robust_ewma_value")
    count = _temporary_name(names, "__lagged_robust_ewma_count")

    staged = frame.with_row_index(row)
    grouped = lambda expr: expr.over(asset_col, order_by=row)
    asset_present = pl.col(asset_col).is_not_null()
    if frame.schema[asset_col] in (pl.Float32, pl.Float64):
        asset_present = asset_present & ~pl.col(asset_col).is_nan()

    # Validate within-asset chronology without requiring globally sorted input.
    order_check = (
        staged.filter(asset_present)
        .with_columns(grouped(pl.col(time_col).shift(1)).alias(previous_time))
        .filter(
            pl.col(time_col).is_null()
            | (pl.col(time_col) < pl.col(previous_time)).fill_null(False)
        )
        .select(pl.len().alias("_bad"))
        .item()
    )
    if order_check:
        raise ValueError("per-asset dates must be non-null and monotone increasing")

    staged = staged.with_columns(
        grouped(pl.col(value_col).cast(pl.Float64, strict=True).shift(1)).alias(lag)
    ).with_columns(
        pl.when(pl.col(lag).is_finite()).then(pl.col(lag)).otherwise(None).alias(finite_lag)
    ).with_columns(
        grouped(pl.col(finite_lag).rolling_mean(
            window_size=_WINDOW, min_samples=_ROLLING_MIN_SAMPLES,
        )).alias(mean),
        grouped(pl.col(finite_lag).rolling_std(
            window_size=_WINDOW, min_samples=_ROLLING_MIN_SAMPLES, ddof=1,
        )).clip(lower_bound=_STD_FLOOR).alias(std),
    ).with_columns(
        (pl.col(mean) - width * pl.col(std)).alias(lower),
        (pl.col(mean) + width * pl.col(std)).alias(upper),
    )

    # pandas clip treats each NaN bound as unconstrained. Guard a missing lag
    # before horizontal min/max: Polars otherwise orders NaN and may clip it.
    low_clipped = pl.when(
        pl.col(lower).is_not_null() & ~pl.col(lower).is_nan()
    ).then(
        pl.max_horizontal(pl.col(lag), pl.col(lower))
    ).otherwise(pl.col(lag))
    clipped_expr = pl.when(
        pl.col(lag).is_null() | pl.col(lag).is_nan()
    ).then(None).otherwise(
        pl.when(pl.col(upper).is_not_null() & ~pl.col(upper).is_nan())
        .then(pl.min_horizontal(low_clipped, pl.col(upper)))
        .otherwise(low_clipped)
    )
    staged = staged.with_columns(clipped_expr.alias(clipped)).with_columns(
        pl.when(pl.col(clipped).is_finite()).then(pl.col(clipped))
        .otherwise(None).alias(ewm_input)
    )

    ewm = pl.col(ewm_input).ewm_mean(
        alpha=alpha,
        adjust=False,
        min_samples=max(1, min_periods),
        ignore_nulls=False,
    )
    staged = staged.with_columns(
        grouped(pl.col(ewm_input).is_not_null().cast(pl.UInt64).cum_sum()).alias(count),
        grouped(ewm.forward_fill()).alias(ewm_value),
    ).with_columns(
        pl.when(
            asset_present & (pl.col(count) >= min_periods)
        ).then(pl.col(ewm_value)).otherwise(None).alias(output_col)
    )
    return staged.drop(
        row, lag, finite_lag, mean, std, lower, upper,
        clipped, ewm_input, ewm_value, count,
    )


__all__ = ["lagged_robust_ewma"]
