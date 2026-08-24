# -*- coding: utf-8 -*-
"""same_clock_lag: minute-level clock-aligned lag operator (TRUE_GAP special timing).

``same_clock_lag(x, lag, clock_unit)`` returns the value of ``x`` at ``lag``
periods earlier on the SAME clock (e.g., same minute-of-day across trading days).

Unlike ``ts_delay`` which shifts along the time axis, ``same_clock_lag`` aligns
on a recurring clock position:
- ``same_clock_lag(x, 1, "minute_of_day")`` returns yesterday's value at the
  exact same minute-of-day.
- ``same_clock_lag(x, 5, "minute_of_day")`` returns the value 5 trading days ago
  at the same minute-of-day.

Clock alignment semantics:
- The output is defined only when the reference clock position exists in history.
- If the lag period does not have data at the same clock position, returns NaN.
- This is a TRUE_GAP operator: it operates on distinct clock-aligned events,
  not forward-filled daily values.
- PIT-safe: only references strictly prior observations.

Use cases:
- Intraday momentum: compare current minute to the same minute yesterday.
- Clock-relative patterns: detect recurring intraday behavior at specific times.
- Multi-day intraday comparison: measure changes at identical session positions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_EPS = 1e-12


def _metadata() -> OperatorMetadata:
    return OperatorMetadata(
        name="same_clock_lag",
        category="time_series",
        description="分钟级同时钟位置滞后（返回 lag 个交易日前相同分钟位置的值）。",
        param_names=["x", "lag", "clock_unit"],
        return_type="series",
        tags=[
            "time_series", "intraday", "minute", "pit_safe", "causal",
            "true_gap", "clock_aligned", "typed_v2",
            "signature:x,lag,clock_unit->series",
            "domain:intraday_clock", "unit:passthrough", "cost:4",
            "min_periods:1",
        ],
    )


def _extract_clock_key(index: pd.DatetimeIndex, clock_unit: str) -> np.ndarray:
    """Extract clock position key from datetime index.

    Args:
        index: DatetimeIndex (minute-level timestamps)
        clock_unit: "minute_of_day" or "slot"

    Returns:
        Array of integer clock keys (minute-of-day: 0-1439)
    """
    if clock_unit == "minute_of_day":
        # Extract minute-of-day (0-1439)
        times = index.to_numpy().astype("datetime64[m]")
        day_start = times.astype("datetime64[D]")
        minutes_since_midnight = (times - day_start).astype("timedelta64[m]").astype(int)
        return minutes_since_midnight
    elif clock_unit == "slot":
        # Use hour*60 + minute as slot key
        return index.hour * 60 + index.minute
    else:
        raise ValueError(f"Unknown clock_unit: {clock_unit!r}, expected 'minute_of_day' or 'slot'")


def _same_clock_lag_pandas(
    x: pd.DataFrame,
    lag: int,
    clock_unit: str = "minute_of_day",
) -> pd.DataFrame:
    """Pandas implementation of same_clock_lag.

    Strategy: For each (date, clock_position), look back `lag` trading days
    to find the same clock_position.
    """
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("same_clock_lag requires DatetimeIndex (minute-level timestamps)")

    lag = int(lag)
    if lag < 0:
        raise ValueError(f"same_clock_lag requires non-negative lag, got {lag}")

    if lag == 0:
        return x.copy()

    # Extract date and clock position
    dates = x.index.date
    clock_keys = _extract_clock_key(x.index, clock_unit)

    # Build a lookup map: (date, clock_key, symbol) -> value
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)

    # Get unique dates sorted
    unique_dates = pd.Series(dates).drop_duplicates().sort_values().values
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    # For each row, find the value lag trading days ago at the same clock position
    for i, (dt, clock_key) in enumerate(zip(dates, clock_keys)):
        date_idx = date_to_idx.get(dt)
        if date_idx is None or date_idx < lag:
            continue

        # Find the target date (lag trading days ago)
        target_date = unique_dates[date_idx - lag]

        # Find rows in x that match (target_date, clock_key)
        target_mask = (dates == target_date) & (clock_keys == clock_key)
        target_indices = np.where(target_mask)[0]

        if len(target_indices) > 0:
            # Take the first matching row (should be exactly one per symbol)
            target_idx = target_indices[0]
            result.iloc[i] = x.iloc[target_idx].values

    return result


def _same_clock_lag_polars(
    x: "pl.DataFrame",
    lag: int,
    clock_unit: str = "minute_of_day",
) -> "pl.DataFrame":
    """Polars implementation of same_clock_lag.

    Uses Polars expressions for efficient groupby-shift on (date, clock_position).
    """
    if pl is None:
        raise ImportError("polars is not installed")

    lag = int(lag)
    if lag < 0:
        raise ValueError(f"same_clock_lag requires non-negative lag, got {lag}")

    if lag == 0:
        return x.clone()

    # Polars strategy: add date and clock_key columns, group by clock_key, shift by lag within each group
    timestamp_col = x.columns[0] if len(x.columns) > 0 else None
    if timestamp_col is None or not x[timestamp_col].dtype == pl.Datetime:
        raise ValueError("same_clock_lag requires first column to be Datetime")

    # Extract date and clock position
    if clock_unit == "minute_of_day":
        clock_expr = (pl.col(timestamp_col).dt.hour() * 60 + pl.col(timestamp_col).dt.minute())
    elif clock_unit == "slot":
        clock_expr = (pl.col(timestamp_col).dt.hour() * 60 + pl.col(timestamp_col).dt.minute())
    else:
        raise ValueError(f"Unknown clock_unit: {clock_unit!r}")

    date_expr = pl.col(timestamp_col).dt.date()

    # Add temporary columns
    temp = x.with_columns([
        date_expr.alias("__date__"),
        clock_expr.alias("__clock__"),
    ])

    # For each clock position, shift by lag within the date sequence
    # This requires sorting by (clock, date) and then shifting
    value_cols = [c for c in x.columns if c != timestamp_col]

    # Build shifted expressions for all value columns
    shifted_exprs = [
        pl.col(col).shift(lag).over("__clock__").alias(col)
        for col in value_cols
    ]

    result = (
        temp
        .sort(["__clock__", "__date__"])
        .with_columns(shifted_exprs)
        .sort(timestamp_col)  # restore original order
        .drop(["__date__", "__clock__"])
    )

    return result


@register_operator(
    name="same_clock_lag",
    category="time_series",
    business_category="intraday_microstructure",
    canonical="same_clock_lag",
    source="same_clock_lag",
    backend="pandas_numpy",
    status="implemented",
)
class SameClockLagPandas(SeriesOperator):
    """分钟级同时钟位置滞后（pandas 后端）。"""

    metadata = _metadata()

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lag: int = 1,
        clock_unit: str = "minute_of_day",
        **_: Any,
    ) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(lag, "lag", minimum=0)
        clock_unit = str(clock_unit)

        return _same_clock_lag_pandas(x, lag, clock_unit)


@register_operator(
    name="same_clock_lag",
    category="time_series",
    business_category="intraday_microstructure",
    canonical="same_clock_lag",
    source="same_clock_lag",
    backend="polars",
    status="implemented",
)
class SameClockLagPolars(SeriesOperator):
    """分钟级同时钟位置滞后（polars 后端）。"""

    metadata = _metadata()

    def _calculate_series(
        self,
        x: "pl.DataFrame",
        lag: int = 1,
        clock_unit: str = "minute_of_day",
        **_: Any,
    ) -> "pl.DataFrame":
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(lag, "lag", minimum=0)
        clock_unit = str(clock_unit)

        return _same_clock_lag_polars(x, lag, clock_unit)


# Register to EXTENDED_ONLY_CANONICALS
import factor_engine.cleaned_operators.operator_surface as _surface
_surface.extend_extended_only({"same_clock_lag"})
