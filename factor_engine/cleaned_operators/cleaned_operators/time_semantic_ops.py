# -*- coding: utf-8 -*-
"""Time semantic operators: report asof, event windows, snapshot lag, calendar effects.

This module implements 6 time-aware semantic operators for PIT-safe temporal analysis.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12


def _positive_int(value: Any, name: str) -> int:
    """Validate positive integer parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    """Validate non-negative integer parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Validate and align input DataFrames."""
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("time semantic operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def pd_report_asof(
    value,
    report_date,
    asof_date,
    max_staleness_days=90,
    **_,
) -> pd.DataFrame:
    """Report date lookup with staleness constraint.

    Finds the most recent value whose report_date is <= asof_date, subject to
    max_staleness_days constraint. Returns NaN if no valid report found.

    Args:
        value: Value panel to look up
        report_date: Report/announcement date panel (datetime)
        asof_date: As-of date panel (datetime)
        max_staleness_days: Maximum days between report_date and asof_date (default 90)

    Returns:
        Panel of most recent values satisfying staleness constraint

    Notes:
        - PIT-safe: never uses future information
        - Returns NaN when no report found within staleness window
        - Backend: pandas_numpy only (datetime operations)
    """
    value, report_date, asof_date = _align(value, report_date, asof_date)
    max_staleness_days = _positive_int(max_staleness_days, "max_staleness_days")

    if not isinstance(report_date.index, pd.DatetimeIndex):
        raise TypeError("report_asof requires DatetimeIndex")

    out = np.full(value.shape, np.nan, dtype=float)

    for col_idx in range(value.shape[1]):
        val_series = value.iloc[:, col_idx].values
        report_series = report_date.iloc[:, col_idx].values
        asof_series = asof_date.iloc[:, col_idx].values

        for row_idx in range(value.shape[0]):
            asof = asof_series[row_idx]
            if pd.isna(asof):
                continue

            asof_dt = pd.Timestamp(asof)

            # Search backwards for most recent valid report
            for search_idx in range(row_idx, -1, -1):
                report = report_series[search_idx]
                val = val_series[search_idx]

                if pd.isna(report) or pd.isna(val):
                    continue

                report_dt = pd.Timestamp(report)
                if report_dt > asof_dt:
                    continue

                staleness = (asof_dt - report_dt).days
                if staleness <= max_staleness_days:
                    out[row_idx, col_idx] = val
                    break

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_event_window_return_asof(
    ret,
    event_date,
    window_before=5,
    window_after=5,
    **_,
) -> pd.DataFrame:
    """Event window cumulative return around event date.

    Computes cumulative return from [event_date - window_before] to
    [event_date + window_after]. PIT-safe: only computes for rows where
    current date >= event_date + window_after.

    Args:
        ret: Returns panel (daily)
        event_date: Event date panel (datetime)
        window_before: Days before event to include (default 5)
        window_after: Days after event to include (default 5)

    Returns:
        Panel of cumulative event window returns

    Notes:
        - PIT-safe: result only available after event window closes
        - Cumulative return computed as product of (1 + ret) - 1
        - Returns NaN if insufficient data or event not yet observable
        - Backend: pandas_numpy only
    """
    ret, event_date = _align(ret, event_date)
    window_before = _nonnegative_int(window_before, "window_before")
    window_after = _nonnegative_int(window_after, "window_after")

    if not isinstance(ret.index, pd.DatetimeIndex):
        raise TypeError("event_window_return_asof requires DatetimeIndex")

    out = np.full(ret.shape, np.nan, dtype=float)
    dates = ret.index

    for col_idx in range(ret.shape[1]):
        ret_series = ret.iloc[:, col_idx].values
        event_series = event_date.iloc[:, col_idx].values

        for row_idx in range(ret.shape[0]):
            current_date = dates[row_idx]
            event = event_series[row_idx]

            if pd.isna(event):
                continue

            event_dt = pd.Timestamp(event)
            window_end = event_dt + pd.Timedelta(days=window_after)

            # Only compute if current_date >= window_end (PIT-safe)
            if current_date < window_end:
                continue

            window_start = event_dt - pd.Timedelta(days=window_before)

            # Find indices for window
            start_idx = None
            end_idx = None
            for idx in range(row_idx + 1):
                if dates[idx] >= window_start and start_idx is None:
                    start_idx = idx
                if dates[idx] <= window_end:
                    end_idx = idx

            if start_idx is None or end_idx is None or start_idx > end_idx:
                continue

            # Compute cumulative return
            window_rets = ret_series[start_idx : end_idx + 1]
            valid = np.isfinite(window_rets)
            if valid.sum() < 1:
                continue

            cum_ret = np.prod(1.0 + window_rets[valid]) - 1.0
            if np.isfinite(cum_ret):
                out[row_idx, col_idx] = cum_ret

    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def pd_financial_snapshot_lag(
    x,
    period_id,
    fiscal_date,
    lag_periods=1,
    **_,
) -> pd.DataFrame:
    """Financial snapshot lag by fiscal periods.

    Returns the value from lag_periods fiscal periods ago, matching on period_id.
    PIT-safe: uses only historically available data.

    Args:
        x: Input financial metric panel
        period_id: Fiscal period identifier panel (e.g., "2024Q3")
        fiscal_date: Fiscal period end date panel (datetime)
        lag_periods: Number of fiscal periods to lag (default 1)

    Returns:
        Panel of lagged values by fiscal period

    Notes:
        - PIT-safe: never uses future information
        - Returns NaN when lagged period not found
        - Backend: pandas_numpy only
    """
    x, period_id, fiscal_date = _align(x, period_id, fiscal_date)
    lag_periods = _positive_int(lag_periods, "lag_periods")

    out = np.full(x.shape, np.nan, dtype=float)

    for col_idx in range(x.shape[1]):
        x_series = x.iloc[:, col_idx].values
        period_series = period_id.iloc[:, col_idx].values
        fiscal_series = fiscal_date.iloc[:, col_idx].values

        for row_idx in range(x.shape[0]):
            current_period = period_series[row_idx]
            if pd.isna(current_period):
                continue

            # Parse period (assuming format like "2024Q3")
            try:
                current_period_str = str(current_period)
                if "Q" in current_period_str:
                    year, quarter = current_period_str.split("Q")
                    year = int(year)
                    quarter = int(quarter)
                    target_quarter = quarter - lag_periods
                    target_year = year
                    while target_quarter < 1:
                        target_quarter += 4
                        target_year -= 1
                    target_period = f"{target_year}Q{target_quarter}"
                else:
                    # Annual periods
                    year = int(current_period_str)
                    target_period = str(year - lag_periods)
            except (ValueError, AttributeError):
                continue

            # Search backwards for target period
            for search_idx in range(row_idx, -1, -1):
                search_period = period_series[search_idx]
                if pd.isna(search_period):
                    continue
                if str(search_period) == target_period:
                    val = x_series[search_idx]
                    if np.isfinite(val):
                        out[row_idx, col_idx] = val
                    break

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_same_calendar_day_mean(
    x,
    window=52,
    **_,
) -> pd.DataFrame:
    """Same calendar day mean (weekly seasonality effect).

    Computes trailing mean of values on the same day-of-week over the past
    window weeks. Captures day-of-week effects.

    Args:
        x: Input panel
        window: Number of weeks to look back (default 52)

    Returns:
        Panel of same-day-of-week trailing means

    Notes:
        - PIT-safe: uses only trailing observations
        - Requires DatetimeIndex with day-of-week information
        - Returns NaN when insufficient observations
        - Backend: pandas_numpy only
    """
    x = _align(x)[0]
    window = _positive_int(window, "window")

    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("same_calendar_day_mean requires DatetimeIndex")

    out = np.full(x.shape, np.nan, dtype=float)
    dow = x.index.dayofweek  # 0=Monday, 6=Sunday

    for col_idx in range(x.shape[1]):
        x_series = x.iloc[:, col_idx].values

        for row_idx in range(x.shape[0]):
            current_dow = dow[row_idx]

            # Collect same-day values from past window weeks
            same_day_values = []
            for search_idx in range(row_idx - 1, -1, -1):
                if dow[search_idx] == current_dow:
                    val = x_series[search_idx]
                    if np.isfinite(val):
                        same_day_values.append(val)
                    if len(same_day_values) >= window:
                        break

            if len(same_day_values) >= 3:
                out[row_idx, col_idx] = np.mean(same_day_values)

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_same_calendar_month_return(
    x,
    window=12,
    **_,
) -> pd.DataFrame:
    """Same calendar month return (monthly seasonality effect).

    Computes trailing mean return for the same calendar month over the past
    window years. Captures month-of-year effects.

    Args:
        x: Input returns panel
        window: Number of years to look back (default 12)

    Returns:
        Panel of same-month trailing mean returns

    Notes:
        - PIT-safe: uses only trailing observations
        - Requires DatetimeIndex with month information
        - Returns NaN when insufficient observations
        - Backend: pandas_numpy only
    """
    x = _align(x)[0]
    window = _positive_int(window, "window")

    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("same_calendar_month_return requires DatetimeIndex")

    out = np.full(x.shape, np.nan, dtype=float)
    months = x.index.month

    for col_idx in range(x.shape[1]):
        x_series = x.iloc[:, col_idx].values

        for row_idx in range(x.shape[0]):
            current_month = months[row_idx]

            # Collect same-month values from past window years
            same_month_values = []
            for search_idx in range(row_idx - 1, -1, -1):
                if months[search_idx] == current_month:
                    val = x_series[search_idx]
                    if np.isfinite(val):
                        same_month_values.append(val)
                    if len(same_month_values) >= window:
                        break

            if len(same_month_values) >= 3:
                out[row_idx, col_idx] = np.mean(same_month_values)

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_same_clock_lag(
    x,
    clock_time,
    lag_minutes=5,
    **_,
) -> pd.DataFrame:
    """Minute-level clock lag (intraday temporal lag).

    Returns the value from lag_minutes ago based on clock_time. Useful for
    intraday analysis with irregular timestamps.

    Args:
        x: Input panel
        clock_time: Clock time panel (datetime with minute precision)
        lag_minutes: Minutes to lag (default 5)

    Returns:
        Panel of lagged values by clock time

    Notes:
        - PIT-safe: uses only past observations
        - Returns NaN when lagged time not found or insufficient data
        - Backend: pandas_numpy only
    """
    x, clock_time = _align(x, clock_time)
    lag_minutes = _positive_int(lag_minutes, "lag_minutes")

    out = np.full(x.shape, np.nan, dtype=float)

    for col_idx in range(x.shape[1]):
        x_series = x.iloc[:, col_idx].values
        time_series = clock_time.iloc[:, col_idx].values

        for row_idx in range(x.shape[0]):
            current_time = time_series[row_idx]
            if pd.isna(current_time):
                continue

            current_dt = pd.Timestamp(current_time)
            target_time = current_dt - pd.Timedelta(minutes=lag_minutes)

            # Search backwards for closest time <= target_time
            best_idx = None
            best_diff = None
            for search_idx in range(row_idx - 1, -1, -1):
                search_time = time_series[search_idx]
                if pd.isna(search_time):
                    continue

                search_dt = pd.Timestamp(search_time)
                if search_dt > target_time:
                    continue

                diff = abs((target_time - search_dt).total_seconds())
                if best_diff is None or diff < best_diff:
                    best_idx = search_idx
                    best_diff = diff

                # Stop searching if we're more than 2x lag_minutes away
                if search_dt < target_time - pd.Timedelta(minutes=lag_minutes * 2):
                    break

            if best_idx is not None:
                val = x_series[best_idx]
                if np.isfinite(val):
                    out[row_idx, col_idx] = val

    return pd.DataFrame(out, index=x.index, columns=x.columns)


# Operator wrapper classes
class _ReportAsof(SeriesOperator):
    metadata = OperatorMetadata(
        name="report_asof",
        category="time_semantic",
        description="Report date lookup with staleness constraint. Finds most recent value "
        "whose report_date <= asof_date within max_staleness_days.",
        param_names=["value", "report_date", "asof_date", "max_staleness_days"],
        return_type="series",
        tags=["time_semantic", "pit_safe", "causal", "daily", "typed_v2", "deterministic"],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_report_asof(*args, **kwargs)


class _EventWindowReturnAsof(SeriesOperator):
    metadata = OperatorMetadata(
        name="event_window_return_asof",
        category="time_semantic",
        description="Event window cumulative return. Computes return from "
        "[event_date - window_before] to [event_date + window_after]. "
        "Result only available after window closes (PIT-safe).",
        param_names=["ret", "event_date", "window_before", "window_after"],
        return_type="series",
        tags=["time_semantic", "pit_safe", "causal", "daily", "typed_v2", "deterministic"],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_event_window_return_asof(*args, **kwargs)


class _FinancialSnapshotLag(SeriesOperator):
    metadata = OperatorMetadata(
        name="financial_snapshot_lag",
        category="time_semantic",
        description="Financial snapshot lag by fiscal periods. Returns value from "
        "lag_periods fiscal periods ago, matching on period_id.",
        param_names=["x", "period_id", "fiscal_date", "lag_periods"],
        return_type="series",
        tags=["time_semantic", "pit_safe", "causal", "daily", "typed_v2", "deterministic"],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_financial_snapshot_lag(*args, **kwargs)


class _SameCalendarDayMean(SeriesOperator):
    metadata = OperatorMetadata(
        name="same_calendar_day_mean",
        category="time_semantic",
        description="Same calendar day mean (weekly seasonality). Computes trailing mean "
        "of values on same day-of-week over past window weeks.",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_semantic", "pit_safe", "causal", "daily", "typed_v2", "deterministic"],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_same_calendar_day_mean(*args, **kwargs)


class _SameCalendarMonthReturn(SeriesOperator):
    metadata = OperatorMetadata(
        name="same_calendar_month_return",
        category="time_semantic",
        description="Same calendar month return (monthly seasonality). Computes trailing "
        "mean return for same calendar month over past window years.",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_semantic", "pit_safe", "causal", "daily", "typed_v2", "deterministic"],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_same_calendar_month_return(*args, **kwargs)


class _SameClockLag(SeriesOperator):
    metadata = OperatorMetadata(
        name="same_clock_lag",
        category="time_semantic",
        description="Minute-level clock lag (intraday temporal lag). Returns value from "
        "lag_minutes ago based on clock_time.",
        param_names=["x", "clock_time", "lag_minutes"],
        return_type="series",
        tags=["time_semantic", "pit_safe", "causal", "intraday", "typed_v2", "deterministic"],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_same_clock_lag(*args, **kwargs)


def register() -> None:
    """Register time semantic operators."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # Check if already registered
    if "same_clock_lag" in OperatorRegistry._operators:
        return

    # Register pandas_numpy backend
    for name, cls in [
        ("report_asof", _ReportAsof),
        ("event_window_return_asof", _EventWindowReturnAsof),
        ("financial_snapshot_lag", _FinancialSnapshotLag),
        ("same_calendar_day_mean", _SameCalendarDayMean),
        ("same_calendar_month_return", _SameCalendarMonthReturn),
        ("same_clock_lag", _SameClockLag),
    ]:
        register_operator(
            name=name,
            category="time_semantic",
            business_category="time_semantic",
            canonical=name,
            source="time_semantic_ops",
            backend="pandas_numpy",
            status="experimental",
        )(cls)

    # Add to extended surface
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(
        {
            "report_asof",
            "event_window_return_asof",
            "financial_snapshot_lag",
            "same_calendar_day_mean",
            "same_calendar_month_return",
            "same_clock_lag",
        }
    )


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
    "report_asof": {
        "scope": "time_semantic",
        "pit_safe": True,
        "min_periods": 1,
    },
    "event_window_return_asof": {
        "scope": "time_semantic",
        "pit_safe": True,
        "min_periods": 1,
    },
    "financial_snapshot_lag": {
        "scope": "time_semantic",
        "pit_safe": True,
        "min_periods": 1,
    },
    "same_calendar_day_mean": {
        "scope": "time_semantic",
        "pit_safe": True,
        "min_periods": 3,
    },
    "same_calendar_month_return": {
        "scope": "time_semantic",
        "pit_safe": True,
        "min_periods": 3,
    },
    "same_clock_lag": {
        "scope": "time_semantic",
        "pit_safe": True,
        "min_periods": 1,
    },
}


register()
