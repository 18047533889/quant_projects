# -*- coding: utf-8 -*-
"""Time-semantic operators: calendar-aware and event-time transformations.

These operators handle special temporal semantics:
- PIT reporting date lookups
- Event window analysis
- Calendar-based aggregations
- Clock-aware lags

All operators are strictly causal (pit_safe=True) and use dual backends.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.overhaul.base import (
    EPS,
    Spec,
    finite_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    register_specs,
    positive_int,
    nonnegative_int,
)


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Validate exact panel alignment."""
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("time_semantic operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 1. report_asof - 报告期时点值(PIT)
# ---------------------------------------------------------------------------
def pd_report_asof(
    value: pd.DataFrame,
    report_date: pd.DataFrame,
    target_date: pd.DataFrame,
    **_: Any,
) -> pd.DataFrame:
    """Retrieve the most recent report value as-of target date (PIT-safe).

    For each (date, instrument), finds the most recent report where
    report_date <= target_date and returns its value.

    Args:
        value: The reported metric panel
        report_date: Panel of report dates (datetime or ordinal)
        target_date: Panel of target dates (datetime or ordinal)

    Returns:
        Panel with PIT-safe report values

    Notes:
        - Strictly causal: only uses reports published before target_date
        - Returns NaN if no report is available as of target_date
        - Time-zone naive comparison (assumes aligned calendars)
    """
    value, report_date, target_date = _align(value, report_date, target_date)
    val_arr = value.to_numpy(dtype=float)
    rpt_arr = report_date.to_numpy(dtype=float)
    tgt_arr = target_date.to_numpy(dtype=float)
    out = np.full(value.shape, np.nan, dtype=float)

    for col in range(value.shape[1]):
        for row in range(value.shape[0]):
            target = tgt_arr[row, col]
            if not _finite(target):
                continue

            # Find most recent report <= target across all prior rows
            best_val = np.nan
            best_date = -np.inf
            for r in range(row + 1):
                rpt = rpt_arr[r, col]
                val = val_arr[r, col]
                if _finite(rpt) and _finite(val) and rpt <= target:
                    if rpt > best_date:
                        best_date = rpt
                        best_val = val

            if _finite(best_val):
                out[row, col] = best_val

    return frame_pd(value, out)


def pl_report_asof(
    value: "pl.DataFrame",
    report_date: "pl.DataFrame",
    target_date: "pl.DataFrame",
    **_: Any,
) -> "pl.DataFrame":
    """Polars implementation of report_asof."""
    if pl is None:
        raise ImportError("polars is not installed")

    import polars.selectors as cs

    # Convert to pandas, compute, convert back
    # (A full native Polars impl would use window functions per instrument)
    value_pd = value.to_pandas()
    report_pd = report_date.to_pandas()
    target_pd = target_date.to_pandas()
    result_pd = pd_report_asof(value_pd, report_pd, target_pd)
    return pl.from_pandas(result_pd)


# ---------------------------------------------------------------------------
# 2. event_window_return_asof - 事件窗口收益率(PIT)
# ---------------------------------------------------------------------------
def pd_event_window_return_asof(
    price: pd.DataFrame,
    event_date: pd.DataFrame,
    window_start: int = 0,
    window_end: int = 5,
    **_: Any,
) -> pd.DataFrame:
    """Event window return from event_date + window_start to event_date + window_end.

    Computes: (price[event + window_end] - price[event + window_start]) / price[event + window_start]
    where prices are looked up by matching dates in the date panel.

    Args:
        price: Price panel (close/adjusted)
        event_date: Panel of event date ordinals (e.g., row indices or timestamps)
        window_start: Days after event for start price (default 0)
        window_end: Days after event for end price (default 5)

    Returns:
        Panel of event window returns

    Notes:
        - Strictly causal: only uses prices available at observation time
        - Returns NaN if prices are not available in the window
        - window_start < window_end required
        - event_date should contain ordinals (e.g., 0, 1, 2, ...) for row-based lookup
    """
    price, event_date = _align(price, event_date)
    window_start = nonnegative_int(window_start, "window_start")
    window_end = positive_int(window_end, "window_end")
    if window_start >= window_end:
        raise ValueError("window_start must be < window_end")

    price_arr = price.to_numpy(dtype=float)
    event_arr = event_date.to_numpy(dtype=float)
    out = np.full(price.shape, np.nan, dtype=float)

    for col in range(price.shape[1]):
        # Build a map from event ordinal to price for this column
        ordinal_to_price = {}
        for r in range(price.shape[0]):
            ord_val = event_arr[r, col]
            price_val = price_arr[r, col]
            if _finite(ord_val) and _finite(price_val):
                # Use int ordinal as key, store price
                ordinal_to_price[int(ord_val)] = price_val

        for row in range(price.shape[0]):
            event = event_arr[row, col]
            if not _finite(event):
                continue

            event_ord = int(event)
            start_ord = event_ord + window_start
            end_ord = event_ord + window_end

            # Check if we have observed both ordinals by this row
            # (strictly causal: can only use data up to current row)
            start_price = None
            end_price = None

            # Check if start_ord and end_ord are visible up to this row
            for r in range(row + 1):
                ord_val = event_arr[r, col]
                if _finite(ord_val):
                    o = int(ord_val)
                    if o == start_ord and start_price is None:
                        start_price = price_arr[r, col] if _finite(price_arr[r, col]) else None
                    if o == end_ord and end_price is None:
                        end_price = price_arr[r, col] if _finite(price_arr[r, col]) else None

            if start_price is not None and end_price is not None and abs(start_price) > EPS:
                out[row, col] = ((end_price - start_price)) / start_price if start_price != 0 else np.nan

    return frame_pd(price, out)


def pl_event_window_return_asof(
    price: "pl.DataFrame",
    event_date: "pl.DataFrame",
    window_start: int = 0,
    window_end: int = 5,
    **_: Any,
) -> "pl.DataFrame":
    """Polars implementation of event_window_return_asof."""
    if pl is None:
        raise ImportError("polars is not installed")
    price_pd = price.to_pandas()
    event_pd = event_date.to_pandas()
    result_pd = pd_event_window_return_asof(price_pd, event_pd, window_start, window_end)
    return pl.from_pandas(result_pd)


# ---------------------------------------------------------------------------
# 3. financial_snapshot_lag - 财务快照滞后
# ---------------------------------------------------------------------------
def pd_financial_snapshot_lag(
    value: pd.DataFrame,
    snapshot_id: pd.DataFrame,
    lag: int = 1,
    **_: Any,
) -> pd.DataFrame:
    """Lag financial value by snapshot ordinal (not calendar days).

    Returns the value from lag snapshots ago, where snapshots are distinct
    reporting events (not daily forward-fill).

    Args:
        value: Financial metric panel
        snapshot_id: Snapshot identifier panel (string or ordinal)
        lag: Number of snapshots to lag (default 1)

    Returns:
        Lagged financial values

    Notes:
        - Works on distinct snapshots, not daily rows
        - Returns NaN if lag snapshots are not available
        - Strictly causal
    """
    value, snapshot_id = _align(value, snapshot_id)
    lag = positive_int(lag, "lag")

    val_arr = value.to_numpy(dtype=float)
    snap_arr = snapshot_id.to_numpy(dtype=object)
    out = np.full(value.shape, np.nan, dtype=float)

    for col in range(value.shape[1]):
        # Build snapshot history per column
        snapshots = []  # list of (row, snap_id, value)
        seen_snaps = set()
        current_snap = None

        for row in range(value.shape[0]):
            snap = snap_arr[row, col]
            val = val_arr[row, col]

            # Record distinct snapshots only when they first appear
            if snap is not None and snap != "" and str(snap) != "nan":
                snap_key = str(snap)
                if snap_key not in seen_snaps and _finite(val):
                    snapshots.append((row, snap_key, val))
                    seen_snaps.add(snap_key)
                current_snap = snap_key

            # At each row, if we're in a snapshot and have enough history,
            # look back lag snapshots from the current snapshot
            if current_snap is not None:
                # Find current snapshot index
                snap_idx = None
                for i, (_, s_id, _) in enumerate(snapshots):
                    if s_id == current_snap:
                        snap_idx = i
                        break

                # If we have lag snapshots before the current one
                if snap_idx is not None and snap_idx >= lag:
                    lagged_val = snapshots[snap_idx - lag][2]
                    out[row, col] = lagged_val

    return frame_pd(value, out)


def pl_financial_snapshot_lag(
    value: "pl.DataFrame",
    snapshot_id: "pl.DataFrame",
    lag: int = 1,
    **_: Any,
) -> "pl.DataFrame":
    """Polars implementation of financial_snapshot_lag."""
    if pl is None:
        raise ImportError("polars is not installed")
    value_pd = value.to_pandas()
    snapshot_pd = snapshot_id.to_pandas()
    result_pd = pd_financial_snapshot_lag(value_pd, snapshot_pd, lag)
    return pl.from_pandas(result_pd)


# ---------------------------------------------------------------------------
# 4. same_calendar_day_mean - 相同日历日均值
# ---------------------------------------------------------------------------
def pd_same_calendar_day_mean(
    value: pd.DataFrame,
    window: int = 252,
    min_periods: int = 3,
    **_: Any,
) -> pd.DataFrame:
    """Mean of values on the same calendar day-of-year across prior years.

    Computes the trailing average of values from the same calendar day
    (month-day) in prior years within the window.

    Args:
        value: Input panel
        window: Maximum lookback days (default 252)
        min_periods: Minimum observations required (default 3)

    Returns:
        Panel of same-day means

    Notes:
        - Uses calendar day-of-year (month-day) matching
        - Strictly causal: only uses prior observations
        - Useful for seasonal pattern analysis
    """
    value = _align(value)[0]
    window = positive_int(window, "window")
    min_periods = positive_int(min_periods, "min_periods")

    val_arr = value.to_numpy(dtype=float)
    out = np.full(value.shape, np.nan, dtype=float)

    if not isinstance(value.index, pd.DatetimeIndex):
        # If not datetime index, cannot extract calendar day
        return frame_pd(value, out)

    dates = value.index

    for row in range(value.shape[0]):
        current_date = dates[row]
        target_month = current_date.month
        target_day = current_date.day

        for col in range(value.shape[1]):
            # Find all prior rows with same month-day within window
            matches = []
            for r in range(row):
                past_date = dates[r]
                if past_date.month == target_month and past_date.day == target_day:
                    days_ago = (current_date - past_date).days
                    if days_ago <= window:
                        v = val_arr[r, col]
                        if _finite(v):
                            matches.append(v)

            if len(matches) >= min_periods:
                out[row, col] = float(np.mean(matches))

    return frame_pd(value, out)


def pl_same_calendar_day_mean(
    value: "pl.DataFrame",
    window: int = 252,
    min_periods: int = 3,
    **_: Any,
) -> "pl.DataFrame":
    """Polars implementation of same_calendar_day_mean."""
    if pl is None:
        raise ImportError("polars is not installed")
    value_pd = value.to_pandas()
    result_pd = pd_same_calendar_day_mean(value_pd, window, min_periods)
    return pl.from_pandas(result_pd)


# ---------------------------------------------------------------------------
# 5. same_calendar_month_return - 相同日历月收益
# ---------------------------------------------------------------------------
def pd_same_calendar_month_return(
    value: pd.DataFrame,
    window: int = 5,
    min_periods: int = 2,
    **_: Any,
) -> pd.DataFrame:
    """Mean return for the same calendar month across prior years.

    Computes the average within-month return for the same calendar month
    in prior years within the window.

    Args:
        value: Price panel
        window: Maximum lookback years (default 5)
        min_periods: Minimum observations required (default 2)

    Returns:
        Panel of same-month average returns

    Notes:
        - Calculates monthly return as (end - start) / start
        - Only uses complete prior months
        - Strictly causal
    """
    value = _align(value)[0]
    window = positive_int(window, "window")
    min_periods = positive_int(min_periods, "min_periods")

    val_arr = value.to_numpy(dtype=float)
    out = np.full(value.shape, np.nan, dtype=float)

    if not isinstance(value.index, pd.DatetimeIndex):
        return frame_pd(value, out)

    dates = value.index

    for row in range(value.shape[0]):
        current_date = dates[row]
        target_month = current_date.month

        for col in range(value.shape[1]):
            # Find monthly returns for same month in prior years
            monthly_returns = []

            for year_offset in range(1, window + 1):
                # Look for data from year_offset years ago, same month
                target_year = current_date.year - year_offset

                # Find start and end of that month
                month_start = None
                month_end = None
                month_start_val = np.nan
                month_end_val = np.nan

                for r in range(row):
                    d = dates[r]
                    if d.year == target_year and d.month == target_month:
                        v = val_arr[r, col]
                        if _finite(v):
                            if month_start is None:
                                month_start = d
                                month_start_val = v
                            month_end = d
                            month_end_val = v

                # Calculate return if we have both start and end
                if month_start is not None and month_end is not None:
                    if _finite(month_start_val) and _finite(month_end_val) and abs(month_start_val) > EPS:
                        ret = ((month_end_val - month_start_val)) / month_start_val if month_start_val != 0 else np.nan
                        monthly_returns.append(ret)

            if len(monthly_returns) >= min_periods:
                out[row, col] = float(np.mean(monthly_returns))

    return frame_pd(value, out)


def pl_same_calendar_month_return(
    value: "pl.DataFrame",
    window: int = 5,
    min_periods: int = 2,
    **_: Any,
) -> "pl.DataFrame":
    """Polars implementation of same_calendar_month_return."""
    if pl is None:
        raise ImportError("polars is not installed")
    value_pd = value.to_pandas()
    result_pd = pd_same_calendar_month_return(value_pd, window, min_periods)
    return pl.from_pandas(result_pd)


# ---------------------------------------------------------------------------
# 6. same_clock_lag - 相同时钟滞后
# ---------------------------------------------------------------------------
def pd_same_clock_lag(
    value: pd.DataFrame,
    clock: pd.DataFrame,
    lag: int = 1,
    **_: Any,
) -> pd.DataFrame:
    """Lag value by clock ordinal (not calendar time).

    Returns the value from lag clock ticks ago, where clock defines
    the time progression (e.g., trading days, business events).

    Args:
        value: Input panel
        clock: Clock ordinal panel (monotonic within each instrument)
        lag: Number of clock ticks to lag (default 1)

    Returns:
        Clock-lagged values

    Notes:
        - Clock must be monotonic (non-decreasing) per instrument
        - Lags by clock ticks, not calendar rows
        - Returns NaN if lag clock ticks not available
    """
    value, clock = _align(value, clock)
    lag = positive_int(lag, "lag")

    val_arr = value.to_numpy(dtype=float)
    clk_arr = clock.to_numpy(dtype=float)
    out = np.full(value.shape, np.nan, dtype=float)

    for col in range(value.shape[1]):
        # Build clock -> (row, value) mapping
        clock_hist = {}  # clock_ordinal -> (row, value)

        for row in range(value.shape[0]):
            clk = clk_arr[row, col]
            val = val_arr[row, col]

            if _finite(clk):
                clock_ordinal = int(clk)

                # Record this clock tick
                if _finite(val):
                    clock_hist[clock_ordinal] = (row, val)

                # Look back lag clock ticks
                target_clock = clock_ordinal - lag
                if target_clock in clock_hist:
                    _, lagged_val = clock_hist[target_clock]
                    out[row, col] = lagged_val

    return frame_pd(value, out)


def pl_same_clock_lag(
    value: "pl.DataFrame",
    clock: "pl.DataFrame",
    lag: int = 1,
    **_: Any,
) -> "pl.DataFrame":
    """Polars implementation of same_clock_lag."""
    if pl is None:
        raise ImportError("polars is not installed")
    value_pd = value.to_pandas()
    clock_pd = clock.to_pandas()
    result_pd = pd_same_clock_lag(value_pd, clock_pd, lag)
    return pl.from_pandas(result_pd)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry as _Reg
    # Idempotent re-registration (P0 collection baseline): ``panel_batch1``
    # already registers ``report_asof``/``event_window_return_asof`` with a
    # different (panel-gap) contract.  A direct import of this module after
    # ``load_all()`` must not re-register those canonicals and trip the R6-157
    # param_specs invariant.  Only register canonicals not already present.
    _specs = {
        "report_asof": Spec(
            "ts",
            ["value", "report_date", "target_date"],
            "PIT-safe report value lookup as-of target date",
            pd_report_asof,
            pl_report_asof,
        ),
        "event_window_return_asof": Spec(
            "ts",
            ["price", "event_date", "window_start", "window_end"],
            "Event window return (strictly causal)",
            pd_event_window_return_asof,
            pl_event_window_return_asof,
        ),
        "financial_snapshot_lag": Spec(
            "ts",
            ["value", "snapshot_id", "lag"],
            "Lag by financial snapshot ordinal (not calendar days)",
            pd_financial_snapshot_lag,
            pl_financial_snapshot_lag,
        ),
        "same_calendar_day_mean": Spec(
            "ts",
            ["value", "window", "min_periods"],
            "Mean of values on same calendar day across prior years",
            pd_same_calendar_day_mean,
            pl_same_calendar_day_mean,
        ),
        "same_calendar_month_return": Spec(
            "ts",
            ["value", "window", "min_periods"],
            "Mean return for same calendar month across prior years",
            pd_same_calendar_month_return,
            pl_same_calendar_month_return,
        ),
        "same_clock_lag": Spec(
            "ts",
            ["value", "clock", "lag"],
            "Lag by clock ordinal (not calendar rows)",
            pd_same_clock_lag,
            pl_same_clock_lag,
        ),
    }
    _to_register = {
        name: spec
        for name, spec in _specs.items()
        if _Reg.get(name, "pandas_numpy", mode="any") is None
    }
    if _to_register:
        register_specs(_to_register)


register()

# Register to EXTENDED_ONLY_CANONICALS
try:
    from factor_engine.cleaned_operators.operator_surface import extend_extended_only

    extend_extended_only([
        "report_asof",
        "event_window_return_asof",
        "financial_snapshot_lag",
        "same_calendar_day_mean",
        "same_calendar_month_return",
        "same_clock_lag",
    ])
except ImportError:  # pragma: no cover
    pass
