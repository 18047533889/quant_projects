# -*- coding: utf-8
"""Minute-native, session-aware transforms.

These helpers deliberately accept Series instead of engine-specific operator
objects so both logical-source runtimes and cleaned operators can share the
same reset semantics.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from factor_engine.runtime.session_calendar import SessionCalendar


def _calendar(calendar: SessionCalendar | None) -> SessionCalendar:
    return calendar or SessionCalendar()


def _session_groups(index: pd.Index, calendar: SessionCalendar | None = None) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("session-aware minute helpers require DatetimeIndex")
    cal = _calendar(calendar)
    return pd.Series(cal.session_key(index), index=index)


def session_pct_change(
    series: pd.Series,
    periods: int = 1,
    *,
    calendar: SessionCalendar | None = None,
) -> pd.Series:
    """Percentage change reset at every explicit trading session."""
    values = pd.Series(series)
    groups = _session_groups(values.index, calendar)
    return values.groupby(groups, group_keys=False).pct_change(max(1, int(periods)))


def session_log_return(
    series: pd.Series,
    periods: int = 1,
    *,
    calendar: SessionCalendar | None = None,
) -> pd.Series:
    """Log return reset at every explicit trading session."""
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    ratio = values.groupby(_session_groups(values.index, calendar), group_keys=False).pct_change(
        max(1, int(periods))
    ) + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(ratio)


def session_rolling(
    series: pd.Series,
    window: int,
    aggregation: str | Callable[[pd.Series], float],
    *,
    min_periods: int | None = None,
    calendar: SessionCalendar | None = None,
) -> pd.Series:
    """Rolling calculation that cannot consume bars from a prior session."""
    values = pd.Series(series)
    width = max(1, int(window))
    minimum = width if min_periods is None else max(1, int(min_periods))

    def calculate(group: pd.Series) -> pd.Series:
        roller = group.rolling(width, min_periods=minimum)
        if callable(aggregation):
            return roller.apply(aggregation, raw=False)
        method = getattr(roller, str(aggregation), None)
        if not callable(method):
            raise ValueError(f"unsupported session rolling aggregation {aggregation!r}")
        return method()

    groups = _session_groups(values.index, calendar)
    return values.groupby(groups, group_keys=False).apply(calculate).reindex(values.index)


def session_cumulative_vwap(
    amount: pd.Series,
    volume: pd.Series,
    *,
    calendar: SessionCalendar | None = None,
) -> pd.Series:
    """Cumulative session VWAP defined as cumulative Amount / Volume."""
    amount_values = pd.to_numeric(pd.Series(amount), errors="coerce")
    volume_values = pd.to_numeric(pd.Series(volume), errors="coerce").reindex(
        amount_values.index
    )
    groups = _session_groups(amount_values.index, calendar)
    numerator = amount_values.groupby(groups).cumsum()
    denominator = volume_values.groupby(groups).cumsum().replace(0, np.nan)
    return numerator / denominator
