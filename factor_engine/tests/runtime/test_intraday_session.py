from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.ashare_intraday import ashare_limit_prices, limit_touch_fraction
from runtime.intraday_session import session_cumulative_vwap, session_log_return, session_rolling
from runtime.session_calendar import SessionCalendar


def test_ashare_calendar_excludes_lunch_and_splits_bars() -> None:
    calendar = SessionCalendar.ashare(timestamp_convention="bar_end")
    timestamps = pd.DatetimeIndex(
        ["2024-01-02 09:31", "2024-01-02 11:30", "2024-01-02 12:00", "2024-01-02 13:01"]
    )
    assert calendar.minute_ordinal(timestamps).tolist() == [0, 119, -1, 120]
    assert calendar.segment_ordinal(timestamps).tolist() == [0, 119, -1, 0]


def test_native_returns_and_rolling_reset_by_session() -> None:
    index = pd.DatetimeIndex(["2024-01-02 14:59", "2024-01-02 15:00", "2024-01-03 09:31"])
    values = pd.Series([100.0, 101.0, 200.0], index=index)
    assert pd.isna(session_log_return(values).iloc[-1])
    assert session_rolling(values, 2, "sum", min_periods=1).iloc[-1] == 200.0


def test_session_cumulative_vwap_is_amount_over_volume() -> None:
    index = pd.DatetimeIndex(["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-03 09:31"])
    amount = pd.Series([1000.0, 2100.0, 600.0], index=index)
    volume = pd.Series([10.0, 20.0, 5.0], index=index)
    out = session_cumulative_vwap(amount, volume)
    assert out.iloc[1] == pytest.approx(3100.0 / 30.0)
    assert out.iloc[2] == pytest.approx(120.0)


def test_ashare_limit_prices_and_touch_fraction() -> None:
    upper, lower = ashare_limit_prices(10.0, "600000.SH", "2024-01-02")
    assert (upper, lower) == (11.0, 9.0)
    bars = pd.DataFrame({"high": [10.9, 11.0], "low": [9.2, 9.0]})
    assert limit_touch_fraction(bars, upper, direction="up") == pytest.approx(0.5)
    assert limit_touch_fraction(bars, lower, direction="down") == pytest.approx(0.5)


def test_chinext_limit_rate_after_registration_reform() -> None:
    upper, lower = ashare_limit_prices(10.0, "300001.SZ", "2024-08-26")
    assert (upper, lower) == (12.0, 8.0)
