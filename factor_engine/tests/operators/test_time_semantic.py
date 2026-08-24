# -*- coding: utf-8 -*-
"""Tests for time_semantic operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.time_semantic import (
    pd_report_asof,
    pd_event_window_return_asof,
    pd_financial_snapshot_lag,
    pd_same_calendar_day_mean,
    pd_same_calendar_month_return,
    pd_same_clock_lag,
)


@pytest.fixture
def simple_panel():
    """Simple panel for testing."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    values = pd.DataFrame(
        {
            "A": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 32.0],
            "B": [100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 115.0, 118.0, 120.0, 122.0],
        },
        index=index,
    )
    return values


@pytest.fixture
def report_panel():
    """Panel with report dates and values."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    values = pd.DataFrame(
        {
            "A": [np.nan, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 125.0, 125.0, 125.0],
            "B": [50.0, 50.0, 50.0, 48.0, 48.0, 48.0, 52.0, 52.0, 52.0, 55.0],
        },
        index=index,
    )
    report_dates = pd.DataFrame(
        {
            "A": [np.nan, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 7.0, 7.0, 7.0],
            "B": [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 9.0],
        },
        index=index,
    )
    target_dates = pd.DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "B": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        },
        index=index,
    )
    return values, report_dates, target_dates


def test_report_asof_basic(report_panel):
    """Test report_asof retrieves most recent report."""
    values, report_dates, target_dates = report_panel
    result = pd_report_asof(values, report_dates, target_dates)

    assert result.shape == values.shape

    # A: no report at target 0
    assert np.isnan(result.iloc[0, 0])
    # A: report at date 1, target 1-3 should get 100
    assert np.isclose(result.iloc[1, 0], 100.0)
    assert np.isclose(result.iloc[2, 0], 100.0)
    assert np.isclose(result.iloc[3, 0], 100.0)
    # A: report at date 4, target 4-6 should get 110
    assert np.isclose(result.iloc[4, 0], 110.0)
    assert np.isclose(result.iloc[5, 0], 110.0)
    assert np.isclose(result.iloc[6, 0], 110.0)
    # A: report at date 7, target 7+ should get 125
    assert np.isclose(result.iloc[7, 0], 125.0)

    # B: report at date 0, target 0-2 should get 50
    assert np.isclose(result.iloc[0, 1], 50.0)
    assert np.isclose(result.iloc[1, 1], 50.0)
    assert np.isclose(result.iloc[2, 1], 50.0)
    # B: report at date 3, target 3-5 should get 48
    assert np.isclose(result.iloc[3, 1], 48.0)


def test_report_asof_no_report():
    """Test report_asof when no report is available."""
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = pd.DataFrame({"A": [np.nan, np.nan, 100.0, 100.0, 100.0]}, index=index)
    report_dates = pd.DataFrame({"A": [np.nan, np.nan, 2.0, 2.0, 2.0]}, index=index)
    target_dates = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=index)

    result = pd_report_asof(values, report_dates, target_dates)

    # No report before target 0-1
    assert np.isnan(result.iloc[0, 0])
    assert np.isnan(result.iloc[1, 0])
    # Report at date 2
    assert np.isclose(result.iloc[2, 0], 100.0)


def test_event_window_return_asof_basic():
    """Test event window return calculation."""
    index = pd.date_range("2024-01-01", periods=11, freq="D")
    price = pd.DataFrame(
        {"A": [100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 112.0, 115.0, 118.0, 120.0, 125.0]},
        index=index,
    )
    # At row 10, event=0 means window [0, 5]
    # Ordinals 0-9 available at rows 0-9, plus event=0 at row 10
    event_date = pd.DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0]},
        index=index,
    )

    result = pd_event_window_return_asof(price, event_date, window_start=0, window_end=5)

    # At row 10: event=0, window [0,5]
    # start_ord=0 (price 100), end_ord=5 (price 110)
    # Return = (110 - 100) / 100 = 0.1
    assert np.isclose(result.iloc[10, 0], 0.1, atol=1e-6)


def test_event_window_return_asof_invalid_window():
    """Test event window return with invalid window."""
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    price = pd.DataFrame({"A": [100.0, 102.0, 105.0, 103.0, 108.0]}, index=index)
    event_date = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=index)

    with pytest.raises(ValueError, match="window_start must be < window_end"):
        pd_event_window_return_asof(price, event_date, window_start=5, window_end=5)


def test_financial_snapshot_lag_basic():
    """Test financial snapshot lag."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    values = pd.DataFrame(
        {"A": [100.0, 100.0, 110.0, 110.0, 125.0, 125.0, 145.0, 145.0]},
        index=index,
    )
    snapshot_id = pd.DataFrame(
        {"A": ["2023Q1", "2023Q1", "2023Q2", "2023Q2", "2023Q3", "2023Q3", "2023Q4", "2023Q4"]},
        index=index,
    )

    result = pd_financial_snapshot_lag(values, snapshot_id, lag=1)

    assert result.shape == values.shape

    # First snapshot (Q1) at rows 0-1, no lag available
    assert np.isnan(result.iloc[0, 0])
    assert np.isnan(result.iloc[1, 0])
    # Second snapshot (Q2) at rows 2-3, lag=1 should get Q1 (100)
    assert np.isclose(result.iloc[2, 0], 100.0)
    assert np.isclose(result.iloc[3, 0], 100.0)
    # Third snapshot (Q3) at rows 4-5, lag=1 should get Q2 (110)
    assert np.isclose(result.iloc[4, 0], 110.0)
    assert np.isclose(result.iloc[5, 0], 110.0)
    # Fourth snapshot (Q4) at rows 6-7, lag=1 should get Q3 (125)
    assert np.isclose(result.iloc[6, 0], 125.0)
    assert np.isclose(result.iloc[7, 0], 125.0)


def test_financial_snapshot_lag_multiple():
    """Test financial snapshot lag with lag=2."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    values = pd.DataFrame(
        {"A": [100.0, np.nan, 110.0, np.nan, 125.0, np.nan, 145.0, np.nan, 160.0, np.nan]},
        index=index,
    )
    snapshot_id = pd.DataFrame(
        {"A": ["Q1", None, "Q2", None, "Q3", None, "Q4", None, "Q5", None]},
        index=index,
    )

    result = pd_financial_snapshot_lag(values, snapshot_id, lag=2)

    # Need at least 2 snapshots before having lag=2
    assert np.isnan(result.iloc[0, 0])
    assert np.isnan(result.iloc[2, 0])
    # At row 4 (Q3), lag=2 should get Q1 (100)
    assert np.isclose(result.iloc[4, 0], 100.0)
    # At row 6 (Q4), lag=2 should get Q2 (110)
    assert np.isclose(result.iloc[6, 0], 110.0)


def test_same_calendar_day_mean_basic():
    """Test same calendar day mean."""
    # Create dates with same month-day across years
    dates = pd.DatetimeIndex([
        "2022-01-15", "2022-02-15", "2022-03-15",
        "2023-01-15", "2023-02-15", "2023-03-15",
        "2024-01-15", "2024-02-15", "2024-03-15",
    ])
    values = pd.DataFrame(
        {"A": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0]},
        index=dates,
    )

    result = pd_same_calendar_day_mean(values, window=1000, min_periods=2)

    # At 2024-01-15 (row 6), should average 2022-01-15 (10) and 2023-01-15 (15) = 12.5
    assert np.isclose(result.iloc[6, 0], 12.5)
    # At 2024-02-15 (row 7), should average 2022-02-15 (20) and 2023-02-15 (25) = 22.5
    assert np.isclose(result.iloc[7, 0], 22.5)


def test_same_calendar_day_mean_insufficient():
    """Test same calendar day mean with insufficient data."""
    dates = pd.DatetimeIndex([
        "2023-01-15", "2023-02-15", "2024-01-15", "2024-02-15",
    ])
    values = pd.DataFrame({"A": [10.0, 20.0, 15.0, 25.0]}, index=dates)

    result = pd_same_calendar_day_mean(values, window=500, min_periods=3)

    # Only 1 prior observation for each date, need 3
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[3, 0])


def test_same_calendar_month_return_basic():
    """Test same calendar month return."""
    dates = pd.DatetimeIndex([
        "2022-01-01", "2022-01-15", "2022-01-31",  # Jan 2022: 100 -> 110
        "2023-01-01", "2023-01-15", "2023-01-31",  # Jan 2023: 105 -> 120
        "2024-01-01", "2024-01-31",                # Jan 2024
    ])
    values = pd.DataFrame(
        {"A": [100.0, 105.0, 110.0, 105.0, 112.0, 120.0, 108.0, 125.0]},
        index=dates,
    )

    result = pd_same_calendar_month_return(values, window=5, min_periods=2)

    # At 2024-01-31 (row 7):
    # Jan 2022 return: (110 - 100) / 100 = 0.1
    # Jan 2023 return: (120 - 105) / 105 ≈ 0.1429
    # Mean ≈ 0.1214
    expected = ((110.0 - 100.0) / 100.0 + (120.0 - 105.0) / 105.0) / 2.0
    assert np.isclose(result.iloc[7, 0], expected, atol=1e-4)


def test_same_calendar_month_return_insufficient():
    """Test same calendar month return with insufficient history."""
    dates = pd.DatetimeIndex([
        "2023-01-01", "2023-01-31",
        "2024-01-01", "2024-01-31",
    ])
    values = pd.DataFrame({"A": [100.0, 110.0, 105.0, 120.0]}, index=dates)

    result = pd_same_calendar_month_return(values, window=5, min_periods=2)

    # Only 1 prior January, need 2
    assert np.isnan(result.iloc[3, 0])


def test_same_clock_lag_basic():
    """Test same clock lag."""
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    values = pd.DataFrame(
        {"A": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 32.0]},
        index=index,
    )
    clock = pd.DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]},
        index=index,
    )

    result = pd_same_clock_lag(values, clock, lag=1)

    # Clock 0 has no lag
    assert np.isnan(result.iloc[0, 0])
    # Clock 1 should get value from clock 0 (10)
    assert np.isclose(result.iloc[1, 0], 10.0)
    # Clock 2 should get value from clock 1 (12)
    assert np.isclose(result.iloc[2, 0], 12.0)
    # Clock 9 should get value from clock 8 (30)
    assert np.isclose(result.iloc[9, 0], 30.0)


def test_same_clock_lag_non_contiguous():
    """Test same clock lag with non-contiguous clock."""
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    values = pd.DataFrame(
        {"A": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0]},
        index=index,
    )
    # Clock with gaps
    clock = pd.DataFrame(
        {"A": [0.0, 1.0, 1.0, 3.0, 5.0, 5.0, 7.0, 9.0]},
        index=index,
    )

    result = pd_same_clock_lag(values, clock, lag=2)

    # Clock 0 at row 0, no lag=2
    assert np.isnan(result.iloc[0, 0])
    # Clock 1 at row 1, no lag=2
    assert np.isnan(result.iloc[1, 0])
    # Clock 3 at row 3, lag=2 should get clock 1 (12)
    assert np.isclose(result.iloc[3, 0], 12.0)
    # Clock 5 at row 4, lag=2 should get clock 3 (18)
    assert np.isclose(result.iloc[4, 0], 18.0)
    # Clock 7 at row 6, lag=2 should get clock 5 (20)
    assert np.isclose(result.iloc[6, 0], 20.0)


def test_same_clock_lag_validation():
    """Test same clock lag parameter validation."""
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = pd.DataFrame({"A": [10.0, 12.0, 15.0, 18.0, 20.0]}, index=index)
    clock = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=index)

    with pytest.raises((TypeError, ValueError)):
        pd_same_clock_lag(values, clock, lag=0)

    with pytest.raises((TypeError, ValueError)):
        pd_same_clock_lag(values, clock, lag=-1)


def test_alignment_validation():
    """Test that operators validate panel alignment."""
    index1 = pd.date_range("2024-01-01", periods=5, freq="D")
    index2 = pd.date_range("2024-01-02", periods=5, freq="D")

    values1 = pd.DataFrame({"A": [10.0, 12.0, 15.0, 18.0, 20.0]}, index=index1)
    values2 = pd.DataFrame({"A": [10.0, 12.0, 15.0, 18.0, 20.0]}, index=index2)

    with pytest.raises(ValueError, match="not aligned"):
        pd_report_asof(values1, values2, values1)


def test_parameter_validation():
    """Test parameter validation."""
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = pd.DataFrame({"A": [10.0, 12.0, 15.0, 18.0, 20.0]}, index=index)

    # Test invalid window
    with pytest.raises((TypeError, ValueError)):
        pd_same_calendar_day_mean(values, window=-1)

    # Test invalid min_periods
    with pytest.raises((TypeError, ValueError)):
        pd_same_calendar_month_return(values, min_periods=0)

    # Test invalid lag
    with pytest.raises((TypeError, ValueError)):
        pd_financial_snapshot_lag(values, values, lag=0)
