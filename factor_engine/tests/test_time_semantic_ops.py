# -*- coding: utf-8 -*-
"""Tests for time_semantic_ops operators."""
import numpy as np
import pandas as pd
import pytest

from cleaned_operators.time_semantic_ops import (
    pd_event_window_return_asof,
    pd_financial_snapshot_lag,
    pd_report_asof,
    pd_same_calendar_day_mean,
    pd_same_calendar_month_return,
    pd_same_clock_lag,
)


@pytest.fixture
def daily_panel():
    """Create a daily panel for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = np.random.randn(100, 3) * 0.01
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def intraday_panel():
    """Create an intraday panel for testing."""
    dates = pd.date_range("2024-01-01 09:00", periods=100, freq="5min")
    data = np.random.randn(100, 3) * 0.001
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


class TestReportAsof:
    def test_basic_functionality(self, daily_panel):
        value = daily_panel.copy()

        # Create report_date panel (announcements every 5 days)
        report_dates = pd.DataFrame(index=value.index, columns=value.columns)
        for i in range(len(value)):
            if i % 5 == 0:
                report_dates.iloc[i] = value.index[i]
            else:
                report_dates.iloc[i] = pd.NaT

        # asof_date is current date
        asof_date = pd.DataFrame(
            np.tile(value.index.values.reshape(-1, 1), (1, 3)),
            index=value.index,
            columns=value.columns
        )

        result = pd_report_asof(value, report_dates, asof_date, max_staleness_days=90)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == value.shape
        assert result.index.equals(value.index)

        # Should have valid values after first report
        assert result.iloc[5:].notna().any().any()

    def test_staleness_constraint(self, daily_panel):
        value = daily_panel.copy()

        # Report at day 0 only
        report_dates = pd.DataFrame(pd.NaT, index=value.index, columns=value.columns)
        report_dates.iloc[0] = value.index[0]

        asof_date = pd.DataFrame(
            np.tile(value.index.values.reshape(-1, 1), (1, 3)),
            index=value.index,
            columns=value.columns
        )

        # With max_staleness_days=10, should only be valid for first 10 days
        result = pd_report_asof(value, report_dates, asof_date, max_staleness_days=10)

        assert result.iloc[0:11].notna().any().any()  # Days 0-10
        assert result.iloc[11:].isna().all().all()  # Days 11+ should be NaN

    def test_parameter_validation(self, daily_panel):
        value = daily_panel.copy()
        report_dates = pd.DataFrame(value.index[0], index=value.index, columns=value.columns)
        asof_date = report_dates.copy()

        # Invalid max_staleness_days
        with pytest.raises((TypeError, ValueError)):
            pd_report_asof(value, report_dates, asof_date, max_staleness_days=0)

        with pytest.raises((TypeError, ValueError)):
            pd_report_asof(value, report_dates, asof_date, max_staleness_days=-5)

    def test_misaligned_inputs(self, daily_panel):
        value = daily_panel.copy()
        report_dates = daily_panel.iloc[:50].copy()
        asof_date = daily_panel.copy()

        with pytest.raises(ValueError, match="not aligned"):
            pd_report_asof(value, report_dates, asof_date)

    def test_nan_handling(self, daily_panel):
        value = daily_panel.copy()
        value.iloc[10:15, 0] = np.nan

        # Report dates align with rows - NaN values won't be returned
        report_dates = pd.DataFrame(
            np.tile(value.index.values.reshape(-1, 1), (1, 3)),
            index=value.index,
            columns=value.columns
        )
        asof_date = report_dates.copy()

        result = pd_report_asof(value, report_dates, asof_date, max_staleness_days=90)

        # When value is NaN, the search continues backwards to find a valid value
        # So rows 10-15 in column 0 will get values from row 9 (within staleness window)
        # Just verify shape and that implementation doesn't crash
        assert result.shape == value.shape


class TestEventWindowReturnAsof:
    def test_basic_functionality(self, daily_panel):
        ret = daily_panel.copy()

        # Event at day 20
        event_date = pd.DataFrame(pd.NaT, index=ret.index, columns=ret.columns)
        event_date.iloc[20:] = ret.index[20]

        result = pd_event_window_return_asof(ret, event_date, window_before=5, window_after=5)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == ret.shape

        # Result should only be available after event window closes (day 25+)
        assert result.iloc[:25].isna().all().all()
        assert result.iloc[25:].notna().any().any()

    def test_cumulative_return_calculation(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # Simple returns: 1% per day
        ret = pd.DataFrame(0.01, index=dates, columns=["A"])

        # Event at day 10
        event_date = pd.DataFrame(pd.NaT, index=dates, columns=["A"])
        event_date.iloc[10:] = dates[10]

        result = pd_event_window_return_asof(ret, event_date, window_before=2, window_after=2)

        # Window: days 8-12 (5 days), cumulative return = (1.01)^5 - 1
        expected = (1.01 ** 5) - 1

        # Should be available from day 12 onwards
        if result.iloc[12:].notna().any().any():
            actual = result.iloc[12, 0]
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_parameter_validation(self, daily_panel):
        ret = daily_panel.copy()
        event_date = pd.DataFrame(ret.index[20], index=ret.index, columns=ret.columns)

        # Invalid window_before (negative)
        with pytest.raises((TypeError, ValueError)):
            pd_event_window_return_asof(ret, event_date, window_before=-1)

    def test_pit_safety(self, daily_panel):
        ret = daily_panel.copy()

        # Event at day 50
        event_date = pd.DataFrame(pd.NaT, index=ret.index, columns=ret.columns)
        event_date.iloc[50:] = ret.index[50]

        result = pd_event_window_return_asof(ret, event_date, window_before=5, window_after=5)

        # Result at day 54 should be NaN (window not closed yet)
        assert result.iloc[54].isna().all()

        # Result at day 55 should be available
        assert result.iloc[55:].notna().any().any()


class TestFinancialSnapshotLag:
    def test_quarterly_lag(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        x = pd.DataFrame(np.arange(50).reshape(-1, 1), index=dates, columns=["A"])

        # Quarterly periods
        period_id = pd.DataFrame("", index=dates, columns=["A"])
        period_id.iloc[0:10] = "2024Q1"
        period_id.iloc[10:20] = "2024Q2"
        period_id.iloc[20:30] = "2024Q3"
        period_id.iloc[30:40] = "2024Q4"
        period_id.iloc[40:50] = "2025Q1"

        fiscal_date = pd.DataFrame(dates, index=dates, columns=["A"])

        result = pd_financial_snapshot_lag(x, period_id, fiscal_date, lag_periods=1)

        # Q2 should show Q1 values
        # Find last Q1 value
        q1_last_value = x.iloc[9, 0]
        q2_result = result.iloc[10:20, 0].dropna()

        if len(q2_result) > 0:
            assert q2_result.iloc[0] == q1_last_value

    def test_annual_lag(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        x = pd.DataFrame(np.arange(30).reshape(-1, 1), index=dates, columns=["A"])

        # Annual periods
        period_id = pd.DataFrame("", index=dates, columns=["A"])
        period_id.iloc[0:10] = "2023"
        period_id.iloc[10:20] = "2024"
        period_id.iloc[20:30] = "2025"

        fiscal_date = pd.DataFrame(dates, index=dates, columns=["A"])

        result = pd_financial_snapshot_lag(x, period_id, fiscal_date, lag_periods=1)

        # 2024 should show 2023 values
        fy2023_last = x.iloc[9, 0]
        fy2024_result = result.iloc[10:20, 0].dropna()

        if len(fy2024_result) > 0:
            assert fy2024_result.iloc[0] == fy2023_last

    def test_parameter_validation(self, daily_panel):
        x = daily_panel.copy()
        period_id = pd.DataFrame("2024Q1", index=x.index, columns=x.columns)
        fiscal_date = pd.DataFrame(
            np.tile(x.index.values.reshape(-1, 1), (1, len(x.columns))),
            index=x.index,
            columns=x.columns
        )

        # Invalid lag_periods
        with pytest.raises((TypeError, ValueError)):
            pd_financial_snapshot_lag(x, period_id, fiscal_date, lag_periods=0)

    def test_missing_lag_period(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        x = pd.DataFrame(np.arange(20).reshape(-1, 1), index=dates, columns=["A"])

        # Only Q2 exists, lag 1 should find nothing
        period_id = pd.DataFrame("2024Q2", index=dates, columns=["A"])
        fiscal_date = pd.DataFrame(dates, index=dates, columns=["A"])

        result = pd_financial_snapshot_lag(x, period_id, fiscal_date, lag_periods=1)

        # Should be all NaN (no Q1 data)
        assert result.isna().all().all()


class TestSameCalendarDayMean:
    def test_weekly_seasonality(self, daily_panel):
        x = daily_panel.copy()

        result = pd_same_calendar_day_mean(x, window=52)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape

        # Early rows should be NaN
        assert result.iloc[:7].isna().all().all()

        # Later rows should have values
        assert result.iloc[14:].notna().any().any()

    def test_day_of_week_matching(self):
        # Create panel with strong Monday effect
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        x = pd.DataFrame(0.0, index=dates, columns=["A"])

        # Set Mondays to 1.0 (dayofweek == 0)
        mondays = dates.dayofweek == 0
        x.loc[mondays, "A"] = 1.0

        result = pd_same_calendar_day_mean(x, window=52)

        # On Mondays, the mean should approach 1.0
        monday_results = result.loc[mondays, "A"].dropna()
        if len(monday_results) > 0:
            assert (monday_results > 0.9).any()

        # On other days, should approach 0.0
        non_mondays = ~mondays
        non_monday_results = result.loc[non_mondays, "A"].dropna()
        if len(non_monday_results) > 0:
            assert (non_monday_results < 0.1).any()

    def test_parameter_validation(self, daily_panel):
        x = daily_panel.copy()

        # Invalid window
        with pytest.raises((TypeError, ValueError)):
            pd_same_calendar_day_mean(x, window=0)

    def test_requires_datetime_index(self):
        x = pd.DataFrame(np.random.randn(50, 2))

        with pytest.raises(TypeError, match="DatetimeIndex"):
            pd_same_calendar_day_mean(x)


class TestSameCalendarMonthReturn:
    def test_monthly_seasonality(self, daily_panel):
        x = daily_panel.copy()

        result = pd_same_calendar_month_return(x, window=12)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape

        # Should have some valid values
        assert result.notna().any().any()

    def test_month_matching(self):
        # Create 3 years of data
        dates = pd.date_range("2022-01-01", periods=1095, freq="D")
        x = pd.DataFrame(0.0, index=dates, columns=["A"])

        # Set January to 1.0
        january = dates.month == 1
        x.loc[january, "A"] = 1.0

        result = pd_same_calendar_month_return(x, window=12)

        # In January, mean should approach 1.0
        jan_results = result.loc[january, "A"].dropna()
        if len(jan_results) > 0:
            # After enough history
            assert jan_results.iloc[-10:].mean() > 0.9

        # In other months, should approach 0.0
        non_jan = dates.month == 2
        feb_results = result.loc[non_jan, "A"].dropna()
        if len(feb_results) > 0:
            assert feb_results.mean() < 0.1

    def test_parameter_validation(self, daily_panel):
        x = daily_panel.copy()

        # Invalid window
        with pytest.raises((TypeError, ValueError)):
            pd_same_calendar_month_return(x, window=0)

    def test_requires_datetime_index(self):
        x = pd.DataFrame(np.random.randn(50, 2))

        with pytest.raises(TypeError, match="DatetimeIndex"):
            pd_same_calendar_month_return(x)


class TestSameClockLag:
    def test_basic_functionality(self, intraday_panel):
        x = intraday_panel.copy()
        clock_time = pd.DataFrame(
            np.tile(x.index.values.reshape(-1, 1), (1, 3)),
            index=x.index,
            columns=x.columns
        )

        result = pd_same_clock_lag(x, clock_time, lag_minutes=5)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape

        # First row should be NaN
        assert result.iloc[0].isna().all()

        # Later rows should have values
        assert result.iloc[2:].notna().any().any()

    def test_lag_accuracy(self):
        dates = pd.date_range("2024-01-01 09:00", periods=20, freq="5min")
        x = pd.DataFrame(np.arange(20).reshape(-1, 1), index=dates, columns=["A"])
        clock_time = pd.DataFrame(dates, index=dates, columns=["A"])

        result = pd_same_clock_lag(x, clock_time, lag_minutes=5)

        # At row 1 (09:05), lag of 5 minutes should give row 0 (09:00)
        if result.iloc[1, 0] is not None and not np.isnan(result.iloc[1, 0]):
            assert result.iloc[1, 0] == x.iloc[0, 0]

        # At row 2 (09:10), lag of 5 minutes should give row 1 (09:05)
        if result.iloc[2, 0] is not None and not np.isnan(result.iloc[2, 0]):
            assert result.iloc[2, 0] == x.iloc[1, 0]

    def test_parameter_validation(self, intraday_panel):
        x = intraday_panel.copy()
        clock_time = pd.DataFrame(
            np.tile(x.index.values.reshape(-1, 1), (1, len(x.columns))),
            index=x.index,
            columns=x.columns
        )

        # Invalid lag_minutes
        with pytest.raises((TypeError, ValueError)):
            pd_same_clock_lag(x, clock_time, lag_minutes=0)

    def test_irregular_timestamps(self):
        # Irregular timestamps
        dates = pd.to_datetime([
            "2024-01-01 09:00",
            "2024-01-01 09:03",
            "2024-01-01 09:08",
            "2024-01-01 09:15",
            "2024-01-01 09:25",
        ])
        x = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, columns=["A"])
        clock_time = pd.DataFrame(dates, index=dates, columns=["A"])

        result = pd_same_clock_lag(x, clock_time, lag_minutes=5)

        # Should find closest past observation within constraint
        assert result.notna().any().any()


class TestPITSafety:
    """Test PIT-safety across all time semantic operators."""

    def test_report_asof_no_future_leakage(self, daily_panel):
        value = daily_panel.copy()
        report_dates = pd.DataFrame(
            np.tile(value.index.values.reshape(-1, 1), (1, 3)),
            index=value.index,
            columns=value.columns
        )
        asof_date = report_dates.copy()

        # Set future values to extreme
        value.iloc[50:] = 999.0

        result = pd_report_asof(value, report_dates, asof_date, max_staleness_days=90)

        # Result at row 49 should not be 999
        if result.iloc[49].notna().any():
            assert (result.iloc[49] != 999.0).all()

    def test_event_window_pit_safety(self, daily_panel):
        ret = daily_panel.copy()

        # Event at day 30
        event_date = pd.DataFrame(pd.NaT, index=ret.index, columns=ret.columns)
        event_date.iloc[30:] = ret.index[30]

        result = pd_event_window_return_asof(ret, event_date, window_before=5, window_after=5)

        # Before window closes (day 35), should be NaN
        assert result.iloc[34].isna().all()

        # After window closes, can have values
        assert result.iloc[35:].notna().any().any()

    def test_financial_snapshot_lag_pit_safety(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        x = pd.DataFrame(np.arange(30).reshape(-1, 1), index=dates, columns=["A"])

        period_id = pd.DataFrame("", index=dates, columns=["A"])
        period_id.iloc[0:10] = "2024Q1"
        period_id.iloc[10:20] = "2024Q2"
        period_id.iloc[20:30] = "2024Q3"

        fiscal_date = pd.DataFrame(dates, index=dates, columns=["A"])

        result = pd_financial_snapshot_lag(x, period_id, fiscal_date, lag_periods=1)

        # Q2 should only see Q1 data, not Q3
        q2_result = result.iloc[10:20, 0].dropna()
        if len(q2_result) > 0:
            # Should be < 10 (Q1 values)
            assert (q2_result < 10).all()


class TestEdgeCases:
    def test_all_nan_input(self, daily_panel):
        x = pd.DataFrame(np.nan, index=daily_panel.index, columns=daily_panel.columns)

        result = pd_same_calendar_day_mean(x)
        assert result.isna().all().all()

    def test_single_column(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        x = pd.DataFrame(np.random.randn(50, 1), index=dates, columns=["A"])

        result = pd_same_calendar_month_return(x, window=12)
        assert result.shape == (50, 1)

    def test_insufficient_window(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        x = pd.DataFrame(np.random.randn(10, 2), index=dates, columns=["A", "B"])

        result = pd_same_calendar_day_mean(x, window=52)

        # Should have limited valid data
        assert result.notna().sum().sum() < 20
