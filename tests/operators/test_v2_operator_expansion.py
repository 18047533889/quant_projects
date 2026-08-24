from __future__ import annotations

import numpy as np
import pandas as pd


def _panel(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"A": values}, index=index, dtype=float)


def test_fundamental_lag_uses_report_period_not_trading_rows():
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import fin_lag

    values = _panel([10, 10, 11, 11, 20, 20, 20, 30, 30])
    period = pd.DataFrame(
        {"A": ["2023Q1"] * 4 + ["2023Q2"] * 3 + ["2023Q3"] * 2},
        index=values.index,
    )
    output = fin_lag(values, period, 1)
    assert np.isnan(output.iloc[3, 0])
    assert output.iloc[4, 0] == 11.0
    assert output.iloc[7, 0] == 20.0


def test_revision_event_is_same_period_only():
    from factor_engine.cleaned_operators.fundamental.transforms_repairs_v2 import (
        fin_revision_delta,
    )

    values = _panel([10, 10, 11, 20, 20])
    period = pd.DataFrame(
        {"A": ["Q1", "Q1", "Q1", "Q2", "Q2"]}, index=values.index
    )
    output = fin_revision_delta(values, period)
    assert list(output["A"])[1:] == [0.0, 1.0, 0.0, 0.0]


def test_bounded_fundamental_streak():
    from factor_engine.cleaned_operators.fundamental.transforms_repairs_v2 import (
        fin_positive_streak,
    )

    values = _panel([1, 2, 3, 4, 5, 6])
    period = pd.DataFrame(
        {"A": [f"Q{i}" for i in range(6)]}, index=values.index
    )
    output = fin_positive_streak(values, period, 4)
    assert output.iloc[-1, 0] == 3.0


def test_structure_pattern_is_prefix_invariant():
    from factor_engine.cleaned_operators.price_volume.structure_patterns_v2 import (
        pattern_sym_triangle,
    )

    count = 80
    time = np.arange(count, dtype=float)
    high = _panel(110 - 0.08 * time + np.sin(time / 2))
    low = _panel(90 + 0.08 * time - np.sin(time / 2))
    full = pattern_sym_triangle(high, low, 2, 2, 50, 3, 0.001)
    prefix_rows = 60
    prefix = pattern_sym_triangle(
        high.iloc[:prefix_rows], low.iloc[:prefix_rows], 2, 2, 50, 3, 0.001
    )
    pd.testing.assert_series_equal(
        full.iloc[:prefix_rows, 0], prefix.iloc[:, 0], check_names=False
    )


def test_kama_is_prefix_invariant():
    from factor_engine.cleaned_operators.technical.indicators_v2 import KAMA

    values = _panel(
        np.linspace(10, 20, 100) + np.sin(np.arange(100, dtype=float) / 3)
    )
    full = KAMA(values, 10, 2, 30)
    prefix = KAMA(values.iloc[:70], 10, 2, 30)
    np.testing.assert_allclose(
        full.iloc[:70, 0],
        prefix.iloc[:, 0],
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


def test_candle_geometry_window_is_prior_based():
    from factor_engine.cleaned_operators.price_volume.candle_geometry_v2 import (
        candle_body_zscore,
    )

    open_ = _panel(np.arange(30) + 10)
    close = open_.copy()
    close.iloc[:, 0] += np.linspace(0.1, 3, 30)
    full = candle_body_zscore(open_, close, 10)
    prefix = candle_body_zscore(open_.iloc[:20], close.iloc[:20], 10)
    np.testing.assert_allclose(
        full.iloc[:20, 0], prefix.iloc[:, 0], equal_nan=True
    )


def _minute_frame(timestamps: pd.Series) -> pd.DataFrame:
    values = np.arange(1, len(timestamps) + 1, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": np.ones(len(values)),
            "amount": values,
        }
    )


def test_intraday_bar_end_clock_does_not_merge_lunch_boundaries():
    from factor_engine.storage.sources.intraday_feature_runtime_v2 import _clock_bars, _ordinal

    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-01-02 11:30",
                "2024-01-02 13:00",
                "2024-01-02 13:01",
                "2024-01-02 13:05",
            ]
        )
    )
    ordinal = _ordinal(
        timestamps,
        "ashare_stock_minute",
        "09:30",
        "15:00",
        "bar_end",
    )
    assert ordinal.iloc[0] == 119
    assert np.isnan(ordinal.iloc[1])  # session marker, not a completed minute bar
    assert ordinal.iloc[2] == 120
    assert ordinal.iloc[3] == 124

    bars = _clock_bars(
        _minute_frame(timestamps),
        5,
        "ashare_stock_minute",
        "09:30",
        "15:00",
        "bar_end",
    )
    assert len(bars) == 2


def test_intraday_bar_start_clock_does_not_merge_lunch_boundaries():
    from factor_engine.storage.sources.intraday_feature_runtime_v2 import _ordinal

    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-01-02 11:29",
                "2024-01-02 11:30",
                "2024-01-02 13:00",
                "2024-01-02 13:01",
            ]
        )
    )
    ordinal = _ordinal(
        timestamps,
        "ashare_stock_minute",
        "09:30",
        "15:00",
        "bar_start",
    )
    assert ordinal.iloc[0] == 119
    assert np.isnan(ordinal.iloc[1])
    assert ordinal.iloc[2] == 120
    assert ordinal.iloc[3] == 121


def test_complete_us_session_has_exactly_78_five_minute_bars():
    from factor_engine.storage.sources.intraday_feature_runtime_v2 import _clock_bars

    timestamps = pd.Series(
        pd.date_range("2024-01-02 09:31", "2024-01-02 16:00", freq="min")
    )
    assert len(timestamps) == 390
    bars = _clock_bars(
        _minute_frame(timestamps),
        5,
        "us_stock_minute",
        "09:30",
        "16:00",
        "bar_end",
    )
    assert len(bars) == 78


def test_cutoff_expected_minutes_excludes_future_session():
    from factor_engine.storage.sources.intraday_feature_runtime_v2 import _effective_minutes

    assert _effective_minutes(
        "us_stock_minute", "09:30", "16:00", "15:50"
    ) == 380
    assert _effective_minutes(
        "ashare_stock_minute", "09:30", "15:00", "14:00"
    ) == 180


def test_liquidity_operator_is_prefix_invariant():
    from factor_engine.cleaned_operators.price_volume.liquidity_v2 import amihud_illiquidity

    close = _panel(np.linspace(10, 12, 80))
    volume = _panel(np.linspace(1e6, 2e6, 80))
    returns = close.pct_change()
    full = amihud_illiquidity(returns, close, volume, 20)
    prefix = amihud_illiquidity(
        returns.iloc[:60], close.iloc[:60], volume.iloc[:60], 20
    )
    np.testing.assert_allclose(
        full.iloc[:60, 0], prefix.iloc[:, 0], equal_nan=True
    )
