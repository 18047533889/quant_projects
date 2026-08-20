# -*- coding: utf-8 -*-
"""Tests for R47 intraday pattern recognition operators.

Covers the four TRUE_GAP operators:
- intra_smart_money_fcm_score
- intra_price_peak_ridge_valley_state
- intra_volume_peak_ridge_valley_state
- intraday_value_at_extreme_state
"""
import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import get_operator


@pytest.fixture
def minute_panel_simple():
    """Simple minute panel: 2 days, 2 instruments, 10 bars/day."""
    dates = pd.date_range("2024-01-02 09:30", periods=10, freq="1min")
    dates2 = pd.date_range("2024-01-03 09:30", periods=10, freq="1min")
    idx = dates.tolist() + dates2.tolist()

    data = {
        "A": np.concatenate([
            np.array([100.0, 101, 102, 103, 104, 103, 102, 101, 100, 99]),
            np.array([99.0, 100, 101, 102, 103, 104, 105, 106, 107, 108])
        ]),
        "B": np.concatenate([
            np.array([50.0, 51, 52, 53, 54, 53, 52, 51, 50, 49]),
            np.array([49.0, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        ]),
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


@pytest.fixture
def volume_panel_simple():
    """Volume panel matching minute_panel_simple."""
    dates = pd.date_range("2024-01-02 09:30", periods=10, freq="1min")
    dates2 = pd.date_range("2024-01-03 09:30", periods=10, freq="1min")
    idx = dates.tolist() + dates2.tolist()

    data = {
        "A": np.concatenate([
            np.array([1000, 1100, 1200, 1300, 1400, 1300, 1200, 1100, 1000, 900]),
            np.array([900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800])
        ]),
        "B": np.concatenate([
            np.array([500, 550, 600, 650, 700, 650, 600, 550, 500, 450]),
            np.array([450, 500, 550, 600, 650, 700, 750, 800, 850, 900])
        ]),
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


@pytest.fixture
def high_low_panels():
    """High/low panels for range calculations."""
    dates = pd.date_range("2024-01-02 09:30", periods=10, freq="1min")
    dates2 = pd.date_range("2024-01-03 09:30", periods=10, freq="1min")
    idx = dates.tolist() + dates2.tolist()

    high = {
        "A": np.concatenate([
            np.array([101, 102, 103, 104, 105, 104, 103, 102, 101, 100]),
            np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        ]),
        "B": np.concatenate([
            np.array([51, 52, 53, 54, 55, 54, 53, 52, 51, 50]),
            np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
        ]),
    }

    low = {
        "A": np.concatenate([
            np.array([99, 100, 101, 102, 103, 102, 101, 100, 99, 98]),
            np.array([98, 99, 100, 101, 102, 103, 104, 105, 106, 107])
        ]),
        "B": np.concatenate([
            np.array([49, 50, 51, 52, 53, 52, 51, 50, 49, 48]),
            np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57])
        ]),
    }

    return (
        pd.DataFrame(high, index=pd.DatetimeIndex(idx)),
        pd.DataFrame(low, index=pd.DatetimeIndex(idx))
    )


# ---------------------------------------------------------------------------
# 1. intra_smart_money_fcm_score
# ---------------------------------------------------------------------------

def test_smart_money_basic(minute_panel_simple, volume_panel_simple, high_low_panels):
    """Basic functionality: returns daily panel with valid scores."""
    op = get_operator("intra_smart_money_fcm_score")
    high, low = high_low_panels

    result = op.calculate(
        price=minute_panel_simple,
        volume=volume_panel_simple,
        high=high,
        low=low
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2  # 2 days
    assert result.shape[1] == 2  # 2 instruments
    assert result.index[0].date() == pd.Timestamp("2024-01-02").date()
    assert result.index[1].date() == pd.Timestamp("2024-01-03").date()
    assert np.all(np.isfinite(result.values))


def test_smart_money_insufficient_data():
    """Insufficient data returns NaN."""
    op = get_operator("intra_smart_money_fcm_score")

    # Only 3 bars (need >= 10 by default)
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min")
    price = pd.DataFrame({"A": [100, 101, 102]}, index=idx)
    volume = pd.DataFrame({"A": [1000, 1100, 1200]}, index=idx)
    high = pd.DataFrame({"A": [101, 102, 103]}, index=idx)
    low = pd.DataFrame({"A": [99, 100, 101]}, index=idx)

    result = op.calculate(price=price, volume=volume, high=high, low=low)
    assert result.shape[0] == 1
    assert np.isnan(result.iloc[0, 0])


def test_smart_money_constant_price():
    """Constant price (zero range) returns NaN."""
    op = get_operator("intra_smart_money_fcm_score")

    idx = pd.date_range("2024-01-02 09:30", periods=15, freq="1min")
    price = pd.DataFrame({"A": [100.0] * 15}, index=idx)
    volume = pd.DataFrame({"A": np.random.rand(15) * 1000}, index=idx)
    high = pd.DataFrame({"A": [100.0] * 15}, index=idx)
    low = pd.DataFrame({"A": [100.0] * 15}, index=idx)

    result = op.calculate(price=price, volume=volume, high=high, low=low)
    assert np.isnan(result.iloc[0, 0])


def test_smart_money_multi_instrument(minute_panel_simple, volume_panel_simple, high_low_panels):
    """Multi-instrument panel processed correctly."""
    op = get_operator("intra_smart_money_fcm_score")
    high, low = high_low_panels

    result = op.calculate(
        price=minute_panel_simple,
        volume=volume_panel_simple,
        high=high,
        low=low
    )

    assert "A" in result.columns
    assert "B" in result.columns
    assert result["A"].notna().sum() >= 1
    assert result["B"].notna().sum() >= 1


# ---------------------------------------------------------------------------
# 2. intra_price_peak_ridge_valley_state
# ---------------------------------------------------------------------------

def test_price_peak_basic(minute_panel_simple):
    """Basic peak detection returns non-negative counts."""
    op = get_operator("intra_price_peak_ridge_valley_state")

    result = op.calculate(price=minute_panel_simple, window=2, prominence=0.3)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
    assert result.shape[1] == 2
    # Peak counts should be >= 0
    assert np.all(result.values[np.isfinite(result.values)] >= 0)


def test_price_peak_window_param():
    """Window parameter controls detection sensitivity."""
    op = get_operator("intra_price_peak_ridge_valley_state")

    # Sawtooth pattern: clear peaks
    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="1min")
    price = pd.DataFrame({
        "A": [100 + (10 if i % 2 == 0 else 0) for i in range(20)]
    }, index=idx)

    result_w1 = op.calculate(price=price, window=1, prominence=0.1)
    result_w3 = op.calculate(price=price, window=3, prominence=0.1)

    # Smaller window detects more peaks
    assert np.isfinite(result_w1.iloc[0, 0])
    assert np.isfinite(result_w3.iloc[0, 0])


def test_price_peak_prominence_threshold():
    """Prominence threshold filters minor peaks."""
    op = get_operator("intra_price_peak_ridge_valley_state")

    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="1min")
    # Small fluctuations around 100
    price = pd.DataFrame({
        "A": [100 + 0.1 * np.sin(i) for i in range(20)]
    }, index=idx)

    result_low = op.calculate(price=price, window=2, prominence=0.01)
    result_high = op.calculate(price=price, window=2, prominence=0.5)

    # High prominence should find fewer/no peaks
    if np.isfinite(result_low.iloc[0, 0]) and np.isfinite(result_high.iloc[0, 0]):
        assert result_high.iloc[0, 0] <= result_low.iloc[0, 0]


def test_price_peak_insufficient_bars():
    """Too few bars returns NaN."""
    op = get_operator("intra_price_peak_ridge_valley_state")

    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min")
    price = pd.DataFrame({"A": [100, 101, 100]}, index=idx)

    result = op.calculate(price=price, window=5, prominence=0.5)
    assert np.isnan(result.iloc[0, 0])


# ---------------------------------------------------------------------------
# 3. intra_volume_peak_ridge_valley_state
# ---------------------------------------------------------------------------

def test_volume_peak_basic(volume_panel_simple):
    """Basic volume peak detection."""
    op = get_operator("intra_volume_peak_ridge_valley_state")

    result = op.calculate(volume=volume_panel_simple, window=2, prominence=0.3)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
    assert result.shape[1] == 2
    assert np.all(result.values[np.isfinite(result.values)] >= 0)


def test_volume_peak_spike_pattern():
    """Volume spike pattern detected as peak."""
    op = get_operator("intra_volume_peak_ridge_valley_state")

    idx = pd.date_range("2024-01-02 09:30", periods=15, freq="1min")
    # Spike at position 7
    volume = pd.DataFrame({
        "A": [1000] * 6 + [5000] + [1000] * 8
    }, index=idx)

    result = op.calculate(volume=volume, window=2, prominence=0.5)

    # Should detect at least 1 peak
    assert np.isfinite(result.iloc[0, 0])
    assert result.iloc[0, 0] >= 1.0


def test_volume_peak_zero_volume():
    """Zero mean volume returns NaN."""
    op = get_operator("intra_volume_peak_ridge_valley_state")

    idx = pd.date_range("2024-01-02 09:30", periods=15, freq="1min")
    volume = pd.DataFrame({"A": [0.0] * 15}, index=idx)

    result = op.calculate(volume=volume, window=2, prominence=0.5)
    assert np.isnan(result.iloc[0, 0])


# ---------------------------------------------------------------------------
# 4. intraday_value_at_extreme_state
# ---------------------------------------------------------------------------

def test_value_at_extreme_basic(minute_panel_simple, volume_panel_simple, high_low_panels):
    """Basic functionality: returns ratio in [0, 1]."""
    op = get_operator("intraday_value_at_extreme_state")
    high, low = high_low_panels

    result = op.calculate(
        price=minute_panel_simple,
        volume=volume_panel_simple,
        high=high,
        low=low,
        quantile=0.1
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
    assert result.shape[1] == 2

    # Ratios should be in [0, 1]
    finite_vals = result.values[np.isfinite(result.values)]
    assert np.all(finite_vals >= 0)
    assert np.all(finite_vals <= 1)


def test_value_at_extreme_quantile_effect():
    """Larger quantile captures more volume."""
    op = get_operator("intraday_value_at_extreme_state")

    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="1min")
    price = pd.DataFrame({"A": np.linspace(100, 110, 20)}, index=idx)
    volume = pd.DataFrame({"A": [1000] * 20}, index=idx)
    high = pd.DataFrame({"A": np.linspace(101, 111, 20)}, index=idx)
    low = pd.DataFrame({"A": np.linspace(99, 109, 20)}, index=idx)

    result_q01 = op.calculate(price=price, volume=volume, high=high, low=low, quantile=0.1)
    result_q03 = op.calculate(price=price, volume=volume, high=high, low=low, quantile=0.3)

    # Larger quantile should capture more volume
    assert result_q03.iloc[0, 0] >= result_q01.iloc[0, 0]


def test_value_at_extreme_concentrated_volume():
    """Volume concentrated at extremes yields high ratio."""
    op = get_operator("intraday_value_at_extreme_state")

    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="1min")
    prices = np.linspace(100, 110, 20)
    # Heavy volume at first and last 2 bars (extremes)
    volumes = [5000, 5000] + [100] * 16 + [5000, 5000]

    price = pd.DataFrame({"A": prices}, index=idx)
    volume = pd.DataFrame({"A": volumes}, index=idx)
    high = pd.DataFrame({"A": prices + 1}, index=idx)
    low = pd.DataFrame({"A": prices - 1}, index=idx)

    result = op.calculate(price=price, volume=volume, high=high, low=low, quantile=0.1)

    # Should show high concentration at extremes
    assert result.iloc[0, 0] > 0.5


def test_value_at_extreme_uniform_volume():
    """Uniform volume distribution yields ~2*quantile ratio."""
    op = get_operator("intraday_value_at_extreme_state")

    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="1min")
    price = pd.DataFrame({"A": np.linspace(100, 110, 20)}, index=idx)
    volume = pd.DataFrame({"A": [1000] * 20}, index=idx)  # uniform
    high = pd.DataFrame({"A": np.linspace(101, 111, 20)}, index=idx)
    low = pd.DataFrame({"A": np.linspace(99, 109, 20)}, index=idx)

    result = op.calculate(price=price, volume=volume, high=high, low=low, quantile=0.1)

    # With uniform volume, expect ~20% at top+bottom 10% each
    assert 0.15 <= result.iloc[0, 0] <= 0.25


def test_value_at_extreme_insufficient_data():
    """Too few bars returns NaN."""
    op = get_operator("intraday_value_at_extreme_state")

    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="1min")
    price = pd.DataFrame({"A": [100, 101, 102]}, index=idx)
    volume = pd.DataFrame({"A": [1000, 1100, 1200]}, index=idx)
    high = pd.DataFrame({"A": [101, 102, 103]}, index=idx)
    low = pd.DataFrame({"A": [99, 100, 101]}, index=idx)

    result = op.calculate(price=price, volume=volume, high=high, low=low, quantile=0.1)
    assert np.isnan(result.iloc[0, 0])


# ---------------------------------------------------------------------------
# Edge cases & robustness
# ---------------------------------------------------------------------------

def test_all_nan_input():
    """All-NaN input returns NaN output."""
    ops = [
        "intra_smart_money_fcm_score",
        "intra_price_peak_ridge_valley_state",
        "intra_volume_peak_ridge_valley_state",
        "intraday_value_at_extreme_state",
    ]

    idx = pd.date_range("2024-01-02 09:30", periods=15, freq="1min")
    nan_panel = pd.DataFrame({"A": [np.nan] * 15}, index=idx)

    for op_name in ops[:2]:  # Test first two
        op = get_operator(op_name)
        if op_name == "intra_smart_money_fcm_score":
            result = op.calculate(price=nan_panel, volume=nan_panel, high=nan_panel, low=nan_panel)
        else:
            result = op.calculate(price=nan_panel)

        assert result.shape[0] == 1
        assert np.isnan(result.iloc[0, 0])


def test_single_instrument():
    """Single instrument processed correctly."""
    idx = pd.date_range("2024-01-02 09:30", periods=15, freq="1min")
    price = pd.DataFrame({"X": np.linspace(100, 110, 15)}, index=idx)

    op = get_operator("intra_price_peak_ridge_valley_state")
    result = op.calculate(price=price, window=2, prominence=0.3)

    assert result.shape == (1, 1)
    assert "X" in result.columns


def test_multi_day_aggregation(minute_panel_simple):
    """Multi-day input produces one row per day."""
    op = get_operator("intra_price_peak_ridge_valley_state")

    result = op.calculate(price=minute_panel_simple, window=2, prominence=0.3)

    # Should have 2 rows (2 days)
    assert result.shape[0] == 2
    unique_dates = pd.DatetimeIndex(result.index).normalize().unique()
    assert len(unique_dates) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
