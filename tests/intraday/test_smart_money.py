# -*- coding: utf-8 -*-
"""Tests for smart money intraday operators.

Tests pandas backend for:
- intra_dynamic_stock_graph_features
- intra_common_trading_intensity
- intra_local_conditional_entropy
- intra_smart_money_vwap_ratio
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def setup_module():
    """Reset registry lifecycle to allow operator registration during test imports.

    The registry gets finalized during normal operation, but test modules that
    import operator classes trigger registration at import time. This hook ensures
    the registry is writable before the imports happen.
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._lifecycle != OperatorRegistry.Lifecycle.BUILDING:
        # Safe to reset for test isolation
        OperatorRegistry._lifecycle = OperatorRegistry.Lifecycle.BUILDING


from factor_engine.cleaned_operators.intraday.smart_money import (
    IntraDynamicStockGraphFeatures,
    IntraCommonTradingIntensity,
    IntraLocalConditionalEntropy,
    IntraSmartMoneyVwapRatio,
)


@pytest.fixture
def minute_panel():
    """Generate a synthetic minute panel (240 bars = 1 day)."""
    dates = pd.date_range("2024-01-02 09:30", periods=240, freq="1min")
    instruments = ["INST_A", "INST_B", "INST_C"]

    np.random.seed(42)
    data = {}
    for inst in instruments:
        base_price = 100.0 + np.random.randn() * 10
        drift = np.random.randn(240) * 0.01
        price = base_price * np.exp(np.cumsum(drift))
        data[inst] = price

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def minute_volume():
    """Generate synthetic volume panel."""
    dates = pd.date_range("2024-01-02 09:30", periods=240, freq="1min")
    instruments = ["INST_A", "INST_B", "INST_C"]

    np.random.seed(43)
    data = {}
    for inst in instruments:
        # Volume with some spikes
        base_vol = 1e6
        vol = base_vol * (1 + np.abs(np.random.randn(240) * 0.5))
        # Add a few large spikes
        vol[100] *= 5
        vol[150] *= 8
        data[inst] = vol

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def minute_amount(minute_panel, minute_volume):
    """Generate amount = price * volume."""
    return minute_panel * minute_volume


# ---------------------------------------------------------------------------
# § Pandas backend tests
# ---------------------------------------------------------------------------

def test_graph_features_pandas(minute_panel):
    """Test intra_dynamic_stock_graph_features (pandas)."""
    op = IntraDynamicStockGraphFeatures()
    result = op.calculate(minute_panel)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # One day
    assert set(result.columns) == {"INST_A", "INST_B", "INST_C"}

    # Check values are finite (autocorrelation proxy)
    for col in result.columns:
        val = result[col].iloc[0]
        assert np.isfinite(val) or pd.isna(val)
        if np.isfinite(val):
            assert -1.0 <= val <= 1.0, "Autocorr should be in [-1, 1]"


def test_trading_intensity_pandas(minute_volume, minute_amount):
    """Test intra_common_trading_intensity (pandas)."""
    op = IntraCommonTradingIntensity()
    result = op.calculate(minute_volume, minute_amount)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns) == {"INST_A", "INST_B", "INST_C"}

    # Intensity should be positive
    for col in result.columns:
        val = result[col].iloc[0]
        assert np.isfinite(val) or pd.isna(val)
        if np.isfinite(val):
            assert val > 0, "Intensity should be positive"


def test_conditional_entropy_pandas(minute_panel, minute_volume):
    """Test intra_local_conditional_entropy (pandas)."""
    op = IntraLocalConditionalEntropy()
    result = op.calculate(minute_panel, minute_volume)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns) == {"INST_A", "INST_B", "INST_C"}

    # Entropy should be non-negative
    for col in result.columns:
        val = result[col].iloc[0]
        assert np.isfinite(val) or pd.isna(val)
        if np.isfinite(val):
            assert val >= 0, "Entropy should be non-negative"


def test_smart_money_vwap_ratio_pandas(minute_panel, minute_amount, minute_volume):
    """Test intra_smart_money_vwap_ratio (pandas)."""
    op = IntraSmartMoneyVwapRatio()
    result = op.calculate(minute_panel, minute_amount, minute_volume)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns) == {"INST_A", "INST_B", "INST_C"}

    # Ratio should be positive (price ratio)
    for col in result.columns:
        val = result[col].iloc[0]
        assert np.isfinite(val) or pd.isna(val)
        if np.isfinite(val):
            assert val > 0, "VWAP ratio should be positive"
            assert 0.5 < val < 2.0, "VWAP ratio should be near 1.0"


# ---------------------------------------------------------------------------
# § Edge case tests
# ---------------------------------------------------------------------------

def test_insufficient_data_pandas():
    """Test that insufficient data returns NaN."""
    dates = pd.date_range("2024-01-02 09:30", periods=3, freq="1min")
    data = pd.DataFrame({"INST_A": [100, 101, 102]}, index=dates)

    op = IntraDynamicStockGraphFeatures()
    result = op.calculate(data)

    assert pd.isna(result["INST_A"].iloc[0])


def test_zero_volume_pandas():
    """Test that zero volume returns NaN."""
    dates = pd.date_range("2024-01-02 09:30", periods=10, freq="1min")
    close = pd.DataFrame({"INST_A": np.full(10, 100.0)}, index=dates)
    volume = pd.DataFrame({"INST_A": np.zeros(10)}, index=dates)
    amount = close * volume

    op = IntraSmartMoneyVwapRatio()
    result = op.calculate(close, amount, volume)

    assert pd.isna(result["INST_A"].iloc[0])


def test_multiday_panel():
    """Test that multiday panel produces one row per day."""
    # Create two separate trading days with proper timestamps
    day1 = pd.date_range("2024-01-02 09:30", periods=240, freq="1min")
    day2 = pd.date_range("2024-01-03 09:30", periods=240, freq="1min")
    dates = day1.union(day2)

    np.random.seed(99)
    data = pd.DataFrame({"INST_A": np.random.randn(480).cumsum() + 100}, index=dates)

    op = IntraDynamicStockGraphFeatures()
    result = op.calculate(data)

    assert len(result) == 2, "Should have 2 rows (one per day)"
