# -*- coding: utf-8 -*-
"""Tests for intra_state_space operators (R47 completion pack).

Tests all 9 operators with dual backend coverage (pandas_numpy + polars).
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


from factor_engine.cleaned_operators.intraday.intra_state_space import (
    IntraFunctionalMotifScore,
    IntraFunctionalMotifScorePolars,
    IntraKalmanLatentPrice,
    IntraKalmanLatentPricePolars,
    IntraMarketProfileCorrExSelf,
    IntraMarketProfileCorrExSelfPolars,
    IntraPricePeakRidgeValleyState,
    IntraPricePeakRidgeValleyStatePolars,
    IntraSmartMoneyFcmScore,
    IntraSmartMoneyFcmScorePolars,
    IntraStateSpaceVolumeComponents,
    IntraStateSpaceVolumeComponentsPolars,
    IntraVisibilityGraphFeatures,
    IntraVisibilityGraphFeaturesPolars,
    IntraVolumePeakRidgeValleyState,
    IntraVolumePeakRidgeValleyStatePolars,
    IntradayValueAtExtremeState,
    IntradayValueAtExtremeStatePolars,
)


@pytest.fixture
def minute_panel_single_day():
    """Single-day minute panel (120 bars: 09:30-11:30)."""
    base = pd.Timestamp("2024-01-15 09:30:00")
    idx = pd.date_range(base, periods=120, freq="1min")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(120) * 0.5)
    volume = 1000 + np.random.rand(120) * 500
    high = close + np.random.rand(120) * 2
    low = close - np.random.rand(120) * 2
    return pd.DataFrame({"close": close, "volume": volume, "high": high, "low": low}, index=idx)


@pytest.fixture
def minute_panel_multi_instrument():
    """Multi-instrument minute panel."""
    base = pd.Timestamp("2024-01-15 09:30:00")
    idx = pd.date_range(base, periods=120, freq="1min")
    np.random.seed(42)
    data = {}
    for sym in ["A", "B", "C"]:
        np.random.seed(42 + ord(sym))
        data[sym] = 1000 + np.cumsum(np.random.randn(120) * 10) + np.random.rand(120) * 50
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# 1. intra_kalman_latent_price
# ---------------------------------------------------------------------------

def test_intra_kalman_latent_price_basic(minute_panel_single_day):
    """Basic smoke test for Kalman latent price."""
    close = minute_panel_single_day[["close"]]
    op = IntraKalmanLatentPrice()
    result = op.calculate(close, min_bars=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0, 0] > 0  # Normalized innovation variance


def test_intra_kalman_latent_price_polars(minute_panel_single_day):
    """Polars backend parity."""
    close = minute_panel_single_day[["close"]]
    op_pd = IntraKalmanLatentPrice()
    op_pl = IntraKalmanLatentPricePolars()
    r_pd = op_pd.calculate(close, min_bars=30)
    r_pl = op_pl.calculate(close, min_bars=30)
    assert len(r_pd) == len(r_pl)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


def test_intra_kalman_latent_price_insufficient_bars(minute_panel_single_day):
    """Insufficient bars returns NaN."""
    close = minute_panel_single_day[["close"]].iloc[:10]
    op = IntraKalmanLatentPrice()
    result = op.calculate(close, min_bars=30)
    assert len(result) == 1
    assert np.isnan(result.iloc[0, 0])


# ---------------------------------------------------------------------------
# 2. intra_state_space_volume_components
# ---------------------------------------------------------------------------

def test_intra_state_space_volume_components_basic(minute_panel_single_day):
    """Volume SSM decomposition returns ratio."""
    volume = minute_panel_single_day[["volume"]]
    op = IntraStateSpaceVolumeComponents()
    result = op.calculate(volume, min_bars=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    val = result.iloc[0, 0]
    assert 0 <= val <= 1  # Energy ratio


def test_intra_state_space_volume_components_polars(minute_panel_single_day):
    """Polars backend parity."""
    volume = minute_panel_single_day[["volume"]]
    op_pd = IntraStateSpaceVolumeComponents()
    op_pl = IntraStateSpaceVolumeComponentsPolars()
    r_pd = op_pd.calculate(volume, min_bars=30)
    r_pl = op_pl.calculate(volume, min_bars=30)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. intra_functional_motif_score
# ---------------------------------------------------------------------------

def test_intra_functional_motif_score_basic(minute_panel_single_day):
    """Functional motif score in [-2, 2] range."""
    close = minute_panel_single_day[["close"]].rename(columns={"close": "A"})
    volume = minute_panel_single_day[["volume"]].rename(columns={"volume": "A"})
    op = IntraFunctionalMotifScore()
    result = op.calculate(close, volume, min_bars=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    val = result.iloc[0, 0]
    assert -2 <= val <= 2  # Correlation difference


def test_intra_functional_motif_score_polars(minute_panel_single_day):
    """Polars backend parity."""
    close = minute_panel_single_day[["close"]].rename(columns={"close": "A"})
    volume = minute_panel_single_day[["volume"]].rename(columns={"volume": "A"})
    op_pd = IntraFunctionalMotifScore()
    op_pl = IntraFunctionalMotifScorePolars()
    r_pd = op_pd.calculate(close, volume, min_bars=30)
    r_pl = op_pl.calculate(close, volume, min_bars=30)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. intra_visibility_graph_features
# ---------------------------------------------------------------------------

def test_intra_visibility_graph_features_basic(minute_panel_single_day):
    """Visibility graph average degree."""
    close = minute_panel_single_day[["close"]]
    op = IntraVisibilityGraphFeatures()
    result = op.calculate(close, min_bars=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    val = result.iloc[0, 0]
    assert val >= 0  # Average degree count


def test_intra_visibility_graph_features_polars(minute_panel_single_day):
    """Polars backend parity."""
    close = minute_panel_single_day[["close"]]
    op_pd = IntraVisibilityGraphFeatures()
    op_pl = IntraVisibilityGraphFeaturesPolars()
    r_pd = op_pd.calculate(close, min_bars=30)
    r_pl = op_pl.calculate(close, min_bars=30)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 5. intra_smart_money_fcm_score
# ---------------------------------------------------------------------------

def test_intra_smart_money_fcm_score_basic(minute_panel_single_day):
    """Smart money FCM composite score."""
    price = minute_panel_single_day[["close"]].rename(columns={"close": "A"})
    volume = minute_panel_single_day[["volume"]].rename(columns={"volume": "A"})
    high = minute_panel_single_day[["high"]].rename(columns={"high": "A"})
    low = minute_panel_single_day[["low"]].rename(columns={"low": "A"})
    op = IntraSmartMoneyFcmScore()
    result = op.calculate(price, volume, high, low)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    # Score is VWAP_dev * (0.5*vol_conc + 0.5*persist), can be any float


def test_intra_smart_money_fcm_score_polars(minute_panel_single_day):
    """Polars backend parity."""
    price = minute_panel_single_day[["close"]].rename(columns={"close": "A"})
    volume = minute_panel_single_day[["volume"]].rename(columns={"volume": "A"})
    high = minute_panel_single_day[["high"]].rename(columns={"high": "A"})
    low = minute_panel_single_day[["low"]].rename(columns={"low": "A"})
    op_pd = IntraSmartMoneyFcmScore()
    op_pl = IntraSmartMoneyFcmScorePolars()
    r_pd = op_pd.calculate(price, volume, high, low)
    r_pl = op_pl.calculate(price, volume, high, low)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 6. intra_price_peak_ridge_valley_state
# ---------------------------------------------------------------------------

def test_intra_price_peak_ridge_valley_state_basic(minute_panel_single_day):
    """Price peak/valley state classification."""
    price = minute_panel_single_day[["close"]]
    op = IntraPricePeakRidgeValleyState()
    result = op.calculate(price, window=5, min_bars=20)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    val = result.iloc[0, 0]
    assert -1 <= val <= 1  # State score


def test_intra_price_peak_ridge_valley_state_polars(minute_panel_single_day):
    """Polars backend parity."""
    price = minute_panel_single_day[["close"]]
    op_pd = IntraPricePeakRidgeValleyState()
    op_pl = IntraPricePeakRidgeValleyStatePolars()
    r_pd = op_pd.calculate(price, window=5, min_bars=20)
    r_pl = op_pl.calculate(price, window=5, min_bars=20)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 7. intra_volume_peak_ridge_valley_state
# ---------------------------------------------------------------------------

def test_intra_volume_peak_ridge_valley_state_basic(minute_panel_single_day):
    """Volume peak/valley state classification."""
    volume = minute_panel_single_day[["volume"]]
    op = IntraVolumePeakRidgeValleyState()
    result = op.calculate(volume, window=5, min_bars=20)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    val = result.iloc[0, 0]
    assert -1 <= val <= 1


def test_intra_volume_peak_ridge_valley_state_polars(minute_panel_single_day):
    """Polars backend parity."""
    volume = minute_panel_single_day[["volume"]]
    op_pd = IntraVolumePeakRidgeValleyState()
    op_pl = IntraVolumePeakRidgeValleyStatePolars()
    r_pd = op_pd.calculate(volume, window=5, min_bars=20)
    r_pl = op_pl.calculate(volume, window=5, min_bars=20)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 8. intraday_value_at_extreme_state
# ---------------------------------------------------------------------------

def test_intraday_value_at_extreme_state_basic(minute_panel_single_day):
    """Value at extreme state returns ratio [0, 1]."""
    price = minute_panel_single_day[["close"]].rename(columns={"close": "A"})
    volume = minute_panel_single_day[["volume"]].rename(columns={"volume": "A"})
    op = IntradayValueAtExtremeState()
    result = op.calculate(price, volume, quantile=0.9, min_bars=20)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    val = result.iloc[0, 0]
    assert 0 <= val <= 1


def test_intraday_value_at_extreme_state_polars(minute_panel_single_day):
    """Polars backend parity."""
    price = minute_panel_single_day[["close"]].rename(columns={"close": "A"})
    volume = minute_panel_single_day[["volume"]].rename(columns={"volume": "A"})
    op_pd = IntradayValueAtExtremeState()
    op_pl = IntradayValueAtExtremeStatePolars()
    r_pd = op_pd.calculate(price, volume, quantile=0.9, min_bars=20)
    r_pl = op_pl.calculate(price, volume, quantile=0.9, min_bars=20)
    np.testing.assert_allclose(r_pd.iloc[0, 0], r_pl.iloc[0, 0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 9. intra_market_profile_corr_ex_self
# ---------------------------------------------------------------------------

def test_intra_market_profile_corr_ex_self_basic(minute_panel_multi_instrument):
    """Market profile correlation ex-self returns correlation in [-1, 1]."""
    volume = minute_panel_multi_instrument
    op = IntraMarketProfileCorrExSelf()
    result = op.calculate(volume, min_bars=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    for col in result.columns:
        val = result[col].iloc[0]
        if np.isfinite(val):
            assert -1 <= val <= 1


def test_intra_market_profile_corr_ex_self_polars(minute_panel_multi_instrument):
    """Polars backend parity."""
    volume = minute_panel_multi_instrument
    op_pd = IntraMarketProfileCorrExSelf()
    op_pl = IntraMarketProfileCorrExSelfPolars()
    r_pd = op_pd.calculate(volume, min_bars=30)
    r_pl = op_pl.calculate(volume, min_bars=30)
    assert r_pd.shape == r_pl.shape
    for col in r_pd.columns:
        if np.isfinite(r_pd[col].iloc[0]) and np.isfinite(r_pl[col].iloc[0]):
            np.testing.assert_allclose(r_pd[col].iloc[0], r_pl[col].iloc[0], rtol=1e-5)


def test_intra_market_profile_corr_ex_self_single_instrument():
    """Single instrument returns NaN (no others to correlate with)."""
    base = pd.Timestamp("2024-01-15 09:30:00")
    idx = pd.date_range(base, periods=120, freq="1min")
    volume = pd.DataFrame({"A": np.random.rand(120) * 1000}, index=idx)
    op = IntraMarketProfileCorrExSelf()
    result = op.calculate(volume, min_bars=30)
    assert len(result) == 1
    assert np.isnan(result.iloc[0, 0])


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

def test_min_bars_validation():
    """min_bars must be positive integer."""
    base = pd.Timestamp("2024-01-15 09:30:00")
    idx = pd.date_range(base, periods=120, freq="1min")
    close = pd.DataFrame({"A": np.random.randn(120) * 10 + 100}, index=idx)
    op = IntraKalmanLatentPrice()
    with pytest.raises(ValueError, match="min_bars"):
        op.calculate(close, min_bars=0)
    with pytest.raises(ValueError, match="min_bars"):
        op.calculate(close, min_bars=-5)


def test_window_validation():
    """window must be positive integer."""
    base = pd.Timestamp("2024-01-15 09:30:00")
    idx = pd.date_range(base, periods=120, freq="1min")
    price = pd.DataFrame({"A": np.random.randn(120) * 10 + 100}, index=idx)
    op = IntraPricePeakRidgeValleyState()
    with pytest.raises(ValueError, match="window"):
        op.calculate(price, window=0)


def test_quantile_validation():
    """quantile must be in valid range."""
    base = pd.Timestamp("2024-01-15 09:30:00")
    idx = pd.date_range(base, periods=120, freq="1min")
    price = pd.DataFrame({"A": np.random.randn(120) * 10 + 100}, index=idx)
    volume = pd.DataFrame({"A": np.random.rand(120) * 1000}, index=idx)
    op = IntradayValueAtExtremeState()
    with pytest.raises(ValueError, match="quantile"):
        op.calculate(price, volume, quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        op.calculate(price, volume, quantile=1.1)
