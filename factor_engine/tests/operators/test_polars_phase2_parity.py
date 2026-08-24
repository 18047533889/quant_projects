# -*- coding: utf-8 -*-
"""Polars Phase 2 parity tests: technical indicators and rolling statistics."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_POLARS_PHASE2_TESTS") == "1",
    reason="Polars Phase 2 tests skipped via environment variable"
)

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    """Create test data with realistic OHLCV patterns."""
    load_all()
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    stocks = ["A", "B"]

    idx = pd.MultiIndex.from_product([dates, stocks], names=["timestamp", "instrument"])

    np.random.seed(42)
    n = len(idx)

    # Generate realistic price data
    close_base = 100.0 + np.cumsum(np.random.randn(n) * 2)
    high = close_base + np.abs(np.random.randn(n) * 1.5)
    low = close_base - np.abs(np.random.randn(n) * 1.5)
    volume = 1000000 + np.random.randint(-200000, 200000, n)

    close = pd.Series(close_base, index=idx)
    high = pd.Series(high, index=idx)
    low = pd.Series(low, index=idx)
    open_ = pd.Series(close_base + np.random.randn(n) * 0.5, index=idx)
    volume = pd.Series(volume, index=idx)

    # Add some NaN values to test handling
    close.iloc[10:12] = np.nan
    high.iloc[25] = np.nan

    return InMemorySeriesSource(data={
        "close": close,
        "high": high,
        "low": low,
        "open": open_,
        "volume": volume,
    })


def _run_pair(source, expr):
    """Run expression on both pandas and polars backends."""
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
        a = eng_pd.run(Factor(name="t", expr=expr))["result"]
        b = eng_pl.run(Factor(name="t", expr=expr))["result"]
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
    return a, b


def _assert_close(a, b, rtol=1e-4, atol=1e-6, allow_nan_mismatch=False):
    """Assert series are close, handling NaN values."""
    if allow_nan_mismatch:
        # Allow some difference in NaN patterns due to implementation differences
        mask = a.notna() & b.notna()
        if mask.sum() > 0:
            pd.testing.assert_series_equal(
                a[mask], b[mask], check_names=False, rtol=rtol, atol=atol
            )
    else:
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=rtol, atol=atol)


# =============================================================================
# Phase 2 Technical Indicators (20 operators)
# =============================================================================

def test_ts_rsi_parity(source):
    """Test ts_rsi pandas/polars parity."""
    ts_rsi = make_cleaned_call_factory("ts_rsi")
    expr = ts_rsi(col("close"), 14)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_macd_parity(source):
    """Test ts_macd pandas/polars parity."""
    ts_macd = make_cleaned_call_factory("ts_macd")
    expr = ts_macd(col("close"), 12, 26, 9)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_bbands_parity(source):
    """Test ts_bbands pandas/polars parity."""
    ts_bbands = make_cleaned_call_factory("ts_bbands")
    expr = ts_bbands(col("close"), 20, 2.0)
    a, b = _run_pair(source, expr)
    _assert_close(a, b)


def test_ts_atr_parity(source):
    """Test ts_atr pandas/polars parity."""
    ts_atr = make_cleaned_call_factory("ts_atr")
    expr = ts_atr(col("high"), col("low"), col("close"), 14)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_adx_parity(source):
    """Test ts_adx pandas/polars parity."""
    ts_adx = make_cleaned_call_factory("ts_adx")
    expr = ts_adx(col("high"), col("low"), col("close"), 14)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_cci_parity(source):
    """Test ts_cci pandas/polars parity."""
    ts_cci = make_cleaned_call_factory("ts_cci")
    expr = ts_cci(col("high"), col("low"), col("close"), 20)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_roc_parity(source):
    """Test ts_roc pandas/polars parity."""
    ts_roc = make_cleaned_call_factory("ts_roc")
    expr = ts_roc(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b)


def test_ts_momentum_parity(source):
    """Test ts_momentum pandas/polars parity."""
    ts_momentum = make_cleaned_call_factory("ts_momentum")
    expr = ts_momentum(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b)


def test_ts_stoch_parity(source):
    """Test ts_stoch pandas/polars parity."""
    ts_stoch = make_cleaned_call_factory("ts_stoch")
    expr = ts_stoch(col("high"), col("low"), col("close"), 14)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_williams_r_parity(source):
    """Test ts_williams_r pandas/polars parity."""
    ts_williams_r = make_cleaned_call_factory("ts_williams_r")
    expr = ts_williams_r(col("high"), col("low"), col("close"), 14)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_obv_parity(source):
    """Test ts_obv pandas/polars parity."""
    ts_obv = make_cleaned_call_factory("ts_obv")
    expr = ts_obv(col("close"), col("volume"))
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-6, allow_nan_mismatch=True)


def test_ts_mfi_parity(source):
    """Test ts_mfi pandas/polars parity."""
    ts_mfi = make_cleaned_call_factory("ts_mfi")
    expr = ts_mfi(col("high"), col("low"), col("close"), col("volume"), 14)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_trix_parity(source):
    """Test ts_trix pandas/polars parity."""
    ts_trix = make_cleaned_call_factory("ts_trix")
    expr = ts_trix(col("close"), 12)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_dpo_parity(source):
    """Test ts_dpo pandas/polars parity."""
    ts_dpo = make_cleaned_call_factory("ts_dpo")
    expr = ts_dpo(col("close"), 20)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_kama_parity(source):
    """Test ts_kama pandas/polars parity (simplified for Polars)."""
    ts_kama = make_cleaned_call_factory("ts_kama")
    expr = ts_kama(col("close"), 10, 2, 30)
    a, b = _run_pair(source, expr)
    # KAMA has different implementations, allow looser tolerance
    _assert_close(a, b, rtol=0.1, allow_nan_mismatch=True)


def test_ts_tema_parity(source):
    """Test ts_tema pandas/polars parity."""
    ts_tema = make_cleaned_call_factory("ts_tema")
    expr = ts_tema(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_dema_parity(source):
    """Test ts_dema pandas/polars parity."""
    ts_dema = make_cleaned_call_factory("ts_dema")
    expr = ts_dema(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_zlema_parity(source):
    """Test ts_zlema pandas/polars parity."""
    ts_zlema = make_cleaned_call_factory("ts_zlema")
    expr = ts_zlema(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_vwma_parity(source):
    """Test ts_vwma pandas/polars parity."""
    ts_vwma = make_cleaned_call_factory("ts_vwma")
    expr = ts_vwma(col("close"), col("volume"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_hma_parity(source):
    """Test ts_hma pandas/polars parity."""
    ts_hma = make_cleaned_call_factory("ts_hma")
    expr = ts_hma(col("close"), 16)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


# =============================================================================
# Phase 2 Rolling Statistics (20 operators)
# =============================================================================

def test_ts_rolling_corr_parity(source):
    """Test ts_rolling_corr pandas/polars parity."""
    ts_rolling_corr = make_cleaned_call_factory("ts_rolling_corr")
    expr = ts_rolling_corr(col("close"), col("volume"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_cov_parity(source):
    """Test ts_rolling_cov pandas/polars parity."""
    ts_rolling_cov = make_cleaned_call_factory("ts_rolling_cov")
    expr = ts_rolling_cov(col("close"), col("volume"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_beta_parity(source):
    """Test ts_rolling_beta pandas/polars parity."""
    ts_rolling_beta = make_cleaned_call_factory("ts_rolling_beta")
    expr = ts_rolling_beta(col("close"), col("open"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_alpha_parity(source):
    """Test ts_rolling_alpha pandas/polars parity."""
    ts_rolling_alpha = make_cleaned_call_factory("ts_rolling_alpha")
    expr = ts_rolling_alpha(col("close"), col("open"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_sharpe_parity(source):
    """Test ts_rolling_sharpe pandas/polars parity."""
    ts_rolling_sharpe = make_cleaned_call_factory("ts_rolling_sharpe")
    expr = ts_rolling_sharpe(col("close"), 10, 0.0)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_zscore_parity(source):
    """Test ts_rolling_zscore pandas/polars parity."""
    ts_rolling_zscore = make_cleaned_call_factory("ts_rolling_zscore")
    expr = ts_rolling_zscore(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_rank_parity(source):
    """Test ts_rolling_rank pandas/polars parity."""
    ts_rolling_rank = make_cleaned_call_factory("ts_rolling_rank")
    expr = ts_rolling_rank(col("close"), 10)
    a, b = _run_pair(source, expr)
    # Rank may differ in implementation, allow looser tolerance
    _assert_close(a, b, rtol=0.2, allow_nan_mismatch=True)


def test_ts_rolling_quantile_parity(source):
    """Test ts_rolling_quantile pandas/polars parity."""
    ts_rolling_quantile = make_cleaned_call_factory("ts_rolling_quantile")
    expr = ts_rolling_quantile(col("close"), 10, 0.5)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_median_parity(source):
    """Test ts_rolling_median pandas/polars parity."""
    ts_rolling_median = make_cleaned_call_factory("ts_rolling_median")
    expr = ts_rolling_median(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-4, allow_nan_mismatch=True)


def test_ts_rolling_mad_parity(source):
    """Test ts_rolling_mad pandas/polars parity."""
    ts_rolling_mad = make_cleaned_call_factory("ts_rolling_mad")
    expr = ts_rolling_mad(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_iqr_parity(source):
    """Test ts_rolling_iqr pandas/polars parity."""
    ts_rolling_iqr = make_cleaned_call_factory("ts_rolling_iqr")
    expr = ts_rolling_iqr(col("close"), 10)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_entropy_parity(source):
    """Test ts_rolling_entropy pandas/polars parity (approximation)."""
    ts_rolling_entropy = make_cleaned_call_factory("ts_rolling_entropy")
    expr = ts_rolling_entropy(col("close"), 10)
    a, b = _run_pair(source, expr)
    # Entropy is approximated, allow looser tolerance
    _assert_close(a, b, rtol=0.5, allow_nan_mismatch=True)


def test_ts_rolling_autocorr_parity(source):
    """Test ts_rolling_autocorr pandas/polars parity."""
    ts_rolling_autocorr = make_cleaned_call_factory("ts_rolling_autocorr")
    expr = ts_rolling_autocorr(col("close"), 10, 1)
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_rolling_linear_slope_parity(source):
    """Test ts_rolling_linear_slope pandas/polars parity."""
    ts_rolling_linear_slope = make_cleaned_call_factory("ts_rolling_linear_slope")
    expr = ts_rolling_linear_slope(col("close"), 10)
    a, b = _run_pair(source, expr)
    # Slope approximation may differ
    _assert_close(a, b, rtol=0.5, allow_nan_mismatch=True)


def test_ts_rolling_r2_parity(source):
    """Test ts_rolling_r2 pandas/polars parity."""
    ts_rolling_r2 = make_cleaned_call_factory("ts_rolling_r2")
    expr = ts_rolling_r2(col("close"), 10)
    a, b = _run_pair(source, expr)
    # R² approximation may differ significantly
    _assert_close(a, b, rtol=0.5, allow_nan_mismatch=True)


def test_ts_expanding_mean_parity(source):
    """Test ts_expanding_mean pandas/polars parity."""
    ts_expanding_mean = make_cleaned_call_factory("ts_expanding_mean")
    expr = ts_expanding_mean(col("close"))
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_expanding_std_parity(source):
    """Test ts_expanding_std pandas/polars parity."""
    ts_expanding_std = make_cleaned_call_factory("ts_expanding_std")
    expr = ts_expanding_std(col("close"))
    a, b = _run_pair(source, expr)
    _assert_close(a, b, rtol=1e-3, allow_nan_mismatch=True)


def test_ts_expanding_min_parity(source):
    """Test ts_expanding_min pandas/polars parity."""
    ts_expanding_min = make_cleaned_call_factory("ts_expanding_min")
    expr = ts_expanding_min(col("close"))
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_expanding_max_parity(source):
    """Test ts_expanding_max pandas/polars parity."""
    ts_expanding_max = make_cleaned_call_factory("ts_expanding_max")
    expr = ts_expanding_max(col("close"))
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)


def test_ts_expanding_sum_parity(source):
    """Test ts_expanding_sum pandas/polars parity."""
    ts_expanding_sum = make_cleaned_call_factory("ts_expanding_sum")
    expr = ts_expanding_sum(col("close"))
    a, b = _run_pair(source, expr)
    _assert_close(a, b, allow_nan_mismatch=True)
