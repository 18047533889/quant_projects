"""Tests for portfolio weighting utilities."""

import numpy as np
import pandas as pd
import pytest
from utils.portfolio.weights import (
    equal_weights,
    market_cap_weights,
    inverse_volatility_weights,
    risk_parity_weights,
    rank_weights,
    normalize_weights,
)


def test_equal_weights_from_int():
    """Test equal weights with integer input."""
    w = equal_weights(5)
    assert len(w) == 5
    assert np.allclose(w, 0.2)
    assert np.isclose(w.sum(), 1.0)


def test_equal_weights_from_list():
    """Test equal weights with asset list."""
    assets = ['A', 'B', 'C']
    w = equal_weights(assets)
    assert len(w) == 3
    assert list(w.index) == assets
    assert np.allclose(w, 1/3)
    assert np.isclose(w.sum(), 1.0)


def test_equal_weights_different_target():
    """Test equal weights with different target sum."""
    w = equal_weights(4, target_sum=2.0)
    assert np.allclose(w, 0.5)
    assert np.isclose(w.sum(), 2.0)


def test_market_cap_weights():
    """Test market cap weighting."""
    caps = pd.Series([100, 200, 300, 400], index=['A', 'B', 'C', 'D'])
    w = market_cap_weights(caps)

    expected = caps / caps.sum()
    assert np.allclose(w, expected)
    assert np.isclose(w.sum(), 1.0)


def test_market_cap_weights_with_max():
    """Test market cap weighting with max weight constraint."""
    caps = pd.Series([100, 900], index=['A', 'B'])
    w = market_cap_weights(caps, max_weight=0.6)

    assert w['B'] <= 0.6
    assert w['A'] >= 0.4 - 1e-6  # Allow small numerical tolerance
    assert np.isclose(w.sum(), 1.0)


def test_market_cap_weights_negative():
    """Test market cap weighting filters negative caps."""
    caps = pd.Series([100, -50, 200], index=['A', 'B', 'C'])
    w = market_cap_weights(caps)

    assert np.isnan(w['B']) or w['B'] == 0
    assert w['A'] > 0
    assert w['C'] > 0


def test_inverse_volatility_weights():
    """Test inverse volatility weighting."""
    vols = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
    w = inverse_volatility_weights(vols)

    # Higher vol should get lower weight
    assert w['A'] > w['B'] > w['C']
    assert np.isclose(w.sum(), 1.0)


def test_risk_parity_weights_diagonal():
    """Test risk parity with diagonal covariance (reduces to inverse vol)."""
    vols = np.array([0.1, 0.2, 0.3])
    cov = pd.DataFrame(np.diag(vols ** 2), index=['A', 'B', 'C'], columns=['A', 'B', 'C'])

    w = risk_parity_weights(cov)

    # Should be similar to inverse vol
    inv_vol_w = inverse_volatility_weights(pd.Series(vols, index=['A', 'B', 'C']))
    assert np.allclose(w, inv_vol_w, atol=0.01)


def test_risk_parity_weights_with_correlation():
    """Test risk parity with non-diagonal covariance."""
    # Simple 2-asset case
    cov = pd.DataFrame([
        [0.01, 0.005],
        [0.005, 0.04]
    ], index=['A', 'B'], columns=['A', 'B'])

    w = risk_parity_weights(cov)

    # Lower variance asset should get higher weight
    assert w['A'] > w['B']
    assert np.isclose(w.sum(), 1.0)

    # Check risk contributions are approximately equal
    port_var = w.values @ cov.values @ w.values
    mrc = (cov.values @ w.values) / np.sqrt(port_var)
    risk_contrib = w.values * mrc

    # Risk contributions should be roughly equal
    assert np.allclose(risk_contrib[0], risk_contrib[1], rtol=0.1)


def test_rank_weights_linear():
    """Test linear rank weighting."""
    signals = pd.Series([3.0, 1.0, 2.0, 5.0], index=['A', 'B', 'C', 'D'])
    w = rank_weights(signals, method='linear')

    # Rank order: B(1), C(2), A(3), D(4)
    # Weights proportional to rank
    assert w['D'] > w['A'] > w['C'] > w['B']
    assert np.isclose(w.sum(), 1.0)


def test_rank_weights_quadratic():
    """Test quadratic rank weighting."""
    signals = pd.Series([1.0, 2.0, 3.0], index=['A', 'B', 'C'])
    w = rank_weights(signals, method='quadratic')

    # Quadratic gives more weight to top ranks
    assert w['C'] > w['B'] > w['A']
    assert np.isclose(w.sum(), 1.0)


def test_rank_weights_long_short():
    """Test long-short rank weighting."""
    signals = pd.Series([1.0, 2.0, 3.0, 4.0], index=['A', 'B', 'C', 'D'])
    w = rank_weights(signals, method='linear', long_short=True, target_sum=0.0)

    # Top half long, bottom half short
    assert w['D'] > 0  # Best signal
    assert w['C'] > 0
    assert w['B'] < 0
    assert w['A'] < 0  # Worst signal

    # Market neutral
    assert np.isclose(w.sum(), 0.0, atol=1e-10)


def test_normalize_weights_basic():
    """Test basic weight normalization."""
    w = pd.Series([1.0, 2.0, 3.0], index=['A', 'B', 'C'])
    normalized = normalize_weights(w)

    assert np.isclose(normalized.sum(), 1.0)
    assert normalized['C'] == 0.5  # 3/6


def test_normalize_weights_with_bounds():
    """Test weight normalization with bounds."""
    w = pd.Series([1.0, 10.0], index=['A', 'B'])
    normalized = normalize_weights(w, max_weight=0.6)

    assert normalized['B'] <= 0.6
    assert np.isclose(normalized.sum(), 1.0)


def test_normalize_weights_array():
    """Test normalize works with arrays."""
    w = np.array([1.0, 2.0, 3.0])
    normalized = normalize_weights(w)

    assert isinstance(normalized, np.ndarray)
    assert np.isclose(normalized.sum(), 1.0)


def test_normalize_weights_all_nan():
    """Test normalize handles all-NaN input."""
    w = pd.Series([np.nan, np.nan], index=['A', 'B'])
    normalized = normalize_weights(w)

    assert np.all(np.isnan(normalized))


def test_normalize_weights_all_zero():
    """Test normalize handles all-zero input."""
    w = pd.Series([0.0, 0.0], index=['A', 'B'])
    normalized = normalize_weights(w)

    assert np.all(np.isnan(normalized))


def test_rank_weights_exponential():
    """Test exponential rank weighting."""
    signals = pd.Series([1.0, 2.0, 3.0], index=['A', 'B', 'C'])
    w = rank_weights(signals, method='exponential')

    # Exponential gives even more weight to top ranks
    assert w['C'] > w['B'] > w['A']
    assert np.isclose(w.sum(), 1.0)


def test_inverse_volatility_zero_vol():
    """Test inverse vol handles zero volatility."""
    vols = pd.Series([0.1, 0.0, 0.2], index=['A', 'B', 'C'])
    w = inverse_volatility_weights(vols)

    # Zero vol should be treated as NaN
    assert w['A'] > 0
    assert w['C'] > 0
    # B should not crash, gets 0 or nan
