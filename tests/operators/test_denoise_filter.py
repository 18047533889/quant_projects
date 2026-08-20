# -*- coding: utf-8 -*-
"""Tests for denoise/regularization filter operators.

Covers:
- ts_ssa_denoise_trailing
- ts_wavelet_shrinkage_trailing
- ts_total_variation_filter_trailing
- ts_l1_trend_filter_trailing

Each operator is tested for:
- Registration and basic calculation
- Shape preservation
- Causality (future-poison test)
- NaN handling
- Parameter validation
- Determinism
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

# Import the modules explicitly to ensure registration
import cleaned_operators.technical.denoise_filter  # noqa: F401

ALL_OPERATORS = [
    "ts_ssa_denoise_trailing",
    "ts_wavelet_shrinkage_trailing",
    "ts_total_variation_filter_trailing",
    "ts_l1_trend_filter_trailing",
]

DEFAULT_PARAMS = {
    "ts_ssa_denoise_trailing": {"window": 20, "n_components": 3},
    "ts_wavelet_shrinkage_trailing": {"window": 20, "threshold": 0.5},
    "ts_total_variation_filter_trailing": {"window": 20, "lambda_tv": 0.1},
    "ts_l1_trend_filter_trailing": {"window": 30, "lambda_l1": 0.1},
}


def _make_panel(rows: int = 100, cols: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create synthetic price panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    # Trending + noise
    trend = np.linspace(100, 120, rows)
    noise = rng.normal(0, 2, (rows, cols))
    data = trend[:, None] + noise
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(cols)])


def _tamper_future(panel: pd.DataFrame, last_n: int = 10, factor: float = 100.0) -> pd.DataFrame:
    """Corrupt the last N rows (future poison test)."""
    p = panel.copy()
    p.iloc[-last_n:] *= factor
    return p


# ---------------------------------------------------------------------------
# Registration and basic calculation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_registration(name: str):
    """All operators are registered."""
    op = OperatorRegistry.get(name)
    assert op is not None
    assert op.metadata.name == name


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_basic_calculation(name: str):
    """Basic calculation returns valid output."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(50, 2)
    params = DEFAULT_PARAMS[name]

    result = op.calculate(panel, **params)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == panel.shape
    assert (result.index == panel.index).all()
    assert (result.columns == panel.columns).all()


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_shape_preservation(name: str):
    """Output shape matches input shape."""
    op = OperatorRegistry.get(name)
    params = DEFAULT_PARAMS[name]

    for rows, cols in [(30, 1), (50, 3), (100, 5)]:
        panel = _make_panel(rows, cols)
        result = op.calculate(panel, **params)
        assert result.shape == (rows, cols)


# ---------------------------------------------------------------------------
# Causality (future-poison test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_causality_future_poison(name: str):
    """Corrupting future rows does not affect past outputs."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(100, 2, seed=123)
    params = DEFAULT_PARAMS[name]

    # Clean output
    clean = op.calculate(panel, **params)

    # Tampered output
    tampered_panel = _tamper_future(panel, last_n=10, factor=1000.0)
    tampered = op.calculate(tampered_panel, **params)

    # Past should be identical (up to second-to-last row to avoid edge effects)
    past_clean = clean.iloc[:-11]
    past_tampered = tampered.iloc[:-11]

    # Allow small numerical differences
    np.testing.assert_allclose(
        past_clean.values,
        past_tampered.values,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_nan_handling_sparse(name: str):
    """Operators handle sparse NaN gracefully."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(50, 2)
    params = DEFAULT_PARAMS[name]

    # Inject sparse NaNs
    panel_nan = panel.copy()
    panel_nan.iloc[10, 0] = np.nan
    panel_nan.iloc[25, 1] = np.nan

    result = op.calculate(panel_nan, **params)

    # Should produce output (may have NaN in affected windows)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == panel.shape


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_nan_handling_dense(name: str):
    """Operators handle dense NaN blocks."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(50, 2)
    params = DEFAULT_PARAMS[name]

    # Create large NaN block
    panel_nan = panel.copy()
    panel_nan.iloc[10:20, 0] = np.nan

    result = op.calculate(panel_nan, **params)

    # Should produce output with NaN in affected regions
    assert isinstance(result, pd.DataFrame)
    assert result.shape == panel.shape
    # The NaN block should propagate
    assert result.iloc[10:20, 0].isna().any()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_determinism(name: str):
    """Multiple runs produce identical output."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(50, 2, seed=999)
    params = DEFAULT_PARAMS[name]

    result1 = op.calculate(panel, **params)
    result2 = op.calculate(panel, **params)

    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_ssa_parameter_validation():
    """SSA validates window and n_components."""
    op = OperatorRegistry.get("ts_ssa_denoise_trailing")
    panel = _make_panel(30, 1)

    # Valid
    result = op.calculate(panel, window=15, n_components=3)
    assert result.shape == panel.shape

    # Window too small (should clamp)
    result = op.calculate(panel, window=3, n_components=2)
    assert result.shape == panel.shape


def test_wavelet_parameter_validation():
    """Wavelet validates threshold."""
    op = OperatorRegistry.get("ts_wavelet_shrinkage_trailing")
    panel = _make_panel(30, 1)

    # Zero threshold (no shrinkage)
    result = op.calculate(panel, window=10, threshold=0.0)
    assert result.shape == panel.shape

    # Large threshold (heavy shrinkage)
    result = op.calculate(panel, window=10, threshold=5.0)
    assert result.shape == panel.shape


def test_tv_parameter_validation():
    """TV filter validates lambda_tv."""
    op = OperatorRegistry.get("ts_total_variation_filter_trailing")
    panel = _make_panel(30, 1)

    # Zero lambda (no regularization)
    result = op.calculate(panel, window=10, lambda_tv=0.0)
    assert result.shape == panel.shape

    # Large lambda (strong smoothing)
    result = op.calculate(panel, window=10, lambda_tv=1.0)
    assert result.shape == panel.shape


def test_l1_trend_parameter_validation():
    """L1 trend validates lambda_l1."""
    op = OperatorRegistry.get("ts_l1_trend_filter_trailing")
    panel = _make_panel(40, 1)

    # Zero lambda (no penalty)
    result = op.calculate(panel, window=15, lambda_l1=0.0)
    assert result.shape == panel.shape

    # Large lambda (strong smoothing)
    result = op.calculate(panel, window=15, lambda_l1=0.5)
    assert result.shape == panel.shape


# ---------------------------------------------------------------------------
# Functional behavior
# ---------------------------------------------------------------------------

def test_ssa_noise_reduction():
    """SSA reduces high-frequency noise."""
    op = OperatorRegistry.get("ts_ssa_denoise_trailing")

    # Clean signal + noise
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    clean_signal = np.sin(np.linspace(0, 4 * np.pi, n)) * 10 + 100
    noisy_signal = clean_signal + rng.normal(0, 2, n)

    panel = pd.DataFrame({"S0": noisy_signal}, index=dates)

    result = op.calculate(panel, window=30, n_components=2)

    # Denoised should be smoother (lower variance in differences)
    original_diff_std = np.nanstd(np.diff(panel["S0"].values))
    denoised_diff_std = np.nanstd(np.diff(result["S0"].values[30:]))  # Skip warm-up

    assert denoised_diff_std < original_diff_std


def test_wavelet_impulse_preservation():
    """Wavelet preserves large impulses while removing noise."""
    op = OperatorRegistry.get("ts_wavelet_shrinkage_trailing")

    # Signal with impulse
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    signal = np.ones(n) * 100
    signal[25] = 120  # Large impulse
    rng = np.random.default_rng(42)
    signal = signal + rng.normal(0, 0.5, n)  # Small noise

    panel = pd.DataFrame({"S0": signal}, index=dates)

    result = op.calculate(panel, window=20, threshold=0.3)

    # Impulse should be preserved (relatively)
    impulse_idx = 25
    assert result["S0"].iloc[impulse_idx] > result["S0"].iloc[impulse_idx - 5:impulse_idx].mean() + 5


def test_tv_edge_preservation():
    """TV filter preserves edges (jumps)."""
    op = OperatorRegistry.get("ts_total_variation_filter_trailing")

    # Piecewise constant signal + noise
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    signal = np.concatenate([np.ones(20) * 100, np.ones(20) * 110, np.ones(20) * 105])
    rng = np.random.default_rng(42)
    signal = signal + rng.normal(0, 1, n)

    panel = pd.DataFrame({"S0": signal}, index=dates)

    result = op.calculate(panel, window=20, lambda_tv=0.15)

    # Should see distinct levels
    assert result["S0"].iloc[15:19].mean() < result["S0"].iloc[25:29].mean()


def test_l1_trend_smoothness():
    """L1 trend produces piecewise-linear smooth trend."""
    op = OperatorRegistry.get("ts_l1_trend_filter_trailing")

    # Trend + oscillation
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = np.linspace(100, 120, n)
    oscillation = np.sin(np.linspace(0, 6 * np.pi, n)) * 3
    signal = trend + oscillation

    panel = pd.DataFrame({"S0": signal}, index=dates)

    result = op.calculate(panel, window=40, lambda_l1=0.2)

    # Extracted trend should be smoother (lower second derivative)
    original_d2 = np.abs(np.diff(panel["S0"].values, n=2)).mean()
    trend_d2 = np.abs(np.diff(result["S0"].values[40:], n=2)).mean()

    assert trend_d2 < original_d2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_single_column(name: str):
    """Operators work with single column."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(40, 1)
    params = DEFAULT_PARAMS[name]

    result = op.calculate(panel, **params)
    assert result.shape == (40, 1)


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_short_window(name: str):
    """Operators handle short windows."""
    op = OperatorRegistry.get(name)
    panel = _make_panel(20, 2)
    params = DEFAULT_PARAMS[name].copy()
    params["window"] = 10

    result = op.calculate(panel, **params)
    assert result.shape == panel.shape


# ---------------------------------------------------------------------------
# Scope and metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_metadata_tags(name: str):
    """All operators have correct metadata tags."""
    op = OperatorRegistry.get(name)
    tags = op.metadata.tags

    assert "time_series" in tags
    assert "pit_safe" in tags


@pytest.mark.parametrize("name", ALL_OPERATORS)
def test_category(name: str):
    """All operators have time_series category."""
    op = OperatorRegistry.get(name)
    assert op.metadata.category == "time_series"
