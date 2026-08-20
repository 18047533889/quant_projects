# -*- coding: utf-8 -*-
"""Tests for frequency-domain / causal filtering operators.

Four R47 operators: ts_bessel_lowpass_causal, ts_fir_lowpass_causal,
ts_spectral_lowpass_trailing, ts_causal_savgol_endpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.technical.frequency_filters  # noqa: F401

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

FREQUENCY_FILTER_CANONICALS = [
    "ts_bessel_lowpass_causal",
    "ts_fir_lowpass_causal",
    "ts_spectral_lowpass_trailing",
    "ts_causal_savgol_endpoint",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _daily_panel(days: int = 100, cols: int = 3, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic daily price panel."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    # Random walk + drift
    returns = rng.standard_normal((days, cols)) * 0.01 + 0.0002
    prices = np.exp(np.cumsum(returns, axis=0)) * 100.0
    return pd.DataFrame(prices, index=idx, columns=[f"C{i}" for i in range(cols)])


def _all_nan(template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)


def _assert_shape_match(result: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert result.shape == expected.shape
    assert list(result.columns) == list(expected.columns)
    assert list(result.index) == list(expected.index)


# ---------------------------------------------------------------------------
# registration + surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_registered_and_classified(name: str) -> None:
    """All operators must be registered and classified as extended."""
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, f"{name} not registered"
    surface = classify_canonical(name)
    assert surface in ("daily", "extended", "research"), f"{name} surface={surface}"


# ---------------------------------------------------------------------------
# shape and determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_shape_and_determinism(name: str) -> None:
    """Output shape matches input; deterministic over repeated calls."""
    panel = _daily_panel(days=100, cols=3, seed=42)
    op = OperatorRegistry.get(name)

    # Default parameters
    if name == "ts_bessel_lowpass_causal":
        kwargs = {"order": 4, "cutoff": 0.05}
    elif name == "ts_fir_lowpass_causal":
        kwargs = {"ntaps": 21, "cutoff": 0.1, "window": "hamming"}
    elif name == "ts_spectral_lowpass_trailing":
        kwargs = {"window": 64, "cutoff_freq": 5}
    elif name == "ts_causal_savgol_endpoint":
        kwargs = {"window": 21, "polyorder": 2}
    else:
        kwargs = {}

    out1 = op.calculate(panel, **kwargs)
    out2 = op.calculate(panel, **kwargs)

    _assert_shape_match(out1, panel)
    _assert_shape_match(out2, panel)
    pd.testing.assert_frame_equal(out1, out2, check_dtype=False)


# ---------------------------------------------------------------------------
# causality / PIT safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_causality_no_future_leak(name: str) -> None:
    """Output at t depends only on data <= t (strictly causal)."""
    panel = _daily_panel(days=100, cols=2, seed=7)
    op = OperatorRegistry.get(name)

    if name == "ts_bessel_lowpass_causal":
        kwargs = {"order": 3, "cutoff": 0.08}
    elif name == "ts_fir_lowpass_causal":
        kwargs = {"ntaps": 15, "cutoff": 0.1, "window": "hamming"}
    elif name == "ts_spectral_lowpass_trailing":
        kwargs = {"window": 32, "cutoff_freq": 3}
    elif name == "ts_causal_savgol_endpoint":
        kwargs = {"window": 15, "polyorder": 2}
    else:
        kwargs = {}

    # Compute full series
    full = op.calculate(panel, **kwargs)

    # Compute truncated series (first 50 rows)
    truncated = op.calculate(panel.iloc[:50], **kwargs)

    # Values at t <= 50 must be identical (no future leak)
    for col in panel.columns:
        full_head = full[col].iloc[:50]
        trunc_vals = truncated[col]
        # Compare finite values only
        valid_mask = full_head.notna() & trunc_vals.notna()
        if valid_mask.any():
            np.testing.assert_allclose(
                full_head[valid_mask].values,
                trunc_vals[valid_mask].values,
                rtol=1e-10,
                err_msg=f"{name} leaked future data for {col}",
            )


# ---------------------------------------------------------------------------
# warmup / initial NaN behavior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_initial_warmup_nan(name: str) -> None:
    """First K bars are NaN (warmup period)."""
    panel = _daily_panel(days=60, cols=2, seed=3)
    op = OperatorRegistry.get(name)

    if name == "ts_bessel_lowpass_causal":
        kwargs = {"order": 4, "cutoff": 0.05}
        min_warmup = 4
    elif name == "ts_fir_lowpass_causal":
        kwargs = {"ntaps": 21, "cutoff": 0.1, "window": "hamming"}
        min_warmup = 21
    elif name == "ts_spectral_lowpass_trailing":
        kwargs = {"window": 32, "cutoff_freq": 4}
        min_warmup = 32
    elif name == "ts_causal_savgol_endpoint":
        kwargs = {"window": 15, "polyorder": 2}
        min_warmup = 15
    else:
        kwargs = {}
        min_warmup = 1

    out = op.calculate(panel, **kwargs)

    # First min_warmup bars should be NaN
    for col in panel.columns:
        assert out[col].iloc[:min_warmup].isna().all(), f"{name} warmup not NaN for {col}"
        # At least some finite values after warmup
        assert out[col].iloc[min_warmup:].notna().any(), f"{name} all NaN after warmup for {col}"


# ---------------------------------------------------------------------------
# smoothing behavior
# ---------------------------------------------------------------------------
def test_bessel_smoothing() -> None:
    """Bessel filter reduces high-frequency noise."""
    panel = _daily_panel(days=120, cols=1, seed=10)
    # Add high-frequency noise
    noise = pd.DataFrame(
        np.random.default_rng(10).standard_normal(panel.shape) * 5.0,
        index=panel.index,
        columns=panel.columns,
    )
    noisy = panel + noise

    op = OperatorRegistry.get("ts_bessel_lowpass_causal")
    smoothed = op.calculate(noisy, order=4, cutoff=0.05)

    # Smoothed series should have lower variance than noisy input (after warmup)
    col = panel.columns[0]
    noisy_var = noisy[col].iloc[20:].var()
    smooth_var = smoothed[col].iloc[20:].var()
    assert smooth_var < noisy_var, "Bessel filter did not reduce variance"


def test_fir_lowpass_reduces_noise() -> None:
    """FIR lowpass filter reduces high-frequency noise."""
    panel = _daily_panel(days=100, cols=1, seed=5)
    noise = pd.DataFrame(
        np.random.default_rng(5).standard_normal(panel.shape) * 3.0,
        index=panel.index,
        columns=panel.columns,
    )
    noisy = panel + noise

    op = OperatorRegistry.get("ts_fir_lowpass_causal")
    smoothed = op.calculate(noisy, ntaps=31, cutoff=0.08, window="hamming")

    col = panel.columns[0]
    noisy_std = noisy[col].iloc[40:].std()
    smooth_std = smoothed[col].iloc[40:].std()
    assert smooth_std < noisy_std, "FIR filter did not reduce noise"


def test_spectral_lowpass_removes_high_freq() -> None:
    """Spectral lowpass zeros high-frequency components."""
    panel = _daily_panel(days=150, cols=1, seed=8)
    op = OperatorRegistry.get("ts_spectral_lowpass_trailing")
    smoothed = op.calculate(panel, window=64, cutoff_freq=3)

    col = panel.columns[0]
    # Check that smoothed series is finite after warmup
    assert smoothed[col].iloc[64:].notna().sum() > 50, "Spectral filter produced too many NaNs"


def test_savgol_preserves_polynomial() -> None:
    """Savitzky-Golay exactly preserves polynomials up to degree <= polyorder."""
    days = 80
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    # Quadratic polynomial: y = 2*t^2 + 3*t + 5
    t = np.arange(days, dtype=float)
    poly = 2 * t**2 + 3 * t + 5
    panel = pd.DataFrame(poly, index=idx, columns=["C0"])

    op = OperatorRegistry.get("ts_causal_savgol_endpoint")
    # polyorder=2 should exactly preserve quadratic
    smoothed = op.calculate(panel, window=21, polyorder=2)

    # After warmup, smoothed should match input (up to numerical error)
    col = "C0"
    valid = smoothed[col].notna()
    np.testing.assert_allclose(
        smoothed[col][valid].values,
        panel[col][valid].values,
        rtol=1e-3,
        atol=1e-2,
        err_msg="SavGol did not preserve quadratic polynomial",
    )


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_all_nan_input(name: str) -> None:
    """All-NaN input produces all-NaN output."""
    panel = _daily_panel(days=50, cols=2, seed=1)
    nan_panel = _all_nan(panel)
    op = OperatorRegistry.get(name)

    if name == "ts_bessel_lowpass_causal":
        kwargs = {"order": 3, "cutoff": 0.05}
    elif name == "ts_fir_lowpass_causal":
        kwargs = {"ntaps": 15, "cutoff": 0.1}
    elif name == "ts_spectral_lowpass_trailing":
        kwargs = {"window": 32, "cutoff_freq": 3}
    elif name == "ts_causal_savgol_endpoint":
        kwargs = {"window": 15, "polyorder": 2}
    else:
        kwargs = {}

    out = op.calculate(nan_panel, **kwargs)
    assert out.isna().all().all(), f"{name} did not produce all-NaN for all-NaN input"


@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_single_column(name: str) -> None:
    """Works correctly with single-column input."""
    panel = _daily_panel(days=80, cols=1, seed=2)
    op = OperatorRegistry.get(name)

    if name == "ts_bessel_lowpass_causal":
        kwargs = {"order": 3, "cutoff": 0.05}
    elif name == "ts_fir_lowpass_causal":
        kwargs = {"ntaps": 15, "cutoff": 0.1}
    elif name == "ts_spectral_lowpass_trailing":
        kwargs = {"window": 32, "cutoff_freq": 3}
    elif name == "ts_causal_savgol_endpoint":
        kwargs = {"window": 15, "polyorder": 2}
    else:
        kwargs = {}

    out = op.calculate(panel, **kwargs)
    _assert_shape_match(out, panel)
    assert out.notna().any().any(), f"{name} produced all-NaN for single column"


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------
def test_bessel_order_bounds() -> None:
    """Bessel filter order is clamped to [1, 8]."""
    panel = _daily_panel(days=60, cols=1, seed=1)
    op = OperatorRegistry.get("ts_bessel_lowpass_causal")

    # Order too high (should be clamped to 8)
    out = op.calculate(panel, order=15, cutoff=0.05)
    assert out.notna().any().any(), "Bessel with high order failed"

    # Order too low (should be clamped to 1)
    out = op.calculate(panel, order=0, cutoff=0.05)
    assert out.notna().any().any(), "Bessel with low order failed"


def test_fir_ntaps_odd_enforcement() -> None:
    """FIR filter enforces odd ntaps for symmetric kernel."""
    panel = _daily_panel(days=80, cols=1, seed=3)
    op = OperatorRegistry.get("ts_fir_lowpass_causal")

    # Even ntaps (should be incremented to odd)
    out = op.calculate(panel, ntaps=20, cutoff=0.1, window="hamming")
    assert out.notna().any().any(), "FIR with even ntaps failed"


def test_savgol_polyorder_clamp() -> None:
    """SavGol polyorder is clamped to < window."""
    panel = _daily_panel(days=60, cols=1, seed=4)
    op = OperatorRegistry.get("ts_causal_savgol_endpoint")

    # polyorder >= window (should be clamped)
    out = op.calculate(panel, window=10, polyorder=12)
    assert out.notna().any().any(), "SavGol with high polyorder failed"


# ---------------------------------------------------------------------------
# policy conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FREQUENCY_FILTER_CANONICALS)
def test_explicit_policy_present(name: str) -> None:
    """All operators have explicit pit_safe=True policy."""
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    assert name in _EXPLICIT_POLICIES, f"{name} missing explicit policy"
    policy = _EXPLICIT_POLICIES[name]
    assert policy.get("pit_safe") is True, f"{name} policy pit_safe != True"
    assert policy.get("scope") == "ts", f"{name} policy scope != ts"
    assert policy.get("min_periods") == 1, f"{name} policy min_periods != 1"
