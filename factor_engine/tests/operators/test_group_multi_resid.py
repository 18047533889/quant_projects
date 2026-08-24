# -*- coding: utf-8 -*-
"""Tests for group_multi_resid operator.

Comprehensive coverage of group-wise multi-feature OLS residuals:
- Normal multi-feature regression (2-5 features)
- Singular matrix detection (zero variance feature, collinear features)
- Sample size constraints (min_obs gate)
- All-NaN groups (fail-closed)
- add_intercept True/False
- Joint missing value filtering
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op


# ---------------------------------------------------------------------------
# Normal multi-feature regression
# ---------------------------------------------------------------------------
def test_group_multi_resid_two_features() -> None:
    """Two-feature regression: y ~ x1 + x2 within groups."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    # Group A: perfect linear relationship y = 2*x1 + 3*x2 + 10
    # Using non-collinear x1 and x2
    # Group B: different relationship y = -1*x1 + 1*x2 + 5
    y = pd.DataFrame([[21.0, 26.0, 22.0, 9.0, 9.0, 3.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[3.0, 4.0, 2.0, 5.0, 6.0, 1.0]], index=idx, columns=cols)
    group = pd.DataFrame([["A", "A", "A", "B", "B", "B"]], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(
        y, group, x1, x2, add_intercept=True, min_obs=None
    )

    # With perfect fit and non-collinear features, residuals should be near zero
    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, np.zeros(6), atol=1e-9)


def test_group_multi_resid_three_features_with_noise() -> None:
    """Three-feature regression with noise."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=1)
    n = 20
    cols = [f"C{i}" for i in range(n)]

    # Single group with noisy linear relationship
    x1_vals = np.random.randn(n)
    x2_vals = np.random.randn(n)
    x3_vals = np.random.randn(n)
    y_vals = 2.0 * x1_vals - 1.5 * x2_vals + 0.5 * x3_vals + 10.0 + 0.1 * np.random.randn(n)

    y = pd.DataFrame([y_vals], index=idx, columns=cols)
    x1 = pd.DataFrame([x1_vals], index=idx, columns=cols)
    x2 = pd.DataFrame([x2_vals], index=idx, columns=cols)
    x3 = pd.DataFrame([x3_vals], index=idx, columns=cols)
    group = pd.DataFrame([["G"] * n], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(
        y, group, x1, x2, x3, add_intercept=True
    )

    result = out.to_numpy()[0]
    # Residuals should have small variance (good fit)
    assert np.nanstd(result) < 0.2
    # Mean residual should be near zero
    assert abs(np.nanmean(result)) < 0.05


def test_group_multi_resid_no_intercept() -> None:
    """Regression without intercept: y ~ x1 + x2 (no constant)."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    # Perfect relationship y = 2*x1 + 3*x2 (no intercept)
    x1_vals = np.array([1.0, 2.0, 3.0, 4.0])
    x2_vals = np.array([1.0, 1.0, 1.0, 1.0])
    y_vals = 2.0 * x1_vals + 3.0 * x2_vals

    y = pd.DataFrame([y_vals], index=idx, columns=cols)
    x1 = pd.DataFrame([x1_vals], index=idx, columns=cols)
    x2 = pd.DataFrame([x2_vals], index=idx, columns=cols)
    group = pd.DataFrame([["G"] * 4], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(
        y, group, x1, x2, add_intercept=False
    )

    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, np.zeros(4), atol=1e-9)


# ---------------------------------------------------------------------------
# Singular matrix / degenerate features
# ---------------------------------------------------------------------------
def test_group_multi_resid_zero_variance_feature() -> None:
    """Feature with zero variance -> fail-closed (NaN)."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    y = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[5.0, 5.0, 5.0, 5.0]], index=idx, columns=cols)  # constant
    group = pd.DataFrame([["G"] * 4], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(y, group, x1, x2)

    # Zero variance in x2 -> singular matrix -> all NaN
    assert out.isna().all().all()


def test_group_multi_resid_collinear_features() -> None:
    """Perfectly collinear features -> singular matrix -> NaN."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDE")

    y = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[2.0, 4.0, 6.0, 8.0, 10.0]], index=idx, columns=cols)  # x2 = 2*x1
    group = pd.DataFrame([["G"] * 5], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(y, group, x1, x2, add_intercept=False)

    # Perfect collinearity -> singular -> NaN
    # Note: with add_intercept=True, lstsq might still succeed due to numerical behavior
    # but with False, the matrix is strictly singular
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# Sample size constraints
# ---------------------------------------------------------------------------
def test_group_multi_resid_insufficient_samples_default() -> None:
    """Default min_obs = n_features + 2 (with intercept)."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABC")

    # 3 samples, 2 features + intercept = 3 parameters -> need 5 samples by default
    y = pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[4.0, 5.0, 6.0]], index=idx, columns=cols)
    group = pd.DataFrame([["G"] * 3], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(y, group, x1, x2, add_intercept=True)

    # 3 < 5 -> fail-closed
    assert out.isna().all().all()


def test_group_multi_resid_custom_min_obs() -> None:
    """Custom min_obs allows smaller sample."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    # Use non-collinear features
    y = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[2.0, 3.0, 1.0, 5.0]], index=idx, columns=cols)  # varying
    group = pd.DataFrame([["G"] * 4], index=idx, columns=cols)

    # Default would need 3 params (2 features + intercept), we have exactly 4 obs
    out = _op("group_multi_resid").calculate(
        y, group, x1, x2, add_intercept=True, min_obs=4
    )

    # Should succeed with custom min_obs=4 (4 obs >= 3 params)
    result = out.to_numpy()[0]
    assert np.isfinite(result).any()


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------
def test_group_multi_resid_joint_missing_filter() -> None:
    """Joint filter: valid only if y AND all x's are finite."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    # A has NaN in x1, B has NaN in x2, C has NaN in y
    # Make x1 and x2 non-collinear for valid samples (D, E, F)
    y = pd.DataFrame([[1.0, 2.0, np.nan, 4.0, 5.0, 6.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[np.nan, 2.0, 3.0, 1.0, 2.0, 3.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[1.0, np.nan, 3.0, 2.0, 3.0, 1.0]], index=idx, columns=cols)
    group = pd.DataFrame([["G"] * 6], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(
        y, group, x1, x2, add_intercept=True, min_obs=3
    )

    result = out.to_numpy()[0]
    # A, B, C have NaN -> only D, E, F are valid (3 samples, exactly min_obs)
    assert np.isnan(result[0])  # A: x1 missing
    assert np.isnan(result[1])  # B: x2 missing
    assert np.isnan(result[2])  # C: y missing
    # D, E, F should have valid residuals
    assert np.isfinite(result[3:]).all()


def test_group_multi_resid_all_nan_group() -> None:
    """Group with all NaN values -> fail-closed."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    y = pd.DataFrame([[1.0, 2.0, 3.0, np.nan, np.nan, np.nan]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0, np.nan, np.nan, np.nan]], index=idx, columns=cols)
    x2 = pd.DataFrame([[2.0, 3.0, 1.0, np.nan, np.nan, np.nan]], index=idx, columns=cols)
    group = pd.DataFrame([["A", "A", "A", "B", "B", "B"]], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(y, group, x1, x2, min_obs=3)

    result = out.to_numpy()[0]
    # Group A: valid residuals
    assert np.isfinite(result[:3]).all()
    # Group B: all NaN -> fail-closed
    assert np.isnan(result[3:]).all()


# ---------------------------------------------------------------------------
# Multi-group independence
# ---------------------------------------------------------------------------
def test_group_multi_resid_independent_groups() -> None:
    """Each group fits independently."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(8)]

    # Group A: y = 2*x1 + 1*x2 + 10
    # Group B: y = -1*x1 + 3*x2 + 5
    # Use non-collinear x1 and x2 values
    x1_vals = [1.0, 2.0, 3.0, 4.0,     # A
               1.0, 2.0, 3.0, 2.0]     # B
    x2_vals = [3.0, 2.0, 4.0, 3.0,     # A
               3.0, 4.0, 2.0, 5.0]     # B

    # Compute y values from the formulas
    y_vals = [
        # Group A: y = 2*x1 + 1*x2 + 10
        2*1.0 + 1*3.0 + 10,  # 15
        2*2.0 + 1*2.0 + 10,  # 16
        2*3.0 + 1*4.0 + 10,  # 20
        2*4.0 + 1*3.0 + 10,  # 21
        # Group B: y = -1*x1 + 3*x2 + 5
        -1*1.0 + 3*3.0 + 5,  # 13
        -1*2.0 + 3*4.0 + 5,  # 15
        -1*3.0 + 3*2.0 + 5,  # 8
        -1*2.0 + 3*5.0 + 5,  # 18
    ]
    group_vals = ["A", "A", "A", "A", "B", "B", "B", "B"]

    y = pd.DataFrame([y_vals], index=idx, columns=cols)
    x1 = pd.DataFrame([x1_vals], index=idx, columns=cols)
    x2 = pd.DataFrame([x2_vals], index=idx, columns=cols)
    group = pd.DataFrame([group_vals], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(y, group, x1, x2, add_intercept=True)

    result = out.to_numpy()[0]
    # Both groups should have near-zero residuals (perfect fit)
    np.testing.assert_allclose(result, np.zeros(8), atol=1e-9)


# ---------------------------------------------------------------------------
# Variadic parameters (x1 to x5)
# ---------------------------------------------------------------------------
def test_group_multi_resid_five_features() -> None:
    """Maximum 5 features."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=1)
    n = 30
    cols = [f"C{i}" for i in range(n)]

    # Generate random features and linear combination
    x_vals = [np.random.randn(n) for _ in range(5)]
    coeffs = [1.0, -0.5, 0.8, -1.2, 0.3]
    y_vals = sum(c * x for c, x in zip(coeffs, x_vals)) + 5.0 + 0.05 * np.random.randn(n)

    y = pd.DataFrame([y_vals], index=idx, columns=cols)
    xs = [pd.DataFrame([x], index=idx, columns=cols) for x in x_vals]
    group = pd.DataFrame([["G"] * n], index=idx, columns=cols)

    out = _op("group_multi_resid").calculate(
        y, group, xs[0], xs[1], xs[2], xs[3], xs[4], add_intercept=True
    )

    result = out.to_numpy()[0]
    # Should have small residuals
    assert np.nanstd(result) < 0.1


def test_group_multi_resid_optional_features_none() -> None:
    """Optional features (x3, x4, x5) can be None."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    y = pd.DataFrame([[10.0, 15.0, 20.0, 25.0]], index=idx, columns=cols)
    x1 = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    x2 = pd.DataFrame([[3.0, 2.0, 4.0, 3.0]], index=idx, columns=cols)  # Non-collinear
    group = pd.DataFrame([["G"] * 4], index=idx, columns=cols)

    # Only x1 and x2, others None
    out = _op("group_multi_resid").calculate(
        y, group, x1, x2, x3=None, x4=None, x5=None, add_intercept=True
    )

    result = out.to_numpy()[0]
    assert np.isfinite(result).any()


# ---------------------------------------------------------------------------
# Unit consistency
# ---------------------------------------------------------------------------
def test_group_multi_resid_output_unit() -> None:
    """Output unit should be same_as:y."""
    op = _op("group_multi_resid")
    meta = getattr(op, "metadata", None)
    assert meta is not None
    # Check metadata declares unit:same_as:y
    tags = getattr(meta, "tags", [])
    unit = getattr(meta, "unit", None) or getattr(meta, "output_unit", None)
    assert any("unit:same_as:y" in str(tag) or "same_as:y" in str(tag) for tag in tags) or \
           (unit and "same_as:y" in str(unit))


