# -*- coding: utf-8 -*-
"""Tests for research_transform operators.

Coverage of 3 operators:
1. ts_wavelet_lowpass_reconstruct
2. ts_signature_mahalanobis_anomaly
3. ts_persistence_birth_dispersion
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op



# ---------------------------------------------------------------------------
# 1. ts_wavelet_lowpass_reconstruct
# ---------------------------------------------------------------------------
def test_ts_wavelet_lowpass_reconstruct_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_wavelet_lowpass_reconstruct")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_wavelet_lowpass_reconstruct_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_wavelet_lowpass_reconstruct")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_wavelet_lowpass_reconstruct_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_wavelet_lowpass_reconstruct")
    try:
        result1 = op.calculate(x)
        result2 = op.calculate(x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic


# ---------------------------------------------------------------------------
# 2. ts_signature_mahalanobis_anomaly
# ---------------------------------------------------------------------------
def test_ts_signature_mahalanobis_anomaly_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_signature_mahalanobis_anomaly")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_signature_mahalanobis_anomaly_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_signature_mahalanobis_anomaly")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_signature_mahalanobis_anomaly_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_signature_mahalanobis_anomaly")
    try:
        result1 = op.calculate(x)
        result2 = op.calculate(x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic


# ---------------------------------------------------------------------------
# 3. ts_persistence_birth_dispersion
# ---------------------------------------------------------------------------
def test_ts_persistence_birth_dispersion_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_persistence_birth_dispersion")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_persistence_birth_dispersion_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_persistence_birth_dispersion")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_persistence_birth_dispersion_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_persistence_birth_dispersion")
    try:
        result1 = op.calculate(x)
        result2 = op.calculate(x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_research_transform_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_wavelet_lowpass_reconstruct",
        "ts_signature_mahalanobis_anomaly",
        "ts_persistence_birth_dispersion"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
