# -*- coding: utf-8 -*-
"""Tests for distribution_break operators.

Coverage of 3 operators:
1. ts_joint_energy_shift
2. ts_energy_break_score
3. ts_copula_central_asymmetry
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


def _calculate(op, x):
    """Exercise the actual panel topology with supported short histories."""
    name = op.metadata.name
    y = x.shift(1) * 0.7 + x * x
    if name == "ts_copula_central_asymmetry":
        return op.calculate(x, y, window=10)
    kwargs = dict(recent_window=2, prior_window=4)
    if name == "ts_energy_break_score":
        kwargs["window"] = 3
    return op.calculate(x, y, x.shift(2) - x, **kwargs)


# ---------------------------------------------------------------------------
# 1. ts_joint_energy_shift
# ---------------------------------------------------------------------------
def test_ts_joint_energy_shift_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_joint_energy_shift")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_joint_energy_shift_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_joint_energy_shift")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_joint_energy_shift_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_joint_energy_shift")
    result1 = _calculate(op, x)
    result2 = _calculate(op, x)
    pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. ts_energy_break_score
# ---------------------------------------------------------------------------
def test_ts_energy_break_score_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_energy_break_score")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_energy_break_score_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_energy_break_score")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_energy_break_score_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_energy_break_score")
    result1 = _calculate(op, x)
    result2 = _calculate(op, x)
    pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. ts_copula_central_asymmetry
# ---------------------------------------------------------------------------
def test_ts_copula_central_asymmetry_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_copula_central_asymmetry")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_copula_central_asymmetry_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_copula_central_asymmetry")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_copula_central_asymmetry_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_copula_central_asymmetry")
    result1 = _calculate(op, x)
    result2 = _calculate(op, x)
    pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_distribution_break_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_joint_energy_shift",
        "ts_energy_break_score",
        "ts_copula_central_asymmetry"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
