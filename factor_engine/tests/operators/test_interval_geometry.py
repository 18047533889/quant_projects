# -*- coding: utf-8 -*-
"""Tests for interval_geometry operators.

Coverage of 5 operators:
1. ts_interval_union_coverage
2. ts_interval_occupancy_entropy
3. ts_interval_occupancy_mode_distance
4. ts_interval_nesting_depth
5. ts_interval_exploration_efficiency
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
# 1. ts_interval_union_coverage
# ---------------------------------------------------------------------------
def test_ts_interval_union_coverage_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_interval_union_coverage")
    try:
        result = op.calculate(x, x)  # low and high are the same for testing
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_interval_union_coverage_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_interval_union_coverage")
    try:
        result = op.calculate(x, x)  # low and high are the same for testing
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_interval_union_coverage_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_interval_union_coverage")
    try:
        result1 = op.calculate(x, x)  # low and high are the same for testing
        result2 = op.calculate(x, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 2. ts_interval_occupancy_entropy
# ---------------------------------------------------------------------------
def test_ts_interval_occupancy_entropy_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_interval_occupancy_entropy")
    try:
        result = op.calculate(x, x)  # low and high are the same for testing
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_interval_occupancy_entropy_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_interval_occupancy_entropy")
    try:
        result = op.calculate(x, x)  # low and high are the same for testing
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_interval_occupancy_entropy_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_interval_occupancy_entropy")
    try:
        result1 = op.calculate(x, x)  # low and high are the same for testing
        result2 = op.calculate(x, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 3. ts_interval_occupancy_mode_distance
# ---------------------------------------------------------------------------
def test_ts_interval_occupancy_mode_distance_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_interval_occupancy_mode_distance")
    try:
        result = op.calculate(x, x, x)  # x, low, high are the same for testing
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_interval_occupancy_mode_distance_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_interval_occupancy_mode_distance")
    try:
        result = op.calculate(x, x, x)  # x, low, high are the same for testing
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_interval_occupancy_mode_distance_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_interval_occupancy_mode_distance")
    try:
        result1 = op.calculate(x, x, x)  # x, low, high are the same for testing
        result2 = op.calculate(x, x, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 4. ts_interval_nesting_depth
# ---------------------------------------------------------------------------
def test_ts_interval_nesting_depth_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_interval_nesting_depth")
    try:
        result = op.calculate(x, x)  # low and high are the same for testing
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_interval_nesting_depth_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_interval_nesting_depth")
    try:
        result = op.calculate(x, x)  # low and high are the same for testing
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_interval_nesting_depth_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_interval_nesting_depth")
    try:
        result1 = op.calculate(x, x)  # low and high are the same for testing
        result2 = op.calculate(x, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 5. ts_interval_exploration_efficiency
# ---------------------------------------------------------------------------
def test_ts_interval_exploration_efficiency_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_interval_exploration_efficiency")
    try:
        result = op.calculate(x, x, x)  # high, low, close are the same for testing
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_interval_exploration_efficiency_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_interval_exploration_efficiency")
    try:
        result = op.calculate(x, x, x)  # high, low, close are the same for testing
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_interval_exploration_efficiency_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_interval_exploration_efficiency")
    try:
        result1 = op.calculate(x, x, x)  # high, low, close are the same for testing
        result2 = op.calculate(x, x, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_interval_geometry_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_interval_union_coverage",
        "ts_interval_occupancy_entropy",
        "ts_interval_occupancy_mode_distance",
        "ts_interval_nesting_depth",
        "ts_interval_exploration_efficiency"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
