# -*- coding: utf-8 -*-
"""Tests for conditional_ext operators.

Coverage of 6 operators:
1. ts_min_if
2. ts_max_if
3. ts_quantile_if
4. ts_corr_if
5. ts_beta_if
6. ts_regression_resid_if
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


def _calculate(op, x: pd.DataFrame) -> pd.DataFrame:
    rows = np.arange(len(x))[:, None]
    condition = pd.DataFrame(
        np.broadcast_to((rows % 3 != 1).astype(float), x.shape).copy(),
        index=x.index, columns=x.columns,
    ).where(x.notna())
    kwargs = {"window": 5, "min_periods": 1}
    if op.metadata.name == "ts_quantile_if":
        return op.calculate(x, condition, q=0.5, **kwargs)
    if op.metadata.name in {"ts_corr_if", "ts_beta_if", "ts_regression_resid_if"}:
        y = 1.5 * x + pd.DataFrame(
            np.broadcast_to(rows.astype(float), x.shape), index=x.index, columns=x.columns
        )
        kwargs["min_periods"] = 3 if op.metadata.name == "ts_regression_resid_if" else 2
        return op.calculate(x, y, condition, **kwargs)
    return op.calculate(x, condition, **kwargs)



# ---------------------------------------------------------------------------
# 1. ts_min_if
# ---------------------------------------------------------------------------
def test_ts_min_if_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_min_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_min_if_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_min_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_min_if_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_min_if")
    try:
        result1 = _calculate(op, x)
        result2 = _calculate(op, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 2. ts_max_if
# ---------------------------------------------------------------------------
def test_ts_max_if_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_max_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_max_if_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_max_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_max_if_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_max_if")
    try:
        result1 = _calculate(op, x)
        result2 = _calculate(op, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 3. ts_quantile_if
# ---------------------------------------------------------------------------
def test_ts_quantile_if_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_quantile_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_quantile_if_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_quantile_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_quantile_if_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_quantile_if")
    try:
        result1 = _calculate(op, x)
        result2 = _calculate(op, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 4. ts_corr_if
# ---------------------------------------------------------------------------
def test_ts_corr_if_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_corr_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_corr_if_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_corr_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_corr_if_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_corr_if")
    try:
        result1 = _calculate(op, x)
        result2 = _calculate(op, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 5. ts_beta_if
# ---------------------------------------------------------------------------
def test_ts_beta_if_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_beta_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_beta_if_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_beta_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_beta_if_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_beta_if")
    try:
        result1 = _calculate(op, x)
        result2 = _calculate(op, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")


# ---------------------------------------------------------------------------
# 6. ts_regression_resid_if
# ---------------------------------------------------------------------------
def test_ts_regression_resid_if_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_regression_resid_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_ts_regression_resid_if_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("ts_regression_resid_if")
    try:
        result = _calculate(op, x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_ts_regression_resid_if_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("ts_regression_resid_if")
    try:
        result1 = _calculate(op, x)
        result2 = _calculate(op, x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception as e:
        pytest.fail(f"Determinism check failed: {e}")



@pytest.mark.parametrize("name,reducer", [("ts_min_if", np.min), ("ts_max_if", np.max)])
def test_min_max_if_mask_oracle_and_prefix_causality(name, reducer) -> None:
    index = pd.date_range("2024-03-01", periods=8)
    x = pd.DataFrame({"A": [5.0, 2.0, 9.0, 4.0, 8.0, 1.0, 7.0, 3.0]}, index=index)
    condition = pd.DataFrame({"A": [1.0, 0.0, 1.0, np.nan, 1.0, 1.0, 0.0, 1.0]}, index=index)
    window = 4
    result = _op(name).calculate(x, condition, window=window, min_periods=1)
    oracle = []
    for row in range(len(x)):
        start = max(0, row - window + 1)
        selected = [
            x.iloc[i, 0] for i in range(start, row + 1)
            if condition.iloc[i, 0] == 1.0 and np.isfinite(x.iloc[i, 0])
        ]
        oracle.append(float(reducer(selected)) if selected else np.nan)
    np.testing.assert_allclose(result["A"], oracle, equal_nan=True)

    changed_future = x.copy()
    changed_future.iloc[5:] = 1000.0
    changed = _op(name).calculate(
        changed_future, condition, window=window, min_periods=1
    )
    pd.testing.assert_series_equal(result["A"].iloc[:5], changed["A"].iloc[:5])


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_conditional_ext_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_min_if",
        "ts_max_if",
        "ts_quantile_if",
        "ts_corr_if",
        "ts_beta_if",
        "ts_regression_resid_if"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
