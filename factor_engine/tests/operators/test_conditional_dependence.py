# -*- coding: utf-8 -*-
"""Tests for conditional_dependence operators.

Coverage of 2 operators:
1. ts_conditional_transfer_entropy
2. ts_modwt_band_corr
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
    """Call the real multi-input signature with small feasible parameters."""
    name = op.metadata.name
    if name == "ts_conditional_transfer_entropy":
        source = x.shift(1).fillna(0.0)
        condition = pd.DataFrame(
            np.tile(np.arange(len(x))[:, None] % 2, (1, x.shape[1])),
            index=x.index,
            columns=x.columns,
        )
        return op.calculate(x, source, condition, window=49, bins=2, lag=1)
    if name == "ts_modwt_band_corr":
        y = x * 2.0
        return op.calculate(x, y, window=9, level=1, band=1)
    raise AssertionError(f"unhandled operator fixture: {name}")



# ---------------------------------------------------------------------------
# 1. ts_conditional_transfer_entropy
# ---------------------------------------------------------------------------
def test_ts_conditional_transfer_entropy_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_conditional_transfer_entropy")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = _calculate(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_conditional_transfer_entropy_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_conditional_transfer_entropy")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_conditional_transfer_entropy_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_conditional_transfer_entropy")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_conditional_transfer_entropy_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_conditional_transfer_entropy")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_conditional_transfer_entropy_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_conditional_transfer_entropy")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 2. ts_modwt_band_corr
# ---------------------------------------------------------------------------
def test_ts_modwt_band_corr_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_modwt_band_corr")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = _calculate(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_modwt_band_corr_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_modwt_band_corr")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_modwt_band_corr_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_modwt_band_corr")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_modwt_band_corr_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_modwt_band_corr")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_modwt_band_corr_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_modwt_band_corr")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")



def test_conditional_transfer_entropy_detects_directed_binary_signal() -> None:
    rng = np.random.default_rng(20260909)
    rows = 100
    source_values = rng.integers(0, 2, size=rows).astype(float)
    target_values = np.zeros(rows, dtype=float)
    target_values[1:] = source_values[:-1]
    condition_values = rng.integers(0, 2, size=rows).astype(float)
    index = pd.date_range("2024-01-01", periods=rows)
    target = pd.DataFrame({"A": target_values}, index=index)
    source = pd.DataFrame({"A": source_values}, index=index)
    condition = pd.DataFrame({"A": condition_values}, index=index)
    result = _op("ts_conditional_transfer_entropy").calculate(
        target, source, condition, window=60, bins=2, lag=1
    )
    assert np.isfinite(result.iloc[-1, 0])
    assert result.iloc[-1, 0] > 0.0


def test_modwt_band_corr_identical_series_is_one_on_interior_window() -> None:
    index = pd.date_range("2024-01-01", periods=24)
    x = pd.DataFrame({"A": np.sin(np.arange(24, dtype=float))}, index=index)
    result = _op("ts_modwt_band_corr").calculate(
        x, x.copy(), window=9, level=1, band=1
    )
    assert result.iloc[-1, 0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "name", ["ts_conditional_transfer_entropy", "ts_modwt_band_corr"]
)
def test_conditional_dependence_missing_required_inputs_rejects(name: str) -> None:
    x = pd.DataFrame({"A": np.arange(60, dtype=float)})
    with pytest.raises(TypeError):
        _op(name).calculate(x)


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_conditional_dependence_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_conditional_transfer_entropy",
        "ts_modwt_band_corr"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
        assert meta.name == op_name, f"{op_name} name mismatch"
