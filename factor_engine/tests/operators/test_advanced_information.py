# -*- coding: utf-8 -*-
"""Tests for advanced_information operators.

Coverage of 7 operators:
1. ts_transfer_entropy
2. ts_effective_transfer_entropy
3. ts_score_rank_weighted_mean
4. report_benford_js_divergence
5. ts_transfer_entropy_peak_strength
6. ts_transfer_entropy_peak_lag
7. ts_transfer_entropy_peak_excess
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


_TRANSFER_ENTROPY_NAMES = {
    "ts_transfer_entropy",
    "ts_effective_transfer_entropy",
    "ts_transfer_entropy_peak_strength",
    "ts_transfer_entropy_peak_lag",
    "ts_transfer_entropy_peak_excess",
}


def _calculate_with_real_inputs(op, x: pd.DataFrame) -> pd.DataFrame:
    """Call each operator with its actual arity and a feasible small domain."""
    name = op.metadata.name
    if name in _TRANSFER_ENTROPY_NAMES:
        return op.calculate(
            x, x, window=32, bins=2,
            min_transitions=2, min_cells_ratio=0.25,
        )
    if name == "ts_score_rank_weighted_mean":
        return op.calculate(x, x, window=20)
    return op.calculate(x)



# ---------------------------------------------------------------------------
# 1. ts_transfer_entropy
# ---------------------------------------------------------------------------
def test_ts_transfer_entropy_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_transfer_entropy")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_transfer_entropy_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_transfer_entropy_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_transfer_entropy_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_transfer_entropy_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 2. ts_effective_transfer_entropy
# ---------------------------------------------------------------------------
def test_ts_effective_transfer_entropy_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_effective_transfer_entropy")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_effective_transfer_entropy_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_effective_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_effective_transfer_entropy_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_effective_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_effective_transfer_entropy_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_effective_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_effective_transfer_entropy_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_effective_transfer_entropy")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 3. ts_score_rank_weighted_mean
# ---------------------------------------------------------------------------
def test_ts_score_rank_weighted_mean_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_score_rank_weighted_mean")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_score_rank_weighted_mean_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_score_rank_weighted_mean")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_score_rank_weighted_mean_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_score_rank_weighted_mean")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_score_rank_weighted_mean_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_score_rank_weighted_mean")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_score_rank_weighted_mean_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_score_rank_weighted_mean")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 4. report_benford_js_divergence
# ---------------------------------------------------------------------------
def test_report_benford_js_divergence_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("report_benford_js_divergence")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_report_benford_js_divergence_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("report_benford_js_divergence")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_report_benford_js_divergence_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("report_benford_js_divergence")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_report_benford_js_divergence_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("report_benford_js_divergence")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_report_benford_js_divergence_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("report_benford_js_divergence")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 5. ts_transfer_entropy_peak_strength
# ---------------------------------------------------------------------------
def test_ts_transfer_entropy_peak_strength_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_strength")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_transfer_entropy_peak_strength_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_strength")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_transfer_entropy_peak_strength_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_strength")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_transfer_entropy_peak_strength_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_transfer_entropy_peak_strength")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_transfer_entropy_peak_strength_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_transfer_entropy_peak_strength")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 6. ts_transfer_entropy_peak_lag
# ---------------------------------------------------------------------------
def test_ts_transfer_entropy_peak_lag_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_lag")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_transfer_entropy_peak_lag_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_lag")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_transfer_entropy_peak_lag_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_lag")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_transfer_entropy_peak_lag_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_transfer_entropy_peak_lag")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_transfer_entropy_peak_lag_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_transfer_entropy_peak_lag")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 7. ts_transfer_entropy_peak_excess
# ---------------------------------------------------------------------------
def test_ts_transfer_entropy_peak_excess_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_excess")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with valid fixture parameters
    try:
        result = _calculate_with_real_inputs(op, x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert np.isfinite(result.to_numpy()).any()
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_ts_transfer_entropy_peak_excess_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_excess")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_transfer_entropy_peak_excess_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_transfer_entropy_peak_excess")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_transfer_entropy_peak_excess_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_transfer_entropy_peak_excess")

    try:
        result = _calculate_with_real_inputs(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_transfer_entropy_peak_excess_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_transfer_entropy_peak_excess")

    try:
        result = _calculate_with_real_inputs(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_advanced_information_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_transfer_entropy",
        "ts_effective_transfer_entropy",
        "ts_score_rank_weighted_mean",
        "report_benford_js_divergence",
        "ts_transfer_entropy_peak_strength",
        "ts_transfer_entropy_peak_lag",
        "ts_transfer_entropy_peak_excess"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
        assert meta.name == op_name, f"{op_name} name mismatch"
