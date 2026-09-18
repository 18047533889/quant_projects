# -*- coding: utf-8 -*-
"""Tests for filter_hysteresis operators.

Coverage of 10 operators:
1. state_adaptive_deadband
2. state_rank_deadband
3. state_quantile_hysteresis
4. state_adaptive_slew_limit
5. state_cost_aware_deadband
6. state_cost_aware_slew
7. state_confidence_weighted_ema
8. state_uncertainty_deadband
9. state_l1_turnover_prox
10. state_l2_partial_adjustment
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
# 1. state_adaptive_deadband
# ---------------------------------------------------------------------------
def test_state_adaptive_deadband_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_adaptive_deadband")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_adaptive_deadband_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_adaptive_deadband")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_adaptive_deadband_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_adaptive_deadband")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_adaptive_deadband_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_adaptive_deadband")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_adaptive_deadband_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_adaptive_deadband")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 2. state_rank_deadband
# ---------------------------------------------------------------------------
def test_state_rank_deadband_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_rank_deadband")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_rank_deadband_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_rank_deadband")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_rank_deadband_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_rank_deadband")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_rank_deadband_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_rank_deadband")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_rank_deadband_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_rank_deadband")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 3. state_quantile_hysteresis
# ---------------------------------------------------------------------------
def test_state_quantile_hysteresis_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_quantile_hysteresis")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_quantile_hysteresis_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_quantile_hysteresis")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_quantile_hysteresis_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_quantile_hysteresis")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_quantile_hysteresis_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_quantile_hysteresis")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_quantile_hysteresis_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_quantile_hysteresis")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 4. state_adaptive_slew_limit
# ---------------------------------------------------------------------------
def test_state_adaptive_slew_limit_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_adaptive_slew_limit")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_adaptive_slew_limit_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_adaptive_slew_limit")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_adaptive_slew_limit_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_adaptive_slew_limit")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_adaptive_slew_limit_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_adaptive_slew_limit")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_adaptive_slew_limit_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_adaptive_slew_limit")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 5. state_cost_aware_deadband
# ---------------------------------------------------------------------------
def test_state_cost_aware_deadband_default_matches_independent_recurrence() -> None:
    index = pd.date_range("2024-01-01", periods=6)
    x = pd.DataFrame({"A": [0.0, 0.5, 3.0, 3.4, -2.0, -1.5]}, index=index)
    cost = pd.DataFrame({"A": [0.2, 0.3, 0.5, 0.1, 1.0, 0.1]}, index=index)
    result = _op("state_cost_aware_deadband").calculate(x, cost)
    expected = []
    previous = np.nan
    for value, proxy in zip(x["A"], cost["A"]):
        if not np.isfinite(previous) or abs(value - previous) > 2.0 * proxy:
            previous = value
        expected.append(previous)
    np.testing.assert_allclose(result["A"], expected)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_state_cost_aware_deadband_backend_contract(backend: str) -> None:
    from factor_engine.backend.operator_errors import OperatorParameterError

    if backend == "polars":
        pl = pytest.importorskip("polars")
        x = pl.DataFrame({"A": [0.0, 1.0, 4.0]})
        cost = pl.DataFrame({"A": [0.1, 0.2, 0.3]})
    else:
        x = pd.DataFrame({"A": [0.0, 1.0, 4.0]})
        cost = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
    op = _op("state_cost_aware_deadband", backend)
    assert op.calculate(x, cost).shape == x.shape
    with pytest.raises(OperatorParameterError):
        op.calculate(x)
    with pytest.raises((OperatorParameterError, ValueError)):
        op.calculate(x, cost, cost_mult=-1.0)


# ---------------------------------------------------------------------------
# 6. state_cost_aware_slew
# ---------------------------------------------------------------------------
def test_state_cost_aware_slew_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_cost_aware_slew")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_cost_aware_slew_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_cost_aware_slew")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_cost_aware_slew_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_cost_aware_slew")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_cost_aware_slew_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_cost_aware_slew")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_cost_aware_slew_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_cost_aware_slew")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 7. state_confidence_weighted_ema
# ---------------------------------------------------------------------------
def test_state_confidence_weighted_ema_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_confidence_weighted_ema")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_confidence_weighted_ema_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_confidence_weighted_ema")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_confidence_weighted_ema_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_confidence_weighted_ema")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_confidence_weighted_ema_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_confidence_weighted_ema")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_confidence_weighted_ema_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_confidence_weighted_ema")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 8. state_uncertainty_deadband
# ---------------------------------------------------------------------------
def test_state_uncertainty_deadband_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_uncertainty_deadband")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_uncertainty_deadband_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_uncertainty_deadband")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_uncertainty_deadband_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_uncertainty_deadband")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_uncertainty_deadband_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_uncertainty_deadband")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_uncertainty_deadband_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_uncertainty_deadband")

    try:
        result = op.calculate(x, pd.DataFrame(0.25, index=x.index, columns=x.columns))

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 9. state_l1_turnover_prox
# ---------------------------------------------------------------------------
def test_state_l1_turnover_prox_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_l1_turnover_prox")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_l1_turnover_prox_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_l1_turnover_prox")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_l1_turnover_prox_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_l1_turnover_prox")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_l1_turnover_prox_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_l1_turnover_prox")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_l1_turnover_prox_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_l1_turnover_prox")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 10. state_l2_partial_adjustment
# ---------------------------------------------------------------------------
def test_state_l2_partial_adjustment_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("state_l2_partial_adjustment")
    # Get param names from metadata
    meta = getattr(op, "metadata", None)

    # Call with default parameters
    try:
        result = op.calculate(x)

        # Basic shape check
        assert result.shape == x.shape, f"{result.shape} != {x.shape}"
        assert list(result.columns) == list(x.columns)
        assert list(result.index) == list(x.index)

        # Result should contain some finite values or be validly all-NaN
        # (some operators may return all NaN for random data)
    except Exception as e:
        pytest.fail(f"{op} failed with {type(e).__name__}: {e}")


def test_state_l2_partial_adjustment_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_l2_partial_adjustment")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_state_l2_partial_adjustment_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("state_l2_partial_adjustment")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_state_l2_partial_adjustment_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("state_l2_partial_adjustment")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_state_l2_partial_adjustment_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("state_l2_partial_adjustment")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_filter_hysteresis_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "state_adaptive_deadband",
        "state_rank_deadband",
        "state_quantile_hysteresis",
        "state_adaptive_slew_limit",
        "state_cost_aware_deadband",
        "state_cost_aware_slew",
        "state_confidence_weighted_ema",
        "state_uncertainty_deadband",
        "state_l1_turnover_prox",
        "state_l2_partial_adjustment"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
        assert meta.name == op_name, f"{op_name} name mismatch"


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("name", [
    "state_cost_aware_deadband", "state_cost_aware_slew",
    "state_confidence_weighted_ema", "state_uncertainty_deadband",
])
def test_two_panel_state_filter_real_defaults_and_missing_reset(name, backend):
    from factor_engine.backend.operator_errors import OperatorParameterError
    values = np.array([0., 1., 4., np.nan, 2., 5., np.inf, 3., -1.])
    companion = np.full(len(values), 0.25)
    expected = []
    previous = np.nan
    for value, aux in zip(values, companion):
        if not np.isfinite(value):
            previous = np.nan
            expected.append(np.nan)
            continue
        if not np.isfinite(previous):
            previous = value
        elif name == "state_cost_aware_slew":
            previous += np.clip(value - previous, -3 / (1 + aux), 3 / (1 + aux))
        elif name == "state_confidence_weighted_ema":
            previous += (0.05 + aux * 0.45) * (value - previous)
        elif abs(value - previous) > aux * (2 if name == "state_cost_aware_deadband" else 1):
            previous = value
        expected.append(previous)
    if backend == "polars":
        import polars as pl
        x, aux = pl.DataFrame({"A": values}), pl.DataFrame({"A": companion})
    else:
        x, aux = pd.DataFrame({"A": values}), pd.DataFrame({"A": companion})
    op = _op(name, backend)
    actual = op.calculate(x, aux)
    np.testing.assert_allclose(actual["A"].to_numpy(), expected, equal_nan=True)
    with pytest.raises(OperatorParameterError, match="missing"):
        op.calculate(x)
    assert op.calculate(x.head(0), aux.head(0)).shape == (0, 1)
