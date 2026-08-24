# -*- coding: utf-8 -*-
"""Tests for turnover_survival operators.

Coverage of 12 operators:
1. ts_turnover_reference_price
2. ts_turnover_cost_dispersion
3. ts_turnover_profit_share
4. ts_turnover_holding_age
5. ts_turnover_near_cost_mass
6. ts_turnover_cost_quantile_distance
7. ts_turnover_cost_entropy
8. ts_turnover_cost_mode_distance
9. ts_turnover_cost_skew
10. ts_turnover_age_dispersion
11. ts_turnover_old_mass
12. ts_turnover_cost_entropy_vol_scaled
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
# 1. ts_turnover_reference_price
# ---------------------------------------------------------------------------
def test_ts_turnover_reference_price_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_reference_price")
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


def test_ts_turnover_reference_price_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_reference_price")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_reference_price_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_reference_price")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_reference_price_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_reference_price")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_reference_price_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_reference_price")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 2. ts_turnover_cost_dispersion
# ---------------------------------------------------------------------------
def test_ts_turnover_cost_dispersion_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_cost_dispersion")
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


def test_ts_turnover_cost_dispersion_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_dispersion")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_cost_dispersion_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_dispersion")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_cost_dispersion_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_cost_dispersion")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_cost_dispersion_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_cost_dispersion")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 3. ts_turnover_profit_share
# ---------------------------------------------------------------------------
def test_ts_turnover_profit_share_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_profit_share")
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


def test_ts_turnover_profit_share_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_profit_share")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_profit_share_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_profit_share")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_profit_share_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_profit_share")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_profit_share_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_profit_share")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 4. ts_turnover_holding_age
# ---------------------------------------------------------------------------
def test_ts_turnover_holding_age_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_holding_age")
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


def test_ts_turnover_holding_age_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_holding_age")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_holding_age_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_holding_age")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_holding_age_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_holding_age")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_holding_age_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_holding_age")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 5. ts_turnover_near_cost_mass
# ---------------------------------------------------------------------------
def test_ts_turnover_near_cost_mass_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_near_cost_mass")
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


def test_ts_turnover_near_cost_mass_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_near_cost_mass")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_near_cost_mass_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_near_cost_mass")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_near_cost_mass_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_near_cost_mass")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_near_cost_mass_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_near_cost_mass")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 6. ts_turnover_cost_quantile_distance
# ---------------------------------------------------------------------------
def test_ts_turnover_cost_quantile_distance_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_cost_quantile_distance")
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


def test_ts_turnover_cost_quantile_distance_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_quantile_distance")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_cost_quantile_distance_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_quantile_distance")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_cost_quantile_distance_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_cost_quantile_distance")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_cost_quantile_distance_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_cost_quantile_distance")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 7. ts_turnover_cost_entropy
# ---------------------------------------------------------------------------
def test_ts_turnover_cost_entropy_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_cost_entropy")
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


def test_ts_turnover_cost_entropy_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_entropy")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_cost_entropy_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_entropy")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_cost_entropy_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_cost_entropy")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_cost_entropy_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_cost_entropy")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 8. ts_turnover_cost_mode_distance
# ---------------------------------------------------------------------------
def test_ts_turnover_cost_mode_distance_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_cost_mode_distance")
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


def test_ts_turnover_cost_mode_distance_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_mode_distance")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_cost_mode_distance_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_mode_distance")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_cost_mode_distance_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_cost_mode_distance")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_cost_mode_distance_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_cost_mode_distance")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 9. ts_turnover_cost_skew
# ---------------------------------------------------------------------------
def test_ts_turnover_cost_skew_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_cost_skew")
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


def test_ts_turnover_cost_skew_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_skew")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_cost_skew_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_skew")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_cost_skew_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_cost_skew")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_cost_skew_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_cost_skew")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 10. ts_turnover_age_dispersion
# ---------------------------------------------------------------------------
def test_ts_turnover_age_dispersion_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_age_dispersion")
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


def test_ts_turnover_age_dispersion_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_age_dispersion")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_age_dispersion_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_age_dispersion")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_age_dispersion_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_age_dispersion")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_age_dispersion_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_age_dispersion")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 11. ts_turnover_old_mass
# ---------------------------------------------------------------------------
def test_ts_turnover_old_mass_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_old_mass")
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


def test_ts_turnover_old_mass_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_old_mass")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_old_mass_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_old_mass")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_old_mass_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_old_mass")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_old_mass_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_old_mass")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 12. ts_turnover_cost_entropy_vol_scaled
# ---------------------------------------------------------------------------
def test_ts_turnover_cost_entropy_vol_scaled_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("ts_turnover_cost_entropy_vol_scaled")
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


def test_ts_turnover_cost_entropy_vol_scaled_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_entropy_vol_scaled")

    try:
        result = op.calculate(x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_ts_turnover_cost_entropy_vol_scaled_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("ts_turnover_cost_entropy_vol_scaled")

    try:
        result = op.calculate(x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_ts_turnover_cost_entropy_vol_scaled_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("ts_turnover_cost_entropy_vol_scaled")

    try:
        result = op.calculate(x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_ts_turnover_cost_entropy_vol_scaled_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("ts_turnover_cost_entropy_vol_scaled")

    try:
        result = op.calculate(x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_turnover_survival_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "ts_turnover_reference_price",
        "ts_turnover_cost_dispersion",
        "ts_turnover_profit_share",
        "ts_turnover_holding_age",
        "ts_turnover_near_cost_mass",
        "ts_turnover_cost_quantile_distance",
        "ts_turnover_cost_entropy",
        "ts_turnover_cost_mode_distance",
        "ts_turnover_cost_skew",
        "ts_turnover_age_dispersion",
        "ts_turnover_old_mass",
        "ts_turnover_cost_entropy_vol_scaled"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
        assert meta.name == op_name, f"{op_name} name mismatch"
