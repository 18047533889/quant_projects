# -*- coding: utf-8 -*-
"""Tests for group_spectrum operators.

Coverage of 7 operators:
1. group_feature_valid_member_count
2. group_feature_coverage_ratio
3. group_feature_mode_share
4. group_feature_effective_rank
5. group_feature_mode_localization
6. group_feature_spectral_gap
7. group_feature_second_mode_localization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _calculate(op, x):
    rng = np.random.default_rng(4199)
    f2 = pd.DataFrame(rng.normal(size=x.shape), index=x.index, columns=x.columns)
    f3 = pd.DataFrame(rng.normal(size=x.shape), index=x.index, columns=x.columns)
    group = pd.DataFrame("G", index=x.index, columns=x.columns)
    return op.calculate(x, f2, f3, group)


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op



# ---------------------------------------------------------------------------
# 1. group_feature_valid_member_count
# ---------------------------------------------------------------------------
def test_group_feature_valid_member_count_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_valid_member_count")
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


def test_group_feature_valid_member_count_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_valid_member_count")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_valid_member_count_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_valid_member_count")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_valid_member_count_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_valid_member_count")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_valid_member_count_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_valid_member_count")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 2. group_feature_coverage_ratio
# ---------------------------------------------------------------------------
def test_group_feature_coverage_ratio_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_coverage_ratio")
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


def test_group_feature_coverage_ratio_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_coverage_ratio")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_coverage_ratio_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_coverage_ratio")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_coverage_ratio_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_coverage_ratio")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_coverage_ratio_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_coverage_ratio")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 3. group_feature_mode_share
# ---------------------------------------------------------------------------
def test_group_feature_mode_share_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_mode_share")
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


def test_group_feature_mode_share_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_mode_share")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_mode_share_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_mode_share")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_mode_share_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_mode_share")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_mode_share_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_mode_share")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 4. group_feature_effective_rank
# ---------------------------------------------------------------------------
def test_group_feature_effective_rank_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_effective_rank")
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


def test_group_feature_effective_rank_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_effective_rank")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_effective_rank_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_effective_rank")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_effective_rank_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_effective_rank")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_effective_rank_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_effective_rank")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 5. group_feature_mode_localization
# ---------------------------------------------------------------------------
def test_group_feature_mode_localization_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_mode_localization")
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


def test_group_feature_mode_localization_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_mode_localization")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_mode_localization_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_mode_localization")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_mode_localization_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_mode_localization")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_mode_localization_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_mode_localization")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 6. group_feature_spectral_gap
# ---------------------------------------------------------------------------
def test_group_feature_spectral_gap_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_spectral_gap")
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


def test_group_feature_spectral_gap_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_spectral_gap")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_spectral_gap_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_spectral_gap")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_spectral_gap_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_spectral_gap")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_spectral_gap_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_spectral_gap")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")


# ---------------------------------------------------------------------------
# 7. group_feature_second_mode_localization
# ---------------------------------------------------------------------------
def test_group_feature_second_mode_localization_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]

    # Create sample data
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("group_feature_second_mode_localization")
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


def test_group_feature_second_mode_localization_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B", "C"]

    # Data with NaNs
    data = [[1.0, np.nan, 3.0]] * 10
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_second_mode_localization")

    try:
        result = _calculate(op, x)

        # Should not raise, should return DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} failed on NaN input: {e}")


def test_group_feature_second_mode_localization_handles_inf() -> None:
    """Infinity handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = ["A", "B"]

    # Data with infinities
    data = [[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]] * 3 + [[5.0, 6.0]]
    x = pd.DataFrame(data, index=idx, columns=cols)

    op = _op("group_feature_second_mode_localization")

    try:
        result = _calculate(op, x)

        # Should handle inf gracefully (typically return NaN)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"{op} failed on inf input: {e}")


def test_group_feature_second_mode_localization_empty_input() -> None:
    """Empty input test."""
    x = pd.DataFrame()

    op = _op("group_feature_second_mode_localization")

    try:
        result = _calculate(op, x)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    except Exception as e:
        # Some operators may validly raise on empty input
        pass


def test_group_feature_second_mode_localization_single_column() -> None:
    """Single column test."""
    idx = pd.date_range("2024-01-01", periods=20)
    x = pd.DataFrame({"A": range(20)}, index=idx)

    op = _op("group_feature_second_mode_localization")

    try:
        result = _calculate(op, x)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
    except Exception as e:
        pytest.fail(f"{op} failed on single column: {e}")



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_group_spectrum_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "group_feature_valid_member_count",
        "group_feature_coverage_ratio",
        "group_feature_mode_share",
        "group_feature_effective_rank",
        "group_feature_mode_localization",
        "group_feature_spectral_gap",
        "group_feature_second_mode_localization"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
        assert meta.name == op_name, f"{op_name} name mismatch"


@pytest.mark.parametrize("name", [
    "group_feature_valid_member_count", "group_feature_coverage_ratio",
    "group_feature_mode_share", "group_feature_effective_rank",
    "group_feature_mode_localization", "group_feature_spectral_gap",
    "group_feature_second_mode_localization",
])
def test_group_spectrum_real_four_panel_contract_and_prefix(name):
    from factor_engine.backend.operator_errors import OperatorParameterError
    rng = np.random.default_rng(4190)
    frames = [pd.DataFrame(rng.normal(size=(5, 32))) for _ in range(3)]
    group = pd.DataFrame("G", index=frames[0].index, columns=frames[0].columns)
    op = _op(name)
    out = op.calculate(*frames, group)
    assert np.isfinite(out.to_numpy()).any()
    pd.testing.assert_frame_equal(out.iloc[:3], op.calculate(*(f.iloc[:3] for f in frames), group.iloc[:3]))
    assert tuple(op.metadata.panel_params) == ("f1", "f2", "f3", "group")
    with pytest.raises(OperatorParameterError, match="missing"):
        op.calculate(frames[0])


def test_group_spectrum_independent_covariance_eigenvalue_oracle():
    rng = np.random.default_rng(4191)
    z = rng.normal(size=(32, 3))
    frames = [pd.DataFrame(z[:, i][None, :]) for i in range(3)]
    group = pd.DataFrame("G", index=frames[0].index, columns=frames[0].columns)
    centered = z - np.median(z, axis=0)
    normalized = centered / (1.4826 * np.median(abs(centered), axis=0))
    eigen = np.linalg.eigvalsh(normalized.T @ normalized)[::-1]
    proportions = eigen / eigen.sum()
    expected = {
        "group_feature_valid_member_count": 32.,
        "group_feature_coverage_ratio": 1.,
        "group_feature_mode_share": proportions[0],
        "group_feature_effective_rank": np.exp(-sum(proportions * np.log(proportions))),
        "group_feature_spectral_gap": proportions[0] - proportions[1],
    }
    for name, value in expected.items():
        actual = _op(name).calculate(*frames, group)
        np.testing.assert_allclose(actual.to_numpy(), value, rtol=1e-10, atol=1e-12)
