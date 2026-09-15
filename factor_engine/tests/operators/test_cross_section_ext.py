# -*- coding: utf-8 -*-
"""Tests for cross_section_ext operators.

Coverage of 5 operators:
1. cs_knn_local_moran
2. cs_isotonic_residual
3. cs_isotonic_residual_lagged_direction
4. group_tail_coexceedance_density
5. group_corr_mst_length
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
# 1. cs_knn_local_moran
# ---------------------------------------------------------------------------
def test_cs_knn_local_moran_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("cs_knn_local_moran")
    result = op.calculate(x, x, x, x, k=3)
    assert result.shape == x.shape
    assert np.isfinite(result.to_numpy()).all()


def test_cs_knn_local_moran_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("cs_knn_local_moran")
    result = op.calculate(x, x, x, x, k=1)
    assert result["B"].isna().all()
    assert np.isfinite(result[["A", "C"]].to_numpy()).all()


def test_cs_knn_local_moran_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("cs_knn_local_moran")
    result1 = op.calculate(x, x, x, x, k=2)
    result2 = op.calculate(x, x, x, x, k=2)
    pd.testing.assert_frame_equal(result1, result2, check_exact=True)


# ---------------------------------------------------------------------------
# 2. cs_isotonic_residual
# ---------------------------------------------------------------------------
def test_cs_isotonic_residual_basic() -> None:
    """A strictly increasing affine relation has exactly zero residual."""
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame([np.arange(10.0)], columns=cols)
    y = 2.0 * x + 3.0
    result = _op("cs_isotonic_residual").calculate(y, x)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_cs_isotonic_residual_handles_nans() -> None:
    """NaN handling test."""
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame([np.arange(10.0)], columns=cols)
    y = x.copy()
    y.iloc[0, 0] = np.nan
    result = _op("cs_isotonic_residual").calculate(y, x)
    assert result.isna().all().all()  # nine aligned names < breadth floor ten


def test_cs_isotonic_residual_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)
    y = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)
    op = _op("cs_isotonic_residual")
    pd.testing.assert_frame_equal(op.calculate(y, x), op.calculate(y, x), check_exact=True)


# ---------------------------------------------------------------------------
# 3. cs_isotonic_residual_lagged_direction
# ---------------------------------------------------------------------------
def test_cs_isotonic_residual_lagged_direction_basic() -> None:
    """Prior increasing direction forces today's decreasing fit to one level."""
    cols = [f"S{i}" for i in range(10)]
    xrow = np.arange(1.0, 11.0)
    x = pd.DataFrame([xrow, xrow], columns=cols)
    y = pd.DataFrame([xrow, xrow[::-1]], columns=cols)
    result = _op("cs_isotonic_residual_lagged_direction").calculate(y, x, lookback=1)
    assert result.iloc[0].isna().all()
    np.testing.assert_allclose(result.iloc[1], xrow[::-1] - 5.5, atol=1e-12)


def test_cs_isotonic_residual_lagged_direction_handles_nans() -> None:
    """NaN handling test."""
    cols = [f"S{i}" for i in range(10)]
    row = np.arange(10.0)
    x = pd.DataFrame([row, row], columns=cols)
    y = x.copy()
    y.iloc[0, 0] = np.nan
    result = _op("cs_isotonic_residual_lagged_direction").calculate(y, x, lookback=1)
    assert result.isna().all().all()  # prior aligned breadth is only nine


def test_cs_isotonic_residual_lagged_direction_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)
    y = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)
    op = _op("cs_isotonic_residual_lagged_direction")
    pd.testing.assert_frame_equal(
        op.calculate(y, x, lookback=3), op.calculate(y, x, lookback=3), check_exact=True
    )


# ---------------------------------------------------------------------------
# 4. group_tail_coexceedance_density
# ---------------------------------------------------------------------------
def test_group_tail_coexceedance_density_basic() -> None:
    """Three identical tail-event paths have zero excess over empirical p_i*p_j when p=1."""
    vals = np.array([1, 2, 3, 10, 20, 30], dtype=float)
    x = pd.DataFrame(np.tile(vals[:, None], (1, 3)), columns=list("ABC"))
    group = pd.DataFrame([["g"] * 3] * len(x), columns=x.columns)
    result = _op("group_tail_coexceedance_density").calculate(
        x, group, window=6, quantile=0.9, side="upper"
    )
    assert result.iloc[:5].isna().all().all()
    np.testing.assert_allclose(result.iloc[5], 0.0, atol=1e-12)


def test_group_tail_coexceedance_density_handles_nans() -> None:
    """NaN handling test."""
    vals = np.array([1, 2, 3, 10, 20, 30], dtype=float)
    x = pd.DataFrame({"A": vals, "B": vals, "C": np.nan})
    group = pd.DataFrame([["g"] * 3] * len(x), columns=x.columns)
    result = _op("group_tail_coexceedance_density").calculate(x, group, window=6)
    assert result.isna().all().all()  # one valid pair of three fails 50% coverage


def test_group_tail_coexceedance_density_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))
    group = pd.DataFrame([["g"] * 5] * 15, index=idx, columns=x.columns)
    op = _op("group_tail_coexceedance_density")
    pd.testing.assert_frame_equal(
        op.calculate(x, group, window=10), op.calculate(x, group, window=10), check_exact=True
    )


# ---------------------------------------------------------------------------
# 5. group_corr_mst_length
# ---------------------------------------------------------------------------
def test_group_corr_mst_length_basic() -> None:
    """rho(A,B)=1 and rho(A,C)=-1 gives MST mean (0+2)/2 = 1."""
    trend = np.arange(20.0)
    x = pd.DataFrame({"A": trend, "B": trend, "C": -trend})
    group = pd.DataFrame([["g"] * 3] * 20, columns=x.columns)
    result = _op("group_corr_mst_length").calculate(x, group, window=20)
    assert result.iloc[:19].isna().all().all()
    np.testing.assert_allclose(result.iloc[19], 1.0, atol=1e-12)


def test_group_corr_mst_length_handles_nans() -> None:
    """NaN handling test."""
    trend = np.arange(20.0)
    x = pd.DataFrame({"A": trend, "B": trend, "C": -trend})
    x.loc[0, "C"] = np.nan
    group = pd.DataFrame([["g"] * 3] * 20, columns=x.columns)
    result = _op("group_corr_mst_length").calculate(x, group, window=20)
    assert result.isna().all().all()  # C has only 19 aligned rows; graph disconnects


def test_group_corr_mst_length_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=25)
    x = pd.DataFrame(np.random.randn(25, 5), index=idx, columns=list("ABCDE"))
    group = pd.DataFrame([["g"] * 5] * 25, index=idx, columns=x.columns)
    op = _op("group_corr_mst_length")
    pd.testing.assert_frame_equal(
        op.calculate(x, group, window=20), op.calculate(x, group, window=20), check_exact=True
    )



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_cross_section_ext_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "cs_knn_local_moran",
        "cs_isotonic_residual",
        "cs_isotonic_residual_lagged_direction",
        "group_tail_coexceedance_density",
        "group_corr_mst_length"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
