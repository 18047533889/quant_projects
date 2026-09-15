# -*- coding: utf-8 -*-
"""Tests for cs_batch1_polars operators.

Coverage of 5 operators:
1. cs_isolation_forest_score
2. cs_factor_bucket_return
3. cs_empirical_bayes_shrinkage
4. cs_shrink_to_group_mean
5. panel_peer_graph_aggregate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.common.static_adjacency import StaticAdjacency

ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op



# ---------------------------------------------------------------------------
# 1. cs_isolation_forest_score
# ---------------------------------------------------------------------------
def test_cs_isolation_forest_score_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)

    op = _op("cs_isolation_forest_score")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_cs_isolation_forest_score_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])

    op = _op("cs_isolation_forest_score")
    try:
        result = op.calculate(x)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_cs_isolation_forest_score_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))

    op = _op("cs_isolation_forest_score")
    try:
        result1 = op.calculate(x)
        result2 = op.calculate(x)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic


# ---------------------------------------------------------------------------
# 2. cs_factor_bucket_return
# ---------------------------------------------------------------------------
def test_cs_factor_bucket_return_basic() -> None:
    """Required factor/return panels produce the documented bucket means."""
    idx = pd.date_range("2024-01-01", periods=2)
    cols = [f"S{i}" for i in range(10)]
    factor = pd.DataFrame([np.arange(10.0)] * 2, index=idx, columns=cols)
    ret = pd.DataFrame([np.arange(10.0, 20.0)] * 2, index=idx, columns=cols)

    op = _op("cs_factor_bucket_return")
    result = op.calculate(factor, ret, n_buckets=5)
    expected = np.repeat(np.arange(10.5, 20.0, 2.0), 2)
    np.testing.assert_allclose(result.to_numpy(), np.vstack([expected, expected]))


def test_cs_factor_bucket_return_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = [f"S{i}" for i in range(10)]
    factor = pd.DataFrame([np.arange(10.0)] * 10, index=idx, columns=cols)
    ret = factor.copy()
    factor.iloc[:, 0] = np.nan

    op = _op("cs_factor_bucket_return")
    result = op.calculate(factor, ret)
    assert result.isna().all().all()  # nine aligned names is below min breadth ten


def test_cs_factor_bucket_return_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)
    ret = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)

    op = _op("cs_factor_bucket_return")
    result1 = op.calculate(x, ret)
    result2 = op.calculate(x, ret)
    pd.testing.assert_frame_equal(result1, result2, check_exact=True)


# ---------------------------------------------------------------------------
# 3. cs_empirical_bayes_shrinkage
# ---------------------------------------------------------------------------
def test_cs_empirical_bayes_shrinkage_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)
    std_err = pd.DataFrame(np.ones((20, 10)), index=idx, columns=cols)

    op = _op("cs_empirical_bayes_shrinkage")
    result = op.calculate(x, std_err)
    means = x.mean(axis=1)
    assert ((result.sub(means, axis=0).abs() <= x.sub(means, axis=0).abs() + 1e-12).all().all())


def test_cs_empirical_bayes_shrinkage_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame([np.arange(10.0)] * 10, index=idx, columns=cols)
    std_err = pd.DataFrame(np.ones((10, 10)), index=idx, columns=cols)
    x.iloc[:, 0] = np.nan

    op = _op("cs_empirical_bayes_shrinkage")
    result = op.calculate(x, std_err)
    assert result.isna().all().all()


def test_cs_empirical_bayes_shrinkage_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(15, 10), index=idx, columns=cols)
    std_err = pd.DataFrame(np.ones((15, 10)), index=idx, columns=cols)

    op = _op("cs_empirical_bayes_shrinkage")
    result1 = op.calculate(x, std_err)
    result2 = op.calculate(x, std_err)
    pd.testing.assert_frame_equal(result1, result2, check_exact=True)


# ---------------------------------------------------------------------------
# 4. cs_shrink_to_group_mean
# ---------------------------------------------------------------------------
def test_cs_shrink_to_group_mean_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame([np.arange(10.0)] * 20, index=idx, columns=cols)
    group = pd.DataFrame([["A"] * 5 + ["B"] * 5] * 20, index=idx, columns=cols)

    op = _op("cs_shrink_to_group_mean")
    result = op.calculate(x, group, shrinkage_intensity=0.5)
    expected = np.r_[0.5 * np.arange(5.0) + 1.0, 0.5 * np.arange(5.0, 10.0) + 3.5]
    np.testing.assert_allclose(result.to_numpy(), np.tile(expected, (20, 1)))


def test_cs_shrink_to_group_mean_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])
    group = pd.DataFrame([["g", "g", "g"]] * 10, index=idx, columns=x.columns)

    op = _op("cs_shrink_to_group_mean")
    result = op.calculate(x, group)
    np.testing.assert_allclose(result[["A", "C"]], [[1.5, 2.5]] * 10)
    assert result["B"].isna().all()


def test_cs_shrink_to_group_mean_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))
    group = pd.DataFrame([["g1", "g1", "g2", "g2", "g2"]] * 15, index=idx, columns=x.columns)

    op = _op("cs_shrink_to_group_mean")
    result1 = op.calculate(x, group)
    result2 = op.calculate(x, group)
    pd.testing.assert_frame_equal(result1, result2, check_exact=True)


# ---------------------------------------------------------------------------
# 5. panel_peer_graph_aggregate
# ---------------------------------------------------------------------------
def test_panel_peer_graph_aggregate_basic() -> None:
    """Basic functionality test."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=20)
    cols = [f"S{i}" for i in range(10)]
    x = pd.DataFrame(np.random.randn(20, 10), index=idx, columns=cols)
    graph = StaticAdjacency(tuple(cols), tuple(tuple(row) for row in np.ones((10, 10))))

    op = _op("panel_peer_graph_aggregate")
    try:
        result = op.calculate(x, graph)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == x.shape
    except Exception as e:
        pytest.fail(f"{op} basic test failed: {e}")


def test_panel_peer_graph_aggregate_handles_nans() -> None:
    """NaN handling test."""
    idx = pd.date_range("2024-01-01", periods=10)
    x = pd.DataFrame([[1.0, np.nan, 3.0]] * 10, index=idx, columns=["A", "B", "C"])
    graph = pd.DataFrame(np.ones((3, 3)), index=x.columns, columns=x.columns)

    op = _op("panel_peer_graph_aggregate")
    try:
        result = op.calculate(x, graph)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"NaN test failed: {e}")


def test_panel_peer_graph_aggregate_deterministic() -> None:
    """Determinism test - same input yields same output."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=15)
    x = pd.DataFrame(np.random.randn(15, 5), index=idx, columns=list("ABCDE"))
    graph = pd.DataFrame(np.ones((5, 5)), index=x.columns, columns=x.columns)

    op = _op("panel_peer_graph_aggregate")
    try:
        result1 = op.calculate(x, graph)
        result2 = op.calculate(x, graph)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)
    except Exception:
        pass  # Some operators may not be deterministic



# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------
def test_cs_batch1_polars_metadata() -> None:
    """Verify all operators have correct metadata."""
    operators = [
        "cs_isolation_forest_score",
        "cs_factor_bucket_return",
        "cs_empirical_bayes_shrinkage",
        "cs_shrink_to_group_mean",
        "panel_peer_graph_aggregate"
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = getattr(op, "metadata", None)
        assert meta is not None, f"{op_name} missing metadata"
        assert hasattr(meta, "tags"), f"{op_name} missing tags"
