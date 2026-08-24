# -*- coding: utf-8 -*-
"""Tests for cs_batch1 operators.

Comprehensive coverage of 5 cross-sectional TRUE_GAP operators:
1. cs_isolation_forest_score - anomaly detection
2. cs_factor_bucket_return - factor bucketing analysis
3. cs_empirical_bayes_shrinkage - Bayesian shrinkage
4. cs_shrink_to_group_mean - group-wise shrinkage
5. panel_peer_graph_aggregate - graph-based aggregation
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
# 1. cs_isolation_forest_score
# ---------------------------------------------------------------------------
def test_cs_isolation_forest_score_normal_distribution() -> None:
    """Normal distribution should have low anomaly scores."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=1)
    n = 50
    cols = [f"C{i}" for i in range(n)]

    # Normal distribution N(0, 1)
    x = pd.DataFrame([np.random.randn(n)], index=idx, columns=cols)

    out = _op("cs_isolation_forest_score").calculate(x, n_trees=100, contamination=0.1, random_seed=42)

    result = out.to_numpy()[0]
    # Most scores should be low (< 0.5) for normal data
    assert np.isfinite(result).all()
    assert np.mean(result) < 0.5
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_cs_isolation_forest_score_with_outliers() -> None:
    """Outliers should have high anomaly scores."""
    np.random.seed(123)
    idx = pd.date_range("2024-01-01", periods=1)
    n = 50
    cols = [f"C{i}" for i in range(n)]

    # Normal data with a few outliers
    x_vals = np.random.randn(n)
    x_vals[0] = 10.0  # Strong outlier
    x_vals[1] = -8.0  # Strong outlier
    x_vals[2] = 7.0   # Moderate outlier

    x = pd.DataFrame([x_vals], index=idx, columns=cols)

    out = _op("cs_isolation_forest_score").calculate(x, n_trees=100, contamination=0.1, random_seed=42)

    result = out.to_numpy()[0]
    # Outliers should have higher scores
    assert result[0] > 0.6  # Strong outlier
    assert result[1] > 0.6  # Strong outlier
    # Normal points should have lower scores
    assert np.mean(result[3:]) < 0.5


def test_cs_isolation_forest_score_insufficient_samples() -> None:
    """Insufficient samples (<10) should fail-closed."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDE")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], index=idx, columns=cols)

    out = _op("cs_isolation_forest_score").calculate(x)

    # < 10 samples -> fail-closed
    assert out.isna().all().all()


def test_cs_isolation_forest_score_missing_values() -> None:
    """Handle missing values correctly."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=1)
    n = 30
    cols = [f"C{i}" for i in range(n)]

    x_vals = np.random.randn(n)
    x_vals[0] = np.nan
    x_vals[5] = np.nan

    x = pd.DataFrame([x_vals], index=idx, columns=cols)

    out = _op("cs_isolation_forest_score").calculate(x)

    result = out.to_numpy()[0]
    # Missing values stay NaN
    assert np.isnan(result[0])
    assert np.isnan(result[5])
    # Valid values get scores
    assert np.isfinite(result[1:5]).all()


# ---------------------------------------------------------------------------
# 2. cs_factor_bucket_return
# ---------------------------------------------------------------------------
def test_cs_factor_bucket_return_basic() -> None:
    """Basic bucketing functionality."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(15)]

    # Factor values: 1, 2, 3, ..., 15
    factor = pd.DataFrame([list(range(1, 16))], index=idx, columns=cols)
    # Returns: same as factor for simplicity
    ret = pd.DataFrame([list(range(1, 16))], index=idx, columns=cols)

    out = _op("cs_factor_bucket_return").calculate(factor, ret, n_buckets=3, ascending=True)

    result = out.to_numpy()[0]
    # Bucket 1: factor [1-5] -> mean return = 3
    # Bucket 2: factor [6-10] -> mean return = 8
    # Bucket 3: factor [11-15] -> mean return = 13
    assert np.isfinite(result).all()
    # Check bucket means (approximate)
    np.testing.assert_allclose(result[0:5], 3.0, atol=0.5)
    np.testing.assert_allclose(result[5:10], 8.0, atol=0.5)
    np.testing.assert_allclose(result[10:15], 13.0, atol=0.5)


def test_cs_factor_bucket_return_descending() -> None:
    """Descending bucketing."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(15)]

    factor = pd.DataFrame([list(range(1, 16))], index=idx, columns=cols)
    ret = pd.DataFrame([list(range(1, 16))], index=idx, columns=cols)

    out = _op("cs_factor_bucket_return").calculate(factor, ret, n_buckets=3, ascending=False)

    result = out.to_numpy()[0]
    # Descending: high factor values in bucket 1
    # Bucket 1: factor [11-15] -> mean return = 13
    # Bucket 3: factor [1-5] -> mean return = 3
    np.testing.assert_allclose(result[10:15], 13.0, atol=0.5)
    np.testing.assert_allclose(result[0:5], 3.0, atol=0.5)


def test_cs_factor_bucket_return_missing_values() -> None:
    """Missing values in factor or return."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(15)]

    factor_vals = list(range(1, 16))
    factor_vals[0] = np.nan
    ret_vals = list(range(1, 16))
    ret_vals[5] = np.nan

    factor = pd.DataFrame([factor_vals], index=idx, columns=cols)
    ret = pd.DataFrame([ret_vals], index=idx, columns=cols)

    out = _op("cs_factor_bucket_return").calculate(factor, ret, n_buckets=3)

    result = out.to_numpy()[0]
    # Missing values stay NaN
    assert np.isnan(result[0])
    assert np.isnan(result[5])
    # Others are finite
    assert np.isfinite(result[1:5]).all()


def test_cs_factor_bucket_return_insufficient_samples() -> None:
    """Insufficient samples (<10) should fail-closed."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    factor = pd.DataFrame([[1, 2, 3, 4, 5, 6]], index=idx, columns=cols)
    ret = pd.DataFrame([[1, 2, 3, 4, 5, 6]], index=idx, columns=cols)

    out = _op("cs_factor_bucket_return").calculate(factor, ret, n_buckets=3)

    # < 10 samples -> fail-closed
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# 3. cs_empirical_bayes_shrinkage
# ---------------------------------------------------------------------------
def test_cs_empirical_bayes_shrinkage_basic() -> None:
    """Basic shrinkage functionality."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(20)]

    # Estimates centered around 10, with one outlier
    estimate_vals = [10.0] * 19 + [50.0]
    # Std errors: most have small error, outlier has large error
    std_err_vals = [1.0] * 19 + [10.0]

    estimate = pd.DataFrame([estimate_vals], index=idx, columns=cols)
    std_err = pd.DataFrame([std_err_vals], index=idx, columns=cols)

    out = _op("cs_empirical_bayes_shrinkage").calculate(estimate, std_err, shrinkage_factor=1.0)

    result = out.to_numpy()[0]
    # Mean estimate ≈ 12 (19*10 + 50) / 20
    cs_mean = np.mean(estimate_vals)

    # Regular stocks (low std_err) should stay near original
    assert np.abs(result[0] - estimate_vals[0]) < 2.0

    # Outlier with high std_err should be pulled toward mean
    assert result[19] < estimate_vals[19]  # 50 -> closer to mean
    assert result[19] > cs_mean  # But still above mean


def test_cs_empirical_bayes_shrinkage_zero_variance() -> None:
    """Zero variance -> all shrink to mean."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(20)]

    # All estimates identical -> variance = 0
    estimate = pd.DataFrame([[10.0] * 20], index=idx, columns=cols)
    std_err = pd.DataFrame([[1.0] * 20], index=idx, columns=cols)

    out = _op("cs_empirical_bayes_shrinkage").calculate(estimate, std_err, shrinkage_factor=1.0)

    result = out.to_numpy()[0]
    # All shrunk to mean (which is 10)
    np.testing.assert_allclose(result, 10.0, atol=1e-6)


def test_cs_empirical_bayes_shrinkage_missing_values() -> None:
    """Missing values handled correctly."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(15)]

    estimate_vals = [float(i) for i in range(1, 16)]
    estimate_vals[0] = np.nan
    std_err_vals = [1.0] * 15
    std_err_vals[5] = np.nan

    estimate = pd.DataFrame([estimate_vals], index=idx, columns=cols)
    std_err = pd.DataFrame([std_err_vals], index=idx, columns=cols)

    out = _op("cs_empirical_bayes_shrinkage").calculate(estimate, std_err)

    result = out.to_numpy()[0]
    assert np.isnan(result[0])
    assert np.isnan(result[5])
    assert np.isfinite(result[1:5]).all()


# ---------------------------------------------------------------------------
# 4. cs_shrink_to_group_mean
# ---------------------------------------------------------------------------
def test_cs_shrink_to_group_mean_basic() -> None:
    """Basic group shrinkage."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    # Group A: [1, 2, 3] -> mean = 2
    # Group B: [4, 5, 6] -> mean = 5
    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], index=idx, columns=cols)
    group = pd.DataFrame([["A", "A", "A", "B", "B", "B"]], index=idx, columns=cols)

    out = _op("cs_shrink_to_group_mean").calculate(x, group, shrinkage_intensity=0.5)

    result = out.to_numpy()[0]
    # Group A: shrunk = x * 0.5 + 2 * 0.5
    # Stock A (x=1): 1*0.5 + 2*0.5 = 1.5
    # Stock B (x=2): 2*0.5 + 2*0.5 = 2.0
    # Stock C (x=3): 3*0.5 + 2*0.5 = 2.5
    np.testing.assert_allclose(result[:3], [1.5, 2.0, 2.5], atol=1e-6)

    # Group B: shrunk = x * 0.5 + 5 * 0.5
    # Stock D (x=4): 4*0.5 + 5*0.5 = 4.5
    # Stock E (x=5): 5*0.5 + 5*0.5 = 5.0
    # Stock F (x=6): 6*0.5 + 5*0.5 = 5.5
    np.testing.assert_allclose(result[3:], [4.5, 5.0, 5.5], atol=1e-6)


def test_cs_shrink_to_group_mean_full_shrinkage() -> None:
    """Full shrinkage (intensity=1) -> all to group mean."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], index=idx, columns=cols)
    group = pd.DataFrame([["A", "A", "A", "B", "B", "B"]], index=idx, columns=cols)

    out = _op("cs_shrink_to_group_mean").calculate(x, group, shrinkage_intensity=1.0)

    result = out.to_numpy()[0]
    # Group A mean = 2, all shrunk to 2
    np.testing.assert_allclose(result[:3], 2.0, atol=1e-6)
    # Group B mean = 20, all shrunk to 20
    np.testing.assert_allclose(result[3:], 20.0, atol=1e-6)


def test_cs_shrink_to_group_mean_no_shrinkage() -> None:
    """No shrinkage (intensity=0) -> keep original."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    x_vals = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    x = pd.DataFrame([x_vals], index=idx, columns=cols)
    group = pd.DataFrame([["A", "A", "A", "B", "B", "B"]], index=idx, columns=cols)

    out = _op("cs_shrink_to_group_mean").calculate(x, group, shrinkage_intensity=0.0)

    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, x_vals, atol=1e-6)


def test_cs_shrink_to_group_mean_invalid_group() -> None:
    """Invalid group labels (NaN/inf/None/empty) -> NaN."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], index=idx, columns=cols)
    # Invalid groups: NaN, inf, None (as empty string), valid groups
    group = pd.DataFrame([[np.nan, np.inf, "", "A", "A", "A"]], index=idx, columns=cols)

    out = _op("cs_shrink_to_group_mean").calculate(x, group, shrinkage_intensity=0.5)

    result = out.to_numpy()[0]
    # Invalid groups -> NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    # Valid group A -> shrunk
    assert np.isfinite(result[3:]).all()


# ---------------------------------------------------------------------------
# 5. panel_peer_graph_aggregate
# ---------------------------------------------------------------------------
def test_panel_peer_graph_aggregate_mean() -> None:
    """Simple mean aggregation."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    # Uniform similarity -> all neighbors equally weighted
    similarity = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=idx, columns=cols)

    out = _op("panel_peer_graph_aggregate").calculate(x, similarity, method="mean", threshold=0.0)

    result = out.to_numpy()[0]
    # Mean of all = (1+2+3+4)/4 = 2.5
    expected_mean = 2.5
    np.testing.assert_allclose(result, expected_mean, atol=1e-6)


def test_panel_peer_graph_aggregate_weighted_mean() -> None:
    """Weighted mean aggregation."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    # Different similarities
    similarity = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], index=idx, columns=cols)

    out = _op("panel_peer_graph_aggregate").calculate(x, similarity, method="weighted_mean", threshold=0.0)

    result = out.to_numpy()[0]
    # Weighted mean = (1*0.1 + 2*0.2 + 3*0.3 + 4*0.4) / (0.1+0.2+0.3+0.4) = 3.0
    expected = (1*0.1 + 2*0.2 + 3*0.3 + 4*0.4) / (0.1+0.2+0.3+0.4)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_panel_peer_graph_aggregate_sum() -> None:
    """Sum aggregation."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    similarity = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=idx, columns=cols)

    out = _op("panel_peer_graph_aggregate").calculate(x, similarity, method="sum", threshold=0.0)

    result = out.to_numpy()[0]
    # Sum = 1+2+3+4 = 10
    expected_sum = 10.0
    np.testing.assert_allclose(result, expected_sum, atol=1e-6)


def test_panel_peer_graph_aggregate_threshold() -> None:
    """Threshold filters low similarity."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], index=idx, columns=cols)
    # Only last 3 stocks have high similarity
    similarity = pd.DataFrame([[0.1, 0.2, 0.3, 0.8, 0.9, 1.0]], index=idx, columns=cols)

    out = _op("panel_peer_graph_aggregate").calculate(x, similarity, method="mean", threshold=0.5)

    result = out.to_numpy()[0]
    # Only stocks with similarity > 0.5 are included: [4, 5, 6]
    # Mean = (4+5+6)/3 = 5.0
    # First 3 should be NaN (below threshold)
    assert np.isnan(result[:3]).all()
    # Last 3 should be mean of themselves
    np.testing.assert_allclose(result[3:], 5.0, atol=1e-6)


def test_panel_peer_graph_aggregate_missing_values() -> None:
    """Missing values in x or similarity."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")

    x_vals = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
    sim_vals = [1.0, 1.0, 1.0, np.nan, 1.0, 1.0]

    x = pd.DataFrame([x_vals], index=idx, columns=cols)
    similarity = pd.DataFrame([sim_vals], index=idx, columns=cols)

    out = _op("panel_peer_graph_aggregate").calculate(x, similarity, method="mean")

    result = out.to_numpy()[0]
    # Valid neighbors: stocks 0, 1, 4, 5 (x and sim both finite)
    # Mean = (1+2+5+6)/4 = 3.5
    assert np.isfinite(result[0])
    assert np.isfinite(result[1])


def test_panel_peer_graph_aggregate_invalid_method() -> None:
    """Invalid method -> fail-closed."""
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCD")

    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=cols)
    similarity = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=idx, columns=cols)

    out = _op("panel_peer_graph_aggregate").calculate(x, similarity, method="invalid")

    # Invalid method -> all NaN
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# Unit consistency checks
# ---------------------------------------------------------------------------
def test_cs_batch1_metadata_units() -> None:
    """Verify metadata declares correct units."""
    # cs_isolation_forest_score: dimensionless
    op1 = _op("cs_isolation_forest_score")
    meta1 = getattr(op1, "metadata", None)
    assert meta1 is not None
    assert any("dimensionless" in str(tag) for tag in meta1.tags)

    # cs_factor_bucket_return: same_as:ret
    op2 = _op("cs_factor_bucket_return")
    meta2 = getattr(op2, "metadata", None)
    assert meta2 is not None
    assert any("same_as:ret" in str(tag) for tag in meta2.tags)

    # cs_empirical_bayes_shrinkage: same_as:estimate
    op3 = _op("cs_empirical_bayes_shrinkage")
    meta3 = getattr(op3, "metadata", None)
    assert meta3 is not None
    assert any("same_as:estimate" in str(tag) for tag in meta3.tags)

    # cs_shrink_to_group_mean: same_as:x
    op4 = _op("cs_shrink_to_group_mean")
    meta4 = getattr(op4, "metadata", None)
    assert meta4 is not None
    assert any("same_as:x" in str(tag) for tag in meta4.tags)

    # panel_peer_graph_aggregate: same_as:x
    op5 = _op("panel_peer_graph_aggregate")
    meta5 = getattr(op5, "metadata", None)
    assert meta5 is not None
    assert any("same_as:x" in str(tag) for tag in meta5.tags)
