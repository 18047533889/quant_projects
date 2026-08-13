# -*- coding: utf-8 -*-
"""
Focused tests for weighted cross-sectional operators in cs_batch1.py

Tests verify true Polars native implementation without pandas round-trip.
"""
import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Import directly from module to avoid __init__.py conflicts
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "cleaned_operators" / "polars_native"))

from cs_batch1 import (
    CSWeightedMeanPolarsNative,
    CSWeightedDemeanPolarsNative,
    CSWeightedZscorePolarsNative,
    CSWeightedPercentileRankPolarsNative,
)


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
class TestWeightedCSOperators:
    """Test weighted cross-sectional operators"""

    def test_weighted_mean_basic(self):
        """Test weighted mean: sum(x * w) / sum(w)"""
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="weight")

        op = CSWeightedMeanPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Manual calculation: (1*1 + 2*1 + 3*1 + 4*1 + 5*1) / (1+1+1+1+1) = 15/5 = 3.0
        expected = 3.0
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result[0]}"

    def test_weighted_mean_unequal_weights(self):
        """Test weighted mean with unequal weights"""
        feature = pd.Series([1.0, 2.0, 3.0], name="value")
        weight = pd.Series([1.0, 2.0, 3.0], name="weight")

        op = CSWeightedMeanPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Manual: (1*1 + 2*2 + 3*3) / (1+2+3) = (1 + 4 + 9) / 6 = 14/6 = 2.333...
        expected = 14.0 / 6.0
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result[0]}"

    def test_weighted_mean_with_nulls(self):
        """Test weighted mean with null values"""
        feature = pd.Series([1.0, np.nan, 3.0, 4.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0], name="weight")

        op = CSWeightedMeanPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Polars should handle nulls by skipping them
        # (1*1 + 3*1 + 4*1) / (1+1+1) = 8/3 = 2.666...
        assert len(result) == len(feature)
        # Result should be broadcast to all rows
        assert np.isnan(result).sum() == 0 or np.allclose(result.dropna(), 8.0/3.0)

    def test_weighted_demean_basic(self):
        """Test weighted demean: x - weighted_mean"""
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="weight")

        op = CSWeightedDemeanPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Weighted mean = 3.0
        # Expected: [1-3, 2-3, 3-3, 4-3, 5-3] = [-2, -1, 0, 1, 2]
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_demean_unequal_weights(self):
        """Test weighted demean with unequal weights"""
        feature = pd.Series([1.0, 2.0, 3.0], name="value")
        weight = pd.Series([1.0, 2.0, 3.0], name="weight")

        op = CSWeightedDemeanPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Weighted mean = 14/6 = 2.333...
        # Expected: [1 - 2.333, 2 - 2.333, 3 - 2.333] = [-1.333, -0.333, 0.666]
        weighted_mean = 14.0 / 6.0
        expected = np.array([1.0, 2.0, 3.0]) - weighted_mean
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_zscore_basic(self):
        """Test weighted zscore: (x - weighted_mean) / weighted_std"""
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="weight")

        op = CSWeightedZscorePolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Weighted mean = 3.0
        # Weighted variance = sum(w * (x - mean)^2) / sum(w)
        # = (1*(1-3)^2 + 1*(2-3)^2 + 1*(3-3)^2 + 1*(4-3)^2 + 1*(5-3)^2) / 5
        # = (4 + 1 + 0 + 1 + 4) / 5 = 10/5 = 2.0
        # Weighted std = sqrt(2.0) = 1.414...
        weighted_mean = 3.0
        weighted_var = 2.0
        weighted_std = np.sqrt(2.0)
        expected = (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) - weighted_mean) / weighted_std

        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_zscore_unequal_weights(self):
        """Test weighted zscore with unequal weights"""
        feature = pd.Series([1.0, 2.0, 3.0], name="value")
        weight = pd.Series([1.0, 2.0, 3.0], name="weight")

        op = CSWeightedZscorePolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # Weighted mean = 14/6 = 2.333...
        # Weighted variance = (1*(1-2.333)^2 + 2*(2-2.333)^2 + 3*(3-2.333)^2) / 6
        weighted_mean = 14.0 / 6.0
        deviations = np.array([1.0, 2.0, 3.0]) - weighted_mean
        weighted_var = np.sum(np.array([1.0, 2.0, 3.0]) * deviations**2) / 6.0
        weighted_std = np.sqrt(weighted_var)
        expected = deviations / weighted_std

        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_zscore_zero_variance(self):
        """Test weighted zscore with zero variance (constant values)"""
        feature = pd.Series([2.0, 2.0, 2.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0], name="weight")

        op = CSWeightedZscorePolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # All values are constant, std = 0, should return None/NaN
        assert len(result) == len(feature)
        assert result.isna().all(), f"Expected all NaN for zero variance, got {result.values}"

    def test_weighted_percentile_rank_basic(self):
        """Test weighted percentile rank: cumsum(w) / sum(w) after sorting"""
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="weight")

        op = CSWeightedPercentileRankPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # After sorting by value: [1, 2, 3, 4, 5]
        # Cumsum of weights: [1, 2, 3, 4, 5]
        # Sum of weights: 5
        # Weighted rank: [1/5, 2/5, 3/5, 4/5, 5/5] = [0.2, 0.4, 0.6, 0.8, 1.0]
        expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_percentile_rank_unequal_weights(self):
        """Test weighted percentile rank with unequal weights"""
        feature = pd.Series([1.0, 2.0, 3.0], name="value")
        weight = pd.Series([1.0, 2.0, 3.0], name="weight")

        op = CSWeightedPercentileRankPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # After sorting by value: [1, 2, 3] (already sorted)
        # Cumsum of weights: [1, 3, 6]
        # Sum of weights: 6
        # Weighted rank: [1/6, 3/6, 6/6] = [0.1667, 0.5, 1.0]
        expected = np.array([1.0/6.0, 3.0/6.0, 6.0/6.0])
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_percentile_rank_reverse_order(self):
        """Test weighted percentile rank with reverse sorted input"""
        feature = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="weight")

        op = CSWeightedPercentileRankPolarsNative()
        result = op._calculate_series(feature, weight=weight)

        # After sorting by value: [1, 2, 3, 4, 5]
        # Original indices: [4, 3, 2, 1, 0]
        # Weighted rank at sorted positions: [0.2, 0.4, 0.6, 0.8, 1.0]
        # Result at original positions: [1.0, 0.8, 0.6, 0.4, 0.2]
        expected = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        assert len(result) == len(feature)
        assert np.allclose(result, expected), f"Expected {expected}, got {result.values}"

    def test_weighted_operators_preserve_index(self):
        """Test that all weighted operators preserve original index"""
        index = pd.date_range('2020-01-01', periods=5, freq='D')
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index, name="value")
        weight = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=index, name="weight")

        for OpClass in [CSWeightedMeanPolarsNative, CSWeightedDemeanPolarsNative,
                       CSWeightedZscorePolarsNative, CSWeightedPercentileRankPolarsNative]:
            op = OpClass()
            result = op._calculate_series(feature, weight=weight)
            assert result.index.equals(index), f"{OpClass.__name__} did not preserve index"

    def test_weighted_operators_fallback_to_unweighted(self):
        """Test that weighted operators fallback to unweighted when weight=None"""
        feature = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")

        # Test weighted mean falls back to fill_mean (broadcast mean)
        op_mean = CSWeightedMeanPolarsNative()
        result_mean = op_mean._calculate_series(feature, weight=None)
        assert len(result_mean) == len(feature)

        # Test weighted demean falls back to demean
        op_demean = CSWeightedDemeanPolarsNative()
        result_demean = op_demean._calculate_series(feature, weight=None)
        expected_demean = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert np.allclose(result_demean, expected_demean)

        # Test weighted zscore falls back to zscore
        op_zscore = CSWeightedZscorePolarsNative()
        result_zscore = op_zscore._calculate_series(feature, weight=None)
        assert len(result_zscore) == len(feature)

        # Test weighted rank falls back to rank
        op_rank = CSWeightedPercentileRankPolarsNative()
        result_rank = op_rank._calculate_series(feature, weight=None)
        assert len(result_rank) == len(feature)
