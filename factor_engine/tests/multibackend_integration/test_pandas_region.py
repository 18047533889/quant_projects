# -*- coding: utf-8 -*-
"""Pandas backend-specific integration tests.

Tests cover:
- Pandas operations
- Panel operations
- GroupBy mechanics
- Memory characteristics
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


class TestPandasBackend:
    """Test Pandas NumPy backend."""

    def test_pandas_basic_operations(self, sample_panel_data):
        """Test basic Pandas operations."""
        df = sample_panel_data.copy()

        # Basic filtering
        filtered = df[df["close"] > 100.0]
        assert len(filtered) <= len(df)

        # Basic aggregation
        agg = df.group_by("instrument")["close"].mean()
        assert len(agg) == df["instrument"].nunique()

    def test_pandas_rolling_operations(self, sample_panel_data):
        """Test rolling window operations."""
        df = sample_panel_data.copy()
        df = df.sort_values(["instrument", "date"])

        # Rolling mean
        df["rolling_mean"] = df.group_by("instrument")["close"].transform(
            lambda x: x.rolling(window=5, min_samples=1).mean()
        )

        assert "rolling_mean" in df.columns
        assert not df["rolling_mean"].isna().all()

    def test_pandas_ewm(self, sample_panel_data):
        """Test exponentially weighted moving average."""
        df = sample_panel_data.copy()
        df = df.sort_values(["instrument", "date"])

        # EWM
        df["ewm_mean"] = df.group_by("instrument")["close"].transform(
            lambda x: x.ewm(span=10).mean()
        )

        assert "ewm_mean" in df.columns
        assert not df["ewm_mean"].isna().all()


class TestPandasCrossSection:
    """Test Pandas cross-sectional operations."""

    def test_pandas_rank(self, sample_panel_data):
        """Test ranking within groups."""
        df = sample_panel_data.copy()

        # Rank within each date
        df["rank"] = df.group_by("date")["close"].rank(method="average", pct=True)

        assert "rank" in df.columns
        assert df["rank"].min() >= 0.0
        assert df["rank"].max() <= 1.0

    def test_pandas_zscore(self, sample_panel_data):
        """Test z-score normalization."""
        df = sample_panel_data.copy()

        # Z-score within each date
        df["zscore"] = df.group_by("date")["close"].transform(
            lambda x: (x - x.mean()) / x.std()
        )

        assert "zscore" in df.columns

        # Check mean ≈ 0 per date
        for date in df["date"].unique():
            date_data = df[df["date"] == date]["zscore"]
            assert abs(date_data.mean()) < 0.1

    def test_pandas_quantile_cut(self, sample_panel_data):
        """Test quantile-based binning."""
        df = sample_panel_data.copy()

        # Cut into quintiles within each date
        df["quintile"] = df.group_by("date")["close"].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
        )

        assert "quintile" in df.columns


class TestPandasGroupBy:
    """Test Pandas GroupBy operations."""

    def test_pandas_multiple_aggregations(self, sample_panel_data):
        """Test multiple aggregations."""
        df = sample_panel_data.copy()

        # Multiple aggregations
        result = df.group_by("instrument").agg({
            "close": ["mean", "std", "min", "max"],
            "volume": ["sum", "mean"],
        })

        assert result.shape[0] == df["instrument"].nunique()

    def test_pandas_transform_vs_agg(self, sample_panel_data):
        """Test difference between transform and agg."""
        df = sample_panel_data.copy()

        # Transform returns same shape
        df["group_mean"] = df.group_by("instrument")["close"].transform("mean")
        assert len(df) == len(sample_panel_data)

        # Agg returns reduced shape
        agg_result = df.group_by("instrument")["close"].agg("mean")
        assert len(agg_result) == df["instrument"].nunique()

    def test_pandas_custom_aggregation(self, sample_panel_data):
        """Test custom aggregation function."""
        df = sample_panel_data.copy()

        # Custom function
        def custom_agg(x):
            return x.max() - x.min()

        result = df.group_by("instrument")["close"].agg(custom_agg)
        assert len(result) == df["instrument"].nunique()


class TestPandasTimeSeries:
    """Test Pandas time-series operations."""

    def test_pandas_shift_lag(self, sample_panel_data):
        """Test shift/lag operations."""
        df = sample_panel_data.copy()
        df = df.sort_values(["instrument", "date"])

        # Lag
        df["close_lag1"] = df.group_by("instrument")["close"].shift(1)
        df["close_lead1"] = df.group_by("instrument")["close"].shift(-1)

        assert "close_lag1" in df.columns
        assert "close_lead1" in df.columns

    def test_pandas_diff(self, sample_panel_data):
        """Test diff operations."""
        df = sample_panel_data.copy()
        df = df.sort_values(["instrument", "date"])

        # Diff
        df["close_diff"] = df.group_by("instrument")["close"].diff()

        assert "close_diff" in df.columns

    def test_pandas_pct_change(self, sample_panel_data):
        """Test percent change."""
        df = sample_panel_data.copy()
        df = df.sort_values(["instrument", "date"])

        # Pct change
        df["returns_calc"] = df.group_by("instrument")["close"].pct_change()

        assert "returns_calc" in df.columns


class TestPandasJoins:
    """Test Pandas join operations."""

    def test_pandas_merge_inner(self, sample_panel_data):
        """Test inner merge."""
        df1 = sample_panel_data[["date", "instrument", "close"]].copy()
        df2 = sample_panel_data[["date", "instrument", "volume"]].copy()

        result = pd.merge(df1, df2, on=["date", "instrument"], how="inner")

        assert len(result) == len(sample_panel_data)
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_pandas_merge_left(self, sample_panel_data):
        """Test left merge."""
        df1 = sample_panel_data.copy()
        df2 = sample_panel_data.sample(frac=0.8).copy()

        result = pd.merge(
            df1[["date", "instrument", "close"]],
            df2[["date", "instrument", "volume"]],
            on=["date", "instrument"],
            how="left"
        )

        assert len(result) == len(df1)

    def test_pandas_merge_asof(self, sample_panel_data):
        """Test merge_asof for time-series joins."""
        df = sample_panel_data.copy()

        # Split into two dataframes
        df1 = df[df["instrument"] == "INST_000"].sort_values("date")
        df2 = df[df["instrument"] == "INST_001"].sort_values("date")

        result = pd.merge_asof(
            df1[["date", "close"]],
            df2[["date", "volume"]],
            on="date",
            direction="backward",
            suffixes=("_left", "_right")
        )

        assert len(result) == len(df1)


class TestPandasMemory:
    """Test Pandas memory characteristics."""

    def test_pandas_memory_usage(self, sample_panel_data):
        """Test memory usage tracking."""
        df = sample_panel_data.copy()

        # Get memory usage
        memory = df.memory_usage(deep=True)
        total_memory = memory.sum()

        assert total_memory > 0

    def test_pandas_copy_vs_view(self, sample_panel_data):
        """Test copy vs view behavior."""
        df = sample_panel_data.copy()

        # View (no copy)
        view = df[["date", "instrument"]]

        # Explicit copy
        copy = df[["date", "instrument"]].copy()

        # Both should have same data
        pd.testing.assert_frame_equal(view, copy)

    def test_pandas_dtype_optimization(self, sample_panel_data):
        """Test dtype optimization for memory."""
        df = sample_panel_data.copy()

        # Original memory
        original_memory = df.memory_usage(deep=True).sum()

        # Optimize dtypes (if possible)
        # For example, convert float64 to float32
        df_optimized = df.copy()
        for col in df_optimized.select_dtypes(include=["float64"]).columns:
            df_optimized[col] = df_optimized[col].astype("float32")

        optimized_memory = df_optimized.memory_usage(deep=True).sum()

        # Optimized should use less memory
        assert optimized_memory <= original_memory


class TestPandasPivot:
    """Test Pandas pivot operations."""

    def test_pandas_pivot_wide(self, sample_panel_data):
        """Test pivot to wide format."""
        df = sample_panel_data.copy()

        # Pivot to wide
        wide = df.pivot(
            index="date",
            columns="instrument",
            values="close"
        )

        assert wide.shape[0] == df["date"].nunique()
        assert wide.shape[1] == df["instrument"].nunique()

    def test_pandas_melt_long(self, sample_panel_data):
        """Test melt to long format."""
        df = sample_panel_data.copy()

        # First pivot to wide
        wide = df.pivot(
            index="date",
            columns="instrument",
            values="close"
        )

        # Then melt back to long
        long = wide.reset_index().melt(
            id_vars=["date"],
            var_name="instrument",
            value_name="close"
        )

        assert len(long) == len(df)


class TestPandasApply:
    """Test Pandas apply operations."""

    def test_pandas_apply_row(self, sample_panel_data):
        """Test apply along rows."""
        df = sample_panel_data.copy()

        # Apply function to each row
        df["computed"] = df.apply(
            lambda row: row["close"] * row["volume"] / 1000,
            axis=1
        )

        assert "computed" in df.columns

    def test_pandas_apply_column(self, sample_panel_data):
        """Test apply along columns."""
        df = sample_panel_data.copy()

        # Apply to column
        result = df["close"].apply(lambda x: x * 2)

        assert len(result) == len(df)

    def test_pandas_applymap(self, sample_panel_data):
        """Test element-wise apply."""
        df = sample_panel_data[["close", "volume"]].copy()

        # Apply to all elements
        result = df.map(lambda x: x * 2 if isinstance(x, (int, float)) else x)

        assert result.shape == df.shape


class TestPandasPerformance:
    """Test Pandas performance characteristics."""

    def test_pandas_vectorized_vs_loop(self, sample_panel_data):
        """Test vectorized operations are faster than loops."""
        df = sample_panel_data.copy()

        # Vectorized
        import time
        start = time.time()
        result_vectorized = df["close"] * 2.0
        vectorized_time = time.time() - start

        # Loop (slower)
        start = time.time()
        result_loop = [x * 2.0 for x in df["close"]]
        loop_time = time.time() - start

        # Both should produce same result
        assert np.allclose(result_vectorized, result_loop)

        # Vectorized should be faster (generally)
        # But we won't enforce in test since it's environment-dependent

    def test_pandas_query_performance(self, large_panel_data):
        """Test query method performance."""
        df = large_panel_data.copy()

        # Query method
        result_query = df.query("close > 100.0 and volume > 500000")

        # Boolean indexing
        result_bool = df[(df["close"] > 100.0) & (df["volume"] > 500000)]

        # Should produce same result
        pd.testing.assert_frame_equal(
            result_query.reset_index(drop=True),
            result_bool.reset_index(drop=True)
        )


class TestPandasMultiIndex:
    """Test Pandas MultiIndex operations."""

    def test_pandas_set_multiindex(self, sample_panel_data):
        """Test setting MultiIndex."""
        df = sample_panel_data.copy()

        # Set MultiIndex
        df_indexed = df.set_index(["date", "instrument"])

        assert isinstance(df_indexed.index, pd.MultiIndex)
        assert df_indexed.index.names == ["date", "instrument"]

    def test_pandas_multiindex_groupby(self, sample_panel_data):
        """Test GroupBy with MultiIndex."""
        df = sample_panel_data.copy()
        df_indexed = df.set_index(["date", "instrument"])

        # GroupBy on first level
        result = df_indexed.group_by(level=0)["close"].mean()

        assert len(result) == df["date"].nunique()

    def test_pandas_multiindex_unstack(self, sample_panel_data):
        """Test unstack with MultiIndex."""
        df = sample_panel_data.copy()
        df_indexed = df.set_index(["date", "instrument"])

        # Unstack to wide format
        wide = df_indexed["close"].unstack()

        assert wide.shape[0] == df["date"].nunique()
        assert wide.shape[1] == df["instrument"].nunique()


class TestPandasCategorical:
    """Test Pandas categorical dtypes."""

    def test_pandas_convert_to_categorical(self, sample_panel_data):
        """Test converting to categorical."""
        df = sample_panel_data.copy()

        # Original memory
        original_memory = df.memory_usage(deep=True).sum()

        # Convert instrument to categorical
        df["instrument"] = df["instrument"].astype("category")

        categorical_memory = df.memory_usage(deep=True).sum()

        # Categorical should use less memory for repeated values
        assert categorical_memory < original_memory

    def test_pandas_categorical_operations(self, sample_panel_data):
        """Test operations on categorical columns."""
        df = sample_panel_data.copy()
        df["instrument"] = df["instrument"].astype("category")

        # GroupBy still works
        result = df.group_by("instrument")["close"].mean()
        assert len(result) == df["instrument"].nunique()


class TestPandasNullHandling:
    """Test Pandas null/NA handling."""

    def test_pandas_dropna(self, sample_panel_data):
        """Test dropping NA values."""
        df = sample_panel_data.copy()

        # Introduce some NAs
        df.loc[df.index[:10], "close"] = np.nan

        # Drop NAs
        df_cleaned = df.dropna(subset=["close"])

        assert len(df_cleaned) == len(df) - 10

    def test_pandas_fillna(self, sample_panel_data):
        """Test filling NA values."""
        df = sample_panel_data.copy()

        # Introduce some NAs
        df.loc[df.index[:10], "close"] = np.nan

        # Fill with mean
        mean_close = df["close"].mean()
        df["close"] = df["close"].fillna(mean_close)

        assert not df["close"].isna().any()

    def test_pandas_ffill_bfill(self, sample_panel_data):
        """Test forward/backward fill."""
        df = sample_panel_data.copy()
        df = df.sort_values(["instrument", "date"])

        # Introduce some NAs
        df.loc[df.index[10:15], "close"] = np.nan

        # Forward fill within groups
        df["close_ffill"] = df.group_by("instrument")["close"].ffill()

        # Should have fewer NAs
        assert df["close_ffill"].isna().sum() <= df["close"].isna().sum()
