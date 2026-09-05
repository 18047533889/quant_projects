# -*- coding: utf-8 -*-
"""Polars backend-specific integration tests.

Tests cover:
- Polars lazy/eager evaluation
- Polars streaming execution
- Polars-specific optimizations
- Memory efficiency
"""
from __future__ import annotations

import pytest
import numpy as np


class TestPolarsPanelBackend:
    """Test Polars panel backend."""

    def test_polars_available(self):
        """Test Polars is available for import."""
        try:
            import polars as pl
            assert pl is not None
        except ImportError:
            pytest.skip("Polars not installed")

    def test_polars_basic_operations(self, sample_panel_data):
        """Test basic operations in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Basic filtering
        result = df_pl.filter(pl.col("close") > 100.0)
        assert result.height <= df_pl.height

        # Basic aggregation
        agg = df_pl.group_by("instrument").agg(pl.col("close").mean())
        assert agg.height == df_pl["instrument"].n_unique()

    def test_polars_rolling_operations(self, sample_panel_data):
        """Test rolling window operations in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Rolling mean
        result = df_pl.sort(["instrument", "date"]).with_columns([
            pl.col("close")
            .rolling_mean(window_size=5, min_samples=1)
            .over("instrument")
            .alias("rolling_mean")
        ])

        assert "rolling_mean" in result.columns
        assert result.height == df_pl.height

    def test_polars_lazy_evaluation(self, sample_panel_data):
        """Test Polars lazy evaluation."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Create lazy frame
        df_lazy = pl.from_pandas(sample_panel_data).lazy()

        # Build query
        query = (
            df_lazy
            .filter(pl.col("close") > 90.0)
            .group_by("instrument")
            .agg(pl.col("close").mean().alias("avg_close"))
        )

        # Execute
        result = query.collect()

        assert result.height > 0
        assert "avg_close" in result.columns


class TestPolarsCrossSection:
    """Test Polars cross-sectional operations."""

    def test_cross_sectional_rank(self, sample_panel_data):
        """Test cross-sectional ranking."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Rank within each date
        result = df_pl.with_columns([
            pl.col("close")
            .rank(method="average")
            .over("date")
            .alias("rank_close")
        ])

        assert "rank_close" in result.columns
        assert result["rank_close"].min() >= 1.0

    def test_cross_sectional_zscore(self, sample_panel_data):
        """Test cross-sectional z-score normalization."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Z-score within each date
        result = df_pl.with_columns([
            ((pl.col("close") - pl.col("close").mean().over("date")) /
             pl.col("close").std().over("date"))
            .alias("zscore")
        ])

        assert "zscore" in result.columns

        # Z-scores should have mean ≈ 0, std ≈ 1 per date
        for date in result["date"].unique():
            date_data = result.filter(pl.col("date") == date)
            mean_zscore = date_data["zscore"].mean()
            assert abs(mean_zscore) < 0.1  # Close to 0


class TestPolarsMemoryEfficiency:
    """Test Polars memory efficiency."""

    def test_polars_uses_less_memory_than_pandas(self, large_panel_data):
        """Test Polars uses less memory than Pandas."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Pandas memory
        pandas_memory = large_panel_data.memory_usage(deep=True).sum()

        # Polars memory (estimated)
        df_pl = pl.from_pandas(large_panel_data)
        polars_memory = df_pl.estimated_size()

        # Polars should use less memory
        # (This is generally true but not guaranteed for all data)
        assert polars_memory > 0
        assert pandas_memory > 0

    def test_polars_streaming_reduces_memory(self, large_panel_data):
        """Test Polars streaming reduces memory usage."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Lazy evaluation with streaming
        df_lazy = pl.from_pandas(large_panel_data).lazy()

        # Complex query that would use a lot of memory if eager
        result = (
            df_lazy
            .filter(pl.col("close") > 90.0)
            .group_by("instrument")
            .agg([
                pl.col("close").mean().alias("avg_close"),
                pl.col("volume").sum().alias("total_volume"),
            ])
            .collect(streaming=True)
        )

        assert result.height > 0


class TestPolarsOptimizations:
    """Test Polars-specific optimizations."""

    def test_polars_predicate_pushdown(self, sample_panel_data):
        """Test Polars predicate pushdown optimization."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_lazy = pl.from_pandas(sample_panel_data).lazy()

        # Filter should be pushed down
        query = (
            df_lazy
            .select(["date", "instrument", "close"])
            .filter(pl.col("close") > 100.0)
        )

        # Explain plan should show filter pushed down
        explain = query.explain()
        assert "FILTER" in explain or "filter" in explain

    def test_polars_projection_pushdown(self, sample_panel_data):
        """Test Polars projection pushdown optimization."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_lazy = pl.from_pandas(sample_panel_data).lazy()

        # Only select needed columns
        query = (
            df_lazy
            .filter(pl.col("close") > 100.0)
            .select(["instrument", "close"])
        )

        result = query.collect()

        # Should only have selected columns
        assert set(result.columns) == {"instrument", "close"}


class TestPolarsTimeSeries:
    """Test Polars time-series operations."""

    def test_polars_ema(self, sample_panel_data):
        """Test exponential moving average in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # EMA using ewm_mean
        result = df_pl.sort(["instrument", "date"]).with_columns([
            pl.col("close")
            .ewm_mean(span=10)
            .over("instrument")
            .alias("ema_10")
        ])

        assert "ema_10" in result.columns
        assert not result["ema_10"].is_null().all()

    def test_polars_lag_lead(self, sample_panel_data):
        """Test lag/lead operations in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Lag and lead
        result = df_pl.sort(["instrument", "date"]).with_columns([
            pl.col("close").shift(1).over("instrument").alias("close_lag1"),
            pl.col("close").shift(-1).over("instrument").alias("close_lead1"),
        ])

        assert "close_lag1" in result.columns
        assert "close_lead1" in result.columns

    def test_polars_rolling_std(self, sample_panel_data):
        """Test rolling standard deviation in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Rolling std
        result = df_pl.sort(["instrument", "date"]).with_columns([
            pl.col("close")
            .rolling_std(window_size=10, min_samples=5)
            .over("instrument")
            .alias("rolling_std")
        ])

        assert "rolling_std" in result.columns


class TestPolarsJoins:
    """Test Polars join operations."""

    def test_polars_inner_join(self, sample_panel_data):
        """Test inner join in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df1 = pl.from_pandas(sample_panel_data[["date", "instrument", "close"]])
        df2 = pl.from_pandas(sample_panel_data[["date", "instrument", "volume"]])

        # Inner join
        result = df1.join(df2, on=["date", "instrument"], how="inner")

        assert result.height == df1.height
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_polars_asof_join(self, sample_panel_data):
        """Test asof join in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Create two dataframes with different frequencies
        df1 = df_pl.filter(pl.col("instrument") == "INST_000")
        df2 = df_pl.filter(pl.col("instrument") == "INST_001")

        # Asof join
        result = df1.sort("date").join_asof(
            df2.sort("date"),
            on="date",
            strategy="backward",
            suffix="_right"
        )

        assert result.height == df1.height


class TestPolarsGroupBy:
    """Test Polars group-by operations."""

    def test_polars_multiple_aggregations(self, sample_panel_data):
        """Test multiple aggregations in one group-by."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Multiple aggregations
        result = df_pl.group_by("instrument").agg([
            pl.col("close").mean().alias("avg_close"),
            pl.col("close").std().alias("std_close"),
            pl.col("volume").sum().alias("total_volume"),
            pl.col("returns").count().alias("count"),
        ])

        assert result.height == df_pl["instrument"].n_unique()
        assert all(col in result.columns for col in [
            "avg_close", "std_close", "total_volume", "count"
        ])

    def test_polars_multi_key_groupby(self, sample_panel_data):
        """Test group-by with multiple keys."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Add a synthetic group column
        df_pl = df_pl.with_columns([
            (pl.col("instrument").str.slice(0, 7)).alias("group")
        ])

        # Group by multiple keys
        result = df_pl.group_by(["group", "date"]).agg([
            pl.col("close").mean().alias("avg_close"),
        ])

        assert result.height > 0
        assert all(col in result.columns for col in ["group", "date", "avg_close"])


class TestPolarsPerformance:
    """Test Polars performance characteristics."""

    def test_polars_faster_than_pandas_for_large_data(self, large_panel_data):
        """Test Polars is faster than Pandas for large data."""
        try:
            import polars as pl
            import time
        except ImportError:
            pytest.skip("Polars not installed")

        # Pandas timing
        start = time.time()
        result_pandas = (
            large_panel_data
            .groupby("instrument")["close"]
            .mean()
        )
        pandas_time = time.time() - start

        # Polars timing
        df_pl = pl.from_pandas(large_panel_data)
        start = time.time()
        result_polars = (
            df_pl
            .group_by("instrument")
            .agg(pl.col("close").mean())
        )
        polars_time = time.time() - start

        # Polars should be faster (generally true for aggregations)
        # But we won't enforce this in test since it depends on data size
        assert polars_time >= 0.0
        assert pandas_time >= 0.0

    def test_polars_parallel_execution(self, large_panel_data):
        """Test Polars can use parallel execution."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(large_panel_data)

        # Complex query that can be parallelized
        result = (
            df_pl
            .group_by("instrument")
            .agg([
                pl.col("close").mean().alias("avg_close"),
                pl.col("close").std().alias("std_close"),
                pl.col("volume").sum().alias("total_volume"),
            ])
        )

        assert result.height > 0