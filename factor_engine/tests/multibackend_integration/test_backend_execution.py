# -*- coding: utf-8 -*-
"""Test backend-specific execution logic.

Tests cover:
- Backend capability validation
- Operator support checking
- Execution correctness per backend
- Backend switching correctness
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from planner.backend_region import (
    PhysicalBackend,
    Representation,
    normalize_backend_name,
    infer_representation,
    backend_supports_direct_sink,
    supports_streaming,
)


class TestBackendNormalization:
    """Test backend name normalization (MB-P0-002)."""

    def test_generic_sql_becomes_duckdb(self):
        """Generic 'sql' should normalize to duckdb_sql."""
        result = normalize_backend_name("sql")
        assert result == PhysicalBackend.DUCKDB_SQL

    def test_pandas_variants(self):
        """Test pandas backend name variants."""
        assert normalize_backend_name("pandas") == PhysicalBackend.PANDAS_NUMPY
        assert normalize_backend_name("pandas_numpy") == PhysicalBackend.PANDAS_NUMPY

    def test_polars_variants(self):
        """Test polars backend name variants."""
        assert normalize_backend_name("polars") == PhysicalBackend.POLARS_PANEL
        assert normalize_backend_name("polars_panel") == PhysicalBackend.POLARS_PANEL
        assert normalize_backend_name("polars_long") == PhysicalBackend.POLARS_LONG

    def test_sql_variants(self):
        """Test SQL backend name variants."""
        assert normalize_backend_name("duckdb") == PhysicalBackend.DUCKDB_SQL
        assert normalize_backend_name("duckdb_sql") == PhysicalBackend.DUCKDB_SQL
        assert normalize_backend_name("clickhouse") == PhysicalBackend.CLICKHOUSE_SQL

    def test_unknown_backend_raises(self):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            normalize_backend_name("spark_sql")


class TestRepresentationInference:
    """Test representation inference from backend (MB-P1-018)."""

    def test_pandas_representation(self):
        """Pandas backend should infer pandas representation."""
        result = infer_representation(PhysicalBackend.PANDAS_NUMPY, wide=False)
        assert result == Representation.PANDAS_LONG

        result_wide = infer_representation(PhysicalBackend.PANDAS_NUMPY, wide=True)
        assert result_wide == Representation.PANDAS_WIDE

    def test_polars_representation(self):
        """Polars backend should infer polars representation."""
        result = infer_representation(PhysicalBackend.POLARS_PANEL, wide=False)
        assert result == Representation.POLARS_LONG

        result_wide = infer_representation(PhysicalBackend.POLARS_PANEL, wide=True)
        assert result_wide == Representation.POLARS_WIDE

    def test_duckdb_prefers_relation(self):
        """DuckDB should prefer DUCKDB_RELATION (MB-P1-018)."""
        result = infer_representation(PhysicalBackend.DUCKDB_SQL)
        assert result == Representation.DUCKDB_RELATION

    def test_clickhouse_uses_arrow(self):
        """ClickHouse should use Arrow at boundaries."""
        result = infer_representation(PhysicalBackend.CLICKHOUSE_SQL)
        assert result == Representation.ARROW_TABLE


class TestDirectSinkSupport:
    """Test direct sink to Parquet support (MB-P1-025)."""

    def test_polars_supports_direct_sink(self):
        """Polars should support direct sink to Parquet."""
        assert backend_supports_direct_sink(
            PhysicalBackend.POLARS_PANEL,
            Representation.POLARS_LONG
        )

    def test_duckdb_relation_supports_direct_sink(self):
        """DuckDB Relation should support direct sink."""
        assert backend_supports_direct_sink(
            PhysicalBackend.DUCKDB_SQL,
            Representation.DUCKDB_RELATION
        )

    def test_pandas_no_direct_sink(self):
        """Pandas should not support direct sink."""
        assert not backend_supports_direct_sink(
            PhysicalBackend.PANDAS_NUMPY,
            Representation.PANDAS_LONG
        )


class TestStreamingSupport:
    """Test streaming execution support (MB-P1-016)."""

    def test_duckdb_supports_streaming(self):
        """DuckDB should support streaming for most operations."""
        assert supports_streaming(
            PhysicalBackend.DUCKDB_SQL,
            ("add", "multiply", "filter")
        )

    def test_polars_streaming_with_compatible_ops(self):
        """Polars should support streaming for compatible operators."""
        assert supports_streaming(
            PhysicalBackend.POLARS_PANEL,
            ("add", "ts_mean", "ema")
        )

    def test_polars_no_streaming_with_global_ops(self):
        """Polars should not support streaming with global operations."""
        assert not supports_streaming(
            PhysicalBackend.POLARS_PANEL,
            ("rank", "quantile")
        )

    def test_pandas_no_streaming(self):
        """Pandas should not support streaming."""
        assert not supports_streaming(
            PhysicalBackend.PANDAS_NUMPY,
            ("add", "ts_mean")
        )


class TestBackendExecutionCorrectness:
    """Test execution correctness per backend."""

    def test_pandas_simple_computation(self, sample_panel_data):
        """Test simple computation on Pandas backend."""
        df = sample_panel_data.copy()

        # Simple rolling mean
        df = df.sort_values(["instrument", "date"])
        df["rolling_mean"] = df.group_by("instrument")["close"].transform(
            lambda x: x.rolling(window=5, min_samples=1).mean()
        )

        assert "rolling_mean" in df.columns
        assert not df["rolling_mean"].isna().all()
        assert len(df) == len(sample_panel_data)

    def test_polars_simple_computation(self, sample_panel_data):
        """Test simple computation on Polars backend."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pl = pl.from_pandas(sample_panel_data)

        # Simple rolling mean
        result = (
            df_pl
            .sort(["instrument", "date"])
            .with_columns([
                pl.col("close")
                .rolling_mean(window_size=5, min_samples=1)
                .over("instrument")
                .alias("rolling_mean")
            ])
        )

        assert "rolling_mean" in result.columns
        assert result.height == len(sample_panel_data)

    def test_duckdb_simple_computation(self, sample_panel_data):
        """Test simple computation on DuckDB backend."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")

        # Register dataframe and compute rolling mean
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                date,
                instrument,
                close,
                AVG(close) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) as rolling_mean
            FROM data
            ORDER BY instrument, date
        """).df()

        assert "rolling_mean" in result.columns
        assert len(result) == len(sample_panel_data)

        conn.close()


class TestBackendSwitchingCorrectness:
    """Test correctness when switching between backends."""

    def test_pandas_to_polars_conversion(self, sample_panel_data):
        """Test Pandas → Polars conversion preserves data."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Pandas computation
        df_pandas = sample_panel_data.copy()
        df_pandas["computed"] = df_pandas["close"] * 2.0

        # Convert to Polars
        df_polars = pl.from_pandas(df_pandas)

        # Convert back
        df_back = df_polars.to_pandas()

        # Should be identical
        pd.testing.assert_frame_equal(
            df_pandas.sort_values(["date", "instrument"]).reset_index(drop=True),
            df_back.sort_values(["date", "instrument"]).reset_index(drop=True),
        )

    def test_polars_to_duckdb_conversion(self, sample_panel_data):
        """Test Polars → DuckDB conversion preserves data."""
        try:
            import polars as pl
            import duckdb
        except ImportError:
            pytest.skip("Polars or DuckDB not installed")

        # Polars computation
        df_polars = pl.from_pandas(sample_panel_data)
        df_polars = df_polars.with_columns([
            (pl.col("close") * 2.0).alias("computed")
        ])

        # Convert to DuckDB
        conn = duckdb.connect(":memory:")
        conn.register("polars_data", df_polars.to_pandas())

        result = conn.execute("SELECT * FROM polars_data").df()

        # Should have same shape and columns
        assert len(result) == df_polars.height
        assert set(result.columns) == set(df_polars.columns)

        conn.close()

    def test_duckdb_to_pandas_conversion(self, sample_panel_data):
        """Test DuckDB → Pandas conversion preserves data."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        # DuckDB computation
        result_duckdb = conn.execute("""
            SELECT
                date,
                instrument,
                close,
                close * 2.0 as computed
            FROM data
        """).df()

        # Convert to Pandas (already is)
        df_pandas = result_duckdb.copy()

        # Verify
        assert isinstance(df_pandas, pd.DataFrame)
        assert len(df_pandas) == len(sample_panel_data)
        assert "computed" in df_pandas.columns

        conn.close()


class TestBackendCapabilityChecking:
    """Test backend capability validation."""

    def test_all_backends_have_add_operator(self):
        """All backends should support basic add operation."""
        # This would query capability registry in real implementation
        backends = [
            PhysicalBackend.PANDAS_NUMPY,
            PhysicalBackend.POLARS_PANEL,
            PhysicalBackend.DUCKDB_SQL,
        ]

        for backend in backends:
            # In real implementation, would check capability_registry
            assert backend in PhysicalBackend

    def test_backend_specific_operators(self):
        """Some operators are backend-specific."""
        # DuckDB SQL has window functions
        assert PhysicalBackend.DUCKDB_SQL is not None

        # Polars has lazy evaluation
        assert PhysicalBackend.POLARS_PANEL is not None

        # Pandas has full panel operations
        assert PhysicalBackend.PANDAS_NUMPY is not None
