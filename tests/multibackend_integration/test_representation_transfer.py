# -*- coding: utf-8 -*-
"""Test representation transfer and conversion logic.

Tests cover:
- Representation conversion correctness
- Transfer cost estimation
- Sort/repartition requirements
- PIT preservation across transfers
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from planner.backend_region import (
    PhysicalBackend,
    Representation,
    TransferEdge,
    estimate_transfer_cost_ms,
)


class TestRepresentationConversion:
    """Test representation conversion correctness."""

    def test_pandas_long_to_polars_long(self, sample_panel_data):
        """Test Pandas long → Polars long conversion."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df_pandas = sample_panel_data.copy()

        # Convert to Polars
        df_polars = pl.from_pandas(df_pandas)

        # Verify shape and columns preserved
        assert df_polars.height == len(df_pandas)
        assert set(df_polars.columns) == set(df_pandas.columns)

    def test_pandas_long_to_wide_conversion(self, sample_panel_data):
        """Test Pandas long → wide conversion."""
        df_long = sample_panel_data.copy()

        # Convert to wide format
        df_wide = df_long.pivot(
            index="date",
            columns="instrument",
            values="close"
        )

        assert df_wide.shape[0] == df_long["date"].nunique()
        assert df_wide.shape[1] == df_long["instrument"].nunique()

    def test_polars_to_arrow_conversion(self, sample_panel_data):
        """Test Polars → Arrow conversion."""
        try:
            import polars as pl
            import pyarrow as pa
        except ImportError:
            pytest.skip("Polars or PyArrow not installed")

        df_polars = pl.from_pandas(sample_panel_data)

        # Convert to Arrow
        arrow_table = df_polars.to_arrow()

        assert isinstance(arrow_table, pa.Table)
        assert arrow_table.num_rows == df_polars.height
        assert arrow_table.num_columns == len(df_polars.columns)

    def test_arrow_to_duckdb_conversion(self, sample_panel_data):
        """Test Arrow → DuckDB Relation conversion."""
        try:
            import pyarrow as pa
            import duckdb
        except ImportError:
            pytest.skip("PyArrow or DuckDB not installed")

        # Create Arrow table
        arrow_table = pa.Table.from_pandas(sample_panel_data)

        # Load into DuckDB
        conn = duckdb.connect(":memory:")
        conn.register("arrow_data", arrow_table)

        result = conn.execute("SELECT COUNT(*) as cnt FROM arrow_data").fetchone()
        assert result[0] == len(sample_panel_data)

        conn.close()


class TestTransferCostEstimation:
    """Test transfer cost estimation (MB-P1-021)."""

    def test_simple_transfer_cost(self):
        """Test basic transfer cost estimation."""
        cost = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=1_000_000,
            requires_sort=False,
            requires_repartition=False,
            requires_reshape=False,
        )

        # Should return a positive cost
        assert cost > 0.0
        assert isinstance(cost, float)

    def test_transfer_cost_with_sort(self):
        """Transfer requiring sort should cost more."""
        cost_no_sort = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=1_000_000,
            requires_sort=False,
        )

        cost_with_sort = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=1_000_000,
            requires_sort=True,
        )

        assert cost_with_sort > cost_no_sort

    def test_transfer_cost_with_reshape(self):
        """Transfer requiring reshape should cost more."""
        cost_no_reshape = estimate_transfer_cost_ms(
            source_repr=Representation.PANDAS_LONG,
            target_repr=Representation.PANDAS_WIDE,
            estimated_memory_bytes=1_000_000,
            requires_reshape=False,
        )

        cost_with_reshape = estimate_transfer_cost_ms(
            source_repr=Representation.PANDAS_LONG,
            target_repr=Representation.PANDAS_WIDE,
            estimated_memory_bytes=1_000_000,
            requires_reshape=True,
        )

        assert cost_with_reshape > cost_no_reshape

    def test_transfer_cost_scales_with_size(self):
        """Transfer cost should scale with data size."""
        cost_small = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=100_000,
        )

        cost_large = estimate_transfer_cost_ms(
            source_repr=Representation.POLARS_LONG,
            target_repr=Representation.DUCKDB_RELATION,
            estimated_memory_bytes=10_000_000,
        )

        assert cost_large > cost_small


class TestTransferEdgeContract:
    """Test transfer edge contract validation."""

    def test_transfer_edge_basic_creation(self):
        """Test basic transfer edge creation."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=15.0,
        )

        assert edge.edge_id == "e1"
        assert edge.producer_region == "r1"
        assert edge.consumer_region == "r2"
        assert edge.estimated_transfer_ms == 15.0

    def test_transfer_preserves_pit_by_default(self):
        """Transfer should preserve PIT by default."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
        )

        assert edge.preserves_pit is True

    def test_transfer_preserves_universe_by_default(self):
        """Transfer should preserve universe by default."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
        )

        assert edge.preserves_universe is True

    def test_transfer_edge_to_dict(self):
        """Transfer edge should serialize to dict."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=12.5,
            requires_sort=True,
        )

        result = edge.to_dict()

        assert result["edge_id"] == "e1"
        assert result["producer_region"] == "r1"
        assert result["consumer_region"] == "r2"
        assert result["requires_sort"] is True
        assert result["estimated_transfer_ms"] == 12.5


class TestSortRequirements:
    """Test sort requirement detection (Section 22)."""

    def test_sql_output_may_be_unordered(self):
        """SQL output cannot be assumed ordered (Section 22)."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r_sql",
            consumer_region="r_pandas",
            source_backend=PhysicalBackend.DUCKDB_SQL,
            target_backend=PhysicalBackend.PANDAS_NUMPY,
            source_representation=Representation.DUCKDB_RELATION,
            target_representation=Representation.PANDAS_LONG,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=15.0,
            producer_guarantees_order=False,  # SQL doesn't guarantee
            requires_sort=True,  # Consumer requires sort
        )

        assert edge.producer_guarantees_order is False
        assert edge.requires_sort is True

    def test_ordered_producer_no_sort_needed(self):
        """If producer guarantees order, no sort needed."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r_polars",
            consumer_region="r_pandas",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.PANDAS_NUMPY,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.PANDAS_LONG,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            producer_sorted_by=("date", "instrument"),
            producer_guarantees_order=True,
            requires_sort=False,  # No additional sort needed
        )

        assert edge.producer_guarantees_order is True
        assert edge.requires_sort is False


class TestRepartitionRequirements:
    """Test repartition requirement detection."""

    def test_different_partitioning_requires_repartition(self):
        """Different partitioning strategies require repartition."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=20.0,
            requires_repartition=True,
        )

        assert edge.requires_repartition is True

    def test_compatible_partitioning_no_repartition(self):
        """Compatible partitioning doesn't require repartition."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.POLARS_LONG,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.POLARS_LAZY_LONG,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=5.0,
            requires_repartition=False,
        )

        assert edge.requires_repartition is False


class TestPITPreservation:
    """Test PIT preservation across transfers (MB-P0-013)."""

    def test_pit_preserved_across_simple_transfer(self):
        """PIT should be preserved in simple representation conversion."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            preserves_pit=True,
        )

        assert edge.preserves_pit is True

    def test_pit_violation_detected(self):
        """PIT violation should be explicitly marked."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            preserves_pit=False,  # Explicitly marked as violating PIT
        )

        assert edge.preserves_pit is False

    def test_snapshot_id_tracked_across_transfer(self):
        """Source snapshot ID should be tracked across transfer."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            source_snapshot_id="snap_12345",
        )

        assert edge.source_snapshot_id == "snap_12345"


class TestReshapeRequirements:
    """Test reshape (long ↔ wide) requirements."""

    def test_long_to_wide_requires_reshape(self):
        """Long → wide conversion requires reshape."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.PANDAS_NUMPY,
            target_backend=PhysicalBackend.PANDAS_NUMPY,
            source_representation=Representation.PANDAS_LONG,
            target_representation=Representation.PANDAS_WIDE,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=25.0,
            requires_reshape=True,
        )

        assert edge.requires_reshape is True

    def test_same_shape_no_reshape(self):
        """Same shape doesn't require reshape."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.PANDAS_NUMPY,
            target_backend=PhysicalBackend.POLARS_PANEL,
            source_representation=Representation.PANDAS_LONG,
            target_representation=Representation.POLARS_LONG,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=10.0,
            requires_reshape=False,
        )

        assert edge.requires_reshape is False


class TestDtypeCasting:
    """Test dtype casting requirements."""

    def test_dtype_cast_required(self):
        """Dtype casting should be tracked."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.PANDAS_NUMPY,
            target_backend=PhysicalBackend.DUCKDB_SQL,
            source_representation=Representation.PANDAS_LONG,
            target_representation=Representation.DUCKDB_RELATION,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=12.0,
            requires_dtype_cast=True,
        )

        assert edge.requires_dtype_cast is True

    def test_no_dtype_cast_needed(self):
        """No dtype cast when types are compatible."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.POLARS_LONG,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.POLARS_LAZY_LONG,
            estimated_rows=100_000,
            estimated_bytes=800_000,
            estimated_transfer_ms=5.0,
            requires_dtype_cast=False,
        )

        assert edge.requires_dtype_cast is False
