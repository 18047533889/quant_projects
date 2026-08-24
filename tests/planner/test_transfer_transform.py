# -*- coding: utf-8 -*-
"""Regression tests for R21-TRANSFER-BOUNDARIES.

Tests formal transfer boundaries for q→Arrow, ClickHouse→Arrow, Polars→NumPy, NumPy→Polars.
"""
from __future__ import annotations

import pytest

from factor_engine.planner.backend_region import (
    PhysicalBackend,
    Representation,
    TransferTransform,
    TransferEdge,
    infer_transfer_transform,
    estimate_transfer_cost_ms,
)


class TestTransferTransformInference:
    """Test infer_transfer_transform function."""

    def test_duckdb_to_arrow(self):
        """DuckDB relation → Arrow table should infer DUCKDB_TO_ARROW."""
        result = infer_transfer_transform(
            Representation.DUCKDB_RELATION, Representation.ARROW_TABLE
        )
        assert result == TransferTransform.DUCKDB_TO_ARROW

    def test_q_to_arrow_table(self):
        """Q table → Arrow table should infer Q_TO_ARROW."""
        result = infer_transfer_transform(
            Representation.Q_TABLE, Representation.ARROW_TABLE
        )
        assert result == TransferTransform.Q_TO_ARROW

    def test_q_to_arrow_vector(self):
        """Q vector → Arrow table should infer Q_TO_ARROW."""
        result = infer_transfer_transform(
            Representation.Q_VECTOR, Representation.ARROW_TABLE
        )
        assert result == TransferTransform.Q_TO_ARROW

    def test_q_to_arrow_keyed_table(self):
        """Q keyed table → Arrow table should infer Q_TO_ARROW."""
        result = infer_transfer_transform(
            Representation.Q_KEYED_TABLE, Representation.ARROW_TABLE
        )
        assert result == TransferTransform.Q_TO_ARROW

    def test_clickhouse_to_arrow_with_backends(self):
        """ClickHouse Arrow → DuckDB Arrow should infer CLICKHOUSE_TO_ARROW."""
        result = infer_transfer_transform(
            Representation.ARROW_TABLE, Representation.ARROW_TABLE,
            source_backend=PhysicalBackend.CLICKHOUSE_SQL,
            target_backend=PhysicalBackend.DUCKDB_SQL,
        )
        assert result == TransferTransform.CLICKHOUSE_TO_ARROW

    def test_polars_to_numpy_long(self):
        """Polars long → NumPy panel should infer POLARS_TO_NUMPY."""
        result = infer_transfer_transform(
            Representation.POLARS_LONG, Representation.NUMPY_PANEL
        )
        assert result == TransferTransform.POLARS_TO_NUMPY

    def test_polars_to_numpy_wide(self):
        """Polars wide → NumPy panel should infer POLARS_TO_NUMPY."""
        result = infer_transfer_transform(
            Representation.POLARS_WIDE, Representation.NUMPY_PANEL
        )
        assert result == TransferTransform.POLARS_TO_NUMPY

    def test_polars_to_numpy_lazy(self):
        """Polars lazy long → NumPy panel should infer POLARS_TO_NUMPY."""
        result = infer_transfer_transform(
            Representation.POLARS_LAZY_LONG, Representation.NUMPY_PANEL
        )
        assert result == TransferTransform.POLARS_TO_NUMPY

    def test_numpy_to_polars_long(self):
        """NumPy panel → Polars long should infer NUMPY_TO_POLARS."""
        result = infer_transfer_transform(
            Representation.NUMPY_PANEL, Representation.POLARS_LONG
        )
        assert result == TransferTransform.NUMPY_TO_POLARS

    def test_numpy_to_polars_wide(self):
        """NumPy panel → Polars wide should infer NUMPY_TO_POLARS."""
        result = infer_transfer_transform(
            Representation.NUMPY_PANEL, Representation.POLARS_WIDE
        )
        assert result == TransferTransform.NUMPY_TO_POLARS

    def test_numpy_to_polars_lazy(self):
        """NumPy panel → Polars lazy long should infer NUMPY_TO_POLARS."""
        result = infer_transfer_transform(
            Representation.NUMPY_PANEL, Representation.POLARS_LAZY_LONG
        )
        assert result == TransferTransform.NUMPY_TO_POLARS

    def test_same_backend_native(self):
        """Same representation should infer SAME_BACKEND_NATIVE."""
        result = infer_transfer_transform(
            Representation.POLARS_LONG, Representation.POLARS_LONG
        )
        assert result == TransferTransform.SAME_BACKEND_NATIVE

    def test_arrow_to_polars(self):
        """Arrow → Polars should infer ARROW_TO_POLARS."""
        result = infer_transfer_transform(
            Representation.ARROW_TABLE, Representation.POLARS_LONG
        )
        assert result == TransferTransform.ARROW_TO_POLARS

    def test_polars_to_pandas(self):
        """Polars → Pandas should infer POLARS_TO_PANDAS."""
        result = infer_transfer_transform(
            Representation.POLARS_LONG, Representation.PANDAS_LONG
        )
        assert result == TransferTransform.POLARS_TO_PANDAS

    def test_pandas_to_polars(self):
        """Pandas → Polars should infer PANDAS_TO_POLARS."""
        result = infer_transfer_transform(
            Representation.PANDAS_LONG, Representation.POLARS_LONG
        )
        assert result == TransferTransform.PANDAS_TO_POLARS


class TestTransferEdgeTransformBinding:
    """Test TransferEdge with transform field."""

    def test_transfer_edge_transform_field(self):
        """TransferEdge should have transform field and serialize it."""
        edge = TransferEdge(
            edge_id="e1",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.Q_KDB,
            target_backend=PhysicalBackend.PANDAS_NUMPY,
            source_representation=Representation.Q_TABLE,
            target_representation=Representation.ARROW_TABLE,
            transform=TransferTransform.Q_TO_ARROW,
            estimated_rows=1000000,
            estimated_bytes=1024 * 1024 * 100,
            estimated_transfer_ms=100.0,
        )
        edge_dict = edge.to_dict()
        assert edge_dict["transform"] == "q_to_arrow"

    def test_transfer_edge_polars_to_numpy(self):
        """TransferEdge with POLARS_TO_NUMPY transform."""
        edge = TransferEdge(
            edge_id="e2",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.POLARS_PANEL,
            target_backend=PhysicalBackend.PANDAS_NUMPY,
            source_representation=Representation.POLARS_LONG,
            target_representation=Representation.NUMPY_PANEL,
            transform=TransferTransform.POLARS_TO_NUMPY,
            estimated_rows=500000,
            estimated_bytes=50 * 1024 * 1024,
            estimated_transfer_ms=50.0,
        )
        edge_dict = edge.to_dict()
        assert edge_dict["transform"] == "polars_to_numpy"

    def test_transfer_edge_numpy_to_polars(self):
        """TransferEdge with NUMPY_TO_POLARS transform."""
        edge = TransferEdge(
            edge_id="e3",
            producer_region="r1",
            consumer_region="r2",
            source_backend=PhysicalBackend.PANDAS_NUMPY,
            target_backend=PhysicalBackend.POLARS_PANEL,
            source_representation=Representation.NUMPY_PANEL,
            target_representation=Representation.POLARS_LONG,
            transform=TransferTransform.NUMPY_TO_POLARS,
            estimated_rows=500000,
            estimated_bytes=50 * 1024 * 1024,
            estimated_transfer_ms=50.0,
        )
        edge_dict = edge.to_dict()
        assert edge_dict["transform"] == "numpy_to_polars"


class TestTransferCostEstimationNewTransforms:
    """Test estimate_transfer_cost_ms with new transfer transforms."""

    def test_q_to_arrow_cost(self):
        """Q → Arrow conversion should have defined cost."""
        cost = estimate_transfer_cost_ms(
            Representation.Q_TABLE,
            Representation.ARROW_TABLE,
            estimated_bytes=100 * 1024 * 1024,  # 100MB
        )
        # Should be positive and reasonable
        assert cost > 0
        assert cost < 1000  # Less than 1 second for 100MB

    def test_polars_to_numpy_cost(self):
        """Polars → NumPy conversion should have defined cost."""
        cost = estimate_transfer_cost_ms(
            Representation.POLARS_LONG,
            Representation.NUMPY_PANEL,
            estimated_bytes=100 * 1024 * 1024,  # 100MB
        )
        # Should be positive and reasonable
        assert cost > 0
        assert cost < 1000  # Less than 1 second for 100MB

    def test_numpy_to_polars_cost(self):
        """NumPy → Polars conversion should have defined cost."""
        cost = estimate_transfer_cost_ms(
            Representation.NUMPY_PANEL,
            Representation.POLARS_LONG,
            estimated_bytes=100 * 1024 * 1024,  # 100MB
        )
        # Should be positive and reasonable
        assert cost > 0
        assert cost < 1000  # Less than 1 second for 100MB

    def test_cost_with_sort(self):
        """Transfer cost should increase with sort requirement."""
        base_cost = estimate_transfer_cost_ms(
            Representation.Q_TABLE,
            Representation.ARROW_TABLE,
            estimated_bytes=100 * 1024 * 1024,
            requires_sort=False,
        )
        sort_cost = estimate_transfer_cost_ms(
            Representation.Q_TABLE,
            Representation.ARROW_TABLE,
            estimated_bytes=100 * 1024 * 1024,
            requires_sort=True,
        )
        assert sort_cost > base_cost

    def test_cost_with_repartition(self):
        """Transfer cost should increase with repartition requirement."""
        base_cost = estimate_transfer_cost_ms(
            Representation.POLARS_LONG,
            Representation.NUMPY_PANEL,
            estimated_bytes=100 * 1024 * 1024,
            requires_repartition=False,
        )
        repartition_cost = estimate_transfer_cost_ms(
            Representation.POLARS_LONG,
            Representation.NUMPY_PANEL,
            estimated_bytes=100 * 1024 * 1024,
            requires_repartition=True,
        )
        assert repartition_cost > base_cost
