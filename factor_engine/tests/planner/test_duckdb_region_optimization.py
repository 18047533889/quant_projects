# -*- coding: utf-8 -*-
"""Test suite for DuckDB Region optimizations (MB-P1-018, Section 21-22).

Verifies:
1. DuckDB → Arrow boundary preference (Section 21)
2. SQL ordering guarantees and sort cost tracking (Section 22)
3. Concrete backend identity (duckdb_sql not generic sql) (MB-P0-002)
"""
from __future__ import annotations

import pytest

from factor_engine.planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    Representation,
)
from factor_engine.planner.duckdb_region_optimizer import (
    estimate_duckdb_to_arrow_cost,
    estimate_duckdb_to_pandas_cost,
    infer_duckdb_region_output_properties,
    optimize_duckdb_region_boundary,
    should_keep_duckdb_relation,
)
from factor_engine.planner.transfer_cost import (
    check_producer_ordering,
    estimate_sort_cost,
    estimate_transfer_edge_cost,
)


def test_duckdb_to_arrow_is_cheaper_than_pandas():
    """Section 21: DuckDB → Arrow conversion is cheaper than DuckDB → Pandas."""
    rows = 1_000_000
    arrow_cost = estimate_duckdb_to_arrow_cost(rows)
    pandas_cost = estimate_duckdb_to_pandas_cost(rows)

    # Arrow conversion should be significantly cheaper
    assert arrow_cost < pandas_cost
    assert arrow_cost < 5.0  # Should be fast
    assert pandas_cost > 5.0  # More expensive


def test_sql_ordering_not_guaranteed_without_explicit_order_by():
    """Section 22: SQL results are unordered unless there's explicit ORDER BY."""
    region = BackendRegion(
        region_id="duckdb_r1",
        backend=PhysicalBackend.DUCKDB_SQL,
        representation=Representation.DUCKDB_RELATION,
        node_ids=("n1", "n2"),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1_000_000,
        estimated_compute_ms=100.0,
        estimated_memory_bytes=8_000_000,
        sorted_by=(),  # No ORDER BY in SQL
    )

    props = infer_duckdb_region_output_properties(region, has_explicit_order_by=False)

    # Without ORDER BY, SQL doesn't guarantee ordering
    assert props.guarantees_order is False
    assert props.sorted_by == ()


def test_sql_ordering_guaranteed_with_order_by():
    """Section 22: SQL with ORDER BY guarantees the specified ordering."""
    region = BackendRegion(
        region_id="duckdb_r1",
        backend=PhysicalBackend.DUCKDB_SQL,
        representation=Representation.DUCKDB_RELATION,
        node_ids=("n1", "n2"),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1_000_000,
        estimated_compute_ms=100.0,
        estimated_memory_bytes=8_000_000,
        sorted_by=("instrument", "datetime"),
    )

    props = infer_duckdb_region_output_properties(region, has_explicit_order_by=True)

    # With ORDER BY, SQL guarantees the ordering
    assert props.guarantees_order is True
    assert props.sorted_by == ("instrument", "datetime")


def test_producer_ordering_check_satisfied():
    """Section 22: Check if producer ordering satisfies consumer requirement."""
    # Producer sorted by (instrument, datetime)
    producer_sorted = ("instrument", "datetime")
    # Consumer needs (instrument, datetime)
    consumer_required = ("instrument", "datetime")

    satisfied = check_producer_ordering(producer_sorted, consumer_required)
    assert satisfied is True  # No sort needed


def test_producer_ordering_check_not_satisfied():
    """Section 22: Consumer needs ordering that producer doesn't provide."""
    # Producer doesn't guarantee ordering
    producer_sorted = ()
    # Consumer needs (instrument, datetime)
    consumer_required = ("instrument", "datetime")

    satisfied = check_producer_ordering(producer_sorted, consumer_required)
    assert satisfied is False  # Sort needed


def test_producer_ordering_check_prefix_match():
    """Section 22: Producer ordering must be complete prefix of consumer requirement."""
    # Producer sorted by (instrument,) only
    producer_sorted = ("instrument",)
    # Consumer needs (instrument, datetime)
    consumer_required = ("instrument", "datetime")

    satisfied = check_producer_ordering(producer_sorted, consumer_required)
    assert satisfied is False  # Partial match doesn't satisfy


def test_transfer_edge_requires_sort_when_ordering_not_satisfied():
    """Section 22: TransferEdge.requires_sort=True when consumer needs ordering."""
    producer = BackendRegion(
        region_id="duckdb_r1",
        backend=PhysicalBackend.DUCKDB_SQL,
        representation=Representation.DUCKDB_RELATION,
        node_ids=("n1",),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1_000_000,
        estimated_compute_ms=100.0,
        estimated_memory_bytes=8_000_000,
        sorted_by=(),  # No ordering guarantee
    )

    consumer = BackendRegion(
        region_id="polars_r2",
        backend=PhysicalBackend.POLARS_LONG,
        representation=Representation.POLARS_LAZY_LONG,
        node_ids=("n2",),
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=1_000_000,
        estimated_compute_ms=50.0,
        estimated_memory_bytes=8_000_000,
    )

    producer_props = infer_duckdb_region_output_properties(producer, has_explicit_order_by=False)

    edge = optimize_duckdb_region_boundary(
        producer_region=producer,
        consumer_region=consumer,
        producer_properties=producer_props,
        consumer_requires_order=("instrument", "datetime"),
        estimated_rows=1_000_000,
    )

    # Section 22: Requires sort because producer doesn't guarantee order
    assert edge.requires_sort is True
    assert edge.producer_guarantees_order is False
    # Transfer cost should include sort cost
    assert edge.estimated_transfer_ms > 0


def test_transfer_edge_no_sort_when_ordering_satisfied():
    """Section 22: No sort needed when producer already provides required ordering."""
    producer = BackendRegion(
        region_id="duckdb_r1",
        backend=PhysicalBackend.DUCKDB_SQL,
        representation=Representation.DUCKDB_RELATION,
        node_ids=("n1",),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1_000_000,
        estimated_compute_ms=100.0,
        estimated_memory_bytes=8_000_000,
        sorted_by=("instrument", "datetime"),
    )

    consumer = BackendRegion(
        region_id="polars_r2",
        backend=PhysicalBackend.POLARS_LONG,
        representation=Representation.POLARS_LAZY_LONG,
        node_ids=("n2",),
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=1_000_000,
        estimated_compute_ms=50.0,
        estimated_memory_bytes=8_000_000,
    )

    producer_props = infer_duckdb_region_output_properties(producer, has_explicit_order_by=True)

    edge = optimize_duckdb_region_boundary(
        producer_region=producer,
        consumer_region=consumer,
        producer_properties=producer_props,
        consumer_requires_order=("instrument", "datetime"),
        estimated_rows=1_000_000,
    )

    # No sort needed
    assert edge.requires_sort is False
    assert edge.producer_guarantees_order is True
    assert edge.producer_sorted_by == ("instrument", "datetime")


def test_duckdb_to_polars_prefers_arrow_boundary():
    """Section 21: DuckDB → Polars should use Arrow boundary, not Pandas."""
    producer = BackendRegion(
        region_id="duckdb_r1",
        backend=PhysicalBackend.DUCKDB_SQL,
        representation=Representation.DUCKDB_RELATION,
        node_ids=("n1",),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1_000_000,
        estimated_compute_ms=100.0,
        estimated_memory_bytes=8_000_000,
    )

    consumer = BackendRegion(
        region_id="polars_r2",
        backend=PhysicalBackend.POLARS_LONG,
        representation=Representation.POLARS_LAZY_LONG,
        node_ids=("n2",),
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=1_000_000,
        estimated_compute_ms=50.0,
        estimated_memory_bytes=8_000_000,
    )

    producer_props = infer_duckdb_region_output_properties(producer)

    edge = optimize_duckdb_region_boundary(
        producer_region=producer,
        consumer_region=consumer,
        producer_properties=producer_props,
        estimated_rows=1_000_000,
    )

    # Section 21: Should use Arrow boundary
    assert edge.source_representation == Representation.ARROW_TABLE
    assert edge.target_representation == Representation.POLARS_LAZY_LONG


def test_duckdb_to_pandas_only_when_necessary():
    """Section 21: DuckDB → Pandas conversion only when consumer needs it."""
    producer = BackendRegion(
        region_id="duckdb_r1",
        backend=PhysicalBackend.DUCKDB_SQL,
        representation=Representation.DUCKDB_RELATION,
        node_ids=("n1",),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1_000_000,
        estimated_compute_ms=100.0,
        estimated_memory_bytes=8_000_000,
    )

    consumer = BackendRegion(
        region_id="pandas_r2",
        backend=PhysicalBackend.PANDAS_NUMPY,
        representation=Representation.PANDAS_LONG,
        node_ids=("n2",),
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=1_000_000,
        estimated_compute_ms=50.0,
        estimated_memory_bytes=8_000_000,
    )

    producer_props = infer_duckdb_region_output_properties(producer)

    edge = optimize_duckdb_region_boundary(
        producer_region=producer,
        consumer_region=consumer,
        producer_properties=producer_props,
        estimated_rows=1_000_000,
    )

    # Only convert to Pandas when consumer explicitly needs it
    assert edge.target_representation == Representation.PANDAS_LONG


def test_should_keep_duckdb_relation_for_sql_downstream():
    """Section 21: Keep DuckDB Relation when downstream is SQL."""
    downstream = [PhysicalBackend.DUCKDB_SQL]

    keep_relation = should_keep_duckdb_relation(downstream)

    # Should keep Relation for SQL-to-SQL transfer
    assert keep_relation is True


def test_should_keep_duckdb_relation_for_polars_downstream():
    """Section 21: Prefer Arrow boundary for Polars (not premature Pandas conversion)."""
    downstream = [PhysicalBackend.POLARS_LONG]

    keep_relation = should_keep_duckdb_relation(downstream)

    # Should keep Relation and convert to Arrow at boundary
    assert keep_relation is True


def test_sort_cost_included_when_required():
    """Section 22: Sort cost must be included in transfer edge when requires_sort=True."""
    rows = 1_000_000
    sort_cost = estimate_sort_cost(PhysicalBackend.POLARS_LONG, rows, sort_key_count=2)

    # Sort cost should be measurable
    assert sort_cost > 0

    # Estimate transfer cost with sort
    cost_with_sort = estimate_transfer_edge_cost(
        source_backend=PhysicalBackend.DUCKDB_SQL,
        target_backend=PhysicalBackend.POLARS_LONG,
        source_repr=Representation.ARROW_TABLE,
        target_repr=Representation.POLARS_LAZY_LONG,
        rows=rows,
        requires_sort=True,
        sort_key_count=2,
    )

    # Estimate without sort
    cost_without_sort = estimate_transfer_edge_cost(
        source_backend=PhysicalBackend.DUCKDB_SQL,
        target_backend=PhysicalBackend.POLARS_LONG,
        source_repr=Representation.ARROW_TABLE,
        target_repr=Representation.POLARS_LAZY_LONG,
        rows=rows,
        requires_sort=False,
    )

    # Cost with sort should be higher
    assert cost_with_sort.total_ms > cost_without_sort.total_ms
    assert cost_with_sort.sort_ms > 0
    assert cost_without_sort.sort_ms == 0


def test_concrete_backend_identity_duckdb_sql():
    """MB-P0-002: Must use concrete backend 'duckdb_sql' not generic 'sql'."""
    from factor_engine.backend.duckdb_pushdown_backend import DuckDBPushdownBackend

    backend = DuckDBPushdownBackend()

    # MB-P0-002: Runtime backend label must be concrete
    assert backend.runtime_backend_label == "duckdb_sql"
    assert backend.runtime_backend_label != "sql"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
