# -*- coding: utf-8 -*-
"""DuckDB Region boundary optimization (MB-P1-018, Section 21-22).

Section 21: DuckDB Region boundary should prefer Arrow/Relation output, avoiding
unnecessary `.df()` conversion to Pandas when downstream can consume Arrow.

Section 22: SQL ordering cannot be assumed. Time-series downstream requiring
sorted_by=(instrument, datetime) must check producer properties; if producer
lacks the property, set TransferEdge.requires_sort=True and account for sort cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from planner.backend_region import (
    BackendRegion,
    PhysicalBackend,
    Representation,
    TransferEdge,
    TransferTransform,
    infer_transfer_transform,
)
from planner.transfer_cost import check_producer_ordering, estimate_transfer_edge_cost


@dataclass(frozen=True)
class DuckDBRegionProperties:
    """Properties of a DuckDB Region output.

    Section 22: Track whether the region guarantees ordering, so downstream
    consumers can decide if they need to explicitly sort.
    """

    region_id: str
    output_representation: Representation
    guarantees_order: bool = False  # Section 22: SQL results may be unordered
    sorted_by: tuple[str, ...] = ()  # Keys the output is sorted by (if any)
    # Section 21: Prefer Arrow/Relation boundary
    prefer_arrow_boundary: bool = True


def infer_duckdb_region_output_properties(
    region: BackendRegion,
    has_explicit_order_by: bool = False,
) -> DuckDBRegionProperties:
    """Infer output properties of a DuckDB Region.

    Section 22: SQL results are unordered unless there's an explicit ORDER BY clause.
    Even with ORDER BY, the guarantee is only for the specified keys.

    Args:
        region: The DuckDB backend region.
        has_explicit_order_by: Whether the SQL query contains ORDER BY clause.

    Returns:
        Properties including ordering guarantees and preferred boundary representation.
    """
    guarantees_order = has_explicit_order_by and len(region.sorted_by) > 0
    sorted_by = region.sorted_by if guarantees_order else ()

    return DuckDBRegionProperties(
        region_id=region.region_id,
        output_representation=region.representation,
        guarantees_order=guarantees_order,
        sorted_by=sorted_by,
        prefer_arrow_boundary=True,  # Section 21
    )


def optimize_duckdb_region_boundary(
    producer_region: BackendRegion,
    consumer_region: BackendRegion,
    producer_properties: DuckDBRegionProperties,
    consumer_requires_order: tuple[str, ...] = (),
    estimated_rows: int = 0,
) -> TransferEdge:
    """Create an optimized transfer edge from DuckDB Region to consumer.

    Section 21: Prefer DuckDB → Arrow boundary over DuckDB → Pandas.
    Section 22: Check if consumer needs ordering that producer doesn't provide,
    and set requires_sort=True with explicit sort cost.

    Args:
        producer_region: DuckDB Region producing data.
        consumer_region: Consuming region (any backend).
        producer_properties: DuckDB Region's output properties.
        consumer_requires_order: Order consumer needs (e.g., ('instrument', 'datetime')).
        estimated_rows: Row count for cost estimation.

    Returns:
        Transfer edge with optimized representation and explicit sort requirement.
    """
    # Section 21: DuckDB boundary optimization
    # Prefer Arrow if consumer can accept it, avoiding Pandas conversion
    consumer_backend = consumer_region.backend

    # Default: DuckDB → Arrow (preferred)
    source_repr = Representation.ARROW_TABLE

    # Choose target representation based on consumer backend
    if consumer_backend == PhysicalBackend.POLARS_LONG:
        target_repr = Representation.POLARS_LAZY_LONG  # Arrow → Polars is efficient
    elif consumer_backend == PhysicalBackend.POLARS_PANEL:
        target_repr = Representation.POLARS_LONG  # Arrow → Polars
    elif consumer_backend == PhysicalBackend.PANDAS_NUMPY:
        # Only convert to Pandas when consumer explicitly needs it
        target_repr = Representation.PANDAS_LONG
    elif consumer_backend in {PhysicalBackend.DUCKDB_SQL, PhysicalBackend.CLICKHOUSE_SQL}:
        # SQL → SQL: can stay in relational form
        source_repr = Representation.DUCKDB_RELATION
        target_repr = Representation.DUCKDB_RELATION
    else:
        target_repr = Representation.PANDAS_LONG

    # Section 22: Check if consumer needs ordering that producer doesn't provide
    requires_sort = False
    if consumer_requires_order:
        order_satisfied = check_producer_ordering(
            producer_properties.sorted_by,
            consumer_requires_order,
        )
        if not order_satisfied:
            # Consumer needs order that producer doesn't guarantee
            requires_sort = True

    # Calculate transfer cost including sort if needed
    cost_estimate = estimate_transfer_edge_cost(
        source_backend=producer_region.backend,
        target_backend=consumer_region.backend,
        source_repr=source_repr,
        target_repr=target_repr,
        rows=estimated_rows,
        requires_sort=requires_sort,
        requires_repartition=False,
        requires_reshape=False,
        requires_dtype_cast=False,
        sort_key_count=len(consumer_requires_order) if requires_sort else 0,
    )

    # Estimate bytes transferred (8 bytes per numeric cell)
    estimated_bytes = estimated_rows * 8 * 3  # timestamp, instrument, value

    # R21-TRANSFER-BOUNDARIES: Infer formal transform type
    transform = infer_transfer_transform(source_repr, target_repr)

    return TransferEdge(
        edge_id=f"{producer_region.region_id}→{consumer_region.region_id}",
        producer_region=producer_region.region_id,
        consumer_region=consumer_region.region_id,
        source_backend=producer_region.backend,
        target_backend=consumer_region.backend,
        source_representation=source_repr,
        target_representation=target_repr,
        transform=transform,  # R21-TRANSFER-BOUNDARIES: formal transform type
        estimated_rows=estimated_rows,
        estimated_bytes=estimated_bytes,
        estimated_transfer_ms=cost_estimate.total_ms,
        requires_sort=requires_sort,
        requires_repartition=False,
        requires_reshape=False,
        requires_dtype_cast=False,
        producer_sorted_by=producer_properties.sorted_by,
        producer_guarantees_order=producer_properties.guarantees_order,
        preserves_pit=True,
        preserves_universe=True,
        preserves_grain=True,
    )


def should_keep_duckdb_relation(
    downstream_backends: list[PhysicalBackend],
) -> bool:
    """Decide if DuckDB should output Relation instead of converting to Pandas.

    Section 21: Keep DuckDB Relation / SQL within regions; only convert at
    boundary when necessary.

    Args:
        downstream_backends: List of backends that will consume this output.

    Returns:
        True if we should keep DuckDB Relation (avoid premature Pandas conversion).
    """
    if not downstream_backends:
        return False

    # If any downstream is SQL-based, keep Relation
    sql_backends = {PhysicalBackend.DUCKDB_SQL, PhysicalBackend.CLICKHOUSE_SQL}
    if any(b in sql_backends for b in downstream_backends):
        return True

    # If all downstream are Polars, prefer Arrow boundary (not Pandas)
    polars_backends = {PhysicalBackend.POLARS_LONG, PhysicalBackend.POLARS_PANEL}
    if all(b in polars_backends for b in downstream_backends):
        return True

    # Mixed consumers or Pandas consumer: need conversion
    return False


def estimate_duckdb_to_arrow_cost(rows: int) -> float:
    """Estimate DuckDB Relation → Arrow conversion cost (milliseconds).

    Section 21: This conversion is cheap (DuckDB native Arrow support),
    much cheaper than DuckDB → Pandas → consumer roundtrip.
    """
    millions = max(rows / 1_000_000.0, 0.001)
    # DuckDB → Arrow is efficient (native zero-copy or minimal conversion)
    return 2.0 * millions


def estimate_duckdb_to_pandas_cost(rows: int) -> float:
    """Estimate DuckDB Relation → Pandas conversion cost (milliseconds).

    Section 21: This should be avoided when possible — prefer Arrow boundary.
    """
    millions = max(rows / 1_000_000.0, 0.001)
    # DuckDB → Pandas is more expensive (involves intermediate conversions)
    return 8.0 * millions
