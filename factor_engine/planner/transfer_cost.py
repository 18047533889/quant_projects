# -*- coding: utf-8 -*-
"""Transfer edge cost calculation for multi-backend region plans.

MB-P1-021: Transfer cost must include conversion, sort, repartition, reshape,
and dtype cast operations. Section 22: SQL results may be unordered — time-series
downstream requiring sorted_by=(instrument, datetime) must check producer properties
and account for sort cost when needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from planner.backend_region import PhysicalBackend, Representation


@dataclass(frozen=True)
class TransferCostEstimate:
    """Detailed breakdown of transfer edge cost components.

    MB-P1-021: Each component is estimated separately and summed to avoid
    double-counting between different cost models.
    """

    conversion_ms: float = 0.0
    sort_ms: float = 0.0
    repartition_ms: float = 0.0
    reshape_ms: float = 0.0
    dtype_cast_ms: float = 0.0
    total_ms: float = 0.0


# MB-P1-021: Conversion costs by representation pair (milliseconds per million rows)
_CONVERSION_COST_MS_PER_MILLION: dict[tuple[str, str], float] = {
    # DuckDB → Arrow (Section 21: preferred boundary)
    ("duckdb_relation", "arrow_table"): 2.0,
    # DuckDB → Pandas (Section 21: avoid when possible)
    ("duckdb_relation", "pandas_long"): 8.0,
    ("duckdb_relation", "pandas_wide"): 10.0,
    # Arrow → Polars (efficient zero-copy or minimal conversion)
    ("arrow_table", "polars_long"): 1.5,
    ("arrow_table", "polars_lazy_long"): 1.0,
    # Arrow → Pandas
    ("arrow_table", "pandas_long"): 5.0,
    ("arrow_table", "pandas_wide"): 7.0,
    # Polars → Pandas
    ("polars_long", "pandas_long"): 6.0,
    ("polars_lazy_long", "pandas_long"): 6.5,
    # Pandas → Polars
    ("pandas_long", "polars_long"): 7.0,
    ("pandas_long", "polars_lazy_long"): 7.5,
    # Wide ↔ Long reshapes (more expensive)
    ("pandas_wide", "pandas_long"): 12.0,
    ("pandas_long", "pandas_wide"): 12.0,
    ("polars_wide", "polars_long"): 10.0,
    ("polars_long", "polars_wide"): 10.0,
}

# Section 22: Sort cost by backend (milliseconds per million rows per key column)
_SORT_COST_MS_PER_MILLION: dict[str, float] = {
    "pandas_numpy": 25.0,
    "polars_panel": 15.0,
    "polars_long": 15.0,
    "duckdb_sql": 10.0,
    "clickhouse_sql": 12.0,
}


def estimate_conversion_cost(
    source_repr: Representation,
    target_repr: Representation,
    rows: int,
) -> float:
    """Estimate representation conversion cost in milliseconds.

    MB-P1-021, Section 21: DuckDB → Arrow is cheap (preferred boundary);
    DuckDB → Pandas is more expensive and should be avoided when downstream
    can consume Arrow/Polars.
    """
    if source_repr == target_repr:
        return 0.0

    key = (source_repr.value, target_repr.value)
    cost_per_million = _CONVERSION_COST_MS_PER_MILLION.get(key)

    if cost_per_million is None:
        # Fallback: generic conversion estimate
        cost_per_million = 8.0

    millions = max(rows / 1_000_000.0, 0.001)
    return cost_per_million * millions


def estimate_sort_cost(
    backend: PhysicalBackend,
    rows: int,
    sort_key_count: int = 2,
) -> float:
    """Estimate sort operation cost in milliseconds.

    Section 22: SQL ordering cannot be assumed. Time-series downstream requiring
    sorted_by=(instrument, datetime) must check producer properties; if producer
    lacks the ordering guarantee, TransferEdge.requires_sort=True and this cost
    is added to the transfer edge.
    """
    if sort_key_count <= 0 or rows <= 0:
        return 0.0

    cost_per_million = _SORT_COST_MS_PER_MILLION.get(backend.value, 20.0)
    millions = max(rows / 1_000_000.0, 0.001)
    # Cost scales roughly linearly with row count and key count
    return cost_per_million * millions * sort_key_count


def estimate_repartition_cost(rows: int, partition_count: int) -> float:
    """Estimate repartition/reshuffle cost in milliseconds.

    MB-P1-021: Cross-region repartition is expensive and should be included
    in transfer edge cost.
    """
    if partition_count <= 1 or rows <= 0:
        return 0.0

    # Base cost + per-row shuffling overhead
    millions = max(rows / 1_000_000.0, 0.001)
    return 10.0 + 5.0 * millions


def estimate_reshape_cost(rows: int, columns: int = 1) -> float:
    """Estimate wide ↔ long reshape cost in milliseconds.

    MB-P1-021: Pivot/unpivot operations are expensive transformations.
    """
    if rows <= 0:
        return 0.0

    millions = max(rows / 1_000_000.0, 0.001)
    col_factor = max(columns, 1)
    # Reshape cost scales with both rows and columns
    return 8.0 + 4.0 * millions * col_factor


def estimate_dtype_cast_cost(rows: int, column_count: int = 1) -> float:
    """Estimate dtype casting cost in milliseconds.

    MB-P1-021: Type conversions (e.g., float64 → float32) have measurable cost.
    """
    if rows <= 0 or column_count <= 0:
        return 0.0

    millions = max(rows / 1_000_000.0, 0.001)
    return 1.0 + 0.5 * millions * column_count


def estimate_transfer_edge_cost(
    source_backend: PhysicalBackend,
    target_backend: PhysicalBackend,
    source_repr: Representation,
    target_repr: Representation,
    rows: int,
    requires_sort: bool = False,
    requires_repartition: bool = False,
    requires_reshape: bool = False,
    requires_dtype_cast: bool = False,
    sort_key_count: int = 2,
    partition_count: int = 1,
    columns: int = 1,
) -> TransferCostEstimate:
    """Calculate complete transfer edge cost with all components.

    MB-P1-021: Transfer cost = conversion + sort + repartition + reshape + cast.
    Each component is calculated separately and summed exactly once to avoid
    double-counting with other cost models.

    Section 22: If requires_sort=True (because producer doesn't guarantee order
    and consumer needs sorted data), sort cost is included here.
    """
    conversion_ms = estimate_conversion_cost(source_repr, target_repr, rows)

    sort_ms = 0.0
    if requires_sort:
        # Section 22: Sort cost added when consumer needs ordering that producer
        # doesn't guarantee (common with SQL results)
        sort_ms = estimate_sort_cost(target_backend, rows, sort_key_count)

    repartition_ms = 0.0
    if requires_repartition:
        repartition_ms = estimate_repartition_cost(rows, partition_count)

    reshape_ms = 0.0
    if requires_reshape:
        reshape_ms = estimate_reshape_cost(rows, columns)

    dtype_cast_ms = 0.0
    if requires_dtype_cast:
        dtype_cast_ms = estimate_dtype_cast_cost(rows, columns)

    total_ms = conversion_ms + sort_ms + repartition_ms + reshape_ms + dtype_cast_ms

    return TransferCostEstimate(
        conversion_ms=conversion_ms,
        sort_ms=sort_ms,
        repartition_ms=repartition_ms,
        reshape_ms=reshape_ms,
        dtype_cast_ms=dtype_cast_ms,
        total_ms=total_ms,
    )


def check_producer_ordering(
    producer_sorted_by: tuple[str, ...],
    consumer_required_order: tuple[str, ...],
) -> bool:
    """Check if producer's output ordering satisfies consumer's requirement.

    Section 22: Time-series downstream requiring sorted_by=(instrument, datetime)
    must check producer properties. If producer doesn't guarantee the order,
    consumer must set requires_sort=True on the TransferEdge.

    Returns:
        True if producer ordering satisfies consumer (no sort needed).
        False if consumer needs to explicitly sort (requires_sort=True).
    """
    if not consumer_required_order:
        # Consumer doesn't require specific order
        return True

    if not producer_sorted_by:
        # Producer doesn't guarantee any order, consumer needs sorting
        return False

    # Check if producer's order is a prefix of consumer's required order
    # E.g., producer=(instrument, datetime) satisfies consumer=(instrument, datetime)
    # but producer=(datetime) doesn't satisfy consumer=(instrument, datetime)
    if len(producer_sorted_by) < len(consumer_required_order):
        return False

    for i, col in enumerate(consumer_required_order):
        if i >= len(producer_sorted_by) or producer_sorted_by[i] != col:
            return False

    return True
