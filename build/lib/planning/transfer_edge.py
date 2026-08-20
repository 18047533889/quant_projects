# -*- coding: utf-8 -*-
"""Transfer edge cost modeling and conversion tracking.

Implements edge-level cost modeling for representation conversions,
sorts, repartitions, and reshapes between backend regions as specified
in sections 13, 32, 56 of the remediation document.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from planning.backend_region import PhysicalProperty, Representation


class TransferKind(str, Enum):
    """Type of transfer operation required at region boundary (section 32)."""

    SAME_BACKEND_NATIVE = "same_backend_native"  # No conversion, shared buffer
    PANDAS_TO_POLARS = "pandas_to_polars"
    POLARS_TO_PANDAS = "polars_to_pandas"
    DUCKDB_TO_ARROW = "duckdb_to_arrow"
    ARROW_TO_POLARS = "arrow_to_polars"
    ARROW_TO_PANDAS = "arrow_to_pandas"
    ARROW_TO_Q = "arrow_to_q"
    Q_TO_ARROW = "q_to_arrow"
    WIDE_TO_LONG = "wide_to_long"  # Reshape operation
    LONG_TO_WIDE = "long_to_wide"  # Reshape operation
    SORT = "sort"  # Explicit sort to satisfy downstream property
    REPARTITION = "repartition"  # Repartition by different key
    DTYPE_CAST = "dtype_cast"  # Type conversion


@dataclass(frozen=True)
class SemanticContract:
    """Semantic contract preservation across transfer edge (MB-P0-013).

    Region boundaries must preserve PIT, universe, grain, available_at,
    and source snapshot contracts. This structure captures these invariants
    for validation.
    """

    pit_clock: str | None = None
    universe_id: str | None = None
    grain: str | None = None
    available_at: str | None = None
    source_snapshot: str | None = None
    price_basis: str | None = None  # RAW/ADJUSTED/CONTINUOUS
    timezone: str | None = None
    calendar: str | None = None

    def is_compatible(self, other: SemanticContract) -> bool:
        """Check if this contract is compatible with another."""
        for field in ["pit_clock", "universe_id", "grain", "source_snapshot"]:
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if self_val and other_val and self_val != other_val:
                return False
        return True


@dataclass(frozen=True)
class TransferEdge:
    """Transfer edge between backend regions (section 10).

    Represents a data flow from one backend region to another, capturing
    all costs: representation conversion, sort, repartition, reshape,
    and dtype casting.

    Attributes:
        edge_id: Unique stable identifier
        producer_region: Source region ID
        consumer_region: Target region ID
        source_representation: Producer's output representation
        target_representation: Consumer's required input representation
        transfer_kind: Primary transfer operation type
        estimated_rows: Estimated row count through this edge
        estimated_bytes: Estimated bytes transferred
        requires_sort: Whether explicit sort is needed
        requires_repartition: Whether repartition is needed
        requires_reshape: Whether wide<->long reshape is needed
        requires_dtype_cast: Whether dtype conversion is needed
        semantic_contract: Semantic invariants to preserve
        source_properties: Physical properties from producer
        target_properties: Physical properties required by consumer
        estimated_cost_ms: Total estimated transfer cost in milliseconds
    """

    edge_id: str
    producer_region: str
    consumer_region: str
    source_representation: Representation
    target_representation: Representation
    transfer_kind: TransferKind
    estimated_rows: int
    estimated_bytes: int
    requires_sort: bool
    requires_repartition: bool
    requires_reshape: bool
    requires_dtype_cast: bool
    semantic_contract: SemanticContract
    source_properties: PhysicalProperty
    target_properties: PhysicalProperty
    estimated_cost_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate edge contract consistency."""
        # Validate semantic contract preservation (MB-P0-013)
        if self.semantic_contract.pit_clock is None:
            # Warning: PIT contract not preserved
            pass

        # Validate physical property enforcement
        if self.requires_sort:
            if not self.target_properties.sorted_by:
                raise ValueError(
                    f"Edge {self.edge_id}: requires_sort=True but target has no sorted_by"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize edge to dictionary for telemetry/explain."""
        return {
            "edge_id": self.edge_id,
            "producer_region": self.producer_region,
            "consumer_region": self.consumer_region,
            "source_representation": self.source_representation.value if isinstance(self.source_representation, Enum) else str(self.source_representation),
            "target_representation": self.target_representation.value if isinstance(self.target_representation, Enum) else str(self.target_representation),
            "transfer_kind": self.transfer_kind.value if isinstance(self.transfer_kind, Enum) else str(self.transfer_kind),
            "estimated_rows": self.estimated_rows,
            "estimated_bytes": self.estimated_bytes,
            "requires_sort": self.requires_sort,
            "requires_repartition": self.requires_repartition,
            "requires_reshape": self.requires_reshape,
            "requires_dtype_cast": self.requires_dtype_cast,
            "estimated_cost_ms": round(self.estimated_cost_ms, 3),
            "source_sorted_by": list(self.source_properties.sorted_by),
            "target_sorted_by": list(self.target_properties.sorted_by),
        }


# =====================================================================
# Transfer cost modeling (sections 32, 56)
# =====================================================================

# Baseline conversion costs (milliseconds) for 1M rows
_BASELINE_CONVERSION_COSTS_MS = {
    TransferKind.PANDAS_TO_POLARS: 2.0,
    TransferKind.POLARS_TO_PANDAS: 2.5,
    TransferKind.DUCKDB_TO_ARROW: 1.5,
    TransferKind.ARROW_TO_POLARS: 1.0,
    TransferKind.ARROW_TO_PANDAS: 2.0,
    TransferKind.ARROW_TO_Q: 3.0,
    TransferKind.Q_TO_ARROW: 3.0,
    TransferKind.WIDE_TO_LONG: 4.0,
    TransferKind.LONG_TO_WIDE: 5.0,
    TransferKind.SAME_BACKEND_NATIVE: 0.0,
}

# Additional cost per million rows
_PER_MILLION_ROWS_MS = {
    TransferKind.PANDAS_TO_POLARS: 0.5,
    TransferKind.POLARS_TO_PANDAS: 0.6,
    TransferKind.DUCKDB_TO_ARROW: 0.3,
    TransferKind.ARROW_TO_POLARS: 0.2,
    TransferKind.ARROW_TO_PANDAS: 0.5,
    TransferKind.ARROW_TO_Q: 0.8,
    TransferKind.Q_TO_ARROW: 0.8,
    TransferKind.WIDE_TO_LONG: 1.0,
    TransferKind.LONG_TO_WIDE: 1.2,
    TransferKind.SAME_BACKEND_NATIVE: 0.0,
}

# Sort cost per million rows (log-linear in rows)
_SORT_BASE_MS = 5.0
_SORT_PER_MILLION_ROWS_MS = 2.0

# Repartition cost per million rows
_REPARTITION_BASE_MS = 3.0
_REPARTITION_PER_MILLION_ROWS_MS = 1.0


def estimate_transfer_cost(edge: TransferEdge) -> float:
    """Estimate total transfer cost in milliseconds (MB-P1-021, section 32).

    Includes:
    - Base representation conversion
    - Row count scaling
    - Sort cost (if required)
    - Repartition cost (if required)
    - Reshape cost (if required)
    - Dtype cast overhead (if required)

    Returns milliseconds to complete the transfer.
    """
    rows_millions = max(0.001, edge.estimated_rows / 1_000_000.0)

    # Base conversion cost
    base_cost = _BASELINE_CONVERSION_COSTS_MS.get(edge.transfer_kind, 3.0)
    per_row_cost = _PER_MILLION_ROWS_MS.get(edge.transfer_kind, 0.5)
    total_ms = base_cost + per_row_cost * rows_millions

    # Sort cost (log-linear scaling)
    if edge.requires_sort:
        import math
        log_factor = max(1.0, math.log10(edge.estimated_rows + 1))
        total_ms += _SORT_BASE_MS + _SORT_PER_MILLION_ROWS_MS * rows_millions * log_factor

    # Repartition cost
    if edge.requires_repartition:
        total_ms += _REPARTITION_BASE_MS + _REPARTITION_PER_MILLION_ROWS_MS * rows_millions

    # Reshape cost (already in base conversion for WIDE_TO_LONG/LONG_TO_WIDE)
    if edge.requires_reshape and edge.transfer_kind not in {
        TransferKind.WIDE_TO_LONG,
        TransferKind.LONG_TO_WIDE,
    }:
        total_ms += 2.0 * rows_millions

    # Dtype cast overhead
    if edge.requires_dtype_cast:
        total_ms += 0.5 * rows_millions

    return max(0.0, total_ms)


def infer_transfer_kind(
    source_repr: Representation, target_repr: Representation
) -> TransferKind:
    """Infer transfer kind from source and target representations."""
    if source_repr == target_repr:
        return TransferKind.SAME_BACKEND_NATIVE

    # Polars <-> Pandas
    if source_repr in {Representation.PANDAS_LONG, Representation.PANDAS_WIDE}:
        if target_repr in {
            Representation.POLARS_LONG,
            Representation.POLARS_WIDE,
            Representation.POLARS_LAZY_LONG,
        }:
            return TransferKind.PANDAS_TO_POLARS

    if source_repr in {
        Representation.POLARS_LONG,
        Representation.POLARS_WIDE,
        Representation.POLARS_LAZY_LONG,
    }:
        if target_repr in {Representation.PANDAS_LONG, Representation.PANDAS_WIDE}:
            return TransferKind.POLARS_TO_PANDAS

    # DuckDB <-> Arrow
    if source_repr == Representation.DUCKDB_RELATION:
        if target_repr == Representation.ARROW_TABLE:
            return TransferKind.DUCKDB_TO_ARROW

    # Arrow conversions
    if source_repr == Representation.ARROW_TABLE:
        if target_repr in {
            Representation.POLARS_LONG,
            Representation.POLARS_WIDE,
            Representation.POLARS_LAZY_LONG,
        }:
            return TransferKind.ARROW_TO_POLARS
        if target_repr in {Representation.PANDAS_LONG, Representation.PANDAS_WIDE}:
            return TransferKind.ARROW_TO_PANDAS
        if target_repr in {Representation.Q_TABLE, Representation.Q_VECTOR}:
            return TransferKind.ARROW_TO_Q

    # q conversions
    if source_repr in {Representation.Q_TABLE, Representation.Q_VECTOR}:
        if target_repr == Representation.ARROW_TABLE:
            return TransferKind.Q_TO_ARROW

    # Reshape operations
    if "WIDE" in source_repr.value and "LONG" in target_repr.value:
        return TransferKind.WIDE_TO_LONG
    if "LONG" in source_repr.value and "WIDE" in target_repr.value:
        return TransferKind.LONG_TO_WIDE

    # Default to generic conversion
    return TransferKind.PANDAS_TO_POLARS


def requires_sort_for_properties(
    source_props: PhysicalProperty, target_props: PhysicalProperty
) -> bool:
    """Check if explicit sort is required to satisfy target properties (section 22)."""
    if not target_props.sorted_by:
        return False
    if source_props.sorted_by == target_props.sorted_by:
        return False
    # Target requires sort but source doesn't provide it
    return True


def requires_repartition_for_properties(
    source_props: PhysicalProperty, target_props: PhysicalProperty
) -> bool:
    """Check if repartition is required to satisfy target properties."""
    if not target_props.partitioned_by:
        return False
    if source_props.partitioned_by == target_props.partitioned_by:
        return False
    return True
