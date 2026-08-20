# -*- coding: utf-8 -*-
"""Backend region contracts and representation types.

Implements typed contracts for backend regions, representation types, and
physical backends as specified in sections 10-11 of the remediation document.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PhysicalBackend(str, Enum):
    """Physical execution backend identity (MB-P2-002)."""

    PANDAS_NUMPY = "pandas_numpy"
    POLARS_LAZY = "polars_lazy"
    POLARS_EAGER = "polars_eager"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_TABLE = "q_table"
    ARROW_COMPUTE = "arrow_compute"
    NUMBA_CPU = "numba_cpu"  # Numba JIT compiled CPU kernels


class Representation(str, Enum):
    """Physical data representation (MB-P2-004, sections 33-34).

    Each representation has distinct memory layout, API surface, and
    conversion costs. The planner must track representation explicitly
    across region boundaries.
    """

    PANDAS_LONG = "pandas_long"  # (datetime, instrument, value) long format
    PANDAS_WIDE = "pandas_wide"  # datetime index, instrument columns
    NUMPY_PANEL = "numpy_panel"  # 3D array (date, instrument, feature)
    POLARS_LONG = "polars_long"  # Polars DataFrame in long format
    POLARS_WIDE = "polars_wide"  # Polars DataFrame in wide format
    POLARS_LAZY_LONG = "polars_lazy_long"  # Polars LazyFrame (not materialized)
    ARROW_TABLE = "arrow_table"  # PyArrow Table
    DUCKDB_RELATION = "duckdb_relation"  # DuckDB Relation object
    Q_TABLE = "q_table"  # q/K table
    Q_VECTOR = "q_vector"  # q/K vector


class ExecutionAxis(str, Enum):
    """Execution axis classification (MB-P2-003, section 36).

    Determines valid sharding strategies and barrier requirements.
    """

    TIME_PER_INSTRUMENT = "time_per_instrument"  # Rolling, EMA, time-series ops
    CROSS_SECTION_PER_DATE = "cross_section_per_date"  # Rank, zscore, neutralize
    GROUP_PER_DATE = "group_per_date"  # Grouped aggregations
    GLOBAL_PANEL = "global_panel"  # Full panel operations
    EVENT_STREAM = "event_stream"  # Event-driven processing
    RELATIONAL = "relational"  # Pure relational algebra
    RECURSIVE_TIME_PER_INSTRUMENT = "recursive_time_per_instrument"  # Stateful recursion


@dataclass(frozen=True)
class PhysicalProperty:
    """Physical properties of a region's output (section 35).

    These properties must be maintained or explicitly enforced at
    region boundaries. Missing properties trigger explicit sort/
    repartition operations.
    """

    sorted_by: tuple[str, ...] = ()
    partitioned_by: tuple[str, ...] = ()
    grouped_by: tuple[str, ...] = ()
    unique_key: tuple[str, ...] = ()
    grain: str | None = None


@dataclass(frozen=True)
class StateContract:
    """State management contract for stateful operators (section 40).

    Recursive operators (EMA, Kalman, deadband) require explicit
    state continuation contracts across time shards.
    """

    has_state: bool = False
    checkpoint_capable: bool = False
    seed_required: bool = False
    sequential_only: bool = False
    state_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class BackendRegion:
    """A contiguous subgraph assigned to a single backend (section 10).

    Represents the fundamental unit of multi-backend execution planning.
    Each region executes entirely within one backend without internal
    representation conversions.

    Attributes:
        region_id: Unique stable identifier for this region
        backend: Physical backend assigned to execute this region
        representation: Output data representation
        node_ids: Logical plan nodes assigned to this region
        execution_axis: Execution axis classification
        required_properties: Physical properties required at region output
        state_contract: State management contract (for stateful regions)
        estimated_rows: Estimated output row count
        estimated_bytes: Estimated memory footprint
        can_stream: Whether this region supports streaming execution
        requires_global_sort: Whether this region requires full dataset sort
        requires_full_group: Whether grouping requires all data in memory
    """

    region_id: str
    backend: PhysicalBackend
    representation: Representation
    node_ids: tuple[str, ...]
    execution_axis: ExecutionAxis
    required_properties: PhysicalProperty
    state_contract: StateContract
    estimated_rows: int = 0
    estimated_bytes: int = 0
    can_stream: bool = False
    requires_global_sort: bool = False
    requires_full_group: bool = False

    def __post_init__(self) -> None:
        """Validate region contract consistency."""
        # Cross-sectional operations cannot be asset-sharded (MB-P0-011)
        if self.execution_axis == ExecutionAxis.CROSS_SECTION_PER_DATE:
            if "instrument" in self.required_properties.partitioned_by:
                raise ValueError(
                    f"Region {self.region_id}: CROSS_SECTION_PER_DATE cannot be "
                    "partitioned by instrument (violates MB-P0-011)"
                )

        # Recursive stateful operations require sequential execution without checkpoint (MB-P0-012)
        if self.execution_axis == ExecutionAxis.RECURSIVE_TIME_PER_INSTRUMENT:
            if not self.state_contract.checkpoint_capable and not self.state_contract.sequential_only:
                raise ValueError(
                    f"Region {self.region_id}: RECURSIVE_TIME without checkpoint "
                    "must be sequential_only=True (violates MB-P0-012)"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize region to dictionary for telemetry/explain."""
        return {
            "region_id": self.region_id,
            "backend": self.backend.value if isinstance(self.backend, Enum) else str(self.backend),
            "representation": self.representation.value if isinstance(self.representation, Enum) else str(self.representation),
            "node_count": len(self.node_ids),
            "execution_axis": self.execution_axis.value if isinstance(self.execution_axis, Enum) else str(self.execution_axis),
            "estimated_rows": self.estimated_rows,
            "estimated_bytes": self.estimated_bytes,
            "can_stream": self.can_stream,
            "requires_global_sort": self.requires_global_sort,
            "requires_full_group": self.requires_full_group,
            "sorted_by": list(self.required_properties.sorted_by),
            "partitioned_by": list(self.required_properties.partitioned_by),
            "has_state": self.state_contract.has_state,
        }
