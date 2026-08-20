# -*- coding: utf-8 -*-
"""PhysicalProperties: Physical data properties that affect execution planning.

Physical properties include:
- sorted_by: Column ordering guarantees
- partitioned_by: Data partitioning scheme
- grouped_by: Grouping structure (for aggregations)
- unique_key: Uniqueness guarantees

These properties:
1. Enable optimizations (avoid redundant sorts)
2. Enforce correctness (detect missing properties)
3. Propagate through region boundaries
4. Affect transfer costs (sort/repartition required?)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence


@dataclass(frozen=True)
class SortOrder:
    """Sort order specification for a column."""

    column: str
    ascending: bool = True
    nulls_first: bool = False

    def __str__(self) -> str:
        direction = "ASC" if self.ascending else "DESC"
        nulls = " NULLS FIRST" if self.nulls_first else ""
        return f"{self.column} {direction}{nulls}"


@dataclass(frozen=True)
class PhysicalProperties:
    """Physical properties of a data buffer.

    These properties describe the physical layout and guarantees of data:
    - sorted_by: Column ordering (enables merge joins, efficient lookups)
    - partitioned_by: Partitioning scheme (enables partition-aware operations)
    - grouped_by: Grouping structure (for grouped aggregations)
    - unique_key: Uniqueness constraint (enables optimizations, detects errors)

    Properties are preserved across region boundaries and enforced by the planner.
    """

    # Sort order: tuple of (column, ascending) pairs
    sorted_by: tuple[SortOrder, ...] = ()

    # Partitioning columns: data is partitioned by these columns
    partitioned_by: tuple[str, ...] = ()

    # Grouping columns: data is grouped (not necessarily sorted) by these columns
    grouped_by: tuple[str, ...] = ()

    # Unique key columns: these columns form a unique key
    unique_key: tuple[str, ...] = ()

    def __post_init__(self):
        """Validate physical properties."""
        # Check for duplicate columns in sort order
        sort_cols = [s.column for s in self.sorted_by]
        if len(sort_cols) != len(set(sort_cols)):
            raise ValueError(f"Duplicate columns in sorted_by: {sort_cols}")

        # Check for duplicate columns in partitioned_by
        if len(self.partitioned_by) != len(set(self.partitioned_by)):
            raise ValueError(f"Duplicate columns in partitioned_by: {self.partitioned_by}")

        # Check for duplicate columns in grouped_by
        if len(self.grouped_by) != len(set(self.grouped_by)):
            raise ValueError(f"Duplicate columns in grouped_by: {self.grouped_by}")

        # Check for duplicate columns in unique_key
        if len(self.unique_key) != len(set(self.unique_key)):
            raise ValueError(f"Duplicate columns in unique_key: {self.unique_key}")

    def is_sorted_by(self, columns: Sequence[str]) -> bool:
        """Check if data is sorted by the given columns (in order)."""
        if len(columns) == 0:
            return True
        if len(columns) > len(self.sorted_by):
            return False
        return all(
            sort_order.column == col
            for sort_order, col in zip(self.sorted_by, columns)
        )

    def is_partitioned_by(self, columns: Sequence[str]) -> bool:
        """Check if data is partitioned by the given columns."""
        return set(columns).issubset(set(self.partitioned_by))

    def is_grouped_by(self, columns: Sequence[str]) -> bool:
        """Check if data is grouped by the given columns."""
        return set(columns).issubset(set(self.grouped_by))

    def has_unique_key(self, columns: Sequence[str]) -> bool:
        """Check if the given columns form a unique key."""
        return tuple(columns) == self.unique_key

    def with_sort(self, *sort_orders: SortOrder) -> PhysicalProperties:
        """Return new properties with updated sort order."""
        return replace(self, sorted_by=tuple(sort_orders))

    def with_partition(self, *columns: str) -> PhysicalProperties:
        """Return new properties with updated partitioning."""
        return replace(self, partitioned_by=tuple(columns))

    def with_grouping(self, *columns: str) -> PhysicalProperties:
        """Return new properties with updated grouping."""
        return replace(self, grouped_by=tuple(columns))

    def with_unique_key(self, *columns: str) -> PhysicalProperties:
        """Return new properties with updated unique key."""
        return replace(self, unique_key=tuple(columns))

    def cleared(self) -> PhysicalProperties:
        """Return properties with all guarantees cleared."""
        return PhysicalProperties()

    def after_projection(self, retained_columns: set[str]) -> PhysicalProperties:
        """Return properties after a projection operation.

        Projection preserves properties if all relevant columns are retained.
        """
        # Sort order: preserve if all columns retained
        new_sorted_by = tuple(
            s for s in self.sorted_by if s.column in retained_columns
        )

        # Partition: preserve if all columns retained
        new_partitioned_by = tuple(
            c for c in self.partitioned_by if c in retained_columns
        )

        # Grouping: preserve if all columns retained
        new_grouped_by = tuple(
            c for c in self.grouped_by if c in retained_columns
        )

        # Unique key: preserve only if all columns retained
        new_unique_key = self.unique_key if all(
            c in retained_columns for c in self.unique_key
        ) else ()

        return PhysicalProperties(
            sorted_by=new_sorted_by,
            partitioned_by=new_partitioned_by,
            grouped_by=new_grouped_by,
            unique_key=new_unique_key,
        )

    def after_filter(self) -> PhysicalProperties:
        """Return properties after a filter operation.

        Filter preserves all properties (ordering, partitioning, grouping, uniqueness).
        """
        return self

    def after_sort(self, *sort_orders: SortOrder) -> PhysicalProperties:
        """Return properties after a sort operation."""
        return replace(self, sorted_by=tuple(sort_orders))

    def after_repartition(self, *columns: str) -> PhysicalProperties:
        """Return properties after repartitioning.

        Repartitioning clears sort order and grouping.
        """
        return PhysicalProperties(
            sorted_by=(),
            partitioned_by=tuple(columns),
            grouped_by=(),
            unique_key=self.unique_key if all(c in columns for c in self.unique_key) else (),
        )

    def after_group_by(self, *columns: str) -> PhysicalProperties:
        """Return properties after a grouping operation."""
        return PhysicalProperties(
            sorted_by=(),
            partitioned_by=self.partitioned_by,
            grouped_by=tuple(columns),
            unique_key=(),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        parts = []
        if self.sorted_by:
            sort_str = ", ".join(str(s) for s in self.sorted_by)
            parts.append(f"sorted_by=[{sort_str}]")
        if self.partitioned_by:
            parts.append(f"partitioned_by={list(self.partitioned_by)}")
        if self.grouped_by:
            parts.append(f"grouped_by={list(self.grouped_by)}")
        if self.unique_key:
            parts.append(f"unique_key={list(self.unique_key)}")
        return "PhysicalProperties(" + ", ".join(parts) + ")" if parts else "PhysicalProperties()"


@dataclass(frozen=True)
class PropertyRequirement:
    """A requirement for physical properties.

    The planner uses requirements to determine when property enforcement
    (sort, repartition, etc.) is needed.
    """

    # Required sort order (None = any order acceptable)
    required_sort: Optional[tuple[SortOrder, ...]] = None

    # Required partitioning (None = any partitioning acceptable)
    required_partition: Optional[tuple[str, ...]] = None

    # Required grouping (None = any grouping acceptable)
    required_grouping: Optional[tuple[str, ...]] = None

    # Required unique key (None = no uniqueness required)
    required_unique_key: Optional[tuple[str, ...]] = None

    def is_satisfied_by(self, properties: PhysicalProperties) -> bool:
        """Check if the given properties satisfy this requirement."""
        # Check sort order
        if self.required_sort is not None:
            if properties.sorted_by != self.required_sort:
                return False

        # Check partitioning
        if self.required_partition is not None:
            if not properties.is_partitioned_by(self.required_partition):
                return False

        # Check grouping
        if self.required_grouping is not None:
            if not properties.is_grouped_by(self.required_grouping):
                return False

        # Check unique key
        if self.required_unique_key is not None:
            if not properties.has_unique_key(self.required_unique_key):
                return False

        return True

    def estimate_enforcement_cost(
        self,
        current: PhysicalProperties,
        estimated_rows: int,
        estimated_bytes: int,
    ) -> float:
        """Estimate cost (in milliseconds) to enforce this requirement.

        Returns 0.0 if already satisfied.
        """
        if self.is_satisfied_by(current):
            return 0.0

        total_cost = 0.0

        # Sort cost: O(n log n) with constants
        if self.required_sort is not None and current.sorted_by != self.required_sort:
            # Base cost + per-row cost
            sort_cost = 50.0 + (estimated_rows / 1_000_000) * 500  # 500ms per million rows
            total_cost += sort_cost

        # Repartition cost: O(n) shuffle
        if self.required_partition is not None and not current.is_partitioned_by(self.required_partition):
            # Shuffle cost
            repartition_cost = 30.0 + (estimated_bytes / 1_000_000_000) * 200  # 200ms per GB
            total_cost += repartition_cost

        # Grouping cost: O(n) with hash
        if self.required_grouping is not None and not current.is_grouped_by(self.required_grouping):
            # Hash-based grouping
            grouping_cost = 20.0 + (estimated_rows / 1_000_000) * 100  # 100ms per million rows
            total_cost += grouping_cost

        # Unique key validation: O(n) with hash
        if self.required_unique_key is not None and not current.has_unique_key(self.required_unique_key):
            # Hash-based uniqueness check
            unique_cost = 15.0 + (estimated_rows / 1_000_000) * 80  # 80ms per million rows
            total_cost += unique_cost

        return total_cost


# Common property patterns for factor computations
COMMON_PROPERTIES = {
    # Time-series data: sorted by (instrument, datetime)
    "timeseries": PhysicalProperties(
        sorted_by=(
            SortOrder("instrument", ascending=True),
            SortOrder("datetime", ascending=True),
        ),
        unique_key=("instrument", "datetime"),
    ),

    # Cross-sectional data: sorted by (datetime, instrument)
    "cross_section": PhysicalProperties(
        sorted_by=(
            SortOrder("datetime", ascending=True),
            SortOrder("instrument", ascending=True),
        ),
        unique_key=("datetime", "instrument"),
    ),

    # Partitioned by instrument (for parallel time-series)
    "partitioned_by_instrument": PhysicalProperties(
        partitioned_by=("instrument",),
    ),

    # Partitioned by date (for parallel cross-sections)
    "partitioned_by_date": PhysicalProperties(
        partitioned_by=("datetime",),
    ),
}


__all__ = [
    "SortOrder",
    "PhysicalProperties",
    "PropertyRequirement",
    "COMMON_PROPERTIES",
]
