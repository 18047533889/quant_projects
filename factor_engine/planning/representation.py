# -*- coding: utf-8 -*-
"""Representation: First-class physical data representation types.

Representation defines the in-memory format of data buffers:
- PANDAS_LONG: Pandas DataFrame in long format (datetime, instrument, columns)
- PANDAS_WIDE: Pandas DataFrame in wide format (datetime as index, instruments as columns)
- POLARS_LAZY_LONG: Polars LazyFrame in long format (deferred execution)
- POLARS_EAGER_LONG: Polars DataFrame in long format (materialized)
- POLARS_LAZY_WIDE: Polars LazyFrame in wide format
- POLARS_EAGER_WIDE: Polars DataFrame in wide format
- ARROW_TABLE: PyArrow Table (columnar, zero-copy capable)
- DUCKDB_RELATION: DuckDB Relation (lazy SQL execution)
- Q_TABLE: q/K table
- Q_KEYED_TABLE: q/K keyed table

Transfer costs and conversion logic depend on source and target representations.
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Literal


class Representation(Enum):
    """Physical data representation in memory.

    Each representation has different characteristics:
    - Memory layout (row vs columnar)
    - Execution model (eager vs lazy)
    - Zero-copy capabilities
    - Backend affinity
    - Conversion costs
    """

    # Pandas representations
    PANDAS_LONG = "PANDAS_LONG"  # (datetime, instrument, ...) row-oriented
    PANDAS_WIDE = "PANDAS_WIDE"  # datetime index, instrument columns

    # Polars representations
    POLARS_LAZY_LONG = "POLARS_LAZY_LONG"  # LazyFrame, long format, deferred
    POLARS_EAGER_LONG = "POLARS_EAGER_LONG"  # DataFrame, long format, materialized
    POLARS_LAZY_WIDE = "POLARS_LAZY_WIDE"  # LazyFrame, wide format, deferred
    POLARS_EAGER_WIDE = "POLARS_EAGER_WIDE"  # DataFrame, wide format, materialized

    # Arrow representation
    ARROW_TABLE = "ARROW_TABLE"  # PyArrow Table, columnar, zero-copy capable

    # DuckDB representation
    DUCKDB_RELATION = "DUCKDB_RELATION"  # DuckDB Relation, lazy SQL

    # q/K representations
    Q_TABLE = "Q_TABLE"  # q table (unkeyed)
    Q_KEYED_TABLE = "Q_KEYED_TABLE"  # q keyed table
    Q_DICTIONARY = "Q_DICTIONARY"  # q dictionary


@dataclass(frozen=True)
class RepresentationCharacteristics:
    """Characteristics of a physical representation."""

    representation: Representation

    # Backend affinity
    native_backend: str  # "pandas" | "polars" | "duckdb" | "q"

    # Execution model
    is_lazy: bool  # Deferred execution (LazyFrame, Relation)
    is_eager: bool  # Materialized in memory

    # Layout
    is_columnar: bool  # Columnar storage (Arrow, DuckDB)
    is_row_oriented: bool  # Row-oriented (some Pandas operations)

    # Format
    is_long: bool  # Long format (datetime, instrument, value)
    is_wide: bool  # Wide format (datetime × instruments)

    # Zero-copy capabilities
    supports_zero_copy_to_arrow: bool
    supports_zero_copy_from_arrow: bool

    # Memory characteristics
    typical_memory_overhead: float  # Multiplier vs raw data (1.0 = no overhead)


# Representation characteristics lookup
REPRESENTATION_CHARACTERISTICS = {
    Representation.PANDAS_LONG: RepresentationCharacteristics(
        representation=Representation.PANDAS_LONG,
        native_backend="pandas",
        is_lazy=False,
        is_eager=True,
        is_columnar=False,
        is_row_oriented=True,
        is_long=True,
        is_wide=False,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=1.5,  # Pandas index + object overhead
    ),
    Representation.PANDAS_WIDE: RepresentationCharacteristics(
        representation=Representation.PANDAS_WIDE,
        native_backend="pandas",
        is_lazy=False,
        is_eager=True,
        is_columnar=False,
        is_row_oriented=True,
        is_long=False,
        is_wide=True,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=1.6,  # Pandas wide has larger index
    ),
    Representation.POLARS_LAZY_LONG: RepresentationCharacteristics(
        representation=Representation.POLARS_LAZY_LONG,
        native_backend="polars",
        is_lazy=True,
        is_eager=False,
        is_columnar=True,
        is_row_oriented=False,
        is_long=True,
        is_wide=False,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=0.1,  # Lazy, minimal memory until collect
    ),
    Representation.POLARS_EAGER_LONG: RepresentationCharacteristics(
        representation=Representation.POLARS_EAGER_LONG,
        native_backend="polars",
        is_lazy=False,
        is_eager=True,
        is_columnar=True,
        is_row_oriented=False,
        is_long=True,
        is_wide=False,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=1.1,  # Columnar, efficient
    ),
    Representation.POLARS_LAZY_WIDE: RepresentationCharacteristics(
        representation=Representation.POLARS_LAZY_WIDE,
        native_backend="polars",
        is_lazy=True,
        is_eager=False,
        is_columnar=True,
        is_row_oriented=False,
        is_long=False,
        is_wide=True,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=0.1,
    ),
    Representation.POLARS_EAGER_WIDE: RepresentationCharacteristics(
        representation=Representation.POLARS_EAGER_WIDE,
        native_backend="polars",
        is_lazy=False,
        is_eager=True,
        is_columnar=True,
        is_row_oriented=False,
        is_long=False,
        is_wide=True,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=1.2,
    ),
    Representation.ARROW_TABLE: RepresentationCharacteristics(
        representation=Representation.ARROW_TABLE,
        native_backend="arrow",
        is_lazy=False,
        is_eager=True,
        is_columnar=True,
        is_row_oriented=False,
        is_long=True,  # Typically long format
        is_wide=False,
        supports_zero_copy_to_arrow=True,  # Already Arrow
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=1.0,  # Minimal overhead
    ),
    Representation.DUCKDB_RELATION: RepresentationCharacteristics(
        representation=Representation.DUCKDB_RELATION,
        native_backend="duckdb",
        is_lazy=True,
        is_eager=False,
        is_columnar=True,
        is_row_oriented=False,
        is_long=True,
        is_wide=False,
        supports_zero_copy_to_arrow=True,
        supports_zero_copy_from_arrow=True,
        typical_memory_overhead=0.1,  # Lazy relation
    ),
    Representation.Q_TABLE: RepresentationCharacteristics(
        representation=Representation.Q_TABLE,
        native_backend="q",
        is_lazy=False,
        is_eager=True,
        is_columnar=True,
        is_row_oriented=False,
        is_long=True,
        is_wide=False,
        supports_zero_copy_to_arrow=False,  # Requires conversion
        supports_zero_copy_from_arrow=False,
        typical_memory_overhead=1.0,
    ),
    Representation.Q_KEYED_TABLE: RepresentationCharacteristics(
        representation=Representation.Q_KEYED_TABLE,
        native_backend="q",
        is_lazy=False,
        is_eager=True,
        is_columnar=True,
        is_row_oriented=False,
        is_long=True,
        is_wide=False,
        supports_zero_copy_to_arrow=False,
        supports_zero_copy_from_arrow=False,
        typical_memory_overhead=1.1,
    ),
    Representation.Q_DICTIONARY: RepresentationCharacteristics(
        representation=Representation.Q_DICTIONARY,
        native_backend="q",
        is_lazy=False,
        is_eager=True,
        is_columnar=False,
        is_row_oriented=False,
        is_long=False,
        is_wide=False,
        supports_zero_copy_to_arrow=False,
        supports_zero_copy_from_arrow=False,
        typical_memory_overhead=1.2,
    ),
}


def is_compatible_backend(representation: Representation, backend: str) -> bool:
    """Check if a representation is compatible with a backend."""
    chars = REPRESENTATION_CHARACTERISTICS[representation]
    return chars.native_backend == backend


def requires_materialization(representation: Representation) -> bool:
    """Check if a representation requires materialization (lazy -> eager)."""
    chars = REPRESENTATION_CHARACTERISTICS[representation]
    return chars.is_lazy


def supports_streaming(representation: Representation) -> bool:
    """Check if a representation supports streaming execution."""
    chars = REPRESENTATION_CHARACTERISTICS[representation]
    # Lazy representations and some eager columnar formats support streaming
    return chars.is_lazy or (chars.is_columnar and representation in {
        Representation.ARROW_TABLE,
        Representation.POLARS_EAGER_LONG,
    })


def get_layout(representation: Representation) -> Literal["long", "wide", "other"]:
    """Get the layout (long/wide) of a representation."""
    chars = REPRESENTATION_CHARACTERISTICS[representation]
    if chars.is_long:
        return "long"
    elif chars.is_wide:
        return "wide"
    else:
        return "other"


def estimate_conversion_cost(
    source: Representation,
    target: Representation,
    estimated_bytes: int,
) -> float:
    """Estimate conversion cost in milliseconds.

    Conversion costs depend on:
    - Zero-copy capability (Arrow interop)
    - Layout transformation (long <-> wide)
    - Materialization (lazy -> eager)
    - Backend boundary crossing
    """
    source_chars = REPRESENTATION_CHARACTERISTICS[source]
    target_chars = REPRESENTATION_CHARACTERISTICS[target]

    # Same representation: no cost
    if source == target:
        return 0.0

    # Zero-copy via Arrow: minimal cost
    if source_chars.supports_zero_copy_to_arrow and target_chars.supports_zero_copy_from_arrow:
        if source_chars.native_backend != target_chars.native_backend:
            # Backend boundary but zero-copy capable: ~5ms base + small per-byte cost
            return 5.0 + (estimated_bytes / 1_000_000_000) * 50  # 50ms per GB

    # Layout transformation (long <-> wide): expensive
    source_layout = get_layout(source)
    target_layout = get_layout(target)
    if source_layout != target_layout and source_layout != "other" and target_layout != "other":
        # Pivot/unpivot operation: ~100ms base + moderate per-byte cost
        reshape_cost = 100.0 + (estimated_bytes / 1_000_000_000) * 200  # 200ms per GB
    else:
        reshape_cost = 0.0

    # Materialization (lazy -> eager)
    if source_chars.is_lazy and target_chars.is_eager:
        materialization_cost = 10.0 + (estimated_bytes / 1_000_000_000) * 100  # 100ms per GB
    else:
        materialization_cost = 0.0

    # Backend conversion (non-zero-copy)
    if source_chars.native_backend != target_chars.native_backend:
        if not (source_chars.supports_zero_copy_to_arrow and target_chars.supports_zero_copy_from_arrow):
            # Full copy: significant cost
            conversion_cost = 20.0 + (estimated_bytes / 1_000_000_000) * 300  # 300ms per GB
        else:
            conversion_cost = 0.0  # Already counted in zero-copy path
    else:
        conversion_cost = 0.0

    # Total cost
    total = reshape_cost + materialization_cost + conversion_cost

    # If no special costs, use generic conversion
    if total == 0.0 and source != target:
        total = 10.0 + (estimated_bytes / 1_000_000_000) * 100  # Generic 100ms per GB

    return total


__all__ = [
    "Representation",
    "RepresentationCharacteristics",
    "REPRESENTATION_CHARACTERISTICS",
    "is_compatible_backend",
    "requires_materialization",
    "supports_streaming",
    "get_layout",
    "estimate_conversion_cost",
]
