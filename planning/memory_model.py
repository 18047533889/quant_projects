# -*- coding: utf-8 -*-
"""MB-P1-004/007/021: Fine-grained memory cost model.

Implements precise memory estimation accounting for:
- Live columns, dtype, sort, hash, conversion overlap (MB-P1-004)
- Edge-specific transfer costs: sort/repartition/cast (MB-P1-007/021)
- Representation-specific buffer sizes (MB-P1-008)
- Liveness/refcount for early free (MB-P1-014)
- Streaming-first for low-memory servers (MB-P1-016)

Design:
    - DataShapeEstimate: metadata-only shape (rows/columns/bytes) (§27)
    - MemoryCost: per-operation memory breakdown
    - EdgeMemoryCost: boundary overlap (source + target + scratch)
    - LivenessTracker: refcount-driven early release (§46)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Representation(Enum):
    """Physical representation (MB-P2-004, §33)."""

    PANDAS_LONG = "pandas_long"
    PANDAS_WIDE = "pandas_wide"
    NUMPY_PANEL = "numpy_panel"
    POLARS_LONG = "polars_long"
    POLARS_WIDE = "polars_wide"
    POLARS_LAZY_LONG = "polars_lazy_long"
    ARROW_TABLE = "arrow_table"
    DUCKDB_RELATION = "duckdb_relation"
    Q_TABLE = "q_table"
    Q_VECTOR = "q_vector"


@dataclass(frozen=True)
class DataShapeEstimate:
    """MB-P1-002, §27: Metadata-only shape estimate (no data loading).

    Sources:
        - Parquet metadata (row groups, stats)
        - DataAccess manifest
        - Schema
        - Universe metadata
        - Calendar sessions
    """

    estimated_rows: int
    estimated_dates: int
    estimated_instruments: int
    estimated_columns: int
    estimated_bytes: int
    average_row_width_bytes: float
    density: float  # sparsity: 0.0-1.0
    frequency: str  # daily, minute, etc.
    bars_per_session: int | None = None
    group_count: int | None = None
    remote: bool = False
    storage_kind: str = "parquet"
    sorted_by: tuple[str, ...] = ()
    partition_by: tuple[str, ...] = ()
    projected_columns: tuple[str, ...] = ()
    rows_known: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_rows": self.estimated_rows,
            "estimated_dates": self.estimated_dates,
            "estimated_instruments": self.estimated_instruments,
            "estimated_columns": self.estimated_columns,
            "estimated_bytes": self.estimated_bytes,
            "average_row_width_bytes": round(self.average_row_width_bytes, 2),
            "density": round(self.density, 3),
            "frequency": self.frequency,
            "bars_per_session": self.bars_per_session,
            "group_count": self.group_count,
            "remote": self.remote,
            "storage_kind": self.storage_kind,
            "sorted_by": self.sorted_by,
            "partition_by": self.partition_by,
            "projected_columns": self.projected_columns,
            "rows_known": self.rows_known,
        }


@dataclass(frozen=True)
class MemoryCost:
    """MB-P1-004: Fine-grained memory cost per operation.

    Accounts for:
        - Live columns (not all columns always in memory)
        - Dtype-specific sizes (float64=8, float32=4, int32=4, etc.)
        - Sort working memory
        - Hash table memory
        - Conversion scratch space
    """

    base_input_bytes: int
    live_columns_bytes: int
    sort_working_bytes: int
    hash_table_bytes: int
    output_bytes: int
    scratch_bytes: int
    peak_bytes: int  # max(inputs + working + output)
    dtype_breakdown: dict[str, int]  # column -> bytes
    requires_sort: bool = False
    requires_hash: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_input_bytes": self.base_input_bytes,
            "live_columns_bytes": self.live_columns_bytes,
            "sort_working_bytes": self.sort_working_bytes,
            "hash_table_bytes": self.hash_table_bytes,
            "output_bytes": self.output_bytes,
            "scratch_bytes": self.scratch_bytes,
            "peak_bytes": self.peak_bytes,
            "dtype_breakdown": self.dtype_breakdown,
            "requires_sort": self.requires_sort,
            "requires_hash": self.requires_hash,
        }


@dataclass(frozen=True)
class EdgeMemoryCost:
    """MB-P1-007/021, §47: Transfer edge memory cost.

    Boundary overlap: source + target + scratch simultaneously live.
    Edge types: sort/repartition/cast/reshape each have specific costs.
    """

    edge_id: str
    source_representation: str
    target_representation: str
    source_bytes: int
    target_bytes: int
    scratch_bytes: int
    peak_boundary_bytes: int  # source + target + scratch
    requires_sort: bool
    requires_repartition: bool
    requires_dtype_cast: bool
    requires_reshape: bool  # wide<->long
    sort_cost_ms: float = 0.0
    repartition_cost_ms: float = 0.0
    cast_cost_ms: float = 0.0
    reshape_cost_ms: float = 0.0

    def total_transfer_ms(self) -> float:
        """Total edge transfer cost (MB-P1-007)."""
        return (
            self.sort_cost_ms
            + self.repartition_cost_ms
            + self.cast_cost_ms
            + self.reshape_cost_ms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_representation": self.source_representation,
            "target_representation": self.target_representation,
            "source_bytes": self.source_bytes,
            "target_bytes": self.target_bytes,
            "scratch_bytes": self.scratch_bytes,
            "peak_boundary_bytes": self.peak_boundary_bytes,
            "requires_sort": self.requires_sort,
            "requires_repartition": self.requires_repartition,
            "requires_dtype_cast": self.requires_dtype_cast,
            "requires_reshape": self.requires_reshape,
            "total_transfer_ms": round(self.total_transfer_ms(), 3),
        }


@dataclass
class LivenessTracker:
    """MB-P1-014, §46: Refcount-driven early buffer release.

    Each buffer maintains remaining_consumer_count. Last consumer triggers release.
    """

    _refcounts: dict[str, int]
    _sizes: dict[str, int]

    def __init__(self) -> None:
        self._refcounts = {}
        self._sizes = {}

    def register_buffer(self, buffer_id: str, size_bytes: int, consumer_count: int) -> None:
        """Register a buffer with initial consumer count."""
        self._refcounts[buffer_id] = consumer_count
        self._sizes[buffer_id] = size_bytes

    def consume(self, buffer_id: str) -> bool:
        """Mark one consumer complete. Returns True if buffer can be released."""
        if buffer_id not in self._refcounts:
            return False
        self._refcounts[buffer_id] -= 1
        if self._refcounts[buffer_id] <= 0:
            return True
        return False

    def release(self, buffer_id: str) -> int:
        """Release buffer, return freed bytes."""
        freed = self._sizes.pop(buffer_id, 0)
        self._refcounts.pop(buffer_id, None)
        return freed

    def total_live_bytes(self) -> int:
        """Current total live memory."""
        return sum(self._sizes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer_count": len(self._refcounts),
            "total_live_bytes": self.total_live_bytes(),
            "buffers": {
                bid: {"refcount": self._refcounts[bid], "bytes": self._sizes[bid]}
                for bid in self._refcounts
            },
        }


def estimate_shape_from_metadata(
    *,
    dates: int | None = None,
    instruments: int | None = None,
    columns: int = 1,
    frequency: str = "daily",
    bars_per_session: int | None = None,
    density: float = 1.0,
    dtype: str = "float64",
    remote: bool = False,
) -> DataShapeEstimate:
    """MB-P1-002, §27-29: Metadata-only shape estimation.

    No data loading. Conservative universe priors (§28):
        CSI300 ≈ 300, CSI500 ≈ 500, CSI1000 ≈ 1000, ALL_A ≈ 5500

    Args:
        dates: Trading sessions count (from calendar)
        instruments: Instrument count (from universe metadata)
        columns: Column count
        frequency: daily, minute, tick
        bars_per_session: For intraday data
        density: Sparsity (0.0-1.0)
        dtype: float64, float32, int32, etc.
        remote: Is data remote (COS, S3, etc.)

    Returns:
        DataShapeEstimate with metadata-only estimates
    """
    # Conservative defaults (§28)
    est_instruments = instruments if instruments is not None else 3000
    est_dates = dates if dates is not None else 252  # ~1 trading year

    # Rows
    if frequency == "daily":
        est_rows = est_dates * est_instruments
    elif frequency == "minute" and bars_per_session:
        est_rows = est_dates * bars_per_session * est_instruments
    else:
        est_rows = est_dates * est_instruments

    # Bytes
    dtype_size = _dtype_bytes(dtype)
    row_width = columns * dtype_size
    est_bytes = int(est_rows * row_width * density)

    return DataShapeEstimate(
        estimated_rows=est_rows,
        estimated_dates=est_dates,
        estimated_instruments=est_instruments,
        estimated_columns=columns,
        estimated_bytes=est_bytes,
        average_row_width_bytes=row_width,
        density=density,
        frequency=frequency,
        bars_per_session=bars_per_session,
        group_count=None,
        remote=remote,
        storage_kind="parquet",
        sorted_by=(),
        partition_by=(),
        projected_columns=(),
        rows_known=True,
    )


def estimate_operation_memory(
    *,
    shape: DataShapeEstimate,
    operation: str,
    live_columns: int,
    requires_sort: bool = False,
    requires_hash: bool = False,
    output_columns: int | None = None,
) -> MemoryCost:
    """MB-P1-004: Fine-grained operation memory cost.

    Args:
        shape: Input data shape
        operation: Operation name (for operation-specific heuristics)
        live_columns: Number of columns actually live in memory
        requires_sort: Operation needs sorting
        requires_hash: Operation needs hash table
        output_columns: Output column count (default = live_columns)

    Returns:
        MemoryCost with detailed breakdown
    """
    out_cols = output_columns if output_columns is not None else live_columns

    # Base input (only live columns)
    base_input = int(shape.estimated_rows * live_columns * 8)  # assume float64
    live_bytes = base_input

    # Sort working memory (§47: radix sort ~2x input)
    sort_working = int(base_input * 2.0) if requires_sort else 0

    # Hash table (§47: ~1.5x for groups + hash structure)
    hash_bytes = int(base_input * 1.5) if requires_hash else 0

    # Output
    output_bytes = int(shape.estimated_rows * out_cols * 8)

    # Scratch (operation-specific)
    scratch = _operation_scratch(operation, base_input)

    # Peak: max overlap
    if requires_sort:
        peak = base_input + sort_working + output_bytes + scratch
    elif requires_hash:
        peak = base_input + hash_bytes + output_bytes + scratch
    else:
        peak = base_input + output_bytes + scratch

    # Dtype breakdown (simplified)
    dtype_breakdown = {f"col_{i}": shape.estimated_rows * 8 for i in range(live_columns)}

    return MemoryCost(
        base_input_bytes=base_input,
        live_columns_bytes=live_bytes,
        sort_working_bytes=sort_working,
        hash_table_bytes=hash_bytes,
        output_bytes=output_bytes,
        scratch_bytes=scratch,
        peak_bytes=peak,
        dtype_breakdown=dtype_breakdown,
        requires_sort=requires_sort,
        requires_hash=requires_hash,
    )


def estimate_edge_memory(
    *,
    edge_id: str,
    source_shape: DataShapeEstimate,
    source_repr: str,
    target_repr: str,
    requires_sort: bool = False,
    requires_repartition: bool = False,
    requires_dtype_cast: bool = False,
    requires_reshape: bool = False,
) -> EdgeMemoryCost:
    """MB-P1-007/021, §32/47: Edge transfer memory cost.

    Boundary overlap: source + target + scratch simultaneously live.

    Args:
        edge_id: Unique edge identifier
        source_shape: Source buffer shape
        source_repr: Source representation
        target_repr: Target representation
        requires_sort: Needs sort
        requires_repartition: Needs repartition
        requires_dtype_cast: Needs dtype cast
        requires_reshape: Needs wide<->long reshape

    Returns:
        EdgeMemoryCost with boundary overlap
    """
    source_bytes = source_shape.estimated_bytes
    target_bytes = source_shape.estimated_bytes  # Conservative: same size

    # Representation-specific adjustments
    if "wide" in source_repr.lower() and "long" in target_repr.lower():
        # Wide->Long: typically expands
        target_bytes = int(source_bytes * 1.2)
    elif "long" in source_repr.lower() and "wide" in target_repr.lower():
        # Long->Wide: typically shrinks or same
        target_bytes = int(source_bytes * 0.9)

    # Scratch for conversion
    scratch = int(source_bytes * 0.3)  # Conservative conversion scratch

    # Sort cost (§32: bytes + ms)
    sort_ms = 0.0
    if requires_sort:
        scratch += int(source_bytes * 2.0)
        sort_ms = _estimate_sort_ms(source_bytes, source_shape.estimated_rows)

    # Repartition cost
    repartition_ms = 0.0
    if requires_repartition:
        scratch += int(source_bytes * 0.5)
        repartition_ms = _estimate_repartition_ms(source_bytes)

    # Dtype cast cost
    cast_ms = 0.0
    if requires_dtype_cast:
        cast_ms = _estimate_cast_ms(source_bytes)

    # Reshape cost
    reshape_ms = 0.0
    if requires_reshape:
        reshape_ms = _estimate_reshape_ms(source_bytes, source_shape.estimated_rows)

    # Peak boundary: source + target + scratch simultaneously live (§47)
    peak_boundary = source_bytes + target_bytes + scratch

    return EdgeMemoryCost(
        edge_id=edge_id,
        source_representation=source_repr,
        target_representation=target_repr,
        source_bytes=source_bytes,
        target_bytes=target_bytes,
        scratch_bytes=scratch,
        peak_boundary_bytes=peak_boundary,
        requires_sort=requires_sort,
        requires_repartition=requires_repartition,
        requires_dtype_cast=requires_dtype_cast,
        requires_reshape=requires_reshape,
        sort_cost_ms=sort_ms,
        repartition_cost_ms=repartition_ms,
        cast_cost_ms=cast_ms,
        reshape_cost_ms=reshape_ms,
    )


def _dtype_bytes(dtype: str) -> int:
    """Map dtype string to bytes."""
    mapping = {
        "float64": 8,
        "float32": 4,
        "int64": 8,
        "int32": 4,
        "int16": 2,
        "int8": 1,
        "bool": 1,
        "datetime64": 8,
    }
    return mapping.get(dtype.lower(), 8)  # default float64


def _operation_scratch(operation: str, input_bytes: int) -> int:
    """Operation-specific scratch memory estimate."""
    op_lower = operation.lower()
    if "rank" in op_lower or "sort" in op_lower:
        return int(input_bytes * 0.5)
    elif "group" in op_lower or "neutralize" in op_lower:
        return int(input_bytes * 1.0)
    elif "corr" in op_lower or "cov" in op_lower or "beta" in op_lower:
        return int(input_bytes * 0.8)
    else:
        return int(input_bytes * 0.2)


def _estimate_sort_ms(bytes_size: int, rows: int) -> float:
    """Estimate sort time (§32)."""
    # O(N log N) with bytes coefficient
    import math

    if rows <= 0:
        return 0.0
    n_log_n = rows * math.log2(max(2, rows))
    # ~1ms per million N log N ops, plus bytes transfer
    return (n_log_n / 1_000_000.0) + (bytes_size / (1024**3) * 50.0)


def _estimate_repartition_ms(bytes_size: int) -> float:
    """Estimate repartition/shuffle time."""
    # Primarily data movement
    return bytes_size / (1024**3) * 80.0  # ~80ms per GB


def _estimate_cast_ms(bytes_size: int) -> float:
    """Estimate dtype cast time."""
    # Fast memory scan
    return bytes_size / (1024**3) * 20.0  # ~20ms per GB


def _estimate_reshape_ms(bytes_size: int, rows: int) -> float:
    """Estimate wide<->long reshape time."""
    # Involves pivot/unpivot
    import math

    base = bytes_size / (1024**3) * 100.0  # ~100ms per GB
    if rows > 0:
        base += rows * math.log2(max(2, rows)) / 500_000.0
    return base
