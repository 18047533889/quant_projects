# -*- coding: utf-8
"""Polars Region optimization — multi-root unified collection and native buffer retention.

Implementation of sections §17-20 from the Multi-Backend DAG Remediation document:

§17: Multi-root unified LazyFrame compilation and collect_all
§18: Polars UDF capability classification refinement
§19: Streaming capability metadata per canonical operator
§20: Native buffer retention for shared Polars nodes

Key principles:
- Multiple factors from same source scope → one large Polars LazyFrame
- Multiple roots use unified collect_all, not per-factor eager collect
- Avoid per-factor Pandas round-trips within a Polars Region
- Classify capabilities: NATIVE_EXPR / NATIVE_GROUP / STREAMING / PYTHON_UDF_DELEGATE
- Shared Polars nodes keep native buffers; conversion happens once per target representation
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


class PolarsCapability(str, Enum):
    """Polars operator capability classification (§18).

    Distinguishes truly native Polars operations from Python UDF delegates.
    """

    POLARS_NATIVE_EXPR = "polars_native_expr"
    """Pure Polars expression with no Python callbacks."""

    POLARS_NATIVE_GROUP = "polars_native_group"
    """Native Polars group_by aggregation."""

    POLARS_NATIVE_STREAMING = "polars_native_streaming"
    """Native Polars operation that supports streaming execution."""

    POLARS_PYTHON_UDF_DELEGATE = "polars_python_udf_delegate"
    """Uses map_elements, Python lambda, or opaque map_groups — NOT fast path."""

    UNSUPPORTED = "unsupported"
    """Not supported in Polars backend."""


class StreamingCapability(str, Enum):
    """Streaming capability metadata per operator (§19).

    Planner uses this to decide collect / stream / sink_parquet.
    """

    STREAMING_SAFE = "streaming_safe"
    """Can execute in streaming mode without materializing full dataset."""

    REQUIRES_GLOBAL_SORT = "requires_global_sort"
    """Needs full dataset sorted before execution."""

    REQUIRES_FULL_GROUP = "requires_full_group"
    """Needs full group materialized (e.g., group quantile)."""

    REQUIRES_RECURSIVE_STATE = "requires_recursive_state"
    """Stateful operator requiring checkpoint/continuation."""


@dataclass(frozen=True)
class PolarsOperatorCapability:
    """Complete capability metadata for a Polars operator (§18, §19).

    Attributes:
        canonical_op: Canonical operator name
        polars_capability: Native vs UDF delegate classification
        streaming_capability: Streaming execution requirements
        supports_lazy: Whether operator can be compiled to LazyFrame
        requires_eager_boundary: Whether must collect before this operator
        python_callback: Whether involves Python callback (not pure Polars)
    """

    canonical_op: str
    polars_capability: PolarsCapability
    streaming_capability: StreamingCapability
    supports_lazy: bool = True
    requires_eager_boundary: bool = False
    python_callback: bool = False


@dataclass(frozen=True)
class NativeBufferDescriptor:
    """Native buffer representation metadata (§20).

    Tracks native Polars buffer state for shared nodes to avoid redundant conversions.

    Attributes:
        semantic_node_id: Logical node identity
        representation: Native representation (POLARS_LAZY_LONG / POLARS_LONG)
        buffer_id: Physical buffer identity
        consumer_count: Number of remaining consumers
        conversion_cache: Cached conversions to other representations
    """

    semantic_node_id: str
    representation: str  # From planner.backend_region.Representation
    buffer_id: str
    consumer_count: int
    conversion_cache: dict[str, Any]  # target_repr -> converted buffer


@dataclass(frozen=True)
class PolarsRegionPlan:
    """Physical plan for a Polars Region with multi-root collection (§17).

    Attributes:
        region_id: Unique region identifier
        root_node_ids: Multiple factor roots to compile together
        shared_node_ids: Shared intermediate nodes
        base_lazy_frame: Common base LazyFrame from source scan
        collection_mode: 'collect_all' | 'collect' | 'stream' | 'sink_parquet'
        estimated_rows: Estimated result size
        streaming_safe: Whether entire region can stream
    """

    region_id: str
    root_node_ids: tuple[str, ...]
    shared_node_ids: tuple[str, ...]
    base_lazy_frame_id: str | None
    collection_mode: str  # 'collect_all' | 'collect' | 'stream' | 'sink_parquet'
    estimated_rows: int
    streaming_safe: bool
    requires_sort: bool = False
    native_fraction: float = 1.0  # 0.0-1.0, excludes UDF delegates


# §18: Capability classification registry
# Maps canonical operator names to their Polars capability metadata

_POLARS_CAPABILITY_REGISTRY: dict[str, PolarsOperatorCapability] = {}


def register_polars_capability(
    canonical_op: str,
    polars_capability: PolarsCapability,
    streaming_capability: StreamingCapability,
    supports_lazy: bool = True,
    requires_eager_boundary: bool = False,
    python_callback: bool = False,
) -> None:
    """Register capability metadata for a Polars operator (§18, §19).

    Args:
        canonical_op: Canonical operator name
        polars_capability: Native vs UDF classification
        streaming_capability: Streaming requirements
        supports_lazy: Can compile to LazyFrame
        requires_eager_boundary: Must collect before
        python_callback: Involves Python callback
    """
    _POLARS_CAPABILITY_REGISTRY[canonical_op] = PolarsOperatorCapability(
        canonical_op=canonical_op,
        polars_capability=polars_capability,
        streaming_capability=streaming_capability,
        supports_lazy=supports_lazy,
        requires_eager_boundary=requires_eager_boundary,
        python_callback=python_callback,
    )


def get_polars_capability(canonical_op: str) -> PolarsOperatorCapability | None:
    """Get capability metadata for a Polars operator."""
    return _POLARS_CAPABILITY_REGISTRY.get(canonical_op)


def is_polars_native_fast_path(canonical_op: str) -> bool:
    """Check if operator is truly native Polars (§18).

    Returns False for map_elements, Python lambda, opaque map_groups.
    """
    cap = get_polars_capability(canonical_op)
    if cap is None:
        return False
    return cap.polars_capability in {
        PolarsCapability.POLARS_NATIVE_EXPR,
        PolarsCapability.POLARS_NATIVE_GROUP,
        PolarsCapability.POLARS_NATIVE_STREAMING,
    }


def classify_polars_capability_from_tier(tier: str) -> PolarsCapability:
    """Map existing tier classification to new capability enum (§18).

    Args:
        tier: Tier from polars_long_policy (native/python_rolling/map_groups/registry)

    Returns:
        Corresponding PolarsCapability enum value
    """
    if tier == "native":
        return PolarsCapability.POLARS_NATIVE_EXPR
    if tier == "stateful":
        # Stateful operators are native Polars but may have streaming constraints
        return PolarsCapability.POLARS_NATIVE_STREAMING
    if tier in {"python_rolling", "map_groups", "registry"}:
        # These use Python callbacks — NOT native fast path
        return PolarsCapability.POLARS_PYTHON_UDF_DELEGATE
    return PolarsCapability.UNSUPPORTED


# §19: Streaming capability metadata
# Initialize with conservative defaults; operators can override

_STREAMING_CAPABILITY_MAP: dict[str, StreamingCapability] = {
    # Streaming-safe operations (no global state needed)
    "add": StreamingCapability.STREAMING_SAFE,
    "subtract": StreamingCapability.STREAMING_SAFE,
    "multiply": StreamingCapability.STREAMING_SAFE,
    "divide": StreamingCapability.STREAMING_SAFE,
    "abs": StreamingCapability.STREAMING_SAFE,
    "log": StreamingCapability.STREAMING_SAFE,
    "exp": StreamingCapability.STREAMING_SAFE,
    "sqrt": StreamingCapability.STREAMING_SAFE,
    "clip": StreamingCapability.STREAMING_SAFE,
    "fillna": StreamingCapability.STREAMING_SAFE,
    "where": StreamingCapability.STREAMING_SAFE,

    # Per-group operations (streaming-safe within groups)
    "ts_mean": StreamingCapability.STREAMING_SAFE,
    "ts_sum": StreamingCapability.STREAMING_SAFE,
    "ts_std": StreamingCapability.STREAMING_SAFE,
    "ts_min": StreamingCapability.STREAMING_SAFE,
    "ts_max": StreamingCapability.STREAMING_SAFE,

    # Cross-sectional operations requiring full groups
    "rank": StreamingCapability.REQUIRES_FULL_GROUP,
    "rank_pct": StreamingCapability.REQUIRES_FULL_GROUP,
    "cs_quantile": StreamingCapability.REQUIRES_FULL_GROUP,
    "zscore": StreamingCapability.REQUIRES_FULL_GROUP,
    "normalize": StreamingCapability.REQUIRES_FULL_GROUP,

    # Operations requiring global sort
    "ts_rank": StreamingCapability.REQUIRES_GLOBAL_SORT,

    # Stateful/recursive operations
    "ts_ema": StreamingCapability.REQUIRES_RECURSIVE_STATE,
    "KAMA": StreamingCapability.REQUIRES_RECURSIVE_STATE,
    "RSI_WILDER": StreamingCapability.REQUIRES_RECURSIVE_STATE,
    "MACD": StreamingCapability.REQUIRES_RECURSIVE_STATE,
}


def get_streaming_capability(canonical_op: str) -> StreamingCapability:
    """Get streaming capability for an operator (§19)."""
    return _STREAMING_CAPABILITY_MAP.get(
        canonical_op,
        StreamingCapability.REQUIRES_FULL_GROUP  # Conservative default
    )


def init_polars_capability_registry() -> None:
    """Initialize capability registry from existing tier classifications (§18).

    Bridges existing polars_long_policy tiers to new capability system.
    """
    from backend.polars_long_policy import (
        POLARS_LONG_NATIVE,
        POLARS_LONG_STATEFUL,
        POLARS_LONG_PYTHON_ROLLING,
        POLARS_LONG_MAP_GROUPS,
    )

    # Register native expressions
    for op in POLARS_LONG_NATIVE:
        streaming_cap = get_streaming_capability(op)
        register_polars_capability(
            canonical_op=op,
            polars_capability=PolarsCapability.POLARS_NATIVE_EXPR,
            streaming_capability=streaming_cap,
            supports_lazy=True,
            python_callback=False,
        )

    # Register stateful operations
    for op in POLARS_LONG_STATEFUL:
        register_polars_capability(
            canonical_op=op,
            polars_capability=PolarsCapability.POLARS_NATIVE_STREAMING,
            streaming_capability=StreamingCapability.REQUIRES_RECURSIVE_STATE,
            supports_lazy=True,
            python_callback=False,
        )

    # Register Python rolling (UDF delegate, NOT native fast path)
    for op in POLARS_LONG_PYTHON_ROLLING:
        register_polars_capability(
            canonical_op=op,
            polars_capability=PolarsCapability.POLARS_PYTHON_UDF_DELEGATE,
            streaming_capability=StreamingCapability.REQUIRES_FULL_GROUP,
            supports_lazy=True,
            python_callback=True,  # §18: NOT native fast path
        )

    # Register map_groups operations (UDF delegate, NOT native fast path)
    for op in POLARS_LONG_MAP_GROUPS:
        register_polars_capability(
            canonical_op=op,
            polars_capability=PolarsCapability.POLARS_PYTHON_UDF_DELEGATE,
            streaming_capability=StreamingCapability.REQUIRES_FULL_GROUP,
            supports_lazy=True,
            python_callback=True,  # §18: NOT native fast path
        )


# Initialize on module load
init_polars_capability_registry()
