# -*- coding: utf-8 -*-
"""ExecutionAxis: First-class execution axis classification and sharding rules.

ExecutionAxis defines how computation is organized and parallelized:
- TIME_PER_INSTRUMENT: time-series operations per asset (e.g., rolling mean)
- CROSS_SECTION_PER_DATE: cross-sectional operations per date (e.g., rank, zscore)
- GROUP_PER_DATE: grouped operations per date (e.g., industry-neutralized)
- GLOBAL_PANEL: operations requiring full panel (e.g., PCA, covariance matrix)
- RECURSIVE_TIME: stateful time-series with recursive state (e.g., EMA with deadband)

Sharding safety rules prevent incorrect parallelization that would violate semantics.
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Literal


class ExecutionAxis(Enum):
    """Execution axis classification for operator scheduling and sharding.

    The execution axis determines:
    1. How computation can be parallelized
    2. Which sharding dimensions are safe
    3. When barriers are required between regions
    4. How state propagates through the computation
    """

    # Time-series per instrument: independent per asset
    # Examples: ts_mean, ts_std, ts_delta, ts_corr
    # Safe sharding: asset buckets (with lookback overlap for rolling)
    # Unsafe sharding: time buckets (requires lookback overlap)
    TIME_PER_INSTRUMENT = "TIME_PER_INSTRUMENT"

    # Cross-sectional per date: requires all instruments per date
    # Examples: rank, zscore, neutralize, cross-sectional regression
    # Safe sharding: date buckets
    # Unsafe sharding: asset buckets (would compute rank on subset)
    CROSS_SECTION_PER_DATE = "CROSS_SECTION_PER_DATE"

    # Grouped cross-sectional per date: requires full group per date
    # Examples: industry-neutralized, sector rank
    # Safe sharding: date buckets, group-aligned asset buckets
    # Unsafe sharding: arbitrary asset buckets (would split groups)
    GROUP_PER_DATE = "GROUP_PER_DATE"

    # Global panel: requires full dataset
    # Examples: PCA, full-panel covariance, global normalization
    # Safe sharding: none (must collect full panel)
    # Unsafe sharding: any dimension split
    GLOBAL_PANEL = "GLOBAL_PANEL"

    # Recursive time-series: stateful computation with dependencies
    # Examples: EMA with deadband, Kalman filter, regime detection
    # Safe sharding: asset buckets (independent state per asset)
    # Unsafe sharding: time buckets (unless checkpoint chain exists)
    RECURSIVE_TIME = "RECURSIVE_TIME"

    # Event-driven streaming: event-by-event processing
    # Examples: tick aggregation, event accumulation
    # Safe sharding: asset buckets
    # Unsafe sharding: time buckets (ordering dependencies)
    EVENT_STREAM = "EVENT_STREAM"

    # Relational: pure relational operations with no axis semantics
    # Examples: filter, projection, join on keys
    # Safe sharding: depends on operation (filter: any, join: by key)
    RELATIONAL = "RELATIONAL"


@dataclass(frozen=True)
class ShardingConstraints:
    """Sharding safety constraints for a given execution axis."""

    execution_axis: ExecutionAxis

    # Can split across asset dimension
    asset_shardable: bool

    # Can split across time dimension
    time_shardable: bool

    # Requires lookback overlap for time shards
    requires_lookback_overlap: bool

    # Requires checkpoint chain for time shards
    requires_checkpoint_chain: bool

    # Must preserve group integrity
    requires_group_integrity: bool

    # Requires full panel (no sharding)
    requires_full_panel: bool


# Sharding safety matrix
SHARDING_CONSTRAINTS = {
    ExecutionAxis.TIME_PER_INSTRUMENT: ShardingConstraints(
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        asset_shardable=True,
        time_shardable=True,
        requires_lookback_overlap=True,
        requires_checkpoint_chain=False,
        requires_group_integrity=False,
        requires_full_panel=False,
    ),
    ExecutionAxis.CROSS_SECTION_PER_DATE: ShardingConstraints(
        execution_axis=ExecutionAxis.CROSS_SECTION_PER_DATE,
        asset_shardable=False,  # CRITICAL: rank/zscore on subset is wrong
        time_shardable=True,
        requires_lookback_overlap=False,
        requires_checkpoint_chain=False,
        requires_group_integrity=False,
        requires_full_panel=False,
    ),
    ExecutionAxis.GROUP_PER_DATE: ShardingConstraints(
        execution_axis=ExecutionAxis.GROUP_PER_DATE,
        asset_shardable=True,  # Only if group-aligned
        time_shardable=True,
        requires_lookback_overlap=False,
        requires_checkpoint_chain=False,
        requires_group_integrity=True,  # CRITICAL: must keep full groups
        requires_full_panel=False,
    ),
    ExecutionAxis.GLOBAL_PANEL: ShardingConstraints(
        execution_axis=ExecutionAxis.GLOBAL_PANEL,
        asset_shardable=False,
        time_shardable=False,
        requires_lookback_overlap=False,
        requires_checkpoint_chain=False,
        requires_group_integrity=False,
        requires_full_panel=True,  # CRITICAL: must see entire panel
    ),
    ExecutionAxis.RECURSIVE_TIME: ShardingConstraints(
        execution_axis=ExecutionAxis.RECURSIVE_TIME,
        asset_shardable=True,
        time_shardable=False,  # Unless checkpoint chain
        requires_lookback_overlap=False,
        requires_checkpoint_chain=True,  # CRITICAL for time sharding
        requires_group_integrity=False,
        requires_full_panel=False,
    ),
    ExecutionAxis.EVENT_STREAM: ShardingConstraints(
        execution_axis=ExecutionAxis.EVENT_STREAM,
        asset_shardable=True,
        time_shardable=False,
        requires_lookback_overlap=False,
        requires_checkpoint_chain=False,
        requires_group_integrity=False,
        requires_full_panel=False,
    ),
    ExecutionAxis.RELATIONAL: ShardingConstraints(
        execution_axis=ExecutionAxis.RELATIONAL,
        asset_shardable=True,
        time_shardable=True,
        requires_lookback_overlap=False,
        requires_checkpoint_chain=False,
        requires_group_integrity=False,
        requires_full_panel=False,
    ),
}


@dataclass(frozen=True)
class AxisTransition:
    """Represents a transition between execution axes requiring a barrier.

    Certain axis transitions require synchronization barriers:
    - TIME → CROSS_SECTION: must complete all time series before ranking
    - CROSS_SECTION → TIME: must complete ranking before time-series on ranks
    - Any → GLOBAL_PANEL: must collect full data
    - RECURSIVE_TIME → any: must complete state propagation
    """

    source_axis: ExecutionAxis
    target_axis: ExecutionAxis
    requires_barrier: bool
    requires_full_collection: bool
    description: str


# Axis transition rules
def requires_barrier(source: ExecutionAxis, target: ExecutionAxis) -> AxisTransition:
    """Determine if transition between execution axes requires a barrier."""

    # Same axis: no barrier
    if source == target:
        return AxisTransition(
            source_axis=source,
            target_axis=target,
            requires_barrier=False,
            requires_full_collection=False,
            description="Same axis, no barrier needed",
        )

    # TIME → CROSS_SECTION: barrier required
    if source == ExecutionAxis.TIME_PER_INSTRUMENT and target == ExecutionAxis.CROSS_SECTION_PER_DATE:
        return AxisTransition(
            source_axis=source,
            target_axis=target,
            requires_barrier=True,
            requires_full_collection=True,
            description="TIME→CS: must complete all time-series before cross-sectional operations",
        )

    # CROSS_SECTION → TIME: barrier required
    if source == ExecutionAxis.CROSS_SECTION_PER_DATE and target == ExecutionAxis.TIME_PER_INSTRUMENT:
        return AxisTransition(
            source_axis=source,
            target_axis=target,
            requires_barrier=True,
            requires_full_collection=True,
            description="CS→TIME: must complete cross-sectional operations before time-series",
        )

    # Any → GLOBAL_PANEL: barrier required
    if target == ExecutionAxis.GLOBAL_PANEL:
        return AxisTransition(
            source_axis=source,
            target_axis=target,
            requires_barrier=True,
            requires_full_collection=True,
            description="→GLOBAL: must collect full panel",
        )

    # GLOBAL_PANEL → any: barrier required
    if source == ExecutionAxis.GLOBAL_PANEL:
        return AxisTransition(
            source_axis=source,
            target_axis=target,
            requires_barrier=True,
            requires_full_collection=True,
            description="GLOBAL→: must complete global operation",
        )

    # RECURSIVE_TIME → any: barrier required
    if source == ExecutionAxis.RECURSIVE_TIME:
        return AxisTransition(
            source_axis=source,
            target_axis=target,
            requires_barrier=True,
            requires_full_collection=False,
            description="RECURSIVE→: must complete state propagation",
        )

    # Default: no barrier for compatible axes
    return AxisTransition(
        source_axis=source,
        target_axis=target,
        requires_barrier=False,
        requires_full_collection=False,
        description="Compatible axes, no barrier needed",
    )


def validate_sharding_plan(
    execution_axis: ExecutionAxis,
    shard_dimension: Literal["asset", "time", "none"],
    has_checkpoint_chain: bool = False,
) -> tuple[bool, str]:
    """Validate if a sharding plan is safe for the given execution axis.

    Returns:
        (is_valid, reason)
    """
    constraints = SHARDING_CONSTRAINTS[execution_axis]

    if shard_dimension == "none":
        return True, "No sharding requested"

    if shard_dimension == "asset":
        if not constraints.asset_shardable:
            return False, f"{execution_axis.value} cannot be sharded by asset (e.g., rank/zscore would compute on subset)"
        if constraints.requires_group_integrity:
            return True, "Asset sharding allowed but must preserve group integrity"
        return True, "Asset sharding allowed"

    if shard_dimension == "time":
        if not constraints.time_shardable:
            if execution_axis == ExecutionAxis.RECURSIVE_TIME:
                if has_checkpoint_chain:
                    return True, "Time sharding allowed with checkpoint chain"
                return False, f"{execution_axis.value} cannot be sharded by time without checkpoint chain"
            return False, f"{execution_axis.value} cannot be sharded by time"
        if constraints.requires_lookback_overlap:
            return True, "Time sharding allowed with lookback overlap"
        return True, "Time sharding allowed"

    return False, f"Unknown shard dimension: {shard_dimension}"


__all__ = [
    "ExecutionAxis",
    "ShardingConstraints",
    "AxisTransition",
    "SHARDING_CONSTRAINTS",
    "requires_barrier",
    "validate_sharding_plan",
]
