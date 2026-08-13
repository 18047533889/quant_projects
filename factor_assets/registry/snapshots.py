"""
Point-in-time registry snapshots for time-travel queries.

Provides immutable views of the registry at specific timestamps.
Supports querying historical states and lifecycle progression.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

from factor_assets.contracts.asset import FactorAsset
from factor_assets.contracts.lifecycle import LifecycleState, StateEvent


@dataclass(frozen=True)
class SnapshotQuery:
    """
    Query specification for a point-in-time snapshot.

    Defines the temporal boundary and optional filters.
    """
    as_of_timestamp: str  # ISO 8601
    factor_id: Optional[str] = None
    lifecycle_state: Optional[LifecycleState] = None
    campaign_id: Optional[str] = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.as_of_timestamp:
            raise ValueError("as_of_timestamp is required")


@dataclass(frozen=True)
class SnapshotResult:
    """
    Result of a point-in-time snapshot query.

    Contains assets matching the query at the specified timestamp.
    """
    query: SnapshotQuery
    assets: tuple[FactorAsset, ...]
    total_count: int
    snapshot_created_at: str

    def __post_init__(self):
        if not self.snapshot_created_at:
            object.__setattr__(
                self, "snapshot_created_at", datetime.now(timezone.utc).isoformat()
            )


class SnapshotManager:
    """
    Manages point-in-time snapshots of the factor registry.

    Reconstructs historical registry state from events and assets.
    Does NOT store snapshots persistently — computes them on demand.
    """

    def __init__(self):
        pass

    def create_snapshot(
        self,
        assets: list[FactorAsset],
        events: list[StateEvent],
        query: SnapshotQuery
    ) -> SnapshotResult:
        """
        Create a point-in-time snapshot from assets and events.

        Args:
            assets: All registered assets
            events: All lifecycle events
            query: Snapshot query specification

        Returns:
            SnapshotResult containing assets as of the specified time
        """
        as_of = query.as_of_timestamp

        # Build historical state map from events
        historical_states = self._reconstruct_states(events, as_of)

        # Filter assets based on query
        matching_assets = []

        for asset in assets:
            # Check if asset was registered before as_of
            if asset.registered_at > as_of:
                continue

            # Get historical state for this asset
            factor_id = asset.factor_id
            historical_state = historical_states.get(factor_id, LifecycleState.REGISTERED)

            # Apply filters
            if query.factor_id and asset.factor_id != query.factor_id:
                continue

            if query.lifecycle_state and historical_state != query.lifecycle_state:
                continue

            if query.campaign_id:
                if asset.lineage.campaign_id != query.campaign_id:
                    continue

            if query.tags:
                if not set(query.tags).issubset(set(asset.tags)):
                    continue

            # Reconstruct asset with historical state
            historical_asset = self._reconstruct_asset_state(asset, events, as_of)
            matching_assets.append(historical_asset)

        return SnapshotResult(
            query=query,
            assets=tuple(matching_assets),
            total_count=len(matching_assets),
            snapshot_created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _reconstruct_states(
        self,
        events: list[StateEvent],
        as_of_timestamp: str
    ) -> dict[str, LifecycleState]:
        """
        Reconstruct lifecycle states as of a timestamp.

        Args:
            events: All lifecycle events
            as_of_timestamp: Point-in-time boundary

        Returns:
            Map of factor_id -> LifecycleState as of the timestamp
        """
        states: dict[str, LifecycleState] = {}

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for event in sorted_events:
            if event.timestamp > as_of_timestamp:
                break

            states[event.factor_id] = event.to_state

        return states

    def _reconstruct_asset_state(
        self,
        asset: FactorAsset,
        events: list[StateEvent],
        as_of_timestamp: str
    ) -> FactorAsset:
        """
        Reconstruct an asset's state as of a timestamp.

        Returns a copy of the asset with lifecycle_state reflecting the historical state.
        """
        from dataclasses import replace

        # Find the latest state before as_of_timestamp
        historical_state = LifecycleState.REGISTERED
        first_evaluated_at = None
        approved_at = None
        production_ready_at = None

        for event in sorted(events, key=lambda e: e.timestamp):
            if event.factor_id != asset.factor_id:
                continue

            if event.timestamp > as_of_timestamp:
                break

            historical_state = event.to_state

            # Track milestone timestamps
            if event.to_state == LifecycleState.EVALUATED and first_evaluated_at is None:
                first_evaluated_at = event.timestamp
            elif event.to_state == LifecycleState.APPROVED and approved_at is None:
                approved_at = event.timestamp
            elif event.to_state == LifecycleState.PRODUCTION_READY and production_ready_at is None:
                production_ready_at = event.timestamp

        return replace(
            asset,
            lifecycle_state=historical_state,
            first_evaluated_at=first_evaluated_at,
            approved_at=approved_at,
            production_ready_at=production_ready_at,
        )

    def get_state_at_time(
        self,
        factor_id: str,
        events: list[StateEvent],
        as_of_timestamp: str
    ) -> LifecycleState:
        """
        Get a factor's lifecycle state at a specific timestamp.

        Args:
            factor_id: Factor identifier
            events: All lifecycle events
            as_of_timestamp: Point-in-time boundary

        Returns:
            LifecycleState as of the timestamp
        """
        state = LifecycleState.REGISTERED

        for event in sorted(events, key=lambda e: e.timestamp):
            if event.factor_id != factor_id:
                continue

            if event.timestamp > as_of_timestamp:
                break

            state = event.to_state

        return state

    def get_state_transitions(
        self,
        factor_id: str,
        events: list[StateEvent],
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None
    ) -> tuple[StateEvent, ...]:
        """
        Get all state transitions for a factor within a time range.

        Args:
            factor_id: Factor identifier
            events: All lifecycle events
            start_timestamp: Optional start boundary
            end_timestamp: Optional end boundary

        Returns:
            Tuple of StateEvent records within the time range
        """
        matching_events = []

        for event in events:
            if event.factor_id != factor_id:
                continue

            if start_timestamp and event.timestamp < start_timestamp:
                continue

            if end_timestamp and event.timestamp > end_timestamp:
                continue

            matching_events.append(event)

        return tuple(sorted(matching_events, key=lambda e: e.timestamp))

    def compare_snapshots(
        self,
        assets: list[FactorAsset],
        events: list[StateEvent],
        timestamp_a: str,
        timestamp_b: str
    ) -> dict:
        """
        Compare registry state between two timestamps.

        Args:
            assets: All registered assets
            events: All lifecycle events
            timestamp_a: First timestamp
            timestamp_b: Second timestamp

        Returns:
            Dictionary with comparison statistics
        """
        states_a = self._reconstruct_states(events, timestamp_a)
        states_b = self._reconstruct_states(events, timestamp_b)

        # Count assets registered between timestamps
        new_registrations = sum(
            1 for asset in assets
            if timestamp_a < asset.registered_at <= timestamp_b
        )

        # Count state changes
        state_changes = {}
        for factor_id in states_b:
            state_a = states_a.get(factor_id, LifecycleState.REGISTERED)
            state_b = states_b[factor_id]

            if state_a != state_b:
                key = f"{state_a.value} -> {state_b.value}"
                state_changes[key] = state_changes.get(key, 0) + 1

        return {
            "timestamp_a": timestamp_a,
            "timestamp_b": timestamp_b,
            "new_registrations": new_registrations,
            "state_changes": state_changes,
            "total_factors_at_a": len(states_a),
            "total_factors_at_b": len(states_b),
        }
