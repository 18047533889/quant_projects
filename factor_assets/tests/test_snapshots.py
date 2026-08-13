"""
Test point-in-time snapshot manager and time-travel queries.
"""

import pytest
from datetime import datetime, timezone, timedelta

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import LifecycleState, StateEvent
from factor_assets.registry.snapshots import (
    SnapshotManager,
    SnapshotQuery,
    SnapshotResult,
)


def create_test_asset(factor_id: str, registered_at: str, campaign_id=None) -> FactorAsset:
    """Helper to create a test asset."""
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr=f"test_{factor_id}",
        canonical_hash=f"hash_{factor_id}",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
        campaign_id=campaign_id,
    )
    return FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=registered_at,
    )


def test_snapshot_manager_empty_registry():
    """Test snapshot of empty registry."""
    manager = SnapshotManager()

    query = SnapshotQuery(as_of_timestamp="2026-01-01T00:00:00Z")
    result = manager.create_snapshot([], [], query)

    assert result.total_count == 0
    assert len(result.assets) == 0


def test_snapshot_manager_basic_snapshot():
    """Test basic snapshot with registered assets."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        create_test_asset("F002", "2026-01-02T00:00:00Z"),
        create_test_asset("F003", "2026-01-03T00:00:00Z"),
    ]

    # Snapshot as of 2026-01-02T12:00:00Z should see F001 and F002
    query = SnapshotQuery(as_of_timestamp="2026-01-02T12:00:00Z")
    result = manager.create_snapshot(assets, [], query)

    assert result.total_count == 2
    factor_ids = {a.factor_id for a in result.assets}
    assert factor_ids == {"F001", "F002"}


def test_snapshot_manager_with_state_transitions():
    """Test snapshot reflecting state transitions."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
    ]

    events = [
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.EVALUATED,
            to_state=LifecycleState.APPROVED,
            timestamp="2026-01-03T00:00:00Z",
            evidence_refs=("gate_results",),
        ),
    ]

    # Snapshot before evaluation
    query1 = SnapshotQuery(as_of_timestamp="2026-01-01T12:00:00Z")
    result1 = manager.create_snapshot(assets, events, query1)
    assert result1.assets[0].lifecycle_state == LifecycleState.REGISTERED

    # Snapshot after evaluation
    query2 = SnapshotQuery(as_of_timestamp="2026-01-02T12:00:00Z")
    result2 = manager.create_snapshot(assets, events, query2)
    assert result2.assets[0].lifecycle_state == LifecycleState.EVALUATED

    # Snapshot after approval
    query3 = SnapshotQuery(as_of_timestamp="2026-01-03T12:00:00Z")
    result3 = manager.create_snapshot(assets, events, query3)
    assert result3.assets[0].lifecycle_state == LifecycleState.APPROVED


def test_snapshot_manager_filter_by_state():
    """Test filtering snapshot by lifecycle state."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        create_test_asset("F002", "2026-01-01T00:00:00Z"),
        create_test_asset("F003", "2026-01-01T00:00:00Z"),
    ]

    events = [
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F002",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F002",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        StateEvent(
            factor_id="F003",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
    ]

    # Query for EVALUATED factors as of 2026-01-02T12:00:00Z
    query = SnapshotQuery(
        as_of_timestamp="2026-01-02T12:00:00Z",
        lifecycle_state=LifecycleState.EVALUATED,
    )
    result = manager.create_snapshot(assets, events, query)

    assert result.total_count == 1
    assert result.assets[0].factor_id == "F002"


def test_snapshot_manager_filter_by_campaign():
    """Test filtering snapshot by campaign ID."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z", campaign_id="camp_001"),
        create_test_asset("F002", "2026-01-01T00:00:00Z", campaign_id="camp_001"),
        create_test_asset("F003", "2026-01-01T00:00:00Z", campaign_id="camp_002"),
    ]

    query = SnapshotQuery(
        as_of_timestamp="2026-01-02T00:00:00Z",
        campaign_id="camp_001",
    )
    result = manager.create_snapshot(assets, [], query)

    assert result.total_count == 2
    factor_ids = {a.factor_id for a in result.assets}
    assert factor_ids == {"F001", "F002"}


def test_snapshot_manager_filter_by_factor_id():
    """Test filtering snapshot by specific factor ID."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        create_test_asset("F002", "2026-01-01T00:00:00Z"),
    ]

    query = SnapshotQuery(
        as_of_timestamp="2026-01-02T00:00:00Z",
        factor_id="F001",
    )
    result = manager.create_snapshot(assets, [], query)

    assert result.total_count == 1
    assert result.assets[0].factor_id == "F001"


def test_snapshot_manager_get_state_at_time():
    """Test getting a factor's state at a specific time."""
    manager = SnapshotManager()

    events = [
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
    ]

    # Before evaluation
    state1 = manager.get_state_at_time("F001", events, "2026-01-01T12:00:00Z")
    assert state1 == LifecycleState.REGISTERED

    # After evaluation
    state2 = manager.get_state_at_time("F001", events, "2026-01-02T12:00:00Z")
    assert state2 == LifecycleState.EVALUATED


def test_snapshot_manager_get_state_transitions():
    """Test getting state transitions within a time range."""
    manager = SnapshotManager()

    events = [
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.EVALUATED,
            to_state=LifecycleState.APPROVED,
            timestamp="2026-01-03T00:00:00Z",
            evidence_refs=("gate_results",),
        ),
        StateEvent(
            factor_id="F002",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
    ]

    # Get transitions for F001 between 2026-01-01 and 2026-01-02
    transitions = manager.get_state_transitions(
        "F001",
        events,
        start_timestamp="2026-01-01T00:00:00Z",
        end_timestamp="2026-01-02T23:59:59Z",
    )

    assert len(transitions) == 2
    assert transitions[0].to_state == LifecycleState.REGISTERED
    assert transitions[1].to_state == LifecycleState.EVALUATED


def test_snapshot_manager_compare_snapshots():
    """Test comparing registry state between two timestamps."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        create_test_asset("F002", "2026-01-02T00:00:00Z"),
        create_test_asset("F003", "2026-01-03T00:00:00Z"),
    ]

    events = [
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        StateEvent(
            factor_id="F002",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F003",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-03T00:00:00Z",
            evidence_refs=(),
        ),
    ]

    comparison = manager.compare_snapshots(
        assets,
        events,
        "2026-01-01T12:00:00Z",
        "2026-01-02T12:00:00Z",
    )

    assert comparison["new_registrations"] == 1  # F002
    assert comparison["total_factors_at_a"] == 1  # Only F001
    assert comparison["total_factors_at_b"] == 2  # F001, F002
    assert "REGISTERED -> EVALUATED" in comparison["state_changes"]


def test_snapshot_manager_reconstruct_timestamps():
    """Test that milestone timestamps are correctly reconstructed."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
    ]

    events = [
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.EVALUATED,
            to_state=LifecycleState.APPROVED,
            timestamp="2026-01-03T00:00:00Z",
            evidence_refs=("gate_results",),
        ),
    ]

    # Snapshot after approval
    query = SnapshotQuery(as_of_timestamp="2026-01-03T12:00:00Z")
    result = manager.create_snapshot(assets, events, query)

    asset = result.assets[0]
    assert asset.first_evaluated_at == "2026-01-02T00:00:00Z"
    assert asset.approved_at == "2026-01-03T00:00:00Z"


def test_snapshot_manager_multiple_factors():
    """Test snapshot with multiple factors at different states."""
    manager = SnapshotManager()

    assets = [
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        create_test_asset("F002", "2026-01-01T00:00:00Z"),
        create_test_asset("F003", "2026-01-01T00:00:00Z"),
    ]

    events = [
        # F001: REGISTERED -> EVALUATED
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        # F002: REGISTERED only
        StateEvent(
            factor_id="F002",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        # F003: REGISTERED -> EVALUATED -> APPROVED
        StateEvent(
            factor_id="F003",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp="2026-01-01T00:00:00Z",
            evidence_refs=(),
        ),
        StateEvent(
            factor_id="F003",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            timestamp="2026-01-02T00:00:00Z",
            evidence_refs=("evaluation_bundle_ref",),
        ),
        StateEvent(
            factor_id="F003",
            from_state=LifecycleState.EVALUATED,
            to_state=LifecycleState.APPROVED,
            timestamp="2026-01-03T00:00:00Z",
            evidence_refs=("gate_results",),
        ),
    ]

    # Snapshot as of 2026-01-02T12:00:00Z
    query = SnapshotQuery(as_of_timestamp="2026-01-02T12:00:00Z")
    result = manager.create_snapshot(assets, events, query)

    states = {a.factor_id: a.lifecycle_state for a in result.assets}

    assert states["F001"] == LifecycleState.EVALUATED
    assert states["F002"] == LifecycleState.REGISTERED
    assert states["F003"] == LifecycleState.EVALUATED
