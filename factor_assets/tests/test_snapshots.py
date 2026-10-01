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
import factor_assets.registry.snapshots as snapshots_module


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


def test_mixed_timezone_offsets_use_instants_for_cutoff_and_state_order():
    manager = SnapshotManager()
    asset = create_test_asset("offset", "2026-01-02T09:00:00+02:00")
    events = [
        StateEvent("offset", LifecycleState.EVALUATED, LifecycleState.APPROVED,
                   "2026-01-02T08:30:00Z", evidence_refs=()),
        StateEvent("offset", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
                   "2026-01-02T10:00:00+02:00", evidence_refs=()),
    ]
    early = manager.create_snapshot([asset], events,
        SnapshotQuery(as_of_timestamp="2026-01-02T08:00:00Z"))
    assert early.total_count == 1
    assert early.assets[0].lifecycle_state == LifecycleState.EVALUATED
    assert early.assets[0].first_evaluated_at == "2026-01-02T10:00:00+02:00"
    assert early.assets[0].approved_at is None
    late = manager.create_snapshot([asset], events,
        SnapshotQuery(as_of_timestamp="2026-01-02T10:30:00+02:00"))
    assert late.assets[0].lifecycle_state == LifecycleState.APPROVED
    assert late.assets[0].approved_at == "2026-01-02T08:30:00Z"


def test_timezone_normalization_preserves_equal_instant_input_order():
    manager = SnapshotManager()
    asset = create_test_asset("offset", "2026-01-02T07:00:00")
    events = [
        StateEvent("offset", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
                   "2026-01-02T10:00:00+02:00", evidence_refs=()),
        StateEvent("offset", LifecycleState.EVALUATED, LifecycleState.APPROVED,
                   "2026-01-02T08:00:00Z", evidence_refs=()),
    ]
    result = manager.create_snapshot([asset], events,
        SnapshotQuery(as_of_timestamp="2026-01-02T03:00:00-05:00"))
    assert result.total_count == 1
    assert result.assets[0].lifecycle_state == LifecycleState.APPROVED
    assert result.assets[0].first_evaluated_at == events[0].timestamp
    assert result.assets[0].approved_at == events[1].timestamp


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


def test_snapshot_reconstructs_many_assets_with_one_event_sort(monkeypatch):
    assets = [
        create_test_asset(f"F{i:03d}", "2026-01-01T00:00:00Z")
        for i in range(40)
    ]
    events = []
    for asset in assets:
        events.extend((
            StateEvent(
                factor_id=asset.factor_id,
                from_state=LifecycleState.REGISTERED,
                to_state=LifecycleState.REGISTERED,
                timestamp="2026-01-01T00:00:00Z",
                evidence_refs=(),
            ),
            StateEvent(
                factor_id=asset.factor_id,
                from_state=LifecycleState.REGISTERED,
                to_state=LifecycleState.EVALUATED,
                timestamp="2026-01-02T00:00:00Z",
                evidence_refs=("evaluation_bundle_ref",),
            ),
        ))

    real_sorted = sorted
    sort_calls = 0
    normalize_calls = 0
    real_normalize = snapshots_module._normalize_ts

    def counted_sorted(*args, **kwargs):
        nonlocal sort_calls
        sort_calls += 1
        return real_sorted(*args, **kwargs)

    def counted_normalize(timestamp):
        nonlocal normalize_calls
        normalize_calls += 1
        return real_normalize(timestamp)

    monkeypatch.setattr(snapshots_module, "sorted", counted_sorted, raising=False)
    monkeypatch.setattr(snapshots_module, "_normalize_ts", counted_normalize)

    result = SnapshotManager().create_snapshot(
        assets, events, SnapshotQuery(as_of_timestamp="2026-01-03T00:00:00Z")
    )

    assert len(result.assets) == len(assets)
    assert all(asset.lifecycle_state is LifecycleState.EVALUATED for asset in result.assets)
    assert sort_calls == 1
    # One query normalization, one sort key and one replay check per event,
    # plus one registration timestamp check per asset.
    assert normalize_calls <= 1 + 2 * len(events) + len(assets)


def test_snapshot_includes_as_of_ties_in_input_order_and_excludes_future_events():
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    events = [
        StateEvent("F001", LifecycleState.REGISTERED, LifecycleState.REGISTERED,
                   "2026-01-01T00:00:00Z", ()),
        StateEvent("F001", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
                   "2026-01-02T00:00:00Z", ("evaluation_bundle_ref",)),
        StateEvent("F001", LifecycleState.EVALUATED, LifecycleState.APPROVED,
                   "2026-01-02T00:00:00Z", ("gate_results",)),
        StateEvent("F001", LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY,
                   "2026-01-03T00:00:00Z", ("certification",)),
    ]
    manager = SnapshotManager()

    at_tie = manager.create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    ).assets[0]
    before_tie = manager.create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-01T23:59:59Z")
    ).assets[0]

    assert at_tie.lifecycle_state is LifecycleState.APPROVED
    assert at_tie.first_evaluated_at == "2026-01-02T00:00:00Z"
    assert at_tie.approved_at == "2026-01-02T00:00:00Z"
    assert at_tie.production_ready_at is None
    assert before_tie.lifecycle_state is LifecycleState.REGISTERED
    assert before_tie.first_evaluated_at is None
    assert before_tie.approved_at is None


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


def test_one_pass_snapshot_matches_legacy_replay_for_offsets_and_ties():
    import factor_assets.scripts.benchmark_snapshot_replay_20261001 as replay_benchmark

    asset = create_test_asset("offset", "2026-01-02T07:00:00")
    events = [
        StateEvent("offset", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
                   "2026-01-02T10:00:00+02:00", ()),
        StateEvent("offset", LifecycleState.EVALUATED, LifecycleState.APPROVED,
                   "2026-01-02T08:00:00Z", ()),
    ]
    query = SnapshotQuery(as_of_timestamp="2026-01-02T03:00:00-05:00")
    actual = SnapshotManager().create_snapshot([asset], events, query)
    reference = replay_benchmark._legacy_snapshot([asset], events, query)

    assert actual.assets == reference.assets
    assert actual.assets[0].lifecycle_state is LifecycleState.APPROVED
    assert actual.assets[0].first_evaluated_at == events[0].timestamp
    assert actual.assets[0].approved_at == events[1].timestamp


def test_snapshot_rereads_mutated_event_source_on_each_call():
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    events = [StateEvent(
        "F001", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
        "2026-01-02T00:00:00Z", ("evaluation_bundle_ref",),
    )]
    manager = SnapshotManager()
    first = manager.create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    ).assets[0]
    assert first.lifecycle_state is LifecycleState.EVALUATED

    added = StateEvent(
        "F001", LifecycleState.EVALUATED, LifecycleState.APPROVED,
        "2026-01-03T00:00:00Z", ("gate_results",),
    )
    events.append(added)
    second = manager.create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-03T00:00:00Z")
    ).assets[0]
    assert second.lifecycle_state is LifecycleState.APPROVED
    assert second.approved_at == added.timestamp
