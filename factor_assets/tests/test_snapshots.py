"""
Test point-in-time snapshot manager and time-travel queries.
"""

from dataclasses import replace

import pytest
from datetime import datetime, timezone, timedelta

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import HealthState, LifecycleState, StateEvent, StateEventKind
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository
from factor_assets.registry.health_history import HealthHistoryCoverage
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


def health_event(
    timestamp: str,
    health_from: HealthState,
    health_to: HealthState,
    lifecycle_state: LifecycleState = LifecycleState.REGISTERED,
) -> StateEvent:
    return StateEvent(
        factor_id="F001",
        from_state=lifecycle_state,
        to_state=lifecycle_state,
        timestamp=timestamp,
        evidence_refs=(),
        event_kind=StateEventKind.HEALTH_TRANSITION,
        health_from=health_from,
        health_to=health_to,
    )


def test_snapshot_reconstructs_health_from_repository_events_as_of(tmp_path):
    repo = SQLiteLifecycleRepository(tmp_path / "snapshot-health.sqlite")
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    repo.register(asset.metadata, asset.lineage)
    repo.commit_transition(
        "F001", LifecycleState.EVALUATED, evidence_refs=("evaluation_bundle_ref",)
    )
    repo.update_asset_health("F001", HealthState.DEPRECATED)
    current_asset = repo.get("F001")
    events = repo.get_events("F001")
    health_event_record = next(
        event for event in events
        if event.event_kind is StateEventKind.HEALTH_TRANSITION
    )
    before_health = (
        datetime.fromisoformat(health_event_record.timestamp) - timedelta(microseconds=1)
    ).isoformat()
    manager = SnapshotManager()

    before = manager.create_snapshot(
        [current_asset], events, SnapshotQuery(as_of_timestamp=before_health)
    ).assets[0]
    after = manager.create_snapshot(
        [current_asset], events,
        SnapshotQuery(as_of_timestamp=health_event_record.timestamp),
    ).assets[0]

    assert current_asset.health_state is HealthState.DEPRECATED
    assert before.health_state is HealthState.ACTIVE
    assert after.health_state is HealthState.DEPRECATED
    assert after.lifecycle_state is LifecycleState.EVALUATED
    assert after.first_evaluated_at is not None


def test_snapshot_does_not_treat_health_event_as_lifecycle_or_milestone():
    asset = replace(
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        health_state=HealthState.DEPRECATED,
    )
    events = [
        StateEvent("F001", LifecycleState.REGISTERED, LifecycleState.REGISTERED,
                   "2026-01-01T00:00:00Z", ()),
        health_event(
            "2026-01-02T00:00:00Z", HealthState.ACTIVE, HealthState.DEPRECATED,
            lifecycle_state=LifecycleState.EVALUATED,
        ),
    ]
    historical = SnapshotManager().create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-03T00:00:00Z")
    ).assets[0]
    assert historical.lifecycle_state is LifecycleState.REGISTERED
    assert historical.first_evaluated_at is None
    assert historical.health_state is HealthState.DEPRECATED


@pytest.mark.parametrize("health_first", [False, True])
def test_equal_instant_health_and_lifecycle_events_replay_independently(health_first):
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    registration = StateEvent(
        "F001", LifecycleState.REGISTERED, LifecycleState.REGISTERED,
        "2026-01-01T00:00:00Z", (),
    )
    lifecycle = StateEvent(
        "F001", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
        "2026-01-02T00:00:00Z", ("evaluation_bundle_ref",),
    )
    # Same instant expressed with different offsets; health carries a stale
    # lifecycle snapshot, but only its typed fields are replayed.
    health = health_event(
        "2026-01-02T01:00:00+01:00", HealthState.ACTIVE, HealthState.DEPRECATED,
        lifecycle_state=LifecycleState.REGISTERED,
    )
    events = [registration, health, lifecycle] if health_first else [registration, lifecycle, health]
    historical = SnapshotManager().create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    ).assets[0]
    assert historical.lifecycle_state is LifecycleState.EVALUATED
    assert historical.first_evaluated_at == "2026-01-02T00:00:00Z"
    assert historical.health_state is HealthState.DEPRECATED


def test_snapshot_uses_registration_health_default_without_health_events():
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    historical = SnapshotManager().create_snapshot(
        [asset], [], SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    ).assets[0]
    assert historical.health_state is HealthState.ACTIVE


def test_snapshot_rejects_non_active_current_health_without_health_history():
    asset = replace(
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        health_state=HealthState.RETIRED,
    )
    with pytest.raises(ValueError, match="missing health history"):
        SnapshotManager().create_snapshot(
            [asset], [], SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
        )


def test_unmatched_asset_without_health_history_does_not_fail_other_snapshot():
    retired = replace(
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        health_state=HealthState.RETIRED,
    )
    active = create_test_asset("F002", "2026-01-01T00:00:00Z")
    result = SnapshotManager().create_snapshot(
        [retired, active], [],
        SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z", factor_id="F002"),
    )
    assert [item.factor_id for item in result.assets] == ["F002"]
    assert result.assets[0].health_state is HealthState.ACTIVE


def test_snapshot_rejects_inconsistent_health_event_chain():
    asset = replace(
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        health_state=HealthState.RETIRED,
    )
    events = [
        health_event("2026-01-02T00:00:00Z", HealthState.ACTIVE, HealthState.DEPRECATED),
        health_event("2026-01-03T00:00:00Z", HealthState.ACTIVE, HealthState.RETIRED),
    ]
    with pytest.raises(ValueError, match="Inconsistent health event history"):
        SnapshotManager().create_snapshot(
            [asset], events, SnapshotQuery(as_of_timestamp="2026-01-04T00:00:00Z")
        )


def test_snapshot_uses_first_health_from_before_future_first_transition():
    asset = replace(
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        health_state=HealthState.RETIRED,
    )
    event = health_event(
        "2026-01-03T00:00:00Z", HealthState.DEPRECATED, HealthState.RETIRED
    )
    result = SnapshotManager().create_snapshot(
        [asset], [event], SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    )
    assert result.assets[0].health_state is HealthState.DEPRECATED


def test_strict_health_coverage_rejects_truncated_suffix_but_replays_complete_history():
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    complete_events = [
        health_event(
            "2026-01-02T00:00:00Z", HealthState.ACTIVE, HealthState.DEPRECATED
        ),
        health_event(
            "2026-01-04T00:00:00Z", HealthState.DEPRECATED, HealthState.ACTIVE
        ),
    ]
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F001": HealthState.ACTIVE}
    )
    manager = SnapshotManager()
    query = SnapshotQuery(as_of_timestamp="2026-01-01T12:00:00Z")

    result = manager.create_snapshot(
        [asset], complete_events, query, health_history_coverage=coverage
    )
    assert result.assets[0].health_state is HealthState.ACTIVE

    with pytest.raises(ValueError, match="Inconsistent covered health event history"):
        manager.create_snapshot(
            [asset], complete_events[1:], query, health_history_coverage=coverage
        )


def test_strict_health_coverage_baseline_overrides_current_nonactive_default():
    asset = replace(
        create_test_asset("F001", "2026-01-01T00:00:00Z"),
        health_state=HealthState.RETIRED,
    )
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F001": HealthState.ACTIVE}
    )
    result = SnapshotManager().create_snapshot(
        [asset], [], SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z"),
        health_history_coverage=coverage,
    ).assets[0]
    assert result.health_state is HealthState.ACTIVE

@pytest.mark.parametrize(
    "as_of",
    ["2025-12-31T23:59:59Z", "2026-01-05T00:00:01Z"],
)
def test_strict_health_coverage_rejects_snapshot_cut_outside_interval(as_of):
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F001": HealthState.ACTIVE}
    )
    with pytest.raises(ValueError, match="outside health history coverage interval"):
        SnapshotManager().create_snapshot(
            [create_test_asset("F001", "2026-01-01T00:00:00Z")],
            [],
            SnapshotQuery(as_of_timestamp=as_of),
            health_history_coverage=coverage,
        )


def test_strict_health_coverage_requires_baseline_for_each_matched_asset():
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {}
    )
    with pytest.raises(ValueError, match="missing health coverage baseline for factor"):
        SnapshotManager().create_snapshot(
            [create_test_asset("F001", "2026-01-01T00:00:00Z")],
            [],
            SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z"),
            health_history_coverage=coverage,
        )


@pytest.mark.parametrize(
    "covered_from,covered_through",
    [
        ("bad", "2026-01-05T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "bad"),
        ("2026-01-05T00:00:00Z", "2026-01-01T00:00:00Z"),
        ("0001-01-01T00:00:00+01:00", "2026-01-05T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "9999-12-31T23:59:59-01:00"),
    ],
)
def test_health_history_coverage_validates_interval_timestamps(covered_from, covered_through):
    with pytest.raises(ValueError):
        HealthHistoryCoverage(covered_from, covered_through, {})


def test_health_history_coverage_rejects_duplicate_and_untyped_baselines_and_copies_mapping():
    with pytest.raises(ValueError, match="duplicate health baseline"):
        HealthHistoryCoverage(
            "2026-01-01T00:00:00Z",
            "2026-01-05T00:00:00Z",
            [("F001", HealthState.ACTIVE), ("F001", HealthState.RETIRED)],
        )
    with pytest.raises(TypeError, match="must be HealthState"):
        HealthHistoryCoverage(
            "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F001": "ACTIVE"}
        )

    source = {"F001": HealthState.ACTIVE}
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", source
    )
    source["F001"] = HealthState.RETIRED
    assert coverage.baseline_health_by_factor["F001"] is HealthState.ACTIVE
    with pytest.raises(TypeError):
        coverage.baseline_health_by_factor["F001"] = HealthState.RETIRED


def test_strict_health_coverage_only_checks_matched_assets_registered_by_cut():
    excluded = create_test_asset("F001", "2026-01-01T00:00:00Z")
    matched = create_test_asset("F002", "2026-01-01T00:00:00Z")
    future_registered = create_test_asset("F003", "2026-01-03T00:00:00Z")
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F002": HealthState.ACTIVE}
    )
    invalid_excluded_history = [
        health_event("2026-01-02T00:00:00Z", HealthState.RETIRED, HealthState.ACTIVE),
    ]

    result = SnapshotManager().create_snapshot(
        [excluded, matched, future_registered],
        invalid_excluded_history,
        SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z", factor_id="F002"),
        health_history_coverage=coverage,
    )

    assert [asset.factor_id for asset in result.assets] == ["F002"]
    assert result.assets[0].health_state is HealthState.ACTIVE


def test_strict_health_coverage_checks_boundary_ties_inclusive_cut_and_future_chain():
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    events = [
        # Valid pre-baseline events are outside this declared coverage interval.
        health_event(
            "2025-12-31T00:00:00Z", HealthState.RETIRED, HealthState.DEPRECATED
        ),
        health_event(
            "2026-01-01T00:00:00Z", HealthState.ACTIVE, HealthState.DEPRECATED
        ),
        health_event(
            "2026-01-02T01:00:00+01:00", HealthState.DEPRECATED, HealthState.RETIRED
        ),
        health_event(
            "2026-01-02T00:00:00Z", HealthState.RETIRED, HealthState.ACTIVE
        ),
        health_event(
            "2026-01-04T00:00:00Z", HealthState.ACTIVE, HealthState.DEPRECATED
        ),
    ]
    lifecycle = StateEvent(
        "F001", LifecycleState.REGISTERED, LifecycleState.EVALUATED,
        "2026-01-02T00:00:00Z", ("evaluation_bundle_ref",),
    )
    events.insert(2, lifecycle)
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F001": HealthState.ACTIVE}
    )

    result = SnapshotManager().create_snapshot(
        [asset], events, SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z"),
        health_history_coverage=coverage,
    ).assets[0]

    assert result.health_state is HealthState.ACTIVE
    assert result.lifecycle_state is LifecycleState.EVALUATED


def test_strict_health_coverage_rejects_bad_matched_event_timestamps():
    coverage = HealthHistoryCoverage(
        "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", {"F001": HealthState.ACTIVE}
    )
    with pytest.raises(ValueError, match="health event timestamp for F001"):
        SnapshotManager().create_snapshot(
            [create_test_asset("F001", "2026-01-01T00:00:00Z")],
            [health_event("not-a-timestamp", HealthState.ACTIVE, HealthState.DEPRECATED)],
            SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z"),
            health_history_coverage=coverage,
        )

def test_truncated_history_can_misstate_health_for_current_active_asset():
    """Current ACTIVE plus an event suffix cannot prove prior history complete.

    This records the present API ambiguity, not a desired inference rule: the
    same current asset and truncated event list reconstruct differently from
    the complete history, and the caller has no completeness marker to expose
    that difference.
    """
    asset = create_test_asset("F001", "2026-01-01T00:00:00Z")
    complete_events = [
        health_event(
            "2026-01-02T00:00:00Z", HealthState.ACTIVE, HealthState.DEPRECATED
        ),
        health_event(
            "2026-01-04T00:00:00Z", HealthState.DEPRECATED, HealthState.ACTIVE
        ),
    ]
    truncated_events = complete_events[1:]
    query = SnapshotQuery(as_of_timestamp="2026-01-01T12:00:00Z")
    manager = SnapshotManager()

    complete = manager.create_snapshot([asset], complete_events, query).assets[0]
    truncated = manager.create_snapshot([asset], truncated_events, query).assets[0]

    assert asset.health_state is HealthState.ACTIVE
    assert complete.health_state is HealthState.ACTIVE
    # The future suffix's health_from is treated as an earlier baseline. With
    # no coverage metadata, the replay cannot know the Jan 2 edge was omitted.
    assert truncated.health_state is HealthState.DEPRECATED
