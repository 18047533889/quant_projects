"""Regression coverage for timestamps whose UTC offset crosses datetime limits."""

import pytest

from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.lifecycle import (
    HealthState,
    LifecycleState,
    StateEvent,
    StateEventKind,
)
from factor_assets.contracts.lineage import LineageRef
from factor_assets.registry.health_history import HealthHistoryCoverage
from factor_assets.registry.snapshots import SnapshotManager, SnapshotQuery, _normalize_ts


_EXTREME_TIMESTAMPS = (
    "0001-01-01T00:00:00+01:00",
    "9999-12-31T23:59:59-01:00",
)
_COVERAGE = HealthHistoryCoverage(
    "0001-01-01T00:00:00Z",
    "9999-12-31T23:59:59Z",
    {"F001": HealthState.ACTIVE},
)


def _asset(factor_id="F001"):
    return FactorAsset(
        metadata=AssetMetadata(
            factor_id, f"factor_{factor_id}", f"hash_{factor_id}",
            "daily", ("price",), "daily",
        ),
        lineage=LineageRef(factor_id, ()),
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at="2026-01-01T00:00:00Z",
    )


def _health_event(factor_id, timestamp):
    return StateEvent(
        factor_id=factor_id,
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.REGISTERED,
        timestamp=timestamp,
        evidence_refs=(),
        event_kind=StateEventKind.HEALTH_TRANSITION,
        health_from=HealthState.ACTIVE,
        health_to=HealthState.DEPRECATED,
    )


@pytest.mark.parametrize("timestamp", _EXTREME_TIMESTAMPS)
def test_normalize_ts_keeps_legacy_raw_fallback_for_utc_overflow(timestamp):
    assert _normalize_ts(timestamp) == timestamp


@pytest.mark.parametrize("timestamp", _EXTREME_TIMESTAMPS)
def test_covered_snapshot_rejects_utc_overflow_query_as_value_error(timestamp):
    query = SnapshotQuery(as_of_timestamp=timestamp)
    with pytest.raises(ValueError, match="as_of_timestamp is outside"):
        SnapshotManager().create_snapshot(
            [_asset()], [], query, health_history_coverage=_COVERAGE
        )


@pytest.mark.parametrize("timestamp", _EXTREME_TIMESTAMPS)
def test_covered_snapshot_rejects_matched_event_utc_overflow_as_value_error(timestamp):
    query = SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    with pytest.raises(ValueError, match="health event timestamp for F001 is outside"):
        SnapshotManager().create_snapshot(
            [_asset()], [_health_event("F001", timestamp)], query,
            health_history_coverage=_COVERAGE,
        )


@pytest.mark.parametrize("timestamp", _EXTREME_TIMESTAMPS)
def test_covered_snapshot_ignores_unrelated_event_utc_overflow(timestamp):
    query = SnapshotQuery(as_of_timestamp="2026-01-02T00:00:00Z")
    result = SnapshotManager().create_snapshot(
        [_asset()], [_health_event("UNMATCHED", timestamp)], query,
        health_history_coverage=_COVERAGE,
    )
    assert result.assets[0].health_state is HealthState.ACTIVE
