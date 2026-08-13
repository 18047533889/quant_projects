"""
Test AssetRepository append-only operations and lifecycle management.
"""

import pytest

from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.lifecycle import LifecycleState, LifecycleConflictError
from factor_assets.registry import (
    AssetRepository,
    DuplicateIdentityError,
    AssetNotFoundError,
)


def test_repository_register_new_asset():
    """Test registering a new factor asset."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    asset = repo.register(metadata, lineage)

    assert asset.factor_id == "F001"
    assert asset.lifecycle_state == LifecycleState.REGISTERED
    assert asset.registered_at
    assert repo.exists("F001")


def test_repository_rejects_duplicate_factor_id():
    """Repository should reject duplicate factor_id."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    repo.register(metadata, lineage)

    # Try to register again
    with pytest.raises(DuplicateIdentityError, match="already registered"):
        repo.register(metadata, lineage)


def test_repository_rejects_duplicate_canonical_hash():
    """Repository should detect canonical hash collisions."""
    repo = AssetRepository()

    metadata1 = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage1 = LineageRef(factor_id="F001", parents=())

    repo.register(metadata1, lineage1)

    # Different factor_id but same canonical_hash
    metadata2 = AssetMetadata(
        factor_id="F002",
        canonical_repr="ts_rank(close, 20)",  # Same semantics
        canonical_hash="abc123",  # Same hash!
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage2 = LineageRef(factor_id="F002", parents=())

    with pytest.raises(DuplicateIdentityError, match="canonical_hash"):
        repo.register(metadata2, lineage2)


def test_repository_get_asset():
    """Test retrieving assets from repository."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    original = repo.register(metadata, lineage)
    retrieved = repo.get("F001")

    assert retrieved.factor_id == original.factor_id
    assert retrieved.metadata.canonical_hash == original.metadata.canonical_hash


def test_repository_get_nonexistent_asset():
    """Repository should raise AssetNotFoundError for missing assets."""
    repo = AssetRepository()

    with pytest.raises(AssetNotFoundError, match="not found"):
        repo.get("F999")


def test_repository_find_by_hash():
    """Test finding assets by canonical hash."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    repo.register(metadata, lineage)

    found = repo.find_by_hash("abc123")
    assert found is not None
    assert found.factor_id == "F001"

    not_found = repo.find_by_hash("xyz999")
    assert not_found is None


def test_repository_transition_state():
    """Test lifecycle state transitions."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    asset = repo.register(metadata, lineage)
    assert asset.lifecycle_state == LifecycleState.REGISTERED

    # Transition to EVALUATED
    updated = repo.transition(
        "F001",
        LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )

    assert updated.lifecycle_state == LifecycleState.EVALUATED
    assert updated.first_evaluated_at is not None


def test_repository_transition_requires_evidence():
    """Test that transitions require appropriate evidence."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    repo.register(metadata, lineage)

    # Try to transition without evidence
    with pytest.raises(LifecycleConflictError, match="Missing required evidence"):
        repo.transition("F001", LifecycleState.EVALUATED, evidence_refs=())


def test_repository_rejects_illegal_transitions():
    """Test that illegal state transitions are rejected."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    repo.register(metadata, lineage)

    # Try to skip EVALUATED and go directly to APPROVED
    with pytest.raises(LifecycleConflictError, match="Illegal transition"):
        repo.transition(
            "F001",
            LifecycleState.APPROVED,
            evidence_refs=("gate_results",),
        )


def test_repository_list_by_state():
    """Test listing assets by lifecycle state."""
    repo = AssetRepository()

    # Register three assets
    for i in range(3):
        metadata = AssetMetadata(
            factor_id=f"F{i:03d}",
            canonical_repr=f"factor_{i}",
            canonical_hash=f"hash_{i:03d}",
            frequency="daily",
            domains=("price",),
            timing="daily",
        )
        lineage = LineageRef(factor_id=f"F{i:03d}", parents=())
        repo.register(metadata, lineage)

    # Transition one to EVALUATED
    repo.transition("F001", LifecycleState.EVALUATED, evidence_refs=("evaluation_bundle_ref",))

    registered = repo.list_by_state(LifecycleState.REGISTERED)
    evaluated = repo.list_by_state(LifecycleState.EVALUATED)

    assert len(registered) == 2
    assert len(evaluated) == 1
    assert evaluated[0].factor_id == "F001"


def test_repository_records_events():
    """Test that state transitions are recorded as events."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    repo.register(metadata, lineage)

    # Should have initial registration event
    events = repo.get_events("F001")
    assert len(events) == 1
    assert events[0].factor_id == "F001"
    assert events[0].to_state == LifecycleState.REGISTERED

    # Transition and check events
    repo.transition(
        "F001",
        LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
        decision_id="dec_001",
        notes="Test transition",
    )

    events = repo.get_events("F001")
    assert len(events) == 2
    assert events[1].to_state == LifecycleState.EVALUATED
    assert events[1].decision_id == "dec_001"
    assert events[1].notes == "Test transition"


def test_repository_stats():
    """Test repository statistics."""
    repo = AssetRepository()

    # Initially empty
    stats = repo.stats()
    assert stats.total_assets == 0
    assert stats.total_events == 0

    # Register assets
    for i in range(5):
        metadata = AssetMetadata(
            factor_id=f"F{i:03d}",
            canonical_repr=f"factor_{i}",
            canonical_hash=f"hash_{i:03d}",
            frequency="daily",
            domains=("price",),
            timing="daily",
        )
        lineage = LineageRef(factor_id=f"F{i:03d}", parents=())
        repo.register(metadata, lineage)

    stats = repo.stats()
    assert stats.total_assets == 5
    assert stats.by_state[LifecycleState.REGISTERED] == 5
    assert stats.total_events == 5  # One registration event per asset
    assert stats.first_registration is not None
    assert stats.last_registration is not None


def test_repository_complete_lifecycle_flow():
    """Test complete lifecycle flow through repository."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    # Register
    asset = repo.register(metadata, lineage)
    assert asset.lifecycle_state == LifecycleState.REGISTERED

    # Evaluate
    asset = repo.transition(
        "F001",
        LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )
    assert asset.lifecycle_state == LifecycleState.EVALUATED
    assert asset.first_evaluated_at is not None

    # Approve
    asset = repo.transition(
        "F001",
        LifecycleState.APPROVED,
        evidence_refs=("gate_results",),
    )
    assert asset.lifecycle_state == LifecycleState.APPROVED
    assert asset.approved_at is not None

    # Production ready
    asset = repo.transition(
        "F001",
        LifecycleState.PRODUCTION_READY,
        evidence_refs=("certification",),
    )
    assert asset.lifecycle_state == LifecycleState.PRODUCTION_READY
    assert asset.production_ready_at is not None

    # Check event history
    events = repo.get_events("F001")
    assert len(events) == 4  # registration + 3 transitions
