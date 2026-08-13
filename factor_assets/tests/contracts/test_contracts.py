"""
Test contract immutability and basic validation.
"""

import pytest
from datetime import datetime, timezone

from factor_assets.contracts.envelope import ContractEnvelope
from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateTransition,
    StateEvent,
    is_legal_transition,
    get_required_evidence,
    validate_transition,
    LifecycleConflictError,
)
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.errors import MissingInputError


def test_contract_envelope_requires_core_fields():
    """Envelope must have schema_version, producer, producer_version, run_id."""
    with pytest.raises(ValueError, match="schema_version"):
        ContractEnvelope(
            schema_version="",
            producer="test",
            producer_version="1.0",
            created_at="2024-01-01T00:00:00Z",
            run_id="run_001",
            factor_ids=("F001",),
        )

    with pytest.raises(ValueError, match="producer"):
        ContractEnvelope(
            schema_version="v1",
            producer="",
            producer_version="1.0",
            created_at="2024-01-01T00:00:00Z",
            run_id="run_001",
            factor_ids=("F001",),
        )


def test_lifecycle_state_enum():
    """Test lifecycle state enumeration."""
    assert LifecycleState.REGISTERED.value == "REGISTERED"
    assert LifecycleState.EVALUATED.value == "EVALUATED"
    assert LifecycleState.APPROVED.value == "APPROVED"
    assert LifecycleState.PRODUCTION_READY.value == "PRODUCTION_READY"


def test_legal_transitions():
    """Test legal state transitions."""
    assert is_legal_transition(LifecycleState.REGISTERED, LifecycleState.EVALUATED)
    assert is_legal_transition(LifecycleState.EVALUATED, LifecycleState.APPROVED)
    assert is_legal_transition(LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY)
    # Re-evaluation is allowed
    assert is_legal_transition(LifecycleState.EVALUATED, LifecycleState.EVALUATED)


def test_illegal_transitions():
    """Test illegal state transitions."""
    # Cannot skip states
    assert not is_legal_transition(LifecycleState.REGISTERED, LifecycleState.APPROVED)
    assert not is_legal_transition(LifecycleState.REGISTERED, LifecycleState.PRODUCTION_READY)
    # Cannot go backwards
    assert not is_legal_transition(LifecycleState.APPROVED, LifecycleState.REGISTERED)
    assert not is_legal_transition(LifecycleState.EVALUATED, LifecycleState.REGISTERED)


def test_required_evidence():
    """Test required evidence for transitions."""
    evidence = get_required_evidence(LifecycleState.REGISTERED, LifecycleState.EVALUATED)
    assert "evaluation_bundle_ref" in evidence

    evidence = get_required_evidence(LifecycleState.EVALUATED, LifecycleState.APPROVED)
    assert "gate_results" in evidence


def test_validate_transition_with_missing_evidence():
    """Test transition validation fails with missing evidence."""
    with pytest.raises(LifecycleConflictError, match="Missing required evidence"):
        validate_transition(
            LifecycleState.REGISTERED,
            LifecycleState.EVALUATED,
            set()  # No evidence
        )


def test_validate_transition_with_sufficient_evidence():
    """Test transition validation succeeds with required evidence."""
    # Should not raise
    validate_transition(
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        {"evaluation_bundle_ref"}
    )


def test_state_event_immutability():
    """State events are immutable."""
    event = StateEvent(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        timestamp="2024-01-01T00:00:00Z",
        evidence_refs=("ev_001",),
    )

    with pytest.raises(Exception):  # FrozenInstanceError in dataclass
        event.factor_id = "F002"  # type: ignore


def test_parent_ref_requires_fields():
    """ParentRef must have factor_id and relationship."""
    with pytest.raises(ValueError, match="factor_id"):
        ParentRef(factor_id="", relationship="mutation")

    with pytest.raises(ValueError, match="relationship"):
        ParentRef(factor_id="F001", relationship="")


def test_lineage_ref_properties():
    """Test LineageRef convenience properties."""
    parent1 = ParentRef(factor_id="F001", relationship="mutation")
    parent2 = ParentRef(factor_id="F002", relationship="combination")

    lineage = LineageRef(
        factor_id="F003",
        parents=(parent1, parent2),
        campaign_id="camp_001",
    )

    assert lineage.has_parents
    assert lineage.parent_ids == ("F001", "F002")
    assert lineage.is_from_campaign


def test_evidence_ref_requires_core_fields():
    """EvidenceRef must have required fields."""
    with pytest.raises(ValueError, match="evidence_id"):
        EvidenceRef(
            evidence_id="",
            evaluation_run_id="run_001",
            metric_name="rank_ic",
            metric_version="1.0",
            timestamp="2024-01-01T00:00:00Z",
            factor_id="F001",
        )


def test_evidence_bundle_ref_properties():
    """Test EvidenceBundleRef properties."""
    bundle = EvidenceBundleRef(
        bundle_id="bundle_001",
        evaluation_run_id="run_001",
        factor_ids=("F001", "F002"),
        timestamp="2024-01-01T00:00:00Z",
        qe_version="0.1.0",
        warnings=("low_coverage", "few_assets"),
    )

    assert bundle.has_warnings
    assert len(bundle.warnings) == 2


def test_asset_metadata_requires_core_fields():
    """AssetMetadata must have factor_id, canonical_hash, frequency."""
    with pytest.raises(MissingInputError, match="factor_id"):
        AssetMetadata(
            factor_id="",
            canonical_repr="ts_rank(close, 20)",
            canonical_hash="abc123",
            frequency="daily",
            domains=("price",),
            timing="daily",
        )


def test_factor_asset_properties():
    """Test FactorAsset convenience properties."""
    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )

    lineage = LineageRef(factor_id="F001", parents=())

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.EVALUATED,
        registered_at="2024-01-01T00:00:00Z",
        first_evaluated_at="2024-01-02T00:00:00Z",
    )

    assert asset.factor_id == "F001"
    assert not asset.is_registered
    assert asset.is_evaluated
    assert not asset.is_approved
    assert not asset.is_production_ready
    assert not asset.has_evidence  # No evidence_ref attached
    assert not asset.has_lineage  # No parents
