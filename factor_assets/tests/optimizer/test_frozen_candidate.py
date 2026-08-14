"""Tests for frozen candidate."""

import pytest
from factor_assets.optimizer.frozen_candidate import (
    FrozenCandidate,
    FrozenState,
    ContaminationKind,
    FrozenCandidateStateMachine,
    LEGAL_TRANSITIONS,
)


def test_frozen_candidate_creation():
    """Test frozen candidate creation."""
    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20, "alpha": 0.5},
    )

    assert candidate.candidate_id == "cand_001"
    assert candidate.asset_id == "asset_123"
    assert candidate.state == FrozenState.PROPOSED
    assert candidate.content_hash != ""
    assert candidate.created_at is not None


def test_frozen_candidate_hash_computation():
    """Test content hash computation."""
    candidate1 = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
    )

    candidate2 = FrozenCandidate(
        candidate_id="cand_002",  # Different ID
        asset_id="asset_123",
        parameters={"window": 20},
    )

    # Same asset and parameters should have same hash
    assert candidate1.content_hash == candidate2.content_hash


def test_frozen_candidate_hash_verification():
    """Test hash verification."""
    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
    )

    assert candidate.verify_hash()


def test_frozen_candidate_is_sealed():
    """Test sealed state detection."""
    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
        state=FrozenState.PROPOSED,
    )
    assert not candidate.is_sealed()

    sealed_candidate = FrozenCandidate(
        candidate_id="cand_002",
        asset_id="asset_123",
        parameters={"window": 20},
        state=FrozenState.TEST_SEALED,
    )
    assert sealed_candidate.is_sealed()


def test_frozen_candidate_contamination():
    """Test contamination flag."""
    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
        contamination_flags={ContaminationKind.PARAMETER_TUNING},
    )

    assert candidate.is_contaminated()
    assert ContaminationKind.PARAMETER_TUNING in candidate.contamination_flags


def test_frozen_candidate_serialization():
    """Test serialization."""
    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
    )

    data = candidate.to_dict()
    assert data["candidate_id"] == "cand_001"
    assert data["state"] == "proposed"

    restored = FrozenCandidate.from_dict(data)
    assert restored.candidate_id == candidate.candidate_id
    assert restored.content_hash == candidate.content_hash


def test_state_machine_register():
    """Test candidate registration."""
    sm = FrozenCandidateStateMachine()

    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
    )

    sm.register(candidate)
    assert sm.get_candidate("cand_001") is not None


def test_state_machine_duplicate_registration():
    """Test duplicate registration fails."""
    sm = FrozenCandidateStateMachine()

    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
    )

    sm.register(candidate)

    with pytest.raises(ValueError):
        sm.register(candidate)


def test_state_machine_legal_transition():
    """Test legal state transition."""
    sm = FrozenCandidateStateMachine()

    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
    )

    sm.register(candidate)

    new_candidate = sm.transition("cand_001", FrozenState.VALIDATED, "Validation passed")
    assert new_candidate.state == FrozenState.VALIDATED
    assert len(new_candidate.state_history) == 1


def test_state_machine_illegal_transition():
    """Test illegal state transition fails."""
    sm = FrozenCandidateStateMachine()

    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
        state=FrozenState.PROPOSED,
    )

    sm.register(candidate)

    with pytest.raises(ValueError):
        sm.transition("cand_001", FrozenState.ADMITTED, "Skip validation")


def test_state_machine_contamination_flag():
    """Test contamination flagging."""
    sm = FrozenCandidateStateMachine()

    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
        state=FrozenState.TEST_SEALED,
    )

    sm.register(candidate)

    contaminated = sm.flag_contamination(
        "cand_001",
        ContaminationKind.PARAMETER_TUNING,
        "Used validation data for tuning",
    )

    assert contaminated.state == FrozenState.TEST_CONTAMINATED
    assert ContaminationKind.PARAMETER_TUNING in contaminated.contamination_flags


def test_state_machine_contamination_wrong_state():
    """Test contamination flag from wrong state fails."""
    sm = FrozenCandidateStateMachine()

    candidate = FrozenCandidate(
        candidate_id="cand_001",
        asset_id="asset_123",
        parameters={"window": 20},
        state=FrozenState.PROPOSED,
    )

    sm.register(candidate)

    with pytest.raises(ValueError):
        sm.flag_contamination("cand_001", ContaminationKind.PARAMETER_TUNING)


def test_state_machine_get_by_state():
    """Test getting candidates by state."""
    sm = FrozenCandidateStateMachine()

    sm.register(FrozenCandidate("cand_001", "asset_123", {}, state=FrozenState.PROPOSED))
    sm.register(FrozenCandidate("cand_002", "asset_124", {}, state=FrozenState.VALIDATED))
    sm.register(FrozenCandidate("cand_003", "asset_125", {}, state=FrozenState.PROPOSED))

    proposed = sm.get_candidates_by_state(FrozenState.PROPOSED)
    assert len(proposed) == 2


def test_state_machine_get_contaminated():
    """Test getting contaminated candidates."""
    sm = FrozenCandidateStateMachine()

    clean = FrozenCandidate("cand_001", "asset_123", {})
    contaminated = FrozenCandidate(
        "cand_002",
        "asset_124",
        {},
        contamination_flags={ContaminationKind.PARAMETER_TUNING},
    )

    sm.register(clean)
    sm.register(contaminated)

    contaminated_list = sm.get_contaminated_candidates()
    assert len(contaminated_list) == 1
    assert contaminated_list[0].candidate_id == "cand_002"
