"""
Test lifecycle state machine validation.
"""

import pytest

from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateEvent,
    LifecycleConflictError,
)
from factor_assets.lifecycle import StateMachine


def test_state_machine_can_transition():
    """Test state machine transition validation."""
    sm = StateMachine()

    # Legal transitions
    assert sm.can_transition(LifecycleState.REGISTERED, LifecycleState.EVALUATED)
    assert sm.can_transition(LifecycleState.EVALUATED, LifecycleState.APPROVED)
    assert sm.can_transition(LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY)

    # Illegal transitions
    assert not sm.can_transition(LifecycleState.REGISTERED, LifecycleState.APPROVED)
    assert not sm.can_transition(LifecycleState.APPROVED, LifecycleState.REGISTERED)


def test_state_machine_validate_with_evidence():
    """Test state machine validation with evidence."""
    sm = StateMachine()

    # Should succeed with required evidence
    sm.validate(
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        {"evaluation_bundle_ref"},
    )


def test_state_machine_validate_without_evidence():
    """Test state machine validation fails without evidence."""
    sm = StateMachine()

    with pytest.raises(LifecycleConflictError, match="Missing required evidence"):
        sm.validate(
            LifecycleState.REGISTERED,
            LifecycleState.EVALUATED,
            set(),  # Missing evidence
        )


def test_state_machine_validate_illegal_transition():
    """Test state machine rejects illegal transitions."""
    sm = StateMachine()

    with pytest.raises(LifecycleConflictError, match="Illegal transition"):
        sm.validate(
            LifecycleState.REGISTERED,
            LifecycleState.APPROVED,
            {"any_evidence"},
        )


def test_state_machine_create_event():
    """Test state machine event creation."""
    sm = StateMachine()

    event = sm.create_event(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        timestamp="2024-01-01T00:00:00Z",
        evidence_refs=("ev_001",),
        decision_id="dec_001",
        policy_version="v1.0",
        actor="test_system",
        notes="Initial evaluation",
    )

    assert isinstance(event, StateEvent)
    assert event.factor_id == "F001"
    assert event.from_state == LifecycleState.REGISTERED
    assert event.to_state == LifecycleState.EVALUATED
    assert event.evidence_refs == ("ev_001",)
    assert event.decision_id == "dec_001"


def test_complete_lifecycle_flow():
    """Test complete lifecycle state flow through state machine."""
    sm = StateMachine()

    # REGISTERED -> EVALUATED
    assert sm.can_transition(LifecycleState.REGISTERED, LifecycleState.EVALUATED)
    sm.validate(
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        {"evaluation_bundle_ref"},
    )

    # EVALUATED -> APPROVED
    assert sm.can_transition(LifecycleState.EVALUATED, LifecycleState.APPROVED)
    sm.validate(
        LifecycleState.EVALUATED,
        LifecycleState.APPROVED,
        {"gate_results"},
    )

    # APPROVED -> PRODUCTION_READY
    assert sm.can_transition(LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY)
    sm.validate(
        LifecycleState.APPROVED,
        LifecycleState.PRODUCTION_READY,
        {"certification"},
    )

    # PRODUCTION_READY -> DEPRECATED
    assert sm.can_transition(LifecycleState.PRODUCTION_READY, LifecycleState.DEPRECATED)
    sm.validate(
        LifecycleState.PRODUCTION_READY,
        LifecycleState.DEPRECATED,
        set(),  # No evidence required
    )
