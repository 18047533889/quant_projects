"""
Test lifecycle orchestration and state transition validation.
"""

import pytest

from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.lifecycle import LifecycleState, LifecycleConflictError
from factor_assets.contracts.lineage import LineageRef
from factor_assets.registry import AssetRepository
from factor_assets.registry.lifecycle import (
    LifecycleOrchestrator,
    TransitionRequest,
    TransitionResult,
)


def _repository_with_asset(factor_id="F001"):
    repository = AssetRepository()
    repository.register(
        AssetMetadata(
            factor_id=factor_id,
            canonical_repr="ts_rank(close, 20)",
            canonical_hash=f"hash-{factor_id}",
            frequency="daily",
            domains=("price",),
            timing="daily",
        ),
        LineageRef(factor_id=factor_id, parents=()),
    )
    return repository


def _orchestrator_with_asset(factor_id="F001", state=LifecycleState.REGISTERED):
    repository = _repository_with_asset(factor_id)
    if state != LifecycleState.REGISTERED:
        repository.transition(
            factor_id,
            LifecycleState.EVALUATED,
            evidence_refs=("evaluation_bundle_ref",),
        )
    return LifecycleOrchestrator(repository), repository


def test_orchestrator_validate_legal_transition():
    """Test validation of legal state transitions."""
    orchestrator, repository = _orchestrator_with_asset()

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )

    # Should not raise
    orchestrator.validate_transition(request)


def test_orchestrator_reject_illegal_transition():
    """Test rejection of illegal state transitions."""
    orchestrator, repository = _orchestrator_with_asset()

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.APPROVED,  # Skip EVALUATED
        evidence_refs=("gate_results",),
    )

    with pytest.raises(LifecycleConflictError, match="Illegal transition"):
        orchestrator.validate_transition(request)


def test_orchestrator_reject_missing_evidence():
    """Test rejection when required evidence is missing."""
    orchestrator, repository = _orchestrator_with_asset()

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=(),  # Missing required evidence
    )

    with pytest.raises(LifecycleConflictError, match="Missing required evidence"):
        orchestrator.validate_transition(request)


def test_orchestrator_execute_transition():
    """Test executing a valid transition."""
    orchestrator, repository = _orchestrator_with_asset()

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
        decision_id="dec_001",
        notes="Initial evaluation",
    )

    result = orchestrator.execute_transition(request)

    assert isinstance(result, TransitionResult)
    assert result.event.factor_id == "F001"
    assert result.event.from_state == LifecycleState.REGISTERED
    assert result.event.to_state == LifecycleState.EVALUATED
    assert result.event.decision_id == "dec_001"
    assert result.event.notes == "Initial evaluation"
    assert result.timestamp


def test_orchestrator_event_listener():
    """Test that event listeners are called on transitions."""
    orchestrator, repository = _orchestrator_with_asset()

    received_events = []

    def listener(event):
        received_events.append(event)

    orchestrator.add_event_listener(listener)

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )

    orchestrator.execute_transition(request)

    assert len(received_events) == 1
    assert received_events[0].factor_id == "F001"
    assert received_events[0].to_state == LifecycleState.EVALUATED


def test_orchestrator_transition_hook():
    """Test that transition-specific hooks are called."""
    orchestrator, repository = _orchestrator_with_asset()

    hook_called = []

    def hook(event):
        hook_called.append(event.factor_id)

    orchestrator.add_transition_hook(
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        hook
    )

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )

    orchestrator.execute_transition(request)

    assert "F001" in hook_called


def test_orchestrator_hook_not_called_for_different_transition():
    """Test that hooks are only called for their specific transition."""
    orchestrator, repository = _orchestrator_with_asset()

    hook_called = []

    def hook(event):
        hook_called.append(event.factor_id)

    # Register hook for EVALUATED -> APPROVED
    orchestrator.add_transition_hook(
        LifecycleState.EVALUATED,
        LifecycleState.APPROVED,
        hook
    )

    # Execute REGISTERED -> EVALUATED
    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )

    orchestrator.execute_transition(request)

    # Hook should not be called
    assert len(hook_called) == 0


def test_orchestrator_get_legal_next_states():
    """Test getting legal next states from current state."""
    orchestrator, repository = _orchestrator_with_asset()

    # From REGISTERED, can go to EVALUATED
    next_states = orchestrator.get_legal_next_states(LifecycleState.REGISTERED)
    assert LifecycleState.EVALUATED in next_states
    assert LifecycleState.APPROVED not in next_states

    # From EVALUATED, can go to APPROVED or re-evaluate
    next_states = orchestrator.get_legal_next_states(LifecycleState.EVALUATED)
    assert LifecycleState.APPROVED in next_states
    assert LifecycleState.EVALUATED in next_states


def test_orchestrator_get_transition_path():
    """Test finding transition path between states."""
    orchestrator, repository = _orchestrator_with_asset()

    # Path from REGISTERED to PRODUCTION_READY
    path = orchestrator.get_transition_path(
        LifecycleState.REGISTERED,
        LifecycleState.PRODUCTION_READY
    )

    assert path is not None
    assert path[0] == LifecycleState.REGISTERED
    assert path[-1] == LifecycleState.PRODUCTION_READY
    assert LifecycleState.EVALUATED in path
    assert LifecycleState.APPROVED in path


def test_orchestrator_no_path_for_illegal_transition():
    """Test that no path exists for illegal backwards transitions."""
    orchestrator, repository = _orchestrator_with_asset()

    # Cannot go from APPROVED back to REGISTERED
    path = orchestrator.get_transition_path(
        LifecycleState.APPROVED,
        LifecycleState.REGISTERED
    )

    assert path is None


def test_orchestrator_same_state_path():
    """Test path when from and to states are the same."""
    orchestrator, repository = _orchestrator_with_asset()

    path = orchestrator.get_transition_path(
        LifecycleState.REGISTERED,
        LifecycleState.REGISTERED
    )

    assert path == (LifecycleState.REGISTERED,)


def test_orchestrator_warnings_on_re_evaluation():
    """Test warnings when re-evaluating without notes."""
    orchestrator, repository = _orchestrator_with_asset(state=LifecycleState.EVALUATED)

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.EVALUATED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
        notes=None,
        expected_revision=1,
    )

    result = orchestrator.execute_transition(request)

    assert len(result.warnings) > 0
    assert any("Re-evaluation" in w for w in result.warnings)


def test_orchestrator_warnings_missing_decision_id():
    """Test warnings when critical transitions lack decision_id."""
    orchestrator, repository = _orchestrator_with_asset(state=LifecycleState.EVALUATED)

    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.EVALUATED,
        to_state=LifecycleState.APPROVED,
        evidence_refs=("gate_results",),
        decision_id=None,
    )

    result = orchestrator.execute_transition(request)

    assert len(result.warnings) > 0
    assert any("decision_id" in w for w in result.warnings)


def test_orchestrator_complete_lifecycle_flow():
    """Test orchestrating complete lifecycle flow with events."""
    orchestrator, repository = _orchestrator_with_asset()

    all_events = []

    def collector(event):
        all_events.append(event)

    orchestrator.add_event_listener(collector)

    # REGISTERED -> EVALUATED
    r1 = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
    )
    orchestrator.execute_transition(r1)

    # EVALUATED -> APPROVED
    r2 = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.EVALUATED,
        to_state=LifecycleState.APPROVED,
        evidence_refs=("gate_results",),
    )
    orchestrator.execute_transition(r2)

    # APPROVED -> PRODUCTION_READY
    r3 = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.APPROVED,
        to_state=LifecycleState.PRODUCTION_READY,
        evidence_refs=("certification",),
    )
    orchestrator.execute_transition(r3)

    assert len(all_events) == 3
    assert all_events[0].to_state == LifecycleState.EVALUATED
    assert all_events[1].to_state == LifecycleState.APPROVED
    assert all_events[2].to_state == LifecycleState.PRODUCTION_READY


def test_orchestrator_uses_repository_event_as_single_authority():
    orchestrator, repository = _orchestrator_with_asset()
    before = len(repository.get_events("F001"))

    result = orchestrator.execute_transition(
        TransitionRequest(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            evidence_refs=("evaluation_bundle_ref",),
            expected_revision=0,
        )
    )

    stored = repository.get("F001")
    events = repository.get_events("F001")
    assert len(events) == before + 1
    assert result.event is events[-1]
    assert result.asset is stored
    assert result.revision == repository.get_revision("F001") == 1
    assert stored.lifecycle_state == events[-1].to_state
    assert stored.first_evaluated_at == events[-1].timestamp


def test_orchestrator_expected_state_and_revision_conflicts_do_not_mutate():
    orchestrator, repository = _orchestrator_with_asset()
    request = TransitionRequest(
        factor_id="F001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=("evaluation_bundle_ref",),
        expected_revision=1,
    )

    with pytest.raises(LifecycleConflictError, match="Expected revision"):
        orchestrator.execute_transition(request)

    assert repository.get("F001").lifecycle_state == LifecycleState.REGISTERED
    assert repository.get_revision("F001") == 0
    assert len(repository.get_events("F001")) == 1


def test_post_commit_hook_failure_cannot_roll_back_or_fabricate_event():
    orchestrator, repository = _orchestrator_with_asset()

    def failing_hook(event):
        raise RuntimeError("observer failed")

    orchestrator.add_transition_hook(
        LifecycleState.REGISTERED, LifecycleState.EVALUATED, failing_hook
    )
    delivered = []
    orchestrator.add_event_listener(lambda event: delivered.append(event))
    result = orchestrator.execute_transition(
        TransitionRequest(
            factor_id="F001",
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            evidence_refs=("evaluation_bundle_ref",),
        )
    )

    assert any("observer failed" in warning for warning in result.warnings)
    assert delivered == [result.event]

    events = repository.get_events("F001")
    assert repository.get("F001").lifecycle_state == LifecycleState.EVALUATED
    assert repository.get_revision("F001") == 1
    assert len(events) == 2
    assert events[-1].to_state == LifecycleState.EVALUATED
