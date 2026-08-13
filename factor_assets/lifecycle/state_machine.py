"""
Lifecycle state machine implementation.

Validates state transitions and maintains transition history.
"""

from typing import Optional

from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateEvent,
    is_legal_transition,
    validate_transition,
    LifecycleConflictError,
)


class StateMachine:
    """
    Lifecycle state machine for factor assets.

    Provides validation and state transition logic.
    Repository is responsible for persistence; this is pure validation.
    """

    @staticmethod
    def can_transition(from_state: LifecycleState, to_state: LifecycleState) -> bool:
        """
        Check if transition is legal.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if transition is legal
        """
        return is_legal_transition(from_state, to_state)

    @staticmethod
    def validate(
        from_state: LifecycleState,
        to_state: LifecycleState,
        evidence_keys: set[str],
    ) -> None:
        """
        Validate a state transition.

        Args:
            from_state: Current state
            to_state: Target state
            evidence_keys: Available evidence

        Raises:
            LifecycleConflictError: If transition is invalid
        """
        validate_transition(from_state, to_state, evidence_keys)

    @staticmethod
    def create_event(
        factor_id: str,
        from_state: LifecycleState,
        to_state: LifecycleState,
        timestamp: str,
        evidence_refs: tuple[str, ...] = (),
        decision_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        actor: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> StateEvent:
        """
        Create a state transition event.

        Args:
            factor_id: Factor identifier
            from_state: Source state
            to_state: Target state
            timestamp: Event timestamp (ISO 8601)
            evidence_refs: Evidence references
            decision_id: Optional decision ID
            policy_version: Optional policy version
            actor: Optional actor
            notes: Optional notes

        Returns:
            StateEvent

        Raises:
            ValueError: If required fields are missing
        """
        return StateEvent(
            factor_id=factor_id,
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp,
            evidence_refs=evidence_refs,
            decision_id=decision_id,
            policy_version=policy_version,
            actor=actor,
            notes=notes,
        )
