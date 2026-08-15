"""
Lifecycle states, transitions, and events for FactorAsset.

Conservative state machine: registered -> evaluated -> approved flow.
All transitions are validated; illegal transitions raise LifecycleConflictError.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.errors import LifecycleConflictError


class LifecycleState(Enum):
    """
    Factor asset lifecycle states.

    Conservative flow: REGISTERED -> EVALUATED -> APPROVED -> PRODUCTION_READY
    Each state requires specific evidence before transition.
    """
    REGISTERED = "REGISTERED"           # Initial registration, identity established
    EVALUATED = "EVALUATED"             # Evidence attached, metrics computed
    APPROVED = "APPROVED"               # Passed gates, ready for selection
    PRODUCTION_READY = "PRODUCTION_READY"  # Certified for production use
    DEPRECATED = "DEPRECATED"           # No longer recommended
    RETIRED = "RETIRED"                 # Archived, not for new use


@dataclass(frozen=True)
class StateTransition:
    """
    Legal state transition specification.

    Defines from_state -> to_state with required evidence/conditions.
    Self-transitions are allowed for specific states (e.g., re-evaluation).
    """
    from_state: LifecycleState
    to_state: LifecycleState
    required_evidence: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class StateEvent:
    """
    Immutable lifecycle state change event.

    Append-only record of each transition with provenance.
    """
    factor_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: str  # ISO 8601
    evidence_refs: tuple[str, ...]
    decision_id: Optional[str] = None
    policy_version: Optional[str] = None
    actor: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.timestamp:
            raise ValueError("timestamp is required")


# Legal transitions per CONTRACT_FREEZE_DRAFT
_LEGAL_TRANSITIONS = [
    StateTransition(
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        required_evidence=("evaluation_bundle_ref",),
        description="Evidence attached from QE evaluation"
    ),
    StateTransition(
        LifecycleState.EVALUATED,
        LifecycleState.APPROVED,
        required_evidence=("gate_results",),
        description="Passed admission gates"
    ),
    StateTransition(
        LifecycleState.APPROVED,
        LifecycleState.PRODUCTION_READY,
        required_evidence=("certification",),
        description="Production certification complete"
    ),
    StateTransition(
        LifecycleState.PRODUCTION_READY,
        LifecycleState.DEPRECATED,
        required_evidence=(),
        description="Marked deprecated"
    ),
    StateTransition(
        LifecycleState.DEPRECATED,
        LifecycleState.RETIRED,
        required_evidence=(),
        description="Archived and retired"
    ),
    # Allow re-evaluation
    StateTransition(
        LifecycleState.EVALUATED,
        LifecycleState.EVALUATED,
        required_evidence=("evaluation_bundle_ref",),
        description="Re-evaluation with new evidence"
    ),
]


def is_legal_transition(from_state: LifecycleState, to_state: LifecycleState) -> bool:
    """
    Check if a state transition is legal.

    Args:
        from_state: Current lifecycle state
        to_state: Target lifecycle state

    Returns:
        True if transition is legal, False otherwise
    """
    # Self-transition is allowed only for EVALUATED (re-evaluation)
    if from_state == to_state:
        return from_state == LifecycleState.EVALUATED

    for transition in _LEGAL_TRANSITIONS:
        if transition.from_state == from_state and transition.to_state == to_state:
            return True
    return False


def get_required_evidence(from_state: LifecycleState, to_state: LifecycleState) -> tuple[str, ...]:
    """
    Get required evidence for a state transition.

    Args:
        from_state: Current lifecycle state
        to_state: Target lifecycle state

    Returns:
        Tuple of required evidence keys

    Raises:
        LifecycleConflictError: If transition is not legal
    """
    for transition in _LEGAL_TRANSITIONS:
        if transition.from_state == from_state and transition.to_state == to_state:
            return transition.required_evidence

    raise LifecycleConflictError(
        f"Illegal transition: {from_state.value} -> {to_state.value}"
    )


def validate_transition(
    from_state: LifecycleState,
    to_state: LifecycleState,
    evidence_keys: set[str]
) -> None:
    """
    Validate a state transition with evidence.

    Args:
        from_state: Current lifecycle state
        to_state: Target lifecycle state
        evidence_keys: Available evidence keys

    Raises:
        LifecycleConflictError: If transition is illegal or evidence is missing
    """
    if not is_legal_transition(from_state, to_state):
        raise LifecycleConflictError(
            f"Illegal transition: {from_state.value} -> {to_state.value}"
        )

    required = get_required_evidence(from_state, to_state)
    missing = set(required) - evidence_keys

    if missing:
        raise LifecycleConflictError(
            f"Missing required evidence for {from_state.value} -> {to_state.value}: {missing}"
        )
