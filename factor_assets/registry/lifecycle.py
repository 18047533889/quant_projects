"""
Lifecycle orchestration and state transition validation.

Centralizes state machine logic, event emission, and transition guards.
Works alongside AssetRepository to enforce lifecycle invariants.
"""

from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime, timezone

from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateEvent,
    is_legal_transition,
    get_required_evidence,
    LifecycleConflictError,
)


@dataclass(frozen=True)
class TransitionRequest:
    """
    Request to transition a factor asset to a new state.

    Encapsulates all transition context including evidence and provenance.
    """
    factor_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    evidence_refs: tuple[str, ...]
    decision_id: Optional[str] = None
    policy_version: Optional[str] = None
    actor: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")


@dataclass(frozen=True)
class TransitionResult:
    """
    Result of a lifecycle transition.

    Contains the event record and any validation warnings.
    """
    event: StateEvent
    warnings: tuple[str, ...] = ()
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc).isoformat())


class LifecycleOrchestrator:
    """
    Orchestrates lifecycle state transitions with validation and events.

    Centralizes transition logic:
    - Pre-transition validation (legal transition, evidence present)
    - Event creation and emission
    - Post-transition hooks (optional observers)
    - Audit trail management

    Does NOT store assets — that is AssetRepository's responsibility.
    """

    def __init__(self):
        self._event_listeners: list[Callable[[StateEvent], None]] = []
        self._transition_hooks: dict[tuple[LifecycleState, LifecycleState], list[Callable]] = {}

    def validate_transition(self, request: TransitionRequest) -> None:
        """
        Validate a transition request without executing it.

        Args:
            request: Transition request

        Raises:
            LifecycleConflictError: If transition is illegal or evidence missing
        """
        if not is_legal_transition(request.from_state, request.to_state):
            raise LifecycleConflictError(
                f"Illegal transition: {request.from_state.value} -> {request.to_state.value} "
                f"for factor {request.factor_id}"
            )

        required = get_required_evidence(request.from_state, request.to_state)
        evidence_set = set(request.evidence_refs)
        missing = set(required) - evidence_set

        if missing:
            raise LifecycleConflictError(
                f"Missing required evidence for {request.from_state.value} -> {request.to_state.value}: "
                f"{sorted(missing)} (factor {request.factor_id})"
            )

    def execute_transition(self, request: TransitionRequest) -> TransitionResult:
        """
        Execute a validated lifecycle transition.

        Args:
            request: Transition request

        Returns:
            TransitionResult with event and warnings

        Raises:
            LifecycleConflictError: If transition validation fails
        """
        self.validate_transition(request)

        now = datetime.now(timezone.utc).isoformat()

        event = StateEvent(
            factor_id=request.factor_id,
            from_state=request.from_state,
            to_state=request.to_state,
            timestamp=now,
            evidence_refs=request.evidence_refs,
            decision_id=request.decision_id,
            policy_version=request.policy_version,
            actor=request.actor,
            notes=request.notes,
        )

        # Execute pre-transition hooks
        hook_key = (request.from_state, request.to_state)
        if hook_key in self._transition_hooks:
            for hook in self._transition_hooks[hook_key]:
                hook(event)

        # Emit event to listeners
        for listener in self._event_listeners:
            listener(event)

        warnings = self._check_transition_warnings(request)

        return TransitionResult(event=event, warnings=warnings, timestamp=now)

    def _check_transition_warnings(self, request: TransitionRequest) -> tuple[str, ...]:
        """
        Check for non-fatal warnings during transition.

        Returns warnings without blocking the transition.
        """
        warnings = []

        # Warn on re-evaluation if moving backward
        if request.from_state == request.to_state == LifecycleState.EVALUATED:
            if not request.notes:
                warnings.append("Re-evaluation without notes — consider documenting the reason")

        # Warn on production-ready without approval
        if request.to_state == LifecycleState.PRODUCTION_READY:
            if request.from_state != LifecycleState.APPROVED:
                warnings.append(f"Production readiness from {request.from_state.value} — expected APPROVED")

        # Warn on missing decision_id for critical transitions
        if request.to_state in (LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY):
            if not request.decision_id:
                warnings.append(f"Transition to {request.to_state.value} without decision_id")

        return tuple(warnings)

    def add_event_listener(self, listener: Callable[[StateEvent], None]) -> None:
        """
        Register an event listener for all state transitions.

        Listener will be called after transition validation but before completion.
        """
        self._event_listeners.append(listener)

    def add_transition_hook(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        hook: Callable[[StateEvent], None]
    ) -> None:
        """
        Register a hook for a specific state transition.

        Hook is called before event emission.
        """
        key = (from_state, to_state)
        if key not in self._transition_hooks:
            self._transition_hooks[key] = []
        self._transition_hooks[key].append(hook)

    def get_legal_next_states(self, current_state: LifecycleState) -> tuple[LifecycleState, ...]:
        """
        Get all legal next states from the current state.

        Args:
            current_state: Current lifecycle state

        Returns:
            Tuple of legal target states
        """
        legal_states = []
        for target_state in LifecycleState:
            if is_legal_transition(current_state, target_state):
                legal_states.append(target_state)
        return tuple(legal_states)

    def get_transition_path(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState
    ) -> Optional[tuple[LifecycleState, ...]]:
        """
        Find the shortest path between two states.

        Returns None if no legal path exists.
        """
        if from_state == to_state:
            return (from_state,)

        # BFS to find shortest path
        from collections import deque

        queue = deque([(from_state, [from_state])])
        visited = {from_state}

        while queue:
            current, path = queue.popleft()

            for next_state in self.get_legal_next_states(current):
                if next_state in visited:
                    continue

                new_path = path + [next_state]

                if next_state == to_state:
                    return tuple(new_path)

                visited.add(next_state)
                queue.append((next_state, new_path))

        return None
