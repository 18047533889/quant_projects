"""Lifecycle orchestration over an authoritative repository transition boundary."""

import logging
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Optional, Protocol

from factor_assets.contracts.asset import FactorAsset
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateEvent,
    get_required_evidence,
    is_legal_transition,
)
from factor_assets.errors import LifecycleConflictError
from factor_assets.registry.repository import LifecycleRepository


@dataclass(frozen=True)
class TransitionRequest:
    """Requested transition plus optimistic-concurrency expectations."""

    factor_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    evidence_refs: tuple[str, ...]
    decision_id: Optional[str] = None
    policy_version: Optional[str] = None
    actor: Optional[str] = None
    notes: Optional[str] = None
    expected_revision: Optional[int] = None
    evidence_bundle_ref: Optional[EvidenceBundleRef] = None
    authorization_ref: Optional[str] = None

class EvidenceAuthorizationResolver(Protocol):
    def resolve(self, ref: str) -> object: ...

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")


@dataclass(frozen=True)
class TransitionResult:
    """The repository's committed event, asset, revision, and warnings."""

    event: StateEvent
    warnings: tuple[str, ...] = ()
    timestamp: str = ""
    asset: Optional[FactorAsset] = None
    revision: Optional[int] = None

    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, "timestamp", self.event.timestamp)


class LifecycleOrchestrator:
    """Validates requests, delegates commit authority, then runs observers."""

    def __init__(self, repository: Optional[LifecycleRepository] = None,
                 authorization_resolver: Optional[EvidenceAuthorizationResolver] = None):
        self._repository = repository
        self._authorization_resolver = authorization_resolver
        self._event_listeners: list[Callable[[StateEvent], None]] = []
        self._transition_hooks: dict[
            tuple[LifecycleState, LifecycleState], list[Callable[[StateEvent], None]]
        ] = {}
        self._observer_lock = RLock()

    def validate_transition(self, request: TransitionRequest) -> None:
        """Validate transition shape and evidence without mutating state."""
        if not is_legal_transition(request.from_state, request.to_state):
            raise LifecycleConflictError(
                f"Illegal transition: {request.from_state.value} -> {request.to_state.value} "
                f"for factor {request.factor_id}"
            )
        required = get_required_evidence(request.from_state, request.to_state)
        available_evidence = set(request.evidence_refs)
        if request.evidence_bundle_ref is not None:
            available_evidence.add("evaluation_bundle_ref")
        missing = set(required) - available_evidence
        if missing:
            raise LifecycleConflictError(
                f"Missing required evidence for {request.from_state.value} -> "
                f"{request.to_state.value}: {sorted(missing)} (factor {request.factor_id})"
            )
        if request.to_state in (LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY):
            if not request.authorization_ref or self._authorization_resolver is None:
                raise LifecycleConflictError("typed trusted authorization is required for approval/production")
            try:
                artifact = self._authorization_resolver.resolve(request.authorization_ref)
            except Exception as exc:
                raise LifecycleConflictError("authorization could not be resolved") from exc
            if getattr(artifact, "factor_id", None) != request.factor_id:
                raise LifecycleConflictError("authorization factor mismatch")
            current_asset = self._repository.get(request.factor_id) if self._repository is not None else None
            authorized_recipe = (getattr(artifact, "recipe_hash", None)
                                 or getattr(artifact, "canonical_hash", None)
                                 or getattr(artifact, "factor_definition_hash", None))
            if authorized_recipe is not None and (current_asset is None or authorized_recipe != current_asset.metadata.canonical_hash):
                raise LifecycleConflictError("authorization recipe mismatch")
            content_hash = getattr(artifact, "content_hash", None)
            if content_hash != request.authorization_ref:
                raise LifecycleConflictError("authorization content hash mismatch")
            decision = getattr(getattr(artifact, "decision", None), "value", getattr(artifact, "decision", None))
            status = getattr(getattr(artifact, "status", None), "value", getattr(artifact, "status", None))
            if request.to_state is LifecycleState.APPROVED and decision != "APPROVED":
                raise LifecycleConflictError("authorization decision is not APPROVED")
            authorized_policy = getattr(artifact, "policy_version", None)
            if authorized_policy is not None and authorized_policy != request.policy_version:
                raise LifecycleConflictError("authorization policy version mismatch")
            if request.to_state is LifecycleState.PRODUCTION_READY and status not in ("CERTIFIED", "APPROVED"):
                raise LifecycleConflictError("production certification is not approved")
            expires_at = getattr(artifact, "expires_at", None) or getattr(artifact, "frozen_until", None)
            if expires_at:
                from datetime import datetime, timezone
                expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                if expiry <= datetime.now(timezone.utc):
                    raise LifecycleConflictError("authorization is expired")

    def execute_transition(self, request: TransitionRequest) -> TransitionResult:
        """Commit once in the repository, then invoke hooks and listeners."""
        self.validate_transition(request)
        if self._repository is None:
            raise LifecycleConflictError(
                "LifecycleOrchestrator requires an authoritative repository to execute transitions"
            )

        committed = self._repository.commit_transition(
            request.factor_id,
            request.to_state,
            request.evidence_refs,
            request.decision_id,
            request.policy_version,
            request.actor,
            request.notes,
            expected_state=request.from_state,
            expected_revision=request.expected_revision,
            evidence_bundle_ref=request.evidence_bundle_ref,
        )

        # The state/event commit is complete before any extension code runs.
        # Observer failure is allowed to propagate, but cannot undo or replace it.
        hook_key = (committed.event.from_state, committed.event.to_state)
        observer_failures = []
        with self._observer_lock:
            observers = (
                *self._transition_hooks.get(hook_key, ()),
                *self._event_listeners,
            )
            for observer in observers:
                try:
                    observer(committed.event)
                except Exception as exc:
                    logging.warning(
                        "Post-commit observer %r failed: %s", observer, exc,
                        exc_info=True,
                    )
                    observer_failures.append(
                        f"Post-commit observer {observer!r} failed: {exc}"
                    )

        warnings = self._check_transition_warnings(request) + tuple(observer_failures)
        return TransitionResult(
            event=committed.event,
            warnings=warnings,
            timestamp=committed.event.timestamp,
            asset=committed.asset,
            revision=committed.revision,
        )

    def _check_transition_warnings(self, request: TransitionRequest) -> tuple[str, ...]:
        warnings = []
        if request.from_state == request.to_state == LifecycleState.EVALUATED and not request.notes:
            warnings.append("Re-evaluation without notes — consider documenting the reason")
        if request.to_state == LifecycleState.PRODUCTION_READY:
            if request.from_state != LifecycleState.APPROVED:
                warnings.append(
                    f"Production readiness from {request.from_state.value} — expected APPROVED"
                )
        if request.to_state in (LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY):
            if not request.decision_id:
                warnings.append(f"Transition to {request.to_state.value} without decision_id")
        return tuple(warnings)

    def add_event_listener(self, listener: Callable[[StateEvent], None]) -> None:
        """Register a post-commit listener for every transition."""
        with self._observer_lock:
            self._event_listeners.append(listener)

    def add_transition_hook(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        hook: Callable[[StateEvent], None],
    ) -> None:
        """Register a post-commit hook for a specific transition."""
        with self._observer_lock:
            self._transition_hooks.setdefault((from_state, to_state), []).append(hook)

    def get_legal_next_states(self, current_state: LifecycleState) -> tuple[LifecycleState, ...]:
        return tuple(
            target_state
            for target_state in LifecycleState
            if is_legal_transition(current_state, target_state)
        )

    def get_transition_path(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
    ) -> Optional[tuple[LifecycleState, ...]]:
        if from_state == to_state:
            return (from_state,)
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
