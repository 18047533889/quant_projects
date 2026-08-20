"""
Authoritative repository interfaces and the ephemeral in-memory implementation.

The in-memory repository is intentionally RESEARCH_ONLY. It provides the same
atomic transition boundary expected from a future durable repository, but it
must not be selected for production use.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Optional, Protocol, runtime_checkable

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.evidence_ref import (
    EvidenceBundleRef,
    evidence_bundle_event_id,
)
from factor_assets.contracts.lifecycle import (
    HealthState,
    LifecycleState,
    StateEvent,
    validate_transition,
)
from factor_assets.contracts.lineage import LineageRef
from factor_assets.errors import (
    DuplicateIdentityError,
    LifecycleConflictError,
)


class AssetNotFoundError(Exception):
    """Raised when a factor asset is not found in the repository."""


@dataclass(frozen=True)
class CommittedTransition:
    """Authoritative result of one committed state mutation and event append."""

    asset: FactorAsset
    event: StateEvent
    revision: int


@runtime_checkable
class LifecycleRepository(Protocol):
    """Repository boundary that owns lifecycle state and event authority."""

    def commit_transition(
        self,
        factor_id: str,
        to_state: LifecycleState,
        evidence_refs: tuple[str, ...] = (),
        decision_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        actor: Optional[str] = None,
        notes: Optional[str] = None,
        *,
        expected_state: Optional[LifecycleState] = None,
        expected_revision: Optional[int] = None,
        evidence_bundle_ref: Optional[EvidenceBundleRef] = None,
    ) -> CommittedTransition:
        """Atomically validate, mutate state, and append exactly one event."""
        ...

    def update_asset_health(
        self,
        factor_id: str,
        health: HealthState,
        *,
        reason: str = "",
        actor: Optional[str] = None,
    ) -> None:
        """Append a health-change event and update the orthogonal health dimension."""
        ...


@dataclass
class RepositoryStats:
    """Statistics about the repository state."""

    total_assets: int
    by_state: dict[LifecycleState, int]
    total_events: int
    first_registration: Optional[str]
    last_registration: Optional[str]


class AssetRepository:
    """Append-only, process-local repository for research and tests only."""

    RESEARCH_ONLY = True
    EPHEMERAL = True
    PRODUCTION_CAPABLE = False

    def __init__(self):
        self._assets: dict[str, FactorAsset] = {}
        self._events: list[StateEvent] = []
        self._canonical_hash_index: dict[str, str] = {}
        self._revisions: dict[str, int] = {}
        self._idempotency_keys: set[tuple[str, str]] = set()
        self._committed_decisions: dict[tuple[str, str], CommittedTransition] = {}
        self._lock = RLock()

    def register(
        self,
        metadata: AssetMetadata,
        lineage: LineageRef,
        tags: tuple[str, ...] = (),
    ) -> FactorAsset:
        """Register an asset and append its initial registration event."""
        factor_id = metadata.factor_id
        with self._lock:
            if factor_id in self._assets:
                raise DuplicateIdentityError(
                    f"Factor {factor_id} is already registered at "
                    f"{self._assets[factor_id].registered_at}"
                )
            if metadata.factor_id != lineage.factor_id:
                raise ValueError(
                    f"Metadata factor_id ({metadata.factor_id}) must match "
                    f"lineage factor_id ({lineage.factor_id})"
                )
            if metadata.canonical_hash in self._canonical_hash_index:
                existing_id = self._canonical_hash_index[metadata.canonical_hash]
                raise DuplicateIdentityError(
                    f"Factor with canonical_hash {metadata.canonical_hash} "
                    f"already exists as {existing_id}"
                )

            now = datetime.now(timezone.utc).isoformat()
            asset = FactorAsset(
                metadata=metadata,
                lineage=lineage,
                lifecycle_state=LifecycleState.REGISTERED,
                registered_at=now,
                tags=tags,
            )
            event = StateEvent(
                factor_id=factor_id,
                from_state=LifecycleState.REGISTERED,
                to_state=LifecycleState.REGISTERED,
                timestamp=now,
                evidence_refs=(),
                notes="Initial registration",
            )
            self._assets[factor_id] = asset
            self._canonical_hash_index[metadata.canonical_hash] = factor_id
            self._revisions[factor_id] = 0
            self._events.append(event)
            return asset

    def get(self, factor_id: str) -> FactorAsset:
        """Get an asset by ID."""
        with self._lock:
            if factor_id not in self._assets:
                raise AssetNotFoundError(f"Factor {factor_id} not found")
            return self._assets[factor_id]

    def get_revision(self, factor_id: str) -> int:
        """Return the current optimistic-concurrency revision."""
        with self._lock:
            if factor_id not in self._revisions:
                raise AssetNotFoundError(f"Factor {factor_id} not found")
            return self._revisions[factor_id]

    def exists(self, factor_id: str) -> bool:
        with self._lock:
            return factor_id in self._assets

    def find_by_hash(self, canonical_hash: str) -> Optional[FactorAsset]:
        with self._lock:
            factor_id = self._canonical_hash_index.get(canonical_hash)
            return None if factor_id is None else self._assets[factor_id]

    def commit_transition(
        self,
        factor_id: str,
        to_state: LifecycleState,
        evidence_refs: tuple[str, ...] = (),
        decision_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        actor: Optional[str] = None,
        notes: Optional[str] = None,
        *,
        expected_state: Optional[LifecycleState] = None,
        expected_revision: Optional[int] = None,
        evidence_bundle_ref: Optional[EvidenceBundleRef] = None,
    ) -> CommittedTransition:
        """Atomically perform the authoritative transition and event append."""
        with self._lock:
            if factor_id not in self._assets:
                raise AssetNotFoundError(f"Factor {factor_id} not found")
            asset = self._assets[factor_id]
            from_state = asset.lifecycle_state
            revision = self._revisions[factor_id]

            if expected_state is not None and from_state != expected_state:
                raise LifecycleConflictError(
                    f"Expected state {expected_state.value} for factor {factor_id}, "
                    f"found {from_state.value}"
                )
            if expected_revision is not None and revision != expected_revision:
                raise LifecycleConflictError(
                    f"Expected revision {expected_revision} for factor {factor_id}, "
                    f"found {revision}"
                )
            if (
                from_state == to_state == LifecycleState.EVALUATED
                and expected_revision is None
                and not decision_id
            ):
                raise LifecycleConflictError(
                    "EVALUATED re-evaluation requires expected_revision or decision_id"
                )
            if decision_id is not None:
                idempotency_key = (factor_id, decision_id)
                stored = self._committed_decisions.get(idempotency_key)
                if stored is not None:
                    stored_refs = stored.event.evidence_refs
                    replay_refs = tuple(dict.fromkeys((
                        *evidence_refs,
                        "evaluation_bundle_ref",
                        evidence_bundle_event_id(evidence_bundle_ref),
                    ))) if evidence_bundle_ref is not None else tuple(evidence_refs)
                    identical = (
                        from_state == stored.event.to_state
                        and revision == stored.revision
                        and stored.event.to_state == to_state
                        and tuple(stored_refs) == replay_refs
                        and stored.event.policy_version == policy_version
                        and stored.event.actor == actor
                        and stored.event.notes == notes
                        and (
                            evidence_bundle_ref is None
                            and stored.asset.latest_evidence_ref is None
                            or evidence_bundle_ref is not None
                            and stored.asset.latest_evidence_ref is not None
                        )
                    )
                    if evidence_bundle_ref is not None and identical:
                        identical = (
                            evidence_bundle_ref == stored.asset.latest_evidence_ref
                        )
                    if identical:
                        return stored
                    raise LifecycleConflictError(
                        f"Conflicting transition decision_id {decision_id} for factor {factor_id}"
                    )
            if evidence_bundle_ref is not None:
                if not isinstance(evidence_bundle_ref, EvidenceBundleRef):
                    raise TypeError("evidence_bundle_ref must be an EvidenceBundleRef")
                if factor_id not in evidence_bundle_ref.factor_ids:
                    raise LifecycleConflictError(
                        f"Evidence bundle {evidence_bundle_ref.bundle_id} does not contain "
                        f"factor {factor_id}"
                    )
                if to_state != LifecycleState.EVALUATED:
                    raise LifecycleConflictError(
                        "evidence_bundle_ref may only be committed with an EVALUATED transition"
                    )

            evidence_refs = tuple(evidence_refs)
            if evidence_bundle_ref is not None:
                evidence_refs = tuple(dict.fromkeys((
                    *evidence_refs,
                    "evaluation_bundle_ref",
                    evidence_bundle_event_id(evidence_bundle_ref),
                )))
            validate_transition(from_state, to_state, set(evidence_refs))
            now = datetime.now(timezone.utc).isoformat()
            updates = {"lifecycle_state": to_state}
            if to_state == LifecycleState.EVALUATED:
                if evidence_bundle_ref is not None:
                    updates["latest_evidence_ref"] = evidence_bundle_ref
                if asset.first_evaluated_at is None:
                    updates["first_evaluated_at"] = now
            elif to_state == LifecycleState.APPROVED and asset.approved_at is None:
                updates["approved_at"] = now
            elif to_state == LifecycleState.PRODUCTION_READY and asset.production_ready_at is None:
                updates["production_ready_at"] = now

            updated_asset = replace(asset, **updates)
            event = StateEvent(
                factor_id=factor_id,
                from_state=from_state,
                to_state=to_state,
                timestamp=now,
                evidence_refs=evidence_refs,
                decision_id=decision_id,
                policy_version=policy_version,
                actor=actor,
                notes=notes,
            )
            new_revision = revision + 1
            self._assets[factor_id] = updated_asset
            self._revisions[factor_id] = new_revision
            self._events.append(event)
            if decision_id is not None:
                self._idempotency_keys.add((factor_id, decision_id))
                self._committed_decisions[(factor_id, decision_id)] = CommittedTransition(
                    updated_asset, event, new_revision
                )
            return CommittedTransition(updated_asset, event, new_revision)

    def update_asset_health(
        self,
        factor_id: str,
        health: HealthState,
        *,
        reason: str = "",
        actor: Optional[str] = None,
    ) -> None:
        """Update the orthogonal health dimension and append one health event."""
        if not isinstance(health, HealthState):
            raise TypeError("health must be a HealthState")
        with self._lock:
            if factor_id not in self._assets:
                raise AssetNotFoundError(f"Factor {factor_id} not found")
            asset = self._assets[factor_id]
            now = datetime.now(timezone.utc).isoformat()
            updated_asset = replace(asset, health_state=health)
            event = StateEvent(
                factor_id=factor_id,
                from_state=asset.lifecycle_state,
                to_state=asset.lifecycle_state,
                timestamp=now,
                evidence_refs=(),
                notes=f"Health change {asset.health_state.value} -> {health.value}"
                + (f": {reason}" if reason else ""),
                actor=actor,
            )
            self._assets[factor_id] = updated_asset
            self._events.append(event)

    def transition(
        self,
        factor_id: str,
        to_state: LifecycleState,
        evidence_refs: tuple[str, ...] = (),
        decision_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        actor: Optional[str] = None,
        notes: Optional[str] = None,
        *,
        expected_state: Optional[LifecycleState] = None,
        expected_revision: Optional[int] = None,
        evidence_bundle_ref: Optional[EvidenceBundleRef] = None,
    ) -> FactorAsset:
        """Compatibility wrapper returning the committed asset."""
        return self.commit_transition(
            factor_id,
            to_state,
            evidence_refs,
            decision_id,
            policy_version,
            actor,
            notes,
            expected_state=expected_state,
            expected_revision=expected_revision,
            evidence_bundle_ref=evidence_bundle_ref,
        ).asset

    def list_by_state(self, state: LifecycleState) -> list[FactorAsset]:
        with self._lock:
            return [asset for asset in self._assets.values() if asset.lifecycle_state == state]

    def list_all(self) -> list[FactorAsset]:
        with self._lock:
            return list(self._assets.values())

    def get_events(self, factor_id: Optional[str] = None) -> list[StateEvent]:
        with self._lock:
            if factor_id is None:
                return list(self._events)
            return [event for event in self._events if event.factor_id == factor_id]

    def stats(self) -> RepositoryStats:
        with self._lock:
            by_state: dict[LifecycleState, int] = {}
            for asset in self._assets.values():
                by_state[asset.lifecycle_state] = by_state.get(asset.lifecycle_state, 0) + 1
            timestamps = [asset.registered_at for asset in self._assets.values()]
            return RepositoryStats(
                total_assets=len(self._assets),
                by_state=by_state,
                total_events=len(self._events),
                first_registration=min(timestamps) if timestamps else None,
                last_registration=max(timestamps) if timestamps else None,
            )
