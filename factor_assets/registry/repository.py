"""
In-memory append-only repository for factor assets.

Initial implementation uses in-memory storage only.
NO SQLite, NO file I/O — just validated append-only operations.
"""

from dataclasses import dataclass, replace
from typing import Optional
from datetime import datetime, timezone

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateEvent,
    validate_transition,
    LifecycleConflictError,
)
from factor_assets.contracts.lineage import LineageRef


class DuplicateIdentityError(Exception):
    """Raised when attempting to register a factor that already exists."""
    pass


class AssetNotFoundError(Exception):
    """Raised when a factor asset is not found in the repository."""
    pass


@dataclass
class RepositoryStats:
    """Statistics about the repository state."""
    total_assets: int
    by_state: dict[LifecycleState, int]
    total_events: int
    first_registration: Optional[str]
    last_registration: Optional[str]


class AssetRepository:
    """
    Append-only in-memory repository for factor assets.

    Properties:
    - Immutable identity: factor_id cannot change once registered
    - Append-only events: state transitions are recorded, never deleted
    - Exact seen history: first registration timestamp preserved
    - Conservative transitions: validated state machine
    - No raw values: only metadata and references stored
    """

    def __init__(self):
        self._assets: dict[str, FactorAsset] = {}
        self._events: list[StateEvent] = []
        self._canonical_hash_index: dict[str, str] = {}  # hash -> factor_id

    def register(
        self,
        metadata: AssetMetadata,
        lineage: LineageRef,
        tags: tuple[str, ...] = ()
    ) -> FactorAsset:
        """
        Register a new factor asset.

        Args:
            metadata: Asset metadata including identity
            lineage: Lineage and provenance information
            tags: Optional classification tags

        Returns:
            Newly registered FactorAsset

        Raises:
            DuplicateIdentityError: If factor_id already exists
            ValueError: If metadata.factor_id != lineage.factor_id
        """
        factor_id = metadata.factor_id

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

        # Check for canonical hash collision
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

        self._assets[factor_id] = asset
        self._canonical_hash_index[metadata.canonical_hash] = factor_id

        # Record initial registration event
        event = StateEvent(
            factor_id=factor_id,
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.REGISTERED,
            timestamp=now,
            evidence_refs=(),
            notes="Initial registration"
        )
        self._events.append(event)

        return asset

    def get(self, factor_id: str) -> FactorAsset:
        """
        Get a factor asset by ID.

        Args:
            factor_id: Factor identifier

        Returns:
            FactorAsset

        Raises:
            AssetNotFoundError: If factor not found
        """
        if factor_id not in self._assets:
            raise AssetNotFoundError(f"Factor {factor_id} not found")
        return self._assets[factor_id]

    def exists(self, factor_id: str) -> bool:
        """Check if a factor exists in the repository."""
        return factor_id in self._assets

    def find_by_hash(self, canonical_hash: str) -> Optional[FactorAsset]:
        """
        Find a factor by its canonical hash.

        Args:
            canonical_hash: Canonical expression hash

        Returns:
            FactorAsset if found, None otherwise
        """
        factor_id = self._canonical_hash_index.get(canonical_hash)
        if factor_id is None:
            return None
        return self._assets[factor_id]

    def transition(
        self,
        factor_id: str,
        to_state: LifecycleState,
        evidence_refs: tuple[str, ...] = (),
        decision_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        actor: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> FactorAsset:
        """
        Transition a factor to a new lifecycle state.

        Args:
            factor_id: Factor identifier
            to_state: Target lifecycle state
            evidence_refs: Evidence supporting the transition
            decision_id: Optional decision record ID
            policy_version: Optional policy version
            actor: Optional actor who initiated transition
            notes: Optional notes

        Returns:
            Updated FactorAsset

        Raises:
            AssetNotFoundError: If factor not found
            LifecycleConflictError: If transition is illegal or evidence missing
        """
        asset = self.get(factor_id)
        from_state = asset.lifecycle_state

        # Validate transition with evidence
        validate_transition(from_state, to_state, set(evidence_refs))

        now = datetime.now(timezone.utc).isoformat()

        # Update timestamps based on target state
        updates = {"lifecycle_state": to_state}
        if to_state == LifecycleState.EVALUATED and asset.first_evaluated_at is None:
            updates["first_evaluated_at"] = now
        elif to_state == LifecycleState.APPROVED and asset.approved_at is None:
            updates["approved_at"] = now
        elif to_state == LifecycleState.PRODUCTION_READY and asset.production_ready_at is None:
            updates["production_ready_at"] = now

        updated_asset = replace(asset, **updates)
        self._assets[factor_id] = updated_asset

        # Record event
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
        self._events.append(event)

        return updated_asset

    def list_by_state(self, state: LifecycleState) -> list[FactorAsset]:
        """List all factors in a given lifecycle state."""
        return [asset for asset in self._assets.values() if asset.lifecycle_state == state]

    def list_all(self) -> list[FactorAsset]:
        """List all registered factors."""
        return list(self._assets.values())

    def get_events(self, factor_id: Optional[str] = None) -> list[StateEvent]:
        """
        Get lifecycle events.

        Args:
            factor_id: If provided, return only events for this factor

        Returns:
            List of StateEvent records
        """
        if factor_id is None:
            return list(self._events)
        return [e for e in self._events if e.factor_id == factor_id]

    def stats(self) -> RepositoryStats:
        """Get repository statistics."""
        by_state: dict[LifecycleState, int] = {}
        for asset in self._assets.values():
            by_state[asset.lifecycle_state] = by_state.get(asset.lifecycle_state, 0) + 1

        timestamps = [asset.registered_at for asset in self._assets.values()]
        first = min(timestamps) if timestamps else None
        last = max(timestamps) if timestamps else None

        return RepositoryStats(
            total_assets=len(self._assets),
            by_state=by_state,
            total_events=len(self._events),
            first_registration=first,
            last_registration=last,
        )
