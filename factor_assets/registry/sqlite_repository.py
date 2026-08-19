"""Durable SQLite lifecycle repository (schema v1)."""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import replace
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.evidence_ref import EvidenceBundleRef, evidence_bundle_event_id
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import LifecycleState, StateEvent, validate_transition
from factor_assets.errors import (
    CapabilityError,
    DuplicateIdentityError,
    FactorAssetsError,
    LifecycleConflictError,
)
from factor_assets.registry.repository import AssetNotFoundError, CommittedTransition, LifecycleRepository, RepositoryStats
from factor_assets.registry.migrations import migrate
from factor_assets.registry.serialization import asset_from_json, asset_to_json, event_from_json, event_to_json, evidence_to_json

class SQLiteLifecycleRepository:
    RESEARCH_ONLY = False
    EPHEMERAL = False
    PRODUCTION_CAPABLE = True

    def __init__(self, db_path: str | Path):
        if not db_path or str(db_path) == ":memory:": raise CapabilityError("SQLite production repository requires an explicit file db_path")
        self.db_path = Path(db_path)
        if self.db_path.exists() and self.db_path.is_dir(): raise ValueError("db_path must be a file")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn: migrate(conn)

    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def register(self, metadata: AssetMetadata, lineage: LineageRef, tags: tuple[str, ...] = ()) -> FactorAsset:
        if metadata.factor_id != lineage.factor_id: raise ValueError("metadata and lineage factor_id must match")
        now = datetime.now(timezone.utc).isoformat()
        asset = FactorAsset(metadata, lineage, LifecycleState.REGISTERED, now, tags=tuple(tags))
        event = StateEvent(metadata.factor_id, LifecycleState.REGISTERED, LifecycleState.REGISTERED, now, (), notes="Initial registration")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("SELECT 1 FROM assets WHERE factor_id=?", (metadata.factor_id,)).fetchone(): raise DuplicateIdentityError(f"Factor {metadata.factor_id} is already registered")
                if conn.execute("SELECT 1 FROM assets WHERE canonical_hash=?", (metadata.canonical_hash,)).fetchone(): raise DuplicateIdentityError(f"canonical_hash {metadata.canonical_hash} already exists")
                conn.execute("INSERT INTO assets VALUES (?,?,?,?)", (metadata.factor_id, metadata.canonical_hash, asset_to_json(asset), 0))
                conn.execute("INSERT INTO lifecycle_events(factor_id,revision,decision_id,payload) VALUES (?,?,?,?)", (metadata.factor_id, 0, None, event_to_json(event)))
                conn.commit(); return asset
            except (sqlite3.Error, OSError, ValueError, TypeError,
                    FactorAssetsError, AssetNotFoundError):
                # Governance/contract errors (DuplicateIdentityError etc.) must
                # roll back and still propagate by their own type.
                conn.rollback(); raise

    def get(self, factor_id: str) -> FactorAsset:
        with self._connection() as conn:
            row = conn.execute("SELECT payload FROM assets WHERE factor_id=?", (factor_id,)).fetchone()
        if row is None: raise AssetNotFoundError(f"Factor {factor_id} not found")
        return asset_from_json(row[0])

    def exists(self, factor_id: str) -> bool:
        with self._connection() as conn: return conn.execute("SELECT 1 FROM assets WHERE factor_id=?", (factor_id,)).fetchone() is not None
    def get_revision(self, factor_id: str) -> int:
        with self._connection() as conn: row = conn.execute("SELECT revision FROM assets WHERE factor_id=?", (factor_id,)).fetchone()
        if row is None: raise AssetNotFoundError(f"Factor {factor_id} not found")
        return row[0]
    def find_by_hash(self, canonical_hash: str) -> Optional[FactorAsset]:
        with self._connection() as conn: row = conn.execute("SELECT payload FROM assets WHERE canonical_hash=?", (canonical_hash,)).fetchone()
        return None if row is None else asset_from_json(row[0])

    def commit_transition(self, factor_id, to_state, evidence_refs=(), decision_id=None, policy_version=None, actor=None, notes=None, *, expected_state=None, expected_revision=None, evidence_bundle_ref=None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT payload,revision FROM assets WHERE factor_id=?", (factor_id,)).fetchone()
                if row is None: raise AssetNotFoundError(f"Factor {factor_id} not found")
                asset, revision = asset_from_json(row[0]), row[1]
                if expected_state is not None and asset.lifecycle_state != expected_state: raise LifecycleConflictError(f"Expected state {expected_state.value} for factor {factor_id}, found {asset.lifecycle_state.value}")
                if expected_revision is not None and revision != expected_revision: raise LifecycleConflictError(f"Expected revision {expected_revision} for factor {factor_id}, found {revision}")
                if from_eval := (asset.lifecycle_state == to_state == LifecycleState.EVALUATED):
                    if expected_revision is None and not decision_id: raise LifecycleConflictError("EVALUATED re-evaluation requires expected_revision or decision_id")
                refs = tuple(dict.fromkeys(tuple(evidence_refs) + (("evaluation_bundle_ref", evidence_bundle_event_id(evidence_bundle_ref)) if evidence_bundle_ref else ())))
                if decision_id is not None:
                    old = conn.execute("SELECT payload,revision FROM lifecycle_events WHERE factor_id=? AND decision_id=?", (factor_id, decision_id)).fetchone()
                    if old is not None:
                        old_event = event_from_json(old[0])
                        if (asset.lifecycle_state == old_event.to_state and revision == old["revision"]
                                and old_event.to_state == to_state
                                and tuple(old_event.evidence_refs) == refs
                                and old_event.policy_version == policy_version
                                and old_event.actor == actor
                                and old_event.notes == notes):
                            old_bundle = asset.latest_evidence_ref
                            if evidence_bundle_ref is None and old_bundle is None:
                                return CommittedTransition(asset, old_event, revision)
                            if evidence_bundle_ref is not None and old_bundle is not None and evidence_to_json(evidence_bundle_ref) == evidence_to_json(old_bundle):
                                return CommittedTransition(asset, old_event, revision)
                        raise LifecycleConflictError(f"Conflicting transition decision_id {decision_id} for factor {factor_id}")
                    if evidence_bundle_ref is not None:
                        if factor_id not in evidence_bundle_ref.factor_ids: raise LifecycleConflictError(f"Evidence bundle {evidence_bundle_ref.bundle_id} does not contain factor {factor_id}")
                        if to_state != LifecycleState.EVALUATED: raise LifecycleConflictError("evidence_bundle_ref may only be committed with an EVALUATED transition")
                if evidence_bundle_ref is not None:
                    if factor_id not in evidence_bundle_ref.factor_ids: raise LifecycleConflictError(f"Evidence bundle {evidence_bundle_ref.bundle_id} does not contain factor {factor_id}")
                    if to_state != LifecycleState.EVALUATED: raise LifecycleConflictError("evidence_bundle_ref may only be committed with an EVALUATED transition")
                validate_transition(asset.lifecycle_state, to_state, set(refs))
                now = datetime.now(timezone.utc).isoformat(); updates = {"lifecycle_state": to_state}
                if to_state == LifecycleState.EVALUATED:
                    if evidence_bundle_ref is not None: updates["latest_evidence_ref"] = evidence_bundle_ref
                    if asset.first_evaluated_at is None: updates["first_evaluated_at"] = now
                elif to_state == LifecycleState.APPROVED and asset.approved_at is None: updates["approved_at"] = now
                elif to_state == LifecycleState.PRODUCTION_READY and asset.production_ready_at is None: updates["production_ready_at"] = now
                updated = replace(asset, **updates); event = StateEvent(factor_id, asset.lifecycle_state, to_state, now, refs, decision_id, policy_version, actor, notes); new_revision = revision + 1
                conn.execute("UPDATE assets SET payload=?,revision=? WHERE factor_id=?", (asset_to_json(updated), new_revision, factor_id)); conn.execute("INSERT INTO lifecycle_events(factor_id,revision,decision_id,payload) VALUES (?,?,?,?)", (factor_id,new_revision,decision_id,event_to_json(event))); conn.commit(); return CommittedTransition(updated,event,new_revision)
            except (sqlite3.Error, OSError, ValueError, TypeError,
                    FactorAssetsError, AssetNotFoundError):
                # Governance/contract errors (DuplicateIdentityError etc.) must
                # roll back and still propagate by their own type.
                conn.rollback(); raise

    def transition(self, *args, **kwargs): return self.commit_transition(*args, **kwargs).asset
    def get_events(self, factor_id=None):
        with self._connection() as conn:
            rows = conn.execute("SELECT payload FROM lifecycle_events" + (" WHERE factor_id=? ORDER BY id" if factor_id else " ORDER BY id"), ((factor_id,) if factor_id else ())).fetchall()
        return [event_from_json(r[0]) for r in rows]
    def list_all(self):
        with self._connection() as conn: rows=conn.execute("SELECT payload FROM assets ORDER BY factor_id").fetchall()
        return [asset_from_json(r[0]) for r in rows]
    def list_by_state(self, state): return [a for a in self.list_all() if a.lifecycle_state == state]
    def stats(self):
        assets=self.list_all(); events=len(self.get_events()); by={}
        for a in assets: by[a.lifecycle_state]=by.get(a.lifecycle_state,0)+1
        ts=[a.registered_at for a in assets]; return RepositoryStats(len(assets),by,events,min(ts) if ts else None,max(ts) if ts else None)
