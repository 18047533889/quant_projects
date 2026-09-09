"""Durable artifact-generation publication coordinated through the outbox.

The metadata transaction stages an immutable artifact registry row, generation
state, and a unique publish intent together. Blob I/O happens only after commit
in :meth:`publish`, which is the ``Outbox`` publisher callback. Readers resolve
only the active COMPLETE generation.
"""
from __future__ import annotations

import json
import hashlib
import uuid
from enum import Enum
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Callable

from quant_platform.app.contracts import (
    ArtifactRef, DeletionReceipt, DeletionStatus, GCObject, GCRootSnapshot,
    GCTombstoneClaim, TombstoneClaimStatus,
)
from quant_platform.app.outbox import Outbox

_UNCONDITIONAL = object()


class DurableGenerationCoordinator:
    """Transactional metadata writer and idempotent post-commit publisher."""

    EVENT_TYPE = "ArtifactGenerationPublishRequested"
    GC_ROOT_KINDS = frozenset({
        "production", "approved_release", "active_read", "retryable_job",
        "retained_research", "rollback", "pending_label",
    })

    def __init__(
        self,
        db: Any,
        artifact_publisher: Any,
        *,
        failure_injector: Callable[[str, str], None] | None = None,
    ) -> None:
        self.db = db
        self.artifact_publisher = artifact_publisher
        self.failure_injector = failure_injector
        self.outbox = Outbox(db, self)
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_gc_state(singleton, reference_epoch) VALUES(1, 1)"
            )

    def stage(self, artifact: ArtifactRef, data: bytes, *, expected_parent_generation=_UNCONDITIONAL) -> str:
        """Atomically register metadata, stage bytes, and enqueue publish intent."""
        if not isinstance(data, bytes):
            raise TypeError("generation payload must be immutable bytes")
        if hashlib.sha256(data).hexdigest() != artifact.content_hash:
            raise ValueError("generation payload content_hash mismatch")
        if len(data) != artifact.size_bytes:
            raise ValueError("generation payload size_bytes mismatch")
        # Equal bytes belonging to different artifacts are distinct generations.
        identity = json.dumps([artifact.artifact_id, artifact.content_hash],
                              separators=(",", ":")).encode("utf-8")
        generation_id = "gen:v2:" + hashlib.sha256(identity).hexdigest()
        payload_json = json.dumps(self._artifact_payload(artifact), sort_keys=True)
        data_hex = data.hex()
        with self.db.transaction() as conn:
            conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch WHERE singleton=1")
            existing = conn.execute(
                "SELECT * FROM artifact_generations WHERE artifact_id=? AND content_hash=?",
                (artifact.artifact_id, artifact.content_hash),
            ).fetchone()
            if existing is not None:
                # Keep pre-v2 identities usable without relabeling old history.
                generation_id = existing["generation_id"]
                previous = json.loads(existing["artifact_json"])
                current = json.loads(payload_json)
                # Retry creation time is observational, not semantic identity.
                previous.pop("created_at", None)
                current.pop("created_at", None)
                if previous != current or existing["payload_hex"] != data_hex:
                    raise ValueError("conflicting immutable generation metadata")
                if expected_parent_generation is not _UNCONDITIONAL:
                    intent = conn.execute(
                        "SELECT payload_json FROM outbox_events WHERE idempotency_key=?",
                        (f"publish:{generation_id}",),
                    ).fetchone()
                    if intent is None:
                        raise ValueError("existing generation has no durable publication intent")
                    staged_parent = json.loads(intent["payload_json"]).get(
                        "expected_parent_generation", _UNCONDITIONAL
                    )
                    if (staged_parent is _UNCONDITIONAL
                            or staged_parent != expected_parent_generation):
                        raise ValueError("idempotent generation retry changed CAS parent")
            elif expected_parent_generation is not _UNCONDITIONAL:
                # Include pending intents: two writers based on one snapshot
                # cannot both stage replacements before either is published.
                latest = conn.execute(
                    "SELECT g.generation_id FROM artifact_generations g JOIN outbox_events e "
                    "ON e.aggregate_id=g.generation_id WHERE g.artifact_id=? AND e.event_type=? "
                    "ORDER BY e.id DESC LIMIT 1", (artifact.artifact_id, self.EVENT_TYPE),
                ).fetchone()
                actual = latest['generation_id'] if latest is not None else None
                if actual != expected_parent_generation:
                    raise ValueError('stale generation parent: CAS conflict')
            tombstoned = conn.execute(
                "SELECT 1 FROM artifact_gc_tombstones WHERE artifact_id=? AND generation_id=?",
                (artifact.artifact_id, generation_id),
            ).fetchone()
            if tombstoned is not None:
                raise ValueError("cannot stage a tombstoned artifact generation")
            conn.execute(
                "INSERT OR IGNORE INTO artifacts (artifact_id, artifact_type, schema_version, "
                "semantic_hash, content_hash, storage_uri, size_bytes, media_type, producer_type, "
                "producer_version, producer_source_ref, snapshot_ref, universe_ref, "
                "security_classification, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact.artifact_id, artifact.artifact_type, artifact.schema_version,
                 artifact.semantic_hash, artifact.content_hash, artifact.storage_uri,
                 artifact.size_bytes, artifact.media_type, artifact.producer_type,
                 artifact.producer_version, artifact.producer_source_ref, artifact.snapshot_ref,
                 artifact.universe_ref,
                 artifact.security_classification.value
                 if isinstance(artifact.security_classification, Enum)
                 else artifact.security_classification,
                 artifact.created_at.isoformat()),
            )
            conn.execute(
                "INSERT OR IGNORE INTO artifact_generations "
                "(generation_id, artifact_id, content_hash, artifact_json, payload_hex, status, active) "
                "VALUES (?, ?, ?, ?, ?, 'STAGED', FALSE)",
                (generation_id, artifact.artifact_id, artifact.content_hash, payload_json, data_hex),
            )
            intent_key = f"publish:{generation_id}"
            exists = conn.execute(
                "SELECT 1 FROM outbox_events WHERE idempotency_key = ?", (intent_key,)
            ).fetchone()
            if exists is None:
                self.outbox.emit(
                    event_type=self.EVENT_TYPE,
                    aggregate_type="artifact_generation",
                    aggregate_id=generation_id,
                    correlation_id=artifact.artifact_id,
                    idempotency_key=intent_key,
                    payload={"generation_id": generation_id, **(
                        {'expected_parent_generation': expected_parent_generation}
                        if expected_parent_generation is not _UNCONDITIONAL else {})},
                )
            self._fail("transaction_before_commit", generation_id)
        return generation_id

    def publish(self, event: dict[str, Any]) -> None:
        """Outbox callback: publish bytes idempotently, then expose COMPLETE."""
        generation_id = str(event["payload"]["generation_id"])
        with self.db.transaction() as conn:
            # Serialize the external exact-version write with root creation and
            # tombstone claims. This prevents a checked-then-tombstoned publish
            # from recreating bytes after GC has issued an absent receipt.
            conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch WHERE singleton=1")
            row = conn.execute(
                "SELECT * FROM artifact_generations WHERE generation_id = ?", (generation_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"unknown generation {generation_id}")
            tombstoned = conn.execute(
                "SELECT 1 FROM artifact_gc_tombstones WHERE artifact_id=? AND generation_id=?",
                (row["artifact_id"], generation_id),
            ).fetchone()
            if tombstoned is not None:
                raise ValueError("cannot publish a tombstoned artifact generation")
            # Delivery replay is not authorization to roll back a newer active
            # generation. Inactive COMPLETE history must remain inactive.
            if row["status"] == "COMPLETE":
                return
            if 'expected_parent_generation' in event['payload']:
                active_parent = conn.execute(
                    "SELECT generation_id FROM artifact_generations WHERE artifact_id=? AND active=TRUE",
                    (row['artifact_id'],),
                ).fetchone()
                actual_parent = active_parent['generation_id'] if active_parent is not None else None
                if actual_parent != event['payload']['expected_parent_generation']:
                    raise ValueError('generation parent not active: ordered publication required')
            intent = conn.execute(
                "SELECT id FROM outbox_events WHERE idempotency_key=?",
                (f"publish:{generation_id}",),
            ).fetchone()
            if intent is None:
                raise ValueError("generation has no durable publication intent")
            # The durable outbox sequence is assigned atomically at stage time.
            # A delayed older intent may complete for audit, but must not
            # replace a newer staged generation that has already completed.
            newer_active = conn.execute(
                "SELECT 1 FROM artifact_generations g JOIN outbox_events e "
                "ON e.aggregate_id=g.generation_id "
                "WHERE g.artifact_id=? AND g.active=TRUE AND e.event_type=? AND e.id>?",
                (row["artifact_id"], self.EVENT_TYPE, intent["id"]),
            ).fetchone()
            artifact = self._artifact_from_payload(json.loads(row["artifact_json"]))
            resolved = self.artifact_publisher.publish(artifact, bytes.fromhex(row["payload_hex"]))
            if resolved.content_hash != row["content_hash"]:
                raise ValueError("publisher returned a different content_hash")
            self._fail("blob_written_before_complete", generation_id)
            if newer_active is None:
                conn.execute(
                    "UPDATE artifact_generations SET active = FALSE WHERE artifact_id = ?",
                    (row["artifact_id"],),
                )
            conn.execute(
                "UPDATE artifact_generations SET status = 'COMPLETE', active = ?, "
                "resolved_storage_uri = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE generation_id = ?",
                (newer_active is None, resolved.storage_uri, generation_id),
            )
            conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch+1 WHERE singleton=1")
            self._fail("complete_before_commit", generation_id)

    def resolve_active(self, artifact_id: str) -> dict[str, Any] | None:
        rows = self.db.query(
            "SELECT * FROM artifact_generations WHERE artifact_id = ? "
            "AND status = 'COMPLETE' AND active = TRUE",
            (artifact_id,),
        )
        return rows[0] if rows else None

    # -- root-aware GC authority -------------------------------------------
    def protect_gc_root(self, root_kind: str, artifact_id: str, generation_id: str) -> None:
        if root_kind not in self.GC_ROOT_KINDS:
            raise ValueError(f"unknown GC root kind {root_kind!r}")
        with self.db.transaction() as conn:
            # Acquire the GC epoch row write lock before checking tombstones.
            conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch WHERE singleton=1")
            if conn.execute(
                "SELECT 1 FROM artifact_gc_tombstones WHERE artifact_id=? AND generation_id=?",
                (artifact_id, generation_id),
            ).fetchone() is not None:
                raise ValueError("cannot create a reference to a tombstoned generation")
            if conn.execute(
                "SELECT 1 FROM artifact_generations WHERE artifact_id=? AND generation_id=?",
                (artifact_id, generation_id),
            ).fetchone() is None:
                raise ValueError("unknown artifact generation")
            changed = conn.execute(
                "INSERT OR IGNORE INTO artifact_gc_roots(root_kind,artifact_id,generation_id) VALUES(?,?,?)",
                (root_kind, artifact_id, generation_id),
            ).rowcount
            if changed:
                conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch+1 WHERE singleton=1")

    def release_gc_root(self, root_kind: str, artifact_id: str, generation_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch WHERE singleton=1")
            changed = conn.execute(
                "DELETE FROM artifact_gc_roots WHERE root_kind=? AND artifact_id=? AND generation_id=?",
                (root_kind, artifact_id, generation_id),
            ).rowcount
            if changed:
                conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch+1 WHERE singleton=1")

    def record_gc_reference(self, child_artifact_id: str, child_generation_id: str,
                            parent_artifact_id: str, parent_generation_id: str) -> None:
        """Validate endpoint versions and persist conservative artifact lineage.

        The current lineage schema is artifact-wide: GC protects every version
        of the parent reachable from any version of the child. The supplied
        generation IDs validate existing, non-tombstoned endpoints; they are
        not an assertion of exact-version edge persistence. This can retain
        extra old versions, but must never authorize their deletion.
        """
        with self.db.transaction() as conn:
            conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch WHERE singleton=1")
            for artifact_id, generation_id in ((child_artifact_id, child_generation_id),
                                               (parent_artifact_id, parent_generation_id)):
                if conn.execute(
                    "SELECT 1 FROM artifact_generations WHERE artifact_id=? AND generation_id=?",
                    (artifact_id, generation_id),
                ).fetchone() is None:
                    raise ValueError("unknown artifact generation in GC reference")
                if conn.execute(
                    "SELECT 1 FROM artifact_gc_tombstones WHERE artifact_id=? AND generation_id=?",
                    (artifact_id, generation_id),
                ).fetchone() is not None:
                    raise ValueError("cannot reference a tombstoned generation")
            changed = conn.execute(
                "INSERT OR IGNORE INTO artifact_lineage(child_artifact_id,parent_artifact_id) VALUES(?,?)",
                (child_artifact_id, parent_artifact_id),
            ).rowcount
            if changed:
                conn.execute("UPDATE artifact_gc_state SET reference_epoch=reference_epoch+1 WHERE singleton=1")

    def snapshot_for_gc(self) -> GCRootSnapshot:
        epoch = int(self.db.query(
            "SELECT reference_epoch FROM artifact_gc_state WHERE singleton=1"
        )[0]["reference_epoch"])
        rows = self.db.query("SELECT * FROM artifact_generations")
        objects_list = []
        for row in rows:
            artifact_payload = json.loads(str(row["artifact_json"]))
            objects_list.append(GCObject(
            object_id=str(row["artifact_id"]), object_version=str(row["generation_id"]),
            storage_uri=str(row.get("resolved_storage_uri") or artifact_payload["storage_uri"]),
            content_hash=str(row["content_hash"]), lifecycle_state=str(row["status"]),
            terminal=(str(row["status"]) == "COMPLETE" and not bool(row["active"])),
            size_bytes=int(artifact_payload["size_bytes"]),
        ))
        objects = tuple(objects_list)
        roots_by_kind: dict[str, set[tuple[str, str]]] = {kind: set() for kind in self.GC_ROOT_KINDS}
        for row in rows:
            if str(row["status"]) == "COMPLETE" and bool(row["active"]):
                roots_by_kind["production"].add((str(row["artifact_id"]), str(row["generation_id"])))
        for row in self.db.query("SELECT root_kind,artifact_id,generation_id FROM artifact_gc_roots"):
            roots_by_kind[str(row["root_kind"])].add((str(row["artifact_id"]), str(row["generation_id"])))
        generations: dict[str, list[tuple[str, str]]] = {}
        for obj in objects:
            generations.setdefault(obj.object_id, []).append(obj.key)
        references: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for edge in self.db.query("SELECT child_artifact_id,parent_artifact_id FROM artifact_lineage"):
            for child in generations.get(str(edge["child_artifact_id"]), ()):
                references.setdefault(child, []).extend(generations.get(str(edge["parent_artifact_id"]), ()))
        return GCRootSnapshot(
            epoch, objects, {key: tuple(values) for key, values in references.items()},
            production_roots=frozenset(roots_by_kind["production"]),
            approved_release_roots=frozenset(roots_by_kind["approved_release"]),
            active_read_roots=frozenset(roots_by_kind["active_read"]),
            retryable_job_roots=frozenset(roots_by_kind["retryable_job"]),
            retained_research_roots=frozenset(roots_by_kind["retained_research"]),
            rollback_roots=frozenset(roots_by_kind["rollback"]),
            pending_label_roots=frozenset(roots_by_kind["pending_label"]),
        )

    def claim_gc_tombstone(self, object_id: str, object_version: str, *, expected_epoch: int) -> GCTombstoneClaim:
        with self.db.transaction() as conn:
            locked = conn.execute(
                "UPDATE artifact_gc_state SET reference_epoch=reference_epoch "
                "WHERE singleton=1 AND reference_epoch=?", (expected_epoch,),
            ).rowcount
            if not locked:
                return GCTombstoneClaim(TombstoneClaimStatus.EPOCH_CHANGED, object_id, object_version,
                                        reason="reference epoch changed")
            prior = conn.execute(
                "SELECT claim_id,claimed_epoch FROM artifact_gc_tombstones WHERE artifact_id=? AND generation_id=?",
                (object_id, object_version),
            ).fetchone()
            if prior is not None:
                return GCTombstoneClaim(TombstoneClaimStatus.ALREADY_TERMINAL, object_id, object_version,
                                        str(prior[0]), int(prior[1]), "already tombstoned")
            root = conn.execute(
                "SELECT root_kind FROM artifact_gc_roots WHERE artifact_id=? AND generation_id=?",
                (object_id, object_version),
            ).fetchone()
            active = conn.execute(
                "SELECT active,status FROM artifact_generations WHERE artifact_id=? AND generation_id=?",
                (object_id, object_version),
            ).fetchone()
            if active is None or str(active[1]) != "COMPLETE" or bool(active[0]):
                return GCTombstoneClaim(TombstoneClaimStatus.ROOTED, object_id, object_version,
                                        reason="not an inactive terminal generation")
            if root is not None:
                kind = str(root[0])
                status = TombstoneClaimStatus.LEASED if kind == "active_read" else TombstoneClaimStatus.ROOTED
                return GCTombstoneClaim(status, object_id, object_version, reason=kind)
            claim_id = f"gc:{uuid.uuid4().hex}"
            conn.execute(
                "INSERT INTO artifact_gc_tombstones(artifact_id,generation_id,claim_id,claimed_epoch) VALUES(?,?,?,?)",
                (object_id, object_version, claim_id, expected_epoch),
            )
            return GCTombstoneClaim(TombstoneClaimStatus.CLAIMED, object_id, object_version,
                                    claim_id, expected_epoch)

    def record_deletion_receipt(self, receipt: DeletionReceipt) -> DeletionReceipt:
        payload = self._receipt_payload(receipt)
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_deletion_receipts(artifact_id,generation_id,receipt_json) VALUES(?,?,?)",
                (receipt.object_id, receipt.object_version, payload),
            )
            row = conn.execute(
                "SELECT receipt_json FROM artifact_deletion_receipts WHERE artifact_id=? AND generation_id=?",
                (receipt.object_id, receipt.object_version),
            ).fetchone()
        return self._receipt_from_payload(str(row[0]))

    def get_deletion_receipt(self, object_id: str, object_version: str) -> DeletionReceipt | None:
        rows = self.db.query(
            "SELECT receipt_json FROM artifact_deletion_receipts WHERE artifact_id=? AND generation_id=?",
            (object_id, object_version),
        )
        return None if not rows else self._receipt_from_payload(str(rows[0]["receipt_json"]))

    @staticmethod
    def _receipt_payload(receipt: DeletionReceipt) -> str:
        payload = dict(receipt.__dict__)
        payload["status"] = receipt.status.value
        payload["deleted_at"] = receipt.deleted_at.isoformat()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _receipt_from_payload(payload: str) -> DeletionReceipt:
        data = json.loads(payload)
        data["status"] = DeletionStatus(data["status"])
        data["deleted_at"] = datetime.fromisoformat(data["deleted_at"])
        return DeletionReceipt(**data)

    def _fail(self, stage: str, generation_id: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(stage, generation_id)

    @staticmethod
    def _artifact_payload(a: ArtifactRef) -> dict[str, Any]:
        values = {item.name: getattr(a, item.name) for item in fields(a)}
        return {name: (value.isoformat() if isinstance(value, datetime)
                       else value.value if isinstance(value, Enum) else value)
                for name, value in values.items()}

    @staticmethod
    def _artifact_from_payload(payload: dict[str, Any]) -> ArtifactRef:
        payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        return ArtifactRef(**payload)
