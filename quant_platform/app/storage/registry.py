"""ArtifactRegistry — artifact registry over memory or sqlite (spec §7.2/§22).

Stores published :class:`quant_platform.app.contracts.ArtifactRef` entries keyed
by their immutable object-bytes ``content_hash``.

Design
------
- ONE implementation behind both "memory" and "sqlite" modes: a sqlite3 database.
  ``db_path=None`` opens ``":memory:"``; a real path opens a persistent file.
  The uniform interface is ``register`` / ``resolve(content_hash)`` /
  ``contains(content_hash)`` / ``list_versions(artifact_id)``.
- Idempotency: re-registering the same ``(content_hash, artifact_type)`` returns
  the existing entry instead of inserting a duplicate (spec: same content_hash,
  same type, repeat register is idempotent → return the existing row).
- Unknown artifact types (outside ``ARTIFACT_TYPES``) are rejected with
  ``UnknownArtifactTypeError`` (a ``ValueError``). The registry re-validates
  defensively even though ``ArtifactRef`` already validates its own shape.
- A per-``artifact_id`` version number is assigned on first registration of that
  id and incremented on each *new* content_hash registration, so
  ``list_versions(artifact_id)`` returns the version history in registration
  order. The ``(content_hash, artifact_type)`` pair is the row primary key —
  the same bytes registered under two different artifact types are two distinct
  entries.

Only ``quant_platform.app.contracts`` (PURE-DTO boundary) is imported.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.app.contracts import ARTIFACT_TYPES, ArtifactRef

__all__ = ["ArtifactRegistry", "UnknownArtifactTypeError"]


class UnknownArtifactTypeError(ValueError):
    """Raised when ``register`` is handed an artifact of an unknown type."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    content_hash   TEXT NOT NULL,
    artifact_type  TEXT NOT NULL,
    artifact_id    TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    storage_uri    TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    producer_type  TEXT NOT NULL,
    producer_version TEXT NOT NULL,
    semantic_hash  TEXT NOT NULL DEFAULT '',
    media_type     TEXT NOT NULL DEFAULT 'application/octet-stream',
    producer_source_ref TEXT,
    snapshot_ref   TEXT,
    universe_ref   TEXT,
    security_classification TEXT,
    version        INTEGER NOT NULL,
    PRIMARY KEY (content_hash, artifact_type)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_by_artifact_id
    ON artifacts (artifact_id, version);
"""

_INSERT_SQL = """
INSERT INTO artifacts (
    content_hash, artifact_type, artifact_id, schema_version, storage_uri,
    size_bytes, created_at, producer_type, producer_version, semantic_hash,
    media_type, producer_source_ref, snapshot_ref, universe_ref,
    security_classification, version
) VALUES (
    :content_hash, :artifact_type, :artifact_id, :schema_version, :storage_uri,
    :size_bytes, :created_at, :producer_type, :producer_version, :semantic_hash,
    :media_type, :producer_source_ref, :snapshot_ref, :universe_ref,
    :security_classification, :version
)
"""

_SELECT_BY_HASH_TYPE = """
SELECT * FROM artifacts WHERE content_hash = ? AND artifact_type = ?
"""

_SELECT_BY_HASH = """
SELECT * FROM artifacts WHERE content_hash = ? ORDER BY rowid ASC LIMIT 1
"""

_SELECT_BY_ARTIFACT_ID = """
SELECT * FROM artifacts WHERE artifact_id = ? ORDER BY version ASC
"""

_SELECT_COUNT_ARTIFACT_ID = """
SELECT COUNT(*) AS n FROM artifacts WHERE artifact_id = ?
"""


def _parse_dt(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


class ArtifactRegistry:
    """In-memory + sqlite unified artifact registry.

    ``db_path=None`` → ``":memory:"`` (ephemeral); otherwise a persistent sqlite
    file. Same API regardless of mode.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path
        if db_path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ---- core API ---------------------------------------------------------

    def register(self, artifact: ArtifactRef) -> ArtifactRef:
        """Register ``artifact``; idempotent per ``(content_hash, artifact_type)``.

        Unknown artifact types are rejected with ``UnknownArtifactTypeError``
        before any row is touched. Returns the stored ``ArtifactRef`` (for a
        duplicate registration, the *existing* entry is returned unchanged).
        """
        artifact_type = str(getattr(artifact, "artifact_type", "") or "")
        if not artifact_type:
            raise ValueError("artifact is missing artifact_type")
        if artifact_type not in ARTIFACT_TYPES:
            raise UnknownArtifactTypeError(
                f"unknown artifact_type: {artifact_type!r}"
            )
        content_hash = getattr(artifact, "content_hash", "")
        existing = self._fetch_row(content_hash, artifact_type)
        if existing is not None:
            return self._row_to_ref(existing)
        version = self._next_version(getattr(artifact, "artifact_id", ""))
        row = self._row_from_artifact(artifact, version)
        self._conn.execute(_INSERT_SQL, row)
        self._conn.commit()
        stored = self._fetch_row(content_hash, artifact_type)
        assert stored is not None  # just inserted
        return self._row_to_ref(stored)

    def resolve(self, content_hash: str) -> ArtifactRef | None:
        """Resolve an artifact by its content hash; ``None`` on miss.

        The same ``content_hash`` registered under multiple artifact types
        resolves to the *first registered* entry (rowid order), which is
        deterministic.
        """
        cur = self._conn.execute(_SELECT_BY_HASH, (content_hash,))
        row = cur.fetchone()
        return self._row_to_ref(row) if row is not None else None

    def contains(self, content_hash: str) -> bool:
        """True iff any entry with ``content_hash`` is registered."""
        return self.resolve(content_hash) is not None

    def list_versions(self, artifact_id: str) -> list[ArtifactRef]:
        """All registered versions of ``artifact_id`` in registration order."""
        cur = self._conn.execute(_SELECT_BY_ARTIFACT_ID, (artifact_id,))
        return [self._row_to_ref(r) for r in cur.fetchall()]

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "ArtifactRegistry":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- internals --------------------------------------------------------

    def _fetch_row(self, content_hash: str, artifact_type: str):
        cur = self._conn.execute(_SELECT_BY_HASH_TYPE, (content_hash, artifact_type))
        return cur.fetchone()

    def _next_version(self, artifact_id: str) -> int:
        cur = self._conn.execute(_SELECT_COUNT_ARTIFACT_ID, (artifact_id,))
        row = cur.fetchone()
        return int(row["n"]) + 1

    def _row_from_artifact(self, artifact: Any, version: int) -> dict[str, Any]:
        created_at = getattr(artifact, "created_at", datetime.now())
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        return {
            "content_hash": getattr(artifact, "content_hash", ""),
            "artifact_type": getattr(artifact, "artifact_type", ""),
            "artifact_id": getattr(artifact, "artifact_id", ""),
            "schema_version": getattr(artifact, "schema_version", ""),
            "storage_uri": getattr(artifact, "storage_uri", ""),
            "size_bytes": int(getattr(artifact, "size_bytes", 0)),
            "created_at": created_at,
            "producer_type": getattr(artifact, "producer_type", ""),
            "producer_version": getattr(artifact, "producer_version", ""),
            "semantic_hash": getattr(artifact, "semantic_hash", "") or "",
            "media_type": getattr(artifact, "media_type", "application/octet-stream")
            or "application/octet-stream",
            "producer_source_ref": getattr(artifact, "producer_source_ref", None),
            "snapshot_ref": getattr(artifact, "snapshot_ref", None),
            "universe_ref": getattr(artifact, "universe_ref", None),
            "security_classification": getattr(
                artifact, "security_classification", None
            ),
            "version": int(version),
        }

    def _row_to_ref(self, row: sqlite3.Row) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=row["artifact_id"],
            artifact_type=row["artifact_type"],
            schema_version=row["schema_version"],
            content_hash=row["content_hash"],
            storage_uri=row["storage_uri"],
            size_bytes=int(row["size_bytes"]),
            created_at=_parse_dt(str(row["created_at"])),
            producer_type=row["producer_type"],
            producer_version=row["producer_version"],
            semantic_hash=row["semantic_hash"] or "",
            media_type=row["media_type"] or "application/octet-stream",
            producer_source_ref=row["producer_source_ref"],
            snapshot_ref=row["snapshot_ref"],
            universe_ref=row["universe_ref"],
            security_classification=row["security_classification"],
        )