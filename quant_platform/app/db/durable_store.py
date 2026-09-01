# -*- coding: utf-8 -*-
"""Durable anti-replay state store for the orchestrator (P0-PLAT-004).

The ``Pipeline`` keeps anti-replay state in memory only — ``_processed_fingerprints``
and ``_consumed_manifests``.  A process restart loses both, so a replayed batch
passes anti-replay.  This module defines the small durable backing store seam:

- :class:`RunStateStore` — the Protocol the ``Pipeline`` constructor accepts as
  ``run_storage``.  When provided, the pipeline seeds its in-memory anti-replay
  state from the store at construction and records each NEW consumption + the
  batch fingerprint so a restart preserves anti-replay.  When NO store is
  provided the pipeline behaves EXACTLY as today (pure in-memory).
- :class:`SqliteRunStateStore` — stdlib ``sqlite3`` implementation backed by a
  file or ``:memory:`` connection.  Idempotent: recording the same fingerprint
  or consuming the same candidate twice is a no-op.
- :class:`PostgresRunStateStore` — implementation backed by the existing
  ``PostgresDb`` backend.  It follows the same fail-closed convention as
  ``postgres_backend.py``: psycopg2 is NOT a declared dependency, so
  construction raises ``PostgresBackendUnavailable`` when the driver is missing
  and every SQL statement is passed through ``PostgresDialect.adapt`` so the
  same ``?``-placeholder statements run against PG unchanged.

Both implementations write through the backend's ``transaction()`` adapter
(the SQLite-style ``execute`` surface), which commits on success and rolls back
on exception — a crash mid-record never leaves a half-applied row.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "RunStateStore",
    "SqliteRunStateStore",
    "PostgresRunStateStore",
]

#: SQL shared by both backends.  ``?`` placeholders are rewritten to ``%s`` by
#: ``PostgresDialect.adapt`` when the statement runs through ``PostgresDb``.
_INSERT_FINGERPRINT = (
    "INSERT OR IGNORE INTO batch_fingerprints (fingerprint, consumed_at) "
    "VALUES (?, ?)"
)
_SELECT_FINGERPRINT = "SELECT 1 FROM batch_fingerprints WHERE fingerprint = ?"
_INSERT_CONSUMED = (
    "INSERT OR IGNORE INTO consumed_manifests "
    "(candidate_id, content_hash, semantic, consumed_at) VALUES (?, ?, ?, ?)"
)
_SELECT_CONSUMED_HASHES = "SELECT content_hash FROM consumed_manifests"


@runtime_checkable
class RunStateStore(Protocol):
    """Durable anti-replay backing store for :class:`Pipeline` (P0-PLAT-004).

    ``Pipeline`` consults the store at construction (seeding) and during
    ``run()`` (anti-replay check + recording).  Implementations must be
    idempotent — recording the same fingerprint or consuming the same candidate
    twice is a no-op.
    """

    def record_fingerprint(self, fingerprint: str) -> None:
        """Durably record a batch fingerprint as consumed."""
        ...

    def has_fingerprint(self, fingerprint: str) -> bool:
        """True if the batch fingerprint was already recorded."""
        ...

    def consumed_fingerprints(self) -> set[str]:
        """All batch fingerprints recorded across the store's lifetime."""
        ...

    def record_consumed(
        self,
        candidate_id: str,
        content_hash: str,
        semantic: str,
        consumed_at: datetime,
    ) -> None:
        """Durably record one NEW candidate consumption."""
        ...

    def consumed_hashes(self) -> set[str]:
        """All content hashes consumed across the store's lifetime."""
        ...


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class SqliteRunStateStore:
    """Stdlib ``sqlite3`` implementation of :class:`RunStateStore`.

    ``path`` may be ``":memory:"`` (default) or a file path.  The store opens
    its own connection and creates the durable tables on construction (all
    ``CREATE TABLE IF NOT EXISTS`` — idempotent across re-opens).
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS batch_fingerprints ("
            " fingerprint TEXT PRIMARY KEY,"
            " consumed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS consumed_manifests ("
            " candidate_id TEXT PRIMARY KEY,"
            " content_hash TEXT NOT NULL,"
            " semantic TEXT NOT NULL,"
            " consumed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_fingerprint(self, fingerprint: str) -> None:
        self._conn.execute(
            _INSERT_FINGERPRINT,
            (fingerprint, _now_utc().isoformat()),
        )
        self._conn.commit()

    def has_fingerprint(self, fingerprint: str) -> bool:
        row = self._conn.execute(
            _SELECT_FINGERPRINT, (fingerprint,)
        ).fetchone()
        return row is not None

    def consumed_fingerprints(self) -> set[str]:
        rows = self._conn.execute("SELECT fingerprint FROM batch_fingerprints").fetchall()
        return {str(r["fingerprint"]) for r in rows}

    def record_consumed(
        self,
        candidate_id: str,
        content_hash: str,
        semantic: str,
        consumed_at: datetime,
    ) -> None:
        self._conn.execute(
            _INSERT_CONSUMED,
            (candidate_id, content_hash, semantic, consumed_at.isoformat()),
        )
        self._conn.commit()

    def consumed_hashes(self) -> set[str]:
        rows = self._conn.execute(_SELECT_CONSUMED_HASHES).fetchall()
        return {str(r["content_hash"]) for r in rows}


class PostgresRunStateStore:
    """PostgreSQL implementation of :class:`RunStateStore`.

    Backed by the existing ``PostgresDb`` backend and its ``transaction()``
    adapter (P0-PLAT-008), which yields a SQLite-style ``execute`` surface.  The
    same ``?``-placeholder SQL above runs unchanged against PG via
    ``PostgresDialect.adapt`` (R55 #92).

    Fail-closed like ``postgres_backend.py``: psycopg2 is NOT a declared
    dependency, so construction raises ``PostgresBackendUnavailable`` when the
    driver is missing.  The durable tables are expected to exist — ``create_schema``
    (or the migration runner, P0-PLAT-015) applies them.
    """

    def __init__(self, db: Any) -> None:
        # ``PostgresDb`` already raised ``PostgresBackendUnavailable`` on a
        # missing driver at construction; we only need the adapter surface.
        self._db = db

    def record_fingerprint(self, fingerprint: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                _INSERT_FINGERPRINT,
                (fingerprint, _now_utc().isoformat()),
            )

    def has_fingerprint(self, fingerprint: str) -> bool:
        rows = self._db.query(_SELECT_FINGERPRINT, (fingerprint,))
        return bool(rows)

    def consumed_fingerprints(self) -> set[str]:
        rows = self._db.query("SELECT fingerprint FROM batch_fingerprints")
        return {str(r["fingerprint"]) for r in rows}

    def record_consumed(
        self,
        candidate_id: str,
        content_hash: str,
        semantic: str,
        consumed_at: datetime,
    ) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                _INSERT_CONSUMED,
                (candidate_id, content_hash, semantic, consumed_at.isoformat()),
            )

    def consumed_hashes(self) -> set[str]:
        rows = self._db.query(_SELECT_CONSUMED_HASHES)
        return {str(r["content_hash"]) for r in rows}
