# -*- coding: utf-8 -*-
"""Minimal honest schema migration runner (P0-PLAT-015).

No SQLAlchemy / Alembic.  A migration is a versioned list of raw DDL
statements; ``apply_migrations`` applies only the unapplied ones, in order,
each inside its own transaction, and records the version in the same
transaction — so a crash mid-migration rolls the DDL back and the version is
never recorded as applied.  Re-runs are no-ops (idempotent across re-runs), and
a partially-applied state resumes correctly.

Version bookkeeping lives in a ``schema_migrations`` table
(``version TEXT PRIMARY KEY, applied_at TIMESTAMP``) created on first use.
``version`` strings sort naturally as migration ids (``v1``..``vN``).

The runner accepts both a raw DB-API connection (``sqlite3.Connection``) and
the ``PostgresDb`` backend: the two shared DDL applications in
``postgres_backend.py`` (``create_schema``, statement-by-statement) are re-used
so a statement's ``CREATE TABLE IF NOT EXISTS`` + PG-compatible dialect rules
stay in ONE place.  On a raw ``sqlite3.Connection``, ``SAVEPOINT``/``ROLLBACK
TO`` is used for the in-transaction rollback (``CREATE TABLE`` is non-transactional
in SQLite without one); ``PostgresDb.transaction()`` provides the same guarantee
natively (a live PG backend is exercised by the failing-closed
``test_postgres_live.py`` convention).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from .schema import (
    BATCH_FINGERPRINTS,
    CONSUMED_MANIFESTS,
    STAGE_IDEMPOTENCY_KEYS,
    WORKFLOW_STAGE_RUNS,
)

__all__ = [
    "Migration",
    "MIGRATIONS",
    "apply_migrations",
    "DEFAULT_CURRENT_VERSION_TABLE",
]

DEFAULT_CURRENT_VERSION_TABLE = "schema_migrations"

#: DDL for the version bookkeeping table (portable TEXT/TIMESTAMP).
_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMP NOT NULL
)
"""


class Migration:
    """One versioned schema migration: an ordered list of DDL statements."""

    def __init__(
        self,
        *,
        id: str,
        description: str,
        statements: tuple[str, ...],
    ) -> None:
        if not id or not description:
            raise ValueError("migration id and description are required")
        if not statements:
            raise ValueError(f"migration {id!r} must contain at least one statement")
        self.id = id
        self.description = description
        self.statements = tuple(statements)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Migration(id={self.id!r}, statements={len(self.statements)})"


# ---------------------------------------------------------------------------
# P0-PLAT-004 durable orchestration tables, in FK-safe order (parents first).
# ---------------------------------------------------------------------------

MIGRATIONS: list[Migration] = [
    Migration(
        id="v1",
        description="durable orchestration state: workflow_stage_runs",
        statements=(WORKFLOW_STAGE_RUNS,),
    ),
    Migration(
        id="v2",
        description="durable orchestration state: consumed_manifests",
        statements=(CONSUMED_MANIFESTS,),
    ),
    Migration(
        id="v3",
        description="durable orchestration state: batch_fingerprints",
        statements=(BATCH_FINGERPRINTS,),
    ),
    Migration(
        id="v4",
        description="durable orchestration state: stage_idempotency_keys",
        statements=(STAGE_IDEMPOTENCY_KEYS,),
    ),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _applied_versions(conn: Any, table: str) -> set[str]:
    """The set of already-applied version ids (empty when the bookkeeping
    table does not exist yet — first run)."""
    try:
        if hasattr(conn, "query"):
            rows = conn.query(f"SELECT version FROM {table}")
        else:
            # Raw sqlite3.Connection may return tuples (default row factory).
            rows = conn.execute(f"SELECT version FROM {table}").fetchall()
    except Exception:
        return set()
    versions: set[str] = set()
    for r in rows:
        if isinstance(r, dict):
            versions.add(str(r["version"]))
        else:
            versions.add(str(r[0]))
    return versions


def apply_migrations(
    conn: Any,
    *,
    current_version_table: str = DEFAULT_CURRENT_VERSION_TABLE,
) -> tuple[str, ...]:
    """Apply unapplied migrations in order; return the applied version ids.

    ``conn`` is either a raw ``sqlite3.Connection`` (``execute`` +
    ``executescript``) or the ``PostgresDb`` backend (``execute`` / ``query`` /
    ``transaction``).  Each migration runs inside one transaction with its
    version row recorded in the SAME transaction, so a failure rolls both back
    and the migration is NOT recorded as applied — a re-run resumes it.

    Idempotent across re-runs: a second call with every version already applied
    is a no-op (no error, nothing re-applied).
    """
    # Ensure the bookkeeping table exists (idempotent).  On the raw sqlite3
    # connection this executes immediately; ``PostgresDb.execute`` is
    # autocommit=False so the DDL lands with the first migration's commit.
    if hasattr(conn, "_conn"):
        conn._conn.executescript(_MIGRATIONS_DDL)
    elif hasattr(conn, "query"):
        conn.execute(_MIGRATIONS_DDL)
    else:
        conn.executescript(_MIGRATIONS_DDL)

    applied = _applied_versions(conn, current_version_table)
    if len(applied) == len(MIGRATIONS) and applied == {m.id for m in MIGRATIONS}:
        return ()

    newly_applied: list[str] = []
    for migration in MIGRATIONS:
        if migration.id in applied:
            continue
        if hasattr(conn, "_conn"):
            # SqliteDb backend: run the DDL + version row through SAVEPOINTs on
            # the raw connection (its ``transaction()`` yields a plain sqlite3
            # connection whose ROLLBACK cannot undo an auto-committed CREATE
            # TABLE in default isolation).
            raw = conn._conn
            _run_sqlite_savepoint_migration(raw, migration)
        elif hasattr(conn, "query"):
            # PostgresDb: one transaction per migration (DDL + version row
            # commit atomically; a failure rolls both back natively).
            with conn.transaction() as tx:
                for statement in migration.statements:
                    tx.execute(statement)
                tx.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?) "
                    "ON CONFLICT (version) DO NOTHING",
                    (migration.id, _now()),
                )
        else:
            # Raw sqlite3.Connection: ``CREATE TABLE`` is non-transactional in
            # SQLite outside a SAVEPOINT, so wrap each migration in one
            # SAVEPOINT that is rolled back to on failure and released on
            # success — the DDL and the version row stay atomic.
            _run_sqlite_savepoint_migration(conn, migration)
        newly_applied.append(migration.id)

    # Raw sqlite3 connections commit on close; persist the version rows now so
    # a later connection sees them.
    if not hasattr(conn, "query") or hasattr(conn, "_conn"):
        try:
            if hasattr(conn, "_conn"):
                conn._conn.commit()
            else:
                conn.commit()
        except Exception:  # pragma: no cover - no active transaction
            pass
    return tuple(newly_applied)


def _run_sqlite_savepoint_migration(conn, migration: Migration) -> None:
    """Run one migration inside a SAVEPOINT on a raw ``sqlite3.Connection``.

    On failure the savepoint is rolled back to AND released so the connection is
    left usable and the partial DDL + version row are both undone.
    """
    savepoint = f"migration_{migration.id}"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for statement in migration.statements:
            conn.execute(statement)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (migration.id, _now()),
        )
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
