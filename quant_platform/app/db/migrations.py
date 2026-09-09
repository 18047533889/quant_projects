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
import sqlite3
import re
import hashlib

from .schema import (
    ARTIFACT_DELETION_RECEIPTS,
    ARTIFACT_GC_ROOTS,
    ARTIFACT_GC_STATE,
    ARTIFACT_GC_TOMBSTONES,
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
    "postgres_event_clock_migration",
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
    Migration(id="v5", description="artifact GC reference epoch", statements=(ARTIFACT_GC_STATE,)),
    Migration(id="v6", description="artifact GC governed roots and leases", statements=(ARTIFACT_GC_ROOTS,)),
    Migration(id="v7", description="artifact GC exact-version tombstones", statements=(ARTIFACT_GC_TOMBSTONES,)),
    Migration(id="v8", description="artifact GC deletion receipts", statements=(ARTIFACT_DELETION_RECEIPTS,)),
    Migration(
        id="v9",
        description="preserve GC roots while adding pending-label root kind",
        statements=(
            """CREATE TABLE artifact_gc_roots_v9 (
                root_kind TEXT NOT NULL CHECK (root_kind IN
                  ('production','approved_release','active_read','retryable_job',
                   'retained_research','rollback','pending_label')),
                artifact_id TEXT NOT NULL,
                generation_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (root_kind, artifact_id, generation_id)
            )""",
            """INSERT INTO artifact_gc_roots_v9(root_kind,artifact_id,generation_id,created_at)
               SELECT root_kind,artifact_id,generation_id,created_at FROM artifact_gc_roots""",
            "DROP TABLE artifact_gc_roots",
            "ALTER TABLE artifact_gc_roots_v9 RENAME TO artifact_gc_roots",
        ),
    ),
]


from .durable_store import CREATE_CANDIDATE_RESERVATIONS

MIGRATIONS.append(Migration(id='v9z_candidate_reservations',
    description='Durable discovered candidate attempts, leases and retry state',
    statements=(CREATE_CANDIDATE_RESERVATIONS,)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def postgres_event_clock_migration(*, legacy_timezone: str) -> Migration:
    """Build (do not execute) the explicit legacy PG event-clock migration.

    Caller must state the zone of old timezone-naive calendar timestamps.
    Apply through the existing transaction/version runner only after approval
    and backup. Already numeric Unix-second columns are unchanged. Existing
    event IDs are retained while sequence defaults resume above the max ID.
    """
    from zoneinfo import ZoneInfo
    ZoneInfo(legacy_timezone)  # never infer the old clock's timezone
    zone = legacy_timezone.replace("'", "''")
    statements = []
    for table, column in (('outbox_events','occurred_at'), ('outbox_events','claimed_at'),
                          ('outbox_events','next_attempt_at'), ('inbox_events','processed_at'),
                          ('inbox_events','next_attempt_at')):
        statements.append(f"""DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema()
                AND table_name='{table}' AND column_name='{column}'
                AND data_type='timestamp without time zone') THEN
                ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT;
                ALTER TABLE {table} ALTER COLUMN {column} TYPE DOUBLE PRECISION
                    USING EXTRACT(EPOCH FROM ({column} AT TIME ZONE '{zone}'));
            ELSIF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema()
                AND table_name='{table}' AND column_name='{column}'
                AND data_type='timestamp with time zone') THEN
                ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT;
                ALTER TABLE {table} ALTER COLUMN {column} TYPE DOUBLE PRECISION
                    USING EXTRACT(EPOCH FROM {column});
            END IF;
        END $$""")
    for table, column in (('outbox_events','id'), ('audit_logs','id'), ('job_attempts','attempt_id')):
        sequence = f'{table}_{column}_seq'
        statements.extend((f'LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE',
            f'CREATE SEQUENCE IF NOT EXISTS {sequence}',
            f'ALTER SEQUENCE {sequence} OWNED BY {table}.{column}',
            f"SELECT setval('{sequence}', COALESCE((SELECT MAX({column}) FROM {table}),0)+1, false)",
            f"ALTER TABLE {table} ALTER COLUMN {column} SET DEFAULT nextval('{sequence}')"))
    return Migration(id='pg_event_epoch_v1_' + hashlib.sha256(legacy_timezone.encode()).hexdigest()[:12],
        description=f'Explicit Unix-second event clocks from legacy zone {legacy_timezone}; retain surrogate IDs',
        statements=tuple(statements))


def _applied_versions(conn: Any, table: str) -> set[str]:
    """The set of already-applied version ids (empty when the bookkeeping
    table does not exist yet — first run)."""
    if hasattr(conn, "query"):
        rows = conn.query(f"SELECT version FROM {table}")
    else:
        rows = conn.execute(f"SELECT version FROM {table}").fetchall()
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
    migrations: tuple[Migration, ...] | None = None,
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
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', current_version_table):
        raise ValueError('invalid migration version table identifier')
    selected = tuple(MIGRATIONS) if migrations is None else tuple(migrations)
    if len({migration.id for migration in selected}) != len(selected):
        raise ValueError('duplicate migration IDs')
    raw = getattr(conn, '_conn', conn)
    sqlite = isinstance(raw, sqlite3.Connection)
    ddl = _MIGRATIONS_DDL.replace('schema_migrations', current_version_table)
    if sqlite:
        raw.executescript(ddl)
    elif hasattr(conn, "query"):
        conn.execute(ddl)
    else:
        raise TypeError('unsupported migration connection')

    applied = _applied_versions(conn, current_version_table)
    if {m.id for m in selected}.issubset(applied):
        return ()

    newly_applied: list[str] = []
    for migration in selected:
        if migration.id in applied:
            continue
        if sqlite:
            # SqliteDb backend: run the DDL + version row through SAVEPOINTs on
            # the raw connection (its ``transaction()`` yields a plain sqlite3
            # connection whose ROLLBACK cannot undo an auto-committed CREATE
            # TABLE in default isolation).
            _run_sqlite_savepoint_migration(raw, migration, current_version_table)
        elif hasattr(conn, "query"):
            # PostgresDb: one transaction per migration (DDL + version row
            # commit atomically; a failure rolls both back natively).
            with conn.transaction() as tx:
                for statement in migration.statements:
                    tx.execute(statement)
                tx.execute(
                    f"INSERT INTO {current_version_table} (version, applied_at) VALUES (?, ?) "
                    "ON CONFLICT (version) DO NOTHING",
                    (migration.id, _now()),
                )
        else:
            # Raw sqlite3.Connection: ``CREATE TABLE`` is non-transactional in
            # SQLite outside a SAVEPOINT, so wrap each migration in one
            # SAVEPOINT that is rolled back to on failure and released on
            # success — the DDL and the version row stay atomic.
            _run_sqlite_savepoint_migration(conn, migration, current_version_table)
        newly_applied.append(migration.id)

    # Raw sqlite3 connections commit on close; persist the version rows now so
    # a later connection sees them.
    if sqlite:
        raw.commit()
    return tuple(newly_applied)


def _run_sqlite_savepoint_migration(conn, migration: Migration, version_table='schema_migrations') -> None:
    """Run one migration inside a SAVEPOINT on a raw ``sqlite3.Connection``.

    On failure the savepoint is rolled back to AND released so the connection is
    left usable and the partial DDL + version row are both undone.
    """
    savepoint = 'migration_' + hashlib.sha256(migration.id.encode()).hexdigest()[:20]
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for statement in migration.statements:
            conn.execute(statement)
        conn.execute(
            f"INSERT INTO {version_table} (version, applied_at) VALUES (?, ?)",
            (migration.id, _now()),
        )
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
