# -*- coding: utf-8 -*-
"""P0-PLAT-015 migration runner tests.

Proves ``apply_migrations`` is honest and idempotent:

* first apply creates the NEW durable orchestration tables (and records every
  version in ``schema_migrations``);
* second apply is a no-op (no error, nothing re-applied);
* a partially-applied state resumes correctly (only the missing versions run);
* a failed migration rolls back atomically (the DDL and the version row land
  together or not at all) on both the raw ``sqlite3`` connection and the
  ``SqliteDb`` backend surface.
"""

from __future__ import annotations

import os
import sqlite3
import sys

# The quant_platform workspace is FLAT-LAYOUT: ``quant_platform/`` *is* the
# package root. Importing ``quant_platform`` requires the *parent* of this
# file's tree — ``.../quant_projects`` — on ``sys.path``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import pytest

from quant_platform.app.db.migrations import MIGRATIONS, Migration, apply_migrations
from quant_platform.app.db.schema import create_schema
from quant_platform.app.db.sqlite_backend import SqliteDb

#: the four durable orchestration tables registered as migrations v1..v4
_DURABLE_TABLES = {
    "workflow_stage_runs",
    "consumed_manifests",
    "batch_fingerprints",
    "stage_idempotency_keys",
}


def _table_names(conn) -> set[str]:
    return {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }


def test_first_apply_creates_new_tables_and_records_versions():
    conn = sqlite3.connect(":memory:")
    applied = apply_migrations(conn)
    assert tuple(applied) == tuple(m.id for m in MIGRATIONS)
    tables = _table_names(conn)
    assert _DURABLE_TABLES <= tables, f"missing durable tables: {_DURABLE_TABLES - tables}"
    rows = conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert [r[0] for r in rows] == [m.id for m in MIGRATIONS]


def test_second_apply_is_noop():
    conn = sqlite3.connect(":memory:")
    assert apply_migrations(conn)
    # Second apply: nothing re-applied, no error.
    assert apply_migrations(conn) == ()
    rows = conn.execute(
        "SELECT count(*) FROM schema_migrations"
    ).fetchone()
    assert rows[0] == len(MIGRATIONS)


def test_partially_applied_state_resumes_correctly():
    conn = sqlite3.connect(":memory:")
    apply_migrations(conn)
    # Simulate a partially-applied state: v2 was recorded but v3/v4 are not
    # yet. Delete them -> a re-run must apply ONLY the missing versions.
    conn.execute(
        "DELETE FROM schema_migrations WHERE version IN ('v3', 'v4')"
    )
    conn.commit()
    applied = apply_migrations(conn)
    assert tuple(applied) == ("v3", "v4")
    rows = conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert [r[0] for r in rows] == [m.id for m in MIGRATIONS]
    # The durable tables still exist and are untouched.
    assert _DURABLE_TABLES <= _table_names(conn)


def test_failed_migration_rolls_back_atomically_on_raw_connection():
    """A failed migration must not leave partial DDL or a recorded version."""
    conn = sqlite3.connect(":memory:")
    original = MIGRATIONS[:]
    import quant_platform.app.db.migrations as mod

    mod.MIGRATIONS = [
        Migration(
            id="bad-1",
            description="two-statement migration, second one is invalid",
            statements=(
                "CREATE TABLE IF NOT EXISTS partial_table (a TEXT)",
                "THIS IS NOT VALID SQL",
            ),
        )
    ]
    try:
        with pytest.raises(Exception):
            apply_migrations(conn)
        tables = _table_names(conn)
        assert "partial_table" not in tables, (
            "partially-applied DDL must be rolled back"
        )
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        assert rows == [], "failed migration must not be recorded as applied"
    finally:
        mod.MIGRATIONS = original


def test_failed_migration_rolls_back_on_sqlite_backend():
    db = SqliteDb(":memory:", create=False)
    original = MIGRATIONS[:]
    import quant_platform.app.db.migrations as mod

    mod.MIGRATIONS = [
        Migration(
            id="bad-1",
            description="backend-surface rollback",
            statements=(
                "CREATE TABLE IF NOT EXISTS partial_table (a TEXT)",
                "THIS IS NOT VALID SQL",
            ),
        )
    ]
    try:
        with pytest.raises(Exception):
            apply_migrations(db)
        tables = {
            str(r["name"])
            for r in db.query(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert "partial_table" not in tables
        assert db.query("SELECT version FROM schema_migrations") == []
    finally:
        mod.MIGRATIONS = original
        db.close()


def test_migrations_apply_after_full_schema():
    """apply_migrations is additive: it works on a DB that already has the full
    platform schema (create_schema), creating only the durable tables + the
    version bookkeeping table."""
    conn = sqlite3.connect(":memory:")
    create_schema(conn)  # full platform schema, without the durable tables
    applied = apply_migrations(conn)
    assert tuple(applied) == tuple(m.id for m in MIGRATIONS)
    tables = _table_names(conn)
    assert _DURABLE_TABLES <= tables
