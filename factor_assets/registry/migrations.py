"""Ordered, checksummed SQLite schema migrations."""
from __future__ import annotations
import hashlib
import sqlite3
from factor_assets.errors import FactorAssetsError, SchemaVersionError

SCHEMA_VERSION = 1
_MIGRATIONS = {1: """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets (factor_id TEXT PRIMARY KEY, canonical_hash TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS lifecycle_events (id INTEGER PRIMARY KEY AUTOINCREMENT, factor_id TEXT NOT NULL REFERENCES assets(factor_id), revision INTEGER NOT NULL, decision_id TEXT, payload TEXT NOT NULL, UNIQUE(factor_id, decision_id));
"""}

def migration_checksum(version: int) -> str:
    try: body = _MIGRATIONS[version]
    except KeyError: raise SchemaVersionError(f"unknown schema migration {version}")
    return hashlib.sha256(body.encode()).hexdigest()

def _validate_schema_v1(conn: sqlite3.Connection) -> None:
    """Verify the physical schema, including constraints, not only its ledger."""
    expected = {
        "schema_meta": {
            "key": ("TEXT", False, 1), "value": ("TEXT", True, 0),
        },
        "assets": {
            "factor_id": ("TEXT", False, 1), "canonical_hash": ("TEXT", True, 0),
            "payload": ("TEXT", True, 0), "revision": ("INTEGER", True, 0),
        },
        "lifecycle_events": {
            "id": ("INTEGER", False, 1), "factor_id": ("TEXT", True, 0),
            "revision": ("INTEGER", True, 0), "decision_id": ("TEXT", False, 0),
            "payload": ("TEXT", True, 0),
        },
        "schema_migrations": {
            "version": ("INTEGER", False, 1), "checksum": ("TEXT", True, 0),
        },
    }
    for table, expected_columns in expected.items():
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        actual = {row[1]: (row[2].upper(), bool(row[3]), row[5]) for row in rows}
        if actual != expected_columns:
            raise SchemaVersionError(f"schema table {table} has unexpected columns")

    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='lifecycle_events'"
    ).fetchone()[0]
    if "AUTOINCREMENT" not in sql.upper():
        raise SchemaVersionError("schema table lifecycle_events is missing AUTOINCREMENT")

    unique_columns = {
        "assets": {"canonical_hash"},
        "lifecycle_events": {"factor_id", "decision_id"},
    }
    for table, required in unique_columns.items():
        found = False
        for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
            if not index[2] or len(index) > 4 and index[4]:
                continue
            columns = {
                row[2] for row in conn.execute(f"PRAGMA index_info({index[1]!r})").fetchall()
            }
            if columns == required:
                found = True
                break
        if not found:
            raise SchemaVersionError(f"schema table {table} is missing required unique constraint")

    foreign_keys = conn.execute("PRAGMA foreign_key_list(lifecycle_events)").fetchall()
    if not any(row[2] == "assets" and row[3] == "factor_id" and row[4] == "factor_id" for row in foreign_keys):
        raise SchemaVersionError("schema table lifecycle_events is missing assets foreign key")


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        migration_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        if migration_table is None:
            existing = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name IN "
                "('schema_meta', 'assets', 'lifecycle_events') LIMIT 1"
            ).fetchone()
            if existing is not None:
                raise SchemaVersionError("schema_migrations table is missing")
            conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, checksum TEXT NOT NULL)")
        try:
            rows = conn.execute("SELECT version, checksum FROM schema_migrations ORDER BY version").fetchall()
        except sqlite3.DatabaseError as exc:
            raise SchemaVersionError("schema_migrations table is malformed") from exc
        known = {v: c for v, c in rows}
        if any(v not in _MIGRATIONS or c != migration_checksum(v) for v, c in known.items()):
            raise SchemaVersionError("schema migration drift or unknown version")
        if known and max(known) > SCHEMA_VERSION: raise SchemaVersionError("database schema is newer than this code")
        for version in range(1, SCHEMA_VERSION + 1):
            if version in known: continue
            for statement in _MIGRATIONS[version].split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, migration_checksum(version)))
        _validate_schema_v1(conn)
        try:
            conn.execute("INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('schema_version',?)", (str(SCHEMA_VERSION),))
        except sqlite3.DatabaseError as exc:
            raise SchemaVersionError("schema_meta table is malformed") from exc
        conn.commit()
    except (sqlite3.Error, OSError, ValueError, TypeError, FactorAssetsError):
        # SchemaVersionError (a ContractError) and other typed governance
        # errors must roll back and propagate by type.
        conn.rollback()
        raise
