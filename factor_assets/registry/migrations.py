"""Ordered, checksummed SQLite schema migrations."""
from __future__ import annotations
import hashlib
import sqlite3
from factor_assets.errors import SchemaVersionError

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

def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, checksum TEXT NOT NULL)")
    rows = conn.execute("SELECT version, checksum FROM schema_migrations ORDER BY version").fetchall()
    known = {v: c for v, c in rows}
    if any(v not in _MIGRATIONS or c != migration_checksum(v) for v, c in known.items()):
        raise SchemaVersionError("schema migration drift or unknown version")
    if known and max(known) > SCHEMA_VERSION: raise SchemaVersionError("database schema is newer than this code")
    for version in range(1, SCHEMA_VERSION + 1):
        if version in known: continue
        conn.executescript(_MIGRATIONS[version])
        conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, migration_checksum(version)))
    conn.execute("INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('schema_version',?)", (str(SCHEMA_VERSION),))
