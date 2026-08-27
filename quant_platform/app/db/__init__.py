"""Platform metadata DB layer.

QRP-P1. The DB is metadata TRUTH. This layer is a thin DB-API dialect
abstraction with a runnable SQLite backend (stdlib ``sqlite3``) and a clear seam
for a PostgreSQL backend (``psycopg``) to be swapped in later.

Design rules:
- No SQLAlchemy / Alembic. Raw DB-API + parameterized SQL only.
- The ``Db`` Protocol is the dialect seam. ``SqliteDb`` implements it today;
  ``PostgresDb`` is a documented adapter that raises if the driver is missing.
- Schema DDL lives in ``schema.py`` and is written PostgreSQL-compatible; the
  SQLite backend runs it with only dialect-neutral constructs.
"""

from __future__ import annotations

from .schema import SCHEMA_DDL, TABLE_NAMES, create_schema
from .sqlite_backend import SqliteDb

__all__ = [
    "SCHEMA_DDL",
    "TABLE_NAMES",
    "create_schema",
    "SqliteDb",
]
