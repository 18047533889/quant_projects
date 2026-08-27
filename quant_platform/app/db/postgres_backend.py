"""PostgreSQL metadata backend (adapter seam).

QRP-P1. Implements the same ``Db`` Protocol as ``SqliteDb`` but against
``psycopg2``. **psycopg2 is NOT installed in this repo/venv** and must not be
blind-installed (project rule: only project-defined pinned deps; none define
psycopg2). This module therefore:

- Imports ``psycopg2`` lazily inside the constructor.
- Raises a clear ``PostgresBackendUnavailable`` if the driver is missing.
- Documents the dialect differences a live PG backend must reconcile.

The schema DDL in ``schema.py`` is written PostgreSQL-compatible (``TEXT`` for
UUID/JSON, portable ``BOOLEAN``/``TIMESTAMP``/``REAL``). The only PG-specific
tuning a live deployment should apply is switching the surrogate
``INTEGER PRIMARY KEY`` columns (``outbox_events.id``, ``audit_logs.id``,
``job_attempts.attempt_id``) to ``BIGSERIAL`` / ``GENERATED ALWAYS AS IDENTITY``
— noted here, not in the shared DDL, so the same DDL stays SQLite-runnable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .schema import create_schema


class PostgresBackendUnavailable(RuntimeError):
    """Raised when psycopg2 is not installed / cannot connect."""


class PostgresDb:
    """PostgreSQL backend implementing ``Db`` (adapter over psycopg2).

    NOT_IMPLEMENTED for live use in this environment: psycopg2 is not installed.
    The class is fully wired so that once a PG server + driver are available it
    can be used without changing the ``Db`` Protocol or the schema.
    """

    def __init__(self, dsn: str, create: bool = True) -> None:
        try:
            import psycopg2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise PostgresBackendUnavailable(
                "psycopg2 is not installed. Install a project-pinned psycopg2 "
                "and point PostgresDb at a live PG server to enable this backend."
            ) from exc
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        if create:
            create_schema(self._conn)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
