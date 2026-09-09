"""SQLite metadata backend.

QRP-P1. Implements the ``Db`` dialect Protocol over stdlib ``sqlite3``. This is
the runnable default backend. A PostgreSQL backend (``postgres_backend.py``)
implements the same Protocol.

The ``Db`` Protocol is the dialect seam:
- ``execute(sql, params)`` — run a single statement, return the cursor.
- ``query(sql, params)`` — run a SELECT, return a list of rows (as dicts).
- ``transaction()`` — context manager yielding a connection whose writes commit
  on success and roll back on exception. Used for the transactional outbox.
- ``row_factory`` — how rows are shaped (dict by default).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

from .schema import create_schema


@runtime_checkable
class Db(Protocol):
    """Dialect-neutral metadata DB seam."""

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """Run a single statement; return the cursor."""
        ...

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run a SELECT; return rows as dicts."""
        ...

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Context manager yielding a connection; commits on success, rolls back
        on exception."""
        ...


class SqliteDb:
    """SQLite backend implementing ``Db``.

    ``path`` may be ``":memory:"`` (default) or a file path. Foreign keys are
    enabled per-connection. The schema is created lazily on first use unless
    ``create=True`` is passed at construction.
    """

    def __init__(self, path: str = ":memory:", create: bool = True) -> None:
        self._path = path
        # check_same_thread=False: FastAPI/TestClient may serve requests on a
        # different thread than the one that created the connection. The
        # connection is guarded by the transaction context manager.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if create:
            create_schema(self._conn)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = self._conn.execute(sql, params)
        return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        try:
            yield self._conn
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
