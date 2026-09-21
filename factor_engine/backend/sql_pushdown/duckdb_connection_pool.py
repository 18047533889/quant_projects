# -*- coding: utf-8 -*-
"""R58: thread-safe DuckDB connection pool + per-connection prepared-statement reuse.

Why this exists
---------------
Several SQL-pushdown code paths called ``duckdb.connect()`` on *every* invocation:

* ``factor_engine/backend/sql_pushdown/sql_registry.py`` (capability probe)
* ``factor_engine/backend/sql_pushdown/duckdb_capabilities.py`` (capability probe)
* ``factor_engine/scripts/calibrate_backend_costs.py`` (per-call backend cost)
* ``data_access/core/engine.py:_execute_scoped_sql_arrow_locked`` (every scoped
  SQL pushdown — the real engine hot path, owned by the shared ``data_access``
  package and out of scope for this patch).

A fresh ``:memory:`` DuckDB connection pays a fixed ~10–18 ms catalog/init cost
each time.  That fixed cost dominated ``duckdb_sql`` single-call latency
(measured ~18 ms/call → ~9000 ms/M vs ~600 ms/M for the polars backends).

Design
------
* A process-wide pool of in-memory DuckDB connections.  ``acquire``/``release``
  are guarded by a ``Condition`` so a connection is handed to exactly one caller
  at a time (DuckDB connections are **not** thread-safe).
* Every pooled connection owns its **own** ``DuckDBPreparedStatementCache``
  instance.  A prepared plan (relation) is therefore *never* shared across
  connections — this is what prevents batch multi-factor landing from aliasing
  results between factors / parameter domains / tables.  The cache key is the
  normalised SQL text (which embeds the table identity and the query shape); the
  per-connection scoping provides the "which connection" dimension.
* The prepared-statement cache is reused via the existing
  ``duckdb_prepared_statement_cache`` API — no parallel implementation.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from factor_engine.runtime.multibackend.duckdb_prepared_statement_cache import (
    DuckDBPreparedStatementCache,
)


class DuckDBConnectionHandle:
    """A checked-out connection plus its private prepared-statement cache."""

    def __init__(self, pool: "DuckDBConnectionPool", conn: Any, cache: Any) -> None:
        self._pool = pool
        self._conn = conn
        self._cache = cache
        self._released = False

    def __enter__(self) -> "DuckDBConnectionHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    @property
    def connection(self) -> Any:
        return self._conn

    def register(self, name: str, obj: Any) -> None:
        self._conn.register(name, obj)

    def unregister(self, name: str) -> None:
        try:
            self._conn.unregister(name)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        """Execute ``sql`` through this connection's prepared-statement cache."""
        return self._cache.execute(self._conn, sql, params)

    def fetch_arrow_table(self, sql: str, params: list[Any] | None = None) -> Any:
        result = self.execute(sql, params)
        if hasattr(result, "to_arrow_table"):
            return result.to_arrow_table()
        return result  # already an arrow table

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._pool._return(self._conn, self._cache)


class DuckDBConnectionPool:
    """Bounded, thread-safe pool of ``:memory:`` DuckDB connections."""

    def __init__(self, max_size: int = 16, conn_factory=None) -> None:
        self._max_size = int(max_size) if max_size and max_size > 0 else 16
        self._factory = conn_factory or self._default_factory
        self._free: list[tuple[Any, Any]] = []
        self._created = 0
        self._cond = threading.Condition(threading.RLock())

    @staticmethod
    def _default_factory() -> Any:
        import duckdb

        return duckdb.connect(":memory:")

    def acquire(self, timeout: float = 60.0) -> DuckDBConnectionHandle:
        deadline = time.monotonic() + float(timeout)
        with self._cond:
            while True:
                if self._free:
                    conn, cache = self._free.pop()
                    return DuckDBConnectionHandle(self, conn, cache)
                if self._created < self._max_size:
                    conn = self._factory()
                    cache = DuckDBPreparedStatementCache()
                    self._created += 1
                    return DuckDBConnectionHandle(self, conn, cache)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("DuckDBConnectionPool exhausted")
                self._cond.wait(remaining)

    def _return(self, conn: Any, cache: Any) -> None:
        with self._cond:
            self._free.append((conn, cache))
            self._cond.notify()

    def close_all(self) -> None:
        with self._cond:
            for conn, _ in self._free:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._free.clear()
            self._created = 0


_shared_pool: DuckDBConnectionPool | None = None
_shared_pool_lock = threading.Lock()


def get_shared_duckdb_pool() -> DuckDBConnectionPool:
    """Return the process-wide DuckDB connection pool (thread-safe singleton)."""
    global _shared_pool
    if _shared_pool is None:
        with _shared_pool_lock:
            if _shared_pool is None:
                _shared_pool = DuckDBConnectionPool()
    return _shared_pool
