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

R55 #92: the shared SQL is written in SQLite's ``?`` placeholder style. A live
PostgreSQL backend must therefore normalize ``?`` to ``%s`` before executing —
``SqlDialect`` (below) is the parameterization seam.  Every ``execute`` /
``query`` on this backend runs SQL through ``_adapt`` so the same
``outbox.py`` / ``inbox.py`` statements work against PG unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence


class PostgresBackendUnavailable(RuntimeError):
    """Raised when psycopg2 is not installed / cannot connect."""


class PostgresDialect:
    """SQL parameterization adapter: SQLite ``?`` -> psycopg2 ``%s``.

    The application layer (outbox/inbox/jobs) writes ``?`` placeholders which
    psycopg2 does not accept.  This adapter rewrites a statement's ``?`` tokens
    (outside string literals) to ``%s`` so the SAME SQL runs on both backends.
    """

    @staticmethod
    def adapt(sql: str) -> str:
        out: list[str] = []
        in_quote: str | None = None
        i = 0
        n = len(sql)
        while i < n:
            ch = sql[i]
            if in_quote:
                out.append(ch)
                if ch == in_quote:
                    # Closing quote; doubled quotes ('' / "") stay literal.
                    if i + 1 < n and sql[i + 1] == in_quote:
                        out.append(sql[i + 1])
                        i += 1
                    else:
                        in_quote = None
            elif ch in ("'", '"'):
                in_quote = ch
                out.append(ch)
            elif ch == "?":
                out.append("%s")
            else:
                out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def adapt_ignore(sql: str) -> str:
        """Normalize SQLite-only ``INSERT OR IGNORE`` to PG ``INSERT ... ON
        CONFLICT DO NOTHING`` (R55 #92).  Only touches the leading keyword and
        leaves the rest (placeholders, columns, VALUES) untouched."""
        stripped = sql.lstrip()
        prefix = "INSERT OR IGNORE"
        if stripped.upper().startswith(prefix):
            indent = sql[: len(sql) - len(stripped)]
            rest = stripped[len(prefix):]
            # Rest begins right after the keyword (a space).  Build
            # INSERT + <rest> + ON CONFLICT DO NOTHING.
            return indent + "INSERT" + rest + " ON CONFLICT DO NOTHING"
        return sql


class PostgresDb:
    """PostgreSQL backend implementing ``Db`` (adapter over psycopg2).

    Every SQL statement is passed through ``PostgresDialect.adapt`` before
    execution, so ``?``-style statements from the shared app layer run against
    PG unchanged (R55 #92).  ``%s`` already present (from a dialect-tuned
    caller) is left untouched.
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
        cur.execute(PostgresDialect.adapt_ignore(PostgresDialect.adapt(sql)), params)
        return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(PostgresDialect.adapt_ignore(PostgresDialect.adapt(sql)), params)
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
