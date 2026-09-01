# -*- coding: utf-8 -*-
"""Production composition: Outbox + Inbox + durable DB + schema (P0-PLAT-005).

The transactional outbox pieces in ``outbox.py`` are backend-agnostic — they
accept any ``Db``-Protocol object.  This module wires the full PRODUCTION
composition so a process can get a working outbox/inbox with one call:

    comp = production_composition(dsn=":memory:", publisher=publisher)
    # outbox.emit(...) / inbox.process(...) / outbox.publish_pending()
    comp.close()

``production_composition``:

- applies the full schema (``create_schema``, all-``IF NOT EXISTS``);
- applies the P0-PLAT-015 migrations (durable orchestration tables);
- constructs a durable ``SqliteDb`` (default) or ``PostgresDb`` (when a PG DSN
  is passed) and the ``Outbox`` / ``Inbox`` over it;
- returns a :class:`Composition` exposing ``outbox`` / ``inbox`` / ``db`` and a
  ``close()`` that commits and cleans up.

The same composition runs on SQLite (runnable here) and against the
``PostgresTransaction`` adapter (psycopg2 missing -> ``PostgresBackendUnavailable``
per the repo's fail-closed convention, exercised by ``test_postgres_live.py``).
"""

from __future__ import annotations

from typing import Any

from .db import (
    PostgresBackendUnavailable,
    PostgresDb,
    SqliteDb,
    create_schema,
)
from .db.migrations import apply_migrations
from .outbox import Inbox, Outbox

__all__ = [
    "Composition",
    "production_composition",
]


class Composition:
    """One composed outbox/inbox stack over a durable DB.

    ``outbox`` / ``inbox`` are the transactional writers/consumers; ``db`` is
    the underlying backend; ``publisher`` is the pluggable delivery seam.
    ``close()`` commits any open writes and closes the backend.
    """

    def __init__(
        self,
        *,
        db: Any,
        outbox: Outbox,
        inbox: Inbox,
        publisher: Any,
    ) -> None:
        self.db = db
        self.outbox = outbox
        self.inbox = inbox
        self.publisher = publisher

    def close(self) -> None:
        """Commit + clean up the backend. Safe to call once; idempotent."""
        try:
            self.db.commit()
        except AttributeError:
            # PostgresDb has no commit(); its transaction() adapter owns commits.
            pass
        self.db.close()


def _build_db(dsn: str | None) -> Any:
    """Construct the durable backend: SqliteDb for ``:memory:``/file, PostgresDb
    for a PostgreSQL DSN (fail-closed when psycopg2 is missing)."""
    if dsn is None:
        return SqliteDb(":memory:")
    if dsn == ":memory:" or dsn.startswith("sqlite:"):
        path = ":memory:" if dsn == ":memory:" else dsn[len("sqlite:"):]
        return SqliteDb(path)
    # A non-sqlite DSN is a PostgreSQL DSN.  ``PostgresDb(dsn, create=True)``
    # applies the schema itself; ``PostgresBackendUnavailable`` surfaces when
    # psycopg2 is not installed (the repo's fail-closed convention).
    return PostgresDb(dsn, create=True)


def production_composition(
    dsn: str | None = None,
    *,
    db: Any | None = None,
    publisher: Any | None = None,
    handler: Any | None = None,
    max_retries: int = 3,
) -> Composition:
    """Wire the full production outbox/inbox composition.

    Parameters
    ----------
    dsn : SQLite path (``":memory:"`` default) or a PostgreSQL DSN. Ignored when
        ``db`` is passed explicitly.
    db : an already-constructed ``SqliteDb`` / ``PostgresDb``; when provided the
        composition uses it as-is (the caller owns schema + lifecycle).
    publisher : the outbox ``Publisher`` seam (an ``InMemoryPublisher`` is
        created when omitted — use a real bus publisher in production).
    handler : the inbox consumer callable; when omitted the Inbox is built with
        a no-op handler (production callers MUST supply one).
    max_retries : inbox retry budget before dead-lettering (default 3).

    Returns a :class:`Composition` with ``outbox`` / ``inbox`` / ``db``.  The
    schema + P0-PLAT-015 migrations are applied on first use.
    """
    backend = db if db is not None else _build_db(dsn)

    # Apply schema + migrations idempotently.  ``create_schema`` handles the
    # raw sqlite3.Connection AND the PostgresDb cursor path; ``apply_migrations``
    # handles both surfaces (see ``migrations.py``).
    if hasattr(backend, "_conn"):
        create_schema(backend._conn)
        backend._conn.commit()
        apply_migrations(backend._conn)
    else:
        create_schema(backend)
        apply_migrations(backend)

    from .outbox import InMemoryPublisher

    pub = publisher if publisher is not None else InMemoryPublisher()
    outbox = Outbox(backend, pub)
    inbox = Inbox(
        backend,
        handler if handler is not None else (lambda event: None),
        max_retries=max_retries,
    )
    return Composition(db=backend, outbox=outbox, inbox=inbox, publisher=pub)
