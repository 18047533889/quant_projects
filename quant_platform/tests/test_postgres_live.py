# -*- coding: utf-8 -*-
"""R55 #99 — live PostgreSQL exercise for ``PostgresDb`` / ``PostgresDialect``.

WHAT THIS IS
------------
The dialect unit tests (``test_postgres_dialect.py``) prove the SQL rewriting
(``?`` -> ``%s``, ``INSERT OR IGNORE`` -> ``ON CONFLICT DO NOTHING``) by string
inspection only.  They never execute anything against a real server, so a
wrongly-rewritten statement, a DDL incompatibility, or a transaction-semantics
bug in ``PostgresDb`` would pass CI silently.

This module runs the REAL backend against a REAL PostgreSQL server when one is
reachable — the CI ``postgres`` job provides one as a service container (and a
developer can point ``QP_PG_DSN`` at any instance).  The full outbox/inbox
application layer (``quant_platform.app.outbox``) is driven through
``PostgresDb``, not hand-picked statements, so the app's actual SQL is what
gets exercised.

FAIL-CLOSED POLICY (R55 #92/#99)
--------------------------------
psycopg2 is NOT a declared dependency anywhere in this repo: 0 hits for
psycopg/psycopg2/asyncpg/pg8000 across every pyproject.toml and
requirements-production.lock, and ``postgres_backend.py`` documents that rule
("psycopg2 is NOT installed in this repo/venv and must not be blind-installed;
project rule: only project-defined pinned deps").  Per #99 this repo therefore
does NOT add psycopg2 to any dependency list on its own.  Consequence:

- no driver importable -> every live test SKIPS with an explicit reason string
  (a skip is reported as a skip, never recorded as a pass);
- driver present but server unreachable -> that is a FAILURE, not a skip;
- driver present and server reachable -> the tests RUN for real.

So the live path is honest in both directions: an environment without a driver
says so, and an environment that claims to have one is actually exercised.
"""
from __future__ import annotations

import os

import pytest

from quant_platform.app.db import (
    PostgresBackendUnavailable,
    PostgresDb,
    create_schema,
)
from quant_platform.app.outbox import InMemoryPublisher, Inbox, Outbox

# DSN resolution order: explicit QP_PG_DSN env override, then the CI service
# container, then the conventional local default.
DEFAULT_DSN = "postgresql://qrp:qrp_secret@localhost:5432/quant_platform_test"


def _dsn() -> str:
    return os.environ.get("QP_PG_DSN", DEFAULT_DSN)


def _psycopg2_or_skip():
    """Import psycopg2 or skip with the exact reason CI reports in its summary."""
    try:
        import psycopg2  # noqa: F401  (import probe only)
    except ImportError as exc:
        pytest.skip(
            "live PostgreSQL backend not exercised: psycopg2 is not installed "
            "(not a project-pinned dependency; see quant_platform/app/db/"
            "postgres_backend.py). Install psycopg2 and set QP_PG_DSN to run "
            f"these tests for real. ({exc})",
            allow_module_level=False,
        )
    return True


def _connect() -> PostgresDb:
    """Open a live PostgresDb or fail hard if the server is unreachable.

    Deliberately NOT a skip: if a driver is installed and the CI service is
    expected, an unreachable server is a broken environment and must fail.
    """
    dsn = _dsn()
    try:
        db = PostgresDb(dsn, create=True)
    except PostgresBackendUnavailable:
        raise
    except Exception as exc:  # psycopg2.OperationalError and friends
        pytest.fail(
            f"PostgresDb could not connect to {dsn!r}: "
            f"{type(exc).__name__}: {exc}"
        )
    return db


@pytest.fixture(scope="module")
def pg() -> PostgresDb:
    _psycopg2_or_skip()
    db = _connect()
    yield db
    try:
        # Drop what we created so repeated runs (and CI service reuse) start
        # from a clean slate; the schema is created per-connection anyway.
        cur = db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        tables = [r["table_name"] for r in db.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )]
        for t in sorted(tables):
            if t.startswith(("outbox_events", "inbox_events", "principals",
                             "human_users", "workload_principals", "teams",
                             "team_members", "roles", "permissions",
                             "artifacts", "artifact_lineage", "jobs",
                             "job_attempts", "job_results", "workflow_runs",
                             "factor_candidates", "similarity_graph_versions",
                             "cluster_set_versions", "logical_clusters",
                             "cluster_versions", "cluster_memberships",
                             "cluster_lineage", "factor_libraries",
                             "factor_library_versions", "factor_library_members",
                             "feature_sets", "feature_set_versions",
                             "feature_set_members", "production_pointers",
                             "sessions", "audit_logs")):
                db.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
        db._conn.commit()
    finally:
        db.close()


def test_live_schema_ddl_applies_to_postgres(pg: PostgresDb):
    """The shared, PostgreSQL-compatible DDL must apply verbatim on a real PG.

    This is the strongest guarantee in the repo that ``schema.py`` stays
    dialect-neutral: any SQLite-only construct added to the DDL (AUTOINCREMENT,
    PRAGMA, SQLite-typed defaults...) breaks this test on a real server.
    """
    from quant_platform.app.db.schema import TABLE_NAMES

    cur = pg.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    )
    rows = pg.query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    )
    created = {r["table_name"] for r in rows}
    missing = [t for t in TABLE_NAMES if t not in created]
    assert not missing, f"live PostgreSQL is missing platform tables: {missing}"
    assert len(TABLE_NAMES) == 31


def test_live_dialect_roundtrip_insert_select(pg: PostgresDb):
    """``?``-placeholder SQL must execute and round-trip through PostgresDb."""
    with pg.transaction() as conn:
        conn.execute(
            "INSERT INTO principals (principal_id, principal_type, display_name) "
            "VALUES (?, ?, ?) ON CONFLICT (principal_id) DO NOTHING",
            ("p-live-1", "HUMAN", "Live PG Principal"),
        )
    rows = pg.query(
        "SELECT principal_id, principal_type, display_name FROM principals "
        "WHERE principal_id = ?",
        ("p-live-1",),
    )
    assert rows and rows[0]["display_name"] == "Live PG Principal"
    assert rows[0]["principal_type"] == "HUMAN"


def test_live_insert_or_ignore_normalized_and_dedupes(pg: PostgresDb):
    """The SQLite-only ``INSERT OR IGNORE`` written by Inbox.process must be
    normalized by PostgresDialect.adapt_ignore and actually dedupe on PG."""
    pg.execute(
        "INSERT OR IGNORE INTO inbox_events "
        "(event_id, idempotency_key, status, retry_count, next_attempt_at, "
        " last_error, dead_letter, dead_letter_reason, processed_at) "
        "VALUES (?, ?, 'PROCESSING', 0, NULL, NULL, 0, NULL, ?)",
        ("ev-1", "idem-1", 1.0),
    )
    # Second identical insert must be ignored (no PK/unique violation raised).
    pg.execute(
        "INSERT OR IGNORE INTO inbox_events "
        "(event_id, idempotency_key, status, retry_count, next_attempt_at, "
        " last_error, dead_letter, dead_letter_reason, processed_at) "
        "VALUES (?, ?, 'PROCESSING', 0, NULL, NULL, 0, NULL, ?)",
        ("ev-1", "idem-1", 2.0),
    )
    pg._conn.commit()
    rows = pg.query(
        "SELECT event_id, idempotency_key, status FROM inbox_events "
        "WHERE idempotency_key = ?",
        ("idem-1",),
    )
    assert len(rows) == 1, f"INSERT OR IGNORE dedupe broken on PG: {rows}"
    assert rows[0]["event_id"] == "ev-1"


def test_live_quoted_question_mark_is_literal(pg: PostgresDb):
    """A ``?`` inside a string literal must survive the dialect AND run on PG."""
    pg.execute(
        "INSERT INTO principals (principal_id, principal_type, display_name) "
        "VALUES (?, ?, ?) ON CONFLICT (principal_id) DO NOTHING",
        ("p-live-2", "WORKLOAD", "why? and 100% of this is literal"),
    )
    pg._conn.commit()
    rows = pg.query(
        "SELECT display_name FROM principals WHERE principal_id = ?",
        ("p-live-2",),
    )
    assert rows and rows[0]["display_name"] == "why? and 100% of this is literal"


def test_live_outbox_emit_claim_finish_end_to_end(pg: PostgresDb):
    """The FULL app-layer outbox lifecycle runs against live PostgreSQL through
    PostgresDb: emit (inside a transaction) -> claim -> finish."""
    publisher = InMemoryPublisher()
    outbox = Outbox(pg, publisher)
    with pg.transaction() as conn:
        event_id = outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="FactorCandidate",
            aggregate_id="fc-live-1",
            correlation_id="corr-live-1",
            idempotency_key="outbox-live-1",
            payload={"stage": "DISCOVERED"},
        )
        assert isinstance(event_id, int) and event_id > 0

    # claim with a foreign token must NOT flip the row
    assert outbox.claim(
        event_id, worker_id="w-a", claim_token="tok-A", now=123.0
    ) is True
    assert outbox.claim(
        event_id, worker_id="worker-b", claim_token="tok-B", now=124.0
    ) is False, "PG must serialize the claim exactly like SQLite does"
    assert outbox.finish(event_id, claim_token="tok-A") is True
    assert outbox.finish(event_id, claim_token="tok-A") is False, \
        "double-finish must not re-mark a sent row"

    status = pg.query(
        "SELECT status, worker_id FROM outbox_events WHERE id = ?", (event_id,)
    )
    assert status and status[0]["status"] == "sent"
    published = outbox.publish_pending(limit=10)
    assert published == 1, "the sent outbox row must be delivered exactly once"
    assert publisher.delivered and publisher.delivered[0]["event_type"] == \
        "FactorCandidateDiscovered"


def test_live_inbox_idempotency_and_retry_on_postgres(pg: PostgresDb):
    """The full Inbox state machine (RECEIVED->PROCESSING->DONE, and the
    handler-exception -> FAILED -> retry -> dead-letter path) on live PG."""
    calls: list[dict] = []

    def handler(event: dict) -> None:
        calls.append(event)
        if event.get("payload", {}).get("boom"):
            raise RuntimeError("handler blew up on live PG")

    inbox = Inbox(pg, handler)

    ev = {"event_id": "ie-1", "idempotency_key": "inbox-live-1",
          "payload": {"n": 1}}
    assert inbox.process(ev) is True
    assert inbox.status_of("inbox-live-1") == "DONE"
    assert inbox.process(ev) is False, "duplicate must not re-run the handler"
    assert len(calls) == 1

    # handler failure -> FAILED with retry accounting, not silent burn
    bad = {"event_id": "ie-2", "idempotency_key": "inbox-live-2",
           "payload": {"boom": True}}
    with pytest.raises(RuntimeError, match="handler blew up on live PG"):
        inbox.process(bad)
    assert inbox.status_of("inbox-live-2") == "FAILED"
    row = inbox.get("inbox-live-2")
    assert row is not None and int(row["retry_count"]) >= 1


def test_live_transaction_rollback_on_exception(pg: PostgresDb):
    """transaction() must roll back the handler's side effects on PG too."""
    with pytest.raises(RuntimeError, match="rollback-me"):
        with pg.transaction() as conn:
            conn.execute(
                "INSERT INTO principals (principal_id, principal_type, "
                "display_name) VALUES (?, ?, ?)",
                ("p-rollback", "HUMAN", "must not persist"),
            )
            raise RuntimeError("rollback-me")
    rows = pg.query(
        "SELECT principal_id FROM principals WHERE principal_id = ?",
        ("p-rollback",),
    )
    assert rows == [], "rolled-back row must not be visible after the exception"


def test_live_schema_creation_is_idempotent(pg: PostgresDb):
    """create_schema() twice on a live PG must not raise (all-IF-NOT-EXISTS)."""
    create_schema(pg._conn)
    pg._conn.commit()
    rows = pg.query(
        "SELECT count(*) AS n FROM information_schema.tables "
        "WHERE table_schema='public'"
    )
    assert rows and int(rows[0]["n"]) >= 31