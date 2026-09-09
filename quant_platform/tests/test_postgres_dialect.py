# -*- coding: utf-8 -*-
"""R55 #92 — Postgres dialect: SQLite `?` -> psycopg2 `%s` + INSERT OR IGNORE
normalization, without requiring a live PG server."""
from __future__ import annotations

import pytest

from quant_platform.app.db import PostgresDialect, PostgresBackendUnavailable, PostgresDb


def test_placeholder_adapt_rewrites_q_to_percent_s():
    sql = "UPDATE outbox_events SET status='claimed', worker_id=? WHERE id=? AND status IN ('pending','claimed')"
    out = PostgresDialect.adapt(sql)
    assert "?" not in out
    assert out.count("%s") == 2
    assert "worker_id=%s" in out


def test_placeholder_adapt_preserves_quoted_question_marks():
    sql = "UPDATE t SET a='?', b=? WHERE c='has ? mark'"
    out = PostgresDialect.adapt(sql)
    assert out.count("%s") == 1
    assert "a='?'" in out
    assert "'has ? mark'" in out


def test_placeholder_adapt_handles_doubled_quotes():
    sql = "INSERT INTO t (v) VALUES ('it''s ? here')"
    out = PostgresDialect.adapt(sql)
    assert out.count("%s") == 0
    assert "'it''s ? here'" in out


def test_ignore_clause_normalized_for_postgres():
    sql = "INSERT OR IGNORE INTO inbox_events (event_id, status) VALUES (?, 'PROCESSING')"
    out = PostgresDialect.adapt_ignore(PostgresDialect.adapt(sql))
    assert out.startswith("INSERT INTO inbox_events")
    assert "ON CONFLICT DO NOTHING" in out
    assert "?" not in out


def test_ignore_clause_passthrough_for_non_insert():
    sql = "SELECT * FROM outbox_events WHERE status='pending'"
    assert PostgresDialect.adapt_ignore(sql) == sql


def test_postgres_db_without_driver_fails_closed(monkeypatch):
    """No psycopg2 in this venv -> clear PostgresBackendUnavailable, never a
    silent NOT_IMPLEMENTED passthrough (R55 #92)."""
    import sys
    monkeypatch.setitem(sys.modules, 'psycopg2', None)
    with pytest.raises(PostgresBackendUnavailable):
        PostgresDb("postgresql://u:p@localhost:5432/db", create=False)


def test_outbox_inbox_app_sql_adapts_cleanly():
    """The exact SQL the app layer sends must be PG-executable after adapt."""
    from quant_platform.app.outbox import Outbox, Inbox  # noqa: F401
    import inspect
    import re

    # Outbox.claim / Inbox.get / Inbox.mark_failed / Inbox.process bodies
    # contain `?` placeholders; every one must survive the dialect.
    for fn in (Outbox.claim, Outbox.finish, Outbox.fail, Outbox.publish_pending,
               Inbox.get, Inbox.status_of, Inbox.mark_failed, Inbox.process):
        src = inspect.getsource(fn)
        for m in re.finditer(r'"[^"\n]*\?"[^"\n]*', src):
            sql = m.group(0).strip('"')
            adapted = PostgresDialect.adapt_ignore(PostgresDialect.adapt(sql))
            assert "?" not in adapted, f"unadapted ? in {fn.__name__}: {sql}"
            assert "%s" in adapted or "VALUES" not in sql
