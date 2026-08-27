"""DB layer tests — schema creation, outbox same-transaction, inbox idempotency."""

from __future__ import annotations

import pytest

from quant_platform.app.db.schema import SCHEMA_DDL, TABLE_NAMES, create_schema
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.outbox import InMemoryPublisher, Inbox, Outbox


def test_schema_creates_all_tables():
    db = SqliteDb(":memory:")
    rows = db.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    created = {r["name"] for r in rows}
    for table in TABLE_NAMES:
        assert table in created, f"missing table {table}"
    assert len(TABLE_NAMES) == len(SCHEMA_DDL)


def test_schema_idempotent():
    db = SqliteDb(":memory:")
    create_schema(db._conn)  # second apply must not raise
    db._conn.commit()


def test_outbox_same_transaction():
    db = SqliteDb(":memory:")
    publisher = InMemoryPublisher()
    outbox = Outbox(db, publisher)

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO principals (principal_id, principal_type, display_name) VALUES (?, ?, ?)",
            ("p1", "HUMAN", "Alice"),
        )
        outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
            payload={"candidate_id": "cand-1"},
        )

    # Event row committed atomically with the state update.
    events = db.query("SELECT * FROM outbox_events")
    assert len(events) == 1
    assert events[0]["status"] == "pending"
    principals = db.query("SELECT * FROM principals")
    assert len(principals) == 1


def test_outbox_rollback_on_exception():
    db = SqliteDb(":memory:")
    publisher = InMemoryPublisher()
    outbox = Outbox(db, publisher)

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO principals (principal_id, principal_type, display_name) VALUES (?, ?, ?)",
                ("p1", "HUMAN", "Alice"),
            )
            outbox.emit(
                event_type="FactorCandidateDiscovered",
                aggregate_type="factor_candidate",
                aggregate_id="cand-1",
                correlation_id="corr-1",
                idempotency_key="ik-1",
            )
            raise RuntimeError("boom")

    # Both the state update and the outbox row must be rolled back together.
    assert db.query("SELECT * FROM principals") == []
    assert db.query("SELECT * FROM outbox_events") == []


def test_outbox_publish_pending_marks_sent():
    db = SqliteDb(":memory:")
    publisher = InMemoryPublisher()
    outbox = Outbox(db, publisher)

    with db.transaction() as conn:
        outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key="ik-1",
        )

    published = outbox.publish_pending()
    assert published == 1
    assert len(publisher.delivered) == 1
    assert db.query("SELECT status FROM outbox_events")[0]["status"] == "sent"


def test_inbox_idempotency():
    db = SqliteDb(":memory:")
    handled = []
    inbox = Inbox(db, lambda ev: handled.append(ev))

    event = {
        "event_id": "evt-1",
        "idempotency_key": "ik-1",
        "event_type": "FactorCandidateDiscovered",
    }
    assert inbox.process(event) is True
    assert inbox.process(event) is False  # duplicate ignored
    assert len(handled) == 1
    assert len(db.query("SELECT * FROM inbox_events")) == 1
