# -*- coding: utf-8 -*-
"""P0-PLAT-005 production composition tests.

``production_composition`` wires Outbox + Inbox + a durable backend + schema +
migrations.  These tests run against the SQLite backend (runnable here —
psycopg2 is NOT installed, live PG tests keep skipping honestly).

* emit -> claim -> finish -> publish end to end;
* inbox processing is idempotent (duplicate delivery is a no-op);
* the dead-letter path: handler failure past max retries lands in the inbox
  dead-letter state, exported by ``Inbox.dead_lettered()``.
"""

from __future__ import annotations

import os
import sys

import pytest

# The quant_platform workspace is FLAT-LAYOUT: ``quant_platform/`` *is* the
# package root. Importing ``quant_platform`` requires the *parent* of this
# file's tree — ``.../quant_projects`` — on ``sys.path``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from quant_platform.app.composition import Composition, production_composition
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.outbox import (
    InMemoryPublisher,
    Inbox,
    InboxEventNotFound,
    Outbox,
)


def _event(key: str = "ik-1", event_id: str = "e1") -> dict:
    return {
        "event_id": event_id,
        "idempotency_key": key,
        "event_type": "FactorCandidateDiscovered",
        "aggregate_type": "factor_candidate",
        "aggregate_id": "cand-1",
    }


def _emit(comp: Composition, *, key: str = "outbox-1") -> int:
    with comp.db.transaction() as conn:
        return comp.outbox.emit(
            event_type="FactorCandidateDiscovered",
            aggregate_type="factor_candidate",
            aggregate_id="cand-1",
            correlation_id="corr-1",
            idempotency_key=key,
            payload={"candidate_id": "cand-1"},
        )


def test_composition_emit_claim_finish_publish():
    """The full outbox lifecycle runs over the composed SQLite stack."""
    publisher = InMemoryPublisher()
    comp = production_composition(":memory:", publisher=publisher)
    try:
        event_id = _emit(comp)
        assert isinstance(event_id, int) and event_id > 0

        # claim -> finish -> the row is sent; publish_pending finds no pending
        # rows (already sent), so the SAME row is never delivered twice.
        assert comp.outbox.claim(
            event_id, worker_id="w-a", claim_token="tok-A", now=123.0
        ) is True
        assert comp.outbox.finish(event_id, claim_token="tok-A") is True
        rows = comp.db.query("SELECT status FROM outbox_events WHERE id = ?", (event_id,))
        assert rows[0]["status"] == "sent"
        assert comp.outbox.publish_pending() == 0
        assert len(publisher.delivered) == 0
        # A fresh emit (pending) is claimed + published end to end.
        event_id2 = _emit(comp, key="outbox-2")
        assert comp.outbox.publish_pending() == 1
        assert len(publisher.delivered) == 1
        assert publisher.delivered[0]["event_type"] == "FactorCandidateDiscovered"
        # A second drain finds nothing pending.
        assert comp.outbox.publish_pending() == 0
    finally:
        comp.close()


def test_composition_schema_and_migrations_applied():
    """The composed backend has the full platform schema PLUS the durable
    orchestration tables (migrations v1..v4)."""
    comp = production_composition(":memory:")
    try:
        tables = {
            r["name"]
            for r in comp.db.query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        for table in (
            "outbox_events",
            "inbox_events",
            "workflow_stage_runs",
            "consumed_manifests",
            "batch_fingerprints",
            "stage_idempotency_keys",
            "schema_migrations",
        ):
            assert table in tables, f"missing table {table}"
    finally:
        comp.close()


def test_composition_inbox_idempotent():
    handled: list[dict] = []
    comp = production_composition(":memory:", handler=lambda ev: handled.append(ev))
    try:
        assert comp.inbox.process(_event()) is True
        assert comp.inbox.process(_event()) is False  # duplicate delivery no-op
        assert len(handled) == 1
        assert comp.inbox.status_of("ik-1") == "DONE"
    finally:
        comp.close()


def test_composition_dead_letter_path():
    """A handler that always fails eventually dead-letters the inbox row, which
    is exported by ``dead_lettered()`` for production ops."""
    calls: list[str] = []

    def boom(event: dict) -> None:
        calls.append(event["idempotency_key"])
        raise RuntimeError("always fails")

    comp = production_composition(":memory:", handler=boom, max_retries=2)
    try:
        ev = _event(key="ik-dl-1", event_id="e-dl-1")
        # attempt 1: handler raises -> FAILED (retry_count=1)
        with pytest.raises(RuntimeError):
            comp.inbox.process(ev)
        # attempt 2: handler raises -> FAILED (retry_count=2)
        with pytest.raises(RuntimeError):
            comp.inbox.process(ev)
        # attempt 3: at max retries -> dead-lettered up-front, InboxEventNotFound
        with pytest.raises(InboxEventNotFound):
            comp.inbox.process(ev)
        assert len(calls) == 2  # the dead-letter attempt never re-ran the handler

        dead = comp.inbox.dead_lettered()
        assert len(dead) == 1
        assert dead[0]["idempotency_key"] == "ik-dl-1"
        assert dead[0]["status"] == "FAILED"
        assert int(dead[0]["retry_count"]) >= 2
        assert "max retries exceeded" in (dead[0]["last_error"] or "")
    finally:
        comp.close()


def test_composition_dead_lettered_empty_by_default():
    comp = production_composition(":memory:", handler=lambda ev: None)
    try:
        assert comp.inbox.dead_lettered() == []
    finally:
        comp.close()


def test_composition_accepts_prebuilt_db():
    """A caller-owned SqliteDb is used as-is (schema/migrations applied on it)."""
    db = SqliteDb(":memory:", create=True)
    comp = production_composition(db=db)
    try:
        assert comp.db is db
        event_id = _emit(comp, key="prebuilt-1")
        assert comp.outbox.publish_pending() == 1
        assert comp.inbox.process(_event(key="prebuilt-inbox-1")) is True
    finally:
        comp.close()
