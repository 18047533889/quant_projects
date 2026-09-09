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
The optional ``quant-platform[postgres-test]`` extra pins the test driver.
It is not a default runtime dependency and should be installed into an isolated
QA environment with a disposable database, never inferred to be production
deployment authorization. Consequence:

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
            "(install the pinned postgres-test extra in an isolated test environment). "
            "Set QP_PG_DSN to a disposable database to run "
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


@pytest.fixture
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
        from quant_platform.app.db.schema import TABLE_NAMES
        for t in sorted(tables):
            if t in TABLE_NAMES:
                db.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
        db._conn.commit()
    finally:
        db.close()


def test_live_transaction_yields_sqlite_style_adapter(pg: PostgresDb):
    """P0-PLAT-008: ``transaction()`` yields a SQLite-style adapter, not the raw
    psycopg2 connection — shared outbox/inbox business code (``conn.execute`` /
    ``conn.execute(...).fetchone()``) must run against PG unchanged."""
    with pg.transaction() as conn:
        conn.execute(
            "INSERT INTO outbox_events "
            "(event_type, aggregate_type, aggregate_id, correlation_id, "
            "idempotency_key, payload_json, occurred_at, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0) RETURNING id",
            ("adapter_test", "agg", "a1", "c1", "k-adapter", "{}", 0),
        )
    # verify the row landed (committed)
    rows = pg.query(
        "SELECT event_type FROM outbox_events WHERE idempotency_key = ?",
        ("k-adapter",),
    )
    assert rows and rows[0]["event_type"] == "adapter_test"


def test_live_transaction_adapter_execute_returning(pg: PostgresDb):
    """P0-PLAT-008: ``execute_returning`` returns the inserted surrogate id."""
    with pg.transaction() as conn:
        rid = conn.execute_returning(
            "INSERT INTO outbox_events "
            "(event_type, aggregate_type, aggregate_id, correlation_id, "
            "idempotency_key, payload_json, occurred_at, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0) RETURNING id",
            ("ret_test", "agg", "a2", "c2", "k-ret", "{}", 0),
        )
    assert isinstance(rid, int) and rid > 0
    rows = pg.query(
        "SELECT event_type FROM outbox_events WHERE idempotency_key = ?",
        ("k-ret",),
    )
    assert rows and rows[0]["event_type"] == "ret_test"


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
    assert len(TABLE_NAMES) == len(set(TABLE_NAMES))


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
        "VALUES (?, ?, 'PROCESSING', 0, NULL, NULL, FALSE, NULL, ?)",
        ("ev-1", "idem-1", 1.0),
    )
    # Second identical insert must be ignored (no PK/unique violation raised).
    pg.execute(
        "INSERT OR IGNORE INTO inbox_events "
        "(event_id, idempotency_key, status, retry_count, next_attempt_at, "
        " last_error, dead_letter, dead_letter_reason, processed_at) "
        "VALUES (?, ?, 'PROCESSING', 0, NULL, NULL, FALSE, NULL, ?)",
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
    assert published == 0, "an already acknowledged row must not be delivered again"
    assert not publisher.delivered
    with pg.transaction():
        outbox.emit(event_type="FactorCandidateDiscovered", aggregate_type="FactorCandidate",
                    aggregate_id="fc-live-2", correlation_id="corr-live-2",
                    idempotency_key="outbox-live-2", payload={"stage":"DISCOVERED"})
    assert outbox.publish_pending(limit=10) == 1
    assert outbox.publish_pending(limit=10) == 0
    assert publisher.delivered and publisher.delivered[0]["event_type"] == \
        "FactorCandidateDiscovered"


def test_live_generation_publish_and_gc_root(pg):
    import hashlib
    from datetime import datetime, timezone
    from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_CANDIDATE
    from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator

    class Publisher:
        def publish(self, artifact, data):
            assert hashlib.sha256(data).hexdigest() == artifact.content_hash
            return artifact

    payload = b'postgres-generation-fixture'
    artifact = ArtifactRef(artifact_id='pg-fixture-generation',
        artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE, schema_version='1',
        content_hash=hashlib.sha256(payload).hexdigest(), storage_uri='test://pg-fixture',
        size_bytes=len(payload), created_at=datetime(2026,1,1,tzinfo=timezone.utc),
        producer_type='fixture', producer_version='1')
    coordinator = DurableGenerationCoordinator(pg, Publisher())
    generation = coordinator.stage(artifact, payload)
    assert coordinator.resolve_active(artifact.artifact_id) is None
    assert coordinator.outbox.publish_pending() == 1
    assert coordinator.resolve_active(artifact.artifact_id)['status'] == 'COMPLETE'
    coordinator.protect_gc_root('active_read', artifact.artifact_id, generation)
    snapshot = coordinator.snapshot_for_gc()
    assert (artifact.artifact_id, generation) in snapshot.active_read_roots
    claim = coordinator.claim_gc_tombstone(artifact.artifact_id, generation,
                                           expected_epoch=snapshot.epoch)
    assert claim.status.value == 'ROOTED'


def test_live_concurrent_first_delivery_runs_handler_once(pg):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    barrier, lock = Barrier(2), Lock()
    calls = []
    connections = [PostgresDb(_dsn(), create=False) for _ in range(2)]
    def consume(db):
        barrier.wait(timeout=10)
        return Inbox(db, handle).process(event)
    def handle(event):
        with lock:
            calls.append(event['idempotency_key'])
    event = {'event_id':'concurrent-first', 'idempotency_key':'concurrent-first', 'payload':{}}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(consume, connections))
        assert sorted(results) == [False, True]
        assert calls == ['concurrent-first']
    finally:
        for db in connections:
            db.close()


@pytest.mark.parametrize("after_write", [False, True])
def test_live_inbox_process_death_rolls_back_claim_and_effect(pg, after_write):
    import subprocess
    import sys
    code = '''
import os, sys
from quant_platform.app.db import PostgresDb
from quant_platform.app.outbox import Inbox
db = PostgresDb(os.environ['QP_PG_DSN'], create=False)
def die(event):
    if sys.argv[1] == 'True':
        db.execute("INSERT INTO teams (team_id,name) VALUES ('crash-effect','test')")
    os._exit(71)
Inbox(db, die).process({'event_id':'crash', 'idempotency_key':'crash'})
'''
    result = subprocess.run([sys.executable, '-c', code, str(after_write)],
                            env={**os.environ, 'QP_PG_DSN': _dsn()}, timeout=20)
    assert result.returncode == 71
    assert pg.query("SELECT * FROM teams WHERE team_id = 'crash-effect'") == []
    assert pg.query("SELECT * FROM inbox_events WHERE idempotency_key = 'crash'") == []
    assert Inbox(pg, lambda event: None).process({'event_id':'crash', 'idempotency_key':'crash'})


def test_live_inbox_sql_error_rolls_back_handler_savepoint(pg):
    event = {'event_id':'sql-error', 'idempotency_key':'sql-error'}
    def fail(event):
        pg.execute("INSERT INTO teams (team_id,name) VALUES ('sql-effect','test')")
        pg.execute("INSERT INTO teams (team_id,name) VALUES ('sql-effect','duplicate')")
    with pytest.raises(Exception, match='duplicate key'):
        Inbox(pg, fail).process(event)
    assert pg.query("SELECT * FROM teams WHERE team_id = 'sql-effect'") == []
    assert Inbox(pg, fail).status_of('sql-error') == 'FAILED'
    assert Inbox(pg, lambda event: None).process(event)


def test_live_outbox_death_after_delivery_replays_without_duplicate_db_effect(pg):
    import subprocess
    import sys
    import time
    with pg.transaction():
        event_id = Outbox(pg, InMemoryPublisher()).emit(event_type='test', aggregate_type='test',
            aggregate_id='1', correlation_id='1', idempotency_key='delivered-before-crash')
    code = '''
import os
from quant_platform.app.db import PostgresDb
from quant_platform.app.outbox import Inbox, Outbox
sender = PostgresDb(os.environ['QP_PG_DSN'], create=False)
receiver = PostgresDb(os.environ['QP_PG_DSN'], create=False)
def handle(event):
    receiver.execute("INSERT INTO teams (team_id,name) VALUES ('delivery-effect','once')")
class Bus:
    def publish(self, event):
        assert Inbox(receiver, handle).process(event)
        os._exit(72)
Outbox(sender, Bus()).publish_pending()
'''
    result = subprocess.run([sys.executable, '-c', code],
                            env={**os.environ, 'QP_PG_DSN': _dsn()}, timeout=20)
    assert result.returncode == 72
    assert pg.query("SELECT status FROM outbox_events WHERE id = ?", (event_id,))[0]['status'] == 'claimed'
    calls = []
    class ReplayBus:
        def publish(self, event):
            assert not Inbox(pg, lambda event: calls.append(event)).process(event)
    outbox = Outbox(pg, ReplayBus())
    assert outbox.recover_expired_claims(now=time.time() + 301, lease_seconds=300) == 1
    # Recovery schedules availability at its supplied clock; use that same
    # explicit fixture clock for the drain, rather than bypassing the backoff.
    from unittest.mock import patch
    with patch('quant_platform.app.outbox.time.time', return_value=time.time() + 302):
        assert outbox.publish_pending() == 1
    assert calls == []
    assert len(pg.query("SELECT * FROM teams WHERE team_id='delivery-effect'")) == 1


def test_live_public_production_composition_migrates_and_delivers(pg):
    from quant_platform.app.composition import production_composition
    calls = []
    composition = production_composition(db=pg, handler=lambda event: calls.append(event))
    with pg.transaction():
        composition.outbox.emit(event_type='test', aggregate_type='test', aggregate_id='1',
                               correlation_id='1', idempotency_key='composed-live')
    assert composition.outbox.publish_pending() == 1
    assert composition.inbox.process(composition.publisher.delivered[0])
    assert len(calls) == 1
    # Re-entering the public composition must replay migrations idempotently.
    production_composition(db=pg)


def test_live_migration_runner_and_explicit_epoch_upgrade(pg):
    from quant_platform.app.db.migrations import apply_migrations, postgres_event_clock_migration
    with pg.transaction() as tx:
        tx.execute('DROP TABLE IF EXISTS pg_fixture_versions')
        tx.execute('ALTER TABLE outbox_events ALTER COLUMN occurred_at TYPE TIMESTAMP WITHOUT TIME ZONE '
                   "USING TIMESTAMP '1970-01-01' + occurred_at * INTERVAL '1 second'")
        tx.execute('ALTER TABLE outbox_events ALTER COLUMN id DROP DEFAULT')
        tx.execute("INSERT INTO outbox_events (id,event_type,aggregate_type,aggregate_id,correlation_id,"
                   "idempotency_key,occurred_at) VALUES (42,'fixture','fixture','fixture','fixture',"
                   "'old-clock', TIMESTAMP '2024-01-01 00:00:00.125')")
    migration = postgres_event_clock_migration(legacy_timezone='UTC')
    kwargs = dict(current_version_table='pg_fixture_versions', migrations=(migration,))
    assert apply_migrations(pg, **kwargs) == (migration.id,)
    assert apply_migrations(pg, **kwargs) == ()
    value = pg.query("SELECT occurred_at FROM outbox_events WHERE id=42")[0]['occurred_at']
    assert value == 1704067200.125
    with pg.transaction():
        event_id = Outbox(pg, InMemoryPublisher()).emit(event_type='fixture', aggregate_type='fixture',
            aggregate_id='new',correlation_id='new',idempotency_key='new-clock')
    assert event_id > 42
    with pg.transaction() as tx:
        tx.execute('DROP TABLE pg_fixture_versions')


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


def test_live_generation_identity_and_reordered_delivery(pg: PostgresDb):
    from datetime import datetime, timezone
    import hashlib
    from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_VALUE
    from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator

    class Publisher:
        def publish(self, artifact, data):
            assert hashlib.sha256(data).hexdigest() == artifact.content_hash
            return artifact

    coordinator = DurableGenerationCoordinator(pg, Publisher())

    def stage(aid, data):
        digest = hashlib.sha256(data).hexdigest()
        return coordinator.stage(ArtifactRef(
            artifact_id=aid, artifact_type=ARTIFACT_TYPE_FACTOR_VALUE,
            schema_version="1", content_hash=digest, storage_uri=f"test://{aid}/{digest}",
            size_bytes=len(data), created_at=datetime.now(timezone.utc),
            producer_type="test", producer_version="1",
        ), data)

    old = stage("stable", b"old")
    other = stage("other", b"old")
    new = stage("stable", b"new")
    assert old != other
    coordinator.publish({"payload": {"generation_id": new}})
    coordinator.publish({"payload": {"generation_id": old}})
    coordinator.outbox.publish_pending()
    assert coordinator.resolve_active("stable")["generation_id"] == new
    assert coordinator.resolve_active("other")["generation_id"] == other


def test_live_targeted_outbox_recovery_preserves_unrelated_rows(pg: PostgresDb, monkeypatch):
    monkeypatch.setattr("quant_platform.app.outbox.time.time", lambda: 1000.0)
    outbox = Outbox(pg, InMemoryPublisher())
    ids = {}
    with pg.transaction():
        for key in ("production-pending", "production-expired", "shadow-target"):
            ids[key] = outbox.emit(event_type="test", aggregate_type="test", aggregate_id=key,
                                   correlation_id="isolation", idempotency_key=key)
        for key in ("production-expired", "shadow-target"):
            assert outbox.claim(ids[key], worker_id="old-worker", claim_token=key, now=1.0)
    before = [dict(row) for row in pg.query("SELECT * FROM outbox_events WHERE id != ? ORDER BY id", (ids["shadow-target"],))]
    assert outbox.publish_pending(idempotency_key="missing-key") == 0
    assert outbox.publish_pending(idempotency_key="shadow-target") == 1
    assert outbox.publish_pending(idempotency_key="shadow-target") == 0
    after = [dict(row) for row in pg.query("SELECT * FROM outbox_events WHERE id != ? ORDER BY id", (ids["shadow-target"],))]
    assert after == before
    assert not outbox.finish(ids["shadow-target"], claim_token="shadow-target")
