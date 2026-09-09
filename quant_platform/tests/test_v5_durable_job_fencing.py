from datetime import datetime, timedelta, timezone
import os
import threading
import uuid

import pytest

from quant_platform.app.contracts import JobResult, JobSpec, JobStatus
from quant_platform.app.db.postgres_backend import PostgresDialect
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.worker.durable_jobs import DurableJobStore
from quant_platform.app.worker.jobs import JobRunner, StaleJobAttemptError


def test_lease_reclaim_fences_late_old_worker_result(tmp_path):
    db_path = str(tmp_path / "fence.db")
    SqliteDb(db_path).close()
    clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    entered = threading.Event()
    release_old = threading.Event()
    old_outcome = []
    spec = JobSpec("qe_metric_instance_shard", "fenced", max_retries=2,
                   timeout_seconds=10)

    def old_handler(_):
        entered.set()
        assert release_old.wait(5)
        return JobResult(("result:old",))

    def run_old():
        db = SqliteDb(db_path)
        try:
            store = DurableJobStore(db, now_fn=lambda: clock[0])
            old_outcome.append(JobRunner("old-worker", store).execute(spec, old_handler))
        except Exception as exc:
            old_outcome.append(exc)
        finally:
            db.close()

    old_thread = threading.Thread(target=run_old)
    old_thread.start()
    assert entered.wait(5)

    clock[0] += timedelta(seconds=11)
    new_db = SqliteDb(db_path)
    new_store = DurableJobStore(new_db, now_fn=lambda: clock[0])
    new_record = JobRunner("new-worker", new_store).execute(
        spec, lambda _: JobResult(("result:new",)))
    assert new_record.status is JobStatus.SUCCEEDED
    assert new_record.result.output_artifact_refs == ("result:new",)

    release_old.set()
    old_thread.join(5)
    assert not old_thread.is_alive()
    assert len(old_outcome) == 1
    assert isinstance(old_outcome[0], StaleJobAttemptError)
    assert "attempt token 1 is stale" in str(old_outcome[0])

    final = new_store.load("fenced")
    assert final.status is JobStatus.SUCCEEDED
    assert final.attempt_count == 2
    assert tuple(a.worker_id for a in final.attempts) == ("old-worker", "new-worker")
    assert final.result.output_artifact_refs == ("result:new",)
    new_db.close()


def test_claim_cas_sql_is_postgres_placeholder_compatible():
    sql = (
        "UPDATE jobs SET attempt_count=? WHERE job_id=? AND attempt_count=? "
        "AND EXISTS (SELECT 1 FROM job_attempts WHERE job_id=? AND worker_id=?)"
    )
    adapted = PostgresDialect.adapt(sql)
    assert "?" not in adapted
    assert adapted.count("%s") == 5
    assert "EXISTS" in adapted


def test_lease_reclaim_fences_late_old_worker_result_postgres():
    dsn = os.environ.get("QP_PG_DSN")
    if not dsn:
        pytest.skip("QP_PG_DSN is required for disposable live PostgreSQL fencing test")
    pytest.importorskip("psycopg2")
    from quant_platform.app.db.postgres_backend import PostgresDb

    key = "fenced-pg-" + uuid.uuid4().hex
    clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    entered = threading.Event()
    release_old = threading.Event()
    old_outcome = []
    spec = JobSpec("qe_metric_instance_shard", key, max_retries=2, timeout_seconds=10)

    setup = PostgresDb(dsn, create=True)
    setup.close()

    def old_handler(_):
        entered.set()
        assert release_old.wait(10)
        return JobResult(("result:old-pg",))

    def run_old():
        db = PostgresDb(dsn, create=False)
        try:
            store = DurableJobStore(db, now_fn=lambda: clock[0])
            old_outcome.append(JobRunner("old-pg-worker", store).execute(spec, old_handler))
        except Exception as exc:
            old_outcome.append(exc)
        finally:
            db.close()

    thread = threading.Thread(target=run_old)
    thread.start()
    assert entered.wait(10)
    clock[0] += timedelta(seconds=11)
    new_db = PostgresDb(dsn, create=False)
    try:
        store = DurableJobStore(new_db, now_fn=lambda: clock[0])
        winner = JobRunner("new-pg-worker", store).execute(
            spec, lambda _: JobResult(("result:new-pg",)))
        assert winner.result.output_artifact_refs == ("result:new-pg",)
        release_old.set()
        thread.join(10)
        assert isinstance(old_outcome[0], StaleJobAttemptError)
        final = store.load(key)
        assert final.attempt_count == 2
        assert final.result.output_artifact_refs == ("result:new-pg",)
        with new_db.transaction() as conn:
            conn.execute("DELETE FROM job_results WHERE job_id=?", ("job:" + key,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=?", ("job:" + key,))
            conn.execute("DELETE FROM jobs WHERE job_id=?", ("job:" + key,))
    finally:
        release_old.set()
        thread.join(10)
        new_db.close()
