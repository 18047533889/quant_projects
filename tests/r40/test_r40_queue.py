"""R40 #97/#98/#99/#100/#101/#102: service.queue remediation tests."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from factor_engine.service.jobstore import JobRecord, JobStore, _utc_now
from factor_engine.service.queue import (
    BoundedJobQueue,
    JobCancelledError,
    _cancel_event_ctx_var,
    check_job_alive,
)


def _old_ts() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


# ---------------------------------------------------------------------------
# #98: task_done() called after each dequeued job (queue.join() returns)
# ---------------------------------------------------------------------------


def test_queue_task_done_called_after_job_completion(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=1, heartbeat_check_interval=0.05)
    q.start(store)
    done: list[str] = []

    def fast(job):
        done.append(job.run_id)

    q.submit(store.create(JobRecord(run_id="td1", status="submitted")), run_fn=fast)
    deadline = time.time() + 5
    while q._pending.unfinished_tasks > 0 and time.time() < deadline:
        time.sleep(0.02)
    assert q._pending.unfinished_tasks == 0
    assert done == ["td1"]
    # queue.join() must return promptly (not block forever)
    join_ok: dict[str, bool] = {}

    def _join():
        q._pending.join()
        join_ok["done"] = True

    t = threading.Thread(target=_join, daemon=True)
    t.start()
    t.join(timeout=5)
    assert join_ok.get("done") is True
    q.stop()


# ---------------------------------------------------------------------------
# #97: drain lets queued jobs complete naturally; stragglers cancelled after grace
# ---------------------------------------------------------------------------


def test_drain_allows_queued_jobs_to_complete(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=2, heartbeat_check_interval=0.05)
    q.start(store)
    done: list[str] = []

    def fast(job):
        done.append(job.run_id)
        job.status = "succeeded"
        job.finished_at = _utc_now()
        store.update(job)

    q.submit(store.create(JobRecord(run_id="a", status="submitted")), run_fn=fast)
    q.drain(timeout=5)
    assert done == ["a"]
    assert store.get("a").status == "succeeded"


def test_drain_cancels_stragglers_after_grace(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=1, heartbeat_check_interval=0.05)
    q.start(store)
    gate = threading.Event()

    def slow(job):
        gate.wait(timeout=10)

    q.submit(store.create(JobRecord(run_id="slow", status="submitted")), run_fn=slow)
    time.sleep(0.2)  # let worker pick it up
    q.drain(timeout=0.4)  # short timeout -> phase 2/3
    status = store.get("slow").status
    assert status in ("interrupted", "cancelled")
    gate.set()
    q.stop()


# ---------------------------------------------------------------------------
# #99: drain state changes are durable (survive store reopen)
# ---------------------------------------------------------------------------


def test_drain_state_changes_are_durable(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=1, heartbeat_check_interval=0.05)
    q.start(store)
    gate = threading.Event()

    def slow(job):
        gate.wait(timeout=10)

    q.submit(store.create(JobRecord(run_id="durable", status="submitted")), run_fn=slow)
    time.sleep(0.2)
    q.drain(timeout=0.4)
    gate.set()
    q.stop()
    reopened = JobStore(tmp_path)
    rec = reopened.get("durable")
    assert rec is not None
    assert rec.status in ("interrupted", "cancelled")
    assert rec.finished_at is not None


# ---------------------------------------------------------------------------
# #100: heartbeat monitor handles null heartbeat (uses started_at)
# ---------------------------------------------------------------------------


def test_heartbeat_monitor_handles_null_heartbeat(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=2, heartbeat_check_interval=0.05)
    q.heartbeat_stale_seconds = 0.1
    q.start(store)
    job = JobRecord(run_id="null_hb", status="running", started_at=_old_ts(), heartbeat_at=None)
    store.create(job)
    time.sleep(0.6)  # let the monitor poll several times
    assert store.get("null_hb").status == "interrupted"
    q.stop()


# ---------------------------------------------------------------------------
# #101: worker checks the cancel event between phases -> stops computation
# ---------------------------------------------------------------------------


def test_check_job_alive_raises_when_cancel_event_set():
    ev = threading.Event()
    ev.set()
    token = _cancel_event_ctx_var.set(ev)
    try:
        with pytest.raises(JobCancelledError):
            check_job_alive(JobRecord(run_id="x", status="running"))
    finally:
        _cancel_event_ctx_var.reset(token)


def test_interrupt_running_sets_cancel_event(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=1, heartbeat_check_interval=0.05)
    q.start(store)
    gate = threading.Event()

    def slow(job):
        gate.wait(timeout=10)

    q.submit(store.create(JobRecord(run_id="hb_worker", status="submitted")), run_fn=slow)
    time.sleep(0.2)
    assert "hb_worker" in q._running
    assert "hb_worker" in q._cancel_events
    q._interrupt_running()
    ev = q._cancel_events.get("hb_worker")
    assert ev is not None and ev.is_set()
    gate.set()
    q.stop()


# ---------------------------------------------------------------------------
# #102: _mark_resource_rejected persistence failure is not silently swallowed
# ---------------------------------------------------------------------------


def test_mark_resource_rejected_persist_failure_journals(tmp_path):
    store = JobStore(tmp_path)
    q = BoundedJobQueue(max_queue=16, max_running=1, heartbeat_check_interval=0.05)
    q.start(store)
    job = JobRecord(run_id="rej", status="submitted")
    calls: list[int] = []

    def boom_update(*args, **kwargs):  # noqa: ARG001
        calls.append(1)
        raise RuntimeError("disk full")

    store.update = boom_update  # type: ignore[method-assign]
    q._store = store
    q._mark_resource_rejected(job)
    assert q.resource_reject_fatal_counter == 1
    journal = store.root / "emergency_journal.jsonl"
    assert journal.exists()
    assert "rej" in journal.read_text(encoding="utf-8")
    q.stop()
