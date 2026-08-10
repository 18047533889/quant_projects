"""R21-055..075: job lifecycle — durable store, restart reconciliation,
cancel propagation, deadline via monotonic clock, corrupt-manifest quarantine.
"""

from __future__ import annotations

import json
import os

import pytest

from service.jobstore import JobRecord, JobStatus, JobStore
from service.queue import (
    BoundedJobQueue,
    JobCancelledError,
    JobDeadlineExceeded,
    check_job_alive,
)


class TestRestartReconciliation:
    def test_stale_running_marked_interrupted_on_restart(self, tmp_path):
        store1 = JobStore(tmp_path)
        store1.create(JobRecord(run_id="stale", status=JobStatus.RUNNING))
        store1.create(JobRecord(run_id="done", status=JobStatus.SUCCEEDED))
        store2 = JobStore(tmp_path)
        assert store2.get("stale").status == JobStatus.INTERRUPTED
        assert store2.get("done").status == JobStatus.SUCCEEDED

    def test_corrupt_manifest_quarantined_not_silent(self, tmp_path):
        manifests = tmp_path / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / "corrupt.json").write_text("{ not valid json", encoding="utf-8")
        store = JobStore(tmp_path)
        assert store.corruption_count >= 1
        assert (tmp_path / "quarantine").exists()

    def test_checksum_detects_tampered_manifest(self, tmp_path):
        store1 = JobStore(tmp_path)
        store1.create(JobRecord(run_id="tampered", status=JobStatus.SUCCEEDED))
        path = tmp_path / "manifests" / "tampered.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["status"] = JobStatus.RUNNING
        raw.pop("_checksum", None)  # tamper without recomputing checksum
        path.write_text(json.dumps(raw), encoding="utf-8")
        store2 = JobStore(tmp_path)
        # the tampered record must be quarantined, not silently restored running
        assert store2.get("tampered") is None or store2.get("tampered").status != JobStatus.RUNNING


class TestJobLifecycle:
    def test_cancel_flag_and_phase(self, tmp_path):
        store = JobStore(tmp_path)
        queue = BoundedJobQueue(max_queue=8, max_running=2)
        queue.start(store)
        job = store.create(JobRecord(run_id="cancellable", status=JobStatus.QUEUED))
        assert queue.cancel("cancellable") is True
        refreshed = store.get("cancellable")
        assert refreshed.status == JobStatus.CANCELLING
        assert refreshed.cancel_requested_at is not None

    def test_check_job_alive_raises_on_cancel(self, tmp_path):
        store = JobStore(tmp_path)
        job = store.create(JobRecord(run_id="c2", status=JobStatus.RUNNING))
        store.update(job)
        queue = BoundedJobQueue(max_queue=8, max_running=2)
        queue.start(store)
        queue.cancel("c2")
        with pytest.raises(JobCancelledError):
            check_job_alive(store.get("c2"))

    def test_deadline_uses_monotonic(self, tmp_path):
        import time

        store = JobStore(tmp_path)
        job = JobRecord(run_id="dl", status=JobStatus.RUNNING)
        job.deadline_monotonic = time.monotonic() - 1.0  # already expired
        job.deadline_at = "2099-01-01T00:00:00+00:00"  # wall clock says far future
        store.create(job)
        # monotonic deadline expired even though wall clock says otherwise
        # (a clock rollback must not disable the timeout, R21-277).
        with pytest.raises(JobDeadlineExceeded):
            check_job_alive(store.get("dl"))


class TestDurableStore:
    def test_sqlite_store_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACTOR_ENGINE_SERVICE_DURABLE_STORE", "sqlite")
        store = JobStore(tmp_path)
        store.create(JobRecord(run_id="s1", status=JobStatus.SUCCEEDED, idempotency_key="ik1"))
        store2 = JobStore(tmp_path)
        assert store2.get("s1").status == JobStatus.SUCCEEDED
        assert store2.get_by_idempotency_key("ik1").run_id == "s1"

    def test_single_process_workers_refused(self, monkeypatch):
        from service.jobstore import check_single_process_workers
        from service.errors import ServiceError

        monkeypatch.setenv("UVICORN_WORKERS", "4")
        with pytest.raises(ServiceError):
            check_single_process_workers()
        monkeypatch.setenv("FACTOR_ENGINE_SERVICE_DURABLE_STORE", "sqlite")
        check_single_process_workers()  # no raise with durable store
