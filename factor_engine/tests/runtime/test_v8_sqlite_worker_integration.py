import hashlib
import json
import os
import uuid

from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker
from factor_engine.runtime.worker_ownership import RunOwnershipContext
from factor_engine.runtime.worker_ownership_sqlite import (
    initialize_worker_store,
    read_worker_event_snapshot_sqlite,
    validate_all_workers_exited_sqlite,
)


def _context(run_dir):
    run_id = uuid.uuid4().hex
    payload = json.dumps({"run_id": run_id, "ownership_store": "sqlite-v1"}).encode()
    (run_dir / "identity.json").write_bytes(payload)
    context = RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())
    initialize_worker_store(run_dir, context=context)
    return context


def _pid():
    return os.getpid()


def test_supervisor_uses_real_sqlite_lifecycle_without_legacy_journal(tmp_path):
    context = _context(tmp_path)
    worker = SupervisedReusableWorker(
        context="spawn", ownership_run_dir=tmp_path, ownership_context=context,
        ownership_role="compute", cancel_grace_seconds=0,
        exit_observation_seconds=2,
    )
    try:
        result = worker.execute(_pid, timeout_seconds=5)
        events = list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
        assert [event["event"] for event in events] == ["STARTING", "BOUND"]
        assert result.value == events[1]["pid"] == worker._process.pid
        assert not validate_all_workers_exited_sqlite(tmp_path, context=context).all_exited
    finally:
        worker.close()
    validation = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert validation.all_exited and validation.instance_count == 1
    assert not (tmp_path / "worker-ownership.jsonl").exists()


def test_spawn_ingestion_uses_real_sqlite_ownership_and_exits_before_return(tmp_path):
    context = _context(tmp_path)
    manifest = FiniteFactorManifest.ingest_supervised(
        [{"name": "alpha"}], tmp_path / "manifest.sqlite3",
        process_context="spawn", ownership_run_dir=tmp_path,
        ownership_context=context, deadline_seconds=5,
    )
    try:
        assert manifest.input_complete and len(manifest) == 1
        events = list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
        assert [event["event"] for event in events] == ["STARTING", "BOUND", "EXITED"]
        assert events[0]["role"] == "manifest-ingestion"
        validation = validate_all_workers_exited_sqlite(tmp_path, context=context)
        assert validation.all_exited and validation.instance_count == 1
        assert not (tmp_path / "worker-ownership.jsonl").exists()
    finally:
        manifest.close()
