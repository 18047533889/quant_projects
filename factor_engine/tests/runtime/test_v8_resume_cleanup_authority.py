import json
import os
import sqlite3
from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.resume_validation import ResumeIdentityError
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker, WorkerQuarantined
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor


def _successful_run(tmp_path):
    policy = resolve_default_policy()
    engine = FakeEngine()
    receipt = pipeline.execute_run_many_durable(
        engine, [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    return policy, engine, receipt


def _corrupt_artifact(receipt):
    with sqlite3.connect(receipt["state_path"]) as db:
        artifact = json.loads(db.execute(
            "select artifact_json from outcomes where ordinal=0"
        ).fetchone()[0])
    artifact_manifest = json.loads(Path(artifact["path"]).read_text())
    chunk = Path(artifact["path"]).parent / artifact_manifest["chunks"][0]["path"]
    payload = bytearray(chunk.read_bytes())
    payload[len(payload) // 2] ^= 1
    chunk.write_bytes(payload)


def test_resume_verification_primary_retains_all_authorities_on_cleanup_quarantine(
    tmp_path, monkeypatch,
):
    policy, engine, first = _successful_run(tmp_path)
    _corrupt_artifact(first)

    closed_workers = []
    original_close = SupervisedReusableWorker.close

    def close_then_report_uncertainty(self):
        pid = self._process.pid
        original_close(self)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        closed_workers.append(self)
        raise WorkerQuarantined(
            "injected uncertainty after real verification-worker exit", pid=pid,
        )

    monkeypatch.setattr(SupervisedReusableWorker, "close", close_then_report_uncertainty)
    with pytest.raises(ResumeIdentityError, match="hash changed") as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )

    primary = caught.value
    assert len(closed_workers) == 1
    assert primary.cleanup_pending is True
    assert len(primary.cleanup_errors) == 1
    assert isinstance(primary.cleanup_errors[0], WorkerQuarantined)
    assert primary.verification_worker is closed_workers[0]
    assert primary.coordinator_lock.owned is True
    assert primary.resume_state.counts() == {"SUCCEEDED": 1}
    assert len(primary.resume_manifest) == 1
    assert primary.broker is engine.resource_broker

    run_dir = Path(first["manifest_path"]).parent
    challenger = pipeline._RunCoordinatorLock(run_dir)
    try:
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            challenger.acquire(allow_dead_owner=True)
    finally:
        primary.resume_state.close()
        primary.resume_manifest.close()
        primary.coordinator_lock.release()

    challenger.acquire(allow_dead_owner=True)
    challenger.release()


def test_direct_resume_worker_quarantine_retains_all_authorities(tmp_path, monkeypatch):
    policy, engine, first = _successful_run(tmp_path)
    workers = []
    original_execute = SupervisedReusableWorker.execute
    original_close = SupervisedReusableWorker.close

    def execute_then_quarantine(self, *args, **kwargs):
        original_execute(self, *args, **kwargs)
        workers.append(self)
        raise WorkerQuarantined("direct verification quarantine", pid=self._process.pid)

    monkeypatch.setattr(SupervisedReusableWorker, "execute", execute_then_quarantine)
    with pytest.raises(WorkerQuarantined, match="direct verification") as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )

    failure = caught.value
    assert workers == [failure.verification_worker]
    assert failure.cleanup_pending is True
    assert failure.coordinator_lock.owned is True
    assert failure.resume_state.counts() == {"SUCCEEDED": 1}
    assert len(failure.resume_manifest) == 1
    assert failure.broker is engine.resource_broker
    try:
        probe = pipeline._RunCoordinatorLock(Path(first["manifest_path"]).parent)
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            probe.acquire(allow_dead_owner=True)
    finally:
        original_close(failure.verification_worker)
        failure.resume_state.close()
        failure.resume_manifest.close()
        failure.coordinator_lock.release()


@pytest.mark.parametrize("authority_type", [PersistentRunState, FiniteFactorManifest])
def test_ordinary_resume_authority_close_error_preserves_primary(
    tmp_path, monkeypatch, authority_type,
):
    policy, engine, first = _successful_run(tmp_path)
    _corrupt_artifact(first)
    original_close = authority_type.close

    def close_then_fail(self):
        original_close(self)
        raise RuntimeError(f"ordinary {authority_type.__name__} close failure")

    monkeypatch.setattr(authority_type, "close", close_then_fail)
    with pytest.raises(ResumeIdentityError, match="hash changed") as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )
    assert any(
        isinstance(error, RuntimeError) and "close failure" in str(error)
        for error in caught.value.cleanup_errors
    )
    probe = pipeline._RunCoordinatorLock(Path(first["manifest_path"]).parent)
    probe.acquire(allow_dead_owner=True)
    probe.release()


def test_cleanup_pending_state_close_retains_remaining_resume_authority(
    tmp_path, monkeypatch,
):
    policy, engine, first = _successful_run(tmp_path)
    _corrupt_artifact(first)
    original_state_close = PersistentRunState.close

    def quarantined_state_close(_self):
        raise WorkerQuarantined("state close ownership uncertain", pid=None)

    monkeypatch.setattr(PersistentRunState, "close", quarantined_state_close)
    with pytest.raises(ResumeIdentityError, match="hash changed") as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )

    primary = caught.value
    assert primary.cleanup_pending is True
    assert isinstance(primary.cleanup_errors[0], WorkerQuarantined)
    assert primary.coordinator_lock.owned is True
    assert primary.resume_state.counts() == {"SUCCEEDED": 1}
    assert len(primary.resume_manifest) == 1
    assert primary.broker is engine.resource_broker
    try:
        probe = pipeline._RunCoordinatorLock(Path(first["manifest_path"]).parent)
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            probe.acquire(allow_dead_owner=True)
    finally:
        original_state_close(primary.resume_state)
        primary.resume_manifest.close()
        primary.coordinator_lock.release()


def test_successful_verification_close_error_is_not_retried_or_unlocked(
    tmp_path, monkeypatch,
):
    policy, engine, first = _successful_run(tmp_path)
    original_close = SupervisedReusableWorker.close
    close_calls = []

    def close_once_then_fail(self):
        close_calls.append(self)
        original_close(self)
        raise RuntimeError("ordinary error from attempted verification close")

    monkeypatch.setattr(SupervisedReusableWorker, "close", close_once_then_fail)
    with pytest.raises(RuntimeError, match="attempted verification close") as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )

    failure = caught.value
    assert close_calls == [failure.verification_worker]
    assert failure.cleanup_pending is True
    assert failure.coordinator_lock.owned is True
    assert failure.resume_state.counts() == {"SUCCEEDED": 1}
    assert len(failure.resume_manifest) == 1
    assert failure.broker is engine.resource_broker
    try:
        probe = pipeline._RunCoordinatorLock(Path(first["manifest_path"]).parent)
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            probe.acquire(allow_dead_owner=True)
    finally:
        failure.resume_state.close()
        failure.resume_manifest.close()
        failure.coordinator_lock.release()
