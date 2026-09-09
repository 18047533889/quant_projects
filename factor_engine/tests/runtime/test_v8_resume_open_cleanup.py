from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.supervised_worker import WorkerQuarantined
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor


@pytest.mark.parametrize("cleanup_pending", [False, True])
def test_resume_state_open_primary_survives_manifest_close_failure(
    tmp_path, monkeypatch, cleanup_pending,
):
    policy = resolve_default_policy()
    engine = FakeEngine()
    first = pipeline.execute_run_many_durable(
        engine, [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    opened = []
    original_manifest_close = FiniteFactorManifest.close
    primary = ValueError("deliberate resume state open failure")
    cleanup = (
        WorkerQuarantined("manifest close ownership uncertain", pid=None)
        if cleanup_pending else RuntimeError("ordinary manifest close failure")
    )

    def open_manifest(path):
        manifest = FiniteFactorManifest(path)
        opened.append(manifest)
        return manifest

    def fail_state_open(*_args, **_kwargs):
        raise primary

    def fail_manifest_close(self):
        if cleanup_pending:
            raise cleanup
        original_manifest_close(self)
        raise cleanup

    monkeypatch.setattr(pipeline, "FiniteFactorManifest", open_manifest)
    monkeypatch.setattr(pipeline, "PersistentRunState", fail_state_open)
    monkeypatch.setattr(FiniteFactorManifest, "close", fail_manifest_close)

    with pytest.raises(ValueError, match="resume state open failure") as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )
    assert caught.value is primary
    assert opened and caught.value.cleanup_errors == [cleanup]

    run_dir = Path(first["manifest_path"]).parent
    challenger = pipeline._RunCoordinatorLock(run_dir)
    if cleanup_pending:
        assert primary.cleanup_pending is True
        assert primary.resume_manifest is opened[0]
        assert primary.coordinator_lock.owned is True
        assert primary.broker is engine.resource_broker
        assert not hasattr(primary, "resume_state")
        assert len(primary.resume_manifest) == 1
        try:
            with pytest.raises(RuntimeError, match="coordinator is still alive"):
                challenger.acquire(allow_dead_owner=True)
        finally:
            original_manifest_close(primary.resume_manifest)
            primary.coordinator_lock.release()
    challenger.acquire(allow_dead_owner=True)
    challenger.release()


def test_resume_validation_quarantine_retains_lock_without_fabricating_databases(
    tmp_path, monkeypatch,
):
    import factor_engine.runtime.resume_validation as resume_validation

    policy = resolve_default_policy()
    engine = FakeEngine()
    first = pipeline.execute_run_many_durable(
        engine, [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
    )
    quarantine = WorkerQuarantined(
        "validation cleanup ownership uncertain", pid=None,
    )
    monkeypatch.setattr(
        resume_validation, "validate_resume_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(quarantine),
    )

    with pytest.raises(WorkerQuarantined) as caught:
        pipeline.execute_run_many_durable(
            engine, (), policy=policy, artifact_root=tmp_path,
            resume_run_id=first["run_id"],
        )
    assert caught.value is quarantine
    assert quarantine.cleanup_pending is True
    assert quarantine.broker is engine.resource_broker
    assert quarantine.coordinator_lock.owned is True
    assert not hasattr(quarantine, "resume_manifest")
    assert not hasattr(quarantine, "resume_state")

    run_dir = Path(first["manifest_path"]).parent
    challenger = pipeline._RunCoordinatorLock(run_dir)
    try:
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            challenger.acquire(allow_dead_owner=True)
    finally:
        quarantine.coordinator_lock.release()
    challenger.acquire(allow_dead_owner=True)
    challenger.release()
