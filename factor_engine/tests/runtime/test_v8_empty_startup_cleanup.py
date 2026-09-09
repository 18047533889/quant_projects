from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker, WorkerQuarantined
from factor_engine.tests.runtime.test_v6_bounded_pipeline import (
    FakeBroker, _build_spawn_engine,
)


def _only_run_dir(root: Path) -> Path:
    runs = [path for path in root.iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def test_empty_direct_run_starts_no_execution_workers(tmp_path, monkeypatch):
    starts = []
    original_start = SupervisedReusableWorker.start

    def observed_start(self):
        starts.append(self)
        return original_start(self)

    monkeypatch.setattr(SupervisedReusableWorker, "start", observed_start)
    receipt = pipeline.execute_run_many_durable(
        None, [], policy=resolve_default_policy(),
        artifact_root=tmp_path / "artifacts",
        run_kwargs={"broker": FakeBroker()},
        engine_factory=_build_spawn_engine,
        engine_factory_config={},
    )

    assert receipt["status"] == "SUCCEEDED"
    assert receipt["requested_total"] == 0
    assert receipt["execution_batches"] == 0
    assert starts == []


@pytest.mark.parametrize("failure_phase", ["state_open", "seal"])
def test_manifest_close_quarantine_retains_startup_authorities_and_primary(
    tmp_path, monkeypatch, failure_phase,
):
    import factor_engine.runtime.resume_validation as resume_validation

    root = tmp_path / "artifacts"
    broker = FakeBroker()

    class CloseQuarantinedManifest:
        input_complete = True

        def __len__(self):
            return 0

        def records(self, **kwargs):
            return iter(())

        def close(self):
            raise WorkerQuarantined("manifest close cannot prove retirement", pid=None)

    manifest = CloseQuarantinedManifest()
    monkeypatch.setattr(
        pipeline.FiniteFactorManifest, "ingest_supervised",
        lambda *_args, **_kwargs: manifest,
    )
    def fail(*args, **kwargs):
        raise ValueError("deliberate startup failure")

    if failure_phase == "state_open":
        monkeypatch.setattr(pipeline, "PersistentRunState", fail)
    else:
        monkeypatch.setattr(resume_validation, "manifest_seal_payload", fail)

    with pytest.raises(ValueError, match="deliberate startup failure") as caught:
        pipeline.execute_run_many_durable(
            None, [], policy=resolve_default_policy(), artifact_root=root,
            run_kwargs={"broker": broker},
        )

    primary = caught.value
    assert len(primary.cleanup_errors) == 1
    assert isinstance(primary.cleanup_errors[0], WorkerQuarantined)
    assert primary.run_manifest is manifest
    assert primary.broker is broker
    assert primary.coordinator_lock.owned is True
    assert hasattr(primary, "run_state") is (failure_phase == "seal")
    assert (root / _only_run_dir(root).name / "state.sqlite3").exists() is (failure_phase == "seal")

    challenger = pipeline._RunCoordinatorLock(_only_run_dir(root))
    try:
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            challenger.acquire(allow_dead_owner=False)
    finally:
        primary.coordinator_lock.release()

    challenger.acquire(allow_dead_owner=False)
    challenger.release()


@pytest.mark.parametrize("has_primary", [False, True])
def test_state_close_quarantine_does_not_close_manifest_or_release_lock(
    tmp_path, monkeypatch, has_primary,
):
    from factor_engine.runtime import resume_validation

    real_state = pipeline.PersistentRunState
    broker = FakeBroker()

    class HeldState(real_state):
        def close(self):
            raise WorkerQuarantined("state close uncertain", pid=None)

    class Manifest:
        input_complete = True
        input_error = None

        def __len__(self):
            return 0

        def records(self, **kwargs):
            return iter(())

        def close(self):
            pytest.fail("manifest close must not follow uncertain state close")

    manifest = Manifest()

    def seal(*args, **kwargs):
        if has_primary:
            raise ValueError("original seal error")
        return {}

    monkeypatch.setattr(pipeline, "PersistentRunState", HeldState)
    monkeypatch.setattr(pipeline.FiniteFactorManifest, "ingest_supervised", lambda *a, **k: manifest)
    monkeypatch.setattr(resume_validation, "manifest_seal_payload", seal)
    with pytest.raises(ValueError if has_primary else WorkerQuarantined) as caught:
        pipeline.execute_run_many_durable(
            None, [], policy=resolve_default_policy(), artifact_root=tmp_path,
            run_kwargs={"broker": broker},
        )
    error = caught.value
    try:
        assert error.cleanup_pending
        assert error.run_manifest is manifest
        assert error.broker is broker
        assert error.coordinator_lock.owned
        assert error.run_state.counts() == {}
        if has_primary:
            assert str(error) == "original seal error"
            assert isinstance(error.cleanup_errors[0], WorkerQuarantined)
        challenger = pipeline._RunCoordinatorLock(_only_run_dir(tmp_path))
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            challenger.acquire(allow_dead_owner=False)
    finally:
        real_state.close(error.run_state)
        error.coordinator_lock.release()
