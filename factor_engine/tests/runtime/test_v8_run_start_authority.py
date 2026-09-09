from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.supervised_worker import WorkerQuarantined
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker, FakeEngine


def _only_run_dir(root: Path) -> Path:
    runs = [path for path in root.iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def test_new_run_lock_is_held_during_ingestion_and_released_on_ordinary_failure(
    tmp_path, monkeypatch,
):
    root = tmp_path / "artifacts"
    observed = {}

    def fail_ingestion(_factors, path, **_limits):
        run_dir = Path(path).parent
        challenger = pipeline._RunCoordinatorLock(run_dir)
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            challenger.acquire(allow_dead_owner=False)
        observed["run_dir"] = run_dir
        raise ValueError("deliberate ordinary ingestion failure")

    monkeypatch.setattr(
        pipeline.FiniteFactorManifest, "ingest_supervised", fail_ingestion,
    )
    with pytest.raises(ValueError, match="ordinary ingestion failure"):
        pipeline.execute_run_many_durable(
            FakeEngine(), [], policy=resolve_default_policy(), artifact_root=root,
            run_kwargs={"broker": FakeBroker()},
        )

    run_dir = observed["run_dir"]
    assert run_dir == _only_run_dir(root)
    replacement = pipeline._RunCoordinatorLock(run_dir)
    replacement.acquire(allow_dead_owner=False)
    try:
        assert replacement.owned is True
    finally:
        replacement.release()


def test_ingestion_quarantine_hands_off_owned_lock_and_broker_only(
    tmp_path, monkeypatch,
):
    root = tmp_path / "artifacts"
    broker = FakeBroker()

    def quarantine(_factors, _path, **_limits):
        raise WorkerQuarantined("fake ingestion retirement uncertainty", pid=None)

    monkeypatch.setattr(
        pipeline.FiniteFactorManifest, "ingest_supervised", quarantine,
    )
    with pytest.raises(WorkerQuarantined) as caught:
        pipeline.execute_run_many_durable(
            FakeEngine(), [], policy=resolve_default_policy(), artifact_root=root,
            run_kwargs={"broker": broker},
        )

    failure = caught.value
    run_dir = _only_run_dir(root)
    assert failure.coordinator_lock.owned is True
    assert failure.coordinator_lock.path == run_dir / "coordinator.lock"
    assert failure.broker is broker
    assert not hasattr(failure, "run_manifest")
    assert not hasattr(failure, "run_state")

    challenger = pipeline._RunCoordinatorLock(run_dir)
    try:
        with pytest.raises(RuntimeError, match="coordinator is still alive"):
            challenger.acquire(allow_dead_owner=False)
    finally:
        failure.coordinator_lock.release()

    challenger.acquire(allow_dead_owner=False)
    challenger.release()
