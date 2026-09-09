import multiprocessing as mp

import threadpoolctl

from factor_engine.runtime import resource_broker
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker
from factor_engine.tests.runtime.test_v8_supervisor_ownership import _ownership_context


def test_owned_native_initialization_waits_for_resource_registration(tmp_path, monkeypatch):
    initialized = mp.get_context("fork").Event()
    context = _ownership_context(tmp_path)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **_: initialized.set())

    def register(_pid):
        assert not initialized.wait(0.25), "native initialization preceded resource registration"

    monkeypatch.setattr(resource_broker, "register_heavy_worker", register)
    worker = SupervisedReusableWorker(
        context="fork", process_environment={"OMP_NUM_THREADS": "1"},
        ownership_run_dir=tmp_path, ownership_context=context,
        cancel_grace_seconds=0, exit_observation_seconds=1,
    )
    try:
        worker.start()
        assert initialized.wait(2)
    finally:
        worker.close()


def test_owned_ingestion_waits_for_resource_registration(tmp_path, monkeypatch):
    entered = mp.get_context("fork").Event()
    context = _ownership_context(tmp_path)

    class Input:
        def __iter__(self):
            entered.set()
            return iter([])

    def register(_pid):
        assert not entered.wait(0.25), "iterator entered before resource registration"

    monkeypatch.setattr(resource_broker, "register_heavy_worker", register)
    manifest = FiniteFactorManifest.ingest_supervised(
        Input(), tmp_path / "manifest.sqlite3", process_context="fork",
        ownership_run_dir=tmp_path, ownership_context=context,
        deadline_seconds=2,
    )
    try:
        assert manifest.input_complete and entered.is_set()
    finally:
        manifest.close()
