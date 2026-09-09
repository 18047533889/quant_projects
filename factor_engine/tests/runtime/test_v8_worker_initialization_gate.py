import multiprocessing as mp

import threadpoolctl

from factor_engine.runtime import supervised_worker as supervised
from factor_engine.tests.runtime.test_v8_supervisor_ownership import _ownership_context


def test_owned_worker_native_initialization_waits_for_durable_binding(tmp_path, monkeypatch):
    initialized = mp.get_context("fork").Event()
    context = _ownership_context(tmp_path)
    original_bind = supervised.bind_worker

    def observe_native_initialization(*args, **kwargs):
        initialized.set()

    def bind_after_observation(*args, **kwargs):
        assert not initialized.wait(0.25), "native initialization preceded durable BOUND"
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", observe_native_initialization)
    monkeypatch.setattr(supervised, "bind_worker", bind_after_observation)
    worker = supervised.SupervisedReusableWorker(
        context="fork", process_environment={"OMP_NUM_THREADS": "1"},
        ownership_run_dir=tmp_path, ownership_context=context,
        cancel_grace_seconds=0, exit_observation_seconds=1,
    )
    try:
        worker.start()
        assert initialized.wait(2), "bound worker never initialized"
    finally:
        worker.close()
