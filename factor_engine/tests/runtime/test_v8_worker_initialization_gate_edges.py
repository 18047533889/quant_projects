import multiprocessing as mp
import os
import time

import pytest
import threadpoolctl

from factor_engine.runtime import supervised_worker as supervised
from factor_engine.tests.runtime.test_v8_supervisor_ownership import _ownership_context


def _assert_pid_exited(pid):
    deadline = time.monotonic() + 2
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_binding_failure_never_initializes_native_runtime(tmp_path, monkeypatch):
    initialized = mp.get_context("fork").Event()
    context = _ownership_context(tmp_path)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **_kwargs: initialized.set())
    monkeypatch.setattr(
        supervised, "bind_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )
    worker = supervised.SupervisedReusableWorker(
        context="fork", process_environment={"OMP_NUM_THREADS": "1"},
        ownership_run_dir=tmp_path, ownership_context=context,
        cancel_grace_seconds=0, exit_observation_seconds=1,
    )
    with pytest.raises(supervised.WorkerQuarantined, match="binding failed"):
        worker.start()
    assert not initialized.wait(.2)
    _assert_pid_exited(worker._process.pid)


@pytest.mark.parametrize("gate", [b"READY:{generation}", b"BOUND:wrong-{generation}"])
def test_forged_or_wrong_generation_gate_never_initializes(gate, monkeypatch):
    context = mp.get_context("fork")
    initialized = context.Event()
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **_kwargs: initialized.set())
    parent, child = context.Pipe()
    generation = "generation-7"
    process = context.Process(
        target=supervised._worker_loop,
        args=(child, generation, None, {"OMP_NUM_THREADS": "1"}, True),
    )
    process.start()
    child.close()
    parent.send_bytes(gate.replace(b"{generation}", generation.encode("ascii")))
    process.join(2)
    try:
        assert process.exitcode not in (None, 0)
        assert not initialized.wait(.2)
    finally:
        if process.is_alive():
            process.kill()
            process.join(1)
        parent.close()
        process.close()


def test_unowned_worker_initializes_without_ownership_gate(monkeypatch):
    initialized = mp.get_context("fork").Event()
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **_kwargs: initialized.set())
    worker = supervised.SupervisedReusableWorker(
        context="fork", process_environment={"OMP_NUM_THREADS": "1"},
        cancel_grace_seconds=0, exit_observation_seconds=1,
    )
    try:
        worker.start()
        assert initialized.wait(2)
    finally:
        worker.close()


def test_gate_send_failure_retires_process_and_preserves_transport_error(
    tmp_path, monkeypatch,
):
    context = _ownership_context(tmp_path)
    worker = supervised.SupervisedReusableWorker(
        context="fork", ownership_run_dir=tmp_path, ownership_context=context,
        cancel_grace_seconds=0, exit_observation_seconds=1,
    )

    class BrokenGateConnection:
        def __init__(self, connection):
            self.connection = connection
            self.closed = False

        def send_bytes(self, _payload):
            raise OSError("gate transport failed")

        def close(self):
            self.closed = True
            return self.connection.close()

        def __getattr__(self, name):
            return getattr(self.connection, name)

    real_context = worker._context
    real_pipe = real_context.Pipe
    wrapped = []

    class ContextProxy:
        def get_start_method(self):
            return real_context.get_start_method()

        def Pipe(self):
            parent, child = real_pipe()
            parent = BrokenGateConnection(parent)
            wrapped.append(parent)
            return parent, child

        def Process(self, *args, **kwargs):
            return real_context.Process(*args, **kwargs)

    worker._context = ContextProxy()
    with pytest.raises(OSError, match="gate transport failed"):
        worker.start()
    assert len(wrapped) == 1 and wrapped[0].closed is True
    _assert_pid_exited(worker._process.pid)
