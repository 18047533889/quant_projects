import gc
import pickle
import time
import weakref

import pytest

from factor_engine.runtime.supervised_worker import (
    SupervisedReusableWorker, WorkerTimedOut, WorkerQuarantined, WorkerTransportFailed,
    _worker_loop,
)


def _double(value):
    return value * 2


def _pause(seconds):
    time.sleep(seconds)


def _fail():
    raise ValueError("task failed")


def _exit_before_ack():
    import os
    os._exit(23)


def _user_eof():
    raise EOFError("user function, not broken transport")


class Lease:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def test_success_releases_lease_after_response():
    lease = Lease()
    worker = SupervisedReusableWorker(context="fork")
    assert worker.execute(_double, 2, timeout_seconds=1, lease=lease).value == 4
    assert lease.released == 1
    worker.close()


def test_spawn_failure_before_process_exists_releases_transferred_lease():
    # A local callable is intentionally not spawn-serializable. No OS worker
    # owns the admission after this failure and cleanup must remain idempotent.
    worker = SupervisedReusableWorker(context="spawn", function=lambda x: x)
    lease = Lease()
    with pytest.raises((AttributeError, TypeError), match="pickle|local"):
        worker.execute(1, timeout_seconds=1, lease=lease)
    assert worker._process is None
    assert lease.released == 1
    worker.close()
    assert lease.released == 1


def test_pid_none_after_entering_start_boundary_is_quarantined(monkeypatch):
    worker = SupervisedReusableWorker(context="spawn")
    lease = Lease()

    class Endpoint:
        def close(self):
            pass

    class AmbiguousProcess:
        pid = None
        _popen = None

        def start(self):
            # Models CPython Popen.__init__ raising after its internal spawn but
            # before BaseProcess can assign process._popen.
            raise OSError("ambiguous post-spawn failure")

    class AmbiguousContext:
        def get_start_method(self):
            return "spawn"

        def Pipe(self):
            return Endpoint(), Endpoint()

        def Process(self, **kwargs):
            return AmbiguousProcess()

    worker._context = AmbiguousContext()
    with pytest.raises(WorkerQuarantined, match="OS-spawn boundary"):
        worker.execute(_double, 1, timeout_seconds=1, lease=lease)
    assert worker.quarantined
    assert worker._process is not None
    assert worker._process.pid is None
    assert lease.released == 0


def test_startup_quarantine_never_releases_transferred_lease(monkeypatch):
    worker = SupervisedReusableWorker()
    lease = Lease()

    def failed_start():
        worker.quarantined = True
        raise WorkerQuarantined("startup retirement unproven", pid=123)

    monkeypatch.setattr(worker, "start", failed_start)
    with pytest.raises(WorkerQuarantined):
        worker.execute(_double, 1, timeout_seconds=1, lease=lease)
    assert lease.released == 0


def test_task_failure_releases_lease_after_response():
    lease = Lease()
    worker = SupervisedReusableWorker(context="fork")
    with pytest.raises(ValueError, match="task failed"):
        worker.execute(_fail, timeout_seconds=1, lease=lease)
    assert lease.released == 1
    assert worker.execute(_double, 2, timeout_seconds=1).value == 4
    worker.close()


def test_broken_pipe_is_typed_only_after_worker_exit():
    worker = SupervisedReusableWorker(context="spawn")
    lease = Lease()
    with pytest.raises(WorkerTransportFailed) as caught:
        worker.execute(_exit_before_ack, timeout_seconds=5, lease=lease)
    assert isinstance(caught.value.__cause__, EOFError)
    assert not worker._process.is_alive()
    assert lease.released == 1
    worker.close()
    assert lease.released == 1


def test_remote_user_eof_is_not_misclassified_as_exited_worker():
    worker = SupervisedReusableWorker(context="spawn")
    try:
        with pytest.raises(EOFError, match="user function"):
            worker.execute(_user_eof, timeout_seconds=5)
        assert worker._process.is_alive()
        assert worker.execute(_double, 2, timeout_seconds=5).value == 4
    finally:
        worker.close()


def test_worker_reuses_generation_and_returns_real_process_result():
    worker = SupervisedReusableWorker(context="fork")
    first = worker.execute(_double, 2, timeout_seconds=1)
    second = worker.execute(_double, 3, timeout_seconds=1)
    assert (first.value, second.value) == (4, 6)
    assert first.generation == second.generation
    worker.close()


def test_timeout_releases_lease_only_after_observed_process_exit():
    lease = Lease()
    worker = SupervisedReusableWorker(
        context="fork", cancel_grace_seconds=0.01, exit_observation_seconds=1
    )
    with pytest.raises(WorkerTimedOut):
        worker.execute(_pause, 2, timeout_seconds=0.01, lease=lease)
    assert lease.released == 1
    assert not worker._process.is_alive()
    worker.close()
    assert lease.released == 1


def test_late_completion_cannot_poison_next_generation():
    worker = SupervisedReusableWorker(context="fork", cancel_grace_seconds=0.2,
                                     exit_observation_seconds=1)
    with pytest.raises(WorkerTimedOut):
        worker.execute(_pause, 0.06, timeout_seconds=0.02)
    assert not worker._process.is_alive()
    assert worker.execute(_double, 4, timeout_seconds=1).value == 8
    worker.close()


def _unserializable_failure():
    error = ValueError("failed")
    error.callback = lambda: None
    raise error


def test_unserializable_exception_is_reported_and_worker_reusable():
    worker = SupervisedReusableWorker(context="fork")
    with pytest.raises(RuntimeError, match="could not be serialized: ValueError"):
        worker.execute(_unserializable_failure, timeout_seconds=1)
    assert worker.execute(_double, 5, timeout_seconds=1).value == 10
    worker.close()


class Payload:
    pass


_last_payload = None


def _remember_payload(value):
    global _last_payload
    _last_payload = weakref.ref(value)


def _previous_payload_released():
    return _last_payload() is None


def test_decoded_request_does_not_retain_previous_payload():
    worker = SupervisedReusableWorker(context="fork")
    worker.execute(_remember_payload, Payload(), timeout_seconds=1)
    assert worker.execute(_previous_payload_released, timeout_seconds=1).value is True
    worker.close()


def test_serialized_request_is_released_before_function_lifecycle():
    class ScriptedConnection:
        def __init__(self, request):
            self.requests = [request, None]
            self.responses = []

        def recv(self):
            return self.requests.pop(0)

        def send(self, response):
            self.responses.append(response)

    serialized = memoryview(pickle.dumps(((), {})))
    assert serialized.nbytes > 0
    serialized_ref = weakref.ref(serialized)
    connection = ScriptedConnection(("request", serialized))
    del serialized
    observed = []

    def oracle():
        gc.collect()
        observed.append(serialized_ref() is None)
        return "ok"

    _worker_loop(connection, "generation", oracle)

    assert observed == [True]
    assert pickle.loads(connection.responses[0][3]) == "ok"


class SlowSend:
    def __init__(self, connection):
        self.connection = connection

    def send(self, payload):
        time.sleep(0.2)
        raise BrokenPipeError("blocked send finally ended")

    def close(self):
        self.connection.close()


def test_blocked_transport_is_finite_and_keeps_lease_until_thread_exit():
    worker = SupervisedReusableWorker(context="fork", cancel_grace_seconds=0,
                                     exit_observation_seconds=0.02)
    worker.start()
    worker._connection = SlowSend(worker._connection)
    lease = Lease()
    before = time.monotonic()
    with pytest.raises(WorkerQuarantined):
        worker.execute(_double, 3, timeout_seconds=0.01, lease=lease)
    assert time.monotonic() - before < 0.15
    assert lease.released == 0
    with pytest.raises(WorkerQuarantined):
        worker.execute(_double, 3, timeout_seconds=1)
    worker._transport.join(1)
    worker.close()
    assert lease.released == 0


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_nonfinite_waits_rejected(timeout):
    worker = SupervisedReusableWorker()
    with pytest.raises(ValueError):
        worker.execute(_double, 1, timeout_seconds=timeout)
    assert worker._process is None


def test_active_guard_registration_failure_does_not_leave_untracked_worker(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module

    def failed_registration(pid):
        raise RuntimeError("active guardian registration failed")

    monkeypatch.setattr(broker_module, "register_heavy_worker", failed_registration)
    worker = SupervisedReusableWorker(context="fork", exit_observation_seconds=1)
    with pytest.raises(RuntimeError, match="active guardian registration failed"):
        worker.execute(_double, 1, timeout_seconds=1)
    assert not worker._process.is_alive()
    worker.close()


class ClosableCallable:
    def __init__(self, marker, pause=0):
        self.marker = marker
        self.pause = pause

    def __call__(self, value):
        return value * 2

    def close(self):
        from pathlib import Path
        time.sleep(self.pause)
        Path(self.marker).write_text("closed")


def test_normal_close_runs_worker_owned_cleanup_before_exit(tmp_path):
    marker = tmp_path / "closed"
    worker = SupervisedReusableWorker(context="fork", function=ClosableCallable(str(marker)),
                                     cancel_grace_seconds=1, exit_observation_seconds=1)
    assert worker.execute(4, timeout_seconds=1).value == 8
    worker.close()
    assert marker.read_text() == "closed"
    assert not worker._process.is_alive()


def test_blocking_worker_cleanup_is_still_finite(tmp_path):
    marker = tmp_path / "closed"
    worker = SupervisedReusableWorker(context="fork", function=ClosableCallable(str(marker), 2),
                                     cancel_grace_seconds=0.01, exit_observation_seconds=1)
    worker.execute(4, timeout_seconds=1)
    before = time.monotonic()
    worker.close()
    assert time.monotonic() - before < 1.5
    assert not worker._process.is_alive()
    assert not marker.exists()
