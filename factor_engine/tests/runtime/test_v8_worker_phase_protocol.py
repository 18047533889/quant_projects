import time

import pytest

from factor_engine.runtime.supervised_worker import (
    SupervisedReusableWorker,
    WorkerTimedOut,
    WorkerTransportFailed,
    emit_worker_progress,
)


def _emit_sequence(sequence, hang=False):
    for ordinal, phase in sequence:
        emit_worker_progress(phase, ordinal=ordinal)
    if hang:
        while True:
            time.sleep(1)
    return "done"


def _raw_progress(frames):
    import factor_engine.runtime.supervised_worker as module

    connection, request_id, generation, send_lock = module._WORKER_PROGRESS
    for payload in frames:
        with send_lock:
            connection.send((request_id, generation, "progress", payload))
    return "done"


def _worker():
    return SupervisedReusableWorker(
        context="spawn", function=_emit_sequence,
        cancel_grace_seconds=.01, exit_observation_seconds=.5,
    )


def test_unphased_request_rejects_progress_and_retires():
    worker = _worker()
    with pytest.raises(WorkerTransportFailed, match="not assigned"):
        worker.execute([(0, "COMPUTE")], timeout_seconds=2)
    assert not worker._process.is_alive()


@pytest.mark.parametrize("sequence,assigned", [
    ([(1, "COMPUTE")], {0}),
    ([(0, "WRITE")], {0}),
    ([(0, "COMPUTE"), (0, "COMPUTE")], {0}),
    ([(0, "COMPUTE"), (0, "VERIFY_DONE")], {0}),
    ([(0, "COMPUTE"), (0, "WRITE"), (0, "COMPUTE")], {0}),
])
def test_unknown_duplicate_skip_and_reverse_phases_fail_closed(sequence, assigned):
    worker = _worker()
    with pytest.raises(WorkerTransportFailed):
        worker.execute(
            sequence, timeout_seconds=2,
            expected_progress_ordinals=set(assigned),
        )
    assert not worker._process.is_alive()


def test_valid_interleaved_per_ordinal_protocol_and_timeout_snapshot():
    worker = _worker()
    sequence = [
        (0, "COMPUTE"), (1, "COMPUTE"),
        (0, "WRITE"), (0, "VERIFY_DONE"), (1, "WRITE"),
    ]
    with pytest.raises(WorkerTimedOut) as caught:
        worker.execute(
            sequence, True, timeout_seconds=1.5,
            expected_progress_ordinals=frozenset({0, 1}),
        )
    assert dict(caught.value.phases) == {0: "VERIFY_DONE", 1: "WRITE"}
    assert caught.value.stage == "WRITE"
    with pytest.raises(TypeError):
        caught.value.phases[0] = "COMPUTE"
    assert not worker._process.is_alive()


def test_async_cancellation_preserves_immutable_per_ordinal_phases():
    worker = _worker()
    call = worker.execute_async(
        [(0, "COMPUTE"), (1, "COMPUTE"), (0, "WRITE")], True,
        timeout_seconds=5, expected_progress_ordinals={0, 1},
    )
    deadline = time.monotonic() + 1.0
    while not (
        isinstance(worker.last_progress, dict)
        and worker.last_progress.get("phase") == "WRITE"
        and worker.last_progress.get("ordinal") == 0
    ):
        if time.monotonic() >= deadline:
            pytest.fail("worker did not reach WRITE before cancellation")
        time.sleep(0.01)
    call.cancel_and_retire()
    assert dict(call.cancelled_phases) == {0: "WRITE", 1: "COMPUTE"}
    with pytest.raises(TypeError):
        call.cancelled_phases[0] = "VERIFY_DONE"


@pytest.mark.parametrize("timestamps", [
    (10**30,),
    (-1.0,),
    (0.0,),
    (1.0,),
    (2.0, 1.0),
])
def test_stale_future_or_decreasing_worker_monotonic_time_is_rejected(timestamps):
    worker = SupervisedReusableWorker(
        context="spawn", function=_raw_progress,
        cancel_grace_seconds=.01, exit_observation_seconds=.5,
    )
    phases = ["COMPUTE", "WRITE"]
    frames = [
        {"phase": phases[index], "ordinal": 0, "monotonic": stamp}
        for index, stamp in enumerate(timestamps)
    ]
    with pytest.raises(WorkerTransportFailed):
        worker.execute(
            frames, timeout_seconds=2,
            expected_progress_ordinals={0},
        )
    assert not worker._process.is_alive()


@pytest.mark.parametrize("bad", [{True}, {0.0}, [0], {-1}])
def test_progress_assignment_requires_exact_nonnegative_integer_set(bad):
    worker = _worker()
    with pytest.raises(ValueError, match="expected_progress_ordinals"):
        worker.execute([], timeout_seconds=1, expected_progress_ordinals=bad)
