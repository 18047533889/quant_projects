from __future__ import annotations

import hashlib
import json
import time
import uuid

import pytest

import factor_engine.runtime.supervised_worker as supervised
from factor_engine.runtime.worker_ownership import (
    RunOwnershipContext,
    WorkerOwnershipError,
    validate_all_workers_exited,
)


def _ownership_context(run_dir) -> RunOwnershipContext:
    run_id = uuid.uuid4().hex
    payload = json.dumps({"run_id": run_id}, separators=(",", ":")).encode()
    (run_dir / "identity.json").write_bytes(payload)
    return RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())


def _journal_events(run_dir):
    return [
        json.loads(line)["event"]
        for line in (run_dir / "worker-ownership.jsonl").read_text().splitlines()
    ]


def _events_observed_by_worker(run_dir):
    return _journal_events(run_dir)


def _pause(seconds):
    time.sleep(seconds)


def _second():
    return "second"


def _owned_worker(run_dir, context, **kwargs):
    exit_observation_seconds = kwargs.pop("exit_observation_seconds", 1)
    return supervised.SupervisedReusableWorker(
        context="fork",
        ownership_run_dir=run_dir,
        ownership_context=context,
        ownership_role="compute",
        exit_observation_seconds=exit_observation_seconds,
        **kwargs,
    )


def test_real_process_is_bound_before_user_dispatch_and_exited_once(tmp_path):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context)

    observed = worker.execute(_events_observed_by_worker, tmp_path, timeout_seconds=2)
    assert observed.value == ["HEADER", "STARTING", "BOUND"]
    assert _journal_events(tmp_path) == ["HEADER", "STARTING", "BOUND"]

    worker.close()
    worker.close()
    assert _journal_events(tmp_path) == ["HEADER", "STARTING", "BOUND", "EXITED"]
    assert validate_all_workers_exited(tmp_path, context=context).all_exited


def test_timeout_records_exit_only_after_real_process_retirement(tmp_path):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(
        tmp_path, context, cancel_grace_seconds=0, exit_observation_seconds=1
    )

    with pytest.raises(supervised.WorkerTimedOut):
        worker.execute(_pause, 2, timeout_seconds=0.01)

    assert not worker._process.is_alive()
    assert validate_all_workers_exited(tmp_path, context=context).all_exited


def test_restart_after_retirement_creates_a_new_owned_instance(tmp_path):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context, cancel_grace_seconds=0)

    with pytest.raises(supervised.WorkerTimedOut):
        worker.execute(_pause, 2, timeout_seconds=0.01)
    first_instance = worker._ownership_instance.instance_id
    assert worker.execute(_second, timeout_seconds=1).value == "second"
    second_instance = worker._ownership_instance.instance_id
    worker.close()

    assert first_instance != second_instance
    assert _journal_events(tmp_path) == [
        "HEADER", "STARTING", "BOUND", "EXITED",
        "STARTING", "BOUND", "EXITED",
    ]
    assert validate_all_workers_exited(tmp_path, context=context).all_exited


def test_cancellation_records_exit_after_real_process_retirement(tmp_path):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context, cancel_grace_seconds=0)
    call = worker.execute_async(_pause, 2, timeout_seconds=5)

    call.cancel_and_retire()

    assert not worker._process.is_alive()
    with pytest.raises(supervised.WorkerCancelled):
        call.result()
    assert validate_all_workers_exited(tmp_path, context=context).all_exited


def test_start_journal_failure_prevents_spawn(tmp_path, monkeypatch):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context)

    def fail_start(*args, **kwargs):
        raise WorkerOwnershipError("injected STARTING failure")

    monkeypatch.setattr(supervised, "start_worker", fail_start)
    with pytest.raises(WorkerOwnershipError, match="STARTING failure"):
        worker.execute(lambda: None, timeout_seconds=1)
    assert worker._process is None


def test_invalid_progress_configuration_does_not_poison_admission_lock():
    worker = supervised.SupervisedReusableWorker(context="fork")
    with pytest.raises(ValueError, match="expected_progress_ordinals"):
        worker.execute(_second, timeout_seconds=1, expected_progress_ordinals=[])

    assert worker.execute(_second, timeout_seconds=1).value == "second"
    worker.close()


def test_process_constructor_failure_closes_both_pipe_endpoints_and_is_pre_spawn(
    tmp_path,
):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context)

    class Endpoint:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    parent, child = Endpoint(), Endpoint()

    class ConstructorFailureContext:
        def get_start_method(self):
            return "fork"

        def Pipe(self):
            return parent, child

        def Process(self, **kwargs):
            raise RuntimeError("injected process constructor failure")

    worker._context = ConstructorFailureContext()
    with pytest.raises(RuntimeError, match="constructor failure"):
        worker.execute(_second, timeout_seconds=1)

    assert parent.closed and child.closed
    assert worker._process is None
    assert not worker.quarantined
    assert _journal_events(tmp_path) == ["HEADER", "STARTING"]
    assert not validate_all_workers_exited(tmp_path, context=context).all_exited


def test_bind_journal_failure_retires_child_and_leaves_starting_incomplete(
    tmp_path, monkeypatch
):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context)

    def fail_bind(*args, **kwargs):
        raise WorkerOwnershipError("injected BOUND failure")

    monkeypatch.setattr(supervised, "bind_worker", fail_bind)
    with pytest.raises(supervised.WorkerQuarantined, match="binding failed"):
        worker.execute(lambda: None, timeout_seconds=1)

    assert worker.quarantined
    assert not worker._process.is_alive()
    assert _journal_events(tmp_path) == ["HEADER", "STARTING"]
    assert not validate_all_workers_exited(tmp_path, context=context).all_exited


def test_bind_failure_reports_unproven_retirement_details(tmp_path, monkeypatch):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context)

    monkeypatch.setattr(
        supervised,
        "bind_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            WorkerOwnershipError("injected BOUND failure")
        ),
    )
    monkeypatch.setattr(
        worker,
        "_retire",
        lambda: (_ for _ in ()).throw(
            supervised.WorkerQuarantined("injected retirement uncertainty", pid=1)
        ),
    )
    try:
        with pytest.raises(
            supervised.WorkerQuarantined,
            match="retirement could not be proven.*injected retirement uncertainty",
        ) as raised:
            worker.execute(_second, timeout_seconds=1)
        assert isinstance(raised.value.__cause__, supervised.WorkerQuarantined)
        assert worker.quarantined
    finally:
        if worker._process is not None and worker._process.is_alive():
            worker._process.terminate()
            worker._process.join(2)
        if worker._connection is not None:
            worker._connection.close()


def test_exit_journal_failure_quarantines_and_keeps_bound_journal_incomplete(
    tmp_path, monkeypatch
):
    context = _ownership_context(tmp_path)
    worker = _owned_worker(tmp_path, context, cancel_grace_seconds=0)
    worker.start()

    def fail_exit(*args, **kwargs):
        raise WorkerOwnershipError("injected EXITED failure")

    monkeypatch.setattr(supervised, "mark_worker_exited", fail_exit)
    with pytest.raises(supervised.WorkerQuarantined, match="exit recording failed"):
        worker.close()

    assert worker.quarantined
    assert not worker._process.is_alive()
    assert _journal_events(tmp_path) == ["HEADER", "STARTING", "BOUND"]
    assert not validate_all_workers_exited(tmp_path, context=context).all_exited


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ownership_run_dir": "/tmp"},
        {"ownership_context": object()},
        {"ownership_role": ""},
    ],
)
def test_ownership_configuration_is_explicit_and_validated(kwargs):
    with pytest.raises(ValueError):
        supervised.SupervisedReusableWorker(**kwargs)
