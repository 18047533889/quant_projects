import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from factor_engine.runtime.worker_ownership import RunOwnershipContext
from factor_engine.runtime.supervised_worker import WorkerQuarantined
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
import factor_engine.runtime.finite_manifest as module


def _run_context(run_dir):
    import hashlib

    run_id = "a" * 32
    payload = json.dumps({"run_id": run_id}, separators=(",", ":")).encode()
    (run_dir / "identity.json").write_bytes(payload)
    return RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())


def _events(run_dir):
    return [json.loads(line)["event"] for line in (
        run_dir / "worker-ownership.jsonl"
    ).read_text().splitlines()]


@pytest.mark.parametrize("factors,expected", [([], 0), ([{"name": "alpha"}], 1)])
def test_owned_empty_and_normal_ingestion_are_bound_and_exited(tmp_path, factors, expected):
    context = _run_context(tmp_path)
    manifest = FiniteFactorManifest.ingest_supervised(
        factors, tmp_path / "manifest.sqlite3", process_context="fork",
        ownership_run_dir=tmp_path, ownership_context=context,
        deadline_seconds=2,
    )
    try:
        assert len(manifest) == expected
        assert _events(tmp_path) == ["HEADER", "STARTING", "BOUND", "EXITED"]
    finally:
        manifest.close()


def test_unconfigured_ingestion_keeps_legacy_behavior_without_ownership_journal(tmp_path):
    manifest = FiniteFactorManifest.ingest_supervised(
        [{"name": "alpha"}], tmp_path / "manifest.sqlite3",
        process_context="fork", deadline_seconds=2,
    )
    try:
        assert len(manifest) == 1
        assert not (tmp_path / "worker-ownership.jsonl").exists()
    finally:
        manifest.close()


class ResistantInput:
    def __init__(self, marker):
        self.marker = marker

    def __iter__(self):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(self.marker).write_text(str(os.getpid()))
        yield {"name": "before-block"}
        while True:
            time.sleep(.05)


def test_owned_timeout_kills_resistant_ingestion_before_exited(tmp_path):
    context = _run_context(tmp_path)
    marker = tmp_path / "entered"
    manifest = FiniteFactorManifest.ingest_supervised(
        ResistantInput(marker), tmp_path / "manifest.sqlite3", process_context="fork",
        ownership_run_dir=tmp_path, ownership_context=context,
        deadline_seconds=.3,
    )
    try:
        assert marker.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
        assert manifest.input_complete is False
        assert _events(tmp_path) == ["HEADER", "STARTING", "BOUND", "EXITED"]
    finally:
        manifest.close()


def test_bind_failure_never_releases_child_into_iterator(tmp_path, monkeypatch):
    import factor_engine.runtime.worker_ownership as ownership

    context = _run_context(tmp_path)
    marker = tmp_path / "entered"
    monkeypatch.setattr(
        ownership, "bind_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )
    with pytest.raises(WorkerQuarantined) as caught:
        FiniteFactorManifest.ingest_supervised(
            ResistantInput(marker), tmp_path / "manifest.sqlite3", process_context="fork",
            ownership_run_dir=tmp_path, ownership_context=context,
            deadline_seconds=1,
        )
    assert not marker.exists()
    assert caught.value.ingestion_process is not None
    assert _events(tmp_path) == ["HEADER", "STARTING"]
    process = caught.value.ingestion_process
    assert not process.is_alive()
    for endpoint in (
        caught.value.ingestion_parent, caught.value.ingestion_child,
        caught.value.ownership_gate_parent, caught.value.ownership_gate_child,
    ):
        endpoint.close()
    process.close()


def test_gate_send_failure_retires_and_records_exit_without_iterator_entry(
    tmp_path, monkeypatch,
):
    context = _run_context(tmp_path)
    marker = tmp_path / "entered"
    real_context = mp.get_context("fork")
    pipe_calls = 0

    class BrokenSender:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def send_bytes(self, _payload):
            raise OSError("gate send failed")

        def __getattr__(self, name):
            return getattr(self.endpoint, name)

    class ContextProxy:
        def Pipe(self, duplex=True):
            nonlocal pipe_calls
            pipe_calls += 1
            receiver, sender = real_context.Pipe(duplex=duplex)
            return (receiver, BrokenSender(sender)) if pipe_calls == 2 else (receiver, sender)

        def Process(self, *args, **kwargs):
            return real_context.Process(*args, **kwargs)

    monkeypatch.setattr(module.mp, "get_context", lambda *_args: ContextProxy())
    with pytest.raises(OSError, match="gate send failed"):
        FiniteFactorManifest.ingest_supervised(
            ResistantInput(marker), tmp_path / "manifest.sqlite3", process_context="fork",
            ownership_run_dir=tmp_path, ownership_context=context,
            deadline_seconds=1,
        )
    assert not marker.exists()
    assert _events(tmp_path) == ["HEADER", "STARTING", "BOUND", "EXITED"]


@pytest.mark.parametrize("start_failure", [
    OSError("partial spawn boundary"),
    KeyboardInterrupt("partial spawn interrupted"),
], ids=["os-error", "keyboard-interrupt"])
def test_partial_spawn_quarantine_retains_exact_process_and_all_pipes(
    tmp_path, monkeypatch, start_failure,
):
    context = _run_context(tmp_path)

    class Endpoint:
        closed = False

        def close(self):
            self.closed = True

    endpoints = [Endpoint() for _ in range(4)]

    class PartialSpawn:
        pid = None
        exitcode = None
        _popen = None

        def start(self):
            raise start_failure

    process = PartialSpawn()

    class Context:
        calls = 0

        def Pipe(self, duplex=True):
            result = tuple(endpoints[self.calls * 2:self.calls * 2 + 2])
            self.calls += 1
            return result

        def Process(self, *args, **kwargs):
            return process

    monkeypatch.setattr(module.mp, "get_context", lambda *_args: Context())
    with pytest.raises(WorkerQuarantined) as caught:
        FiniteFactorManifest.ingest_supervised(
            [], tmp_path / "manifest.sqlite3", process_context="fork",
            ownership_run_dir=tmp_path, ownership_context=context,
            deadline_seconds=1,
        )
    failure = caught.value
    assert failure.ingestion_process is process
    assert failure.ingestion_parent is endpoints[0]
    assert failure.ingestion_child is endpoints[1]
    assert failure.ownership_gate_child is endpoints[2]
    assert failure.ownership_gate_parent is endpoints[3]
    assert not any(endpoint.closed for endpoint in endpoints)
    assert failure.ownership_instance is not None
    assert _events(tmp_path) == ["HEADER", "STARTING"]


def test_exit_journal_quarantine_is_not_retried_or_masked(tmp_path, monkeypatch):
    import factor_engine.runtime.worker_ownership as ownership

    context = _run_context(tmp_path)
    attempts = []
    journal_failure = RuntimeError("EXITED append failed")

    def fail_exit(*_args, **_kwargs):
        attempts.append(True)
        raise journal_failure

    monkeypatch.setattr(ownership, "mark_worker_exited", fail_exit)
    with pytest.raises(WorkerQuarantined, match="exit journal") as caught:
        FiniteFactorManifest.ingest_supervised(
            [{"name": "alpha"}], tmp_path / "manifest.sqlite3",
            process_context="fork", ownership_run_dir=tmp_path,
            ownership_context=context, deadline_seconds=2,
        )
    failure = caught.value
    assert attempts == [True]
    assert failure.__cause__ is journal_failure
    assert failure.ingestion_process is not None
    assert _events(tmp_path) == ["HEADER", "STARTING", "BOUND"]
    process = failure.ingestion_process
    assert not process.is_alive()
    for endpoint in (
        failure.ingestion_parent, failure.ingestion_child,
        failure.ownership_gate_parent, failure.ownership_gate_child,
    ):
        if endpoint is not None:
            endpoint.close()
    process.close()
