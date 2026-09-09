import math
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.supervised_worker import WorkerQuarantined


class Endpoint:
    closed = False

    def close(self):
        self.closed = True


class StuckProcess:
    pid = 123456789
    exitcode = None

    def __init__(self):
        self.actions = []

    def start(self):
        self.actions.append("start")

    def join(self, timeout):
        self.actions.append(("join", timeout))

    def is_alive(self):
        return True

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")


def test_unproven_ingest_retirement_never_opens_manifest(tmp_path, monkeypatch):
    import factor_engine.runtime.finite_manifest as module
    import factor_engine.runtime.resource_broker as broker_module
    parent, child, process = Endpoint(), Endpoint(), StuckProcess()

    class Context:
        def Pipe(self, **kwargs):
            return parent, child

        def Process(self, **kwargs):
            return process

    monkeypatch.setattr(module.mp, "get_context", lambda *_: Context())
    monkeypatch.setattr(broker_module, "register_heavy_worker", lambda *_: None)
    path = tmp_path / "must-not-open.sqlite3"
    with pytest.raises(WorkerQuarantined) as caught:
        FiniteFactorManifest.ingest_supervised([], path, deadline_seconds=0.01)
    assert caught.value.pid == process.pid
    assert caught.value.ingestion_process is process
    assert "terminate" in process.actions and "kill" in process.actions
    assert not path.exists()
    assert parent.closed and child.closed


@pytest.mark.parametrize("partial_popen", [None, object()])
def test_ambiguous_spawn_does_not_reopen_manifest(tmp_path, monkeypatch, partial_popen):
    import factor_engine.runtime.finite_manifest as module
    parent, child = Endpoint(), Endpoint()

    class PartialSpawn:
        pid = None
        exitcode = None
        _popen = partial_popen

        def start(self):
            raise OSError("Popen constructor failed at uncertain spawn boundary")

    class Context:
        def Pipe(self, **kwargs):
            return parent, child

        def Process(self, **kwargs):
            return PartialSpawn()

    monkeypatch.setattr(module.mp, "get_context", lambda *_: Context())
    path = tmp_path / "manifest.sqlite3"
    with pytest.raises(WorkerQuarantined, match="OS-spawn"):
        FiniteFactorManifest.ingest_supervised([], path, deadline_seconds=0.01)
    assert not path.exists()
    assert parent.closed and child.closed


@pytest.mark.parametrize("deadline", [0, -1, math.inf, math.nan, True])
def test_ingestion_deadline_rejected_before_process_creation(tmp_path, monkeypatch, deadline):
    import factor_engine.runtime.finite_manifest as module
    monkeypatch.setattr(module.mp, "get_context", lambda *_: pytest.fail("must reject before spawn"))
    with pytest.raises(ValueError, match="deadline"):
        FiniteFactorManifest.ingest_supervised([], tmp_path / "manifest", deadline_seconds=deadline)


class IgnoreTerminateInput:
    def __init__(self, marker):
        self.marker = marker

    def __iter__(self):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(self.marker).write_text(str(os.getpid()))
        yield {"name": "before_block"}
        while True:
            time.sleep(0.05)


class HugeErrorInput:
    def __iter__(self):
        raise ValueError("测" * 100_000)


def test_large_unicode_input_error_cannot_block_result_pipe(tmp_path):
    started = time.monotonic()
    with FiniteFactorManifest.ingest_supervised(
        HugeErrorInput(), tmp_path / "manifest.sqlite3",
        process_context="fork", deadline_seconds=3,
    ) as manifest:
        assert not manifest.input_complete
        assert manifest.input_error and set(manifest.input_error) == {"测"}
        assert len(manifest.input_error.encode("utf-8")) <= 2048
        assert time.monotonic() - started < 2


def test_real_ingestion_sigterm_resistant_child_is_killed_before_reopen(tmp_path):
    marker = tmp_path / "pid"
    started = time.monotonic()
    manifest = None
    try:
        manifest = FiniteFactorManifest.ingest_supervised(
            IgnoreTerminateInput(str(marker)), tmp_path / "manifest.sqlite3",
            process_context="fork", deadline_seconds=0.5,
        )
        assert marker.exists()
        pid = int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not manifest.input_complete
        assert manifest.input_error == "factor input ingestion deadline exceeded"
        assert len(manifest) == 1
        assert time.monotonic() - started < 5
    finally:
        if manifest is not None:
            manifest.close()
        if marker.exists():
            pid = int(marker.read_text())
            for child in mp.active_children():
                if child.pid == pid:
                    child.kill()
                    child.join(2)
