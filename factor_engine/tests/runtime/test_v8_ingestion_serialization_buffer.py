from __future__ import annotations

import multiprocessing as mp

import pytest

from factor_engine.runtime import finite_manifest as manifest_module


class FakeLease:
    def __init__(self, fail=False): self.released, self.fail = False, fail
    def release(self):
        if self.fail:
            raise RuntimeError("injected lease release failure")
        self.released = True


class FakeBroker:
    def __init__(self, admitted=True, release_fails=False):
        self.admitted, self.calls, self.lease = admitted, [], FakeLease(release_fails)
    def acquire_memory(self, kind, nbytes, *, lease_id=""):
        self.calls.append((str(kind), nbytes, lease_id))
        return self.lease if self.admitted else None


def test_real_spawn_pickleable_list_releases_controlled_buffer_lease(tmp_path):
    broker = FakeBroker()
    manifest = manifest_module.FiniteFactorManifest.ingest_supervised(
        [{"name": f"factor-{i}"} for i in range(3)], tmp_path / "manifest.sqlite3",
        process_context="spawn", deadline_seconds=5, serialization_broker=broker,
    )
    try:
        assert len(manifest) == 3
        assert [row.name for row in manifest.records()] == ["factor-0", "factor-1", "factor-2"]
    finally:
        manifest.close()
    assert broker.calls[0][1] == 2 * (131_072 + 1) + 64 * 1024
    assert broker.lease.released


def test_denied_buffer_admission_does_not_create_process(tmp_path, monkeypatch):
    broker = FakeBroker(admitted=False)
    real_context = mp.get_context("spawn")
    monkeypatch.setattr(real_context, "Process", lambda *_a, **_k: pytest.fail("must not create process"))
    monkeypatch.setattr(manifest_module.mp, "get_context", lambda _name: real_context)
    with pytest.raises(manifest_module.ManifestError, match="admission was denied"):
        manifest_module.FiniteFactorManifest.ingest_supervised(
            [], tmp_path / "manifest.sqlite3", process_context="spawn",
            serialization_broker=broker,
        )
    assert len(broker.calls) == 1


def test_oversize_definition_uses_observed_lower_bound(tmp_path):
    payload, size, oversize = manifest_module._bounded_pickle(
        {"name": "large", "payload": b"x" * 4096}, 128
    )
    assert (payload, size, oversize) == (None, 129, True)
    manifest = manifest_module.FiniteFactorManifest.ingest(
        [{"name": "large", "payload": b"x" * 4096}],
        tmp_path / "manifest.sqlite3", max_definition_bytes=128,
    )
    try:
        record = next(manifest.records())
        assert record.error_code == "DEFINITION_TOO_LARGE"
        assert record.definition_bytes == 129
    finally:
        manifest.close()


def test_release_failure_retains_exact_lease_authority(tmp_path):
    broker = FakeBroker(release_fails=True)
    with pytest.raises(RuntimeError, match="lease release failure") as caught:
        manifest_module.FiniteFactorManifest.ingest_supervised(
            [{"name": "factor"}], tmp_path / "manifest.sqlite3",
            process_context="spawn", deadline_seconds=5,
            serialization_broker=broker,
        )
    assert caught.value.cleanup_pending is True
    assert caught.value.serialization_buffer_lease is broker.lease
    assert not broker.lease.released


def test_endpoint_close_failure_after_worker_exit_still_releases_lease(
    tmp_path, monkeypatch,
):
    broker = FakeBroker()
    real_context = mp.get_context("spawn")
    processes = []

    class FailingCloseEndpoint:
        def __init__(self, endpoint): self.endpoint = endpoint
        def poll(self, *args): return self.endpoint.poll(*args)
        def recv(self): return self.endpoint.recv()
        def close(self):
            self.endpoint.close()
            raise RuntimeError("injected endpoint close failure")

    class ContextProxy:
        def Process(self, *args, **kwargs):
            process = real_context.Process(*args, **kwargs)
            processes.append(process)
            return process
        def Pipe(self, *args, **kwargs):
            parent, child = real_context.Pipe(*args, **kwargs)
            return FailingCloseEndpoint(parent), child

    monkeypatch.setattr(manifest_module.mp, "get_context", lambda _name: ContextProxy())
    with pytest.raises(RuntimeError, match="endpoint close failure"):
        manifest_module.FiniteFactorManifest.ingest_supervised(
            [{"name": "factor"}], tmp_path / "manifest.sqlite3",
            process_context="spawn", deadline_seconds=5,
            serialization_broker=broker,
        )
    assert broker.lease.released
    assert len(processes) == 1
    assert processes[0].exitcode is not None
    assert not processes[0].is_alive()


def test_primary_error_survives_endpoint_close_failure(tmp_path, monkeypatch):
    import factor_engine.runtime.resource_broker as resource_broker

    broker = FakeBroker()
    real_context = mp.get_context("spawn")

    class FailingCloseEndpoint:
        def __init__(self, endpoint): self.endpoint = endpoint
        def close(self):
            self.endpoint.close()
            raise RuntimeError("injected endpoint cleanup failure")
        def __getattr__(self, name): return getattr(self.endpoint, name)

    class ContextProxy:
        Process = real_context.Process
        def Pipe(self, *args, **kwargs):
            parent, child = real_context.Pipe(*args, **kwargs)
            return FailingCloseEndpoint(parent), child

    monkeypatch.setattr(manifest_module.mp, "get_context", lambda _name: ContextProxy())
    monkeypatch.setattr(
        resource_broker, "register_heavy_worker",
        lambda _pid: (_ for _ in ()).throw(ValueError("original registration failure")),
    )
    with pytest.raises(ValueError, match="original registration failure") as caught:
        manifest_module.FiniteFactorManifest.ingest_supervised(
            [{"name": "factor"}], tmp_path / "manifest.sqlite3",
            process_context="spawn", serialization_broker=broker,
        )
    assert any("endpoint cleanup failure" in str(error)
               for error in caught.value.cleanup_errors)
    assert broker.lease.released


def test_primary_error_survives_release_failure_with_exact_authority(tmp_path, monkeypatch):
    import factor_engine.runtime.resource_broker as resource_broker

    broker = FakeBroker(release_fails=True)
    monkeypatch.setattr(
        resource_broker, "register_heavy_worker",
        lambda _pid: (_ for _ in ()).throw(ValueError("original registration failure")),
    )
    with pytest.raises(ValueError, match="original registration failure") as caught:
        manifest_module.FiniteFactorManifest.ingest_supervised(
            [{"name": "factor"}], tmp_path / "manifest.sqlite3",
            process_context="spawn", serialization_broker=broker,
        )
    assert caught.value.cleanup_pending is True
    assert caught.value.serialization_buffer_lease is broker.lease
    assert any("lease release failure" in str(error)
               for error in caught.value.cleanup_errors)


@pytest.mark.parametrize("invalid", [True, 1.5])
def test_definition_limit_type_is_rejected_before_context(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(manifest_module.mp, "get_context", lambda _name: pytest.fail("too late"))
    with pytest.raises(ValueError, match="positive integer"):
        manifest_module.FiniteFactorManifest.ingest_supervised(
            [], tmp_path / "manifest.sqlite3", process_context="spawn",
            max_definition_bytes=invalid,
        )
