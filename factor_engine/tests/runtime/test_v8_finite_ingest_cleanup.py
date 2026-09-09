import sqlite3

import pytest


from factor_engine.runtime import finite_manifest as module
FiniteFactorManifest = module.FiniteFactorManifest


def _capture_connection(monkeypatch):
    connections = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    return connections


def _assert_closed(connection):
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("select 1")


def test_iter_creation_failure_closes_open_manifest(tmp_path, monkeypatch):
    connections = _capture_connection(monkeypatch)
    primary = LookupError("iter creation failed")

    class IterFails:
        def __iter__(self):
            raise primary

    with pytest.raises(LookupError) as caught:
        FiniteFactorManifest.ingest(IterFails(), tmp_path / "manifest.sqlite3")
    assert caught.value is primary
    assert len(connections) == 1
    _assert_closed(connections[0])


def test_iterator_close_failure_does_not_mask_primary(tmp_path, monkeypatch):
    connections = _capture_connection(monkeypatch)
    primary = ValueError("iteration failed")
    cleanup = RuntimeError("iterator close failed")

    class Iterator:
        exited = False

        def __iter__(self):
            return self

        def __next__(self):
            raise primary

        def close(self):
            self.exited = True
            raise cleanup

    iterator = Iterator()
    with pytest.raises(ValueError) as caught:
        FiniteFactorManifest.ingest(iterator, tmp_path / "manifest.sqlite3")
    assert caught.value is primary
    assert caught.value.cleanup_errors == [cleanup]
    assert iterator.exited is True
    _assert_closed(connections[0])


def test_iterator_close_failure_after_success_closes_manifest(tmp_path, monkeypatch):
    connections = _capture_connection(monkeypatch)
    cleanup = RuntimeError("iterator close failed after success")

    class Iterator:
        exited = False
        yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.yielded:
                raise StopIteration
            self.yielded = True
            return {"name": "alpha"}

        def close(self):
            self.exited = True
            raise cleanup

    iterator = Iterator()
    with pytest.raises(RuntimeError) as caught:
        FiniteFactorManifest.ingest(iterator, tmp_path / "manifest.sqlite3")
    assert caught.value is cleanup
    assert iterator.exited is True
    _assert_closed(connections[0])


@pytest.mark.parametrize("fail_during_iteration", [True, False])
def test_iterator_close_lookup_failure_is_contained_like_close_call_failure(
    tmp_path, monkeypatch, fail_during_iteration,
):
    connections = _capture_connection(monkeypatch)
    primary = ValueError("iteration failed")
    cleanup = RuntimeError("close lookup failed")

    class Iterator:
        yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if fail_during_iteration:
                raise primary
            if self.yielded:
                raise StopIteration
            self.yielded = True
            return {"name": "alpha"}

        @property
        def close(self):
            raise cleanup

    if fail_during_iteration:
        with pytest.raises(ValueError) as caught:
            FiniteFactorManifest.ingest(Iterator(), tmp_path / "manifest.sqlite3")
        assert caught.value is primary
        assert caught.value.cleanup_errors == [cleanup]
    else:
        with pytest.raises(RuntimeError) as caught:
            FiniteFactorManifest.ingest(Iterator(), tmp_path / "manifest.sqlite3")
        assert caught.value is cleanup
    _assert_closed(connections[0])
