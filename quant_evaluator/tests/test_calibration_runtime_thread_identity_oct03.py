"""Runtime identity must include active CPU thread configuration."""
import importlib.metadata
import sys
import types

import pytest

from quant_evaluator.runtime import backend_calibration


THREAD_ENV_KEYS = (
    "BLIS_NUM_THREADS", "MKL_DYNAMIC", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
POOL_KEYS = ("prefix", "internal_api", "user_api", "version", "architecture",
             "threading_layer", "num_threads")


def test_runtime_fingerprint_tracks_thread_environment_and_numba_runtime(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    first = backend_calibration._runtime_fingerprint(None)
    assert first["thread_environment"]["OPENBLAS_NUM_THREADS"] == "7"
    assert set(first["thread_environment"]) == set(THREAD_ENV_KEYS)

    try:
        numba_version = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        assert "numba" not in first["packages"]
        assert first["numba_threads"] is None
    else:
        from numba import get_num_threads
        assert first["packages"]["numba"] == numba_version
        assert first["numba_threads"] == get_num_threads()
        assert first["numba_threads"] > 0
        assert isinstance(first["numba_threading_layer"], str) and first["numba_threading_layer"]
        assert first["packages"]["llvmlite"] == importlib.metadata.version("llvmlite")

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "9")
    second = backend_calibration._runtime_fingerprint(None)
    assert first["thread_environment"] != second["thread_environment"]


def test_runtime_fingerprint_normalizes_threadpools_and_omits_host_paths(monkeypatch):
    real_version = importlib.metadata.version

    def version(name):
        return "test-threadpoolctl" if name == "threadpoolctl" else real_version(name)

    monkeypatch.setattr(backend_calibration.importlib.metadata, "version", version)
    pool = {
        "prefix": "libopenblas", "internal_api": "openblas", "user_api": "blas",
        "version": "0.3.26", "architecture": "Cooperlake",
        "threading_layer": "pthreads", "num_threads": 4,
    }
    records = [
        {**pool, "filepath": "/host/one/libopenblas.so"},
        {"prefix": "libomp", "internal_api": "openmp", "user_api": "openmp",
         "version": "5.0", "architecture": "generic", "threading_layer": "omp",
         "num_threads": 8, "filepath": "/host/two/libomp.so"},
        {**pool, "filepath": "/another-host/libopenblas.so"},
    ]
    fake = types.SimpleNamespace(threadpool_info=lambda: list(records))
    monkeypatch.setitem(sys.modules, "threadpoolctl", fake)
    first = backend_calibration._runtime_fingerprint(None)
    records.reverse()
    records[0]["filepath"] = "/changed/path.so"
    second = backend_calibration._runtime_fingerprint(None)

    assert first["threadpoolctl_available"] is True
    assert first["packages"]["threadpoolctl"] == "test-threadpoolctl"
    assert first["blas_threadpools"] == second["blas_threadpools"]
    assert first["blas_threadpools"].count(pool) == 2
    assert all(set(item) == set(POOL_KEYS) for item in first["blas_threadpools"])
    assert len(first["blas_threadpools"]) == 3
    assert all("filepath" not in item for item in first["blas_threadpools"])


def test_runtime_fingerprint_distinguishes_unavailable_threadpoolctl(monkeypatch):
    real_version = importlib.metadata.version

    def version(name):
        if name == "threadpoolctl":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(backend_calibration.importlib.metadata, "version", version)
    runtime = backend_calibration._runtime_fingerprint(None)
    assert runtime["threadpoolctl_available"] is False
    assert runtime["blas_threadpools"] == []
    assert "threadpoolctl" not in runtime["packages"]


def test_installed_numba_probe_failure_is_not_reported_as_missing(monkeypatch):
    real_version = importlib.metadata.version

    def version(name):
        if name in ("numba", "llvmlite"):
            return "test-" + name
        if name == "threadpoolctl":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    def fail_probe():
        raise RuntimeError("Numba thread probe failed")

    monkeypatch.setattr(backend_calibration.importlib.metadata, "version", version)
    monkeypatch.setitem(sys.modules, "numba", types.SimpleNamespace(
        get_num_threads=fail_probe, threading_layer=lambda: "omp"))
    with pytest.raises(RuntimeError, match="Numba thread probe failed"):
        backend_calibration._runtime_fingerprint(None)


def test_installed_threadpoolctl_probe_failure_is_not_reported_as_missing(monkeypatch):
    real_version = importlib.metadata.version

    def version(name):
        if name in ("numba", "llvmlite"):
            return "test-" + name
        if name == "threadpoolctl":
            return "test-threadpoolctl"
        return real_version(name)

    def fail_probe():
        raise RuntimeError("BLAS threadpool probe failed")

    monkeypatch.setattr(backend_calibration.importlib.metadata, "version", version)
    monkeypatch.setitem(sys.modules, "numba", types.SimpleNamespace(
        get_num_threads=lambda: 6, threading_layer=lambda: "omp"))
    monkeypatch.setitem(sys.modules, "threadpoolctl", types.SimpleNamespace(
        threadpool_info=fail_probe))
    with pytest.raises(RuntimeError, match="BLAS threadpool probe failed"):
        backend_calibration._runtime_fingerprint(None)
