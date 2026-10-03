"""Tests for bounded actual runtime identity capture."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from quant_evaluator.runtime.source_runtime_identity import (
    capture_source_runtime_identity,
    is_qualified_source_runtime_identity,
    source_runtime_identity_digest,
)


_OPTIONAL = ("numpy", "pandas", "scipy", "numba", "llvmlite", "cupy", "threadpoolctl")


def _cold_modules(monkeypatch):
    for name in _OPTIONAL:
        monkeypatch.delitem(sys.modules, name, raising=False)


def _pool_module(records=None):
    return types.SimpleNamespace(
        __version__="test-threadpoolctl",
        threadpool_info=lambda: list(records or []),
    )


def test_cold_capture_never_imports_optional_runtime_modules(monkeypatch):
    _cold_modules(monkeypatch)
    real_import = builtins.__import__

    def block_optional(name, *args, **kwargs):
        if name.split(".", 1)[0] in _OPTIONAL:
            raise AssertionError(f"capture attempted optional import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_optional)
    snapshot = capture_source_runtime_identity()
    assert all(not item["loaded"] for item in snapshot["packages"].values())
    assert snapshot["threadpools"]["status"] == "unavailable_not_loaded"
    assert (
        "threadpoolctl_not_loaded_actual_pools_unknown"
        in snapshot["incomplete_reasons"]
    )
    assert snapshot["complete"] is False
    assert is_qualified_source_runtime_identity(snapshot) is False


def test_thread_environment_is_not_proof_of_live_pool_threads(monkeypatch):
    _cold_modules(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "73")
    first = capture_source_runtime_identity()
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "74")
    second = capture_source_runtime_identity()
    assert first["thread_environment"]["OPENBLAS_NUM_THREADS"] == "73"
    assert first["identity_digest"] != second["identity_digest"]
    assert first["threadpools"]["pools"] == []
    assert first["complete"] is False


def test_loaded_numba_and_threadpool_states_change_identity(monkeypatch):
    _cold_modules(monkeypatch)
    state = {"threads": 3, "pool_threads": 4}
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(__version__="test-numpy"),
    )
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(__version__="test-pandas"),
    )
    monkeypatch.setitem(
        sys.modules,
        "numba",
        types.SimpleNamespace(
            __version__="test-numba",
            get_num_threads=lambda: state["threads"],
            threading_layer=lambda: "omp",
        ),
    )
    pool = {
        "prefix": "libopenblas",
        "internal_api": "openblas",
        "user_api": "blas",
        "version": "0.3",
        "architecture": "generic",
        "threading_layer": "pthreads",
        "num_threads": 4,
        "filepath": "/must/not/leak/libopenblas.so",
    }
    pool_module = types.SimpleNamespace(
        __version__="test-threadpoolctl",
        threadpool_info=lambda: [{**pool, "num_threads": state["pool_threads"]}],
    )
    monkeypatch.setitem(sys.modules, "threadpoolctl", pool_module)

    first = capture_source_runtime_identity()
    assert first["complete"] is True
    assert first["packages"]["numpy"] == {
        "loaded": True,
        "version": "test-numpy",
    }
    assert first["numba"]["threads"] == 3
    assert first["threadpools"]["pools"][0]["num_threads"] == 4
    assert "filepath" not in first["threadpools"]["pools"][0]
    state["threads"] = 5
    second = capture_source_runtime_identity()
    assert second["identity_digest"] != first["identity_digest"]
    state["threads"] = 3
    state["pool_threads"] = 6
    third = capture_source_runtime_identity()
    assert third["identity_digest"] != first["identity_digest"]

    malformed_pool = {**first}
    malformed_pool["threadpools"] = {
        "status": "captured",
        "pools": [{**first["threadpools"]["pools"][0], "prefix": 17}],
    }
    malformed_pool["identity_digest"] = source_runtime_identity_digest(malformed_pool)
    assert not is_qualified_source_runtime_identity(malformed_pool)

    mismatched_numba = {**first}
    mismatched_numba["packages"] = {
        **first["packages"],
        "numba": {"loaded": False, "version": None},
    }
    mismatched_numba["identity_digest"] = source_runtime_identity_digest(
        mismatched_numba
    )
    assert not is_qualified_source_runtime_identity(mismatched_numba)


def test_bad_loaded_pool_probe_is_incomplete_not_unavailable(monkeypatch):
    _cold_modules(monkeypatch)

    def fail_probe():
        raise RuntimeError("private path must not escape")

    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        types.SimpleNamespace(
            __version__="test-threadpoolctl", threadpool_info=fail_probe
        ),
    )
    snapshot = capture_source_runtime_identity()
    assert snapshot["threadpools"]["status"] == "probe_failed"
    assert "threadpoolctl_probe_failed" in snapshot["incomplete_reasons"]
    assert snapshot["complete"] is False
    assert "private path" not in repr(snapshot)


def test_malformed_pool_fields_fail_closed_and_drop_unbounded_values(monkeypatch):
    _cold_modules(monkeypatch)
    malformed = [
        {
            "prefix": "x" * 1000,
            "internal_api": "openblas",
            "user_api": "blas",
            "num_threads": "many",
        },
        object(),
    ]
    monkeypatch.setitem(sys.modules, "threadpoolctl", _pool_module(malformed))
    snapshot = capture_source_runtime_identity()
    assert snapshot["complete"] is False
    assert len(snapshot["threadpools"]["pools"][0]["prefix"]) <= 160
    assert snapshot["threadpools"]["pools"][0]["num_threads"] is None
    assert "threadpool_record_malformed" in snapshot["incomplete_reasons"]
    assert "threadpool_thread_count_unavailable" in snapshot["incomplete_reasons"]


def test_cupy_probe_never_creates_context_and_tracks_existing_device(monkeypatch):
    _cold_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "threadpoolctl", _pool_module())
    calls = []
    state = {"context": None, "device": 0}

    def current_context():
        calls.append("current_context")
        return state["context"]

    def get_device():
        calls.append("get_device")
        return state["device"]

    driver = types.SimpleNamespace(
        ctxGetCurrent=current_context,
        ctxGetDevice=get_device,
    )
    cupy = types.SimpleNamespace(
        __version__="test-cupy",
        cuda=types.SimpleNamespace(
            driver=driver,
            runtime=types.SimpleNamespace(
                deviceGetPCIBusId=lambda device: f"0000:{device + 1:02x}:00.0",
                getDeviceProperties=lambda device: {
                    "uuid": b"1234567",
                    "name": f"GPU {device}",
                    "totalGlobalMem": 24 * 1024**3,
                    "major": 8,
                    "minor": 9,
                }
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    lazy = capture_source_runtime_identity()
    assert lazy["complete"] is True
    assert lazy["cuda"]["status"] == "no_current_context"
    assert lazy["cuda"]["device"] is None
    assert calls == ["current_context"]

    state["context"] = object()
    active = capture_source_runtime_identity()
    assert is_qualified_source_runtime_identity(active)
    device = active["cuda"]["device"]
    assert device["identity_scope"] == "host_pci_bus_v1"
    assert device["pci_bus_id"] == "0000:01:00.0"
    assert len(device["host_identity_sha256"]) == 64
    assert device["total_memory_bytes"] == 24 * 1024**3
    assert device["compute_capability"] == [8, 9]
    assert device["uuid"] is None
    assert device["uuid_status"] == "unavailable_truncated_or_missing"
    for field, value in (
        ("pci_bus_id", "not-pci"), ("total_memory_bytes", True),
        ("total_memory_bytes", 0), ("compute_capability", [True, 9]),
        ("compute_capability", [8]), ("host_identity_sha256", "not-a-hash"),
        ("uuid", "1234567"), ("uuid_status", "captured_full_16_bytes"),
        ("identity_scope", "globally_unique_uuid"),
    ):
        bad = {**active, "cuda": {**active["cuda"], "device": {**device, field: value}}}
        bad["identity_digest"] = source_runtime_identity_digest(bad)
        assert not is_qualified_source_runtime_identity(bad)
    state["device"] = 1
    changed = capture_source_runtime_identity()
    assert changed["identity_digest"] != active["identity_digest"]
    assert changed["cuda"]["device"]["pci_bus_id"] == "0000:02:00.0"
    cupy.cuda.runtime.getDeviceProperties = lambda device: {
        "uuid": b"1234567890abcdef", "name": "GPU 1",
        "totalGlobalMem": 24 * 1024**3, "major": 8, "minor": 9,
    }
    full_uuid = capture_source_runtime_identity()
    assert is_qualified_source_runtime_identity(full_uuid)
    assert full_uuid["cuda"]["device"]["uuid"] == b"1234567890abcdef".hex()
    assert full_uuid["cuda"]["device"]["uuid_status"] == "captured_full_16_bytes"


def test_real_cupy_device_api_opt_in(monkeypatch):
    import os

    if os.environ.get("QE_SOURCE_RUNTIME_CUPY_DEVICE_TEST") != "1":
        pytest.skip("set QE_SOURCE_RUNTIME_CUPY_DEVICE_TEST=1 to initialize CUDA")
    import cupy

    # A single-byte allocation initializes CUDA only under explicit opt-in.
    cupy.empty((1,), dtype=cupy.uint8)
    assert callable(cupy.cuda.driver.ctxGetCurrent)
    assert callable(cupy.cuda.driver.ctxGetDevice)
    assert callable(cupy.cuda.runtime.getDeviceProperties)
    assert callable(cupy.cuda.runtime.deviceGetPCIBusId)
    snapshot = capture_source_runtime_identity()
    assert snapshot["cuda"]["context_initialized"] is True
    assert snapshot["cuda"]["status"] == "captured"
    device = snapshot["cuda"]["device"]
    assert device["identity_scope"] == "host_pci_bus_v1"
    assert device["pci_bus_id"]
    assert device["total_memory_bytes"] > 0
    if device["uuid_status"] == "captured_full_16_bytes":
        assert len(device["uuid"]) == 32
    else:
        assert device["uuid_status"] == "unavailable_truncated_or_missing"
        assert device["uuid"] is None


def test_cupy_lazy_load_invalidates_qualified_identity(monkeypatch):
    _cold_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "threadpoolctl", _pool_module())
    before = capture_source_runtime_identity()
    assert is_qualified_source_runtime_identity(before)
    cupy = types.SimpleNamespace(
        __version__="test-cupy",
        cuda=types.SimpleNamespace(
            driver=types.SimpleNamespace(ctxGetCurrent=lambda: None)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    after = capture_source_runtime_identity()
    assert is_qualified_source_runtime_identity(after)
    assert before["identity_digest"] != after["identity_digest"]


def test_digest_is_deterministic_bounded_and_fail_closed():
    snapshot = {
        "schema": "source-runtime-identity-v2",
        "complete": True,
        "incomplete_reasons": [],
        "sample": {"b": 2, "a": 1},
    }
    first = source_runtime_identity_digest(snapshot)
    second = source_runtime_identity_digest(
        {
            "sample": {"a": 1, "b": 2},
            "incomplete_reasons": [],
            "complete": True,
            "schema": "source-runtime-identity-v2",
        }
    )
    assert first == second
    qualified = {**snapshot, "identity_digest": first}
    assert not is_qualified_source_runtime_identity(qualified)
    assert not is_qualified_source_runtime_identity({**qualified, "sample": {"a": 9}})
    with pytest.raises(TypeError):
        source_runtime_identity_digest(None)
    with pytest.raises(ValueError):
        source_runtime_identity_digest({"schema": "wrong"})
    with pytest.raises(ValueError):
        source_runtime_identity_digest(
            {"schema": "source-runtime-identity-v2", "huge": "x" * 70000}
        )
