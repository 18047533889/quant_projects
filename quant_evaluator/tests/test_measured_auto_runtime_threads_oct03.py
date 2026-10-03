"""Measured-auto candidate runtime thread identity integration tests."""
from __future__ import annotations
import importlib.metadata
import os
import sys
import types
import numpy as np
import pytest
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime import auto_calibration, backend_calibration
from quant_evaluator.runtime.backend_calibration import BoundedCalibrationCache, CalibrationPolicy
from quant_evaluator.runtime import measured_auto_registry as registry

ENV_KEYS = ("BLIS_NUM_THREADS", "MKL_DYNAMIC", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
            "NUMBA_THREADING_LAYER", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _inputs():
    times = AxisRef("time", "int64", 4, np.arange(4, dtype=np.int64))
    assets = AxisRef("asset", "str", 3, np.array(["a", "b", "c"]))
    batch = FactorBatch(("f1",), times, assets, np.arange(12.).reshape(4, 3, 1))
    labels = LabelBundle("thread-runtime", np.arange(12.).reshape(4, 3), 1,
        decision_time=(0, 1, 2, 3), label_start_time=(1, 2, 3, 4),
        label_end_time=(2, 3, 4, 5), asset_axis=assets)
    return batch, labels


def _probes(monkeypatch):
    state = {"numba_threads": 2, "blas_threads": 4, "numba_layer": "workqueue",
        "versions": {"numpy": "2.0", "scipy": "1.13", "numba": "0.61",
                     "llvmlite": "0.44", "threadpoolctl": "3.6"},
        "set_calls": [], "limit_calls": []}
    for key in ENV_KEYS:
        monkeypatch.setenv(key, "2" if key.endswith("NUM_THREADS") else "default")
    numba = types.ModuleType("numba")
    numba.get_num_threads = lambda: state["numba_threads"]
    numba.threading_layer = lambda: state["numba_layer"]
    numba.set_num_threads = lambda n: state["set_calls"].append(n)
    threadpoolctl = types.ModuleType("threadpoolctl")
    threadpoolctl.threadpool_info = lambda: [{
        "user_api": "blas", "internal_api": "openblas", "prefix": "libopenblas",
        "version": "0.3.26", "architecture": "Cooperlake",
        "threading_layer": "pthreads", "num_threads": state["blas_threads"],
        "filepath": "/host/specific/path/libopenblas.so",
    }]
    class Limits:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
    def limits(*args, **kwargs):
        state["limit_calls"].append((args, kwargs))
        return Limits()
    threadpoolctl.threadpool_limits = limits
    monkeypatch.setitem(sys.modules, "numba", numba)
    monkeypatch.setitem(sys.modules, "threadpoolctl", threadpoolctl)
    def version(name):
        if name not in state["versions"]:
            raise importlib.metadata.PackageNotFoundError(name)
        return state["versions"][name]
    monkeypatch.setattr(backend_calibration.importlib.metadata, "version", version)
    return state


def _register(monkeypatch, batch, labels):
    registry.clear_registry_for_tests()
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(backend_calibration, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(backend_calibration, "_source_fingerprint", lambda: "stable-source")
    monkeypatch.setattr(backend_calibration, "_device_admission",
                        lambda _policy: (None, {"device_id": 0, "name": "mock-device"}))
    policy, gpu, metrics = CalibrationPolicy(), GPUExecutionPolicy(), ("coverage",)
    identity = backend_calibration.calibration_identity(
        batch, labels, metrics=metrics, calibration_policy=policy, gpu_policy=gpu)
    record = {"winner": "cpu", "parity": "pass", "selection_basis": "steady_state_median",
        "cpu_median_seconds": 1.0, "cuda_median_seconds": 5.0,
        "repetitions": policy.repetitions, "warmups": policy.warmups}
    cache = BoundedCalibrationCache()
    cache.put(identity["key"], record)
    assert registry.register_candidate(
        cache, identity["key"], status="calibrated", winner="cpu",
        calibration_record=record, source_check={"mode": "strict_full_content"},
        process_source_drifted=False, calibration_policy=policy, gpu_policy=gpu,
        descriptor=auto_calibration.measured_auto_descriptor(batch, labels, metrics),
        setup_seconds=.1, batch=batch, label=labels, metrics=metrics,
        request_digest=identity["request_digest"])
    return gpu, cache


def _select(batch, labels, gpu):
    return auto_calibration.select_measured_auto_backend(
        batch, labels, metrics=("coverage",), gpu_policy=gpu,
        static_backend="cuda_strict", static_reason="static")


def test_unchanged_runtime_reuses_candidate_without_thread_mutation(monkeypatch):
    state = _probes(monkeypatch)
    batch, labels = _inputs()
    gpu, cache = _register(monkeypatch, batch, labels)
    env_before = {key: os.environ.get(key) for key in ENV_KEYS}
    runtime_before = (state["numba_threads"], state["blas_threads"], state["numba_layer"])
    assert _select(batch, labels, gpu)[:3] == ("cpu", "measured_auto_exact_candidate", True)
    assert {key: os.environ.get(key) for key in ENV_KEYS} == env_before
    assert (state["numba_threads"], state["blas_threads"], state["numba_layer"]) == runtime_before
    assert state["set_calls"] == [] and state["limit_calls"] == []


@pytest.mark.parametrize("change", ["environment", "numba_threads", "blas_threads",
                                    "numba_layer", "library_version"])
def test_runtime_thread_or_library_change_invalidates_candidate(monkeypatch, change):
    state = _probes(monkeypatch)
    batch, labels = _inputs()
    gpu, cache = _register(monkeypatch, batch, labels)
    assert _select(batch, labels, gpu)[:3] == ("cpu", "measured_auto_exact_candidate", True)
    if change == "environment": os.environ["OPENBLAS_NUM_THREADS"] = "7"
    elif change == "numba_threads": state["numba_threads"] = 7
    elif change == "blas_threads": state["blas_threads"] = 7
    elif change == "numba_layer": state["numba_layer"] = "omp"
    else: state["versions"]["numba"] = "0.62"
    selected = _select(batch, labels, gpu)
    assert selected[0] == "cuda_strict" and selected[2] is False
    assert selected[3]["rejection_reason"] == "identity_mismatch"
    assert registry.registry_size() == 0
    assert state["set_calls"] == [] and state["limit_calls"] == []
