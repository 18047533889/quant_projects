"""Whole-request CPU/CUDA calibration for materialized QE batches.

Cold calibration remains opt-in because it executes both backends. Passing
exact candidates may later be reused by :func:`evaluate`'s default auto router
after strict content, source, runtime, and device-admission checks. The wall-
time limit is a soft admission budget: Python cannot safely interrupt an
in-flight public evaluation.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable
import weakref

import numpy as np
from quant_evaluator.contracts._hashutil import canonicalize

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.source_identity import ProcessSourceIdentity


@dataclass(frozen=True)
class CalibrationPolicy:
    """Hard bounds on replays/cache; wall time is a soft admission limit."""

    max_wall_time_seconds: float = 120.0
    repetitions: int = 3
    warmups: int = 1
    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-12
    source_check_mode: str = "strict_full_content"

    def __post_init__(self):
        if self.source_check_mode not in ("strict_full_content", "stat_guarded"):
            raise ValueError("source_check_mode must be strict_full_content or stat_guarded")
        if not np.isfinite(self.max_wall_time_seconds) or self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive and finite")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 9:
            raise ValueError("repetitions must be an integer in [1, 9]")
        if type(self.warmups) is not int or not 0 <= self.warmups <= 2:
            raise ValueError("warmups must be an integer in [0, 2]")
        for name in ("relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


# Keep only weak references so cache instances remain collectible. A forked
# process has its own process-local cache state; inherited route records must
# not be reused there, and inherited locks may have been held by vanished
# parent threads.
_CALIBRATION_CACHES = weakref.WeakSet()


def _reset_calibration_caches_after_fork():
    for cache in list(_CALIBRATION_CACHES):
        cache._lock = threading.RLock()
        cache._records.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_calibration_caches_after_fork)



class BoundedCalibrationCache:
    """Thread-safe LRU of exact-request route/timing records, never results."""

    def __init__(self, max_entries: int = 32, ttl_seconds: float = 3600.0):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if not np.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive and finite")
        self.max_entries = max_entries
        self.ttl_seconds = float(ttl_seconds)
        self._records = OrderedDict()
        self._lock = threading.RLock()
        _CALIBRATION_CACHES.add(self)

    def get(self, key: str):
        with self._lock:
            record = self._records.get(key)
            if record is not None and time.monotonic() - record.get("_cached_at", 0) < self.ttl_seconds:
                self._records.move_to_end(key)
                return {k: v for k, v in record.items() if k != "_cached_at"}
            if record is not None:
                del self._records[key]
            return None

    def put(self, key: str, record: dict) -> None:
        with self._lock:
            self._records[key] = {**record, "_cached_at": time.monotonic()}
            self._records.move_to_end(key)
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._records)


@dataclass(frozen=True)
class CalibratedEvaluation:
    bundle: Any
    metadata: dict


_PROCESS_CACHE_NONCE = secrets.token_hex(16)
_PROCESS_SOURCE_DIGEST = None
_PROCESS_SOURCE_DRIFTED = False
_SOURCE_IDENTITY = ProcessSourceIdentity(Path(__file__).resolve().parent.parent)


def _hash_array(hasher, array: np.ndarray, *, chunk_bytes: int = 4 * 1024**2) -> None:
    """Hash logical C-order bytes using bounded temporary chunks."""
    array = np.asarray(array)
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    if array.dtype.hasobject:
        # Object-array buffers contain process-specific pointers; hash the
        # immutable scalar values rather than their addresses.
        for value in array.flat:
            hasher.update(json.dumps(canonicalize(value), sort_keys=True,
                                     separators=(",", ":")).encode("utf-8"))
            hasher.update(b"\0")
        return
    if array.ndim == 0:
        array = array.reshape(1)
    itemsize = max(1, array.dtype.itemsize)
    iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C",
                         buffersize=max(1, chunk_bytes // itemsize))
    for chunk in iterator:
        contiguous = np.ascontiguousarray(chunk)
        # NumPy datetime/timedelta arrays reject the Python buffer format
        # exposed by memoryview(array). A uint8 view preserves exact C bytes
        # for every non-object dtype, with no additional chunk-sized copy.
        hasher.update(memoryview(contiguous.view(np.uint8)).cast("B"))


def _request_fingerprint(batch: FactorBatch, label: LabelBundle, metrics) -> str:
    digest = hashlib.sha256()
    digest.update(b"QE-CalibratedBatch-Input-v1\0")
    for text in (batch.factor_ids, tuple(metrics), label.target_id,
                 batch.time_axis.name, batch.time_axis.dtype,
                 batch.time_axis.size, batch.asset_axis.name,
                 batch.asset_axis.dtype, batch.asset_axis.size,
                 label.schema_version,
                 None if label.asset_axis is None else
                 (label.asset_axis.name, label.asset_axis.dtype, label.asset_axis.size)):
        digest.update(json.dumps(canonicalize(text), sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
    for name, value in (("factor_values", batch.values),
                        ("factor_validity", batch.validity),
                        ("time_coordinates", batch.time_axis.values),
                        ("asset_coordinates", batch.asset_axis.values),
                        ("label_asset_coordinates", None if label.asset_axis is None
                         else label.asset_axis.values)):
        digest.update(name.encode("ascii") + b"\0")
        if value is None:
            digest.update(b"none\0")
        else:
            _hash_array(digest, value)
    digest.update(label.content_hash.encode("ascii"))
    digest.update(json.dumps(canonicalize({
        "layout": batch.layout, "dtype": batch.dtype, "value_hash": batch.value_hash,
        "context_refs": batch.context_refs,
    }), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _source_fingerprint() -> str:
    """Bounded full-content identity; does not identify loaded Python code."""
    return _SOURCE_IDENTITY.identify(strict_full_content=True).digest


def _runtime_fingerprint(device_info: dict | None) -> dict:
    packages = {}
    for name in ("numpy", "scipy", "cupy-cuda12x", "cupy-cuda11x"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "device": device_info}


def calibration_identity(batch: FactorBatch, label_bundle: LabelBundle, *, metrics,
                         calibration_policy: CalibrationPolicy,
                         gpu_policy: GPUExecutionPolicy | None = None,
                         started: float | None = None) -> dict:
    """Build the exact content/source/runtime key shared by measured auto.

    Callers should only invoke this after a cheap measured-auto candidate
    match. It intentionally performs the full input fingerprint and source
    check, so it is not part of ordinary certified auto's no-candidate path.
    """
    if not isinstance(batch, FactorBatch) or not isinstance(label_bundle, LabelBundle):
        raise InvalidContractError("calibration requires FactorBatch and LabelBundle")
    if not isinstance(calibration_policy, CalibrationPolicy):
        raise InvalidContractError("calibration_policy must be CalibrationPolicy")
    gpu_policy = gpu_policy or GPUExecutionPolicy()
    if not isinstance(gpu_policy, GPUExecutionPolicy):
        raise InvalidContractError("gpu_policy must be GPUExecutionPolicy")
    if isinstance(metrics, (str, bytes)):
        raise InvalidContractError("metrics must be a sequence")
    metrics = tuple(metrics)
    if (not metrics or any(not isinstance(x, str) or not x for x in metrics)
            or len(set(metrics)) != len(metrics)):
        raise InvalidContractError("metrics must be nonempty unique metric ids")

    global _PROCESS_SOURCE_DIGEST, _PROCESS_SOURCE_DRIFTED
    started = time.monotonic() if started is None else started
    request_digest = _request_fingerprint(batch, label_bundle, metrics)
    hash_seconds = time.monotonic() - started
    if hash_seconds >= calibration_policy.max_wall_time_seconds:
        raise TimeoutError("input fingerprint exceeded soft calibration time budget")
    source_receipt = None
    if calibration_policy.source_check_mode == "stat_guarded":
        source_receipt = _SOURCE_IDENTITY.identify(strict_full_content=False)
        source_digest = source_receipt.digest
        if source_receipt.drifted:
            _PROCESS_SOURCE_DRIFTED = True
    else:
        source_digest = _source_fingerprint()
    if _PROCESS_SOURCE_DIGEST is None:
        _PROCESS_SOURCE_DIGEST = source_digest
    elif source_digest != _PROCESS_SOURCE_DIGEST:
        _PROCESS_SOURCE_DRIFTED = True
    admission_reason, device_info = _device_admission(gpu_policy)
    runtime = _runtime_fingerprint(device_info)
    key_payload = {"request": request_digest, "source": source_digest,
                   "process_nonce": (_PROCESS_CACHE_NONCE, _SOURCE_IDENTITY.process_nonce),
                   "runtime": runtime, "metrics": metrics,
                   "gpu_policy": {field.name: getattr(gpu_policy, field.name)
                                  for field in fields(gpu_policy)},
                   "calibration_policy": {
                       field.name: getattr(calibration_policy, field.name)
                       for field in fields(calibration_policy)}}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=str,
                                    separators=(",", ":")).encode()).hexdigest()
    setup_seconds = time.monotonic() - started
    source_metadata = {
        "mode": calibration_policy.source_check_mode,
        "identity_scope": "disk_source_not_loaded_python",
        "files_hashed": None if source_receipt is None else source_receipt.files_hashed,
        "bytes_hashed": None if source_receipt is None else source_receipt.bytes_hashed,
    }
    return {
        "request_digest": request_digest, "source_digest": source_digest,
        "runtime": runtime, "key": key, "source_metadata": source_metadata,
        "input_fingerprint_seconds": hash_seconds, "setup_seconds": setup_seconds,
        "device_admission": admission_reason, "device_info": device_info,
        "process_source_drifted": _PROCESS_SOURCE_DRIFTED,
    }


def _device_admission(policy: GPUExecutionPolicy):
    from quant_evaluator.runtime.evaluator import (
        _AUTO_BATCH_MIN_EFFECTIVE_VRAM_BYTES, _auto_batch_cuda_rejection,
    )
    rejection = _auto_batch_cuda_rejection(policy, _AUTO_BATCH_MIN_EFFECTIVE_VRAM_BYTES)
    if rejection:
        return rejection, None
    import cupy as cp
    device_id = policy.device_ids[0]
    with cp.cuda.Device(device_id):
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        driver = cp.cuda.runtime.driverGetVersion()
        runtime = cp.cuda.runtime.runtimeGetVersion()
    info = {"device_id": device_id, "name": str(name), "driver": driver,
            "cuda_runtime": runtime}
    return None, info


def _metric_fields(value):
    return (value.metric_id, value.valid, value.observation_count,
            value.value, value.metric_version, value.sample_unit)


def _compare_metric_values(left, right, *, rtol, atol, path):
    if type(left) is not type(right):
        return f"{path}: result types differ"
    from quant_evaluator.contracts.metric_artifacts import MetricArtifact
    if isinstance(left, MetricArtifact):
        if type(left) is not type(right):
            return f"{path}: artifact kinds differ"
        for item in fields(left):
            name = item.name
            if name == "provenance":
                mismatch = _compare_metric_values(
                    _semantic_provenance(left.provenance), _semantic_provenance(right.provenance),
                    rtol=rtol, atol=atol, path=f"{path}.provenance")
            else:
                mismatch = _compare_metric_values(
                    getattr(left, name), getattr(right, name), rtol=rtol,
                    atol=atol, path=f"{path}.{name}")
            if mismatch:
                return mismatch
        return None
    if hasattr(left, "metric_id") and hasattr(left, "observation_count"):
        a, b = _metric_fields(left), _metric_fields(right)
        if a[:3] != b[:3] or a[4:] != b[4:]:
            return f"{path}: metric identity/validity/count differs"
        if a[3] is None or b[3] is None:
            return None if a[3] is b[3] else f"{path}: one value is missing"
        if not np.isclose(a[3], b[3], rtol=rtol, atol=atol, equal_nan=True):
            return f"{path}: scalar values differ"
        return None
    if isinstance(left, np.ndarray):
        if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
            return f"{path}: array shape or dtype kind differs"
        if left.dtype.kind in "fc":
            if not np.array_equal(np.isfinite(left), np.isfinite(right)):
                return f"{path}: finite-value masks differ"
            if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
                return f"{path}: array values differ"
        elif not np.array_equal(left, right):
            return f"{path}: array values differ"
        return None
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return f"{path}: mapping keys differ"
        for key in left:
            mismatch = _compare_metric_values(left[key], right[key], rtol=rtol,
                                              atol=atol, path=f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{path}: sequence lengths differ"
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = _compare_metric_values(a, b, rtol=rtol, atol=atol,
                                              path=f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if is_dataclass(left):
        for item in fields(left):
            mismatch = _compare_metric_values(
                getattr(left, item.name), getattr(right, item.name), rtol=rtol,
                atol=atol, path=f"{path}.{item.name}")
            if mismatch:
                return mismatch
        return None
    if left != right:
        # Counts and discrete metadata must match exactly, even at large values.
        # Only floating/complex measurements are eligible for numerical tolerance.
        if isinstance(left, (float, np.floating, complex, np.complexfloating)):
            return None if np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True) else f"{path}: values differ"
        return f"{path}: values differ"
    return None


def _semantic_provenance(value):
    """Drop execution identity while retaining all data/config provenance."""
    if isinstance(value, Mapping):
        execution_keys = {"backend", "backend_name", "execution_backend",
                          "device", "device_id", "device_name"}
        return {key: _semantic_provenance(item) for key, item in value.items()
                if str(key).lower() not in execution_keys}
    if isinstance(value, (tuple, list)):
        return tuple(_semantic_provenance(item) for item in value)
    return value


def _parity_mismatch(cpu, cuda, policy: CalibrationPolicy) -> str | None:
    if tuple(cpu.factor_ids) != tuple(cuda.factor_ids):
        return "factor coordinate differs"
    for name in ("metric_values", "grouped_metrics", "diagnostics", "artifacts",
                 "factor_artifacts", "metric_versions", "instance_results",
                 "instance_specs", "warnings"):
        cpu_has = hasattr(cpu, name)
        cuda_has = hasattr(cuda, name)
        if cpu_has != cuda_has:
            return f"{name}: result field is missing on one backend"
        if not cpu_has:
            # Lightweight injected test doubles may omit all result fields.
            continue
        mismatch = _compare_metric_values(
            getattr(cpu, name), getattr(cuda, name),
            rtol=policy.relative_tolerance, atol=policy.absolute_tolerance, path=name)
        if mismatch:
            return mismatch
    return None


def _call_evaluate(evaluate_fn, batch, label, metrics, backend, gpu_policy):
    return evaluate_fn(batch, label, metrics=metrics, backend=backend,
                       gpu_policy=gpu_policy)


def evaluate_calibrated_batch(
    batch: FactorBatch,
    label_bundle: LabelBundle,
    *,
    metrics,
    calibration_policy: CalibrationPolicy,
    cache: BoundedCalibrationCache,
    gpu_policy: GPUExecutionPolicy | None = None,
    evaluate_fn: Callable | None = None,
) -> CalibratedEvaluation:
    """Calibrate explicit whole-request CPU/CUDA routes for one exact batch.

    Unsupported advanced inputs are intentionally absent from this initial
    API. Cache hits reuse only a passing exact-content calibration, recheck
    current device admission, and run the chosen backend explicitly.
    """
    if not isinstance(batch, FactorBatch) or not isinstance(label_bundle, LabelBundle):
        raise InvalidContractError("calibration requires FactorBatch and LabelBundle")
    if not isinstance(calibration_policy, CalibrationPolicy):
        raise InvalidContractError("calibration_policy must be CalibrationPolicy")
    if not isinstance(cache, BoundedCalibrationCache):
        raise InvalidContractError("cache must be BoundedCalibrationCache")
    if isinstance(metrics, (str, bytes)):
        raise InvalidContractError("metrics must be a sequence")
    metrics = tuple(metrics)
    if not metrics or any(not isinstance(x, str) or not x for x in metrics) or len(set(metrics)) != len(metrics):
        raise InvalidContractError("metrics must be nonempty unique metric ids")
    gpu_policy = gpu_policy or GPUExecutionPolicy()
    if not isinstance(gpu_policy, GPUExecutionPolicy):
        raise InvalidContractError("gpu_policy must be GPUExecutionPolicy")
    started = time.monotonic()
    public_evaluate = __import__(
        "quant_evaluator.runtime.evaluator", fromlist=["evaluate"]).evaluate
    injected_evaluator = evaluate_fn is not None and evaluate_fn is not public_evaluate
    evaluate_fn = evaluate_fn or public_evaluate
    identity = calibration_identity(
        batch, label_bundle, metrics=metrics,
        calibration_policy=calibration_policy, gpu_policy=gpu_policy,
        started=started,
    )
    request_digest = identity["request_digest"]
    hash_seconds = identity["input_fingerprint_seconds"]
    source_metadata = identity["source_metadata"]
    admission_reason = identity["device_admission"]
    key = identity["key"]
    setup_seconds = identity["setup_seconds"]
    # Injectable evaluators are test seams, not stable cache namespaces.
    cache_record = (cache.get(key) if admission_reason is None and not injected_evaluator
                    and not _PROCESS_SOURCE_DRIFTED else None)
    if cache_record is not None:
        route = cache_record["winner"]
        if route == "cuda_strict" and admission_reason is not None:
            route = "cpu"
        bundle = _call_evaluate(evaluate_fn, batch, label_bundle, metrics, route, gpu_policy)
        return CalibratedEvaluation(bundle, {
            "status": "cache_hit", "cache_key": key, "winner": route,
            "source_check": source_metadata,
            "input_fingerprint_seconds": hash_seconds, "setup_seconds": setup_seconds,
            "device_admission": "pass" if admission_reason is None else admission_reason,
            "calibration_record": cache_record,
        })

    if admission_reason is not None:
        bundle = _call_evaluate(evaluate_fn, batch, label_bundle, metrics, "cpu", gpu_policy)
        return CalibratedEvaluation(bundle, {
            "status": "cpu_only_device_admission", "winner": "cpu",
            "source_check": source_metadata,
            "device_admission": admission_reason, "setup_seconds": setup_seconds,
        })

    # Complete public evaluations cannot be interrupted safely. Check the soft
    # budget between calls and alternate route order to limit order bias.
    timings = {"cpu": [], "cuda_strict": []}
    first_call_seconds = {}
    last_results = {}
    calls = 0
    parity_mismatch = None
    cpu_result = None
    for iteration in range(calibration_policy.warmups + calibration_policy.repetitions):
        routes = ("cpu", "cuda_strict") if iteration % 2 == 0 else ("cuda_strict", "cpu")
        for route in routes:
            if time.monotonic() - started >= calibration_policy.max_wall_time_seconds:
                raise TimeoutError("calibration soft time budget exhausted between backend calls")
            call_started = time.monotonic()
            result = _call_evaluate(evaluate_fn, batch, label_bundle, metrics, route, gpu_policy)
            elapsed = time.monotonic() - call_started
            last_results[route] = result
            first_call_seconds.setdefault(route, elapsed)
            if route == "cpu":
                cpu_result = result
            if iteration >= calibration_policy.warmups:
                timings[route].append(elapsed)
            calls += 1
        # Each paired repetition must agree. A later matching pair cannot hide
        # nondeterminism or a transient backend divergence observed earlier.
        pair_mismatch = _parity_mismatch(last_results["cpu"], last_results["cuda_strict"],
                                         calibration_policy)
        if pair_mismatch is not None:
            parity_mismatch = pair_mismatch
            break

    mismatch = parity_mismatch
    cpu_seconds = float(np.median(timings["cpu"])) if timings["cpu"] else None
    cuda_seconds = float(np.median(timings["cuda_strict"])) if timings["cuda_strict"] else None
    fastest = ("cpu" if cpu_seconds is not None and cuda_seconds is not None
               and cpu_seconds <= cuda_seconds else
               ("cuda_strict" if cpu_seconds is not None and cuda_seconds is not None else "cpu"))
    record = {"winner": fastest, "cpu_median_seconds": cpu_seconds,
              "cuda_median_seconds": cuda_seconds,
              "cpu_first_call_seconds": first_call_seconds.get("cpu"),
              "cuda_first_call_seconds": first_call_seconds.get("cuda_strict"),
              "selection_basis": "steady_state_median",
              "parity": "pass" if mismatch is None else "fail",
              "parity_mismatch": mismatch, "repetitions": calibration_policy.repetitions,
              "warmups": calibration_policy.warmups}
    if mismatch is None and not injected_evaluator and not _PROCESS_SOURCE_DRIFTED:
        cache.put(key, record)
    winner = fastest if mismatch is None else "cpu"
    selected = last_results[winner] if mismatch is None else cpu_result
    return CalibratedEvaluation(selected, {
        "status": "calibrated" if mismatch is None else "parity_failed_cpu_fallback",
        "source_check": source_metadata,
        "cache_key": key, "winner": winner, "input_fingerprint_seconds": hash_seconds,
        "setup_seconds": setup_seconds, "device_admission": "pass",
        "calibration_record": record, "process_source_drifted": _PROCESS_SOURCE_DRIFTED,
        "cache_scope": "process_local",
        "total_public_calls": calls,
    })
