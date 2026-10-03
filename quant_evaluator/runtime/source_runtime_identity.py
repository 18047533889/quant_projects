"""Bounded actual-runtime identity capture for source qualification.

Capture deliberately inspects only modules already present in ``sys.modules``.
It does not import optional runtimes, load threadpoolctl, or initialize CuPy or
CUDA. An incomplete snapshot is diagnostic only and must fail closed where a
qualified identity is required.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from collections.abc import Mapping
from typing import Any


_SCHEMA = "source-runtime-identity-v2"
_MAX_TEXT = 160
_MAX_JSON_BYTES = 64 * 1024
_MAX_POOLS = 128
_VERSIONED_MODULES = ("numpy", "pandas", "scipy", "numba", "llvmlite", "cupy", "threadpoolctl")
_THREAD_ENVIRONMENT = (
    "BLIS_NUM_THREADS",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_POOL_FIELDS = (
    "prefix",
    "internal_api",
    "user_api",
    "version",
    "architecture",
    "threading_layer",
    "num_threads",
)


def _bounded_text(value: Any) -> tuple[str | None, bool]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None, False
    if not isinstance(value, str):
        return None, False
    if len(value) > _MAX_TEXT:
        return value[:_MAX_TEXT], False
    return value, True


def _safe_module_version(name: str, incomplete: list[str]) -> dict[str, Any]:
    module = sys.modules.get(name)
    if module is None:
        return {"loaded": False, "version": None}
    version, valid = _bounded_text(getattr(module, "__version__", None))
    if not valid:
        incomplete.append(f"{name}_version_unavailable_or_unbounded")
    return {"loaded": True, "version": version}


def _capture_numba(incomplete: list[str]) -> dict[str, Any]:
    module = sys.modules.get("numba")
    if module is None:
        return {"loaded": False, "threads": None, "threading_layer": None}
    get_threads = getattr(module, "get_num_threads", None)
    get_layer = getattr(module, "threading_layer", None)
    threads = None
    layer = None
    try:
        value = get_threads() if callable(get_threads) else None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError
        threads = value
    except Exception:
        incomplete.append("numba_thread_count_unavailable")
    try:
        value = get_layer() if callable(get_layer) else None
        layer, valid = _bounded_text(value)
        if not valid:
            raise ValueError
    except Exception:
        incomplete.append("numba_threading_layer_unavailable")
    return {"loaded": True, "threads": threads, "threading_layer": layer}


def _pool_record(record: Any, incomplete: list[str]) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        incomplete.append("threadpool_record_malformed")
        return None
    normalized: dict[str, Any] = {}
    for field in _POOL_FIELDS:
        value = record.get(field)
        if field == "num_threads":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                incomplete.append("threadpool_thread_count_unavailable")
                normalized[field] = None
            else:
                normalized[field] = value
        elif value is None:
            normalized[field] = None
        else:
            text, valid = _bounded_text(value)
            normalized[field] = text
            if not valid:
                incomplete.append(f"threadpool_{field}_unavailable_or_unbounded")
    return normalized


def _capture_threadpools(incomplete: list[str]) -> dict[str, Any]:
    module = sys.modules.get("threadpoolctl")
    if module is None:
        incomplete.append("threadpoolctl_not_loaded_actual_pools_unknown")
        return {"status": "unavailable_not_loaded", "pools": []}
    probe = getattr(module, "threadpool_info", None)
    if not callable(probe):
        incomplete.append("threadpoolctl_probe_unavailable")
        return {"status": "probe_unavailable", "pools": []}
    try:
        records = probe()
    except Exception:
        incomplete.append("threadpoolctl_probe_failed")
        return {"status": "probe_failed", "pools": []}
    if not isinstance(records, (list, tuple)):
        incomplete.append("threadpoolctl_result_malformed")
        return {"status": "result_malformed", "pools": []}
    if len(records) > _MAX_POOLS:
        incomplete.append("threadpool_count_exceeds_limit")
        records = records[:_MAX_POOLS]
    pools = [
        pool
        for pool in (_pool_record(record, incomplete) for record in records)
        if pool is not None
    ]
    pools.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return {"status": "captured", "pools": pools}


def _device_text(value: Any) -> str | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    text, valid = _bounded_text(value)
    return text if valid and text else None


def _capture_cuda(incomplete: list[str]) -> dict[str, Any]:
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return {
            "status": "cupy_not_loaded",
            "context_initialized": False,
            "device": None,
        }
    cuda = getattr(cupy, "cuda", None)
    driver = getattr(cuda, "driver", None) if cuda is not None else None
    current_context = getattr(driver, "ctxGetCurrent", None)
    if not callable(current_context):
        incomplete.append("cupy_context_probe_unavailable")
        return {
            "status": "context_probe_unavailable",
            "context_initialized": None,
            "device": None,
        }
    try:
        context = current_context()
    except Exception:
        incomplete.append("cupy_context_probe_failed")
        return {
            "status": "context_probe_failed",
            "context_initialized": None,
            "device": None,
        }
    if context is None or context == 0:
        return {
            "status": "no_current_context",
            "context_initialized": False,
            "device": None,
        }

    get_device = getattr(driver, "ctxGetDevice", None)
    runtime = getattr(cuda, "runtime", None)
    get_properties = getattr(runtime, "getDeviceProperties", None)
    get_pci_bus = getattr(runtime, "deviceGetPCIBusId", None)
    if not all(callable(probe) for probe in (get_device, get_properties, get_pci_bus)):
        incomplete.append("cupy_device_identity_probe_unavailable")
        return {
            "status": "device_identity_probe_unavailable",
            "context_initialized": True,
            "device": None,
        }
    try:
        # ctxGetDevice() queries the current thread's active context and takes
        # no context argument. Device properties are read only after the
        # existing context has been verified above.
        device_id = get_device()
        properties = get_properties(device_id)
        if not isinstance(properties, Mapping):
            raise ValueError
        raw_uuid = properties.get("uuid")
        # Some CuPy versions expose UUID bytes truncated at a zero byte. Such
        # values are not a complete UUID; bind a narrower host+PCI identity.
        uuid = raw_uuid.hex() if type(raw_uuid) is bytes and len(raw_uuid) == 16 else None
        uuid_status = "captured_full_16_bytes" if uuid else "unavailable_truncated_or_missing"
        name = _device_text(properties.get("name"))
        pci_bus = _device_text(get_pci_bus(device_id))
        host_name, host_valid = _bounded_text(platform.node())
        host_digest = hashlib.sha256(host_name.encode("utf-8")).hexdigest() if host_valid and host_name else None
        total_memory = properties.get("totalGlobalMem")
        major, minor = properties.get("major"), properties.get("minor")
        if (
            isinstance(device_id, bool)
            or not isinstance(device_id, int)
            or device_id < 0
        ):
            raise ValueError
        if (name is None or pci_bus is None or host_digest is None
                or re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci_bus) is None
                or type(total_memory) is not int or total_memory <= 0
                or type(major) is not int or major < 0
                or type(minor) is not int or minor < 0):
            raise ValueError
    except Exception:
        incomplete.append("cupy_device_identity_probe_failed")
        return {
            "status": "device_identity_probe_failed",
            "context_initialized": True,
            "device": None,
        }
    return {
        "status": "captured",
        "context_initialized": True,
        "device": {
            "id": device_id,
            "identity_scope": "host_pci_bus_v1",
            "host_identity_sha256": host_digest,
            "pci_bus_id": pci_bus.lower(),
            "name": name,
            "total_memory_bytes": total_memory,
            "compute_capability": [major, minor],
            "uuid": uuid,
            "uuid_status": uuid_status,
        },
    }


def _thread_environment(incomplete: list[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for key in _THREAD_ENVIRONMENT:
        value = os.environ.get(key)
        if value is None:
            values[key] = None
            continue
        bounded, valid = _bounded_text(value)
        values[key] = bounded
        if not valid:
            incomplete.append(f"thread_environment_{key}_unbounded")
    return values


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    def validate(value: Any, depth: int = 0) -> None:
        if depth > 16:
            raise ValueError("runtime identity nesting exceeds limit")
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            if value.bit_length() > 128:
                raise ValueError("runtime identity integer exceeds limit")
            return
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("runtime identity floats must be finite")
            return
        if isinstance(value, str):
            if len(value) > _MAX_TEXT:
                raise ValueError("runtime identity text exceeds limit")
            return
        if isinstance(value, (list, tuple)):
            if len(value) > 1024:
                raise ValueError("runtime identity sequence exceeds limit")
            for item in value:
                validate(item, depth + 1)
            return
        if isinstance(value, Mapping):
            if len(value) > 128:
                raise ValueError("runtime identity mapping exceeds limit")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > _MAX_TEXT:
                    raise ValueError("runtime identity keys must be bounded strings")
                validate(item, depth + 1)
            return
        raise ValueError("runtime identity contains an unsupported value")

    validate(payload)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime identity must contain bounded JSON values") from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("runtime identity exceeds the bounded JSON limit")
    return encoded


def source_runtime_identity_digest(snapshot: Mapping[str, Any]) -> str:
    """Hash a normalized snapshot, excluding its self-referential digest field."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("runtime identity snapshot must be a mapping")
    payload = {
        key: value for key, value in snapshot.items() if key != "identity_digest"
    }
    if payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported runtime identity schema")
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def is_qualified_source_runtime_identity(snapshot: Any) -> bool:
    """Return true only for complete snapshots whose canonical digest verifies."""
    if not isinstance(snapshot, Mapping):
        return False
    if snapshot.get("complete") is not True:
        return False
    if snapshot.get("incomplete_reasons") != []:
        return False
    required = {
        "schema",
        "python",
        "packages",
        "numba",
        "threadpools",
        "thread_environment",
        "cuda",
        "complete",
        "incomplete_reasons",
        "identity_digest",
    }
    if set(snapshot) != required or snapshot.get("schema") != _SCHEMA:
        return False
    python_version, valid_python = _bounded_text(snapshot.get("python"))
    if not valid_python or not python_version:
        return False
    packages = snapshot.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(_VERSIONED_MODULES):
        return False
    for package in _VERSIONED_MODULES:
        item = packages.get(package)
        if not isinstance(item, Mapping) or set(item) != {"loaded", "version"}:
            return False
        if not isinstance(item.get("loaded"), bool):
            return False
        version = item.get("version")
        if item["loaded"]:
            if not isinstance(version, str) or not version or len(version) > _MAX_TEXT:
                return False
        elif version is not None:
            return False
    numba = snapshot.get("numba")
    if not isinstance(numba, Mapping) or set(numba) != {
        "loaded",
        "threads",
        "threading_layer",
    }:
        return False
    if not isinstance(numba.get("loaded"), bool):
        return False
    if numba["loaded"]:
        if (
            isinstance(numba.get("threads"), bool)
            or not isinstance(numba.get("threads"), int)
            or numba["threads"] <= 0
            or not isinstance(numba.get("threading_layer"), str)
        ):
            return False
    elif numba.get("threads") is not None or numba.get("threading_layer") is not None:
        return False
    if packages["numba"]["loaded"] != numba["loaded"]:
        return False
    pools = snapshot.get("threadpools")
    if not isinstance(pools, Mapping) or set(pools) != {"status", "pools"}:
        return False
    if pools.get("status") != "captured" or not isinstance(pools.get("pools"), list):
        return False
    if not packages["threadpoolctl"]["loaded"]:
        return False
    for pool in pools["pools"]:
        if not isinstance(pool, Mapping) or set(pool) != set(_POOL_FIELDS):
            return False
        if (
            isinstance(pool.get("num_threads"), bool)
            or not isinstance(pool.get("num_threads"), int)
            or pool["num_threads"] <= 0
        ):
            return False
        if any(
            value is not None and (not isinstance(value, str) or len(value) > _MAX_TEXT)
            for field, value in pool.items()
            if field != "num_threads"
        ):
            return False
    environment = snapshot.get("thread_environment")
    if not isinstance(environment, Mapping) or set(environment) != set(
        _THREAD_ENVIRONMENT
    ):
        return False
    if any(
        value is not None and not isinstance(value, str)
        for value in environment.values()
    ):
        return False
    cuda = snapshot.get("cuda")
    if not isinstance(cuda, Mapping) or set(cuda) != {
        "status",
        "context_initialized",
        "device",
    }:
        return False
    if packages["cupy"]["loaded"]:
        if cuda.get("status") == "no_current_context":
            if (
                cuda.get("context_initialized") is not False
                or cuda.get("device") is not None
            ):
                return False
        elif cuda.get("status") == "captured":
            device = cuda.get("device")
            if (
                cuda.get("context_initialized") is not True
                or not isinstance(device, Mapping)
                or set(device) != {"id", "identity_scope", "host_identity_sha256",
                                   "pci_bus_id", "name", "total_memory_bytes",
                                   "compute_capability", "uuid", "uuid_status"}
                or isinstance(device.get("id"), bool)
                or not isinstance(device.get("id"), int)
                or device["id"] < 0
                or not isinstance(device.get("name"), str)
                or not device["name"]
                or device.get("identity_scope") != "host_pci_bus_v1"
                or type(device.get("host_identity_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", device["host_identity_sha256"]) is None
                or type(device.get("pci_bus_id")) is not str
                or re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", device["pci_bus_id"]) is None
                or type(device.get("total_memory_bytes")) is not int
                or device["total_memory_bytes"] <= 0
                or type(device.get("compute_capability")) is not list
                or len(device["compute_capability"]) != 2
                or any(type(v) is not int or v < 0 for v in device["compute_capability"])
            ):
                return False
            if device.get("uuid_status") == "captured_full_16_bytes":
                if (type(device.get("uuid")) is not str
                        or re.fullmatch(r"[0-9a-f]{32}", device["uuid"]) is None):
                    return False
            elif device.get("uuid_status") == "unavailable_truncated_or_missing":
                if device.get("uuid") is not None:
                    return False
            else:
                return False
        else:
            return False
    elif (
        cuda.get("status") != "cupy_not_loaded"
        or cuda.get("context_initialized") is not False
        or cuda.get("device") is not None
    ):
        return False
    digest = snapshot.get("identity_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        return digest == source_runtime_identity_digest(snapshot)
    except (TypeError, ValueError):
        return False


def capture_source_runtime_identity() -> dict[str, Any]:
    """Capture loaded runtime state without importing optional runtimes.

    Thread environment values are contextual data, never proof of active thread
    counts. Missing threadpoolctl leaves actual native pool state unknown and
    therefore marks the snapshot incomplete. CuPy is inspected only when it is
    already in ``sys.modules``; the sole context probe is ``ctxGetCurrent``.
    """
    incomplete: list[str] = []
    packages = {
        name: _safe_module_version(name, incomplete) for name in _VERSIONED_MODULES
    }
    payload: dict[str, Any] = {
        "schema": _SCHEMA,
        "python": platform.python_version(),
        "packages": packages,
        "numba": _capture_numba(incomplete),
        "threadpools": _capture_threadpools(incomplete),
        "thread_environment": _thread_environment(incomplete),
        "cuda": _capture_cuda(incomplete),
    }
    reasons = sorted(set(incomplete))
    payload["complete"] = not reasons
    payload["incomplete_reasons"] = reasons
    payload["identity_digest"] = source_runtime_identity_digest(payload)
    return payload


__all__ = [
    "capture_source_runtime_identity",
    "is_qualified_source_runtime_identity",
    "source_runtime_identity_digest",
]
