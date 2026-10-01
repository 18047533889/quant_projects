"""Stable content hashing utilities for quant_evaluator contracts.

Python's builtin :func:`hash` is randomized per-process (``PYTHONHASHSEED``),
so it can never be used for cross-process content identity. These helpers
produce canonical SHA-256 digests over a stable serialization:

- dicts are serialized with ``sort_keys=True``;
- numpy arrays are serialized as dtype + shape + base64(raw bytes) so the
  digest covers the exact payload bytes regardless of process;
- scalars / sequences fall back to a canonical JSON representation.

QE-P0-06 (hash codec hardening): :func:`canonicalize` is **fail-closed**:

- object-dtype ndarrays are rejected (``TypeError``) rather than silently
  hashed by an unstable representation;
- mapping keys must be ``str`` (non-string keys raise ``TypeError``);
- the old ``default=repr`` fallback is removed — unsupported object types
  raise ``TypeError`` instead of being reduced to a process-dependent repr;
- explicit codecs are provided for ``datetime``/``date``/``timedelta``,
  ``Decimal``, ``Enum``, ``bytes``, and numpy scalars.

Exposed functions:

- ``canonicalize(value) -> Any``     — JSON-friendly canonical form
- ``stable_content_hex(tag, fields) -> str``  — canonical SHA-256 hex digest
- ``stable_hash(hexdigest) -> int``  — in-range ``__hash__`` int derived from a
  digest (deterministic across processes, unlike builtin ``hash()``)
"""

from typing import Any, Mapping

import base64
import datetime
import decimal
import enum
import hashlib
import json
import sys

import numpy as np
from ._array_hash_cache import ArrayHashStateCache


def canonicalize(value: Any) -> Any:
    """Return a JSON-friendly canonical form of ``value`` (fail-closed).

    Arrays become ``{"__ndarray__": true, "dtype", "shape", "data_b64"}``.
    numpy scalars are converted to Python scalars.  Mappings must have string
    keys.  Unsupported object types raise ``TypeError`` rather than falling
    back to ``repr`` (which is not stable across processes).
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError(
                "canonicalize: object-dtype or structured-dtype ndarray is not supported (fail-closed)"
            )
        array = np.asarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, datetime.datetime):
        tz = value.tzinfo.tzname(value) if value.tzinfo else None
        return {"__datetime__": value.isoformat(), "tz": tz}
    if isinstance(value, datetime.date):
        return {"__date__": value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"__timedelta__": value.total_seconds()}
    if isinstance(value, decimal.Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, enum.Enum):
        return {
            "__enum__": [type(value).__module__, type(value).__qualname__, value.name],
        }
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        result = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    "canonicalize: mapping keys must be str, got "
                    f"{type(k).__name__} (fail-closed)"
                )
            result[k] = canonicalize(v)
        reserved = {'__ndarray__', '__datetime__', '__date__', '__timedelta__',
                    '__decimal__', '__enum__', '__bytes__', '__mapping__'}
        if reserved.intersection(result):
            # User mappings resembling a typed canonical value must not share
            # its identity. Ordinary string-key maps retain their old hashes.
            return {'__mapping__': sorted(result.items())}
        return result
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"canonicalize: unsupported type {type(value).__name__} (fail-closed)"
    )


def stable_content_hex(*, tag: str, fields: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 hex digest of ``fields`` under ``tag``.

    Deterministic across processes regardless of ``PYTHONHASHSEED``.
    """
    canonical = canonicalize(fields)
    blob = json.dumps(
        [tag, canonical],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_ARRAY_HASH_STATE_CACHE = ArrayHashStateCache()


def _bounded_probe_metadata(value, *, remaining_nodes, remaining_text):
    """Bound work before canonicalizing non-array fields in a cache probe."""
    pending = [value]
    nodes = 0
    text_size = 0
    while pending:
        current = pending.pop()
        nodes += 1
        if nodes > remaining_nodes:
            return None
        current_type = type(current)
        if current is None or current_type in (bool, float):
            continue
        if current_type is int:
            if current.bit_length() > 256:
                return None
            continue
        if current_type is str:
            text_size += len(current)
            if text_size > remaining_text:
                return None
            continue
        if current_type in (tuple, list):
            if len(current) > remaining_nodes - nodes + 1:
                return None
            pending.extend(current)
            continue
        if current_type is dict:
            if len(current) * 2 > remaining_nodes - nodes + 1:
                return None
            for key, item in current.items():
                if type(key) is not str:
                    return None
                text_size += len(key)
                if text_size > remaining_text:
                    return None
                pending.append(item)
            continue
        return None
    return nodes, text_size

def stable_content_hex_streamed_arrays(
    *, tag: str, fields: Mapping[str, Any], array_keys: tuple[str, ...],
    chunk_bytes: int = 3 * 1024 * 1024,
) -> str:
    """Hash large top-level arrays with the existing canonical JSON codec.

    C-contiguous arrays are base64-encoded in bounded chunks while preserving
    the exact digest of stable_content_hex. Other fields retain canonicalize's
    fail-closed rules. Non-contiguous arrays use the original allocating codec.
    """
    return _stable_content_hex_streamed_arrays(
        tag=tag, fields=fields, array_keys=array_keys, chunk_bytes=chunk_bytes,
        probe_only=False,
    )


def streamed_arrays_cache_ready(*, tag: str, fields: Mapping[str, Any],
                                array_keys: tuple[str, ...],
                                max_metadata_nodes: int = 4096,
                                max_metadata_text: int = 1024 * 1024) -> bool | None:
    """Return hit, miss, or unknown without hashing missing array payloads."""
    if (type(max_metadata_nodes) is not int or max_metadata_nodes < 1
            or type(max_metadata_text) is not int or max_metadata_text < 0):
        return False
    try:
        return _stable_content_hex_streamed_arrays(
            tag=tag, fields=fields, array_keys=array_keys, chunk_bytes=3 * 1024 * 1024,
            probe_only=True, max_metadata_nodes=max_metadata_nodes,
            max_metadata_text=max_metadata_text,
        )
    except Exception:
        return None


def _stable_content_hex_streamed_arrays(
    *, tag: str, fields: Mapping[str, Any], array_keys: tuple[str, ...],
    chunk_bytes: int, probe_only: bool, max_metadata_nodes: int = 4096,
    max_metadata_text: int = 1024 * 1024,
):
    if not isinstance(fields, Mapping) or not isinstance(array_keys, tuple):
        if probe_only:
            return None
        raise TypeError("fields must be a mapping and array_keys a tuple")
    if probe_only and (type(fields) is not dict or len(fields) > 512
                       or len(array_keys) > 16):
        return None
    if probe_only:
        if (type(tag) is not str or len(tag) > 256
                or any(type(key) is not str for key in fields)
                or any(type(key) is not str or len(key) > 256 for key in array_keys)
                or sum(len(key) for key in fields) > max_metadata_text
                or sum(len(key) for key in array_keys) > max_metadata_text):
            return None
    if (isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int)
            or chunk_bytes < 3):
        raise ValueError("chunk_bytes must be an integer >= 3")
    if any(not isinstance(key, str) for key in fields):
        raise TypeError("canonicalize: mapping keys must be str (fail-closed)")
    if any(not isinstance(key, str) for key in array_keys):
        raise TypeError("array_keys must contain only str")
    reserved = {'__ndarray__', '__datetime__', '__date__', '__timedelta__',
                '__decimal__', '__enum__', '__bytes__', '__mapping__'}
    if reserved.intersection(fields):
        if probe_only:
            return None
        return stable_content_hex(tag=tag, fields=fields)

    digest = hashlib.sha256()
    def emit(value: str | bytes) -> None:
        digest.update(value.encode("utf-8") if isinstance(value, str) else value)

    emit('[')
    emit(json.dumps(tag, ensure_ascii=True))
    emit(',{')
    selected = frozenset(array_keys)
    stride = chunk_bytes - chunk_bytes % 3
    metadata_nodes = 0
    metadata_text = 0
    for index, key in enumerate(sorted(fields)):
        if index:
            emit(',')
        emit(json.dumps(key, ensure_ascii=True))
        emit(':')
        value = fields[key]
        if key not in selected or not isinstance(value, np.ndarray):
            if probe_only:
                if type(key) is not str:
                    return None
                metadata_text += len(key)
                measured = _bounded_probe_metadata(
                        value, remaining_nodes=max_metadata_nodes - metadata_nodes,
                        remaining_text=max_metadata_text - metadata_text)
                if metadata_text > max_metadata_text or measured is None:
                    return None
                metadata_nodes += measured[0]
                metadata_text += measured[1]
            emit(json.dumps(canonicalize(value), sort_keys=True,
                            separators=(',', ':'), ensure_ascii=True))
            continue
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError(
                "canonicalize: object-dtype or structured-dtype ndarray is not supported (fail-closed)"
            )
        array = np.asarray(value)
        if not array.flags.c_contiguous or not array.size or array.dtype.kind in "Mm":
            if probe_only:
                return None
            emit(json.dumps(canonicalize(array), sort_keys=True,
                            separators=(',', ':'), ensure_ascii=True))
            continue
        prefix = digest.copy()
        cached_state = (_ARRAY_HASH_STATE_CACHE.peek(array, prefix) if probe_only
                        else _ARRAY_HASH_STATE_CACHE.get(array, prefix))
        if cached_state is not None:
            digest = cached_state
            continue
        if probe_only:
            if not _ARRAY_HASH_STATE_CACHE._eligible(array, _ARRAY_HASH_STATE_CACHE.min_nbytes):
                return None
            return False
        emit('{"__ndarray__":true,"data_b64":"')
        raw = memoryview(array).cast('B')
        for offset in range(0, len(raw), stride):
            emit(base64.b64encode(raw[offset:offset + stride]))
        emit('","dtype":')
        emit(json.dumps(array.dtype.str, ensure_ascii=True))
        emit(',"shape":')
        emit(json.dumps(list(array.shape), separators=(',', ':')))
        emit('}')
        _ARRAY_HASH_STATE_CACHE.store(array, prefix, digest)
    emit('}]')
    return True if probe_only else digest.hexdigest()
#: Largest value CPython allows ``hash()`` to return (``sys.hash_info.modulus``).
_HASH_MODULUS = getattr(sys.hash_info, "modulus", 2**63 - 1)


def stable_hash(hexdigest: str) -> int:
    """Reduce a canonical SHA-256 hex digest to an in-range Python hash int.

    Python requires ``__hash__`` to return an ``int``; a raw 256-bit digest
    overflows ``sys.hash_info``.  Modular reduction yields a stable,
    collision-respecting integer that is identical across processes.
    """
    return int(hexdigest, 16) % _HASH_MODULUS
