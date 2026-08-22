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


def canonicalize(value: Any) -> Any:
    """Return a JSON-friendly canonical form of ``value`` (fail-closed).

    Arrays become ``{"__ndarray__": true, "dtype", "shape", "data_b64"}``.
    numpy scalars are converted to Python scalars.  Mappings must have string
    keys.  Unsupported object types raise ``TypeError`` rather than falling
    back to ``repr`` (which is not stable across processes).
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                "canonicalize: object-dtype ndarray is not supported (fail-closed)"
            )
        array = np.ascontiguousarray(value)
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
    canonical = {str(k): canonicalize(v) for k, v in fields.items()}
    blob = json.dumps(
        [tag, canonical],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


#: Largest value CPython allows ``hash()`` to return (``sys.hash_info.modulus``).
_HASH_MODULUS = getattr(sys.hash_info, "modulus", 2**63 - 1)


def stable_hash(hexdigest: str) -> int:
    """Reduce a canonical SHA-256 hex digest to an in-range Python hash int.

    Python requires ``__hash__`` to return an ``int``; a raw 256-bit digest
    overflows ``sys.hash_info``.  Modular reduction yields a stable,
    collision-respecting integer that is identical across processes.
    """
    return int(hexdigest, 16) % _HASH_MODULUS
