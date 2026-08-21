"""Stable content hashing utilities for quant_evaluator contracts.

Python's builtin :func:`hash` is randomized per-process (``PYTHONHASHSEED``),
so it can never be used for cross-process content identity. These helpers
produce canonical SHA-256 digests over a stable serialization:

- dicts are serialized with ``sort_keys=True``;
- numpy arrays are serialized as dtype + shape + base64(raw bytes) so the
  digest covers the exact payload bytes regardless of process;
- scalars / sequences fall back to a canonical JSON representation.

Exposed functions:

- ``canonicalize(value) -> Any``     — JSON-friendly canonical form
- ``stable_content_hex(tag, fields) -> str``  — canonical SHA-256 hex digest
- ``stable_hash(hexdigest) -> int``  — in-range ``__hash__`` int derived from a
  digest (deterministic across processes, unlike builtin ``hash()``)
"""

from typing import Any, Mapping

import base64
import hashlib
import json
import sys

import numpy as np


def canonicalize(value: Any) -> Any:
    """Return a JSON-friendly canonical form of ``value``.

    Arrays become ``{"__ndarray__": true, "dtype", "shape", "data_b64"}``.
    numpy scalars are converted to Python scalars.  Mappings are keyed by
    ``str(key)`` so mixed key types cannot produce unstable output.
    """
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    return value


def stable_content_hex(*, tag: str, fields: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 hex digest of ``fields`` under ``tag``.

    Deterministic across processes regardless of ``PYTHONHASHSEED``.
    """
    canonical = {str(k): canonicalize(v) for k, v in fields.items()}
    blob = json.dumps(
        [tag, canonical],
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
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
