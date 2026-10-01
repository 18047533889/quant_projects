"""Canonical QE identity for ndarray inputs."""

import hashlib
import json

import numpy as np

from quant_evaluator.contracts._array_hash_cache import ArrayHashStateCache


# Separate from the JSON serializer's cache: this memo is only for the raw
# payload segment in QE's historical authoritative-array digest.
_RAW_ARRAY_HASH_STATE_CACHE = ArrayHashStateCache(capacity=32)


def _update_raw_payload(digest, array: np.ndarray) -> None:
    """Feed C-order array bytes without materializing a second full copy."""
    digest.update(memoryview(array.reshape(-1).view(np.uint8)))


def authoritative_array_hash(value: np.ndarray) -> str:
    """Hash ndarray values using the historical dtype+shape+payload contract.

    Numeric and other non-object arrays hash exact C-order bytes. Object arrays
    use the lossless element codec; structured arrays fail closed. Only exact,
    contiguous arrays backed by immutable bytes can reuse a raw-payload SHA
    state. All other inputs follow the uncached historical path.
    """
    original = np.asarray(value)
    array = np.ascontiguousarray(original)
    digest = hashlib.sha256()
    digest.update(str(original.dtype).encode())
    digest.update(str(original.shape).encode())
    if array.dtype.hasobject:
        from quant_evaluator.contracts._ndarray_codec import encode_ndarray

        encoded = encode_ndarray(original)
        digest.update(json.dumps(
            encoded["object_values"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
    elif array.dtype.fields is not None:
        raise TypeError("structured arrays require an explicit schema codec")
    else:
        # Scalar and non-contiguous values retain the existing contiguous-copy
        # behavior and skip memoization. For eligible inputs, preserve this
        # exact prefix in the cache key, then save the state after raw bytes.
        cacheable_layout = original.ndim > 0 and original.flags.c_contiguous
        prefix = digest.copy() if cacheable_layout else None
        if prefix is not None:
            cached = _RAW_ARRAY_HASH_STATE_CACHE.get(original, prefix)
            if cached is not None:
                return cached.hexdigest()
        _update_raw_payload(digest, array)
        if prefix is not None:
            _RAW_ARRAY_HASH_STATE_CACHE.store(original, prefix, digest)
    return digest.hexdigest()
