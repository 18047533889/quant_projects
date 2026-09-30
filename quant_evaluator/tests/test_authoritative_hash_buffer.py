"""The zero-copy numeric hash retains the historical byte contract."""
import hashlib

import numpy as np
import pytest

from quant_evaluator.runtime.evaluator import authoritative_array_hash


@pytest.mark.parametrize("value", [
    np.array(3.0),
    np.empty((0, 3)),
    np.empty((2, 0, 4), dtype=np.int64),
    np.array([np.nan, np.inf, -np.inf, -0.0, 0.0]),
    np.arange(24).reshape(4, 6)[:, ::2],
    np.asfortranarray(np.arange(24).reshape(4, 6)),
    np.arange(12)[::-1],
    np.array([1, 2], dtype=">i8"),
    np.array([True, False]),
    np.array([1 + 2j, 3 - 4j]),
    np.array(["2026-01-01"], dtype="datetime64[D]"),
    np.array(["abc", "xyz"], dtype="U3"),
])
def test_numeric_hash_matches_historical_recipe(value):
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    assert authoritative_array_hash(value) == digest.hexdigest()


def test_numeric_hash_does_not_call_tobytes(monkeypatch):
    class NoBytesArray(np.ndarray):
        def tobytes(self, *args, **kwargs):
            raise AssertionError("full-panel allocation is forbidden")
    value = np.arange(12)
    expected = authoritative_array_hash(value)
    original = np.ascontiguousarray
    monkeypatch.setattr(np, "ascontiguousarray", lambda a: original(a).view(NoBytesArray))
    assert authoritative_array_hash(value) == expected
