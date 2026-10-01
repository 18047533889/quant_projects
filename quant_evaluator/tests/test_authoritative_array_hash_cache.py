import hashlib
import json

import numpy as np
import pytest

from quant_evaluator.contracts import array_identity
from quant_evaluator.contracts._array_hash_cache import ArrayHashStateCache
from quant_evaluator.contracts._ndarray_codec import encode_ndarray
from quant_evaluator.runtime.evaluator import authoritative_array_hash


def _historical_hash(value):
    original = np.asarray(value)
    array = np.ascontiguousarray(original)
    digest = hashlib.sha256()
    digest.update(str(original.dtype).encode())
    digest.update(str(original.shape).encode())
    if array.dtype.hasobject:
        encoded = encode_ndarray(original)
        digest.update(json.dumps(
            encoded["object_values"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
    elif array.dtype.fields is not None:
        raise TypeError("structured arrays require an explicit schema codec")
    else:
        digest.update(memoryview(array.reshape(-1).view(np.uint8)))
    return digest.hexdigest()


@pytest.mark.parametrize("value", [
    np.asarray(7, dtype=np.int64),
    np.arange(12, dtype=np.float32).reshape(3, 4),
    np.asarray([1, 2, 3], dtype=">i8"),
    np.asarray([1, 2, 3], dtype="<i8"),
    np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
    np.arange(16, dtype=np.float64).reshape(4, 4)[::2, ::2],
    np.frombuffer(np.asarray([
        0x7ff8000000000001, 0x8000000000000000,
        0x0000000000000000, 0x7ff8000000000042,
    ], dtype=np.uint64).tobytes(), dtype=np.float64),
    np.array(["2026-01-01"], dtype="datetime64[D]"),
    np.array(["alpha", "beta"], dtype="U5"),
])
def test_authoritative_hash_matches_historical_bytes(value):
    assert authoritative_array_hash(value) == _historical_hash(value)


def test_object_codec_and_structured_failure_are_unchanged():
    left = np.array([("asset", 1), {"session": "A"}], dtype=object)
    assert authoritative_array_hash(left) == _historical_hash(left)
    structured = np.array([(1, 2.0)], dtype=[("i", "i4"), ("x", "f8")])
    with pytest.raises(TypeError, match="structured arrays"):
        authoritative_array_hash(structured)


def test_immutable_payload_hit_skips_raw_hash_helper(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=32)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", cache)
    original = array_identity._update_raw_payload
    calls = []

    def counted(digest, array):
        calls.append(array.nbytes)
        original(digest, array)

    monkeypatch.setattr(array_identity, "_update_raw_payload", counted)
    values = np.arange(32, dtype=np.float64)
    immutable = np.frombuffer(values.tobytes(), dtype=values.dtype)
    expected = _historical_hash(immutable)
    assert authoritative_array_hash(immutable) == expected
    assert len(calls) == 1
    assert authoritative_array_hash(immutable) == expected
    assert len(calls) == 1


def test_raw_hash_cache_state_distinguishes_eligible_cold_and_resident(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", cache)
    values = np.arange(8, dtype=np.float64)
    immutable = np.frombuffer(values.tobytes(), dtype=values.dtype)
    mutable = np.arange(8, dtype=np.float64)

    assert array_identity.authoritative_array_hash_cache_state(immutable) is False
    assert array_identity.authoritative_array_hash_cache_state(mutable) is None
    authoritative_array_hash(immutable)
    assert array_identity.authoritative_array_hash_cache_state(immutable) is True


def test_readonly_mutable_backing_bypasses_cache_and_observes_changes(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=32)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", cache)
    backing = bytearray(np.asarray([1.0, 2.0], dtype=np.float64).tobytes())
    readonly = np.frombuffer(memoryview(backing).toreadonly(), dtype=np.float64)
    assert not readonly.flags.writeable
    before = authoritative_array_hash(readonly)
    backing[:8] = np.asarray([8.0], dtype=np.float64).tobytes()
    after = authoritative_array_hash(readonly)
    assert before != after
    assert len(cache._entries) == 0


def test_same_object_shape_and_dtype_changes_miss_cache(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=32)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", cache)
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    immutable = np.frombuffer(values.tobytes(), dtype=values.dtype)

    first = authoritative_array_hash(immutable)
    assert first == _historical_hash(immutable)
    immutable.shape = (2, 2)
    reshaped = authoritative_array_hash(immutable)
    assert reshaped == _historical_hash(immutable)
    assert reshaped != first

    immutable.dtype = np.dtype(">f8")
    retyped = authoritative_array_hash(immutable)
    assert retyped == _historical_hash(immutable)
    assert retyped != reshaped
    assert len(cache._entries) == 3


def test_mutable_subclass_and_noncontiguous_arrays_do_not_enter_raw_cache(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=32)
    monkeypatch.setattr(array_identity, "_RAW_ARRAY_HASH_STATE_CACHE", cache)

    mutable = np.arange(8, dtype=np.float64)
    class ArraySubclass(np.ndarray):
        pass
    subclass = mutable.view(ArraySubclass)
    frozen = np.frombuffer(mutable.tobytes(), dtype=mutable.dtype)
    strided = frozen[::2]
    scalar = np.asarray(3.0)

    for value in (mutable, subclass, strided, scalar):
        assert authoritative_array_hash(value) == _historical_hash(value)
    assert len(cache._entries) == 0


def test_f32_coverage_readiness_requires_raw_and_config_json_cache_hits(monkeypatch):
    from types import SimpleNamespace
    import quant_evaluator.contracts._hashutil as hashutil
    from quant_evaluator.runtime.evaluator import (
        _apply_f32_coverage_cache_route,
        _f32_coverage_identity_cache_ready,
    )

    batch = SimpleNamespace(values=np.empty((1,)), validity=np.empty((1,), dtype=bool))
    label = SimpleNamespace(values=np.empty((1,)))
    raw_state = {"hit": True}
    json_state = {"hit": True}
    monkeypatch.setattr(
        array_identity, "authoritative_array_hash_cache_state",
        lambda _: raw_state["hit"],
    )
    monkeypatch.setattr(
        hashutil, "streamed_arrays_cache_ready",
        lambda **_: json_state["hit"],
    )

    ready = _f32_coverage_identity_cache_ready(batch, label, {})
    assert ready
    assert _apply_f32_coverage_cache_route(
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage", ready,
    ) == ("cuda_strict", "certified_single_metric_real_cos_f32_coverage")
    json_state["hit"] = False
    ready = _f32_coverage_identity_cache_ready(batch, label, {})
    assert not ready
    assert _apply_f32_coverage_cache_route(
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage", ready,
    )[0] == "cpu"
    json_state["hit"] = True
    raw_state["hit"] = False
    ready = _f32_coverage_identity_cache_ready(batch, label, {})
    assert not ready
    assert _apply_f32_coverage_cache_route(
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage", ready,
    )[0] == "cpu"
    raw_state["hit"] = True
    json_state["hit"] = None
    assert not _f32_coverage_identity_cache_ready(batch, label, {})
    assert _apply_f32_coverage_cache_route(
        "cuda_strict", "certified_single_metric_real_cos_f32_coverage", False,
    )[0] == "cpu"
    assert _apply_f32_coverage_cache_route("cpu", "explicit", False) == ("cpu", "explicit")
