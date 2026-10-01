import gc
import hashlib
import multiprocessing
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from quant_evaluator.contracts._array_hash_cache import ArrayHashStateCache
from quant_evaluator.contracts._hashutil import (
    stable_content_hex,
    stable_content_hex_streamed_arrays,
)


def _immutable_array(values):
    return np.frombuffer(np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64)


def _digest(array, *, prefix="p", suffix="s"):
    return stable_content_hex_streamed_arrays(
        tag="qe.array-hash-cache.test",
        fields={"a": prefix, "factor_values": array, "z": suffix},
        array_keys=("factor_values",), chunk_bytes=9,
    )


def test_cached_encoding_is_byte_exact_and_suffix_is_still_hashed(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(
        "quant_evaluator.contracts._hashutil._ARRAY_HASH_STATE_CACHE", cache,
    )
    array = _immutable_array([1.0, 2.0, 3.0, 4.0])
    for prefix, suffix in (("p", "s1"), ("p", "s2"), ("p2", "s2")):
        fields = {"a": prefix, "factor_values": array, "z": suffix}
        assert _digest(array, prefix=prefix, suffix=suffix) == stable_content_hex(
            tag="qe.array-hash-cache.test", fields=fields,
        )
    assert len(cache._entries) == 2


def test_cache_hit_skips_array_encoding(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(
        "quant_evaluator.contracts._hashutil._ARRAY_HASH_STATE_CACHE", cache,
    )
    import quant_evaluator.contracts._hashutil as hashutil
    original = hashutil.base64.b64encode
    calls = []

    def counted(payload):
        calls.append(len(payload))
        return original(payload)

    monkeypatch.setattr(hashutil.base64, "b64encode", counted)
    array = _immutable_array(np.arange(16, dtype=np.float64))
    first = _digest(array)
    first_call_count = len(calls)
    assert first_call_count > 0
    assert _digest(array) == first
    assert len(calls) == first_call_count


def test_readonly_mutable_backing_is_not_cached_and_changes_are_seen(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(
        "quant_evaluator.contracts._hashutil._ARRAY_HASH_STATE_CACHE", cache,
    )
    backing = bytearray(np.asarray([1.0, 2.0], dtype=np.float64).tobytes())
    array = np.frombuffer(backing, dtype=np.float64)
    array.flags.writeable = False
    assert not array.flags.writeable
    assert not cache._eligible(array, 0)
    before = _digest(array)
    backing[:8] = np.asarray([8.0], dtype=np.float64).tobytes()
    assert _digest(array) != before
    assert len(cache._entries) == 0


def test_ineligible_array_owners_layouts_and_dtypes(tmp_path):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)

    backing = bytearray(np.asarray([1.0, 2.0], dtype=np.float64).tobytes())
    readonly_view = np.frombuffer(memoryview(backing).toreadonly(), dtype=np.float64)
    assert not readonly_view.flags.writeable

    mapped_path = tmp_path / "mapped.dat"
    writable_map = np.memmap(mapped_path, dtype=np.float64, mode="w+", shape=(2,))
    writable_map[:] = [1.0, 2.0]
    writable_map.flush()
    del writable_map
    mapped = np.memmap(mapped_path, dtype=np.float64, mode="r", shape=(2,))

    class ArraySubclass(np.ndarray):
        pass

    subclass = _immutable_array([1.0, 2.0]).view(ArraySubclass)
    base = _immutable_array([1.0, 2.0, 3.0, 4.0])
    noncontiguous = base[::2]
    object_array = np.array([object()], dtype=object)
    structured = np.array([(1, 2.0)], dtype=[("i", "i4"), ("x", "f8")])
    datetime_array = np.array(["2020-01-01"], dtype="datetime64[D]")

    for array in (readonly_view, mapped, subclass, noncontiguous,
                  object_array, structured, datetime_array):
        assert not cache._eligible(array, 0)


def test_shape_and_dtype_mutation_of_same_immutable_array_misses_cache(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(
        "quant_evaluator.contracts._hashutil._ARRAY_HASH_STATE_CACHE", cache,
    )
    array = _immutable_array([1.0, 2.0, 3.0, 4.0])
    prefix = hashlib.sha256(b"same-prefix")
    cache.store(array, prefix, hashlib.sha256(b"completed"))

    original = _digest(array)
    array.shape = (2, 2)
    reshaped = _digest(array)
    assert reshaped != original
    assert reshaped == stable_content_hex(
        tag="qe.array-hash-cache.test",
        fields={"a": "p", "factor_values": array, "z": "s"},
    )

    array.dtype = np.dtype(">f8")
    retyped = _digest(array)
    assert retyped != reshaped
    assert retyped == stable_content_hex(
        tag="qe.array-hash-cache.test",
        fields={"a": "p", "factor_values": array, "z": "s"},
    )


def test_invalid_cache_configuration():
    for value in (-1, True, 1.5, "1"):
        with pytest.raises(ValueError):
            ArrayHashStateCache(min_nbytes=value)
    for value in (0, -1, True, 1.5, "1"):
        with pytest.raises(ValueError):
            ArrayHashStateCache(capacity=value)


def test_identity_and_metadata_are_part_of_cache_key():
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    first = _immutable_array([1.0, 2.0, 3.0, 4.0])
    prefix = hashlib.sha256(b"prefix")
    assert cache.get(first, prefix) is None
    completed = hashlib.sha256(b"completed")
    cache.store(first, prefix, completed)
    assert cache.get(first, prefix) is not None
    assert cache.get(first.reshape(2, 2), prefix) is None
    assert cache.get(_immutable_array([1.0, 2.0, 3.0, 4.0]), prefix) is None
    assert cache.get(first, hashlib.sha256(b"other-prefix")) is None


def test_weak_collection_and_lru_bound():
    cache = ArrayHashStateCache(min_nbytes=0, capacity=2)
    arrays = [_immutable_array([float(i)]) for i in range(3)]
    refs = [weakref.ref(array) for array in arrays]
    for array in arrays:
        cache.store(array, hashlib.sha256(b"p"), hashlib.sha256(b"done"))
    assert len(cache._entries) == 2
    arrays.clear()
    del array
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert len(cache._entries) == 0


def test_concurrent_hashes_are_exact(monkeypatch):
    cache = ArrayHashStateCache(min_nbytes=0, capacity=8)
    monkeypatch.setattr(
        "quant_evaluator.contracts._hashutil._ARRAY_HASH_STATE_CACHE", cache,
    )
    array = _immutable_array(np.arange(64, dtype=np.float64))
    expected = stable_content_hex(
        tag="qe.array-hash-cache.test",
        fields={"a": "p", "factor_values": array, "z": "s"},
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(lambda _: _digest(array), range(32))) == {expected}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_clears_entries_and_replaces_inherited_lock():
    cache = ArrayHashStateCache(min_nbytes=0, capacity=4)
    array = _immutable_array([1.0])
    cache.store(array, hashlib.sha256(b"p"), hashlib.sha256(b"done"))
    assert len(cache._entries) == 1
    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        with cache._lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(2)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def check_child():
        with cache._lock:
            sender.send(len(cache._entries) == 0)
        sender.close()

    child = context.Process(target=check_child)
    try:
        child.start()
        assert receiver.poll(3)
        assert receiver.recv() is True
        child.join(1)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(1)
        release.set()
        holder.join(1)
        receiver.close()
        sender.close()
