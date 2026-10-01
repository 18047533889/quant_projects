"""Small process-local cache for completed immutable-array hash segments."""

import os
import threading
import weakref
from collections import OrderedDict
from typing import Any

import numpy as np


class ArrayHashStateCache:
    """Memoize SHA states after encoding immutable byte-backed arrays.

    The cache retains weak references to arrays and copies of hashlib state;
    it never owns array storage or serialized payloads.
    """

    def __init__(self, *, min_nbytes: int = 8 * 1024 * 1024, capacity: int = 32):
        if type(min_nbytes) is not int or min_nbytes < 0:
            raise ValueError("min_nbytes must be a nonnegative integer")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.min_nbytes = min_nbytes
        self.capacity = capacity
        self._lock = threading.RLock()
        self._entries = OrderedDict()
        _CACHES.add(self)

    @staticmethod
    def _eligible(array: np.ndarray, min_nbytes: int) -> bool:
        if (type(array) is not np.ndarray or array.nbytes < min_nbytes
                or not array.size or not array.flags.c_contiguous
                or array.flags.writeable or array.dtype.hasobject
                or array.dtype.fields is not None
                or array.dtype.kind not in "biufc"):
            return False
        owner: Any = array
        seen = set()
        while isinstance(owner, np.ndarray):
            if type(owner) is not np.ndarray or id(owner) in seen:
                return False
            seen.add(id(owner))
            owner = owner.base
        # A readonly flag alone is insufficient: ndarray flags can be reset
        # when the underlying storage is mutable.
        return type(owner) is bytes

    @staticmethod
    def _key(array: np.ndarray, prefix: Any) -> tuple:
        return (prefix.hexdigest(), id(array), array.dtype.str, array.shape,
                array.strides, array.nbytes)

    def get(self, array: np.ndarray, prefix: Any) -> Any | None:
        if not self._eligible(array, self.min_nbytes):
            return None
        key = self._key(array, prefix)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry[0]() is not array:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry[1].copy()

    def peek(self, array: np.ndarray, prefix: Any) -> Any | None:
        """Copy a cached digest state without changing LRU order or contents."""
        if not self._eligible(array, self.min_nbytes):
            return None
        key = self._key(array, prefix)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0]() is not array:
                return None
            return entry[1].copy()

    def store(self, array: np.ndarray, prefix: Any, completed: Any) -> None:
        if not self._eligible(array, self.min_nbytes):
            return
        key = self._key(array, prefix)
        cache_ref = weakref.ref(self)

        def discard(dead_ref, cache_ref=cache_ref, key=key):
            cache = cache_ref()
            if cache is None:
                return
            with cache._lock:
                current = cache._entries.get(key)
                if current is not None and current[0] is dead_ref:
                    del cache._entries[key]

        array_ref = weakref.ref(array, discard)
        with self._lock:
            self._entries[key] = (array_ref, completed.copy())
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def _after_fork_child(self) -> None:
        self._lock = threading.RLock()
        self._entries.clear()


_CACHES = weakref.WeakSet()


def _reset_after_fork_child() -> None:
    for cache in list(_CACHES):
        cache._after_fork_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork_child)
