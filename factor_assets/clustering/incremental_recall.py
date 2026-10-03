"""Bounded request-local helpers for exact incremental recall."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Sequence

import numpy as np


def unit_cosine_scores(unit_query: np.ndarray, unit_matrix: np.ndarray) -> np.ndarray:
    """Clamp only Float64 dot-product endpoint roundoff, not arbitrary scores."""
    scores = unit_query @ unit_matrix.T
    tolerance = 64 * np.finfo(np.float64).eps
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.0 + tolerance):
        raise ValueError("unit cosine scores exceed the finite Float64 envelope")
    return np.clip(scores, -1.0, 1.0)


DEFAULT_MEMBER_MATRIX_CACHE_BYTES = 32 * 1024**2


class RequestLocalMemberMatrixCache:
    """Lazy LRU retaining at most the configured bytes of prepared matrices.

    The estimate includes matrix payload and conservative per-row Python list
    bookkeeping. Oversized matrices are used for the current calculation and
    released. This bounds retained cache memory, not transient workspace or
    total process RSS.
    """

    def __init__(self, *, max_bytes: int = DEFAULT_MEMBER_MATRIX_CACHE_BYTES):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, tuple[list[str], object, int]] = OrderedDict()
        self.retained_bytes = 0

    def get(self, key: str, loader: Callable[[], tuple[list[str], Optional[object]]]):
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached[0], cached[1]
        ids, matrix = loader()
        if not ids or matrix is None:
            return ids, matrix
        cost = int(matrix.nbytes + 64 * len(ids) + 256)
        if cost > self._max_bytes:
            return ids, matrix
        while self._entries and self.retained_bytes + cost > self._max_bytes:
            _key, (_ids, _matrix, evicted_bytes) = self._entries.popitem(last=False)
            self.retained_bytes -= evicted_bytes
        self._entries[key] = (ids, matrix, cost)
        self.retained_bytes += cost
        return ids, matrix
