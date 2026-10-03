"""Bounded request-local helpers for exact incremental recall."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Sequence

import numpy as np


def _validated_unit_cosine_scores(scores: np.ndarray) -> np.ndarray:
    tolerance = 64 * np.finfo(np.float64).eps
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.0 + tolerance):
        raise ValueError("unit cosine scores exceed the finite Float64 envelope")
    return np.clip(scores, -1.0, 1.0)


def unit_cosine_scores(unit_query: np.ndarray, unit_matrix: np.ndarray) -> np.ndarray:
    """Clamp only Float64 dot-product endpoint roundoff, not arbitrary scores."""
    scores = unit_query @ unit_matrix.T
    return _validated_unit_cosine_scores(scores)


def _rowwise_unit_cosine_scores(
    unit_query: np.ndarray, unit_matrix: np.ndarray,
) -> np.ndarray:
    """Score rows with a fixed reduction independent of chunk batch shape."""
    scores = np.einsum("ij,j->i", unit_matrix, unit_query, optimize=False)
    return _validated_unit_cosine_scores(scores)


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


DEFAULT_MEMBER_SCAN_CHUNK_ROWS = 256
DEFAULT_MEMBER_SCAN_CHUNK_BYTES = 4 * 1024**2
_MEMBER_SCAN_SCRATCH_ARRAYS = 3


def scan_exact_member_winner(
    fingerprints_by_id,
    member_ids: Sequence[str],
    unit_query: np.ndarray,
    *,
    to_embedding: Callable[[object], np.ndarray],
    normalize_rows: Callable[[np.ndarray], np.ndarray],
    max_chunk_rows: int = DEFAULT_MEMBER_SCAN_CHUNK_ROWS,
    max_chunk_bytes: int = DEFAULT_MEMBER_SCAN_CHUNK_BYTES,
) -> Optional[tuple[str, float]]:
    """Scan exact member cosine scores with bounded row-matrix scratch.

    Returns ``(factor_id, similarity)`` for the first member attaining the
    maximum score, or ``None`` when no requested member has a fingerprint.
    Missing members are skipped and fingerprint-key identity is checked just
    as in the full-matrix collection path. Every chunk is scored and validated
    before moving on, even after a provisional winner has been found.

    ``max_chunk_bytes`` reserves a conservative three-array numeric workspace
    for row assembly, normalization, and scoring; it is not a total RSS limit.
    At least one row is processed at a time, so a single embedding wider than
    that budget remains a one-row operation rather than materializing a cluster.
    """
    if not isinstance(unit_query, np.ndarray) or unit_query.ndim != 1:
        raise ValueError("exact unit recall query must be a vector")
    if isinstance(max_chunk_rows, bool) or not isinstance(max_chunk_rows, int) or max_chunk_rows <= 0:
        raise ValueError("max_chunk_rows must be a positive integer")
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be a positive integer")

    width = int(unit_query.size)
    row_bytes = width * np.dtype(np.float64).itemsize
    rows_by_bytes = max(
        1,
        max_chunk_bytes // max(1, row_bytes * _MEMBER_SCAN_SCRATCH_ARRAYS),
    )
    chunk_capacity = min(max_chunk_rows, rows_by_bytes)

    winner_id: Optional[str] = None
    winner_score = -np.inf
    chunk_ids: list[str] = []
    chunk_rows = np.empty((chunk_capacity, width), dtype=np.float64)
    used_rows = 0

    def score_chunk(count: int) -> None:
        nonlocal winner_id, winner_score
        unit_matrix = normalize_rows(chunk_rows[:count])
        scores = _rowwise_unit_cosine_scores(unit_query, unit_matrix)
        position = int(np.argmax(scores))
        score = float(scores[position])
        # Strict comparison keeps the first member on equal rowwise scores;
        # no tolerance is used, so a strictly greater near-tie can still win.
        if winner_id is None or score > winner_score:
            winner_id = chunk_ids[position]
            winner_score = score

    for fid in member_ids:
        fingerprint = fingerprints_by_id.get(fid)
        if fingerprint is None:
            continue
        if fingerprint.factor_id != fid:
            raise ValueError("fingerprint mapping key must match fingerprint.factor_id")
        embedding = np.asarray(to_embedding(fingerprint), dtype=np.float64)
        if embedding.ndim != 1 or embedding.shape != (width,):
            raise ValueError("member embedding must be a vector matching the query width")
        chunk_rows[used_rows] = embedding
        chunk_ids.append(fid)
        used_rows += 1
        if used_rows == chunk_capacity:
            score_chunk(used_rows)
            used_rows = 0
            chunk_ids.clear()

    if used_rows:
        score_chunk(used_rows)
    if winner_id is None:
        return None
    return winner_id, winner_score

DEFAULT_MEMBER_SCAN_MAX_BATCH_QUERIES = 256


def scan_exact_member_winners(
    fingerprints_by_id,
    member_ids: Sequence[str],
    unit_queries: np.ndarray,
    *,
    to_embedding: Callable[[object], np.ndarray],
    normalize_rows: Callable[[np.ndarray], np.ndarray],
    max_chunk_rows: int = DEFAULT_MEMBER_SCAN_CHUNK_ROWS,
    max_chunk_bytes: int = DEFAULT_MEMBER_SCAN_CHUNK_BYTES,
    max_batch_queries: int = DEFAULT_MEMBER_SCAN_MAX_BATCH_QUERIES,
) -> list[Optional[tuple[str, float]]]:
    """Return exact member winners for a bounded query batch.

    Each member embedding is converted once and each member chunk is normalized
    once for the whole query batch. Queries are then scored one at a time with
    the same fixed rowwise Float64 reduction as :func:`scan_exact_member_winner`;
    no query-by-member score matrix is formed.

    Output order matches the query rows. Missing members are skipped, and each
    query keeps the first member attaining the maximum score. Every member
    chunk is scored and validated for every query, including chunks after a
    provisional winner. The query array is bounded by ``max_batch_queries``; member
    scratch follows the same row and byte caps as the single-query helper.
    These limits cover helper-owned workspace, not input arrays or total RSS.
    """
    if not isinstance(unit_queries, np.ndarray) or unit_queries.ndim != 2:
        raise ValueError("exact unit recall queries must be a two-dimensional matrix")
    if unit_queries.dtype.kind not in "fiu" or not np.isfinite(unit_queries).all():
        raise ValueError("exact unit recall queries must be finite real numeric values")
    if unit_queries.shape[1] <= 0:
        raise ValueError("exact unit recall query width must be positive")
    if (isinstance(max_batch_queries, bool) or not isinstance(max_batch_queries, int)
            or max_batch_queries <= 0):
        raise ValueError("max_batch_queries must be a positive integer")
    if unit_queries.shape[0] > max_batch_queries:
        raise ValueError(
            f"exact unit recall query batch exceeds {max_batch_queries} rows"
        )
    if isinstance(max_chunk_rows, bool) or not isinstance(max_chunk_rows, int) or max_chunk_rows <= 0:
        raise ValueError("max_chunk_rows must be a positive integer")
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be a positive integer")

    query_count, width = unit_queries.shape
    if query_count == 0:
        return []
    row_bytes = width * np.dtype(np.float64).itemsize
    rows_by_bytes = max(
        1,
        max_chunk_bytes // max(1, row_bytes * _MEMBER_SCAN_SCRATCH_ARRAYS),
    )
    chunk_capacity = min(max_chunk_rows, rows_by_bytes)

    winner_ids: list[Optional[str]] = [None] * query_count
    winner_scores = np.full(query_count, -np.inf, dtype=np.float64)
    chunk_ids: list[str] = []
    chunk_rows = np.empty((chunk_capacity, width), dtype=np.float64)
    used_rows = 0

    def score_chunk(count: int) -> None:
        unit_matrix = normalize_rows(chunk_rows[:count])
        for query_index in range(query_count):
            scores = _rowwise_unit_cosine_scores(
                unit_queries[query_index], unit_matrix
            )
            position = int(np.argmax(scores))
            score = float(scores[position])
            # Strict comparison preserves the first member on exact ties.
            if winner_ids[query_index] is None or score > winner_scores[query_index]:
                winner_ids[query_index] = chunk_ids[position]
                winner_scores[query_index] = score

    for fid in member_ids:
        fingerprint = fingerprints_by_id.get(fid)
        if fingerprint is None:
            continue
        if fingerprint.factor_id != fid:
            raise ValueError("fingerprint mapping key must match fingerprint.factor_id")
        embedding = np.asarray(to_embedding(fingerprint), dtype=np.float64)
        if embedding.ndim != 1 or embedding.shape != (width,):
            raise ValueError("member embedding must be a vector matching the query width")
        chunk_rows[used_rows] = embedding
        chunk_ids.append(fid)
        used_rows += 1
        if used_rows == chunk_capacity:
            score_chunk(used_rows)
            used_rows = 0
            chunk_ids.clear()

    if used_rows:
        score_chunk(used_rows)
    return [None if fid is None else (fid, float(winner_scores[i]))
            for i, fid in enumerate(winner_ids)]
