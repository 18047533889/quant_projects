"""Bounded result-row admission for exact-flat matrix search."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")
from factor_assets.similarity.ann import FaissANNIndex


def _index():
    index = FaissANNIndex(3)
    index.build(
        ["A", "B", "C"],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    return index


def test_single_query_batch_matches_scalar_search():
    index = _index()
    query = np.array([[0.8, 0.6, 0.0]])
    batch = index.search_batch(query, k=2)
    scalar = index.search(query[0], k=2)

    assert len(batch) == 1
    assert [item.factor_id for item in batch[0]] == [item.factor_id for item in scalar]
    assert [item.similarity_score for item in batch[0]] == pytest.approx(
        [item.similarity_score for item in scalar]
    )


def test_empty_query_batch_returns_empty_without_backend_call():
    index = _index()

    class SearchSpy:
        calls = 0

        def search(self, queries, k):
            self.calls += 1
            raise AssertionError("empty batch reached FAISS")

    spy = SearchSpy()
    index._index = spy
    assert index.search_batch(np.empty((0, 3)), k=3) == []
    assert spy.calls == 0


def test_invalid_limit_rejected_for_empty_query_batch():
    with pytest.raises(ValueError, match="positive built-in integer"):
        _index().search_batch(np.empty((0, 3)), k=3, max_result_rows=0)


def test_invalid_limit_rejected_for_empty_index():
    empty_index = FaissANNIndex(3)
    with pytest.raises(ValueError, match="positive built-in integer"):
        empty_index.search_batch(np.ones((1, 3)), max_result_rows=0)


def test_exact_result_row_limit_is_admitted_once():
    index = _index()
    real_index = index._index

    class SearchSpy:
        calls = 0

        def search(self, queries, k):
            self.calls += 1
            return real_index.search(queries, k)

    spy = SearchSpy()
    index._index = spy
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = index.search_batch(queries, k=1, max_result_rows=2)

    assert spy.calls == 1
    assert len(result) == 2
    assert all(len(rows) == 1 for rows in result)


def test_over_limit_uses_effective_k_and_rejects_before_backend():
    index = _index()

    class SearchSpy:
        calls = 0

        def search(self, queries, k):
            self.calls += 1
            raise AssertionError("over-limit batch reached FAISS")

    spy = SearchSpy()
    index._index = spy
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # k=99 is clamped to three indexed rows: 2 queries × 3 results = 6.
    with pytest.raises(ValueError, match="chunk queries"):
        index.search_batch(queries, k=99, max_result_rows=5)
    assert spy.calls == 0


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, None, 2_000_001])
def test_result_row_limit_must_be_bounded_positive_builtin_int(limit):
    index = _index()
    with pytest.raises(ValueError):
        index.search_batch(
            np.array([[1.0, 0.0, 0.0]]), k=1, max_result_rows=limit,
        )


def test_hard_result_limit_itself_is_a_valid_option():
    index = _index()
    result = index.search_batch(
        np.array([[1.0, 0.0, 0.0]]), k=1, max_result_rows=2_000_000,
    )
    assert len(result) == 1
    assert result[0][0].factor_id == "A"
