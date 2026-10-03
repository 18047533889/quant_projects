"""Bounded exact-flat batch search tests with an independent cosine oracle."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")
from factor_assets.similarity.ann import FaissANNIndex


def _reference_unit_rows(values: np.ndarray) -> np.ndarray:
    """Independent scale-safe float64 -> float32 FAISS normalization model."""
    wide = np.asarray(values, dtype=np.float64)
    scaled = wide / np.max(np.abs(wide), axis=1, keepdims=True)
    unit64 = scaled / np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    unit32 = np.ascontiguousarray(unit64, dtype=np.float32)
    norms32 = np.sqrt(np.sum(unit32 * unit32, axis=1, keepdims=True, dtype=np.float32))
    return np.ascontiguousarray(unit32 / norms32, dtype=np.float32)


def _result_map(results):
    return {result.factor_id: result.similarity_score for result in results}


def test_faiss_batch_matches_independent_oracle_and_scalar_loop():
    rng = np.random.default_rng(20261003)
    library_size, query_count, dimensions = 1000, 32, 24
    ids = [f"F{i:04d}" for i in range(library_size)]
    embeddings = rng.normal(size=(library_size, dimensions)).astype(np.float64)
    embeddings[1] = embeddings[0]
    embeddings[2] = embeddings[0]
    embeddings[3] = -embeddings[0]
    queries = rng.normal(size=(query_count, dimensions)).astype(np.float64)
    queries[0] = embeddings[0]
    queries[1] = -embeddings[0]
    embeddings_before, queries_before = embeddings.copy(), queries.copy()

    index = FaissANNIndex(dimensions)
    index.build(ids, embeddings)
    batch = index.search_batch(queries, k=library_size)
    scalar = [index.search(query, k=library_size) for query in queries]

    assert len(batch) == query_count
    assert np.array_equal(embeddings, embeddings_before)
    assert np.array_equal(queries, queries_before)
    query_unit = _reference_unit_rows(queries)
    library_unit = _reference_unit_rows(embeddings)
    oracle_scores = query_unit @ library_unit.T

    for query_number, (batch_results, scalar_results) in enumerate(zip(batch, scalar)):
        assert len(batch_results) == library_size
        assert len({result.factor_id for result in batch_results}) == library_size
        batch_map = _result_map(batch_results)
        scalar_map = _result_map(scalar_results)
        expected = {
            factor_id: float(oracle_scores[query_number, i])
            for i, factor_id in enumerate(ids)
        }
        assert batch_map.keys() == scalar_map.keys() == expected.keys()
        for factor_id, score in expected.items():
            assert batch_map[factor_id] == pytest.approx(score, abs=3e-6)
            assert scalar_map[factor_id] == pytest.approx(score, abs=3e-6)
        observed_order = [result.similarity_score for result in batch_results]
        assert observed_order == sorted(observed_order, reverse=True)

    # All identical top vectors are included; their order is intentionally free.
    assert {item.factor_id for item in batch[0][:3]} == set(ids[:3])
    assert batch[1][0].factor_id == ids[3]
    assert batch[1][0].similarity_score == pytest.approx(1.0, abs=2e-6)

    threshold = 0.8
    filtered = index.search_batch(queries[:4], k=library_size, min_similarity=threshold)
    for query_number, results in enumerate(filtered):
        expected_ids = {
            ids[i] for i, value in enumerate(oracle_scores[query_number])
            if float(value) >= threshold
        }
        assert {result.factor_id for result in results} == expected_ids


def test_batch_validation_empty_shapes_and_atomic_failure():
    index = FaissANNIndex(3)
    index.build(["A", "B"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert index.search_batch(np.empty((0, 3)), k=1) == []
    assert index.search_batch(np.ones((2, 3)), k=9)[0]

    class SearchSpy:
        calls = 0

        def search(self, queries, k):
            self.calls += 1
            raise AssertionError("invalid batch reached FAISS")

    spy = SearchSpy()
    real_index = index._index
    index._index = spy
    invalid_batches = (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [np.nan, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [np.inf, 0.0, 1.0]]),
    )
    for values in invalid_batches:
        with pytest.raises(ValueError):
            index.search_batch(values)
    assert spy.calls == 0
    index._index = real_index

    with pytest.raises(ValueError, match="2D numpy array"):
        index.search_batch(np.ones(3))
    with pytest.raises(ValueError, match="Expected embedding_dim"):
        index.search_batch(np.ones((2, 4)))
    with pytest.raises(ValueError, match="positive integer"):
        index.search_batch(np.ones((2, 3)), k=0)
    with pytest.raises(ValueError, match="finite"):
        index.search_batch(np.ones((0, 3)), min_similarity=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        index.search_batch(np.ones((1, 3)), k=True)
    with pytest.raises(TypeError, match="real number"):
        index.search_batch(np.ones((1, 3)), min_similarity=True)


def test_batch_makes_one_faiss_call_and_cutoff_ties_need_no_stable_id_order():
    index = FaissANNIndex(2)
    index.build(
        ["A", "B", "C", "D"],
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    real_index = index._index

    class CountingIndex:
        calls = 0

        def search(self, queries, k):
            self.calls += 1
            return real_index.search(queries, k)

    counting = CountingIndex()
    index._index = counting
    result = index.search_batch(np.array([[1.0, 0.0], [0.0, 1.0]]), k=2)
    assert counting.calls == 1
    assert len(result) == 2
    # Three equal top scores compete for two slots: membership/order is free.
    assert len(result[0]) == 2
    assert {row.factor_id for row in result[0]} <= {"A", "B", "C"}
    assert all(row.similarity_score == pytest.approx(1.0) for row in result[0])
    assert len(result[1]) == 2
    assert result[1][0].factor_id == "D"


def test_batch_on_empty_index_returns_one_empty_list_per_query():
    index = FaissANNIndex(3)
    assert index.search_batch(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])) == [[], []]


def test_batch_safe_extreme_magnitudes_and_build_validation():
    extreme = np.array([
        [1e300, 1e300, 0.0],
        [1e-300, 1e-300, 0.0],
        [np.nextafter(0.0, 1.0), 0.0, 0.0],
        [-1e300, -1e300, 0.0],
    ], dtype=np.float64)
    before = extreme.copy()
    queries = np.array([[1e300, 1e300, 0.0], [-1e-300, -1e-300, 0.0]])
    queries_before = queries.copy()
    index = FaissANNIndex(3)
    index.build(["large", "small", "subnormal", "opposite"], extreme)
    results = index.search_batch(queries, k=4)
    assert {results[0][0].factor_id, results[0][1].factor_id} == {"large", "small"}
    assert results[1][0].factor_id == "opposite"
    assert all(row[0].similarity_score == pytest.approx(1.0, abs=2e-6) for row in results)
    assert np.array_equal(extreme, before)
    assert np.array_equal(queries, queries_before)

    for invalid in (
        np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, np.inf, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ):
        with pytest.raises(ValueError):
            FaissANNIndex(3).build(["A", "B"], invalid)
