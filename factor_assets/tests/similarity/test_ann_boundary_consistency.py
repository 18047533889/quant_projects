"""Boundary contracts for the Annoy backend adapter."""
from __future__ import annotations

import numpy as np
import pytest

from factor_assets.similarity import ann
from factor_assets.similarity.unit_vectors import _canonical_cosine


@pytest.mark.parametrize("dim", [0, -1, True, 2.0, np.int64(2)])
def test_invalid_dimension_rejected_before_native_allocation(monkeypatch, dim):
    allocated = []
    monkeypatch.setattr(ann, "ANNOY_AVAILABLE", True)
    monkeypatch.setattr(ann, "AnnoyIndex", lambda *args: allocated.append(args))
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        ann.AnnoyANNIndex(dim)
    assert allocated == []


@pytest.mark.parametrize("trees", [0, -1, True, 2.0, np.int64(2)])
def test_invalid_tree_count_rejected_before_native_allocation(monkeypatch, trees):
    allocated = []
    monkeypatch.setattr(ann, "ANNOY_AVAILABLE", True)
    monkeypatch.setattr(ann, "AnnoyIndex", lambda *args: allocated.append(args))
    with pytest.raises(ValueError, match="n_trees must be a positive integer"):
        ann.AnnoyANNIndex(2, n_trees=trees)
    assert allocated == []


@pytest.mark.skipif(not ann.ANNOY_AVAILABLE, reason="annoy unavailable")
def test_false_native_build_result_preserves_previous_index(monkeypatch):
    index = ann.AnnoyANNIndex(2, n_trees=4)
    index.build(["old"], np.array([[1.0, 0.0]]))
    previous_index = index._index

    class FalseBuildIndex:
        def add_item(self, item_id, vector):
            pass

        def build(self, n_trees):
            return False

    monkeypatch.setattr(ann, "AnnoyIndex", lambda *args: FalseBuildIndex())
    with pytest.raises(RuntimeError, match="Annoy index build failed"):
        index.build(["new"], np.array([[0.0, 1.0]]))

    assert index._index is previous_index
    assert index.num_factors == 1
    assert index.search(np.array([1.0, 0.0]), k=1)[0].factor_id == "old"


def test_canonical_cosine_clamps_roundoff_and_rejects_values_outside_envelope():
    tolerance = 16 * np.finfo(np.float32).eps
    assert _canonical_cosine(1.0 + tolerance) == 1.0
    assert _canonical_cosine(-1.0 - tolerance) == -1.0
    with pytest.raises(ValueError, match="outside its finite numeric envelope"):
        _canonical_cosine(np.nextafter(1.0 + tolerance, np.inf))
    with pytest.raises(ValueError, match="outside its finite numeric envelope"):
        _canonical_cosine(np.nextafter(-1.0 - tolerance, -np.inf))


@pytest.mark.skipif(not ann.ANNOY_AVAILABLE, reason="annoy unavailable")
def test_search_by_id_rejects_invalid_native_cosine_envelope(monkeypatch):
    index = ann.AnnoyANNIndex(2)
    index.build(["source", "other"], np.array([[1.0, 0.0], [-1.0, 0.0]]))
    tolerance = 16 * np.finfo(np.float32).eps

    class InvalidDistanceIndex:
        def get_nns_by_item(self, item_id, count, include_distances):
            return [0, 1], [0.0, np.sqrt(4.0 + 4.0 * tolerance)]

    monkeypatch.setattr(index, "_index", InvalidDistanceIndex())
    with pytest.raises(ValueError, match="outside its finite numeric envelope"):
        index.search_by_id("source", k=1)


@pytest.mark.skipif(not ann.ANNOY_AVAILABLE, reason="annoy unavailable")
def test_real_annoy_search_by_id_preserves_signed_cosine_scores():
    if not ann.ANNOY_AVAILABLE:
        pytest.skip("annoy unavailable")
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    ids = ["positive", "negative", "orthogonal"]
    index = ann.AnnoyANNIndex(2, n_trees=20)
    index.build(ids, vectors)

    results = index.search_by_id("positive", k=2)
    scores = {result.factor_id: result.similarity_score for result in results}
    vector_results = index.search(index._embeddings[0], k=3)
    vector_scores = {
        result.factor_id: result.similarity_score for result in vector_results
    }
    assert set(scores) == {"negative", "orthogonal"}
    expected = {
        factor_id: _canonical_cosine(float(np.dot(index._embeddings[0], row)))
        for factor_id, row in zip(ids[1:], index._embeddings[1:])
    }
    for factor_id in expected:
        assert scores[factor_id] == pytest.approx(expected[factor_id], abs=1e-6)
        assert scores[factor_id] == pytest.approx(vector_scores[factor_id], abs=1e-6)
    assert scores["negative"] < 0.0


@pytest.mark.skipif(not ann.ANNOY_AVAILABLE, reason="annoy unavailable")
def test_search_by_id_clamps_roundoff_at_negative_cosine_endpoint(monkeypatch):
    index = ann.AnnoyANNIndex(2)
    index.build(["source", "other"], np.array([[1.0, 0.0], [-1.0, 0.0]]))
    tolerance = 16 * np.finfo(np.float32).eps

    class RoundedDistanceIndex:
        def get_nns_by_item(self, item_id, count, include_distances):
            return [0, 1], [0.0, np.sqrt(4.0 + 2.0 * tolerance)]

    monkeypatch.setattr(index, "_index", RoundedDistanceIndex())
    result = index.search_by_id("source", k=1)
    assert result[0].similarity_score == -1.0

def test_missing_annoy_validates_parameters_before_dependency_error(monkeypatch):
    allocations = []
    monkeypatch.setattr(ann, "ANNOY_AVAILABLE", False)
    monkeypatch.setattr(ann, "AnnoyIndex", lambda *args: allocations.append(args))

    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        ann.AnnoyANNIndex(0)
    with pytest.raises(ValueError, match="n_trees must be a positive integer"):
        ann.AnnoyANNIndex(2, n_trees=0)
    with pytest.raises(ImportError, match="annoy is required"):
        ann.AnnoyANNIndex(2, n_trees=3)
    assert allocations == []
