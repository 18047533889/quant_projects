"""
Tests for ANN-based similarity search.
"""

import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from factor_assets.similarity.ann import (
        ANNBackend,
        ANNSearchResult,
        FaissANNIndex,
        AnnoyANNIndex,
        create_ann_index,
    )
    ANN_MODULE_AVAILABLE = True
except ImportError:
    ANN_MODULE_AVAILABLE = False

# Check individual backend availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False


pytestmark = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")


class TestANNSearchResult:
    """Tests for ANNSearchResult dataclass."""

    def test_creation(self):
        """ANNSearchResult can be created with valid data."""
        if not ANN_MODULE_AVAILABLE:
            pytest.skip("ANN module not available")

        result = ANNSearchResult(
            factor_id="F001",
            distance=0.5,
            similarity_score=0.8,
            evidence_ref_id="evidence-001",
        )

        assert result.factor_id == "F001"
        assert result.distance == 0.5
        assert result.similarity_score == 0.8
        assert result.evidence_ref_id == "evidence-001"

    def test_requires_factor_id(self):
        """ANNSearchResult requires factor_id."""
        if not ANN_MODULE_AVAILABLE:
            pytest.skip("ANN module not available")

        with pytest.raises(ValueError, match="factor_id is required"):
            ANNSearchResult(
                factor_id="",
                distance=0.5,
            )

    def test_requires_non_negative_distance(self):
        """Distance must be non-negative."""
        if not ANN_MODULE_AVAILABLE:
            pytest.skip("ANN module not available")

        with pytest.raises(ValueError, match="distance must be finite and non-negative"):
            ANNSearchResult(
                factor_id="F001",
                distance=-0.1,
            )


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not available")
class TestFaissANNIndex:
    """Tests for FAISS-based ANN index."""

    def test_build_and_search(self):
        """Build index and search for neighbors."""
        from factor_assets.similarity.ann import FaissANNIndex

        # Create simple embeddings
        factor_ids = ["F001", "F002", "F003", "F004"]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        index = FaissANNIndex(embedding_dim=3, normalize=True)
        index.build(factor_ids, embeddings)

        assert index.num_factors == 4

        # Search for nearest neighbors to F001
        query = embeddings[0]
        results = index.search(query, k=3)

        assert len(results) <= 3
        # F001 and F002 should be similar (both have high first dimension)
        factor_ids_found = [r.factor_id for r in results]
        assert "F001" in factor_ids_found
        assert "F002" in factor_ids_found

    def test_search_by_id(self):
        """Search for neighbors by factor ID."""
        from factor_assets.similarity.ann import FaissANNIndex

        factor_ids = ["F001", "F002", "F003"]
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])

        index = FaissANNIndex(embedding_dim=2, normalize=True)
        index.build(factor_ids, embeddings)

        # Find neighbors of F001
        results = index.search_by_id("F001", k=2)

        # Should not include F001 itself
        factor_ids_found = [r.factor_id for r in results]
        assert "F001" not in factor_ids_found
        # F002 should be closest
        assert "F002" in factor_ids_found

    def test_search_with_threshold(self):
        """Search with minimum similarity threshold."""
        from factor_assets.similarity.ann import FaissANNIndex

        factor_ids = ["F001", "F002", "F003"]
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],  # Similar to F001
            [0.0, 1.0],  # Orthogonal to F001
        ])

        index = FaissANNIndex(embedding_dim=2, normalize=True)
        index.build(factor_ids, embeddings)

        query = embeddings[0]
        # High threshold should filter out F003
        results = index.search(query, k=3, min_similarity=0.8)

        factor_ids_found = [r.factor_id for r in results]
        assert "F002" in factor_ids_found or "F001" in factor_ids_found

    def test_empty_index(self):
        """Empty index returns empty results."""
        from factor_assets.similarity.ann import FaissANNIndex

        index = FaissANNIndex(embedding_dim=3)
        query = np.array([1.0, 0.0, 0.0])

        results = index.search(query, k=5)
        assert len(results) == 0

    def test_dimension_mismatch(self):
        """Build fails with dimension mismatch."""
        from factor_assets.similarity.ann import FaissANNIndex

        index = FaissANNIndex(embedding_dim=3)
        factor_ids = ["F001", "F002"]
        embeddings = np.array([
            [1.0, 0.0],  # Wrong dimension
            [0.0, 1.0],
        ])

        with pytest.raises(ValueError, match="Expected embedding_dim"):
            index.build(factor_ids, embeddings)

    def test_consecutive_build_replaces_index_and_mapping(self):
        index = FaissANNIndex(embedding_dim=2, normalize=True)
        index.build(["OLD1", "OLD2"], np.array([[1.0, 0.0], [0.9, 0.1]]))
        index.build(["NEW1"], np.array([[0.0, 1.0]]))

        assert index.num_factors == 1
        assert index.search_by_id("OLD1") == []
        assert [result.factor_id for result in index.search(np.array([1.0, 0.0]), k=10)] == ["NEW1"]

    def test_duplicate_factor_ids_rejected_without_changing_index(self):
        index = FaissANNIndex(embedding_dim=2)
        index.build(["OLD"], np.array([[1.0, 0.0]]))

        with pytest.raises(ValueError, match="factor_ids must be unique"):
            index.build(["DUP", "DUP"], np.array([[1.0, 0.0], [0.0, 1.0]]))

        assert index.num_factors == 1
        assert index.search(np.array([1.0, 0.0]), k=1)[0].factor_id == "OLD"

    def test_failed_rebuild_preserves_previous_index(self, monkeypatch):
        import factor_assets.similarity.ann as ann_module

        index = FaissANNIndex(embedding_dim=2)
        index.build(["OLD"], np.array([[1.0, 0.0]]))

        class FailingIndex:
            def add(self, embeddings):
                raise RuntimeError("injected add failure")

        monkeypatch.setattr(ann_module.faiss, "IndexFlatIP", lambda dim: FailingIndex())
        with pytest.raises(RuntimeError, match="injected add failure"):
            index.build(["NEW"], np.array([[0.0, 1.0]]))

        assert index.num_factors == 1
        assert index.search(np.array([1.0, 0.0]), k=1)[0].factor_id == "OLD"


@pytest.mark.skipif(not NUMPY_AVAILABLE or not ANNOY_AVAILABLE, reason="numpy or annoy not available")
class TestAnnoyANNIndex:
    """Tests for Annoy-based ANN index."""

    def test_build_and_search(self):
        """Build Annoy index and search."""
        from factor_assets.similarity.ann import AnnoyANNIndex

        factor_ids = ["F001", "F002", "F003", "F004"]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        index = AnnoyANNIndex(embedding_dim=3, n_trees=5)
        index.build(factor_ids, embeddings)

        assert index.num_factors == 4

        # Search
        query = embeddings[0]
        results = index.search(query, k=3)

        assert len(results) <= 3
        # F001 should be found
        factor_ids_found = [r.factor_id for r in results]
        assert "F001" in factor_ids_found

    def test_search_by_id(self):
        """Search by factor ID in Annoy index."""
        from factor_assets.similarity.ann import AnnoyANNIndex

        factor_ids = ["F001", "F002", "F003"]
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])

        index = AnnoyANNIndex(embedding_dim=2, n_trees=5)
        index.build(factor_ids, embeddings)

        results = index.search_by_id("F001", k=2)

        # Should not include F001 itself
        factor_ids_found = [r.factor_id for r in results]
        assert "F001" not in factor_ids_found

    def test_not_built(self):
        """Search on unbuilt index returns empty."""
        from factor_assets.similarity.ann import AnnoyANNIndex

        index = AnnoyANNIndex(embedding_dim=3, n_trees=5)
        query = np.array([1.0, 0.0, 0.0])

        results = index.search(query, k=5)
        assert len(results) == 0


@pytest.mark.skipif(not NUMPY_AVAILABLE or not ANN_MODULE_AVAILABLE, reason="numpy or ANN module not available")
class TestCreateANNIndex:
    """Tests for ANN index factory."""

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not available")
    def test_create_faiss_index(self):
        """Factory creates FAISS index."""
        from factor_assets.similarity.ann import create_ann_index, ANNBackend

        index = create_ann_index(ANNBackend.FAISS, embedding_dim=10)
        assert index is not None

    @pytest.mark.skipif(not ANNOY_AVAILABLE, reason="annoy not available")
    def test_create_annoy_index(self):
        """Factory creates Annoy index."""
        from factor_assets.similarity.ann import create_ann_index, ANNBackend

        index = create_ann_index(ANNBackend.ANNOY, embedding_dim=10, n_trees=5)
        assert index is not None

    def test_invalid_backend(self):
        """Factory rejects invalid backend."""
        try:
            from factor_assets.similarity.ann import create_ann_index
        except ImportError:
            pytest.skip("ANN backends not available")

        with pytest.raises(ValueError, match="Unsupported backend"):
            create_ann_index("invalid", embedding_dim=10)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")
@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not available")
class TestANNIntegration:
    """Integration tests for ANN search."""

    def test_similarity_ordering(self):
        """Results are ordered by similarity."""
        try:
            from factor_assets.similarity.ann import FaissANNIndex
        except ImportError:
            pytest.skip("FAISS not available")

        # Create factors with varying similarity to query
        factor_ids = ["F001", "F002", "F003", "F004"]
        embeddings = np.array([
            [1.0, 0.0],  # Query
            [0.99, 0.01],  # Very similar
            [0.7, 0.3],   # Moderately similar
            [0.0, 1.0],   # Dissimilar
        ])

        index = FaissANNIndex(embedding_dim=2, normalize=True)
        index.build(factor_ids, embeddings)

        query = embeddings[0]
        results = index.search(query, k=4)

        # Check similarity is descending (distance is ascending for inner product)
        # Higher similarity should come first
        if len(results) >= 2:
            assert results[0].factor_id in ["F001", "F002"]

    def test_consistent_results(self):
        """Multiple searches return consistent results."""
        try:
            from factor_assets.similarity.ann import FaissANNIndex
        except ImportError:
            pytest.skip("FAISS not available")

        factor_ids = ["F001", "F002", "F003"]
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])

        index = FaissANNIndex(embedding_dim=2, normalize=True)
        index.build(factor_ids, embeddings)

        query = embeddings[0]
        results1 = index.search(query, k=2)
        results2 = index.search(query, k=2)

        # Should return same results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.factor_id == r2.factor_id
            assert abs(r1.distance - r2.distance) < 1e-6


    @pytest.mark.parametrize("distance", [float("nan"), float("inf"), float("-inf"), -0.1])
    def test_distance_must_be_finite_and_nonnegative(self, distance):
        with pytest.raises(ValueError, match="finite and non-negative"):
            ANNSearchResult(factor_id="F001", distance=distance)


class TestBackendIndependentANNContracts:
    @pytest.mark.parametrize("k", [0, -1, 1.5, True, "1", None])
    def test_validate_query_rejects_nonpositive_noninteger_k(self, k):
        from factor_assets.similarity.ann import _validate_query

        with pytest.raises(ValueError, match="k must be a positive integer"):
            _validate_query(np.array([1.0, 0.0]), 2, k)


class ANNContractMixin:
    backend_class = None

    def make_index(self):
        kwargs = {"n_trees": 20} if self.backend_class is AnnoyANNIndex else {"normalize": True}
        return self.backend_class(embedding_dim=2, **kwargs)

    def test_signed_score_and_distance_contract(self):
        index = self.make_index()
        index.build(["same", "orthogonal", "opposite"], np.array([[1., 0.], [0., 1.], [-1., 0.]]))
        results = index.search(np.array([1., 0.]), k=3)
        by_id = {result.factor_id: result for result in results}
        assert by_id["same"].similarity_score == pytest.approx(1.0, abs=1e-5)
        assert by_id["orthogonal"].similarity_score == pytest.approx(0.0, abs=1e-5)
        assert by_id["opposite"].similarity_score == pytest.approx(-1.0, abs=1e-5)
        assert all(result.distance >= 0 for result in results)
        assert [r.similarity_score for r in results] == sorted(
            (r.similarity_score for r in results), reverse=True
        )

    @pytest.mark.parametrize("k", [0, -1, 1.5, True])
    def test_invalid_k_rejected(self, k):
        index = self.make_index()
        index.build(["F001"], np.array([[1., 0.]]))
        with pytest.raises(ValueError, match="k must be a positive integer"):
            index.search(np.array([1., 0.]), k=k)

    @pytest.mark.parametrize("k", [0, -1, 1.5, True])
    def test_invalid_k_rejected_before_unknown_id(self, k):
        index = self.make_index()
        index.build(["F001"], np.array([[1., 0.]]))
        with pytest.raises(ValueError, match="k must be a positive integer"):
            index.search_by_id("UNKNOWN", k=k)

    def test_query_shape_dimension_and_zero_rejected(self):
        index = self.make_index()
        index.build(["F001"], np.array([[1., 0.]]))
        for query, message in [
            (np.array([[1., 0.]]), "1D"),
            (np.array([1., 0., 0.]), "Expected embedding_dim"),
            (np.array([0., 0.]), "zero query vectors"),
        ]:
            with pytest.raises(ValueError, match=message):
                index.search(query, k=1)

    def test_duplicate_and_zero_build_failures_preserve_index(self):
        index = self.make_index()
        index.build(["OLD"], np.array([[1., 0.]]))
        with pytest.raises(ValueError, match="factor_ids must be unique"):
            index.build(["DUP", "DUP"], np.array([[1., 0.], [0., 1.]]))
        with pytest.raises(ValueError, match="zero vectors"):
            index.build(["ZERO"], np.array([[0., 0.]]))
        assert index.num_factors == 1
        assert index.search(np.array([1., 0.]), k=1)[0].factor_id == "OLD"

    def test_rebuild_replaces_previous_contents(self):
        index = self.make_index()
        index.build(["OLD"], np.array([[1., 0.]]))
        index.build(["NEW"], np.array([[0., 1.]]))
        assert index.search_by_id("OLD") == []
        assert index.num_factors == 1


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not available")
class TestFaissANNContract(ANNContractMixin):
    backend_class = FaissANNIndex


@pytest.mark.skipif(not ANNOY_AVAILABLE, reason="annoy not available")
class TestAnnoyANNContract(ANNContractMixin):
    backend_class = AnnoyANNIndex
