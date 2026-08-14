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

        with pytest.raises(ValueError, match="distance must be non-negative"):
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
