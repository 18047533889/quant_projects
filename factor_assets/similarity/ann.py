"""
Approximate nearest neighbor search for factor similarity.

Uses ANN indices (faiss or annoy) for fast similarity search over large factor sets.
Works with EvidenceRef-based embeddings, not raw factor values.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, List, Tuple
from enum import Enum
import warnings

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    AnnoyIndex = None


class ANNBackend(Enum):
    """Supported ANN backends."""
    FAISS = "faiss"
    ANNOY = "annoy"


@dataclass(frozen=True)
class ANNSearchResult:
    """
    Result from ANN similarity search.

    Contains factor ID, distance/similarity score, and metadata.
    """
    factor_id: str
    distance: float
    similarity_score: Optional[float] = None
    evidence_ref_id: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if self.distance < 0:
            raise ValueError("distance must be non-negative")


class ANNIndex(Protocol):
    """
    Protocol for ANN similarity indices.

    Defines interface for building and querying approximate nearest neighbor indices.
    """

    def build(self, factor_ids: List[str], embeddings: "np.ndarray") -> None:
        """
        Build index from factor embeddings.

        Args:
            factor_ids: List of factor identifiers
            embeddings: 2D array of shape (n_factors, embedding_dim)
        """
        ...

    def search(
        self,
        query_embedding: "np.ndarray",
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[ANNSearchResult]:
        """
        Find k nearest neighbors to query embedding.

        Args:
            query_embedding: 1D array of shape (embedding_dim,)
            k: Number of neighbors to return
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of ANNSearchResult ordered by distance (ascending)
        """
        ...

    def search_by_id(
        self,
        factor_id: str,
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[ANNSearchResult]:
        """
        Find k nearest neighbors to a factor already in the index.

        Args:
            factor_id: Factor identifier
            k: Number of neighbors to return
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of ANNSearchResult ordered by distance (ascending)
        """
        ...


class FaissANNIndex:
    """
    FAISS-based ANN index for factor similarity.

    Uses inner product similarity (cosine after L2 normalization).
    Requires embeddings to be pre-normalized for cosine similarity.
    """

    def __init__(self, embedding_dim: int, normalize: bool = True):
        """
        Args:
            embedding_dim: Dimensionality of embeddings
            normalize: Whether to L2-normalize embeddings (for cosine similarity)

        Raises:
            ImportError: If faiss is not available
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for FaissANNIndex. "
                "Install with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product index
        self._factor_ids: List[str] = []
        self._id_to_idx: dict[str, int] = {}

    def build(self, factor_ids: List[str], embeddings: "np.ndarray") -> None:
        """
        Build FAISS index from factor embeddings.

        Args:
            factor_ids: List of factor identifiers
            embeddings: 2D array of shape (n_factors, embedding_dim)
        """
        if len(factor_ids) != embeddings.shape[0]:
            raise ValueError("Number of factor_ids must match embeddings rows")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embeddings.shape[1]}")

        # Store factor IDs
        self._factor_ids = list(factor_ids)
        self._id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}

        # Normalize embeddings if requested
        if self.normalize:
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)

        # Build index
        self._index.add(embeddings.astype(np.float32))

    def search(
        self,
        query_embedding: "np.ndarray",
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[ANNSearchResult]:
        """
        Find k nearest neighbors using FAISS.

        Args:
            query_embedding: 1D array of shape (embedding_dim,)
            k: Number of neighbors to return
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of ANNSearchResult ordered by similarity (descending)
        """
        if len(self._factor_ids) == 0:
            return []

        # Ensure query is 2D and correct dtype
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if self.normalize:
            faiss.normalize_L2(query)

        # Search
        k_actual = min(k, len(self._factor_ids))
        distances, indices = self._index.search(query, k_actual)

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            # Convert inner product to similarity (already normalized if requested)
            similarity = float(dist) if self.normalize else None

            # Apply threshold
            if min_similarity is not None and similarity is not None:
                if similarity < min_similarity:
                    continue

            results.append(ANNSearchResult(
                factor_id=self._factor_ids[idx],
                distance=float(dist),
                similarity_score=similarity,
            ))

        return results

    def search_by_id(
        self,
        factor_id: str,
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[ANNSearchResult]:
        """
        Find k nearest neighbors to a factor in the index.

        Args:
            factor_id: Factor identifier
            k: Number of neighbors to return (excludes query factor itself)
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of ANNSearchResult ordered by similarity (descending)
        """
        if factor_id not in self._id_to_idx:
            return []

        idx = self._id_to_idx[factor_id]
        query_embedding = self._index.reconstruct(idx)

        # Get k+1 neighbors (to exclude self)
        results = self.search(query_embedding, k=k+1, min_similarity=min_similarity)

        # Filter out query factor itself
        return [r for r in results if r.factor_id != factor_id][:k]

    @property
    def num_factors(self) -> int:
        """Number of factors in index."""
        return len(self._factor_ids)


class AnnoyANNIndex:
    """
    Annoy-based ANN index for factor similarity.

    Uses angular distance (equivalent to cosine similarity).
    Suitable for larger datasets with trade-off between speed and accuracy.
    """

    def __init__(self, embedding_dim: int, n_trees: int = 10):
        """
        Args:
            embedding_dim: Dimensionality of embeddings
            n_trees: Number of trees (more trees = better accuracy, slower build)

        Raises:
            ImportError: If annoy is not available
        """
        if not ANNOY_AVAILABLE:
            raise ImportError(
                "annoy is required for AnnoyANNIndex. "
                "Install with: pip install annoy"
            )

        self.embedding_dim = embedding_dim
        self.n_trees = n_trees
        self._index = AnnoyIndex(embedding_dim, 'angular')
        self._factor_ids: List[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._built = False

    def build(self, factor_ids: List[str], embeddings: "np.ndarray") -> None:
        """
        Build Annoy index from factor embeddings.

        Args:
            factor_ids: List of factor identifiers
            embeddings: 2D array of shape (n_factors, embedding_dim)
        """
        if len(factor_ids) != embeddings.shape[0]:
            raise ValueError("Number of factor_ids must match embeddings rows")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embeddings.shape[1]}")

        # Store factor IDs
        self._factor_ids = list(factor_ids)
        self._id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}

        # Add embeddings to index
        for idx, embedding in enumerate(embeddings):
            self._index.add_item(idx, embedding)

        # Build index
        self._index.build(self.n_trees)
        self._built = True

    def search(
        self,
        query_embedding: "np.ndarray",
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[ANNSearchResult]:
        """
        Find k nearest neighbors using Annoy.

        Args:
            query_embedding: 1D array of shape (embedding_dim,)
            k: Number of neighbors to return
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of ANNSearchResult ordered by similarity (descending)
        """
        if not self._built or len(self._factor_ids) == 0:
            return []

        # Search (Annoy returns indices and distances)
        k_actual = min(k, len(self._factor_ids))
        indices, distances = self._index.get_nns_by_vector(
            query_embedding.flatten().tolist(),
            k_actual,
            include_distances=True
        )

        # Convert to results
        results = []
        for idx, dist in zip(indices, distances):
            # Angular distance to cosine similarity: sim = 1 - (dist^2 / 2)
            # For small distances, sim ≈ 1 - dist
            similarity = 1.0 - (dist ** 2 / 2.0)

            # Apply threshold
            if min_similarity is not None and similarity < min_similarity:
                continue

            results.append(ANNSearchResult(
                factor_id=self._factor_ids[idx],
                distance=float(dist),
                similarity_score=similarity,
            ))

        return results

    def search_by_id(
        self,
        factor_id: str,
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[ANNSearchResult]:
        """
        Find k nearest neighbors to a factor in the index.

        Args:
            factor_id: Factor identifier
            k: Number of neighbors to return (excludes query factor itself)
            min_similarity: Optional minimum similarity threshold

        Returns:
            List of ANNSearchResult ordered by similarity (descending)
        """
        if not self._built or factor_id not in self._id_to_idx:
            return []

        idx = self._id_to_idx[factor_id]

        # Get k+1 neighbors (to exclude self)
        k_actual = min(k + 1, len(self._factor_ids))
        indices, distances = self._index.get_nns_by_item(
            idx,
            k_actual,
            include_distances=True
        )

        # Convert to results and filter out self
        results = []
        for result_idx, dist in zip(indices, distances):
            fid = self._factor_ids[result_idx]
            if fid == factor_id:
                continue

            similarity = 1.0 - (dist ** 2 / 2.0)

            if min_similarity is not None and similarity < min_similarity:
                continue

            results.append(ANNSearchResult(
                factor_id=fid,
                distance=float(dist),
                similarity_score=similarity,
            ))

        return results[:k]

    @property
    def num_factors(self) -> int:
        """Number of factors in index."""
        return len(self._factor_ids)


def create_ann_index(
    backend: ANNBackend,
    embedding_dim: int,
    **kwargs
) -> ANNIndex:
    """
    Factory function to create ANN index.

    Args:
        backend: ANN backend to use
        embedding_dim: Dimensionality of embeddings
        **kwargs: Backend-specific parameters

    Returns:
        ANNIndex implementation

    Raises:
        ImportError: If required backend is not available
        ValueError: If backend is not supported
    """
    if backend == ANNBackend.FAISS:
        normalize = kwargs.get('normalize', True)
        return FaissANNIndex(embedding_dim, normalize=normalize)
    elif backend == ANNBackend.ANNOY:
        n_trees = kwargs.get('n_trees', 10)
        return AnnoyANNIndex(embedding_dim, n_trees=n_trees)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


__all__ = [
    "ANNBackend",
    "ANNSearchResult",
    "ANNIndex",
    "FaissANNIndex",
    "AnnoyANNIndex",
    "create_ann_index",
]
