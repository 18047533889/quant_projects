"""
Approximate nearest neighbor search for factor similarity.

Uses ANN indices (faiss or annoy) for fast similarity search over large factor sets.
Works with EvidenceRef-based embeddings, not raw factor values.
"""

from dataclasses import dataclass
import numbers
from typing import Optional, Protocol, List, Tuple
from enum import Enum
import warnings

from factor_assets.similarity.unit_vectors import _canonical_cosine, _unit_rows


def _validate_embedding_dim(embedding_dim: int) -> None:
    if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
        raise ValueError("embedding_dim must be a positive integer")


def _validate_positive_n_trees(n_trees: int) -> None:
    if not isinstance(n_trees, int) or isinstance(n_trees, bool) or n_trees <= 0:
        raise ValueError("n_trees must be a positive integer")


def _validate_build_inputs(factor_ids: List[str], embeddings: "np.ndarray", embedding_dim: int) -> None:
    _validate_embedding_dim(embedding_dim)
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D numpy array")
    if len(factor_ids) != embeddings.shape[0]:
        raise ValueError("Number of factor_ids must match embeddings rows")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Expected embedding_dim={embedding_dim}, got {embeddings.shape[1]}")
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be unique")
    if embeddings.dtype.kind not in "fiu":
        raise ValueError("embeddings must contain real numeric values")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    if np.any(~np.any(embeddings != 0, axis=1)):
        raise ValueError("zero vectors are not valid embeddings")


def _validate_positive_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")


DEFAULT_MAX_BATCH_RESULT_ROWS = 100_000
MAX_BATCH_RESULT_ROWS = 2_000_000


def _validate_max_result_rows(max_result_rows: int) -> None:
    if type(max_result_rows) is not int or max_result_rows <= 0:
        raise ValueError("max_result_rows must be a positive built-in integer")
    if max_result_rows > MAX_BATCH_RESULT_ROWS:
        raise ValueError(
            f"max_result_rows must not exceed {MAX_BATCH_RESULT_ROWS}"
        )


def _validate_min_similarity(min_similarity: Optional[float]) -> None:
    """Require an optional similarity threshold to be a finite real number."""
    if min_similarity is None:
        return
    if isinstance(min_similarity, bool) or not isinstance(min_similarity, numbers.Real):
        raise TypeError("min_similarity must be a finite real number or None")
    # Compare in the input's own numeric precision: math.isfinite() first
    # converts NumPy longdouble values to C double and can misclassify finite
    # values above the double range as infinity.
    if (min_similarity != min_similarity
            or min_similarity == float("inf")
            or min_similarity == float("-inf")):
        raise ValueError("min_similarity must be finite")


def _validate_query(query_embedding: "np.ndarray", embedding_dim: int, k: int) -> "np.ndarray":
    _validate_positive_k(k)
    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D numpy array")
    if query_embedding.shape[0] != embedding_dim:
        raise ValueError(f"Expected embedding_dim={embedding_dim}, got {query_embedding.shape[0]}")
    if query_embedding.dtype.kind not in "fiu":
        raise ValueError("query_embedding must contain real numeric values")
    if not np.isfinite(query_embedding).all():
        raise ValueError("query_embedding must contain only finite values")
    if not np.any(query_embedding != 0):
        raise ValueError("zero query vectors are not valid embeddings")
    return np.ascontiguousarray(_unit_rows(query_embedding.reshape(1, -1))[0], dtype=np.float32)

def _validate_query_batch(query_embeddings: "np.ndarray", embedding_dim: int, k: int) -> "np.ndarray":
    """Validate and normalize a complete query matrix before backend access."""
    _validate_positive_k(k)
    if not isinstance(query_embeddings, np.ndarray) or query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must be a 2D numpy array")
    if query_embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Expected embedding_dim={embedding_dim}, got {query_embeddings.shape[1]}"
        )
    if query_embeddings.dtype.kind not in "fiu":
        raise ValueError("query_embeddings must contain real numeric values")
    if not np.isfinite(query_embeddings).all():
        raise ValueError("query_embeddings must contain only finite values")
    if query_embeddings.shape[0] == 0:
        return np.empty((0, embedding_dim), dtype=np.float32)
    if np.any(~np.any(query_embeddings != 0, axis=1)):
        raise ValueError("zero query vectors are not valid embeddings")
    return np.ascontiguousarray(_unit_rows(query_embeddings), dtype=np.float32)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

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


class IndexCapability(Enum):
    """Capability of a search index.

    An index that answers queries EXACTLY (brute-force over every stored
    vector) is NOT an approximate-nearest-neighbour (ANN) index, even when it
    is backed by the ``faiss`` library.  Naming it correctly matters: a caller
    selecting on ``capability`` must not be told a flat exact index is "ANN".
    """
    EXACT_FLAT = "exact_flat"        # exact brute-force; e.g. faiss.IndexFlatIP
    APPROXIMATE = "approximate"      # real ANN; e.g. Annoy angular index


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
        if not np.isfinite(self.distance) or self.distance < 0:
            raise ValueError("distance must be finite and non-negative")
        if self.similarity_score is not None and not -1.0 <= self.similarity_score <= 1.0:
            raise ValueError("similarity_score must be in [-1, 1]")


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


class BatchANNIndex(Protocol):
    """Optional matrix-query capability; legacy ANNIndex is unchanged."""

    def search_batch(
        self,
        query_embeddings: "np.ndarray",
        k: int = 10,
        min_similarity: Optional[float] = None,
        *,
        max_result_rows: int = DEFAULT_MAX_BATCH_RESULT_ROWS,
    ) -> List[List[ANNSearchResult]]:
        """Return one ordered neighbor list per query row."""
        ...


class FaissANNIndex:
    """
    FAISS-backed EXACT flat index for factor similarity.

    Uses ``faiss.IndexFlatIP`` which performs EXACT brute-force inner-product
    search (cosine after L2 normalization).  This is **not** an approximate
    nearest-neighbour (ANN) index — its ``capability`` is ``EXACT_FLAT``, and
    callers must not assume approximation.  Requires embeddings to be
    pre-normalized for cosine similarity.
    """

    #: Honest capability: this is an exact flat index, not ANN.
    capability = IndexCapability.EXACT_FLAT

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
        _validate_embedding_dim(embedding_dim)
        if not normalize:
            raise ValueError("normalize=False is unsupported; canonical score is cosine similarity")
        self.normalize = True
        self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product index
        self._factor_ids: List[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._embeddings = np.empty((0, embedding_dim), dtype=np.float32)

    def build(self, factor_ids: List[str], embeddings: "np.ndarray") -> None:
        """
        Build FAISS index from factor embeddings.

        Args:
            factor_ids: List of factor identifiers
            embeddings: 2D array of shape (n_factors, embedding_dim)
        """
        _validate_build_inputs(factor_ids, embeddings, self.embedding_dim)

        staged_embeddings = np.ascontiguousarray(_unit_rows(embeddings), dtype=np.float32)
        if self.normalize:
            faiss.normalize_L2(staged_embeddings)

        staged_index = faiss.IndexFlatIP(self.embedding_dim)
        staged_index.add(staged_embeddings)
        staged_factor_ids = list(factor_ids)
        staged_id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}

        self._index = staged_index
        self._factor_ids = staged_factor_ids
        self._id_to_idx = staged_id_to_idx
        self._embeddings = staged_embeddings

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
        _validate_min_similarity(min_similarity)
        query = _validate_query(query_embedding, self.embedding_dim, k)
        if len(self._factor_ids) == 0:
            return []

        query = query.reshape(1, -1)

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

            # FAISS IndexFlatIP returns inner-product similarity; expose the
            # backend's equivalent non-negative distance alongside canonical score.
            similarity = _canonical_cosine(dist) if self.normalize else None
            backend_distance = 1.0 - similarity if self.normalize else abs(float(dist))

            if min_similarity is not None and (similarity is None or similarity < min_similarity):
                continue

            results.append(ANNSearchResult(
                factor_id=self._factor_ids[idx],
                distance=backend_distance,
                similarity_score=similarity,
            ))

        return results

    def search_batch(
        self,
        query_embeddings: "np.ndarray",
        k: int = 10,
        min_similarity: Optional[float] = None,
        *,
        max_result_rows: int = DEFAULT_MAX_BATCH_RESULT_ROWS,
    ) -> List[List[ANNSearchResult]]:
        """Search many queries in one exact FAISS call.

        The outer result position corresponds to the same input query row;
        each inner list follows scalar search semantics. Equal-score neighbors
        have no stable factor-ID ordering guarantee.
        Result rows are admitted against `Q * min(k, index size)` before FAISS
        is called. Reduce `k` or chunk queries when a call exceeds the cap.
        This row-count cap is not a query-workspace or total-process RSS limit.

        The whole query matrix is validated before FAISS is called, so an
        invalid row rejects the batch without partially running a search.
        An empty matrix returns [] after validating its shape and options.
        """
        _validate_max_result_rows(max_result_rows)
        _validate_min_similarity(min_similarity)
        queries = _validate_query_batch(query_embeddings, self.embedding_dim, k)
        if len(queries) == 0:
            return []

        k_actual = min(k, len(self._factor_ids))
        requested_rows = len(queries) * k_actual
        if requested_rows > max_result_rows:
            raise ValueError(
                f"batch may return {requested_rows} result rows, above "
                f"max_result_rows={max_result_rows}; reduce k or chunk queries"
            )
        if k_actual == 0:
            return [[] for _ in queries]

        if self.normalize:
            faiss.normalize_L2(queries)
        distances, indices = self._index.search(queries, k_actual)

        batch_results: List[List[ANNSearchResult]] = []
        for query_distances, query_indices in zip(distances, indices):
            results: List[ANNSearchResult] = []
            for dist, idx in zip(query_distances, query_indices):
                if idx == -1:
                    continue
                similarity = _canonical_cosine(dist) if self.normalize else None
                backend_distance = 1.0 - similarity if self.normalize else abs(float(dist))
                if min_similarity is not None and (
                    similarity is None or similarity < min_similarity
                ):
                    continue
                results.append(ANNSearchResult(
                    factor_id=self._factor_ids[idx],
                    distance=backend_distance,
                    similarity_score=similarity,
                ))
            batch_results.append(results)
        return batch_results

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
        _validate_positive_k(k)
        _validate_min_similarity(min_similarity)
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
    Annoy-based approximate nearest neighbour (ANN) index for factor similarity.

    Uses angular distance (equivalent to cosine similarity).  This is a real
    ANN index — ``capability`` is ``APPROXIMATE``.  Suitable for larger
    datasets with trade-off between speed and accuracy.
    """

    #: Honest capability: real approximate nearest-neighbour search.
    capability = IndexCapability.APPROXIMATE

    def __init__(self, embedding_dim: int, n_trees: int = 10):
        """
        Args:
            embedding_dim: Dimensionality of embeddings
            n_trees: Number of trees (more trees = better accuracy, slower build)

        Raises:
            ImportError: If annoy is not available
        """
        _validate_embedding_dim(embedding_dim)
        _validate_positive_n_trees(n_trees)
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
        self._embeddings = np.empty((0, embedding_dim), dtype=np.float32)

    def build(self, factor_ids: List[str], embeddings: "np.ndarray") -> None:
        """
        Build Annoy index from factor embeddings.

        Args:
            factor_ids: List of factor identifiers
            embeddings: 2D array of shape (n_factors, embedding_dim)
        """
        _validate_build_inputs(factor_ids, embeddings, self.embedding_dim)
        staged_embeddings = np.ascontiguousarray(_unit_rows(embeddings), dtype=np.float32)
        staged_index = AnnoyIndex(self.embedding_dim, 'angular')

        for idx, embedding in enumerate(staged_embeddings):
            staged_index.add_item(idx, embedding.tolist())

        if staged_index.build(self.n_trees) is False:
            raise RuntimeError("Annoy index build failed")
        self._index = staged_index
        self._factor_ids = list(factor_ids)
        self._id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}
        self._embeddings = staged_embeddings
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
        _validate_min_similarity(min_similarity)
        query = _validate_query(query_embedding, self.embedding_dim, k)
        if not self._built or len(self._factor_ids) == 0:
            return []

        indices, distances = self._index.get_nns_by_vector(
            query.tolist(), min(k, len(self._factor_ids)), include_distances=True
        )

        # Convert to results
        results = []
        for idx, dist in zip(indices, distances):
            # Annoy angular distance is backend distance; canonical signed
            # cosine similarity is 1 - distance^2 / 2.
            similarity = _canonical_cosine(1.0 - (float(dist) ** 2 / 2.0))

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
        _validate_positive_k(k)
        _validate_min_similarity(min_similarity)
        if not self._built or factor_id not in self._id_to_idx:
            return []

        _validate_query(self._embeddings[self._id_to_idx[factor_id]], self.embedding_dim, k)
        idx = self._id_to_idx[factor_id]
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

            similarity = _canonical_cosine(1.0 - (float(dist) ** 2 / 2.0))

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
    "BatchANNIndex",
    "FaissANNIndex",
    "AnnoyANNIndex",
    "create_ann_index",
    "IndexCapability",
]
