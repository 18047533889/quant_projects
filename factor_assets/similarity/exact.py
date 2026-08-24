"""
Correlation-based similarity measurement.

Computes factor similarity using correlation metrics.
FA stores only bounded summary statistics, not full value arrays.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Optional


class SimilarityMethod(Enum):
    """Similarity computation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


@dataclass(frozen=True)
class SimilarityResult:
    """
    Result of similarity computation between two factors.

    Contains only summary statistics, not raw factor values.
    """
    factor_id_a: str
    factor_id_b: str
    similarity_score: float  # [-1, 1] for correlation methods
    method: SimilarityMethod
    timestamp: str  # ISO 8601
    sample_size: int
    universe_ref: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    confidence: Optional[float] = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.factor_id_a:
            raise ValueError("factor_id_a is required")
        if not self.factor_id_b:
            raise ValueError("factor_id_b is required")
        if not -1.0 <= self.similarity_score <= 1.0:
            raise ValueError("similarity_score must be in [-1, 1]")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")

    @property
    def has_warnings(self) -> bool:
        """Check if result has warnings."""
        return len(self.warnings) > 0

    def is_high_similarity(self, threshold: float = 0.7) -> bool:
        """Check if similarity exceeds threshold."""
        return abs(self.similarity_score) >= threshold

    def is_significant(self, min_samples: int = 30) -> bool:
        """Check if sample size is sufficient for significance."""
        return self.sample_size >= min_samples


class SimilarityMeasure(Protocol):
    """
    Protocol for computing factor similarity.

    FA does not implement full correlation computation over raw values.
    This protocol defines the boundary for similarity providers.
    """

    def compute_similarity(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod = SimilarityMethod.PEARSON,
        universe_ref: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Optional[SimilarityResult]:
        """
        Compute similarity between two factors.

        Args:
            factor_id_a: First factor identifier
            factor_id_b: Second factor identifier
            method: Similarity computation method
            universe_ref: Optional universe constraint
            period_start: Optional period start
            period_end: Optional period end

        Returns:
            SimilarityResult if computation succeeds, None otherwise
        """
        ...

    def find_similar(
        self,
        factor_id: str,
        threshold: float = 0.7,
        max_results: int = 10,
        method: SimilarityMethod = SimilarityMethod.PEARSON,
        universe_ref: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> list[SimilarityResult]:
        """
        Find factors similar to the given factor.

        Args:
            factor_id: Factor identifier
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            method: Similarity computation method

        Returns:
            List of SimilarityResult ordered by descending similarity
        """
        ...


class QEPairwiseSimilarity:
    """
    QE-based pairwise similarity computation.

    Delegates to QuantEvaluator for pairwise correlation computation.
    Works with EvidenceRef, not raw factor values.
    """

    def __init__(self, qe_adapter=None):
        """
        Args:
            qe_adapter: Optional QE adapter for computing correlations.
                       If None, falls back to stub mode.
        """
        self._qe_adapter = qe_adapter
        self._cache: dict[tuple[str, str, str, str, str, str], SimilarityResult] = {}

    def compute_similarity(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod = SimilarityMethod.PEARSON,
        universe_ref: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Optional[SimilarityResult]:
        """
        Compute pairwise correlation via QE adapter.

        In production mode, delegates to QE for correlation computation.
        In stub mode, returns cached results only.

        Args:
            factor_id_a: First factor identifier
            factor_id_b: Second factor identifier
            method: Similarity computation method
            universe_ref: Optional universe constraint
            period_start: Optional period start
            period_end: Optional period end

        Returns:
            SimilarityResult if computation succeeds, None otherwise
        """
        # Check cache first
        cached = self._get_from_cache(
            factor_id_a, factor_id_b, method,
            universe_ref, period_start, period_end
        )
        if cached:
            return cached

        # Delegate to QE adapter if available
        if self._qe_adapter:
            result = self._qe_adapter.compute_pairwise_correlation(
                factor_id_a=factor_id_a,
                factor_id_b=factor_id_b,
                method=method.value,
                universe_ref=universe_ref,
                period_start=period_start,
                period_end=period_end,
            )
            if result:
                # Convert to SimilarityResult and cache
                similarity_result = self._convert_qe_result(result, method)
                self.add_result(similarity_result)
                return similarity_result

        return None

    def find_similar(
        self,
        factor_id: str,
        threshold: float = 0.7,
        max_results: int = 10,
        method: SimilarityMethod = SimilarityMethod.PEARSON,
        universe_ref: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> list[SimilarityResult]:
        """
        Find factors similar to given factor via QE.

        Args:
            factor_id: Factor identifier
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            method: Similarity computation method

        Returns:
            List of SimilarityResult ordered by descending similarity
        """
        # In production, would delegate to QE batch correlation
        # For now, use cache
        results = []
        seen_factor_ids = set()

        for key, result in self._cache.items():
            # key is (fid_a, fid_b, method_val, universe, start, end)
            fid_a, fid_b, method_val, universe, start, end = key

            if fid_a != factor_id or method_val != method.value:
                continue
            if universe_ref is not None and universe != universe_ref:
                continue
            if period_start is not None and start != period_start:
                continue
            if period_end is not None and end != period_end:
                continue

            # Avoid duplicates (we store symmetric pairs)
            if fid_b in seen_factor_ids:
                continue

            if result.is_high_similarity(threshold):
                results.append(result)
                seen_factor_ids.add(fid_b)

        results.sort(key=lambda r: abs(r.similarity_score), reverse=True)
        return results[:max_results]

    def add_result(self, result: SimilarityResult) -> None:
        """
        Add precomputed similarity result to cache.

        Args:
            result: Similarity result to cache
        """
        key_a = self._make_cache_key(
            result.factor_id_a, result.factor_id_b, result.method,
            result.universe_ref, result.period_start, result.period_end
        )
        key_b = self._make_cache_key(
            result.factor_id_b, result.factor_id_a, result.method,
            result.universe_ref, result.period_start, result.period_end
        )

        # Store with both orderings for symmetric lookup
        self._cache[key_a] = result
        self._cache[key_b] = result

    def _make_cache_key(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod,
        universe_ref: Optional[str],
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> tuple[str, str, str, str, str, str]:
        """
        Build comprehensive cache key including all distinguishing parameters.

        Prevents cache collisions when same factor pair is evaluated with
        different methods, universes, or time periods.
        """
        return (
            factor_id_a,
            factor_id_b,
            method.value,
            universe_ref or "",
            period_start or "",
            period_end or "",
        )

    def _get_from_cache(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod,
        universe_ref: Optional[str],
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> Optional[SimilarityResult]:
        """Retrieve from cache using comprehensive key."""
        key = self._make_cache_key(
            factor_id_a, factor_id_b, method,
            universe_ref, period_start, period_end
        )
        return self._cache.get(key)

    def _convert_qe_result(self, qe_result, method: SimilarityMethod) -> SimilarityResult:
        """Convert QE correlation result to SimilarityResult."""
        # Placeholder for QE result conversion
        # In practice, qe_result would have structure like:
        # {correlation: float, sample_size: int, timestamp: str, ...}
        return SimilarityResult(
            factor_id_a=qe_result.get("factor_id_a"),
            factor_id_b=qe_result.get("factor_id_b"),
            similarity_score=qe_result.get("correlation"),
            method=method,
            timestamp=qe_result.get("timestamp"),
            sample_size=qe_result.get("sample_size"),
            universe_ref=qe_result.get("universe_ref"),
            period_start=qe_result.get("period_start"),
            period_end=qe_result.get("period_end"),
        )

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def count(self) -> int:
        """Get total number of cached results (accounting for symmetric storage)."""
        return len(self._cache) // 2


class CorrelationSimilarity:
    """
    Stub implementation of correlation-based similarity.

    In production, this would delegate to a similarity service or DA.
    For testing, stores precomputed similarity results in memory.
    """

    def __init__(self):
        self._cache: dict[tuple[str, str, str, str, str, str], SimilarityResult] = {}

    def add_result(self, result: SimilarityResult) -> None:
        """
        Add a precomputed similarity result.

        Args:
            result: Similarity result to cache
        """
        key_a = self._make_cache_key(
            result.factor_id_a, result.factor_id_b, result.method,
            result.universe_ref, result.period_start, result.period_end
        )
        key_b = self._make_cache_key(
            result.factor_id_b, result.factor_id_a, result.method,
            result.universe_ref, result.period_start, result.period_end
        )

        # Store with both orderings for symmetric lookup
        self._cache[key_a] = result
        self._cache[key_b] = result

    def _make_cache_key(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod,
        universe_ref: Optional[str],
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> tuple[str, str, str, str, str, str]:
        """
        Build comprehensive cache key including all distinguishing parameters.

        Prevents cache collisions when same factor pair is evaluated with
        different methods, universes, or time periods.
        """
        return (
            factor_id_a,
            factor_id_b,
            method.value,
            universe_ref or "",
            period_start or "",
            period_end or "",
        )

    def compute_similarity(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod = SimilarityMethod.PEARSON,
        universe_ref: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Optional[SimilarityResult]:
        """
        Retrieve cached similarity result.

        In production, this would compute correlation using DA.
        """
        key = self._make_cache_key(
            factor_id_a, factor_id_b, method,
            universe_ref, period_start, period_end
        )
        return self._cache.get(key)

    def find_similar(
        self,
        factor_id: str,
        threshold: float = 0.7,
        max_results: int = 10,
        method: SimilarityMethod = SimilarityMethod.PEARSON,
        universe_ref: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> list[SimilarityResult]:
        """
        Find cached similar factors above threshold.

        In production, this would query a similarity index.
        """
        results = []
        seen_factor_ids = set()

        for key, result in self._cache.items():
            # key is (fid_a, fid_b, method_val, universe, start, end)
            fid_a, fid_b, method_val, universe, start, end = key

            if fid_a != factor_id or method_val != method.value:
                continue
            if universe_ref is not None and universe != universe_ref:
                continue
            if period_start is not None and start != period_start:
                continue
            if period_end is not None and end != period_end:
                continue

            # Avoid duplicates (we store symmetric pairs)
            if fid_b in seen_factor_ids:
                continue

            if result.is_high_similarity(threshold):
                results.append(result)
                seen_factor_ids.add(fid_b)

        # Sort by descending similarity score
        results.sort(key=lambda r: abs(r.similarity_score), reverse=True)

        return results[:max_results]

    def count(self) -> int:
        """Get total number of cached similarity results (accounting for symmetric storage)."""
        # Divide by 2 since we store symmetric pairs
        return len(self._cache) // 2

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
