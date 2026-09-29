"""
Correlation-based similarity measurement.

Computes factor similarity using correlation metrics.
FA stores only bounded summary statistics, not full value arrays.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol, Optional


def _similarity_scope_key(result: "SimilarityResult") -> tuple[tuple[bool, str], ...]:
    """Return a stable ordering key for otherwise tied scope measurements."""
    return tuple(
        (value is not None, value or "")
        for value in (result.universe_ref, result.period_start, result.period_end)
    )


def _deduplicate_similar_neighbors(
    candidates: list[tuple[str, "SimilarityResult"]],
) -> list["SimilarityResult"]:
    """Keep each neighbor's strongest score with deterministic scope tie breaks."""
    best_by_neighbor: dict[str, tuple[tuple[float, tuple[tuple[bool, str], ...]], "SimilarityResult"]] = {}
    for neighbor_id, result in candidates:
        rank = (-abs(result.similarity_score), _similarity_scope_key(result))
        current = best_by_neighbor.get(neighbor_id)
        if current is None or rank < current[0]:
            best_by_neighbor[neighbor_id] = (rank, result)

    ordered = sorted(
        best_by_neighbor.items(),
        key=lambda item: (item[1][0][0], item[0], item[1][0][1]),
    )
    return [result for _, (_, result) in ordered]


def _orient_similarity_result(
    result: "SimilarityResult", factor_id_a: str, factor_id_b: str
) -> "SimilarityResult":
    """Return symmetric cached evidence in the caller's requested orientation."""
    if result.factor_id_a == factor_id_a and result.factor_id_b == factor_id_b:
        return result
    if result.factor_id_a == factor_id_b and result.factor_id_b == factor_id_a:
        return replace(result, factor_id_a=factor_id_a, factor_id_b=factor_id_b)
    raise ValueError("cached similarity result does not match the requested factor pair")


class SimilarityMethod(Enum):
    """Similarity computation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class SimilarityMeasurementStatus(Enum):
    """Measurement status of a similarity score (DLIB-FA-016).

    Distinguishes a genuinely computed value (including a computed ``0.0``)
    from an unknown / insufficient / constant-input / invalid measurement.  A
    consumer must never conflate ``UNKNOWN`` with a computed zero.
    """

    COMPUTED_VALUE = "COMPUTED_VALUE"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CONSTANT_INPUT = "CONSTANT_INPUT"
    INVALID = "INVALID"


@dataclass(frozen=True)
class SimilarityResult:
    """
    Result of similarity computation between two factors.

    Contains only summary statistics, not raw factor values.

    DLIB-FA-016: a vanishing ``0.0`` similarity must be a deliberate, recorded
    measurement, never an accidental byproduct of an empty / constant /
    insufficient sample.  ``measurement_status`` distinguishes the computed
    ``COMPUTED_VALUE`` (including a real computed zero) from ``UNKNOWN`` /
    ``INSUFFICIENT_DATA`` / ``CONSTANT_INPUT`` / ``INVALID``.  A consumer must
    never conflate ``UNKNOWN`` with a computed zero.  ``value_is_admissible``
    reports whether the ``similarity_score`` can be trusted as a measurement.
    """
    factor_id_a: str
    factor_id_b: str
    similarity_score: Optional[float]  # None unless a value is admissible
    method: SimilarityMethod
    timestamp: str  # ISO 8601
    sample_size: int
    universe_ref: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    confidence: Optional[float] = None
    warnings: tuple[str, ...] = ()
    # DLIB-FA-016: typed measurement status.  Defaults to COMPUTED_VALUE so a
    # legacy constructed result behaves exactly as before.
    measurement_status: SimilarityMeasurementStatus = SimilarityMeasurementStatus.COMPUTED_VALUE

    def __post_init__(self):
        if not self.factor_id_a:
            raise ValueError("factor_id_a is required")
        if not self.factor_id_b:
            raise ValueError("factor_id_b is required")
        if not isinstance(self.method, SimilarityMethod):
            raise TypeError("method must be a SimilarityMethod")
        if self.measurement_status is SimilarityMeasurementStatus.COMPUTED_VALUE:
            if self.similarity_score is None or not -1.0 <= self.similarity_score <= 1.0:
                raise ValueError("similarity_score must be in [-1, 1] for computed evidence")
        elif self.similarity_score is not None and not -1.0 <= self.similarity_score <= 1.0:
            raise ValueError("similarity_score must be None or in [-1, 1]")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        if not isinstance(self.measurement_status, SimilarityMeasurementStatus):
            raise TypeError("measurement_status must be a SimilarityMeasurementStatus")

    @property
    def has_warnings(self) -> bool:
        """Check if result has warnings."""
        return len(self.warnings) > 0

    @property
    def value_is_admissible(self) -> bool:
        """Whether the ``similarity_score`` is a trustworthy measurement.

        Only a COMPUTED_VALUE (including a real computed zero) is admissible.
        UNKNOWN / INSUFFICIENT_DATA / CONSTANT_INPUT / INVALID results must not
        be treated as evidence of (dis)similarity.
        """
        return self.measurement_status is SimilarityMeasurementStatus.COMPUTED_VALUE

    def is_high_similarity(self, threshold: float = 0.7) -> bool:
        """Check if similarity exceeds threshold.

        A non-computed result (UNKNOWN / INSUFFICIENT_DATA / CONSTANT_INPUT /
        INVALID) never counts as high similarity — an unmeasured pair is not
        evidence of duplication.
        """
        if not self.value_is_admissible:
            return False
        return abs(self.similarity_score) >= threshold

    def is_significant(self, min_samples: int = 30) -> bool:
        """Check if sample size is sufficient for significance.

        A non-computed result is never significant, regardless of sample size.
        """
        if not self.value_is_admissible:
            return False
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
        self._cache: dict[tuple[str, str, str, Optional[str], Optional[str], Optional[str]], SimilarityResult] = {}
        # Per-left-factor key maps preserve _cache insertion order while making
        # find_similar proportional to this factor's degree, not total cache size.
        self._factor_keys: dict[str, dict[tuple[str, str, str, Optional[str], Optional[str], Optional[str]], None]] = {}

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
        if cached is not None:
            return _orient_similarity_result(cached, factor_id_a, factor_id_b)

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
                return _orient_similarity_result(
                    similarity_result, factor_id_a, factor_id_b
                )

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
        if max_results < 0:
            raise ValueError("max_results must be non-negative")
        # In production, would delegate to QE batch correlation
        # For now, use cache
        candidates = []
        factor_keys = self._factor_keys.get(factor_id, {})
        for key in factor_keys:
            result = self._cache[key]
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

            if result.is_high_similarity(threshold):
                candidates.append((fid_b, _orient_similarity_result(result, fid_a, fid_b)))

        results = _deduplicate_similar_neighbors(candidates)
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
        for key in (key_a, key_b):
            # Dict assignment to an existing cache key preserves its global
            # insertion position; only first insertion belongs in the index.
            if key not in self._cache:
                factor_id = key[0]
                self._factor_keys.setdefault(factor_id, {})[key] = None
            self._cache[key] = result

    def _make_cache_key(
        self,
        factor_id_a: str,
        factor_id_b: str,
        method: SimilarityMethod,
        universe_ref: Optional[str],
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> tuple[str, str, str, Optional[str], Optional[str], Optional[str]]:
        """
        Build comprehensive cache key including all distinguishing parameters.

        Prevents cache collisions when same factor pair is evaluated with
        different methods, universes, or time periods.
        """
        return (
            factor_id_a,
            factor_id_b,
            method.value,
            universe_ref,
            period_start,
            period_end,
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
        result = self._cache.get(key)
        if result is None:
            return None
        return _orient_similarity_result(result, factor_id_a, factor_id_b)

    def _convert_qe_result(self, qe_result, method: SimilarityMethod) -> SimilarityResult:
        """Convert typed QE evidence; legacy aggregate-only dicts stay UNKNOWN."""
        if hasattr(qe_result, "correlation"):
            windows = tuple(getattr(qe_result, "windows", ()))
            computed = getattr(getattr(qe_result, "status", None), "value", None) == "computed"
            window_computed = all(
                getattr(getattr(w, "status", None), "value", None) == "computed"
                and w.confidence_interval[0] is not None
                for w in windows
            ) and bool(windows)
            signs = {1 if w.correlation > 0 else -1 if w.correlation < 0 else 0
                     for w in windows if w.correlation is not None}
            admissible = computed and window_computed and len(signs - {0}) <= 1
            return SimilarityResult(
                factor_id_a=qe_result.factor_id_a,
                factor_id_b=qe_result.factor_id_b,
                similarity_score=(float(qe_result.correlation) if admissible else None),
                method=method,
                timestamp="",
                sample_size=qe_result.pair_count,
                universe_ref=qe_result.universe_ref,
                period_start=qe_result.window_ref,
                period_end=qe_result.window_ref,
                measurement_status=(SimilarityMeasurementStatus.COMPUTED_VALUE
                                    if admissible else SimilarityMeasurementStatus.UNKNOWN),
            )
        # Aggregate-only legacy payload has no per-window stability or
        # uncertainty evidence. Preserve its value origin but do not certify it.
        return SimilarityResult(
            factor_id_a=qe_result.get("factor_id_a"),
            factor_id_b=qe_result.get("factor_id_b"),
            similarity_score=None,
            method=method,
            timestamp=qe_result.get("timestamp"),
            sample_size=qe_result.get("sample_size"),
            universe_ref=qe_result.get("universe_ref"),
            period_start=qe_result.get("period_start"),
            period_end=qe_result.get("period_end"),
            measurement_status=SimilarityMeasurementStatus.UNKNOWN,
        )

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._factor_keys.clear()

    def count(self) -> int:
        """Get total number of cached results (accounting for symmetric storage)."""
        self_pairs = sum(key[0] == key[1] for key in self._cache)
        return (len(self._cache) + self_pairs) // 2


class CorrelationSimilarity:
    """
    Stub implementation of correlation-based similarity.

    In production, this would delegate to a similarity service or DA.
    For testing, stores precomputed similarity results in memory.
    """

    def __init__(self):
        self._cache: dict[tuple[str, str, str, Optional[str], Optional[str], Optional[str]], SimilarityResult] = {}

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
    ) -> tuple[str, str, str, Optional[str], Optional[str], Optional[str]]:
        """
        Build comprehensive cache key including all distinguishing parameters.

        Prevents cache collisions when same factor pair is evaluated with
        different methods, universes, or time periods.
        """
        return (
            factor_id_a,
            factor_id_b,
            method.value,
            universe_ref,
            period_start,
            period_end,
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
        result = self._cache.get(key)
        if result is None:
            return None
        return _orient_similarity_result(result, factor_id_a, factor_id_b)

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
        if max_results < 0:
            raise ValueError("max_results must be non-negative")
        candidates = []

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

            if result.is_high_similarity(threshold):
                candidates.append((fid_b, _orient_similarity_result(result, fid_a, fid_b)))

        results = _deduplicate_similar_neighbors(candidates)

        return results[:max_results]

    def count(self) -> int:
        """Get total number of cached similarity results (accounting for symmetric storage)."""
        # Divide by 2 since we store symmetric pairs
        return len(self._cache) // 2

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
