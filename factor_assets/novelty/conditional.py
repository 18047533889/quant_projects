"""
Conditional novelty computation with reverse cache and result identity.

Provides true conditional novelty assessment that considers:
- Result identity: Two factors with different formulas but identical outputs
- Cache identity: Reverse lookup from results to find equivalent factors
- Conditional assessment: Novelty relative to existing factor pool, not absolute
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Set, Tuple, runtime_checkable
import hashlib


@dataclass(frozen=True)
class NoveltyResult:
    """
    Result of conditional novelty assessment.

    Indicates whether a factor provides novel information relative to
    an existing pool, not just whether it's syntactically different.
    """
    factor_id: str
    is_novel: bool
    novelty_score: float  # 0.0 = completely redundant, 1.0 = completely novel
    similar_factor_ids: tuple[str, ...]
    assessment_method: str  # "exact", "result_identity", "conditional_ic", etc.
    pool_ref: str  # Reference to comparison pool
    evidence_refs: tuple[str, ...] = ()
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not (0.0 <= self.novelty_score <= 1.0):
            raise ValueError(f"novelty_score must be in [0,1], got {self.novelty_score}")
        if not self.assessment_method:
            raise ValueError("assessment_method is required")


@dataclass(frozen=True)
class ResultIdentity:
    """
    Identity based on factor outputs, not formula.

    Two factors with different formulas but identical outputs have
    the same result identity.  The identity is only meaningful within a
    fully-specified evaluation context: the data snapshot, universe, period,
    factor version, coordinate space and calculation spec all participate in
    the identity so that two results computed on different data or with
    different factor versions are never conflated.
    """
    result_hash: str  # Hash of factor values
    factor_id: str
    universe_ref: str
    period_start: str
    period_end: str
    num_observations: int
    data_snapshot_ref: Optional[str] = None  # data snapshot the values were computed on
    factor_version: Optional[str] = None    # version of the factor definition
    coordinate_hash: Optional[str] = None    # hash of the coordinate space (e.g. asset universe)
    calculation_spec_hash: Optional[str] = None  # hash of the calculation spec

    @classmethod
    def from_values(
        cls,
        factor_id: str,
        values: bytes,  # Serialized factor values
        universe_ref: str,
        period_start: str,
        period_end: str,
        num_observations: int,
        data_snapshot_ref: Optional[str] = None,
        factor_version: Optional[str] = None,
        coordinate_hash: Optional[str] = None,
        calculation_spec_hash: Optional[str] = None,
    ) -> "ResultIdentity":
        """
        Create result identity from factor values.

        Args:
            factor_id: Factor identifier
            values: Serialized factor values (e.g., parquet bytes)
            universe_ref: Universe reference
            period_start: Period start date
            period_end: Period end date
            num_observations: Number of observations
            data_snapshot_ref: Data snapshot the values were computed on
            factor_version: Version of the factor definition
            coordinate_hash: Hash of the coordinate space
            calculation_spec_hash: Hash of the calculation spec

        Returns:
            ResultIdentity
        """
        result_hash = hashlib.sha256(values).hexdigest()
        return cls(
            result_hash=result_hash,
            factor_id=factor_id,
            universe_ref=universe_ref,
            period_start=period_start,
            period_end=period_end,
            num_observations=num_observations,
            data_snapshot_ref=data_snapshot_ref,
            factor_version=factor_version,
            coordinate_hash=coordinate_hash,
            calculation_spec_hash=calculation_spec_hash,
        )


class ResultIdentityCache:
    """
    Reverse cache from result identity to factor IDs.

    Enables efficient lookup: "Do we already have a factor with these outputs?"

    The cache key includes EVERY result-comparability-relevant identity field
    (result hash, universe, data snapshot, period, observation count, factor
    version, coordinate hash, calculation spec hash) so that two factors are
    only considered equivalent when their full evaluation context matches.
    """

    def __init__(self):
        self._cache: dict[tuple, list[str]] = {}

    @staticmethod
    def _identity_key(
        result_identity: ResultIdentity,
    ) -> tuple:
        return (
            result_identity.result_hash,
            result_identity.universe_ref,
            result_identity.data_snapshot_ref or "",
            result_identity.period_start,
            result_identity.period_end,
            result_identity.num_observations,
            result_identity.factor_version or "",
            result_identity.coordinate_hash or "",
            result_identity.calculation_spec_hash or "",
        )

    def add(self, result_identity: ResultIdentity) -> None:
        """
        Add a result identity to the cache.

        Args:
            result_identity: Result identity to add
        """
        identity_key = self._identity_key(result_identity)
        if identity_key not in self._cache:
            self._cache[identity_key] = []
        if result_identity.factor_id not in self._cache[identity_key]:
            self._cache[identity_key].append(result_identity.factor_id)

    def find_equivalent(self, result_identity: ResultIdentity) -> tuple[str, ...]:
        """Find factors with equivalent results in the same evaluation context."""
        return tuple(self._cache.get(self._identity_key(result_identity), []))

    def has_equivalent(self, result_identity: ResultIdentity) -> bool:
        """Check for equivalent results in the same evaluation context."""
        identity_key = self._identity_key(result_identity)
        return identity_key in self._cache and bool(self._cache[identity_key])

    def count(self) -> int:
        """Get total number of unique result identities."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


@runtime_checkable
class ConditionalNoveltyProvider(Protocol):
    """
    Protocol for conditional novelty assessment.

    FA defines this protocol; implementations compute novelty metrics.
    """

    def assess_novelty(
        self,
        factor_id: str,
        result_identity: ResultIdentity,
        pool_factor_ids: tuple[str, ...],
        pool_ref: str = "default_pool",
    ) -> NoveltyResult:
        """
        Assess whether a factor is novel relative to a pool.

        Args:
            factor_id: Candidate factor ID
            pool_factor_ids: Existing factor pool for comparison
            evidence_ref: Optional evidence reference

        Returns:
            NoveltyResult with assessment
        """
        ...


class ResultIdentityNoveltyAssessor:
    """
    Exact result-identity novelty assessor using the result identity cache.

    This is EXACT result-identity novelty: a factor is novel iff no pool
    factor shares its full result identity (same outputs on the same data
    snapshot, universe, period, factor version, coordinate and calculation
    spec).  It is NOT conditional residual novelty — residual/conditional
    novelty (incremental signal on top of a similar admitted factor) is the
    separate :class:`factor_assets.adapters.residual_novelty.ResidualICNoveltyProducer`.
    """

    def __init__(self, result_cache: ResultIdentityCache):
        """
        Initialize assessor.

        Args:
            result_cache: Result identity cache
        """
        self.result_cache = result_cache

    def assess_novelty(
        self,
        factor_id: str,
        result_identity: ResultIdentity,
        pool_factor_ids: tuple[str, ...],
        pool_ref: str = "default_pool",
    ) -> NoveltyResult:
        """
        Assess novelty using result identity.

        Args:
            factor_id: Candidate factor ID
            result_identity: Result identity of candidate
            pool_factor_ids: Existing factor pool
            pool_ref: Pool reference

        Returns:
            NoveltyResult
        """
        # Check cache for equivalent results
        equivalent_ids = self.result_cache.find_equivalent(result_identity)

        # Filter to only pool factors
        pool_equivalents = tuple(fid for fid in equivalent_ids if fid in pool_factor_ids)

        if pool_equivalents:
            # Found exact result match in pool - not novel
            return NoveltyResult(
                factor_id=factor_id,
                is_novel=False,
                novelty_score=0.0,
                similar_factor_ids=pool_equivalents,
                assessment_method="result_identity",
                pool_ref=pool_ref,
                notes=f"Identical results to {len(pool_equivalents)} existing factors",
            )

        # No exact match - novel
        return NoveltyResult(
            factor_id=factor_id,
            is_novel=True,
            novelty_score=1.0,
            similar_factor_ids=(),
            assessment_method="result_identity",
            pool_ref=pool_ref,
            notes="No result-equivalent factors in pool",
        )


__all__ = [
    "NoveltyResult",
    "ResultIdentity",
    "ResultIdentityCache",
    "ConditionalNoveltyProvider",
    "ResultIdentityNoveltyAssessor",
]
