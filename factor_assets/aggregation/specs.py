"""
Aggregation specifications for factor weighting and combination.

Defines weighting schemes (equal, IC-based, inverse-variance) and aggregation specs.
FA stores only metadata and references, not raw factor values or weights.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class WeightingScheme(Enum):
    """Factor weighting schemes for aggregation."""
    EQUAL = "equal"                    # Equal weight across all factors
    IC_WEIGHTED = "ic_weighted"        # Weight by information coefficient
    INVERSE_VARIANCE = "inverse_variance"  # Weight by 1/variance
    RANK_IC = "rank_ic"                # Weight by rank IC
    MEAN_ABSOLUTE_IC = "mean_abs_ic"   # Weight by mean absolute IC
    CUSTOM = "custom"                  # User-provided weights


@dataclass(frozen=True)
class AggregationSpec:
    """
    Specification for factor aggregation within a family or group.

    Defines how to combine multiple factors into a composite signal.
    Does not store raw factor values or computed weights.
    """
    spec_id: str
    name: str
    weighting_scheme: WeightingScheme
    factor_ids: tuple[str, ...]
    universe_ref: Optional[str] = None
    frequency: Optional[str] = None
    # Weighting parameters
    lookback_periods: Optional[int] = None
    min_ic_threshold: Optional[float] = None
    max_weight: Optional[float] = None
    min_weight: Optional[float] = None
    # Custom weights (only if scheme=CUSTOM)
    custom_weights: Optional[tuple[float, ...]] = None
    # Metadata
    family: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.spec_id:
            raise ValueError("spec_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.factor_ids:
            raise ValueError("factor_ids cannot be empty")

        if self.weighting_scheme == WeightingScheme.CUSTOM:
            if not self.custom_weights:
                raise ValueError("custom_weights required for CUSTOM scheme")
            if len(self.custom_weights) != len(self.factor_ids):
                raise ValueError("custom_weights length must match factor_ids")

        if self.custom_weights and self.weighting_scheme != WeightingScheme.CUSTOM:
            raise ValueError("custom_weights only valid for CUSTOM scheme")

        if self.max_weight is not None and self.min_weight is not None:
            if self.max_weight < self.min_weight:
                raise ValueError("max_weight must be >= min_weight")

    @property
    def num_factors(self) -> int:
        """Get number of factors in aggregation."""
        return len(self.factor_ids)

    @property
    def has_weight_bounds(self) -> bool:
        """Check if weight bounds are specified."""
        return self.max_weight is not None or self.min_weight is not None

    @property
    def is_equal_weighted(self) -> bool:
        """Check if using equal weighting."""
        return self.weighting_scheme == WeightingScheme.EQUAL

    @property
    def is_ic_based(self) -> bool:
        """Check if using IC-based weighting."""
        return self.weighting_scheme in (
            WeightingScheme.IC_WEIGHTED,
            WeightingScheme.RANK_IC,
            WeightingScheme.MEAN_ABSOLUTE_IC,
        )


@dataclass(frozen=True)
class AggregationResult:
    """
    Result of factor aggregation computation.

    Contains metadata about the aggregation but not the raw aggregated values.
    References the computed composite factor by ID.
    """
    result_id: str
    spec: AggregationSpec
    composite_factor_id: str
    timestamp: str  # ISO 8601
    # Weight summary (bounded, not full time series)
    computed_weights: Optional[tuple[float, ...]] = None
    weight_computation_method: Optional[str] = None
    # Quality metrics
    effective_num_factors: Optional[float] = None  # Inverse Herfindahl
    max_weight_used: Optional[float] = None
    min_weight_used: Optional[float] = None
    # Aggregation metadata
    sample_size: Optional[int] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.result_id:
            raise ValueError("result_id is required")
        if not self.composite_factor_id:
            raise ValueError("composite_factor_id is required")
        if not self.timestamp:
            raise ValueError("timestamp is required")

        if self.computed_weights:
            if len(self.computed_weights) != self.spec.num_factors:
                raise ValueError("computed_weights length must match spec.num_factors")

    @property
    def has_warnings(self) -> bool:
        """Check if aggregation produced warnings."""
        return len(self.warnings) > 0

    @property
    def has_computed_weights(self) -> bool:
        """Check if weights were computed."""
        return self.computed_weights is not None

    @property
    def weight_concentration(self) -> Optional[float]:
        """
        Compute weight concentration (Herfindahl index).

        Returns:
            Value in [1/n, 1] where 1/n = equal weight, 1 = single factor
        """
        if not self.computed_weights:
            return None
        return sum(w * w for w in self.computed_weights)
