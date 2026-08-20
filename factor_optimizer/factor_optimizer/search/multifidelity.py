"""Multi-fidelity evaluation: L0-L4 tier mapping for progressive evaluation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FidelityTier(Enum):
    """
    Evaluation fidelity tiers from cheapest to most expensive.

    L0: Ultra-fast screening (small sample, single metric)
    L1: Fast screening (moderate sample, basic metrics)
    L2: Standard evaluation (full sample, core metrics)
    L3: High-fidelity (full sample, extended metrics, robustness)
    L4: Production-grade (full sample, all metrics, cross-validation)
    """
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


@dataclass(frozen=True)
class FidelitySpec:
    """
    Specification for evaluation at a specific fidelity tier.

    Attributes:
        tier: Fidelity tier level
        sample_fraction: Fraction of full dataset to use
        num_metrics: Number of metrics to compute
        cross_validate: Whether to perform cross-validation
        compute_robustness: Whether to compute robustness checks
        cost_multiplier: Relative cost compared to L0
        typical_duration_ms: Typical evaluation duration
    """
    tier: FidelityTier
    sample_fraction: float
    num_metrics: int
    cross_validate: bool = False
    compute_robustness: bool = False
    cost_multiplier: float = 1.0
    typical_duration_ms: int = 100

    def __post_init__(self):
        if not 0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.num_metrics < 1:
            raise ValueError("num_metrics must be >= 1")
        if self.cost_multiplier <= 0:
            raise ValueError("cost_multiplier must be > 0")


# Standard fidelity tier configurations
FIDELITY_TIERS: Dict[FidelityTier, FidelitySpec] = {
    FidelityTier.L0: FidelitySpec(
        tier=FidelityTier.L0,
        sample_fraction=0.05,
        num_metrics=1,
        cross_validate=False,
        compute_robustness=False,
        cost_multiplier=1.0,
        typical_duration_ms=50,
    ),
    FidelityTier.L1: FidelitySpec(
        tier=FidelityTier.L1,
        sample_fraction=0.15,
        num_metrics=3,
        cross_validate=False,
        compute_robustness=False,
        cost_multiplier=3.0,
        typical_duration_ms=150,
    ),
    FidelityTier.L2: FidelitySpec(
        tier=FidelityTier.L2,
        sample_fraction=0.50,
        num_metrics=5,
        cross_validate=False,
        compute_robustness=False,
        cost_multiplier=10.0,
        typical_duration_ms=500,
    ),
    FidelityTier.L3: FidelitySpec(
        tier=FidelityTier.L3,
        sample_fraction=1.0,
        num_metrics=8,
        cross_validate=True,
        compute_robustness=True,
        cost_multiplier=25.0,
        typical_duration_ms=2000,
    ),
    FidelityTier.L4: FidelitySpec(
        tier=FidelityTier.L4,
        sample_fraction=1.0,
        num_metrics=12,
        cross_validate=True,
        compute_robustness=True,
        cost_multiplier=50.0,
        typical_duration_ms=5000,
    ),
}


@dataclass
class PromotionCriteria:
    """
    Criteria for promoting a candidate to higher fidelity.

    Attributes:
        min_score: Minimum score at current tier to promote
        top_k_fraction: Fraction of top candidates to promote
        min_improvement: Minimum improvement over baseline
        require_all: Whether all criteria must be met (vs any)
    """
    min_score: Optional[float] = None
    top_k_fraction: Optional[float] = None
    min_improvement: Optional[float] = None
    require_all: bool = True

    def should_promote(
        self,
        score: float,
        rank: int,
        total: int,
        baseline_score: Optional[float] = None
    ) -> bool:
        """
        Check if candidate meets promotion criteria.

        Args:
            score: Candidate's score at current tier
            rank: Rank among peers (0 = best)
            total: Total number of candidates at this tier
            baseline_score: Baseline score for improvement comparison

        Returns:
            True if candidate should be promoted
        """
        checks = []

        if self.min_score is not None:
            checks.append(score >= self.min_score)

        if self.top_k_fraction is not None:
            threshold_rank = 0
            if total > 0 and self.top_k_fraction > 0:
                threshold_rank = max(1, int(total * self.top_k_fraction))
            checks.append(rank < threshold_rank)

        if self.min_improvement is not None and baseline_score is not None:
            improvement = score - baseline_score
            checks.append(improvement >= self.min_improvement)

        if not checks:
            return True

        return all(checks) if self.require_all else any(checks)


class MultiFidelityScheduler:
    """
    Schedules evaluation across fidelity tiers.

    Progressive evaluation: start at L0, promote promising candidates to higher tiers.
    """

    def __init__(
        self,
        tier_specs: Optional[Dict[FidelityTier, FidelitySpec]] = None,
        promotion_criteria: Optional[Dict[FidelityTier, PromotionCriteria]] = None,
    ):
        """
        Initialize scheduler.

        Args:
            tier_specs: Custom tier specifications (uses defaults if None)
            promotion_criteria: Criteria for promoting between tiers
        """
        self.tier_specs = tier_specs or FIDELITY_TIERS
        self.promotion_criteria = promotion_criteria or self._default_promotion()

    def _default_promotion(self) -> Dict[FidelityTier, PromotionCriteria]:
        """Default promotion criteria for each tier."""
        return {
            FidelityTier.L0: PromotionCriteria(top_k_fraction=0.5),
            FidelityTier.L1: PromotionCriteria(top_k_fraction=0.4),
            FidelityTier.L2: PromotionCriteria(top_k_fraction=0.3),
            FidelityTier.L3: PromotionCriteria(top_k_fraction=0.2),
        }

    def get_spec(self, tier: FidelityTier) -> FidelitySpec:
        """Get specification for a fidelity tier."""
        return self.tier_specs[tier]

    def next_tier(self, current: FidelityTier) -> Optional[FidelityTier]:
        """Get next higher fidelity tier."""
        if current == FidelityTier.L4:
            return None
        return FidelityTier(current.value + 1)

    def should_promote(
        self,
        current_tier: FidelityTier,
        score: float,
        rank: int,
        total: int,
        baseline_score: Optional[float] = None,
    ) -> bool:
        """
        Check if candidate should be promoted to next tier.

        Args:
            current_tier: Current fidelity tier
            score: Candidate's score at current tier
            rank: Rank among peers
            total: Total candidates at this tier
            baseline_score: Optional baseline for improvement check

        Returns:
            True if candidate meets promotion criteria
        """
        if current_tier not in self.promotion_criteria:
            return False

        criteria = self.promotion_criteria[current_tier]
        return criteria.should_promote(score, rank, total, baseline_score)

    def estimate_cost(self, tier: FidelityTier, num_evaluations: int = 1) -> float:
        """Estimate cost for evaluations at given tier."""
        spec = self.get_spec(tier)
        return spec.cost_multiplier * num_evaluations

    def estimate_duration_ms(self, tier: FidelityTier, num_evaluations: int = 1) -> int:
        """Estimate duration for evaluations at given tier."""
        spec = self.get_spec(tier)
        return spec.typical_duration_ms * num_evaluations

    def optimal_tier_for_budget(self, remaining_budget: float) -> FidelityTier:
        """Determine highest tier affordable with remaining budget."""
        for tier in reversed(list(FidelityTier)):
            if self.estimate_cost(tier) <= remaining_budget:
                return tier
        return FidelityTier.L0
