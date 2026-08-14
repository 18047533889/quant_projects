"""
Multi-fidelity evaluation policy with L0-L4 tiers.

Typed promotion criteria with plateau-based gates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FidelityTier(Enum):
    """Evaluation fidelity tiers."""

    L0 = "L0"  # 5% sample, quick screening
    L1 = "L1"  # 15% sample, initial validation
    L2 = "L2"  # 50% sample, mid-fidelity
    L3 = "L3"  # 100% sample, full evaluation
    L4 = "L4"  # 100% sample + robustness analysis


@dataclass
class FidelitySpec:
    """
    Specification for a fidelity tier.

    Attributes:
        tier: Tier identifier
        sample_fraction: Fraction of data to use
        universe_fraction: Fraction of universe to include
        metrics: Metrics to compute at this tier
        cost_multiplier: Cost relative to L0
        typical_duration_s: Expected evaluation time
        description: Human-readable description
    """

    tier: FidelityTier
    sample_fraction: float
    universe_fraction: float
    metrics: List[str]
    cost_multiplier: float
    typical_duration_s: int
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tier": self.tier.value,
            "sample_fraction": self.sample_fraction,
            "universe_fraction": self.universe_fraction,
            "metrics": list(self.metrics),
            "cost_multiplier": self.cost_multiplier,
            "typical_duration_s": self.typical_duration_s,
            "description": self.description,
        }


@dataclass
class PromotionCriteria:
    """
    Criteria for promoting to next tier.

    Attributes:
        min_metric_value: Minimum metric value required
        metric_name: Metric to check
        min_stability: Minimum plateau stability (for L3→L4)
        min_neighbor_survival: Minimum neighbor survival rate (for L3→L4)
        max_candidates_per_tier: Maximum candidates to promote
        confidence_threshold: Minimum confidence for promotion
    """

    min_metric_value: float
    metric_name: str
    min_stability: Optional[float] = None
    min_neighbor_survival: Optional[float] = None
    max_candidates_per_tier: int = 100
    confidence_threshold: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "min_metric_value": self.min_metric_value,
            "metric_name": self.metric_name,
            "min_stability": self.min_stability,
            "min_neighbor_survival": self.min_neighbor_survival,
            "max_candidates_per_tier": self.max_candidates_per_tier,
            "confidence_threshold": self.confidence_threshold,
        }


# Default fidelity specifications
DEFAULT_FIDELITY_SPECS: Dict[FidelityTier, FidelitySpec] = {
    FidelityTier.L0: FidelitySpec(
        tier=FidelityTier.L0,
        sample_fraction=0.05,
        universe_fraction=0.5,
        metrics=["rank_ic"],
        cost_multiplier=1.0,
        typical_duration_s=5,
        description="Quick screening on small sample",
    ),
    FidelityTier.L1: FidelitySpec(
        tier=FidelityTier.L1,
        sample_fraction=0.15,
        universe_fraction=0.7,
        metrics=["rank_ic", "turnover"],
        cost_multiplier=3.0,
        typical_duration_s=15,
        description="Initial validation on medium sample",
    ),
    FidelityTier.L2: FidelitySpec(
        tier=FidelityTier.L2,
        sample_fraction=0.5,
        universe_fraction=1.0,
        metrics=["rank_ic", "turnover", "sharpe", "max_drawdown"],
        cost_multiplier=10.0,
        typical_duration_s=60,
        description="Mid-fidelity evaluation on half data",
    ),
    FidelityTier.L3: FidelitySpec(
        tier=FidelityTier.L3,
        sample_fraction=1.0,
        universe_fraction=1.0,
        metrics=["rank_ic", "turnover", "sharpe", "max_drawdown", "calmar"],
        cost_multiplier=20.0,
        typical_duration_s=120,
        description="Full evaluation on complete data",
    ),
    FidelityTier.L4: FidelitySpec(
        tier=FidelityTier.L4,
        sample_fraction=1.0,
        universe_fraction=1.0,
        metrics=[
            "rank_ic",
            "turnover",
            "sharpe",
            "max_drawdown",
            "calmar",
            "stability",
            "robustness",
        ],
        cost_multiplier=40.0,
        typical_duration_s=300,
        description="Full evaluation with robustness analysis",
    ),
}


class MultiFidelityPolicy:
    """
    Multi-fidelity evaluation policy.

    Manages promotion between fidelity tiers based on typed criteria.
    """

    def __init__(
        self,
        fidelity_specs: Optional[Dict[FidelityTier, FidelitySpec]] = None,
        promotion_criteria: Optional[Dict[FidelityTier, PromotionCriteria]] = None,
    ):
        """
        Initialize policy.

        Args:
            fidelity_specs: Tier specifications (uses defaults if None)
            promotion_criteria: Promotion criteria per tier (uses defaults if None)
        """
        self.fidelity_specs = fidelity_specs or DEFAULT_FIDELITY_SPECS
        self.promotion_criteria = promotion_criteria or self._default_promotion_criteria()

    def _default_promotion_criteria(self) -> Dict[FidelityTier, PromotionCriteria]:
        """Create default promotion criteria."""
        return {
            FidelityTier.L0: PromotionCriteria(
                min_metric_value=0.01,
                metric_name="rank_ic",
                max_candidates_per_tier=100,
            ),
            FidelityTier.L1: PromotionCriteria(
                min_metric_value=0.02,
                metric_name="rank_ic",
                max_candidates_per_tier=50,
            ),
            FidelityTier.L2: PromotionCriteria(
                min_metric_value=0.03,
                metric_name="rank_ic",
                max_candidates_per_tier=20,
            ),
            FidelityTier.L3: PromotionCriteria(
                min_metric_value=0.04,
                metric_name="rank_ic",
                min_stability=0.7,
                min_neighbor_survival=0.6,
                max_candidates_per_tier=10,
            ),
        }

    def get_spec(self, tier: FidelityTier) -> FidelitySpec:
        """Get specification for tier."""
        return self.fidelity_specs[tier]

    def should_promote(
        self,
        current_tier: FidelityTier,
        metrics: Dict[str, float],
        plateau_analysis: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if candidate should be promoted to next tier.

        Args:
            current_tier: Current evaluation tier
            metrics: Metric values from current tier
            plateau_analysis: Optional plateau analysis (required for L3→L4)

        Returns:
            True if candidate should be promoted
        """
        # Cannot promote from L4
        if current_tier == FidelityTier.L4:
            return False

        # Get promotion criteria
        criteria = self.promotion_criteria.get(current_tier)
        if not criteria:
            return False

        # Check metric threshold
        metric_value = metrics.get(criteria.metric_name, float('-inf'))
        if metric_value < criteria.min_metric_value:
            return False

        # For L3→L4, check plateau stability and neighbor survival
        if current_tier == FidelityTier.L3:
            if not plateau_analysis:
                return False

            stability = plateau_analysis.get("plateau_stability", 0.0)
            if criteria.min_stability and stability < criteria.min_stability:
                return False

            neighbor_survival = plateau_analysis.get("neighbor_survival_rate", 0.0)
            if criteria.min_neighbor_survival and neighbor_survival < criteria.min_neighbor_survival:
                return False

        return True

    def get_next_tier(self, current_tier: FidelityTier) -> Optional[FidelityTier]:
        """Get next tier after current tier."""
        tier_order = [
            FidelityTier.L0,
            FidelityTier.L1,
            FidelityTier.L2,
            FidelityTier.L3,
            FidelityTier.L4,
        ]

        try:
            current_idx = tier_order.index(current_tier)
            if current_idx < len(tier_order) - 1:
                return tier_order[current_idx + 1]
        except ValueError:
            pass

        return None

    def estimate_cost(
        self,
        tier: FidelityTier,
        num_candidates: int = 1,
    ) -> float:
        """
        Estimate evaluation cost.

        Args:
            tier: Evaluation tier
            num_candidates: Number of candidates

        Returns:
            Estimated cost (in L0 units)
        """
        spec = self.fidelity_specs[tier]
        return spec.cost_multiplier * num_candidates

    def select_tier_for_budget(
        self,
        budget: float,
        num_candidates: int,
    ) -> FidelityTier:
        """
        Select highest tier within budget.

        Args:
            budget: Available budget (in L0 units)
            num_candidates: Number of candidates to evaluate

        Returns:
            Highest affordable tier
        """
        tier_order = [
            FidelityTier.L4,
            FidelityTier.L3,
            FidelityTier.L2,
            FidelityTier.L1,
            FidelityTier.L0,
        ]

        for tier in tier_order:
            cost = self.estimate_cost(tier, num_candidates)
            if cost <= budget:
                return tier

        return FidelityTier.L0
