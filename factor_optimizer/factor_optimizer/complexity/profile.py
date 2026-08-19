"""ComplexityProfile: adapter projection of FE complexity analysis."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ComplexityProfile:
    """
    Complexity estimate for a factor or mutation.

    This is a projection/adapter over FE's internal complexity analysis.
    FO does not reimplement complexity calculation; it consumes FE's output.

    Attributes:
        operator_count: Total number of operators
        max_depth: Maximum AST depth
        lookback_periods: Maximum lookback window
        stateful_operators: Number of stateful operators
        cross_sectional_operators: Number of CS operators
        nonlinear_operators: Number of nonlinear operators
        estimated_cost: Estimated compute cost (arbitrary units)
        domains: Data domains required
        sources: Data sources required
        estimated_latency_ms: Estimated execution time
        metadata: Additional complexity metadata from FE
    """

    operator_count: int = 0
    max_depth: int = 0
    lookback_periods: int = 0
    stateful_operators: int = 0
    cross_sectional_operators: int = 0
    nonlinear_operators: int = 0
    estimated_cost: float = 0.0
    domains: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    estimated_latency_ms: Optional[float] = None
    memory_estimate: Optional[float] = None  # Estimated memory in MB
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operator_count": self.operator_count,
            "max_depth": self.max_depth,
            "lookback_periods": self.lookback_periods,
            "stateful_operators": self.stateful_operators,
            "cross_sectional_operators": self.cross_sectional_operators,
            "nonlinear_operators": self.nonlinear_operators,
            "estimated_cost": self.estimated_cost,
            "domains": list(self.domains),
            "sources": list(self.sources),
            "estimated_latency_ms": self.estimated_latency_ms,
            "memory_estimate": self.memory_estimate,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplexityProfile":
        """Deserialize from dictionary (fail-closed on unknown fields)."""
        known = {
            "operator_count", "max_depth", "lookback_periods", "stateful_operators",
            "cross_sectional_operators", "nonlinear_operators", "estimated_cost",
            "domains", "sources", "estimated_latency_ms", "memory_estimate",
            "metadata",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ComplexityProfile fields: {sorted(unknown)}")
        return cls(**data)

    def is_within_budget(self, max_cost: float) -> bool:
        """Check if complexity is within budget."""
        return self.estimated_cost <= max_cost


class ComplexityEstimator:
    """
    Estimate complexity through FE adapter.

    This does NOT reimplement FE's complexity analysis. It wraps the FE adapter
    to provide a consistent interface for FO.
    """

    def __init__(self, fe_adapter=None):
        """
        Initialize estimator.

        Args:
            fe_adapter: Optional FE adapter with complexity analysis capability
        """
        self.fe_adapter = fe_adapter

    def estimate(self, factor_definition: Any) -> ComplexityProfile:
        """
        Estimate complexity of a factor definition.

        Args:
            factor_definition: Factor definition (format depends on FE adapter)

        Returns:
            ComplexityProfile with estimates

        Raises:
            RuntimeError: If FE adapter not available
        """
        if self.fe_adapter is None:
            raise RuntimeError("FE adapter required for complexity estimation")

        if not hasattr(self.fe_adapter, "estimate_complexity"):
            # Fallback: return minimal profile
            return ComplexityProfile(operator_count=1, max_depth=1, estimated_cost=1.0)

        # Delegate to FE adapter
        fe_result = self.fe_adapter.estimate_complexity(factor_definition)

        # Convert FE result to ComplexityProfile
        return self._convert_fe_result(fe_result)

    def _convert_fe_result(self, fe_result: Any) -> ComplexityProfile:
        """
        Convert FE complexity result to ComplexityProfile.

        This is adapter-specific logic. Real implementation depends on FE's
        actual complexity output format.
        """
        if isinstance(fe_result, dict):
            # Assume dict with matching field names
            return ComplexityProfile.from_dict(fe_result)
        elif hasattr(fe_result, "to_dict"):
            # Assume FE object with to_dict method
            return ComplexityProfile.from_dict(fe_result.to_dict())
        else:
            # Fallback: minimal profile
            return ComplexityProfile(
                operator_count=1,
                max_depth=1,
                estimated_cost=1.0,
                metadata={"fe_result": str(fe_result)},
            )

    def estimate_mutation_delta(
        self, parent_profile: ComplexityProfile, mutation_type: str, parameters: Dict[str, Any]
    ) -> ComplexityProfile:
        """
        Estimate complexity change from a mutation.

        This is a rough heuristic estimate. Precise calculation requires FE compilation.

        Args:
            parent_profile: Parent factor complexity
            mutation_type: Mutation operation type
            parameters: Mutation parameters

        Returns:
            Estimated new ComplexityProfile
        """
        # Simple heuristics for common mutations
        new_profile = ComplexityProfile(
            operator_count=parent_profile.operator_count,
            max_depth=parent_profile.max_depth,
            lookback_periods=parent_profile.lookback_periods,
            stateful_operators=parent_profile.stateful_operators,
            cross_sectional_operators=parent_profile.cross_sectional_operators,
            nonlinear_operators=parent_profile.nonlinear_operators,
            estimated_cost=parent_profile.estimated_cost,
            domains=list(parent_profile.domains),
            sources=list(parent_profile.sources),
        )

        # Adjust based on mutation type
        if mutation_type == "window_adjust":
            new_window = parameters.get("new_window", parent_profile.lookback_periods)
            # A non-numeric or non-positive window silently produced a
            # garbage lookback (or ZeroDivision-by-1-masked cost) profile.
            # NaN/inf must be rejected too: NaN comparisons are always False
            # and inf poisons the cost estimate.
            if (
                not isinstance(new_window, (int, float))
                or isinstance(new_window, bool)
                or not math.isfinite(new_window)
                or new_window <= 0
            ):
                raise ValueError(
                    "window_adjust requires a positive finite numeric new_window"
                )
            new_profile.lookback_periods = new_window
            # Cost scales roughly with window
            cost_ratio = new_window / max(parent_profile.lookback_periods, 1)
            new_profile.estimated_cost = parent_profile.estimated_cost * cost_ratio

        elif mutation_type == "linear_combination":
            # Composition adds operators and increases cost
            new_profile.operator_count += 1
            new_profile.max_depth = max(parent_profile.max_depth + 1, new_profile.max_depth)
            new_profile.estimated_cost = parent_profile.estimated_cost * 1.5

        elif mutation_type in {"parameter_tune", "decay_adjust", "threshold_adjust"}:
            # Parameter changes don't affect structure, minimal cost change
            new_profile.estimated_cost = parent_profile.estimated_cost * 1.05

        elif mutation_type == "operator_swap":
            # Depends on operator, assume similar cost
            new_profile.estimated_cost = parent_profile.estimated_cost * 1.1

        new_profile.metadata["mutation_type"] = mutation_type
        new_profile.metadata["estimated_from_heuristics"] = True

        return new_profile
