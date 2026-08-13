"""Complexity budget tracking and enforcement."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from factor_optimizer.errors import BudgetExceededError
from .profile import ComplexityProfile


@dataclass
class ComplexityBudget:
    """
    Complexity budget constraints for optimization runs.

    Budgets can be specified per-dimension (cost, operator_count, lookback, etc.)
    or as aggregate limits. Used to prevent runaway complexity growth during search.

    Attributes:
        max_cost: Maximum estimated cost
        max_operator_count: Maximum number of operators
        max_lookback_periods: Maximum lookback window
        max_depth: Maximum AST depth
        max_stateful_operators: Maximum stateful operators
        max_memory_mb: Maximum estimated memory usage in MB
        strict: If True, reject any mutation exceeding budget; if False, warn only
    """

    max_cost: Optional[float] = None
    max_operator_count: Optional[int] = None
    max_lookback_periods: Optional[int] = None
    max_depth: Optional[int] = None
    max_stateful_operators: Optional[int] = None
    max_memory_mb: Optional[float] = None
    strict: bool = True

    def check(self, profile: ComplexityProfile) -> Dict[str, bool]:
        """
        Check if profile is within budget.

        Args:
            profile: ComplexityProfile to check

        Returns:
            Dictionary mapping dimension to whether it's within budget
        """
        results = {}

        if self.max_cost is not None:
            results["cost"] = profile.estimated_cost <= self.max_cost

        if self.max_operator_count is not None:
            results["operator_count"] = profile.operator_count <= self.max_operator_count

        if self.max_lookback_periods is not None:
            results["lookback_periods"] = profile.lookback_periods <= self.max_lookback_periods

        if self.max_depth is not None:
            results["depth"] = profile.max_depth <= self.max_depth

        if self.max_stateful_operators is not None:
            results["stateful_operators"] = (
                profile.stateful_operators <= self.max_stateful_operators
            )

        if self.max_memory_mb is not None and profile.metadata.get("memory_estimate_mb") is not None:
            results["memory"] = profile.metadata["memory_estimate_mb"] <= self.max_memory_mb

        return results

    def is_within_budget(self, profile: ComplexityProfile) -> bool:
        """
        Check if profile satisfies all budget constraints.

        Args:
            profile: ComplexityProfile to check

        Returns:
            True if all constraints satisfied
        """
        results = self.check(profile)
        return all(results.values()) if results else True

    def enforce(self, profile: ComplexityProfile) -> None:
        """
        Enforce budget constraints, raising exception if violated.

        Args:
            profile: ComplexityProfile to check

        Raises:
            BudgetExceededError: If strict=True and budget exceeded
        """
        results = self.check(profile)

        if not all(results.values()):
            violations = [dim for dim, ok in results.items() if not ok]
            message = f"Complexity budget exceeded: {', '.join(violations)}"

            if self.strict:
                raise BudgetExceededError(message)


@dataclass
class BudgetTracker:
    """
    Track complexity budget usage during optimization.

    Maintains history of complexity for all evaluated candidates and provides
    statistics about budget utilization.
    """

    budget: ComplexityBudget
    history: List[ComplexityProfile] = field(default_factory=list)
    violations: List[Dict[str, any]] = field(default_factory=list)

    def record(self, profile: ComplexityProfile, candidate_id: Optional[str] = None) -> bool:
        """
        Record complexity profile and check against budget.

        Args:
            profile: ComplexityProfile to record
            candidate_id: Optional identifier for candidate

        Returns:
            True if within budget, False otherwise
        """
        self.history.append(profile)

        is_within = self.budget.is_within_budget(profile)

        if not is_within:
            self.violations.append(
                {
                    "candidate_id": candidate_id,
                    "profile": profile,
                    "failed_checks": self.budget.check(profile),
                }
            )

        return is_within

    def enforce(self, profile: ComplexityProfile, candidate_id: Optional[str] = None) -> None:
        """
        Record and enforce budget constraint.

        Args:
            profile: ComplexityProfile to check
            candidate_id: Optional identifier for candidate

        Raises:
            BudgetExceededError: If budget violated and strict=True
        """
        is_within = self.record(profile, candidate_id)

        if not is_within and self.budget.strict:
            self.budget.enforce(profile)

    def stats(self) -> Dict[str, any]:
        """
        Compute statistics about budget utilization.

        Returns:
            Dictionary with utilization statistics
        """
        if not self.history:
            return {
                "total_evaluated": 0,
                "violations_count": 0,
                "violation_rate": 0.0,
            }

        total = len(self.history)
        violation_count = len(self.violations)

        # Compute max observed values
        max_cost = max((p.estimated_cost for p in self.history), default=0.0)
        max_operators = max((p.operator_count for p in self.history), default=0)
        max_lookback = max((p.lookback_periods for p in self.history), default=0)

        return {
            "total_evaluated": total,
            "violations_count": violation_count,
            "violation_rate": violation_count / total if total > 0 else 0.0,
            "max_observed_cost": max_cost,
            "max_observed_operators": max_operators,
            "max_observed_lookback": max_lookback,
            "budget_cost_utilization": (
                max_cost / self.budget.max_cost if self.budget.max_cost else None
            ),
            "budget_operators_utilization": (
                max_operators / self.budget.max_operator_count
                if self.budget.max_operator_count
                else None
            ),
        }


def create_default_budget(
    cost_multiplier: float = 1.0,
    operator_multiplier: float = 1.0,
    strict: bool = True,
) -> ComplexityBudget:
    """
    Create sensible default budget.

    Args:
        cost_multiplier: Scale default cost limit
        operator_multiplier: Scale default operator limit
        strict: Whether to enforce strictly

    Returns:
        ComplexityBudget with default limits
    """
    return ComplexityBudget(
        max_cost=100.0 * cost_multiplier,
        max_operator_count=int(20 * operator_multiplier),
        max_lookback_periods=252,  # One trading year
        max_depth=10,
        max_stateful_operators=5,
        strict=strict,
    )
