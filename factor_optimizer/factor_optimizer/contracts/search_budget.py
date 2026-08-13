"""SearchBudget: tracking for trials, evaluations, and compute costs."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchBudget:
    """
    Budget limits for a search session.

    Attributes:
        max_trials: Maximum number of mutation proposals to generate
        max_evaluations: Maximum number of full QE evaluations
        max_cost_units: Maximum compute cost (arbitrary units)
        max_llm_calls: Maximum LLM API calls (if LLM proposal enabled)
    """

    max_trials: int = 100
    max_evaluations: int = 50
    max_cost_units: float = 1000.0
    max_llm_calls: Optional[int] = None

    def __post_init__(self):
        """Validate budget limits."""
        if self.max_trials < 1:
            raise ValueError("max_trials must be >= 1")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        if self.max_cost_units <= 0:
            raise ValueError("max_cost_units must be > 0")
        if self.max_llm_calls is not None and self.max_llm_calls < 0:
            raise ValueError("max_llm_calls must be >= 0")


@dataclass
class BudgetTracker:
    """
    Runtime budget consumption tracker.

    Attributes:
        budget: The budget limits
        trials_used: Number of trials generated
        evaluations_used: Number of evaluations completed
        cost_used: Compute cost consumed
        llm_calls_used: LLM calls made
    """

    budget: SearchBudget
    trials_used: int = 0
    evaluations_used: int = 0
    cost_used: float = 0.0
    llm_calls_used: int = 0

    def can_propose_trial(self) -> bool:
        """Check if budget allows another trial."""
        return self.trials_used < self.budget.max_trials

    def can_evaluate(self) -> bool:
        """Check if budget allows another evaluation."""
        return self.evaluations_used < self.budget.max_evaluations

    def can_spend(self, cost: float) -> bool:
        """Check if budget allows spending cost units."""
        return self.cost_used + cost <= self.budget.max_cost_units

    def can_call_llm(self) -> bool:
        """Check if budget allows another LLM call."""
        if self.budget.max_llm_calls is None:
            return False
        return self.llm_calls_used < self.budget.max_llm_calls

    def record_trial(self) -> None:
        """Record a trial generation."""
        self.trials_used += 1

    def record_evaluation(self, cost: float = 1.0) -> None:
        """Record an evaluation and its cost."""
        self.evaluations_used += 1
        self.cost_used += cost

    def record_llm_call(self) -> None:
        """Record an LLM API call."""
        self.llm_calls_used += 1

    def is_exhausted(self) -> bool:
        """Check if any budget limit is reached."""
        if self.trials_used >= self.budget.max_trials:
            return True
        if self.evaluations_used >= self.budget.max_evaluations:
            return True
        if self.cost_used >= self.budget.max_cost_units:
            return True
        if self.budget.max_llm_calls is not None and self.llm_calls_used >= self.budget.max_llm_calls:
            return True
        return False

    def remaining_trials(self) -> int:
        """Return remaining trial budget."""
        return max(0, self.budget.max_trials - self.trials_used)

    def remaining_evaluations(self) -> int:
        """Return remaining evaluation budget."""
        return max(0, self.budget.max_evaluations - self.evaluations_used)

    def utilization_report(self) -> dict:
        """Return budget utilization summary."""
        return {
            "trials": {"used": self.trials_used, "max": self.budget.max_trials},
            "evaluations": {"used": self.evaluations_used, "max": self.budget.max_evaluations},
            "cost": {"used": self.cost_used, "max": self.budget.max_cost_units},
            "llm_calls": {"used": self.llm_calls_used, "max": self.budget.max_llm_calls},
            "exhausted": self.is_exhausted(),
        }
