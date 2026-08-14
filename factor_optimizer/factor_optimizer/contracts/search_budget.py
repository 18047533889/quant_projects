"""SearchBudget: tracking for trials, evaluations, and compute costs."""

from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
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
        if not isfinite(self.max_cost_units) or self.max_cost_units <= 0:
            raise ValueError("max_cost_units must be finite and > 0")
        if self.max_llm_calls is not None and self.max_llm_calls < 0:
            raise ValueError("max_llm_calls must be >= 0")


@dataclass
class BudgetTracker:
    """Runtime budget consumption tracker with atomic evaluation reservations."""

    budget: SearchBudget
    trials_used: int = 0
    evaluations_used: int = 0
    cost_used: float = 0.0
    llm_calls_used: int = 0
    evaluations_reserved: int = 0
    cost_reserved: float = 0.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def can_propose_trial(self) -> bool:
        """Check if budget allows another trial."""
        return self.trials_used < self.budget.max_trials

    def can_evaluate(self) -> bool:
        """Check if budget allows another evaluation."""
        with self._lock:
            return self.evaluations_used + self.evaluations_reserved < self.budget.max_evaluations

    def can_spend(self, cost: float) -> bool:
        """Check if budget allows spending cost units."""
        if not isfinite(cost) or cost < 0:
            return False
        with self._lock:
            return self.cost_used + self.cost_reserved + cost <= self.budget.max_cost_units

    def reserve_evaluation(self, cost: float) -> bool:
        """Atomically reserve one evaluation and its maximum expected cost."""
        if not isfinite(cost) or cost < 0:
            raise ValueError("evaluation cost reservation must be finite and >= 0")
        with self._lock:
            if self.evaluations_used + self.evaluations_reserved >= self.budget.max_evaluations:
                return False
            if self.cost_used + self.cost_reserved + cost > self.budget.max_cost_units:
                return False
            self.evaluations_reserved += 1
            self.cost_reserved += cost
            return True

    def commit_evaluation(self, reserved_cost: float, actual_cost: float) -> None:
        """Commit a reservation and refund any unused reserved cost."""
        if not isfinite(actual_cost) or actual_cost < 0:
            raise ValueError("actual evaluation cost must be finite and >= 0")
        if not isfinite(reserved_cost) or reserved_cost < 0:
            raise ValueError("reserved evaluation cost must be finite and >= 0")
        if actual_cost > reserved_cost:
            raise ValueError("actual evaluation cost exceeds reserved cost")
        with self._lock:
            self._require_reservation(reserved_cost)
            self.evaluations_reserved -= 1
            self.cost_reserved -= reserved_cost
            self.evaluations_used += 1
            self.cost_used += actual_cost

    def release_evaluation(self, reserved_cost: float) -> None:
        """Refund a reservation after an evaluation fails."""
        if not isfinite(reserved_cost) or reserved_cost < 0:
            raise ValueError("reserved evaluation cost must be finite and >= 0")
        with self._lock:
            self._require_reservation(reserved_cost)
            self.evaluations_reserved -= 1
            self.cost_reserved -= reserved_cost

    def _require_reservation(self, reserved_cost: float) -> None:
        if self.evaluations_reserved < 1 or self.cost_reserved < reserved_cost:
            raise RuntimeError("evaluation reservation is not active")

    def can_call_llm(self) -> bool:
        """Check if budget allows another LLM call."""
        if self.budget.max_llm_calls is None:
            return False
        return self.llm_calls_used < self.budget.max_llm_calls

    def record_trial(self) -> None:
        """Record a trial generation."""
        self.trials_used += 1

    def record_evaluation(self, cost: float = 1.0) -> None:
        """Atomically record an evaluation and its cost."""
        if not self.reserve_evaluation(cost):
            raise RuntimeError("evaluation budget unavailable")
        self.commit_evaluation(cost, cost)

    def record_llm_call(self) -> None:
        """Record an LLM API call."""
        self.llm_calls_used += 1

    def is_exhausted(self) -> bool:
        """Check if any budget limit is reached or fully reserved."""
        if self.trials_used >= self.budget.max_trials:
            return True
        with self._lock:
            if self.evaluations_used + self.evaluations_reserved >= self.budget.max_evaluations:
                return True
            if self.cost_used + self.cost_reserved >= self.budget.max_cost_units:
                return True
        if self.budget.max_llm_calls is not None and self.llm_calls_used >= self.budget.max_llm_calls:
            return True
        return False

    def remaining_trials(self) -> int:
        """Return remaining trial budget."""
        return max(0, self.budget.max_trials - self.trials_used)

    def remaining_evaluations(self) -> int:
        """Return remaining unreserved evaluation budget."""
        with self._lock:
            return max(0, self.budget.max_evaluations - self.evaluations_used - self.evaluations_reserved)

    def remaining_cost(self) -> float:
        """Return remaining unreserved cost budget."""
        with self._lock:
            return max(0.0, self.budget.max_cost_units - self.cost_used - self.cost_reserved)

    def utilization_report(self) -> dict:
        """Return budget utilization summary."""
        with self._lock:
            evaluations_reserved = self.evaluations_reserved
            cost_reserved = self.cost_reserved
        return {
            "trials": {"used": self.trials_used, "max": self.budget.max_trials},
            "evaluations": {
                "used": self.evaluations_used,
                "reserved": evaluations_reserved,
                "max": self.budget.max_evaluations,
            },
            "cost": {
                "used": self.cost_used,
                "reserved": cost_reserved,
                "max": self.budget.max_cost_units,
            },
            "llm_calls": {"used": self.llm_calls_used, "max": self.budget.max_llm_calls},
            "exhausted": self.is_exhausted(),
        }
