"""
Split handling contracts for SearchRunner.

These are STUBS for future implementation. factor_optimizer currently
uses mock evaluation and does not handle real splits. These types document
the intended architecture when real QE integration is built.

Design Principles:
1. SplitPlan comes from caller - optimizer never creates splits (FO-001)
2. Search only sees train+validation - test hidden (FO-003)
3. Test evaluation sealed until search frozen (FO-004)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


class SplitType(Enum):
    """Split type identifier."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class SplitPlan:
    """
    Immutable split plan from external source.

    FUTURE: Passed to SearchRunner.run(), never created by optimizer.
    Ensures splits are consistent across all trials in search.
    """

    split_id: str
    train_mask: Any  # Future: numpy/pandas mask
    validation_mask: Any
    test_mask: Any
    metadata: Dict[str, Any]

    def __post_init__(self):
        raise NotImplementedError(
            "SplitPlan is a stub. Implement when QE integration is real."
        )


@dataclass
class SearchEvaluationResult:
    """
    Evaluation result visible during search.

    FUTURE: Only train + validation metrics exposed.
    Test metrics not accessible - enforces FO-003.
    """

    evaluation_id: str
    trial_id: str
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    diagnostics: Dict[str, Any]

    # NO test_metrics property

    def selection_score(self, metric: str = "rank_ic") -> float:
        """
        Get score for trial selection.

        Uses VALIDATION split only - train would overfit, test would contaminate.
        """
        return self.validation_metrics[metric]

    def __post_init__(self):
        raise NotImplementedError(
            "SearchEvaluationResult is a stub. Implement when splits are real."
        )


@dataclass
class SealedTestResult:
    """
    Test evaluation result - only accessible after search frozen.

    FUTURE: SearchSession.seal_and_test() returns this.
    Prevents test contamination (FO-004).
    """

    trial_id: str
    test_metrics: Dict[str, float]
    frozen_at: datetime
    search_session_id: str

    @classmethod
    def from_search_session(
        cls,
        session: Any,  # SearchSession
        split_plan: SplitPlan,
    ) -> 'SealedTestResult':
        """
        Re-evaluate best trial on test split after search complete.

        Raises:
            RuntimeError: If search not finished
            RuntimeError: If test already evaluated
        """
        raise NotImplementedError(
            "SealedTestResult is a stub. Implement when QE integration is real."
        )

    def __post_init__(self):
        # Allow construction with warning
        import warnings
        warnings.warn(
            "SealedTestResult is a stub. Real implementation needed for production.",
            FutureWarning
        )


# Future integration points:

def validate_split_plan(split_plan: SplitPlan) -> Dict[str, Any]:
    """
    Validate split plan before search.

    FUTURE: Check for:
    - No overlap between train/validation/test
    - Sufficient samples in each split
    - Gap between train/test if time-series
    - Consistent with evaluation period
    """
    raise NotImplementedError("Implement when splits are real")


def create_split_aware_evaluation_fn(
    qe_adapter: Any,
    split_plan: SplitPlan,
) -> Any:  # Callable[[Trial, int], SearchEvaluationResult]
    """
    Create evaluation function that respects split plan.

    FUTURE: Wraps QE adapter, enforces split permissions.
    Returns SearchEvaluationResult with train+val only.
    """
    raise NotImplementedError("Implement when QE integration is real")


__all__ = [
    "SplitType",
    "SplitPlan",
    "SearchEvaluationResult",
    "SealedTestResult",
    "validate_split_plan",
    "create_split_aware_evaluation_fn",
]
