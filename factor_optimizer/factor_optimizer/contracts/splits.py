"""Split contracts for SearchRunner.

The validated boundary here is structural only: it does not sandbox evaluator
access to data or prove PIT/temporal correctness. A trusted QE adapter remains
responsible for enforcing the plan during evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict


class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitPlan:
    """Immutable, externally supplied split boundary description."""

    split_id: str
    train_mask: Any
    validation_mask: Any
    test_mask: Any
    metadata: Dict[str, Any]

    def __post_init__(self):
        validate_split_plan(self)


@dataclass(frozen=True)
class EvaluationProtocol:
    """Evaluator plus a validated split plan for SearchRunner safe mode.

    This is a contract boundary, not a data sandbox: the callback can still
    violate the plan unless its adapter enforces it.
    """

    split_plan: SplitPlan
    evaluator: Callable[[Any, int], Dict[str, Any]]

    def __post_init__(self):
        validate_split_plan(self.split_plan)
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")

    def evaluate(self, trial: Any, fidelity: int) -> Dict[str, Any]:
        return self.evaluator(trial, fidelity)


@dataclass
class SearchEvaluationResult:
    evaluation_id: str
    trial_id: str
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    diagnostics: Dict[str, Any]

    def selection_score(self, metric: str = "rank_ic") -> float:
        return self.validation_metrics[metric]


@dataclass(frozen=True)
class SealedTestHandle:
    """Immutable authority for one test-split evaluation of a frozen winner."""

    search_session_id: str
    trial_id: str
    split_id: str
    evaluation_ref: str
    frozen_at: datetime

    def __post_init__(self):
        for name in ("search_session_id", "trial_id", "split_id", "evaluation_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frozen_at, datetime):
            raise TypeError("frozen_at must be a datetime")


@dataclass(frozen=True)
class SealedTestResult:
    """Result produced by consuming a sealed handle through ``SearchSession``."""

    trial_id: str
    test_metrics: Dict[str, float]
    frozen_at: datetime
    search_session_id: str
    split_id: str
    evaluation_ref: str

    def __post_init__(self):
        if not isinstance(self.test_metrics, dict):
            raise TypeError("test_metrics must be a dict")
        for name in ("trial_id", "search_session_id", "split_id", "evaluation_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frozen_at, datetime):
            raise TypeError("frozen_at must be a datetime")


def validate_split_plan(split_plan: SplitPlan) -> Dict[str, Any]:
    """Validate non-empty, equal-length, pairwise-disjoint boolean masks."""
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    if not isinstance(split_plan.split_id, str) or not split_plan.split_id.strip():
        raise ValueError("split_id must be a non-empty string")
    if not isinstance(split_plan.metadata, dict):
        raise TypeError("metadata must be a dict")
    masks = (split_plan.train_mask, split_plan.validation_mask, split_plan.test_mask)
    lengths = []
    for name, mask in zip(("train", "validation", "test"), masks):
        if mask is None or not hasattr(mask, "__len__"):
            raise ValueError(f"{name}_mask must be a non-empty sized boundary")
        try:
            length = len(mask)
        except TypeError as exc:
            raise ValueError(f"{name}_mask must be a non-empty sized boundary") from exc
        if length == 0:
            raise ValueError(f"{name}_mask must not be empty")
        # Require exact built-in bool values; bool-like scalar types are not
        # valid mask elements at this contract boundary.
        if any(type(value) is not bool for value in mask):
            raise ValueError(f"{name}_mask must contain boolean boundaries")
        lengths.append(length)
    if len(set(lengths)) != 1:
        raise ValueError("split masks must have equal lengths")
    try:
        if any(
            (type(a) is bool and type(b) is bool and a and b)
            for i, left in enumerate(masks)
            for right in masks[i + 1:]
            for a, b in zip(left, right)
        ):
            raise ValueError("train, validation, and test masks must be disjoint")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "disjoint" in str(exc):
            raise
        raise ValueError("split masks must contain boolean boundaries") from exc
    return {"split_id": split_plan.split_id, "n_samples": lengths[0], "validated": True}


def create_split_aware_evaluation_fn(qe_adapter: Any, split_plan: SplitPlan) -> EvaluationProtocol:
    """Build the safe boundary around an adapter's evaluator."""
    if not callable(getattr(qe_adapter, "evaluate", None)):
        raise TypeError("qe_adapter must expose a callable evaluate method")
    return EvaluationProtocol(split_plan, qe_adapter.evaluate)


__all__ = [
    "SplitType", "SplitPlan", "EvaluationProtocol", "SearchEvaluationResult",
    "SealedTestHandle", "SealedTestResult", "validate_split_plan",
    "create_split_aware_evaluation_fn",
]
