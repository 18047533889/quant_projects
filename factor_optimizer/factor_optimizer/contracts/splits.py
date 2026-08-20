"""Split contracts for SearchRunner.

The validated boundary here is structural plus (optionally) temporal:
purge/embargo/label-horizon leakage is rejected when the plan carries a
time axis. It still does not sandbox evaluator access to data or prove
PIT correctness; a trusted QE adapter remains responsible for enforcing
the plan during evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitPlan:
    """Immutable, externally supplied split boundary description.

    Optional temporal-leakage fields (backward compatible: existing plans
    without them validate exactly as before):

    - ``time_index``: positional ordering of samples along the time axis.
      ``None`` means the plan is purely positional and leakage constraints
      below are unexpressible (setting any of them then fails closed).
    - ``label_horizon``: number of trailing positions a training label
      reaches into the future.
    - ``purge``: extra positions removed around the train/test boundary.
    - ``embargo``: positions after a test segment that train data must not
      occupy (coefficient/execution leakage).
    """

    split_id: str
    train_mask: Any
    validation_mask: Any
    test_mask: Any
    metadata: Dict[str, Any]
    time_index: Optional[Tuple] = None
    label_horizon: int = 0
    purge: int = 0
    embargo: int = 0

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
    _validate_temporal_leakage(split_plan, lengths[0])
    return {"split_id": split_plan.split_id, "n_samples": lengths[0], "validated": True}


def _validate_temporal_leakage(split_plan: SplitPlan, n_samples: int) -> None:
    """Fail-closed purge/embargo/label-horizon leakage check.

    Only meaningful when the plan carries a time axis: without one the
    masks are purely positional and the constraints are unexpressible,
    so requesting them without ``time_index`` is rejected outright.
    """
    label_horizon = split_plan.label_horizon
    purge = split_plan.purge
    embargo = split_plan.embargo
    for name, value in (("label_horizon", label_horizon), ("purge", purge), ("embargo", embargo)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if label_horizon == 0 and purge == 0 and embargo == 0:
        return
    time_index = split_plan.time_index
    if time_index is None:
        raise ValueError(
            "purge/embargo/label_horizon require a time_index; purely "
            "positional masks cannot express temporal leakage constraints"
        )
    if (
        not hasattr(time_index, "__len__")
        or isinstance(time_index, (str, bytes))
        or len(time_index) != n_samples
    ):
        raise ValueError(
            "time_index must be a sequence aligned with the masks "
            f"({n_samples} positions)"
        )
    # Map each mask position onto its rank on the time axis.
    try:
        order = sorted(range(n_samples), key=lambda pos: time_index[pos])
    except TypeError as exc:
        raise ValueError("time_index values must be mutually comparable") from exc
    rank_of = {pos: rank for rank, pos in enumerate(order)}

    train_positions = [
        i for i, flag in enumerate(split_plan.train_mask) if flag
    ]
    validation_positions = [
        i for i, flag in enumerate(split_plan.validation_mask) if flag
    ]
    test_positions = [i for i, flag in enumerate(split_plan.test_mask) if flag]

    # (a) A test position whose label window reaches back into a
    # train/validation position's forward-label span is leakage: the
    # train/validation label at position p spans positions
    # p..p+label_horizon, so any test position within
    # purge + label_horizon positions AFTER a train/validation position
    # (on the time axis) is rejected.
    lookback = purge + label_horizon
    if lookback > 0:
        fitted_positions = sorted(rank_of[p] for p in train_positions + validation_positions)
        for test_pos in test_positions:
            test_rank = rank_of[test_pos]
            for fitted_rank in fitted_positions:
                if fitted_rank < test_rank <= fitted_rank + lookback:
                    raise ValueError(
                        "test position falls within purge + label_horizon "
                        f"({lookback}) positions after a train/validation "
                        "position: forward-label overlap into the test segment"
                    )

    # (b) Embargo: train positions within `embargo` positions AFTER a
    # test position are rejected.
    if embargo > 0:
        test_ranks = sorted(rank_of[p] for p in test_positions)
        for train_pos in train_positions:
            train_rank = rank_of[train_pos]
            for test_rank in test_ranks:
                if test_rank < train_rank <= test_rank + embargo:
                    raise ValueError(
                        "train position falls within embargo "
                        f"({embargo}) positions after a test position"
                    )


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
