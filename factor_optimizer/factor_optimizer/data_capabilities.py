"""Real data capabilities: authorization-only scope boundaries.

P0-10: the train/validation/test evaluation contexts are logical wrappers that
all call the same ``evaluation_fn``.  This module introduces a capability
model that authorizes evaluation against a specific data scope WITHOUT holding
or referencing the underlying data.  A ``DataCapability`` knows only its scope
and an immutable boolean mask; it can say whether a split plan is consistent
with that scope, and nothing else.
"""

from enum import Enum
from typing import Dict, Tuple

from factor_optimizer.contracts.splits import SplitPlan


class DataScope(Enum):
    """The only three data scopes a split plan can authorize."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _as_bool_tuple(mask) -> Tuple[bool, ...]:
    """Coerce a validated plan mask to an immutable tuple of plain bools."""
    if not hasattr(mask, "__len__"):
        raise ValueError("mask must be a sized sequence of booleans")
    values = tuple(bool(value) for value in mask)
    if not values:
        raise ValueError("mask must not be empty")
    return values


class DataCapability:
    """Authorization for evaluating against a single data scope.

    Deliberately tiny: holds ``scope`` and an immutable boolean ``allowed_mask``.
    It does NOT hold or reference any data (train/validation/test payloads live
    in the evaluator/adapters, never in the capability).
    """

    scope: DataScope

    def __init__(self, scope: DataScope, allowed_mask):
        if not isinstance(scope, DataScope):
            raise TypeError("scope must be a DataScope")
        self._scope = scope
        self._allowed_mask = _as_bool_tuple(allowed_mask)

    @property
    def scope(self) -> DataScope:
        return self._scope

    @property
    def allowed_mask(self) -> Tuple[bool, ...]:
        return self._allowed_mask

    def can_evaluate(self, split_plan: SplitPlan) -> bool:
        """True if ``split_plan`` is consistent with this capability's scope.

        The capability is consistent when the plan's mask for this scope is
        non-empty.  The capability is deliberately scope-only: it never
        requires the plan to match the capability's own mask bit-for-bit
        (a plan may legitimately present a different split while the runner
        still holds the capability authorizing the scope).
        """
        if not isinstance(split_plan, SplitPlan):
            return False
        if self.scope is DataScope.TRAIN:
            mask = split_plan.train_mask
        elif self.scope is DataScope.VALIDATION:
            mask = split_plan.validation_mask
        else:
            mask = split_plan.test_mask
        return any(bool(value) for value in mask)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scope={self.scope.value!r}, n={len(self.allowed_mask)})"


class TrainDataCapability(DataCapability):
    """Authorization for the train scope only."""

    def __init__(self, allowed_mask):
        super().__init__(DataScope.TRAIN, allowed_mask)


class ValidationDataCapability(DataCapability):
    """Authorization for the validation scope only."""

    def __init__(self, allowed_mask):
        super().__init__(DataScope.VALIDATION, allowed_mask)


class TestDataCapability(DataCapability):
    """Authorization for the test scope only.

    This is the only capability that authorizes the test scope, and it is the
    only place a sealed-test executor may look.  The search runner's train and
    validation contexts never receive one.
    """

    def __init__(self, allowed_mask):
        super().__init__(DataScope.TEST, allowed_mask)


def build_data_capabilities(split_plan: SplitPlan) -> Dict[DataScope, DataCapability]:
    """Build one capability per scope from a validated split plan."""
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    return {
        DataScope.TRAIN: TrainDataCapability(split_plan.train_mask),
        DataScope.VALIDATION: ValidationDataCapability(split_plan.validation_mask),
        DataScope.TEST: TestDataCapability(split_plan.test_mask),
    }


__all__ = [
    "DataScope",
    "DataCapability",
    "TrainDataCapability",
    "ValidationDataCapability",
    "TestDataCapability",
    "build_data_capabilities",
]
