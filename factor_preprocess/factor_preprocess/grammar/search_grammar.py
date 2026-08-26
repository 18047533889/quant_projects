"""
Search grammar for the auto-treatment optimizer.

Defines the *legal* ordering of treatment stages and a small set of certified
order templates. The grammar prevents arbitrary transform-permutation
explosion: only certified templates (and RAW) are admissible.

Stage grammar
-------------
A treatment pipeline follows the canonical stage order:

    A Missingness -> B Outlier -> C Temporal Stabilization
    -> D Neutralization -> E Representation/Scaling

Each certified template declares its preconditions and causality class.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Stage(str, Enum):
    """Canonical treatment stages in order."""

    MISSINGNESS = "A_Missingness"
    OUTLIER = "B_Outlier"
    TEMPORAL = "C_TemporalStabilization"
    NEUTRALIZATION = "D_Neutralization"
    REPRESENTATION = "E_RepresentationScaling"


# Canonical stage order.
STAGE_ORDER: Tuple[Stage, ...] = (
    Stage.MISSINGNESS,
    Stage.OUTLIER,
    Stage.TEMPORAL,
    Stage.NEUTRALIZATION,
    Stage.REPRESENTATION,
)


class CausalityClass(str, Enum):
    """Causality class of a template."""

    CAUSAL = "causal"
    CROSS_SECTIONAL = "cross_sectional"
    NO_OP = "no_op"


@dataclass(frozen=True)
class OrderTemplate:
    """A certified legal stage ordering.

    ``stages`` is the ordered list of stages in the template. ``preconditions``
    are human-readable conditions that must hold for the template to be legal.
    ``causality_class`` describes the causal safety of the template.
    """

    name: str
    stages: Tuple[Stage, ...]
    preconditions: Tuple[str, ...] = field(default_factory=tuple)
    causality_class: CausalityClass = CausalityClass.CAUSAL

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))

    def stage_names(self) -> List[str]:
        """Return the ordered stage names."""
        return [s.value for s in self.stages]


# Certified order templates. Do NOT add arbitrary permutations.
CERTIFIED_TEMPLATES: Tuple[OrderTemplate, ...] = (
    OrderTemplate(
        name="SMOOTH_NEUTRALIZE_RANK",
        stages=(
            Stage.TEMPORAL,
            Stage.NEUTRALIZATION,
            Stage.REPRESENTATION,
        ),
        preconditions=(
            "temporal smoothing precedes neutralization",
            "neutralization precedes rank representation",
        ),
        causality_class=CausalityClass.CAUSAL,
    ),
    OrderTemplate(
        name="NEUTRALIZE_SMOOTH_RANK",
        stages=(
            Stage.NEUTRALIZATION,
            Stage.TEMPORAL,
            Stage.REPRESENTATION,
        ),
        preconditions=(
            "neutralization precedes temporal smoothing",
            "temporal smoothing precedes rank representation",
        ),
        causality_class=CausalityClass.CAUSAL,
    ),
    OrderTemplate(
        name="WINSOR_NEUTRALIZE_ZSCORE",
        stages=(
            Stage.OUTLIER,
            Stage.NEUTRALIZATION,
            Stage.REPRESENTATION,
        ),
        preconditions=(
            "winsor (outlier) precedes neutralization",
            "neutralization precedes zscore representation",
        ),
        causality_class=CausalityClass.CAUSAL,
    ),
    OrderTemplate(
        name="RAW",
        stages=(),
        preconditions=("no treatment applied",),
        causality_class=CausalityClass.NO_OP,
    ),
)


def is_certified_template(stages: Tuple[Stage, ...]) -> bool:
    """True if the given stage sequence matches a certified template."""
    for template in CERTIFIED_TEMPLATES:
        if template.stages == tuple(stages):
            return True
    return False


def validate_stage_order(stages: Tuple[Stage, ...]) -> Tuple[bool, List[str]]:
    """Validate that a stage sequence respects the canonical stage order.

    Returns ``(valid, errors)``. A sequence is valid if it is a certified
    template OR a subsequence of the canonical stage order (no stage appears
    out of order).
    """
    errors: List[str] = []

    # RAW (empty) is always valid.
    if not stages:
        return True, errors

    # Certified templates are always valid.
    if is_certified_template(stages):
        return True, errors

    # Otherwise, stages must appear in canonical order (no reordering).
    order_index = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    last = -1
    for stage in stages:
        idx = order_index.get(stage)
        if idx is None:
            errors.append(f"unknown stage {stage!r}")
            continue
        if idx < last:
            errors.append(f"stage {stage.value} out of canonical order")
        last = idx

    return len(errors) == 0, errors


__all__ = [
    "Stage",
    "STAGE_ORDER",
    "CausalityClass",
    "OrderTemplate",
    "CERTIFIED_TEMPLATES",
    "is_certified_template",
    "validate_stage_order",
]
