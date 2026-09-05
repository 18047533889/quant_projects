"""Fitness adapter: dimension/desirability mapping for generalized treatment decisions.

Pure helpers shared between the decision policy and the search pipeline.
No grading logic or thresholds live here (FO has no grading authority);
grades flow in through the FA health views and are mapped to a direction
(which FA-side grade alphabet already encodes as a fixed ordering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from factor_optimizer.ports.factor_intelligence import (
    HealthDimension,
    HealthGrade,
)

__all__ = [
    "grade_to_desirability",
    "grade_lower_is_better",
    "GRADE_ORDER_ASC",
    "LOWER_IS_BETTER_DIMENSIONS",
]

#: Canonical grade order weakest -> strongest.
GRADE_ORDER_ASC = ("D", "C", "B", "B+", "A", "A+", "S", "S+")


def grade_to_desirability(grade: str) -> Optional[float]:
    """Map one FA grade to a unit desirability (0..1), None for NONE/unknown.

    The mapping is derived from the canonical grade ORDER (weakest -> strongest)
    and is monotone: every valid grade maps to a strictly larger desirability
    than the grade below it.  ``HealthGrade.NONE`` (missing) maps to ``None``
    — a missing grade is never fabricated as 0.0.
    """
    if not grade or grade == HealthGrade.NONE:
        return None
    if grade not in GRADE_ORDER_ASC:
        raise ValueError(f"unknown health grade {grade!r}")
    # Evenly space the ordered grades across (0, 1] so a monotone mapping
    # exists without hard-coding a grade->number table.
    return (GRADE_ORDER_ASC.index(grade) + 1) / len(GRADE_ORDER_ASC)


#: Health dimensions where a LOWER raw value is better (turnover etc.); the
#: FA grade alphabet already orders them so a worse (higher) raw value yields a
#: worse grade.  Mirrors the RAW-relative delta normalization: a positive delta
#: on these dimensions means better-than-RAW.
LOWER_IS_BETTER_DIMENSIONS = frozenset(
    {
        HealthDimension.TURNOVER,
        HealthDimension.CAPACITY,  # low capacity constrains; high capacity is better
        HealthDimension.COST_DRAG,
        HealthDimension.TAIL_RISK,
        HealthDimension.REGIME_SENSITIVITY,
        HealthDimension.COMPLEXITY,
    }
)


def grade_lower_is_better(dimension: str) -> bool:
    """True when the FA dimension's raw value is lower-is-better."""
    return dimension in LOWER_IS_BETTER_DIMENSIONS
