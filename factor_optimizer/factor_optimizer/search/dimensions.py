"""Dimension aggregation primitives for factor optimization.

Provides the per-dimension naming, a frozen per-dimension score container, and the
bottleneck/geomean aggregation primitives (low-quantile / geometric mean with a min
penalty) that the sibling-owned RobustBalancedUtility will call. The final utility
is intentionally NOT defined here.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List


class DimensionName(enum.Enum):
    """The balanced-quality dimensions a candidate factor is scored on."""

    PREDICTIVE = "predictive"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"
    TRADABILITY = "tradability"
    PURITY_EXPOSURE = "purity_exposure"
    DATA_QUALITY = "data_quality"
    # COMPLEXITY is a penalty dimension, kept separate from the balanced set.
    COMPLEXITY = "complexity"


BALANCED_DIMENSIONS = frozenset(
    {
        DimensionName.PREDICTIVE,
        DimensionName.STABILITY,
        DimensionName.ROBUSTNESS,
        DimensionName.TRADABILITY,
        DimensionName.PURITY_EXPOSURE,
        DimensionName.DATA_QUALITY,
    }
)


@dataclass(frozen=True)
class DimensionScores:
    """Per-dimension desirability plus the raw metrics that produced it."""

    dimension: DimensionName
    desirability: float
    raw_metrics: Dict[str, float] = field(default_factory=dict)


def bottleneck_min(desirabilities: List[float]) -> float:
    """Lowest dimension desirability: the strictest bottleneck."""
    if not desirabilities:
        raise ValueError("desirabilities must be non-empty")
    return min(desirabilities)


def geomean_floor(
    desirabilities: List[float], floor: float = 1e-6, min_penalty: float = 0.5
) -> float:
    """Geometric mean with a min penalty.

    A single very poor dimension drags the aggregate far below the arithmetic mean
    because the product is dominated by its weakest member. Values are clamped to
    ``floor`` before taking logs to avoid zeroing out the whole aggregate.
    ``min_penalty`` blends toward the minimum so a lone bad dimension still bites
    even when the geometric mean alone would not be punitive enough.
    """
    if not desirabilities:
        raise ValueError("desirabilities must be non-empty")
    clamped = [max(v, floor) for v in desirabilities]
    gm = math.exp(sum(math.log(v) for v in clamped) / len(clamped))
    mn = min(clamped)
    # Blend geometric mean with the minimum: a bad dimension dominates.
    return gm * (1.0 - min_penalty) + mn * min_penalty


def aggregate_dimension(
    desirabilities: List[float],
    *,
    method: str = "bottleneck_geomean",
    low_quantile: float = 0.2,
    min_penalty: float = 0.5,
) -> float:
    """Aggregate per-dimension desirabilities into a single score.

    Default ``bottleneck_geomean`` uses a geometric mean blended with the minimum,
    so a very high metric cannot fully mask a very poor one. ``low_quantile``
    selects the q-th quantile of the sorted desirabilities (a low-quantile blend),
    and ``min`` returns the raw minimum.
    """
    if not desirabilities:
        raise ValueError("desirabilities must be non-empty")

    if method == "min":
        return bottleneck_min(desirabilities)
    if method == "geomean":
        return geomean_floor(desirabilities, min_penalty=min_penalty)
    if method == "bottleneck_geomean":
        gm = geomean_floor(desirabilities, min_penalty=min_penalty)
        mn = bottleneck_min(desirabilities)
        # Weighted blend: keep the geometric-mean signal but never fully hide the min.
        return 0.5 * gm + 0.5 * mn
    if method == "low_quantile":
        if not 0.0 < low_quantile <= 1.0:
            raise ValueError("low_quantile must be in (0, 1]")
        sorted_vals = sorted(desirabilities)
        idx = max(0, min(len(sorted_vals) - 1, int(low_quantile * len(sorted_vals))))
        # Blend the chosen low quantile with the minimum so a tail below it still bites.
        return 0.6 * sorted_vals[idx] + 0.4 * sorted_vals[0]
    raise ValueError(f"unknown aggregation method: {method!r}")
