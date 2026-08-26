"""Soft desirability floors for factor optimization.

Maps a raw metric to a continuous [0,1] satisfaction score with NO hard-threshold
cliff (0.0249 vs 0.0251 must not jump 0->1). Support monotone-increasing and
monotone-decreasing targets via (x, d) anchor points: linearly interpolate between
anchors and clamp/plateau beyond the ends.

A separate, optional catastrophic floor exists for research-integrity-class values
(truly unacceptable), distinct from the soft floor.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Anchor = Tuple[float, float]


def desirability_for(direction: str, anchors: Sequence[Anchor], value: float) -> float:
    """Return a continuous [0,1] desirability for ``value`` given ``anchors``.

    ``direction`` is one of "increasing" (higher raw value is better) or
    "decreasing" (lower raw value is better). ``anchors`` is a list of (x, d)
    pairs, assumed already sorted in ascending ``x``. Linear interpolation between
    anchors; clamped/plateaued beyond the ends.
    """
    if not anchors:
        raise ValueError("anchors must be non-empty")
    sorted_anchors = sorted(anchors, key=lambda p: p[0])

    if direction == "increasing":
        xs = [x for x, _ in sorted_anchors]
        ds = [d for _, d in sorted_anchors]
    elif direction == "decreasing":
        # For a decreasing target, desirability is high at low x and low at high x.
        # Anchors are supplied ascending in x with descending d; take d directly.
        xs = [x for x, _ in sorted_anchors]
        ds = [d for _, d in sorted_anchors]
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    return _interp_clamped(xs, ds, value)


def _interp_clamped(xs: List[float], ds: List[float], value: float) -> float:
    """Linear interpolation with plateau beyond the anchor range."""
    n = len(xs)
    if value <= xs[0]:
        return ds[0]
    if value >= xs[-1]:
        return ds[-1]
    # Locate bracketing interval.
    lo = 0
    hi = n - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if xs[mid] <= value:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    d0, d1 = ds[lo], ds[hi]
    if x1 == x0:
        return d0
    frac = (value - x0) / (x1 - x0)
    return d0 + frac * (d1 - d0)


def catastrophic_floor(value: float, catastrophic_threshold: float = 0.01) -> float:
    """Hard floor for research-integrity-class unacceptable values.

    Distinct from the soft floor: returns ``value`` unchanged when above the
    integrity threshold, and collapses to 0.0 only when the value is truly
    unacceptable. Soft-range values are unaffected.
    """
    if value < catastrophic_threshold:
        return 0.0
    return value


# Default anchor maps for common metrics. Keys are used by the Desirability catalog.
# Each anchor list is ascending in raw x with desirability d in [0, 1].
DEFAULT_MAPS: Dict[str, Dict[str, Sequence[Anchor]]] = {
    "rank_ic": {
        "direction": "increasing",
        "anchors": [
            (0.010, 0.05),
            (0.020, 0.35),
            (0.030, 0.72),
            (0.040, 0.90),
            (0.050, 1.00),
        ],
    },
    "icir": {
        "direction": "increasing",
        "anchors": [
            (0.0, 0.05),
            (0.2, 0.30),
            (0.4, 0.55),
            (0.6, 0.75),
            (0.8, 0.88),
            (1.0, 1.00),
        ],
    },
    "turnover": {
        "direction": "decreasing",
        "anchors": [
            (0.05, 1.00),
            (0.15, 0.72),
            (0.30, 0.42),
            (0.50, 0.18),
            (0.70, 0.06),
        ],
    },
    "worst_slice": {
        "direction": "increasing",
        "anchors": [
            (-0.02, 0.05),
            (0.00, 0.20),
            (0.01, 0.45),
            (0.02, 0.72),
            (0.03, 0.90),
            (0.04, 1.00),
        ],
    },
    "cost_adjusted_alpha": {
        "direction": "increasing",
        "anchors": [
            (-0.02, 0.05),
            (0.00, 0.30),
            (0.01, 0.60),
            (0.02, 0.80),
            (0.03, 0.95),
            (0.05, 1.00),
        ],
    },
    "coverage": {
        "direction": "increasing",
        "anchors": [
            (0.50, 0.05),
            (0.70, 0.30),
            (0.85, 0.60),
            (0.95, 0.85),
            (1.00, 1.00),
        ],
    },
}


class Desirability:
    """Small catalog of common metric maps."""

    def __init__(self, maps: Dict[str, Dict[str, Sequence[Anchor]]] | None = None) -> None:
        self._maps = maps if maps is not None else DEFAULT_MAPS

    def score(self, metric: str, value: float) -> float:
        if metric not in self._maps:
            raise KeyError(f"no desirability map registered for metric: {metric!r}")
        spec = self._maps[metric]
        return desirability_for(spec["direction"], spec["anchors"], value)

    def keys(self):
        return list(self._maps.keys())
