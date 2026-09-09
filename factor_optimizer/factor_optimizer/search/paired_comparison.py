"""V5 paired common-draw factor comparison.

This module consumes resampled values, not marginal confidence intervals.  A
nonlinear utility is recomputed inside every common draw before its paired
difference is classified.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence, Tuple


class ComparisonStatus(str, Enum):
    SUPERIOR = "SUPERIOR"
    NON_INFERIOR_CHEAPER = "NON_INFERIOR_CHEAPER"
    EQUIVALENT = "EQUIVALENT"
    TRADE_OFF = "TRADE_OFF"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID_CONTEXT = "INVALID_CONTEXT"


@dataclass(frozen=True)
class PairedDraws:
    candidate_id: str
    baseline_id: str
    draw_ids: Tuple[str, ...]
    candidate_metrics: Mapping[str, Sequence[float]]
    baseline_metrics: Mapping[str, Sequence[float]]
    context_identity: str
    candidate_cost: Sequence[float] = ()
    baseline_cost: Sequence[float] = ()


@dataclass(frozen=True)
class ComparisonThresholds:
    minimum_improvement: float
    maximum_noninferiority_loss: float
    equivalence_bound: float
    minimum_cost_improvement: float = 0.0
    confidence_level: float = 0.95

    def __post_init__(self):
        for name in ("minimum_improvement", "maximum_noninferiority_loss",
                     "equivalence_bound", "minimum_cost_improvement"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class PairedComparisonResult:
    status: ComparisonStatus
    difference_interval: Tuple[float, float] | None
    mean_difference: float | None
    wins: int
    losses: int
    ties: int
    draw_count: int
    reason: str


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def compare_paired_draws(
    evidence: PairedDraws,
    thresholds: ComparisonThresholds,
    *,
    utility: Callable[[Mapping[str, float]], float],
    expected_context_identity: str,
) -> PairedComparisonResult:
    """Classify B(candidate)-A(baseline) on identical resampling draws."""
    if not expected_context_identity or evidence.context_identity != expected_context_identity:
        return PairedComparisonResult(ComparisonStatus.INVALID_CONTEXT, None, None,
                                      0, 0, 0, 0, "comparison context identity mismatch")
    names = set(evidence.candidate_metrics)
    n = len(evidence.draw_ids)
    valid = (n > 0 and len(set(evidence.draw_ids)) == n
             and names == set(evidence.baseline_metrics)
             and all(len(evidence.candidate_metrics[k]) == n and
                     len(evidence.baseline_metrics[k]) == n for k in names))
    if not valid:
        return PairedComparisonResult(ComparisonStatus.INVALID_CONTEXT, None, None,
                                      0, 0, 0, 0, "draw identity/shape/metric set mismatch")
    differences = []
    for i in range(n):
        c = {name: float(evidence.candidate_metrics[name][i]) for name in names}
        b = {name: float(evidence.baseline_metrics[name][i]) for name in names}
        try:
            difference = float(utility(c)) - float(utility(b))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return PairedComparisonResult(ComparisonStatus.INVALID_CONTEXT, None, None,
                                          0, 0, 0, 0, "utility could not be evaluated")
        if not math.isfinite(difference):
            return PairedComparisonResult(ComparisonStatus.INVALID_CONTEXT, None, None,
                                          0, 0, 0, 0, "non-finite paired utility")
        differences.append(difference)
    wins = sum(v > 0 for v in differences)
    losses = sum(v < 0 for v in differences)
    ties = n - wins - losses
    alpha = (1 - thresholds.confidence_level) / 2
    interval = (_percentile(differences, alpha), _percentile(differences, 1-alpha))
    mean = sum(differences) / n
    lo, hi = interval
    if lo >= thresholds.minimum_improvement:
        status, reason = ComparisonStatus.SUPERIOR, "paired lower bound clears minimum improvement"
    else:
        cost_ok = False
        if evidence.candidate_cost or evidence.baseline_cost:
            if len(evidence.candidate_cost) != n or len(evidence.baseline_cost) != n:
                return PairedComparisonResult(ComparisonStatus.INVALID_CONTEXT, None, None,
                                              0, 0, 0, 0, "cost draws are not paired")
            savings = [float(a)-float(b) for a, b in zip(evidence.baseline_cost,
                                                         evidence.candidate_cost)]
            cost_ok = _percentile(savings, alpha) >= thresholds.minimum_cost_improvement
        if lo >= -thresholds.maximum_noninferiority_loss and cost_ok:
            status, reason = ComparisonStatus.NON_INFERIOR_CHEAPER, "signed noninferiority and cost bounds pass"
        elif lo >= -thresholds.equivalence_bound and hi <= thresholds.equivalence_bound:
            status, reason = ComparisonStatus.EQUIVALENT, "paired interval is contained in equivalence bounds"
        elif lo > -thresholds.maximum_noninferiority_loss and hi >= thresholds.minimum_improvement:
            status, reason = ComparisonStatus.INCONCLUSIVE, "paired interval crosses a decision boundary"
        elif lo < -thresholds.equivalence_bound and hi > thresholds.equivalence_bound:
            status, reason = ComparisonStatus.TRADE_OFF, "material outcomes occur in both directions"
        else:
            status, reason = ComparisonStatus.INCONCLUSIVE, "evidence does not establish a declared relation"
    return PairedComparisonResult(status, interval, mean, wins, losses, ties, n, reason)


__all__ = ["ComparisonStatus", "PairedDraws", "ComparisonThresholds",
           "PairedComparisonResult", "compare_paired_draws"]
