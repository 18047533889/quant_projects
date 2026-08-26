"""Robust winner selector for factor auto-treatment optimization.

This module implements the L3 robust winner selector: among the Pareto
frontier of candidate factor recipes, pick the single best one using a
versioned, weighted utility that penalizes any single terrible dimension.

The core insight (the user's "RankIC high can't compensate terrible
Turnover" requirement) is that a candidate whose one dimension is terrible
cannot win even if all others are excellent — the ``min`` term drags it
down.  This is the robust balanced utility:

    U = alpha*min(dims) + beta*geomean(dims) + gamma*robustness - lambda*complexity

All weights live in a versioned :class:`WinnerPolicy` — no magic constants
scattered in the code.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from factor_optimizer.search.pareto import ParetoPoint


# Tolerance below which two utilities are treated as statistically
# indistinguishable for the complexity tie-break.
_UTILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class WinnerPolicy:
    """Versioned policy controlling the robust winner selector.

    Attributes:
        alpha: Weight on the minimum dimension (robustness to a single
            terrible dimension).  Must be >= 0.
        beta: Weight on the geometric mean of dimensions.  Must be >= 0.
        gamma: Weight on the robustness score.  Must be >= 0.
        lambda_: Weight on the complexity penalty (subtracted).  Must be >= 0.
        policy_id: Stable identifier for the policy family.
        policy_version: Semantic version of this policy.  Changing any
            weight must bump the version so results are reproducible and
            auditable.
    """

    alpha: float
    beta: float
    gamma: float
    lambda_: float
    policy_id: str
    policy_version: str

    def __post_init__(self):
        for name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("lambda_", self.lambda_),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty (policies are versioned)")


def _safe_geomean(values: Sequence[float]) -> float:
    """Geometric mean of ``values``, guarding non-positive / zero entries.

    The geometric mean is only defined for strictly positive values.  A
    zero (or negative) dimension is a degenerate case: the true geomean
    would be 0 (or undefined).  We return 0.0 so the term contributes
    nothing and the caller never sees NaN or a crash — the ``min`` term
    already punishes a zero dimension.
    """
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def RobustBalancedUtility(
    dimension_desirabilities: List[float],
    robustness_score: float,
    complexity_score: float,
    policy: WinnerPolicy,
) -> float:
    """Compute the robust balanced utility of a candidate recipe.

    This is the L3 robust winner selector.  A candidate whose one dimension
    is terrible cannot win even if all others are excellent: the ``min``
    term drags it down.

    Args:
        dimension_desirabilities: Desirability (0..1, higher better) of each
            objective dimension (e.g. RankIC, Turnover, Capacity).
        robustness_score: Robustness of the recipe (0..1, higher better).
        complexity_score: Complexity of the recipe (0..1, higher = more
            complex = worse; subtracted).
        policy: The versioned :class:`WinnerPolicy` controlling the weights.

    Returns:
        The scalar utility ``U``.  Higher is better.
    """
    if not dimension_desirabilities:
        raise ValueError("dimension_desirabilities cannot be empty")

    min_dim = min(dimension_desirabilities)
    geomean = _safe_geomean(dimension_desirabilities)

    return (
        policy.alpha * min_dim
        + policy.beta * geomean
        + policy.gamma * robustness_score
        - policy.lambda_ * complexity_score
    )


def augmented_tchebycheff(
    dimension_desirabilities: List[float],
    robustness_score: float,
    complexity_score: float,
    policy: WinnerPolicy,
    reference: Optional[Sequence[float]] = None,
) -> float:
    """Alternative winner metric: augmented weighted Tchebycheff scalarization.

    Provided as an illustration / alternative to :func:`RobustBalancedUtility`.
    The Tchebycheff metric minimizes the worst weighted deviation from a
    reference (ideal) point, which also punishes a single terrible dimension
    but in a max-deviation sense rather than a min-term sense.

    ``U = -max_i(w_i * (ref_i - dim_i)) - lambda*complexity``

    Higher is better (negated max-deviation).  Not the primary selector.
    """
    if not dimension_desirabilities:
        raise ValueError("dimension_desirabilities cannot be empty")

    if reference is None:
        reference = [1.0] * len(dimension_desirabilities)
    if len(reference) != len(dimension_desirabilities):
        raise ValueError("reference must match dimension count")

    # Weights: reuse alpha/beta as per-dimension weights, normalized.
    weights = [policy.alpha, policy.beta] + [policy.gamma] * (
        len(dimension_desirabilities) - 2
    )
    if len(dimension_desirabilities) == 1:
        weights = [policy.alpha]

    worst_deviation = max(
        w * max(0.0, ref - dim)
        for w, ref, dim in zip(weights, reference, dimension_desirabilities)
    )
    return -worst_deviation - policy.lambda_ * complexity_score


def select_winner(
    pareto_candidates: Iterable[ParetoPoint],
    robustness_scores: Dict[str, float],
    complexity_scores: Dict[str, float],
    policy: WinnerPolicy,
) -> ParetoPoint:
    """Select the single best candidate among the Pareto frontier.

    Picks the candidate with the maximum :func:`RobustBalancedUtility`.

    Complexity tie-break: when two candidates' utility is within a small
    tolerance (statistically indistinguishable), prefer the SIMPLER recipe
    (lower ``complexity_score``).  If still tied, fall back to the
    lexicographically smallest trial_id for determinism.

    Args:
        pareto_candidates: The non-dominated candidates (Pareto frontier).
        robustness_scores: Mapping trial_id -> robustness score.
        complexity_scores: Mapping trial_id -> complexity score.
        policy: The versioned :class:`WinnerPolicy`.

    Returns:
        The winning :class:`ParetoPoint`.
    """
    candidates = list(pareto_candidates)
    if not candidates:
        raise ValueError("cannot select a winner from an empty candidate set")

    def utility(point: ParetoPoint) -> float:
        return RobustBalancedUtility(
            dimension_desirabilities=list(point.objectives),
            robustness_score=robustness_scores.get(point.trial_id, 0.0),
            complexity_score=complexity_scores.get(point.trial_id, 0.0),
            policy=policy,
        )

    scored = [(utility(p), p) for p in candidates]

    # Sort by utility desc, then complexity asc (simpler preferred), then
    # trial_id asc for determinism.
    scored.sort(
        key=lambda up: (
            -up[0],
            complexity_scores.get(up[1].trial_id, 0.0),
            up[1].trial_id,
        )
    )

    best_utility, best = scored[0]

    # Complexity tie-break: if the runner-up is within tolerance on utility,
    # prefer the simpler recipe.  The sort already orders by complexity asc
    # within equal utility, so the first element is the simplest among the
    # tied group; we only need to confirm the tie actually exists.
    for utility_i, point in scored[1:]:
        if abs(utility_i - best_utility) <= _UTILITY_TOLERANCE:
            if complexity_scores.get(point.trial_id, 0.0) < complexity_scores.get(
                best.trial_id, 0.0
            ):
                best = point
                best_utility = utility_i
        else:
            break

    return best
