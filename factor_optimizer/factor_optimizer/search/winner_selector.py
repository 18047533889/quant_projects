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


@dataclass(frozen=True)
class WinnerSetPolicy:
    """Predeclared policy for a small complementary winner set."""

    max_winners: int = 3
    max_per_family: int = 1
    utility_epsilon: float = 1e-6
    duplicate_correlation: float = 0.995
    minimum_incremental_value: float = 0.0

    def __post_init__(self):
        if self.max_winners < 1 or self.max_per_family < 1:
            raise ValueError("winner and family caps must be positive")
        if self.utility_epsilon < 0:
            raise ValueError("utility_epsilon must be non-negative")
        if not 0 <= self.duplicate_correlation <= 1:
            raise ValueError("duplicate_correlation must be in [0,1]")


def _validate_desirabilities_and_scores(
    dimension_desirabilities: Sequence[float],
    robustness_score: float,
    complexity_score: float,
) -> None:
    """Strictly validate every desirability and score.

    Fail-closed contract: each dimension desirability and both scores must be
    finite AND within [0, 1] (their documented range).  A missing or non-finite
    value, or a value outside [0, 1], raises :class:`ValueError` naming the
    offending value — it is never clamped or silently coerced, so a poisoned
    (NaN/Inf/out-of-range) input cannot slide through as a plausible winner.

    NaN and +/-Infinity are rejected explicitly: NaN would otherwise propagate
    silently through ``min``/``log`` (yielding NaN utility) and Infinity would
    break the ``min``/deviation terms.
    """
    for idx, des in enumerate(dimension_desirabilities):
        if isinstance(des, bool) or not isinstance(des, (int, float)):
            raise ValueError(
                f"dimension_desirabilities[{idx}] must be numeric, got {des!r}"
            )
        if not math.isfinite(float(des)):
            raise ValueError(
                f"dimension_desirabilities[{idx}] must be finite, got {des!r}"
            )
        if not 0.0 <= des <= 1.0:
            raise ValueError(
                f"dimension_desirabilities[{idx}] must be in [0, 1], got {des!r}"
            )

    for name, value in (
        ("robustness_score", robustness_score),
        ("complexity_score", complexity_score),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}")


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

    Raises:
        ValueError: If ``dimension_desirabilities`` is empty, or if any
            desirability / ``robustness_score`` / ``complexity_score`` is
            non-finite or outside [0, 1].  Inputs are validated strictly and
            fail-closed; they are never clamped.
    """
    if not dimension_desirabilities:
        raise ValueError("dimension_desirabilities cannot be empty")

    _validate_desirabilities_and_scores(
        dimension_desirabilities, robustness_score, complexity_score
    )

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

    _validate_desirabilities_and_scores(
        dimension_desirabilities, robustness_score, complexity_score
    )

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

    # Fail-closed: every candidate on the Pareto frontier MUST have both a
    # robustness and a complexity score.  A missing score is never defaulted to
    # 0.0 (which would pretend a missing robustness is worst and a missing
    # complexity is simplest — silently biasing the winner).  Missing entries
    # raise a clear error naming the trial and the missing score.
    for point in candidates:
        if point.trial_id not in robustness_scores:
            raise KeyError(
                f"missing robustness score for candidate trial_id="
                f"{point.trial_id!r} on the Pareto frontier"
            )
        if point.trial_id not in complexity_scores:
            raise KeyError(
                f"missing complexity score for candidate trial_id="
                f"{point.trial_id!r} on the Pareto frontier"
            )

    def utility(point: ParetoPoint) -> float:
        return RobustBalancedUtility(
            dimension_desirabilities=list(point.objectives),
            robustness_score=robustness_scores[point.trial_id],
            complexity_score=complexity_scores[point.trial_id],
            policy=policy,
        )

    scored = [(utility(p), p) for p in candidates]

    # Sort by utility desc, then complexity asc (simpler preferred), then
    # trial_id asc for determinism.  Direct lookups are safe here: the
    # missing-score check above guarantees every candidate has an entry.
    scored.sort(
        key=lambda up: (
            -up[0],
            complexity_scores[up[1].trial_id],
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
            if complexity_scores[point.trial_id] < complexity_scores[best.trial_id]:
                best = point
                best_utility = utility_i
        else:
            break

    return best


def select_complementary_winners(
    pareto_candidates: Iterable[ParetoPoint],
    robustness_scores: Dict[str, float],
    complexity_scores: Dict[str, float],
    utility_policy: WinnerPolicy,
    set_policy: WinnerSetPolicy,
    *,
    pairwise_correlations: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[ParetoPoint]:
    """Select zero or more eligible, incremental, non-duplicate candidates.

    ``eligible``, ``family_id``, ``purpose`` and ``incremental_value`` are read
    from point metadata. Missing eligibility/incremental evidence fails closed.
    This keeps hard gates outside scalar utility and makes an empty set valid.
    """
    correlations = pairwise_correlations or {}
    eligible = []
    for point in pareto_candidates:
        if point.metadata.get("eligible") is not True:
            continue
        incremental = point.metadata.get("incremental_value")
        if (
            isinstance(incremental, bool)
            or not isinstance(incremental, (int, float))
            or not math.isfinite(float(incremental))
            or incremental < set_policy.minimum_incremental_value
        ):
            continue
        # Reuse the strict required-score validation in select_winner.
        if point.trial_id not in robustness_scores or point.trial_id not in complexity_scores:
            raise KeyError(f"missing required winner evidence for {point.trial_id!r}")
        score = RobustBalancedUtility(
            list(point.objectives), robustness_scores[point.trial_id],
            complexity_scores[point.trial_id], utility_policy,
        )
        eligible.append((score, point))

    eligible.sort(
        key=lambda item: (
            -round(item[0] / max(set_policy.utility_epsilon, 1e-15)),
            complexity_scores[item[1].trial_id], item[1].trial_id,
        )
    )
    selected: List[ParetoPoint] = []
    family_counts: Dict[str, int] = {}
    for _, point in eligible:
        family = str(point.metadata.get("family_id", point.trial_id))
        if family_counts.get(family, 0) >= set_policy.max_per_family:
            continue
        duplicate = False
        for prior in selected:
            rho = correlations.get((point.trial_id, prior.trial_id))
            if rho is None:
                rho = correlations.get((prior.trial_id, point.trial_id))
            if rho is not None and math.isfinite(rho) and abs(rho) >= set_policy.duplicate_correlation:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(point)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= set_policy.max_winners:
            break
    return selected
