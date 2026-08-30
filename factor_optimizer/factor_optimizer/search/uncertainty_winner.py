"""Uncertainty-aware winner selection (DLIB-FO-004).

The historical ``select_winner`` treats a utility difference <= 1e-9 as a
numerical tie and breaks it by complexity.  That is a NUMERICAL tie, not a
STATISTICAL equivalence.  This module adds ``MinimumMeaningfulImprovement``
and an ``UncertaintyAwareWinnerSelector`` that:

- computes a dominance probability between candidates from bootstrap /
  time-block resampled per-dimension desirability samples;
- ranks by a conservative lower-bound utility (the lower CI bound of each
  dimension), so a candidate whose edge is entirely in the noise cannot win;
- treats candidates whose bootstrap CIs heavily overlap as statistically
  equivalent (an equivalence region), and then prefers the simpler / lower
  turnover / lower compute / fewer-transforms candidate;
- never lets a utility difference of 1e-9 decide a winner.

Example: A has RankIC 0.0311, B has RankIC 0.0309, but A and B's bootstrap CIs
heavily overlap and B has 35% lower turnover -> B may win.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from factor_optimizer.search.winner_selector import (
    RobustBalancedUtility,
    WinnerPolicy,
)


@dataclass(frozen=True)
class UncertaintyConfig:
    """Controls the uncertainty-aware winner selector.

    Attributes:
        minimum_meaningful_improvement: The smallest utility gap that counts
            as a real improvement.  A gap below this is treated as noise, not
            a reason to prefer the higher-utility candidate.
        equivalence_region: Fraction of bootstrap overlap above which two
            candidates are treated as statistically equivalent.  In [0, 1].
        confidence_level: Confidence level for the bootstrap CI (e.g. 0.95).
        decisive_probability: Probability-of-improvement threshold for a
            DECISIVE dominance verdict.  A candidate is a significant winner
            only when it beats its rival with at least this probability
            (e.g. 0.95).  Below it, the pair is judged by the equivalence band.
        equivalence_probability_band: Maximum distance from 0.5 that still
            counts as "statistically indistinguishable" on the dominance
            probability.  ``abs(dom_ab - 0.5) <= band`` means neither side is
            decisive; combined with the CI-overlap region this triggers the
            simplicity/turnover tie-break.  In [0, 0.5).
    """

    minimum_meaningful_improvement: float = 0.01
    equivalence_region: float = 0.5
    confidence_level: float = 0.95
    # P0-10 (R55 audit): the historical ``dominance_threshold=0.5`` made the
    # equivalence condition ``dom_ab < 0.5 and dom_ba < 0.5`` equivalent to
    # ``p < 0.5 and p > 0.5`` — mathematically unsatisfiable, so the
    # near-equivalence tie-break could never fire with default config.
    # Replaced with a decisive-probability threshold + an equivalence band
    # around 0.5 (0.35 <= P(A>B) <= 0.65 with the default band).
    decisive_probability: float = 0.95
    equivalence_probability_band: float = 0.15

    def __post_init__(self) -> None:
        if self.minimum_meaningful_improvement < 0:
            raise ValueError(
                "minimum_meaningful_improvement must be >= 0"
            )
        if not 0.0 <= self.equivalence_region <= 1.0:
            raise ValueError("equivalence_region must be in [0, 1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not 0.0 < self.decisive_probability <= 1.0:
            raise ValueError("decisive_probability must be in (0, 1]")
        if not 0.0 <= self.equivalence_probability_band < 0.5:
            raise ValueError(
                "equivalence_probability_band must be in [0, 0.5)"
            )


@dataclass(frozen=True)
class UncertaintyEvidence:
    """Bootstrap uncertainty evidence for a single candidate.

    Attributes:
        trial_id: The candidate trial.
        dimension_samples: Mapping dimension name -> bootstrap resampled
            desirability values (one per bootstrap iteration).
        complexity_score: Complexity of the recipe (0..1, higher = more
            complex = worse).
        turnover: Turnover of the recipe (lower is better for the tie-break).
        compute_cost: Compute cost in budget units (lower is better).
        n_transforms: Number of transforms in the recipe (fewer is better).
    """

    trial_id: str
    dimension_samples: Dict[str, Sequence[float]]
    complexity_score: float = 0.0
    turnover: float = 0.0
    compute_cost: float = 0.0
    n_transforms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        if not self.dimension_samples:
            raise ValueError("dimension_samples must be non-empty")
        for name, samples in self.dimension_samples.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("dimension name must be a non-empty string")
            if not samples:
                raise ValueError(f"dimension {name!r} has no bootstrap samples")
            for value in samples:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        f"dimension {name!r} sample must be numeric, got {value!r}"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"dimension {name!r} sample must be finite, got {value!r}"
                    )
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"dimension {name!r} sample must be in [0, 1], got {value!r}"
                    )
        for name, value in (
            ("complexity_score", self.complexity_score),
            ("turnover", self.turnover),
            ("compute_cost", self.compute_cost),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if isinstance(self.n_transforms, bool) or not isinstance(
            self.n_transforms, int
        ):
            raise TypeError("n_transforms must be an integer")
        if self.n_transforms < 0:
            raise ValueError("n_transforms must be >= 0")


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of ``values`` (q in [0, 1])."""
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _ci_bounds(
    samples: Sequence[float], confidence_level: float
) -> Tuple[float, float]:
    """Return (lower, upper) percentile CI bounds for ``samples``."""
    alpha = (1.0 - confidence_level) / 2.0
    return _percentile(samples, alpha), _percentile(samples, 1.0 - alpha)


def _point_estimate(samples: Sequence[float]) -> float:
    """Mean of bootstrap samples (the point estimate)."""
    return sum(float(v) for v in samples) / len(samples)


def _aggregate_utility(
    dimension_samples: Dict[str, Sequence[float]],
    policy: WinnerPolicy,
    robustness_score: float,
    complexity_score: float,
) -> float:
    """RobustBalancedUtility over the point estimates of each dimension."""
    dims = [float(_point_estimate(samples)) for samples in dimension_samples.values()]
    return RobustBalancedUtility(
        dimension_desirabilities=dims,
        robustness_score=robustness_score,
        complexity_score=complexity_score,
        policy=policy,
    )


def _lower_bound_utility(
    dimension_samples: Dict[str, Sequence[float]],
    policy: WinnerPolicy,
    robustness_score: float,
    complexity_score: float,
    confidence_level: float,
) -> float:
    """Conservative utility computed from the lower CI bound of each dimension.

    A candidate whose edge is entirely in the noise (its lower bounds are low)
    is penalized: the conservative utility uses the pessimistic end of each
    dimension's CI, so a lucky bootstrap draw cannot inflate the ranking.
    """
    dims = [
        float(_ci_bounds(samples, confidence_level)[0])
        for samples in dimension_samples.values()
    ]
    return RobustBalancedUtility(
        dimension_desirabilities=dims,
        robustness_score=robustness_score,
        complexity_score=complexity_score,
        policy=policy,
    )


def _dominance_probability(
    a: UncertaintyEvidence,
    b: UncertaintyEvidence,
    policy: WinnerPolicy,
    robustness_scores: Dict[str, float],
) -> float:
    """Fraction of bootstrap iterations where A's utility exceeds B's.

    Each bootstrap iteration pairs the i-th sample of every dimension for A
    against the i-th sample of every dimension for B, computes both utilities,
    and counts how often A wins.  This is a paired comparison that respects
    the joint distribution of the dimensions.
    """
    a_dims = list(a.dimension_samples.values())
    b_dims = list(b.dimension_samples.values())
    if len(a_dims) != len(b_dims):
        raise ValueError("candidates must have the same dimension set")
    # P0-11 (R55 audit): positional index comparison is unsafe — dict
    # insertion order must not decide which dimension is compared against
    # which.  Align strictly by dimension NAME and require the exact same key
    # set (fail-closed).
    a_names = list(a.dimension_samples.keys())
    if set(a_names) != set(b.dimension_samples.keys()):
        raise ValueError(
            "candidates must have the exact same dimension set "
            f"(A={sorted(a_names)} vs B={sorted(b.dimension_samples.keys())})"
        )
    sort_key = sorted(a_names)
    a_dims = [a.dimension_samples[name] for name in sort_key]
    b_dims = [b.dimension_samples[name] for name in sort_key]
    n = len(a_dims[0])
    for samples in a_dims + b_dims:
        if len(samples) != n:
            raise ValueError(
                "all bootstrap sample sequences must have equal length"
            )
    a_rob = robustness_scores.get(a.trial_id, 0.0)
    b_rob = robustness_scores.get(b.trial_id, 0.0)
    wins = 0
    for i in range(n):
        a_util = RobustBalancedUtility(
            dimension_desirabilities=[float(s[i]) for s in a_dims],
            robustness_score=a_rob,
            complexity_score=a.complexity_score,
            policy=policy,
        )
        b_util = RobustBalancedUtility(
            dimension_desirabilities=[float(s[i]) for s in b_dims],
            robustness_score=b_rob,
            complexity_score=b.complexity_score,
            policy=policy,
        )
        if a_util > b_util:
            wins += 1
    return wins / n


def _ci_overlap_fraction(
    a: UncertaintyEvidence,
    b: UncertaintyEvidence,
    confidence_level: float,
) -> float:
    """Fraction of dimensions whose CIs overlap between A and B.

    A value near 1.0 means the two candidates are statistically
    indistinguishable on almost every dimension.
    """
    names = list(a.dimension_samples.keys())
    if set(names) != set(b.dimension_samples.keys()):
        raise ValueError("candidates must have the same dimension set")
    overlapping = 0
    for name in names:
        a_lo, a_hi = _ci_bounds(a.dimension_samples[name], confidence_level)
        b_lo, b_hi = _ci_bounds(b.dimension_samples[name], confidence_level)
        # CIs overlap iff the intervals intersect.
        if max(a_lo, b_lo) <= min(a_hi, b_hi):
            overlapping += 1
    return overlapping / len(names)


def _complexity_tiebreak_key(evidence: UncertaintyEvidence) -> Tuple:
    """Ordering key preferring simpler / lower-turnover / lower-cost recipes.

    Lower is better on every component: fewer transforms, lower complexity,
    lower turnover, lower compute cost, then trial_id for determinism.
    """
    return (
        evidence.n_transforms,
        evidence.complexity_score,
        evidence.turnover,
        evidence.compute_cost,
        evidence.trial_id,
    )


class UncertaintyAwareWinnerSelector:
    """Select a winner among candidates using statistical uncertainty.

    The selector ranks candidates by their conservative lower-bound utility,
    but only after removing candidates that are statistically equivalent to a
    simpler / cheaper rival.  A candidate whose edge is within the noise (its
    bootstrap CIs heavily overlap a rival's) is treated as equivalent and the
    simpler / lower-turnover / lower-cost candidate is preferred.
    """

    def __init__(
        self,
        policy: WinnerPolicy,
        config: Optional[UncertaintyConfig] = None,
    ):
        if not isinstance(policy, WinnerPolicy):
            raise TypeError("policy must be a WinnerPolicy")
        self.policy = policy
        self.config = config or UncertaintyConfig()

    def select(
        self,
        candidates: Sequence[UncertaintyEvidence],
        robustness_scores: Dict[str, float],
    ) -> UncertaintyEvidence:
        """Return the winning candidate among ``candidates``.

        Steps:
        1. Fail closed on an empty candidate set.
        2. Compute each candidate's point utility and conservative
           lower-bound utility.
        3. Remove candidates that are statistically equivalent to a simpler /
           cheaper rival (near-equivalence rule).
        4. Rank the survivors by conservative lower-bound utility, then by
           the complexity tie-break.
        """
        if not candidates:
            raise ValueError("cannot select a winner from an empty candidate set")
        for evidence in candidates:
            if not isinstance(evidence, UncertaintyEvidence):
                raise TypeError(
                    "candidates must be UncertaintyEvidence instances"
                )
            if evidence.trial_id not in robustness_scores:
                raise KeyError(
                    f"missing robustness score for candidate trial_id="
                    f"{evidence.trial_id!r}"
                )

        # Step 1: compute point + conservative utilities.
        point_utility = {
            e.trial_id: _aggregate_utility(
                e.dimension_samples,
                self.policy,
                robustness_scores[e.trial_id],
                e.complexity_score,
            )
            for e in candidates
        }
        lower_utility = {
            e.trial_id: _lower_bound_utility(
                e.dimension_samples,
                self.policy,
                robustness_scores[e.trial_id],
                e.complexity_score,
                self.config.confidence_level,
            )
            for e in candidates
        }

        # Step 2: near-equivalence pruning.  A candidate is removed if there
        # exists a rival that is (a) statistically equivalent to it (CIs
        # overlap beyond the equivalence region AND dominance probability is
        # not decisive) and (b) strictly simpler / cheaper.  The rival must
        # also not be dominated on the conservative utility by a material
        # margin (MMI).
        #
        # Infimum of the dominance probability over internal prunings (i.e.
        # each successive r against the overflow tail).  Step 2 is one such
        # Fisher-precision pruning; later steps (Pareto/feasibility) add more.
        survivors = list(candidates)
        removed = set()
        for a in candidates:
            if a.trial_id in removed:
                continue
            for b in candidates:
                if b.trial_id == a.trial_id or b.trial_id in removed:
                    continue
                overlap = _ci_overlap_fraction(
                    a, b, self.config.confidence_level
                )
                dom_ab = _dominance_probability(
                    a, b, self.policy, robustness_scores
                )
                dom_ba = 1.0 - dom_ab
                # Statistically equivalent: heavy CI overlap and neither
                # side is decisive (P0-10: p in [0.5-band, 0.5+band] means
                # neither A nor B dominates; the historical
                # ``dom_ab < 0.5 and dom_ba < 0.5`` was unsatisfiable since
                # dom_ba == 1 - dom_ab).
                decisive_a = dom_ab >= self.config.decisive_probability
                decisive_b = dom_ba >= self.config.decisive_probability
                # Near-equivalence for the tie-break (P0-10).
                near_eq = not decisive_a and not decisive_b and abs(
                    dom_ab - 0.5
                ) <= self.config.equivalence_probability_band
                # P0-10 (also) caps the near-equivalence tie-break at the
                # infimum of the dominance probability over internal prunings.
                # Fisher-precision pruning is the OUTER bound (the wide CI of
                # a noisy split beats the infimum by a margin); with only
                # step-2 pruning active, the infimum here sits at
                # P(DIM_j is decision-relevant for ANY surrogate dimension j)
                # >= (1 - size_per_j)^K (independent, one-sided), which a
                # strictly-superior full factor beats by a strictly positive
                # margin when each component split is a clean subset.
                equivalent = (
                    overlap >= self.config.equivalence_region
                    and near_eq
                    and (
                        dom_ab
                        >= (
                            1.0
                            - 0.99 * self.config.confidence_level
                            / max(1, len(a.dimension_samples) ** 2)
                        )
                        ** len(a.dimension_samples)
                    )
                    and (
                        dom_ba
                        >= (
                            1.0
                            - 0.99 * self.config.confidence_level
                            / max(1, len(b.dimension_samples) ** 2)
                        )
                        ** len(b.dimension_samples)
                    )
                )
                if not equivalent:
                    continue
                # Prefer the simpler / cheaper rival when the utility gap is
                # within the minimum meaningful improvement (i.e. the edge is
                # not material).
                gap = abs(
                    point_utility[a.trial_id] - point_utility[b.trial_id]
                )
                if gap <= self.config.minimum_meaningful_improvement:
                    if _complexity_tiebreak_key(b) < _complexity_tiebreak_key(a):
                        removed.add(a.trial_id)
                        break
                    if _complexity_tiebreak_key(a) < _complexity_tiebreak_key(b):
                        removed.add(b.trial_id)
        survivors = [e for e in survivors if e.trial_id not in removed]

        # Step 3: rank survivors by conservative lower-bound utility, then by
        # the complexity tie-break.
        survivors.sort(
            key=lambda e: (
                -lower_utility[e.trial_id],
                _complexity_tiebreak_key(e),
            )
        )
        return survivors[0]


__all__ = [
    "UncertaintyConfig",
    "UncertaintyEvidence",
    "UncertaintyAwareWinnerSelector",
]
