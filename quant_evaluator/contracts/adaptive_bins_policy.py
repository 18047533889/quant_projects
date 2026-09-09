"""
Adaptive quantile-bins policy contract (plan section 14.1).

The A-share daily cross-section is used at 10 bins for first-pass screening
and 20 bins for suspicious/surviving factors (plan §14.1).  A fixed global
bin count is wrong twice: a small date/universe cannot support 20 bins with
enough effective names per bin, and a 20-bin evaluation of a factor with
fewer distinct names fabricates empty/sparse buckets.  This module defines
the versioned policy object that selects the largest feasible bin count and
*records* what was actually used:

- ``AdaptiveBinsPolicy(preferred_bins=20, fallback_bins=(10, 5),
  min_effective_names_per_bin=100)`` (plan §14.1) — the exact parameter
  triple from the plan.
- ``resolve_bin_count(names_per_date, policy=None)`` — the deterministic
  resolution rule: the largest bin count in ``(preferred_bins, *fallback_bins)``
  for which every date in the panel has at least
  ``names_per_date * min_effective_names_per_bin`` weighted-thousand shares
  (i.e. ``min_effective_names_per_bin * bins <= names_per_date``) is selected.
  The *actual* bin count is always recorded (the ``adaptive_quantile_count``
  metric), never assumed from the preferred value.
- ``AdaptiveBinsResolution`` — frozen result carrying the selected bin count
  and the reason fallback was applied (``"preferred"`` /
  ``"fallback"`` / ``"insufficient"``).

Everything here is a pure CPU reference: no GPU path, no state, no imports
from the metric registry.  Missing evidence (no feasible bin count) is an
explicit ``insufficient`` resolution — never a silent 0 — so callers can map
it to the existing ``EvidenceStatus.INSUFFICIENT_DATA`` vocabulary.

QE-BINS-P0-001: ``resolve_bin_count`` is required to return the SAME result
for the same (names_per_date, policy) inputs on every platform — the rule is
pure arithmetic over the plan's parameter triple.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "AdaptiveBinsPolicy",
    "AdaptiveBinsResolution",
    "AdaptiveBinsDateEvidence",
    "AdaptiveBinsFactorEvidence",
    "resolve_bin_count",
    "resolve_bin_count_from_counts",
]


@dataclass(frozen=True)
class AdaptiveBinsPolicy:
    """Versioned adaptive-bin selection policy (plan §14.1).

    Attributes:
        preferred_bins: Preferred bin count for surviving/suspicious factors
            (20 in the plan).
        fallback_bins: Ordered fallback bin counts (10 then 5 in the plan).
        min_effective_names_per_bin: Minimum average effective names each
            bin must contain (100 in the plan — A-share daily panels provide
            thousands of names, so 20x100=2000 names is a normal floor).
        policy_id / policy_version: Versioned-policy identity so consumers
            can record which policy produced a bin count.
    """

    preferred_bins: int = 20
    fallback_bins: Sequence[int] = (10, 5)
    min_effective_names_per_bin: int = 100
    policy_id: str = "QE_ADAPTIVE_BINS"
    policy_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if isinstance(self.preferred_bins, bool) or not isinstance(self.preferred_bins, Integral) or self.preferred_bins < 2:
            raise ValueError(
                f"AdaptiveBinsPolicy.preferred_bins must be >= 2, got "
                f"{self.preferred_bins}"
            )
        fallback = tuple(self.fallback_bins)
        if any(isinstance(b,bool) or not isinstance(b,Integral) or b < 2 for b in fallback):
            raise ValueError(
                f"AdaptiveBinsPolicy.fallback_bins must all be >= 2, got "
                f"{fallback}"
            )
        if len(set((self.preferred_bins, *fallback))) != 1 + len(fallback):
            raise ValueError(
                "AdaptiveBinsPolicy bin counts (preferred + fallback) must be "
                "distinct"
            )
        if any(a <= b for a,b in zip((self.preferred_bins,*fallback), fallback)):
            raise ValueError("AdaptiveBinsPolicy candidates must be strictly descending")
        if isinstance(self.min_effective_names_per_bin,bool) or not isinstance(self.min_effective_names_per_bin,Integral) or self.min_effective_names_per_bin < 1:
            raise ValueError(
                f"AdaptiveBinsPolicy.min_effective_names_per_bin must be >= 1, "
                f"got {self.min_effective_names_per_bin}"
            )
        if not self.policy_id.strip() or not self.policy_version.strip():
            raise ValueError(
                "AdaptiveBinsPolicy policy_id/policy_version must be non-empty"
            )
        object.__setattr__(self, "fallback_bins", fallback)

    def candidate_bin_counts(self):
        """Bin counts tried in order of preference, longest first."""
        return (self.preferred_bins, *self.fallback_bins)

    def to_dict(self) -> Dict[str, object]:
        """Canonical JSON-safe policy representation for request transport."""
        return {
            "preferred_bins": self.preferred_bins,
            "fallback_bins": tuple(self.fallback_bins),
            "min_effective_names_per_bin": self.min_effective_names_per_bin,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AdaptiveBinsPolicy":
        """Normalize a transported mapping through the policy validator."""
        if not isinstance(value, Mapping):
            raise TypeError("AdaptiveBinsPolicy.from_dict requires a mapping")
        allowed = {
            "preferred_bins", "fallback_bins", "min_effective_names_per_bin",
            "policy_id", "policy_version",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown AdaptiveBinsPolicy fields: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class AdaptiveBinsResolution:
    """Frozen outcome of an adaptive-bin resolution.

    Attributes:
        bin_count: The selected bin count, or ``None`` when the panel cannot
            support even the smallest fallback (explicit insufficiency —
            never a fabricated 0).
        reason: ``"preferred"`` when the preferred count was feasible,
            ``"fallback"`` when a fallback count was selected, or
            ``"insufficient"`` when no count was feasible.
        min_names_per_bin: The observed minimum ``names_per_date /
            bin_count`` across dates at the selected count
            (``None`` when insufficient).
        policy_id / policy_version: Which policy produced the resolution.
    """

    bin_count: Optional[int]
    reason: str
    min_names_per_bin: Optional[float] = None
    policy_id: str = "QE_ADAPTIVE_BINS"
    policy_version: str = "1.0.0"

    @property
    def is_insufficient(self) -> bool:
        """True when no feasible bin count exists (evidence is missing)."""
        return self.reason == "insufficient"

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "bin_count": self.bin_count,
            "reason": self.reason,
            "min_names_per_bin": self.min_names_per_bin,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class AdaptiveBinsDateEvidence:
    """Tie-aware feasibility evidence for one factor on one date."""

    date_index: int
    applicable: bool
    finite_names: int
    distinct_levels: int
    candidate_min_bucket_counts: Tuple[Tuple[int, Optional[int]], ...]
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "date_index": self.date_index,
            "applicable": self.applicable,
            "finite_names": self.finite_names,
            "distinct_levels": self.distinct_levels,
            "candidate_min_bucket_counts": {
                str(q): count for q, count in self.candidate_min_bucket_counts
            },
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AdaptiveBinsFactorEvidence:
    """Fixed-Q adaptive-bin decision and its complete per-date coverage."""

    factor_id: str
    resolution: AdaptiveBinsResolution
    tie_policy: str
    comparison_policy: str
    dates: Tuple[AdaptiveBinsDateEvidence, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "factor_id": self.factor_id,
            "selected_q": self.resolution.bin_count,
            "fallback_reason": self.resolution.reason,
            "min_names_per_bin": self.resolution.min_names_per_bin,
            "policy_id": self.resolution.policy_id,
            "policy_version": self.resolution.policy_version,
            "tie_policy": self.tie_policy,
            "comparison_policy": self.comparison_policy,
            "applicable_dates": sum(row.applicable for row in self.dates),
            "total_dates": len(self.dates),
            "dates": tuple(row.to_dict() for row in self.dates),
        }


def resolve_bin_count(
    names_per_date: Sequence[int],
    policy: Optional[AdaptiveBinsPolicy] = None,
) -> AdaptiveBinsResolution:
    """Select the largest feasible quantile bin count for a panel.

    Args:
        names_per_date: Effective-name counts per date (cross-sectional
            sizes of the panel, in the same order as the panel dates).
        policy: The versioned policy (defaults to the plan §14.1 triple
            ``AdaptiveBinsPolicy()``).

    Returns:
        Frozen :class:`AdaptiveBinsResolution` with the selected bin count
        and reason.  The count is ``None`` only when even the smallest
        fallback is infeasible — the caller must map that to explicit
        ``INSUFFICIENT_DATA`` evidence, never 0.
    """
    if policy is None:
        policy = AdaptiveBinsPolicy()
    if not isinstance(policy, AdaptiveBinsPolicy):
        raise TypeError(
            f"resolve_bin_count expects AdaptiveBinsPolicy, got "
            f"{type(policy).__name__}"
        )
    if len(names_per_date) == 0:
        return AdaptiveBinsResolution(
            bin_count=None,
            reason="insufficient",
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
        )
    if any(isinstance(n,bool) or not isinstance(n,Integral) or n < 0 for n in names_per_date):
        raise ValueError("names_per_date must contain non-negative integers, never bool/float")
    valid = list(names_per_date)
    if not valid:
        return AdaptiveBinsResolution(
            bin_count=None,
            reason="insufficient",
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
        )
    min_names = min(valid)
    for bins in policy.candidate_bin_counts():
        if min_names >= bins * policy.min_effective_names_per_bin:
            return AdaptiveBinsResolution(
                bin_count=bins,
                reason="preferred" if bins == policy.preferred_bins else "fallback",
                min_names_per_bin=min_names / bins,
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
            )
    return AdaptiveBinsResolution(
        bin_count=None,
        reason="insufficient",
        min_names_per_bin=min_names / min(policy.candidate_bin_counts()),
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
    )


def resolve_bin_count_from_counts(
    per_bin_counts: Dict[int, Sequence[int]],
    policy: Optional[AdaptiveBinsPolicy] = None,
) -> AdaptiveBinsResolution:
    """Resolve from observed per-bin effective-name counts.

    This variant accepts a mapping ``bin_count -> per-date per-bin counts``
    (e.g. produced by a quantile-assignment kernel over the whole panel) and
    checks that the *minimum* per-bin count over all dates stays at or above
    ``min_effective_names_per_bin``.  If a candidate's per-bin counts are not
    provided, it is skipped (never assumed feasible).

    Args:
        per_bin_counts: ``{bin_count: [per-date min effective names per bin]}``
        policy: The versioned policy (defaults to the plan triple).

    Returns:
        Frozen :class:`AdaptiveBinsResolution` (``insufficient`` when no
        feasible count is observed).
    """
    if policy is None:
        policy = AdaptiveBinsPolicy()
    if not isinstance(policy, AdaptiveBinsPolicy):
        raise TypeError("policy must be AdaptiveBinsPolicy")
    for bins, observed in per_bin_counts.items():
        if isinstance(bins,bool) or not isinstance(bins,Integral) or bins < 2:
            raise ValueError("per_bin_counts keys must be integer bin counts >=2")
        if any(isinstance(n,bool) or not isinstance(n,Integral) or n < 0 for n in observed):
            raise ValueError("per_bin_counts must contain non-negative integers")
    for bins in policy.candidate_bin_counts():
        observed = per_bin_counts.get(bins)
        if observed is None or len(observed) == 0:
            continue
        if min(observed) >= policy.min_effective_names_per_bin:
            return AdaptiveBinsResolution(
                bin_count=bins,
                reason="preferred" if bins == policy.preferred_bins else "fallback",
                min_names_per_bin=float(min(observed)),
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
            )
    return AdaptiveBinsResolution(
        bin_count=None,
        reason="insufficient",
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
    )
