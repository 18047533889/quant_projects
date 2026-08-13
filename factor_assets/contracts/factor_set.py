"""
FactorSet and selection specifications.

Groups of factors selected for downstream use (e.g., preprocessing, modeling).
References factors by ID only — no raw values stored.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FactorSetSpec:
    """
    Specification for factor set selection.

    Defines criteria and constraints for assembling a FactorSet.
    """
    set_id: str
    name: str
    selection_policy: str  # "family_robust", "pareto_front", "manual", etc.
    universe_ref: Optional[str] = None
    frequency: Optional[str] = None
    max_factors: Optional[int] = None
    min_evidence_date: Optional[str] = None
    required_domains: tuple[str, ...] = ()
    excluded_domains: tuple[str, ...] = ()
    min_lifecycle_state: Optional[str] = None
    family_constraints: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if not self.set_id:
            raise ValueError("set_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.selection_policy:
            raise ValueError("selection_policy is required")


@dataclass(frozen=True)
class FactorSet:
    """
    A selected set of factor assets.

    Contains factor IDs and references only — no raw factor values.
    Used to pass selected factors to FactorPreprocess or other consumers.
    """
    set_id: str
    name: str
    factor_ids: tuple[str, ...]
    created_at: str  # ISO 8601
    spec: Optional[FactorSetSpec] = None
    universe_ref: Optional[str] = None
    frequency: Optional[str] = None
    # Selection metadata
    selection_run_id: Optional[str] = None
    parent_set_id: Optional[str] = None
    # Aggregation info (if factors are grouped)
    families: tuple[str, ...] = ()
    representatives: tuple[str, ...] = ()
    description: Optional[str] = None

    def __post_init__(self):
        if not self.set_id:
            raise ValueError("set_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.factor_ids:
            raise ValueError("factor_ids cannot be empty")
        if not self.created_at:
            raise ValueError("created_at is required")

    @property
    def size(self) -> int:
        """Get number of factors in the set."""
        return len(self.factor_ids)

    @property
    def has_families(self) -> bool:
        """Check if factors are organized into families."""
        return len(self.families) > 0

    @property
    def has_representatives(self) -> bool:
        """Check if representative factors are designated."""
        return len(self.representatives) > 0

    def contains(self, factor_id: str) -> bool:
        """Check if factor_id is in this set."""
        return factor_id in self.factor_ids
