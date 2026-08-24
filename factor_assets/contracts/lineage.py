"""
Lineage tracking for factor assets.

Records parent factors, campaigns, mutations, and search trials.
No factor values stored — only identity references.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParentRef:
    """
    Reference to a parent factor in the lineage.

    Captures parent identity and relationship without storing values.
    """
    factor_id: str
    relationship: str  # "mutation", "combination", "derived", "manual"
    mutation_id: Optional[str] = None
    operation: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.relationship:
            raise ValueError("relationship is required")


@dataclass(frozen=True)
class LineageRef:
    """
    Complete lineage reference for a factor asset.

    Tracks provenance, parents, campaign context, and trial information.
    All references are identifiers only — no raw values.
    """
    factor_id: str
    parents: tuple[ParentRef, ...]
    campaign_id: Optional[str] = None
    trial_id: Optional[str] = None
    batch_id: Optional[str] = None
    origin: str = "manual"  # "manual", "llm", "search", "corpus", "migration"
    origin_ref: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")

    @property
    def has_parents(self) -> bool:
        """Check if factor has parent factors."""
        return len(self.parents) > 0

    @property
    def parent_ids(self) -> tuple[str, ...]:
        """Get all parent factor IDs."""
        return tuple(p.factor_id for p in self.parents)

    @property
    def is_from_campaign(self) -> bool:
        """Check if factor originated from a search campaign."""
        return self.campaign_id is not None
