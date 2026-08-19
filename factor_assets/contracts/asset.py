"""
FactorAsset core contract.

Identity, definition reference, lineage, evidence references, lifecycle state.
NO raw factor values stored — only references and metadata.
"""

from dataclasses import dataclass
from typing import Optional

from factor_assets.contracts.lifecycle import (
    HealthState,
    LifecycleState,
    ValidationStatus,
)
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.errors import MissingInputError


@dataclass(frozen=True)
class AssetMetadata:
    """
    Core metadata for a factor asset.

    Provenance and descriptive information without computation logic.
    """
    factor_id: str
    canonical_repr: str
    canonical_hash: str
    frequency: str
    domains: tuple[str, ...]
    timing: str  # "daily", "intraday", "weekly", etc.
    source_definition_ref: Optional[str] = None
    fe_identity_ref: Optional[str] = None
    fe_compiler_generation: Optional[str] = None
    lookback_days: Optional[int] = None
    required_fields: tuple[str, ...] = ()
    complexity_score: Optional[float] = None
    description: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise MissingInputError("factor_id is required")
        if not self.canonical_hash:
            raise MissingInputError("canonical_hash is required")
        if not self.frequency:
            raise MissingInputError("frequency is required")


@dataclass(frozen=True)
class FactorAsset:
    """
    Complete factor asset record.

    Contains:
    - Identity and metadata
    - Lineage and provenance
    - Evidence references (not raw evidence)
    - Lifecycle state and events
    - Compact derived fingerprints (bounded)

    Does NOT contain:
    - Raw factor values (those live in DA)
    - Full metric results (those live in QE)
    - Materialization paths or state
    """
    metadata: AssetMetadata
    lineage: LineageRef
    lifecycle_state: LifecycleState
    registered_at: str  # ISO 8601 timestamp
    # Orthogonal lifecycle dimensions (P0 #12): evidence freshness and
    # usability are tracked independently of the pipeline stage.
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    health_state: HealthState = HealthState.ACTIVE
    first_evaluated_at: Optional[str] = None
    approved_at: Optional[str] = None
    production_ready_at: Optional[str] = None
    latest_evidence_ref: Optional[EvidenceBundleRef] = None
    # Compact derived fingerprints (bounded, not full values)
    structural_fingerprint: Optional[str] = None
    semantic_fingerprint: Optional[str] = None
    # Tags and classification
    tags: tuple[str, ...] = ()
    family: Optional[str] = None

    def __post_init__(self):
        if not self.registered_at:
            raise MissingInputError("registered_at is required")

    @property
    def factor_id(self) -> str:
        """Get factor ID from metadata."""
        return self.metadata.factor_id

    @property
    def is_registered(self) -> bool:
        """Check if asset is in REGISTERED state."""
        return self.lifecycle_state == LifecycleState.REGISTERED

    @property
    def is_evaluated(self) -> bool:
        """Check if asset has been evaluated."""
        return self.lifecycle_state in (
            LifecycleState.EVALUATED,
            LifecycleState.APPROVED,
            LifecycleState.PRODUCTION_READY,
        )

    @property
    def is_approved(self) -> bool:
        """Check if asset is approved for use."""
        return self.lifecycle_state in (
            LifecycleState.APPROVED,
            LifecycleState.PRODUCTION_READY,
        )

    @property
    def is_production_ready(self) -> bool:
        """Check if asset is production certified."""
        return self.lifecycle_state == LifecycleState.PRODUCTION_READY

    @property
    def is_usable(self) -> bool:
        """Check health dimension: usable for new assemblies."""
        return self.health_state == HealthState.ACTIVE

    @property
    def is_validation_current(self) -> bool:
        """Check validation dimension: evidence is complete and fresh."""
        return self.validation_status == ValidationStatus.VALIDATED

    @property
    def has_evidence(self) -> bool:
        """Check if asset has evaluation evidence."""
        return self.latest_evidence_ref is not None

    @property
    def has_lineage(self) -> bool:
        """Check if asset has parent factors."""
        return self.lineage.has_parents
