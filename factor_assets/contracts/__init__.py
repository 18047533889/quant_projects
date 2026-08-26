"""
Core contracts for factor_assets package.

Typed contracts for FactorAsset, FactorSet, EvidenceRef, and lifecycle states.
All contracts are immutable and follow the envelope pattern from CONTRACT_FREEZE_DRAFT.
"""

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.factor_set import (
    FactorMembership,
    FactorSet,
    FactorSetArtifact,
    FactorSetSpec,
)
from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.contracts.lifecycle import (
    HealthState,
    LifecycleState,
    StateTransition,
    StateEvent,
    ValidationStatus,
)
from factor_assets.contracts.envelope import ContractEnvelope
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.treatment_policy import TreatmentPolicyRef
from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact

__all__ = [
    "FactorAsset",
    "AssetMetadata",
    "FactorMembership",
    "FactorSet",
    "FactorSetArtifact",
    "FactorSetSpec",
    "EvidenceRef",
    "EvidenceBundleRef",
    "HealthState",
    "LifecycleState",
    "StateTransition",
    "StateEvent",
    "ValidationStatus",
    "ContractEnvelope",
    "LineageRef",
    "ParentRef",
    "TreatmentPolicyRef",
    "TreatmentSelectionArtifact",
]
