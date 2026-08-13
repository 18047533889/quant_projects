"""
Core contracts for factor_assets package.

Typed contracts for FactorAsset, FactorSet, EvidenceRef, and lifecycle states.
All contracts are immutable and follow the envelope pattern from CONTRACT_FREEZE_DRAFT.
"""

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.factor_set import FactorSet, FactorSetSpec
from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateTransition,
    StateEvent,
)
from factor_assets.contracts.envelope import ContractEnvelope
from factor_assets.contracts.lineage import LineageRef, ParentRef

__all__ = [
    "FactorAsset",
    "AssetMetadata",
    "FactorSet",
    "FactorSetSpec",
    "EvidenceRef",
    "EvidenceBundleRef",
    "LifecycleState",
    "StateTransition",
    "StateEvent",
    "ContractEnvelope",
    "LineageRef",
    "ParentRef",
]
