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
from factor_assets.contracts.similarity import (
    SimilarityArtifact,
    SimilarityView,
    SimilarityViewRegistry,
    EdgeAffinityPolicy,
)
from factor_assets.contracts.cluster_governance import (
    GraphCompletenessClass,
    UnknownEdgeSemantics,
    SimilarityGraphArtifact,
    ClusterSetVersionArtifact,
    LogicalCluster,
    ClusterVersionArtifact,
    ClusterMembership,
    ClusterLineageEdge,
    ClusterVersionMatch,
    ClusterVersionMatcher,
    IncrementalAssignmentKind,
    IncrementalClusterAssignment,
    ClusterResolutionSelector,
    ClusterScale,
)
from factor_assets.contracts.library_governance import (
    FactorLibraryStatus,
    FactorLibraryDefinition,
    FactorLibraryMembership,
    FactorLibraryVersionArtifact,
    NewLibraryProposal,
)
from factor_assets.contracts.assembly_evidence import (
    EvidenceMaturity,
    AssemblyEvidence,
    LibraryCandidateEvidence,
    AssemblyPolicy,
)
from factor_assets.contracts.fingerprint import (
    SimilarityFingerprintArtifact,
    ANNIndexArtifact,
    ANNIndexCapability,
)

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
    # Similarity (DLIB-FA-004/46-50)
    "SimilarityArtifact",
    "SimilarityView",
    "SimilarityViewRegistry",
    "EdgeAffinityPolicy",
    # Cluster governance (DLIB-FA-007/008/009/010/011)
    "GraphCompletenessClass",
    "UnknownEdgeSemantics",
    "SimilarityGraphArtifact",
    "ClusterSetVersionArtifact",
    "LogicalCluster",
    "ClusterVersionArtifact",
    "ClusterMembership",
    "ClusterLineageEdge",
    "ClusterVersionMatch",
    "ClusterVersionMatcher",
    "IncrementalAssignmentKind",
    "IncrementalClusterAssignment",
    "ClusterResolutionSelector",
    "ClusterScale",
    # Library governance (DLIB-FA-012)
    "FactorLibraryStatus",
    "FactorLibraryDefinition",
    "FactorLibraryMembership",
    "FactorLibraryVersionArtifact",
    "NewLibraryProposal",
    # Assembly evidence (DLIB-FA-013)
    "EvidenceMaturity",
    "AssemblyEvidence",
    "LibraryCandidateEvidence",
    "AssemblyPolicy",
    # Fingerprint / ANN (DLIB-FA-006/52)
    "SimilarityFingerprintArtifact",
    "ANNIndexArtifact",
    "ANNIndexCapability",
]
