"""
Factor Assets package — identity, registry, lifecycle, and governance.

Core contracts for FactorAsset, FactorSet, EvidenceRef, and asset lifecycle.
Append-only repository with immutable identity and lineage.
No raw factor values, no file I/O, no parser duplication, protocol-based FE integration.

Public API authority (DLIB-FA-003): the canonical artifacts are
:class:`FactorSetArtifact`, :class:`FactorMembership`,
:class:`FactorAdmissionArtifact`, :class:`SimilarityArtifact`,
:class:`TreatmentSelectionArtifact`, etc. The legacy :class:`FactorSet` is a
deprecated read-only compatibility view (produced via
``FactorSetArtifact.to_legacy_view()``); new code must not construct it.
"""

__version__ = "0.1.0"

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.factor_set import (
    FactorSet,
    FactorSetSpec,
    FactorSetArtifact,
    FactorMembership,
)
from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.lifecycle import (
    LifecycleState,
    StateTransition,
    StateEvent,
    LifecycleConflictError,
)
from factor_assets.contracts.admission import FactorAdmissionArtifact
from factor_assets.contracts.similarity import SimilarityArtifact
from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact
from factor_assets.registry.repository import (
    AssetRepository,
    DuplicateIdentityError,
    AssetNotFoundError,
    RepositoryStats,
    LifecycleRepository,
    CommittedTransition,
)
from factor_assets.registry.factory import create_platform_lifecycle, create_repository
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository
from factor_assets.registry.lifecycle import (
    LifecycleOrchestrator,
    TransitionRequest,
    TransitionResult,
)
from factor_assets.identity.canonical import (
    FactorIdentityProvider,
    FactorIdentity,
    create_factor_id,
)
from factor_assets.seen_index.exact import (
    SeenRecord,
    SeenIndex,
)
from factor_assets.errors import (
    FactorAssetsError,
    ContractError,
    SchemaVersionError,
    MissingInputError,
    InvalidContractError,
    TimingContractError,
    SnapshotMismatchError,
    CapabilityError,
    UnsupportedTransformError,
    OptionalDependencyMissing,
    DataError,
    InsufficientObservations,
    InvalidValidityMask,
    EvidenceUnavailableError,
    StaleEvidenceError,
    ExecutionError,
    NumericalFailure,
    OverflowOrNonFiniteError,
    BudgetExceededError,
    CancellationError,
    GovernanceError,
    LifecycleConflictError as LifecycleConflictErrorNew,
    DuplicateIdentityError as DuplicateIdentityErrorNew,
    CollisionError,
    ContractChangeRequired,
)

__all__ = [
    # Canonical artifacts (public API authority — DLIB-FA-003)
    "FactorSetArtifact",
    "FactorMembership",
    "FactorAdmissionArtifact",
    "SimilarityArtifact",
    "TreatmentSelectionArtifact",
    # Contracts
    "FactorAsset",
    "AssetMetadata",
    "FactorSet",
    "FactorSetSpec",
    "EvidenceRef",
    "EvidenceBundleRef",
    "LifecycleState",
    "StateTransition",
    "StateEvent",
    "LifecycleConflictError",
    "LineageRef",
    "ParentRef",
    # Registry
    "AssetRepository",
    "DuplicateIdentityError",
    "AssetNotFoundError",
    "LifecycleRepository",
    "CommittedTransition",
    "RepositoryStats",
    "create_repository",
    "create_platform_lifecycle",
    "SQLiteLifecycleRepository",
    "LifecycleOrchestrator",
    "TransitionRequest",
    "TransitionResult",
    # Identity
    "FactorIdentityProvider",
    "FactorIdentity",
    "create_factor_id",
    # Seen index
    "SeenRecord",
    "SeenIndex",
    # Errors
    "FactorAssetsError",
    "ContractError",
    "SchemaVersionError",
    "MissingInputError",
    "InvalidContractError",
    "TimingContractError",
    "SnapshotMismatchError",
    "CapabilityError",
    "UnsupportedTransformError",
    "OptionalDependencyMissing",
    "DataError",
    "InsufficientObservations",
    "InvalidValidityMask",
    "EvidenceUnavailableError",
    "StaleEvidenceError",
    "ExecutionError",
    "NumericalFailure",
    "OverflowOrNonFiniteError",
    "BudgetExceededError",
    "CancellationError",
    "GovernanceError",
    "CollisionError",
    "ContractChangeRequired",
]
