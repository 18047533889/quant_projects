"""
Factor Preprocess: Model-input preparation layer.

Public API for transforming selected factors into model-ready features.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# DLIB-FP-001: canonical top-level public API.
#
# The primary public API is the registry-based, deeply-immutable contract
# objects. ``PreprocessingPolicy`` / ``TransformSpec`` are retained ONLY as a
# deprecated compatibility view (their real authority is
# ``registry.policies.PolicyPreset`` / ``TransformStep``).
# ---------------------------------------------------------------------------

# Canonical policy + transform registries and their deeply-immutable contracts.
from factor_preprocess.registry.policies import (
    PolicyPreset,
    PolicyRegistry,
    PolicyLevel,
    TransformStep,
    create_default_policies,
    get_default_policy_registry,
)
from factor_preprocess.registry.transforms import (
    TransformRegistry,
    TransformMetadata,
    TransformCategory,
    create_default_registry,
    get_default_registry,
)
from factor_preprocess.contracts.treatment_lineage import (
    TransformStage,
    TransformSemanticID,
    TransformLineage,
    ExistingTreatmentSignature,
    ExistingTreatmentStatus,
    build_signature_from_lineage,
    map_fe_dsl_to_semantic,
)
from factor_preprocess.contracts.treatment_recipe import (
    TreatmentRecipe,
    RecipeStep,
    FitBoundary,
    RecipeSchemaVersion,
)
from factor_preprocess.contracts.factor_profile import FactorProfileArtifact
from factor_preprocess.contracts.state import FittedState
from factor_preprocess.contracts.feature_bundle import (
    FeatureBundle,
    AxisRef,
    ChannelRef,
    FeatureManifest,
)
from factor_preprocess.neutralization.diagnostics_artifact import (
    NeutralizationDiagnostics,
    RankDeficientResolution,
)
from factor_preprocess.neutralization.spec import NeutralizationSpec

# Treatment identity split (P0-FP #103): SPEC vs MATERIALIZATION typed
# identities + the A股 PIT/unit materialization contract.
from factor_preprocess.contracts.treatment_spec import (
    PriceBasis,
    ValueUnit,
    MaterializationSplit,
    TreatmentSpecIdentity,
    TreatmentMaterializationIdentity,
    neutralization_spec_identity,
    neutralization_binding,
    ensure_materialization_split_valid,
    materialize_identity,
    spec_to_materialization_identity,
    ASHARE_INDUSTRY_SCHEMA,
    ASHARE_SIZE_DEFINITION,
)

# Deprecated compatibility view — NOT primary. Kept only so existing importers
# keep working. See ``registry.policies`` for the canonical authority.
from factor_preprocess.contracts.policy import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
from factor_preprocess.errors import (
    FactorPreprocessError,
    ContractError,
    SchemaVersionError,
    MissingInputError,
    InvalidContractError,
    TimingContractError,
    SnapshotMismatchError,
    TreatmentMaterializationError,
    CapabilityError,
    UnsupportedTransformError,
    OptionalDependencyMissing,
    DataError,
    InsufficientObservations,
    InvalidValidityMask,
    MissingFittedStateError,
    StaleFittedStateError,
    ExecutionError,
    NumericalFailure,
    OverflowOrNonFiniteError,
    BudgetExceededError,
    CancellationError,
    GovernanceError,
    FullSampleFitError,
    FittedStateMismatchError,
    ContractChangeRequired,
)

__all__ = [
    # Canonical registries
    "PolicyPreset",
    "PolicyRegistry",
    "PolicyLevel",
    "TransformStep",
    "create_default_policies",
    "get_default_policy_registry",
    "TransformRegistry",
    "TransformMetadata",
    "TransformCategory",
    "create_default_registry",
    "get_default_registry",
    # Treatment lineage / recipe / profile contracts
    "TransformStage",
    "TransformSemanticID",
    "TransformLineage",
    "ExistingTreatmentSignature",
    "ExistingTreatmentStatus",
    "build_signature_from_lineage",
    "map_fe_dsl_to_semantic",
    "TreatmentRecipe",
    "RecipeStep",
    "FitBoundary",
    "RecipeSchemaVersion",
    "FactorProfileArtifact",
    "NeutralizationDiagnostics",
    "RankDeficientResolution",
    "NeutralizationSpec",
    # Treatment identity split (P0-FP #103)
    "PriceBasis",
    "ValueUnit",
    "MaterializationSplit",
    "TreatmentSpecIdentity",
    "TreatmentMaterializationIdentity",
    "neutralization_spec_identity",
    "neutralization_binding",
    "ensure_materialization_split_valid",
    "materialize_identity",
    "spec_to_materialization_identity",
    "ASHARE_INDUSTRY_SCHEMA",
    "ASHARE_SIZE_DEFINITION",
    # Core contracts
    "FittedState",
    "PreprocessingPolicy",
    "TransformSpec",
    "TransformKind",
    "TransformMode",
    "FeatureBundle",
    "AxisRef",
    "ChannelRef",
    "FeatureManifest",
    # Errors
    "FactorPreprocessError",
    "ContractError",
    "SchemaVersionError",
    "MissingInputError",
    "InvalidContractError",
    "TimingContractError",
    "SnapshotMismatchError",
    "TreatmentMaterializationError",
    "CapabilityError",
    "UnsupportedTransformError",
    "OptionalDependencyMissing",
    "DataError",
    "InsufficientObservations",
    "InvalidValidityMask",
    "MissingFittedStateError",
    "StaleFittedStateError",
    "ExecutionError",
    "NumericalFailure",
    "OverflowOrNonFiniteError",
    "BudgetExceededError",
    "CancellationError",
    "GovernanceError",
    "FullSampleFitError",
    "FittedStateMismatchError",
    "ContractChangeRequired",
]


def package_info():
    """Return package metadata and capability information."""
    return {
        "name": "factor_preprocess",
        "version": __version__,
        "capabilities": {
            "stateless_transforms": True,
            "fitted_transforms": True,
            "cross_sectional": True,
            "time_series": True,
            "neutralization": True,
            "representation": True,
        },
        "production_ready": False,
        "notes": "Initial bounded implementation; fitted transforms are basic only",
    }
