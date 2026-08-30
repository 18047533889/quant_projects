"""
FactorOptimizer: Evidence-guided factor mutation and search.

Top-level API:
    package_info: Return package metadata and capability information
    capability and error types: Shared package contracts and failure classes

Mutation grammar, validation, registry, and search-budget contracts are exposed
from the ``factor_optimizer.grammar`` and ``factor_optimizer.contracts`` modules.
This package does NOT execute factors, compute metrics, or make admission decisions.
It coordinates FE/QE through adapter protocols only.
"""

__version__ = "0.1.0"

from factor_optimizer.capabilities import (
    CapabilityStatus,
    ExecutionMode,
    PRODUCTION_CAPABILITY,
    ProductionCapability,
    require_production_capability,
)
from factor_optimizer.contracts.validator import (
    TrialValidatorIdentity,
    MutationGrammarValidator,
)
from factor_optimizer.errors import (
    FactorOptimizerError,
    ContractError,
    SchemaVersionError,
    MissingInputError,
    InvalidContractError,
    TimingContractError,
    SnapshotMismatchError,
    CapabilityError,
    UnsupportedMutationError,
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
    IllegalMutationError,
    DuplicateIdentityError,
    CollisionError,
    ContractChangeRequired,
    TreatmentIntegrityError,
)

# R55 P0-9: real treatment-integrity evidence (fail-closed gates).
from factor_optimizer.contracts.treatment_integrity import (
    EVIDENCE_SCHEMA_VERSION,
    RAW_TREATMENT_KIND,
    IntegrityCheckResult,
    TreatmentIntegrityEvidence,
    TreatmentIntegrityStatus,
    build_integrity_evidence,
    describe_integrity_problem,
    digest_value,
    require_integrity_evidence,
)


def package_info():
    """Return package metadata."""
    return {
        "name": "factor-optimizer",
        "version": __version__,
        "production_capability": PRODUCTION_CAPABILITY.as_dict(),
        "capabilities": [
            "mutation_grammar",
            "legality_validation",
            "search_budget_tracking",
            "seen_cache",
        ],
        "adapters": {
            "factor_engine": "optional",
            "quant_evaluator": "optional",
        },
    }


__all__ = [
    "package_info",
    "CapabilityStatus",
    "ExecutionMode",
    "PRODUCTION_CAPABILITY",
    "ProductionCapability",
    "require_production_capability",
    # Production contracts
    "TrialValidatorIdentity",
    "MutationGrammarValidator",
    # Errors
    "FactorOptimizerError",
    "ContractError",
    "SchemaVersionError",
    "MissingInputError",
    "InvalidContractError",
    "TimingContractError",
    "SnapshotMismatchError",
    "CapabilityError",
    "UnsupportedMutationError",
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
    "IllegalMutationError",
    "DuplicateIdentityError",
    "CollisionError",
    "ContractChangeRequired",
    "TreatmentIntegrityError",
    # Treatment integrity evidence (R55 P0-9)
    "EVIDENCE_SCHEMA_VERSION",
    "RAW_TREATMENT_KIND",
    "IntegrityCheckResult",
    "TreatmentIntegrityEvidence",
    "TreatmentIntegrityStatus",
    "build_integrity_evidence",
    "describe_integrity_problem",
    "digest_value",
    "require_integrity_evidence",
]
