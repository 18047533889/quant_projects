"""
FactorOptimizer: Evidence-guided factor mutation and search.

Public API:
    validate_mutation: Check if a mutation spec is legal
    mutation_catalog: List available mutation operations
    search_budget: Create budget tracking objects

This package does NOT execute factors, compute metrics, or make admission decisions.
It coordinates FE/QE through adapter protocols only.
"""

__version__ = "0.1.0"

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
)


def package_info():
    """Return package metadata."""
    return {
        "name": "factor-optimizer",
        "version": __version__,
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
]
