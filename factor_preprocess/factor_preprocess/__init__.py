"""
Factor Preprocess: Model-input preparation layer.

Public API for transforming selected factors into model-ready features.
"""

__version__ = "0.1.0"

from factor_preprocess.contracts.policy import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
from factor_preprocess.contracts.state import FittedState
from factor_preprocess.contracts.feature_bundle import FeatureBundle
from factor_preprocess.errors import (
    FactorPreprocessError,
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
    "PreprocessingPolicy",
    "TransformSpec",
    "TransformKind",
    "TransformMode",
    "FittedState",
    "FeatureBundle",
    # Errors
    "FactorPreprocessError",
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
