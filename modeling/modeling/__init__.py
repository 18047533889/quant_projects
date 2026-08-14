"""
Modeling: Model input preparation layer.

Provides contracts and adapters for preparing factor data for predictive models,
with strict temporal contracts to prevent future leakage.

This package defines the interface between factor selection (FactorAssets) and
actual model training. It reuses factor_preprocess implementations where possible.
"""

__version__ = "0.1.0"

from modeling.contracts import (
    PreprocessContract,
    FitWindow,
    SplitSpec,
    OutOfFoldSpec,
    ModelReadyData,
)
from modeling.errors import (
    ModelingError,
    FutureLeakageError,
    FitWindowError,
    SplitError,
    ContractViolation,
)

__all__ = [
    # Contracts
    "PreprocessContract",
    "FitWindow",
    "SplitSpec",
    "OutOfFoldSpec",
    "ModelReadyData",
    # Errors
    "ModelingError",
    "FutureLeakageError",
    "FitWindowError",
    "SplitError",
    "ContractViolation",
]


def package_info():
    """Return package metadata."""
    return {
        "name": "modeling",
        "version": __version__,
        "scope": "Model input preparation - NOT model training",
        "capabilities": {
            "temporal_contracts": True,
            "split_validation": True,
            "oof_contracts": True,
            "leakage_detection": True,
            "preprocess_adapters": True,
        },
        "production_ready": False,
        "notes": "Minimal production skeleton - adapters to factor_preprocess",
    }
