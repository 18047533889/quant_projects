"""
Modeling Adapters: model input contracts and adapters only.

The authoritative FactorEngine production package remains top-level ``modeling``.
This standalone distribution intentionally owns only ``modeling_adapters``.
"""

__version__ = "0.1.0"

from modeling_adapters.contracts import (
    PreprocessContract,
    FitWindow,
    SplitSpec,
    OutOfFoldSpec,
    ModelReadyData,
    TransformMode,
)
from modeling_adapters.errors import (
    ModelingError,
    FutureLeakageError,
    FitWindowError,
    SplitError,
    ContractViolation,
    AdapterError,
    InsufficientDataError,
)

__all__ = [
    # Contracts
    "PreprocessContract",
    "FitWindow",
    "SplitSpec",
    "OutOfFoldSpec",
    "ModelReadyData",
    "TransformMode",
    # Errors
    "ModelingError",
    "FutureLeakageError",
    "FitWindowError",
    "SplitError",
    "ContractViolation",
    "AdapterError",
    "InsufficientDataError",
    "package_info",
]


def package_info():
    """Return package metadata."""
    return {
        "name": "modeling-adapters",
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
