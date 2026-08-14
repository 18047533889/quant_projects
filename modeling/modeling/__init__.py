"""
Modeling: Model input preparation layer (ADAPTERS ONLY).

**SCOPE DECLARATION (MODEL2-P0-006)**:
This package provides MINIMAL CONTRACTS for adapting factor_preprocess to
factor_engine. It does NOT implement model training, walk-forward, or enforcement.

For actual model training with temporal leakage prevention, use
``factor_engine.modeling`` (the authoritative package at
``/home/shw/quant_projects/factor_engine/modeling/``).

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
