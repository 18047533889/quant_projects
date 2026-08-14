"""
Factor Assets Assembly: Factor set construction and aggregation coordination.

Assembles optimized factor sets for downstream consumption.
"""

from factor_assets.assembly.set_builder import (
    FactorSetBuilder,
    AssemblySpec,
    AssemblyResult,
)
from factor_assets.assembly.selection_policy import (
    SelectionPolicy,
    SelectionCriteria,
    SelectionMode,
)
from factor_assets.assembly.diversification import (
    SimilarityProvider,
    DiversificationConfig,
    DiversificationResult,
    DiversifiedSelector,
)

__all__ = [
    "FactorSetBuilder",
    "AssemblySpec",
    "AssemblyResult",
    "SelectionPolicy",
    "SelectionCriteria",
    "SelectionMode",
    "SimilarityProvider",
    "DiversificationConfig",
    "DiversificationResult",
    "DiversifiedSelector",
]
