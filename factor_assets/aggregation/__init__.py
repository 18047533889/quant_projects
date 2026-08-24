"""
Factor aggregation and family representative selection.

Provides specifications for weighting schemes and methods for selecting
representative factors from families based on IC, correlation, or equal weighting.
"""

from factor_assets.aggregation.specs import (
    WeightingScheme,
    AggregationSpec,
    AggregationResult,
)
from factor_assets.aggregation.representatives import (
    RepresentativeSelectionMethod,
    RepresentativeSelection,
    FamilyRepresentativeSelector,
)

__all__ = [
    "WeightingScheme",
    "AggregationSpec",
    "AggregationResult",
    "RepresentativeSelectionMethod",
    "RepresentativeSelection",
    "FamilyRepresentativeSelector",
]
