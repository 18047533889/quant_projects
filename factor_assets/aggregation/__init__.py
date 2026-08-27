"""
Factor aggregation and family representative selection.

Provides specifications for weighting schemes and methods for selecting
representative factors from families based on IC, correlation, or equal weighting.
"""

from factor_assets.aggregation.specs import (
    WeightingScheme,
    AggregationSpec,
    AggregationResult,
    AggregationFitArtifact,
)
from factor_assets.aggregation.representatives import (
    RepresentativeSelectionMethod,
    RepresentativeSelection,
    FamilyRepresentativeSelector,
)
from factor_assets.aggregation.composite import (
    CompositeEvaluator,
    CompactCompositeValue,
    HorizonMapping,
    OrientationMapping,
    RegimeMapping,
    HorizonType,
    OrientationType,
    RegimeType,
)

__all__ = [
    "WeightingScheme",
    "AggregationSpec",
    "AggregationResult",
    "AggregationFitArtifact",
    "RepresentativeSelectionMethod",
    "RepresentativeSelection",
    "FamilyRepresentativeSelector",
    "CompositeEvaluator",
    "CompactCompositeValue",
    "HorizonMapping",
    "OrientationMapping",
    "RegimeMapping",
    "HorizonType",
    "OrientationType",
    "RegimeType",
]
