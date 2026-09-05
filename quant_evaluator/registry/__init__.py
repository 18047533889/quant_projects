"""
Registry for metric specifications, presets, and named EvidenceProfiles.

Central catalog of available metrics with metadata and named preset bundles,
plus the versioned EvidenceProfile registry (R61-FI-020) that binds named
metric sets for CN-daily evaluation workflows.
"""

from quant_evaluator.registry.metrics import (
    MetricSpec,
    MetricStatus,
    MetricTier,
    get_metric,
    list_metrics,
    list_metrics_by_status,
    list_metrics_by_tier,
)
from quant_evaluator.registry.presets import (
    CostClass,
    EvidenceProfile,
    GPUPreference,
    TargetFrequency,
    UnknownEvidenceProfileError,
    get_preset,
    get_profile,
    get_profile_or_none,
    list_presets,
    list_profile_versions,
    list_profiles,
    register_profile,
    FACTOR_CORE,
    FACTOR_EXTENDED,
    PRODUCTION_DAILY,
    CHEAP_SCREEN_CN_1D,
    SHAPE_DIAGNOSTIC_CN_1D,
    FULL_VALIDATION_CN_1D,
    EXPENSIVE_STATISTICAL_CN_1D,
    MODEL_FEATURE_DIAGNOSTIC_CN_1D,
)

__all__ = [
    "MetricSpec",
    "MetricStatus",
    "MetricTier",
    "get_metric",
    "list_metrics",
    "list_metrics_by_status",
    "list_metrics_by_tier",
    "get_preset",
    "list_presets",
    "FACTOR_CORE",
    "FACTOR_EXTENDED",
    "PRODUCTION_DAILY",
    # R61-FI-020 EvidenceProfile registry surface.
    "EvidenceProfile",
    "CostClass",
    "GPUPreference",
    "TargetFrequency",
    "UnknownEvidenceProfileError",
    "register_profile",
    "get_profile",
    "get_profile_or_none",
    "list_profiles",
    "list_profile_versions",
    "CHEAP_SCREEN_CN_1D",
    "SHAPE_DIAGNOSTIC_CN_1D",
    "FULL_VALIDATION_CN_1D",
    "EXPENSIVE_STATISTICAL_CN_1D",
    "MODEL_FEATURE_DIAGNOSTIC_CN_1D",
]
