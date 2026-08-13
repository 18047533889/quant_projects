"""
Registry for metric specifications and presets.

Central catalog of available metrics with metadata and named preset bundles.
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
    get_preset,
    list_presets,
    FACTOR_CORE,
    FACTOR_EXTENDED,
    PRODUCTION_DAILY,
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
]
