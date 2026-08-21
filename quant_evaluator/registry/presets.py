"""
Named metric preset bundles.

Curated collections of metrics for common evaluation workflows.
"""

from typing import Dict, List, Set
from quant_evaluator.registry.metrics import MetricSpec, get_metric


class MetricPreset:
    """
    Named collection of metrics for a specific evaluation workflow.

    Attributes:
        name: Preset identifier
        display_name: Human-readable name
        description: Workflow description
        metric_names: List of metric names in this preset
    """
    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        metric_names: List[str],
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.metric_names = metric_names

    def get_metrics(self) -> List[MetricSpec]:
        """
        Retrieve all metric specifications in this preset.

        Returns:
            List of MetricSpec objects

        Raises:
            KeyError: If any metric name is not registered
        """
        return [get_metric(name) for name in self.metric_names]

    def __repr__(self) -> str:
        return (
            f"MetricPreset(name={self.name!r}, "
            f"display_name={self.display_name!r}, "
            f"metrics={len(self.metric_names)})"
        )


# Preset catalog
_PRESET_CATALOG: Dict[str, MetricPreset] = {}


def register_preset(preset: MetricPreset) -> None:
    """
    Register a metric preset.

    Args:
        preset: MetricPreset to register

    Raises:
        ValueError: If preset name already registered
    """
    if preset.name in _PRESET_CATALOG:
        raise ValueError(f"Preset '{preset.name}' already registered")
    _PRESET_CATALOG[preset.name] = preset


def get_preset(name: str) -> MetricPreset:
    """
    Retrieve metric preset by name.

    Args:
        name: Preset identifier

    Returns:
        MetricPreset for the requested preset

    Raises:
        KeyError: If preset not found
    """
    if name not in _PRESET_CATALOG:
        raise KeyError(f"Preset '{name}' not found in registry")
    return _PRESET_CATALOG[name]


def list_presets() -> List[str]:
    """
    List all registered preset names.

    Returns:
        Sorted list of preset names
    """
    return sorted(_PRESET_CATALOG.keys())


# Define standard presets
FACTOR_CORE = MetricPreset(
    name="factor_core",
    display_name="Factor Core Metrics",
    description="Essential metrics for daily factor evaluation",
    metric_names=[
        "mean_ic",
        "ic_std",
        "ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
    ],
)

FACTOR_EXTENDED = MetricPreset(
    name="factor_extended",
    display_name="Factor Extended Metrics",
    description="Comprehensive factor evaluation including robustness and temporal properties",
    metric_names=[
        # Core metrics
        "mean_ic",
        "ic_std",
        "ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
        # Extended metrics
        "hac_tstat",
        "subsample_stability",
        "ic_autocorr_lag1",
        "rank_stability",
        "half_life",
    ],
)

PRODUCTION_DAILY = MetricPreset(
    name="production_daily",
    display_name="Production Daily Report",
    description="Fast metrics for daily production monitoring",
    metric_names=[
        "mean_ic",
        "ic_ir",
        "coverage",
        "turnover",
    ],
)

# Register all presets
register_preset(FACTOR_CORE)
register_preset(FACTOR_EXTENDED)
register_preset(PRODUCTION_DAILY)
