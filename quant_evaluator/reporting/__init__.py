"""
Reporting layer for quant_evaluator.

Provides chart specifications, artifact management, tear sheet generation,
and library-level reporting capabilities.
"""

from quant_evaluator.reporting.chart_spec import ChartSpec
from quant_evaluator.reporting.artifacts import ChartArtifactStore, ArtifactInfo

# Tear-sheet metric artifact model, evaluation container, and full
# institutional panel registry
from quant_evaluator.reporting.tear_sheet import (
    MetricArtifact,
    metric_artifact,
    NOT_COMPUTED,
    UNAVAILABLE,
    INSTITUTIONAL_PANELS,
    EvaluationResult,
    generate_tear_sheet,
)

# Backward compat alias
ArtifactStore = ChartArtifactStore

# Lazy imports to avoid circular dependencies
def generate_library_report(*args, **kwargs):
    raise NotImplementedError("library_reports module not available")

def compare_libraries(*args, **kwargs):
    raise NotImplementedError("library_reports module not available")

def generate_adversarial_report(*args, **kwargs):
    raise NotImplementedError("library_reports module not available")


class LibraryReport:
    """Placeholder for LibraryReport."""
    pass


class ComparisonReport:
    """Placeholder for ComparisonReport."""
    pass


class AdversarialReport:
    """Placeholder for AdversarialReport."""
    pass


__all__ = [
    "ChartSpec",
    "ArtifactStore",
    "ChartArtifactStore",
    "ArtifactInfo",
    "MetricArtifact",
    "metric_artifact",
    "NOT_COMPUTED",
    "UNAVAILABLE",
    "INSTITUTIONAL_PANELS",
    "EvaluationResult",
    "generate_tear_sheet",
    "generate_library_report",
    "compare_libraries",
    "generate_adversarial_report",
    "LibraryReport",
    "ComparisonReport",
    "AdversarialReport",
]
