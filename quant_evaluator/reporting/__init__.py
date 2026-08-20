"""
Reporting layer for quant_evaluator.

Provides chart specifications, artifact management, tear sheet generation,
and library-level reporting capabilities.
"""

from quant_evaluator.reporting.chart_spec import ChartSpec
from quant_evaluator.reporting.artifacts import ArtifactStore, ArtifactInfo
from quant_evaluator.reporting.tear_sheet import generate_tear_sheet, EvaluationResult
from quant_evaluator.reporting.library_reports import (
    generate_library_report,
    compare_libraries,
    generate_adversarial_report,
    LibraryReport,
    ComparisonReport,
    AdversarialReport,
)

__all__ = [
    "ChartSpec",
    "ArtifactStore",
    "ArtifactInfo",
    "generate_tear_sheet",
    "EvaluationResult",
    "generate_library_report",
    "compare_libraries",
    "generate_adversarial_report",
    "LibraryReport",
    "ComparisonReport",
    "AdversarialReport",
]
