"""Public API for target-position analysis."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from ..core.contracts import OutputBundle
from .artifact_loader import artifacts_from_output_bundle, load_artifact_directory
from .config import AnalysisConfigError, load_analysis_config
from .contracts import (
    AnalysisEnrichment,
    OptimizationArtifacts,
    PositionAnalysisResult,
)
from .position_analyzer import PositionAnalyzer
from .report import write_analysis_outputs


def analyze_position_output(
    *,
    config: str | Path | Mapping[str, Any],
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
    data_store: Any | None = None,
) -> PositionAnalysisResult:
    """Analyze an optimizer artifact directory and optionally persist results."""

    resolved, config_path = load_analysis_config(config)
    started = time.perf_counter()
    if input_dir is not None:
        root = Path(input_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        resolved["input"]["optimization_dir"] = str(root.resolve())
    source = resolved["input"]["optimization_dir"]
    if not source:
        raise AnalysisConfigError(
            "input.optimization_dir or --input-dir is required"
        )
    if output_dir is not None:
        root = Path(output_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        resolved["output"]["directory"] = str(root.resolve())
    if overwrite is not None:
        resolved["output"]["overwrite"] = bool(overwrite)

    validation = resolved["validation"]
    artifacts = load_artifact_directory(
        source,
        reconstruction_tolerance=float(
            validation["reconstruction_tolerance"]
        ),
    )
    analyzer = PositionAnalyzer(resolved, data_store=data_store)
    result = analyzer.analyze(artifacts)
    result.metadata["elapsed_ms"] = (
        time.perf_counter() - started
    ) * 1000.0
    destination = resolved["output"]["directory"]
    if destination is None:
        destination = str((Path(source) / "analysis").resolve())
        resolved["output"]["directory"] = destination
    write_analysis_outputs(
        result,
        destination,
        config=resolved,
        config_path=config_path,
        overwrite=bool(resolved["output"]["overwrite"]),
    )
    return result


def analyze_output_bundle(
    bundle: OutputBundle,
    *,
    config: str | Path | Mapping[str, Any],
    enrichment: AnalysisEnrichment | None = None,
    data_store: Any | None = None,
) -> PositionAnalysisResult:
    """Analyze an in-memory OutputBundle without writing files."""

    resolved, _ = load_analysis_config(config)
    artifacts = artifacts_from_output_bundle(
        bundle,
        reconstruction_tolerance=float(
            resolved["validation"]["reconstruction_tolerance"]
        ),
    )
    return PositionAnalyzer(resolved, data_store=data_store).analyze(
        artifacts, enrichment=enrichment
    )


__all__ = [
    "AnalysisConfigError",
    "AnalysisEnrichment",
    "OptimizationArtifacts",
    "PositionAnalysisResult",
    "PositionAnalyzer",
    "analyze_output_bundle",
    "analyze_position_output",
]
