"""
R32-P0-089: Dependency extraction production fail-closed.

Manifest/import/parser/DSL dependency resolution must fail in production
and automated research modes when unresolved. No silent fallback to default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class DependencyResolutionError(ValidationError):
    """Raised when dependency extraction fails in strict mode."""

    unresolved_leaves: tuple[str, ...]
    context: str

    def __str__(self) -> str:
        leaves = ", ".join(self.unresolved_leaves)
        return (
            f"Dependency extraction failed in {self.context}: "
            f"unresolved leaves [{leaves}]. "
            "Production and automated research modes require explicit resolution."
        )


@dataclass(frozen=True)
class DatasetDependency:
    """Resolved dataset dependency with explicit source."""

    dataset: str
    source: str  # "explicit", "inferred", "default"
    fields: tuple[str, ...]
    confidence: str = "high"  # "high", "medium", "low"


class DependencyExtractor:
    """Extract dataset dependencies with fail-closed semantics.

    R32-P0-089: Production/automated research must not silently fall back
    to default dataset when leaves are unresolved.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    def extract_from_manifest(
        self, manifest: Any, declared_datasets: Sequence[str]
    ) -> list[DatasetDependency]:
        """Extract dependencies from manifest metadata.

        Args:
            manifest: Manifest object with metadata
            declared_datasets: Known registered dataset names

        Returns:
            List of resolved dependencies

        Raises:
            DependencyResolutionError: In strict mode when unresolved leaves exist
        """
        dependencies: list[DatasetDependency] = []
        unresolved: list[str] = []

        # Extract explicit references from manifest metadata
        if hasattr(manifest, "metadata") and manifest.metadata:
            meta = manifest.metadata
            if "dependencies" in meta:
                for dep_name in meta["dependencies"]:
                    if dep_name in declared_datasets:
                        dependencies.append(
                            DatasetDependency(
                                dataset=dep_name,
                                source="explicit",
                                fields=(),
                                confidence="high",
                            )
                        )
                    else:
                        unresolved.append(dep_name)

        if unresolved and self.strict:
            raise DependencyResolutionError(
                unresolved_leaves=tuple(unresolved),
                context="manifest dependency extraction",
            )

        return dependencies

    def extract_from_dsl(
        self, dsl_text: str, declared_datasets: Sequence[str]
    ) -> list[DatasetDependency]:
        """Extract dependencies from DSL/formula text.

        Args:
            dsl_text: Formula or DSL text
            declared_datasets: Known registered dataset names

        Returns:
            List of resolved dependencies

        Raises:
            DependencyResolutionError: In strict mode when unresolved references exist
        """
        dependencies: list[DatasetDependency] = []
        unresolved: list[str] = []

        # Simple pattern: look for dataset references in DSL
        # This is a placeholder - real implementation would use proper parser
        import re

        pattern = r'\b(?:from|dataset|source)\s*[=:]\s*["\']?(\w+)["\']?'
        matches = re.findall(pattern, dsl_text, re.IGNORECASE)

        for ref in matches:
            if ref in declared_datasets:
                dependencies.append(
                    DatasetDependency(
                        dataset=ref,
                        source="inferred",
                        fields=(),
                        confidence="medium",
                    )
                )
            else:
                unresolved.append(ref)

        if unresolved and self.strict:
            raise DependencyResolutionError(
                unresolved_leaves=tuple(unresolved),
                context="DSL dependency extraction",
            )

        return dependencies

    def extract_from_imports(
        self, import_spec: Any, declared_datasets: Sequence[str]
    ) -> list[DatasetDependency]:
        """Extract dependencies from import specifications.

        Args:
            import_spec: Import specification dict/object
            declared_datasets: Known registered dataset names

        Returns:
            List of resolved dependencies

        Raises:
            DependencyResolutionError: In strict mode when unresolved imports exist
        """
        dependencies: list[DatasetDependency] = []
        unresolved: list[str] = []

        if isinstance(import_spec, dict):
            for key in ("sources", "imports", "datasets"):
                if key in import_spec:
                    sources = import_spec[key]
                    if isinstance(sources, (list, tuple)):
                        for src in sources:
                            ds_name = str(src) if isinstance(src, str) else src.get("dataset")
                            if ds_name:
                                if ds_name in declared_datasets:
                                    dependencies.append(
                                        DatasetDependency(
                                            dataset=ds_name,
                                            source="explicit",
                                            fields=(),
                                            confidence="high",
                                        )
                                    )
                                else:
                                    unresolved.append(ds_name)

        if unresolved and self.strict:
            raise DependencyResolutionError(
                unresolved_leaves=tuple(unresolved),
                context="import specification",
            )

        return dependencies


def is_production_or_automated() -> bool:
    """Check if current mode requires strict dependency resolution."""
    import os

    # Check production mode
    prod_mode = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if prod_mode:
        return True

    # Check FactorEngine run mode (automated research)
    fe_mode = os.environ.get("FACTOR_ENGINE_RUN_MODE", "")
    if fe_mode in {"automated_research", "production"}:
        return True

    return False


def create_dependency_extractor() -> DependencyExtractor:
    """Create extractor with appropriate strictness for current mode."""
    return DependencyExtractor(strict=is_production_or_automated())


__all__ = [
    "DependencyExtractor",
    "DependencyResolutionError",
    "DatasetDependency",
    "create_dependency_extractor",
    "is_production_or_automated",
]
