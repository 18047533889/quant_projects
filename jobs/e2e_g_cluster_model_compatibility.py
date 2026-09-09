"""Outer composition for E2E-G cluster-to-model compatibility acceptance."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from quant_platform.app.contracts import FeatureSetVersion
from quant_platform.app.modeling_manifest_bridge import (
    ModelFeatureCompatibilityEvidence,
    ModelManifestCompatibilityDecision,
    build_modeling_feature_manifest,
    select_model_compatible_manifest,
)


def assess_research_feature_set_for_model(
    deployed: FeatureSetVersion,
    research: FeatureSetVersion,
    *,
    deployed_model_artifact_ref: str,
    replacement_model_artifact_ref: str | None = None,
    compatibility_evidence: ModelFeatureCompatibilityEvidence | None = None,
    resolve_model_artifact: Callable[[str], Mapping | None] | None = None,
) -> tuple[ModelManifestCompatibilityDecision, dict, dict]:
    """Build both manifests and make a read-only activation decision."""
    deployed_manifest = build_modeling_feature_manifest(deployed)
    research_manifest = build_modeling_feature_manifest(research)
    if compatibility_evidence is not None:
        if resolve_model_artifact is None:
            raise ValueError("compatibility evidence requires a model artifact resolver")
        resolved = resolve_model_artifact(compatibility_evidence.model_artifact_ref)
        if not isinstance(resolved, Mapping):
            raise ValueError("compatibility evidence model artifact cannot be resolved")
        if (resolved.get("model_artifact_ref") != compatibility_evidence.model_artifact_ref or
                resolved.get("feature_set_version_ref") != compatibility_evidence.feature_set_version_ref):
            raise ValueError("resolved model artifact does not match compatibility evidence")
    decision = select_model_compatible_manifest(
        deployed,
        research,
        deployed_model_artifact_ref=deployed_model_artifact_ref,
        replacement_model_artifact_ref=replacement_model_artifact_ref,
        compatibility_evidence=compatibility_evidence,
    )
    return decision, deployed_manifest, research_manifest
