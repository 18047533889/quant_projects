"""Bridge immutable platform feature snapshots into modeling manifests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant_platform.app.contracts import FeatureSetVersion


@dataclass(frozen=True)
class ModelManifestCompatibilityDecision:
    """Non-mutating decision for activating a research feature manifest."""

    status: str
    active_feature_set_version_ref: str
    research_feature_set_version_ref: str
    activated: bool
    reason: str
    active_model_artifact_ref: str


@dataclass(frozen=True)
class ModelFeatureCompatibilityEvidence:
    """Explicit compatibility/retrain result bound to one exact manifest."""

    evidence_ref: str
    model_artifact_ref: str
    feature_set_version_ref: str
    outcome: str

    def __post_init__(self) -> None:
        for name in ("evidence_ref", "model_artifact_ref", "feature_set_version_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if self.outcome not in {"COMPATIBLE", "RETRAINED"}:
            raise ValueError("outcome must be COMPATIBLE or RETRAINED")


def select_model_compatible_manifest(
    deployed: FeatureSetVersion,
    research: FeatureSetVersion,
    *,
    deployed_model_artifact_ref: str,
    replacement_model_artifact_ref: str | None = None,
    compatibility_evidence: ModelFeatureCompatibilityEvidence | None = None,
) -> ModelManifestCompatibilityDecision:
    """Keep deployed inputs until evidence binds the exact research manifest.

    The function only returns a decision; it never writes a production pointer.
    ``compatibility_evidence`` records an explicit model compatibility
    validation or completed retrain and binds its model artifact to the exact
    candidate feature-set identity.
    """
    if not isinstance(deployed, FeatureSetVersion) or not isinstance(research, FeatureSetVersion):
        raise TypeError("deployed and research must be FeatureSetVersion instances")
    if not isinstance(deployed_model_artifact_ref, str) or not deployed_model_artifact_ref.strip():
        raise ValueError("deployed_model_artifact_ref is required")
    for label, version in (("deployed", deployed), ("research", research)):
        expected_hash = FeatureSetVersion(
            feature_set_id=version.feature_set_id, version=version.version,
            ordered_members=version.ordered_members,
        ).schema_hash
        if version.schema_hash != expected_hash:
            raise ValueError(f"{label} FeatureSetVersion schema_hash is not canonical")
    deployed_manifest = build_modeling_feature_manifest(deployed)
    research_manifest = build_modeling_feature_manifest(research)
    for label, manifest in (("deployed", deployed_manifest), ("research", research_manifest)):
        if manifest.get("model_ready") is not True:
            raise ValueError(f"{label} modeling manifest must be model-ready")
        if not manifest.get("feature_set_version_ref"):
            raise ValueError(f"{label} modeling manifest lacks exact feature-set identity")

    deployed_ref = str(deployed_manifest["feature_set_version_ref"])
    research_ref = str(research_manifest["feature_set_version_ref"])
    if deployed_ref == research_ref:
        if deployed_manifest != research_manifest:
            raise ValueError("feature-set identity collision: same ref has different manifest content")
        return ModelManifestCompatibilityDecision(
            "UNCHANGED", deployed_ref, research_ref, True,
            "research and deployed manifests have the same immutable identity",
            deployed_model_artifact_ref,
        )
    if compatibility_evidence is not None and not isinstance(
        compatibility_evidence, ModelFeatureCompatibilityEvidence
    ):
        raise TypeError("compatibility_evidence must be ModelFeatureCompatibilityEvidence")
    if compatibility_evidence is None or compatibility_evidence.feature_set_version_ref != research_ref:
        return ModelManifestCompatibilityDecision(
            "RETRAIN_REQUIRED", deployed_ref, research_ref, False,
            "changed research manifest is not bound to a compatible/retrained model",
            deployed_model_artifact_ref,
        )
    expected_model_ref = (
        deployed_model_artifact_ref
        if compatibility_evidence.outcome == "COMPATIBLE"
        else replacement_model_artifact_ref
    )
    if (not expected_model_ref or
            compatibility_evidence.model_artifact_ref != expected_model_ref or
            (compatibility_evidence.outcome == "RETRAINED" and
             replacement_model_artifact_ref == deployed_model_artifact_ref)):
        return ModelManifestCompatibilityDecision(
            "EVIDENCE_MODEL_MISMATCH", deployed_ref, research_ref, False,
            "compatibility evidence is bound to a foreign or unintended model artifact",
            deployed_model_artifact_ref,
        )
    return ModelManifestCompatibilityDecision(
        compatibility_evidence.outcome, research_ref, research_ref, True,
        "compatibility/retrain evidence binds the exact research manifest",
        expected_model_ref,
    )


def feature_set_version_ref(version: FeatureSetVersion) -> str:
    """Exact immutable identity; logical feature-set names are insufficient."""
    return f"{version.feature_set_id}@{version.version}#{version.schema_hash}"


def build_modeling_feature_manifest(version: FeatureSetVersion) -> dict[str, Any]:
    """Build a readiness-aware manifest without inventing provenance."""
    if not isinstance(version, FeatureSetVersion):
        raise TypeError("version must be FeatureSetVersion")
    fields = []
    missing = []
    for member in version.ordered_members:
        cluster_version_ref = member.metadata.get("cluster_version_ref")
        state_ref = member.metadata.get("preprocess_state_ref")
        required = {
            "raw_value_ref": member.raw_value_ref,
            "treatment_selection_ref": member.treatment_selection_ref,
            "preprocess_state_ref": state_ref,
            "cluster_version_ref": cluster_version_ref,
        }
        missing.extend(
            f"{member.feature_name}.{name}" for name, value in required.items()
            if not isinstance(value, str) or not value
        )
        fields.append({
            "name": member.feature_name,
            "value_ref": member.treated_feature_ref or member.raw_value_ref,
            "recipe_ref": member.treatment_selection_ref,
            "state_ref": state_ref,
            "dtype": member.dtype or "float64",
            "mask_ref": member.metadata.get("mask_ref"),
            "cluster_version_ref": cluster_version_ref,
        })
    if not fields:
        raise ValueError("feature set version has no members")
    lineage = [{
        "name": member.feature_name,
        "factor_definition_ref": member.factor_definition_ref,
        "evaluation_ref": member.metadata.get("evaluation_ref"),
        "metric_policy_ref": member.metadata.get("metric_policy_ref"),
        "health_policy_ref": member.metadata.get("health_policy_ref"),
        "health_verdict_ref": member.metadata.get("health_verdict_ref"),
        "library_version_ref": member.metadata.get("library_version_ref"),
    } for member in version.ordered_members]
    return {
        "manifest_version": "feature-manifest-v1",
        "consumer_profile": version.consumer_profile or "modeling",
        "feature_set_version_ref": feature_set_version_ref(version),
        "columns": [field["name"] for field in fields],
        "fields": fields,
        "lineage": lineage,
        "complete": not missing,
        "model_ready": not missing,
        "missing_provenance": sorted(missing),
    }
