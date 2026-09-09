import pytest

from modeling.dataset import FeatureField, FeatureSchema
from quant_platform.app.contracts import FeatureMemberRef, FeatureSetVersion
from quant_platform.app.modeling_manifest_bridge import (
    ModelFeatureCompatibilityEvidence,
    build_modeling_feature_manifest, feature_set_version_ref,
    select_model_compatible_manifest,
)


def version(tag, cluster="cluster:A@v1#abc"):
    return FeatureSetVersion(
        feature_set_id="fs", version=tag, consumer_profile="production",
        ordered_members=(FeatureMemberRef(
            position=0, feature_name="alpha", factor_definition_ref="factor:def:1",
            raw_value_ref="value:1", treatment_selection_ref="recipe:1", dtype="float64",
            source_artifact_id="artifact:1", metadata={
                "cluster_version_ref": cluster, "preprocess_state_ref": "state:1"
            },
        ),),
    )


def test_bridge_populates_exact_modeling_identities():
    source = version("v1")
    payload = build_modeling_feature_manifest(source)
    fields = tuple(FeatureField(**item) for item in payload["fields"])
    schema = FeatureSchema(
        columns=tuple(payload["columns"]), fields=fields,
        consumer_profile=payload["consumer_profile"],
        feature_set_version_ref=payload["feature_set_version_ref"],
    )
    assert schema.is_complete
    assert schema.feature_set_version_ref == feature_set_version_ref(source)
    assert schema.fields[0].cluster_version_ref == "cluster:A@v1#abc"


def test_logical_cluster_without_exact_version_fails_closed():
    payload = build_modeling_feature_manifest(version("v1", cluster=""))
    assert payload["model_ready"] is False
    assert "alpha.cluster_version_ref" in payload["missing_provenance"]


def test_new_immutable_version_changes_exact_model_identity():
    before = build_modeling_feature_manifest(version("v1"))
    after = build_modeling_feature_manifest(version("v2", "cluster:A@v2#def"))
    assert before["feature_set_version_ref"] != after["feature_set_version_ref"]
    assert before["fields"][0]["cluster_version_ref"] != after["fields"][0]["cluster_version_ref"]


def test_model_compatibility_selection_requires_exact_research_manifest_ref():
    deployed = version("v1")
    research = version("v2", "cluster:A@v2#def")
    blocked = select_model_compatible_manifest(
        deployed, research, deployed_model_artifact_ref="model:old"
    )
    assert blocked.activated is False
    assert blocked.active_feature_set_version_ref == feature_set_version_ref(deployed)
    stale = select_model_compatible_manifest(
        deployed, research, deployed_model_artifact_ref="model:old",
        compatibility_evidence=ModelFeatureCompatibilityEvidence(
            "compat:stale", "model:old", feature_set_version_ref(deployed), "COMPATIBLE")
    )
    assert stale.activated is False
    accepted = select_model_compatible_manifest(
        deployed, research, deployed_model_artifact_ref="model:old",
        replacement_model_artifact_ref="model:new",
        compatibility_evidence=ModelFeatureCompatibilityEvidence(
            "retrain:new", "model:new", feature_set_version_ref(research), "RETRAINED")
    )
    assert accepted.activated is True
    assert accepted.active_feature_set_version_ref == feature_set_version_ref(research)
    assert accepted.active_model_artifact_ref == "model:new"
    compatible = select_model_compatible_manifest(
        deployed, research, deployed_model_artifact_ref="model:old",
        compatibility_evidence=ModelFeatureCompatibilityEvidence(
            "compat:same-model", "model:old", feature_set_version_ref(research), "COMPATIBLE")
    )
    assert compatible.status == "COMPATIBLE"
    assert compatible.active_model_artifact_ref == "model:old"


def test_model_compatibility_selection_rejects_incomplete_manifest():
    deployed = version("v1")
    incomplete = version("v2", "")
    with pytest.raises(ValueError, match="research modeling manifest must be model-ready"):
        select_model_compatible_manifest(
            deployed, incomplete, deployed_model_artifact_ref="model:old"
        )


def test_same_ref_with_spoofed_schema_hash_or_different_fields_is_rejected():
    deployed = version("v1")
    changed = version("v1", "cluster:A@v2#different")
    spoofed = FeatureSetVersion(
        feature_set_id=changed.feature_set_id, version=changed.version,
        ordered_members=changed.ordered_members, consumer_profile=changed.consumer_profile,
        schema_hash=deployed.schema_hash,
    )
    with pytest.raises(ValueError, match="schema_hash is not canonical"):
        select_model_compatible_manifest(
            deployed, spoofed, deployed_model_artifact_ref="model:old"
        )


def test_same_width_cluster_change_and_foreign_model_evidence_stay_blocked():
    deployed = version("v1")
    research = version("v2", "cluster:A@v2#def")
    foreign = select_model_compatible_manifest(
        deployed, research, deployed_model_artifact_ref="model:old",
        replacement_model_artifact_ref="model:intended",
        compatibility_evidence=ModelFeatureCompatibilityEvidence(
            "retrain:foreign", "model:foreign", feature_set_version_ref(research), "RETRAINED"),
    )
    assert foreign.status == "EVIDENCE_MODEL_MISMATCH"
    assert foreign.activated is False
    assert build_modeling_feature_manifest(deployed)["columns"] == build_modeling_feature_manifest(research)["columns"]


def test_compatibility_evidence_type_is_strict():
    with pytest.raises(TypeError, match="ModelFeatureCompatibilityEvidence"):
        select_model_compatible_manifest(
            version("v1"), version("v2", "cluster:A@v2#def"),
            deployed_model_artifact_ref="model:old",
            compatibility_evidence={"outcome": "COMPATIBLE"},
        )
