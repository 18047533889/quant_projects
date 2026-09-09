from factor_assets.clustering.incremental import (
    CertifiedPairwiseEvidence,
    CertifiedWindowEvidence,
    IncrementalPolicy,
    PairwiseEvidenceStatus,
    build_incremental_cluster_version,
    build_incremental_lineage_edges,
    incremental_assign,
)
from factor_assets.contracts.cluster_governance import (
    ClusterScale,
    ClusterVersionArtifact,
    IncrementalAssignmentKind,
)
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact
from jobs.e2e_g_cluster_model_compatibility import assess_research_feature_set_for_model
from quant_platform.app.contracts import FeatureMemberRef, FeatureSetVersion
from quant_platform.app.modeling_manifest_bridge import (
    ModelFeatureCompatibilityEvidence,
    feature_set_version_ref,
)
import pytest


def _fp(fid, embedding):
    return SimilarityFingerprintArtifact(
        factor_id=fid, embedding=embedding, embedding_spec="e2e-g-spec",
        snapshot="2026-09-07", universe="ASHARE", window="250d",
        preprocessing_ref="preprocess:e2e-g", mask_policy="usable-and-finite",
        direction="signed", aggregation_method="daily-cs-then-time",
        embedding_model_version="e2e-g-v1", value_ref=f"value:{fid}",
        profile_ref=f"profile:{fid}",
    )


def _cluster(cid, members):
    return ClusterVersionArtifact(
        logical_cluster_id=cid, cluster_set_version_ref="clusters:v1",
        member_factor_ids=members, representative_factor_id=members[0],
        scale=ClusterScale.MICRO_CLUSTER,
    )


def _pair(new_id, member_id, affinity):
    window = CertifiedWindowEvidence(
        "250d", affinity, PairwiseEvidenceStatus.CERTIFIED, 100, 100,
        (affinity - .02, min(1.0, affinity + .01)),
        "raw_correlation_hac_daily_corr",
    )
    return CertifiedPairwiseEvidence(
        new_id, member_id, affinity, PairwiseEvidenceStatus.CERTIFIED,
        100, "250d", "ASHARE", "qe-sample:e2e-g", (window,),
    )


def _feature_set(version, cluster_ref):
    return FeatureSetVersion(
        feature_set_id="production-alpha", version=version,
        consumer_profile="production",
        ordered_members=(FeatureMemberRef(
            position=0, feature_name="alpha_fast",
            factor_definition_ref="factor-definition:fast",
            raw_value_ref="value:fast", treatment_selection_ref="recipe:fast",
            treated_feature_ref="treated:fast", dtype="float64",
            source_artifact_id="artifact:fast",
            metadata={"preprocess_state_ref": "state:fast", "cluster_version_ref": cluster_ref},
        ),),
    )


def test_e2e_g_research_cluster_change_cannot_silently_change_deployed_model_columns():
    members = {
        "FAST_1": _fp("FAST_1", (1.0, 0.0, 0.0)),
        "FAST_2": _fp("FAST_2", (0.99, 0.01, 0.0)),
        "SLOW_1": _fp("SLOW_1", (0.0, 1.0, 0.0)),
        "SLOW_2": _fp("SLOW_2", (0.01, 0.99, 0.0)),
    }
    production_clusters = {
        "FAST": _cluster("FAST", ("FAST_1", "FAST_2")),
        "SLOW": _cluster("SLOW", ("SLOW_1", "SLOW_2")),
    }
    production_snapshot = tuple(
        (key, value.content_hash, value.member_factor_ids)
        for key, value in sorted(production_clusters.items())
    )

    ambiguous = _fp("AMBIGUOUS", (0.7, 0.7, 0.0))
    ambiguous_evidence = {
        ("AMBIGUOUS", "FAST_1"): _pair("AMBIGUOUS", "FAST_1", .82),
        ("AMBIGUOUS", "FAST_2"): _pair("AMBIGUOUS", "FAST_2", .80),
        ("AMBIGUOUS", "SLOW_1"): _pair("AMBIGUOUS", "SLOW_1", .81),
        ("AMBIGUOUS", "SLOW_2"): _pair("AMBIGUOUS", "SLOW_2", .80),
    }
    ambiguous_result = incremental_assign(
        [ambiguous], production_clusters, {**members, ambiguous.factor_id: ambiguous},
        IncrementalPolicy(affinity_threshold=.75, ambiguity_gap=.02,
                          cluster_support_k=2, min_cluster_support=2,
                          require_medoid_support=True),
        batch_id="request:e2e-g:ambiguous",
        certified_pairwise=ambiguous_evidence,
    )
    assert ambiguous_result.assignments[0].kind is IncrementalAssignmentKind.AMBIGUOUS
    assert {candidate.support_count for candidate in ambiguous_result.candidates} == {2}

    new_fast = _fp("FAST_3", (0.995, 0.005, 0.0))
    assigned_evidence = {
        ("FAST_3", "FAST_1"): _pair("FAST_3", "FAST_1", .96),
        ("FAST_3", "FAST_2"): _pair("FAST_3", "FAST_2", .94),
    }
    assigned = incremental_assign(
        [new_fast], production_clusters, {**members, new_fast.factor_id: new_fast},
        IncrementalPolicy(affinity_threshold=.8, cluster_support_k=2,
                          min_cluster_support=2, require_medoid_support=True),
        batch_id="request:e2e-g:assigned", certified_pairwise=assigned_evidence,
    )
    assert assigned.assignments[0].kind is IncrementalAssignmentKind.ASSIGNED
    assert assigned.candidates[0].support_member_ids == ("FAST_1", "FAST_2")
    overlays = build_incremental_cluster_version(
        production_clusters, assigned.assignments, new_cluster_set_version_id="clusters:v2",
    )
    lineage = build_incremental_lineage_edges(
        production_clusters, overlays, assigned.assignments,
        requested_by="parent-trial:e2e-g",
    )
    assert production_snapshot == tuple(
        (key, value.content_hash, value.member_factor_ids)
        for key, value in sorted(production_clusters.items())
    )
    assert lineage[0].parent_cluster_version_ref != lineage[0].new_cluster_version_ref

    deployed = _feature_set("v1", lineage[0].parent_cluster_version_ref)
    research = _feature_set("v2", lineage[0].new_cluster_version_ref)
    blocked, deployed_manifest, research_manifest = assess_research_feature_set_for_model(
        deployed, research, deployed_model_artifact_ref="model:deployed"
    )
    assert blocked.status == "RETRAIN_REQUIRED"
    assert blocked.activated is False
    assert blocked.active_feature_set_version_ref == feature_set_version_ref(deployed)
    assert deployed_manifest["columns"] == ["alpha_fast"]
    assert deployed_manifest["fields"][0]["cluster_version_ref"] == lineage[0].parent_cluster_version_ref
    assert research_manifest["fields"][0]["cluster_version_ref"] == lineage[0].new_cluster_version_ref

    stale, _, _ = assess_research_feature_set_for_model(
        deployed, research, deployed_model_artifact_ref="model:deployed",
        compatibility_evidence=ModelFeatureCompatibilityEvidence(
            "compatibility:stale", "model:old", feature_set_version_ref(deployed),
            "COMPATIBLE",
        ), resolve_model_artifact=lambda ref: {
            "model_artifact_ref": ref,
            "feature_set_version_ref": feature_set_version_ref(deployed),
        },
    )
    assert stale.activated is False
    accepted, _, _ = assess_research_feature_set_for_model(
        deployed, research, deployed_model_artifact_ref="model:deployed",
        replacement_model_artifact_ref="model:replacement",
        compatibility_evidence=ModelFeatureCompatibilityEvidence(
            "retrain:e2e-g", "model:replacement", feature_set_version_ref(research),
            "RETRAINED",
        ),
        resolve_model_artifact=lambda ref: {
            "model_artifact_ref": ref,
            "feature_set_version_ref": feature_set_version_ref(research),
        },
    )
    assert accepted.status == "RETRAINED"
    assert accepted.activated is True
    assert accepted.active_feature_set_version_ref == feature_set_version_ref(research)

    foreign = ModelFeatureCompatibilityEvidence(
        "retrain:foreign", "model:foreign", feature_set_version_ref(research), "RETRAINED"
    )
    foreign_decision, _, _ = assess_research_feature_set_for_model(
        deployed, research, deployed_model_artifact_ref="model:deployed",
        replacement_model_artifact_ref="model:replacement",
        compatibility_evidence=foreign,
        resolve_model_artifact=lambda ref: {
            "model_artifact_ref": ref,
            "feature_set_version_ref": feature_set_version_ref(research),
        },
    )
    assert foreign_decision.status == "EVIDENCE_MODEL_MISMATCH"
    assert foreign_decision.activated is False

    with pytest.raises(ValueError, match="resolved model artifact does not match"):
        assess_research_feature_set_for_model(
            deployed, research, deployed_model_artifact_ref="model:deployed",
            replacement_model_artifact_ref="model:replacement",
            compatibility_evidence=ModelFeatureCompatibilityEvidence(
                "retrain:spoof", "model:replacement", feature_set_version_ref(research),
                "RETRAINED",
            ),
            resolve_model_artifact=lambda ref: {
                "model_artifact_ref": ref,
                "feature_set_version_ref": feature_set_version_ref(deployed),
            },
        )
