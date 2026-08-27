"""Versioning contract tests: four-layer model, promotion/rollback, FeatureSetDiff.

Covers:
(a) FactorLibraryVersion immutable + v103 candidate->promote->rollback to v102
    (history preserved, v103 not deleted);
(b) feature-set content hash changes on treatment/orientation change but NOT on
    metadata-only change;
(c) FeatureSetDiff classifies each category and only membership/treatment/
    orientation/schema/label/data-revision trigger ModelRetrainRequired;
(d) cluster algorithm-label vs logical-id separation + lineage continuity across
    split/merge.
"""

from datetime import datetime, timezone

import pytest

import quant_platform.app.contracts as c
from quant_platform.app.contracts.feature_set import (
    classify_feature_set_diff,
    compute_feature_set_content_hash,
)
from quant_platform.app.contracts.cluster import (
    build_lineage_edges,
    classify_cluster_label_drift,
    resolve_incremental_cluster,
)
from quant_platform.app.contracts.cluster_library import (
    ClusterConversionType,
    ClusterLineageEdge,
    ClusterSetVersion,
    ClusterVersion,
    LogicalCluster,
    SimilarityGraphVersion,
)
from quant_platform.app.contracts.factor_library import (
    FactorLibraryVersion,
    LibraryActivePointer,
    LibraryMembership,
    LibraryPromotionStep,
    LibraryStatus,
    promote_library_version,
    rollback_library_version,
)
from quant_platform.app.contracts.feature_set import (
    FeatureMemberRef,
    FeatureSetDiffCategory,
    FeatureSetVersion,
)
from quant_platform.app.contracts.model_version import (
    DataSnapshot,
    LabelDefinition,
    ModelVersion,
    ModelWeight,
    SplitPlan,
)
from quant_platform.app.contracts.versioning import (
    VersionChain,
    VersionChainLink,
    VersionChainStatus,
    build_version_chain,
)


def _member(position, name="f", treatment="T1", orientation="LONG", dtype="float64"):
    return FeatureMemberRef(
        position=position,
        feature_name=name,
        factor_definition_ref="FD1",
        treatment_selection_ref=treatment,
        orientation=orientation,
        dtype=dtype,
        source_artifact_id="ART1",
        availability_semantics="daily",
    )


def _fs(version, members, label="L1", data_rev="DR1"):
    return FeatureSetVersion(
        feature_set_id="FS1",
        version=version,
        ordered_members=members,
        label_definition_ref=label,
        data_revision_ref=data_rev,
    )


# ---- (a) library immutable + promotion/rollback ----
def test_library_version_is_immutable_and_promotes():
    m = LibraryMembership(
        factor_definition_id="FD1",
        selected_treatment_id="T1",
        orientation="LONG",
        cluster_id="CL_PV_MOM_017",
        evidence_ref="E1",
    )
    v102 = FactorLibraryVersion(
        library_version_id="FLV_102",
        logical_library_id="CORE_LOW_REDUNDANCY",
        cluster_set_version_id="CS1",
        members=(m,),
        status="PRODUCTION",
    )
    v103 = FactorLibraryVersion(
        library_version_id="FLV_103",
        logical_library_id="CORE_LOW_REDUNDANCY",
        cluster_set_version_id="CS1",
        members=(m,),
        status="CANDIDATE",
    )
    pointer = LibraryActivePointer(
        logical_library_id="CORE_LOW_REDUNDANCY",
        active_version_id="FLV_102",
        active_status="PRODUCTION",
    )

    # v102 history preserved (immutable object untouched).
    assert v102.status == "PRODUCTION"

    # Promote v103 CANDIDATE -> SHADOW -> APPROVED -> PRODUCTION.
    v103s, ptr1, rec1 = promote_library_version(v103, pointer)
    assert v103s.status == "SHADOW"
    assert rec1.from_status == "CANDIDATE" and rec1.to_status == "SHADOW"

    v103a, ptr2, _ = promote_library_version(v103s, ptr1)
    assert v103a.status == "APPROVED"

    v103p, ptr3, _ = promote_library_version(v103a, ptr2)
    assert v103p.status == "PRODUCTION"
    assert ptr3.active_version_id == "FLV_103"
    assert ptr3.active_status == "PRODUCTION"

    # Cannot promote past PRODUCTION.
    with pytest.raises(ValueError):
        promote_library_version(v103p, ptr3)

    # v102 still exists in history.
    assert v102.library_version_id == "FLV_102"
    assert v102.status == "PRODUCTION"


def test_rollback_flips_pointer_and_never_deletes_v103():
    m = LibraryMembership(factor_definition_id="FD1")
    v102 = FactorLibraryVersion(
        library_version_id="FLV_102",
        logical_library_id="CORE_LOW_REDUNDANCY",
        cluster_set_version_id="CS1",
        members=(m,),
        status="PRODUCTION",
    )
    v103 = FactorLibraryVersion(
        library_version_id="FLV_103",
        logical_library_id="CORE_LOW_REDUNDANCY",
        cluster_set_version_id="CS1",
        members=(m,),
        status="PRODUCTION",
    )
    pointer = LibraryActivePointer(
        logical_library_id="CORE_LOW_REDUNDANCY",
        active_version_id="FLV_103",
        active_status="PRODUCTION",
    )
    new_pointer, record = rollback_library_version(pointer, "FLV_102")
    assert new_pointer.active_version_id == "FLV_102"
    assert record.from_status == "PRODUCTION" and record.to_status == "PRODUCTION"
    # v103 is NOT deleted — it remains in history.
    assert v103.library_version_id == "FLV_103"
    assert v103.status == "PRODUCTION"

    # Rolling back to the already-active version is an error.
    with pytest.raises(ValueError):
        rollback_library_version(new_pointer, "FLV_102")


def test_library_membership_has_no_weight_in_model():
    m = LibraryMembership(factor_definition_id="FD1")
    assert not hasattr(m, "weight_in_model")


# ---- (b) feature-set content hash ----
def test_content_hash_changes_on_treatment_change():
    a = _fs("1.0", (_member(0, treatment="T1"),))
    b = _fs("1.0", (_member(0, treatment="T2"),))
    assert compute_feature_set_content_hash(a.ordered_members) != compute_feature_set_content_hash(
        b.ordered_members
    )


def test_content_hash_changes_on_orientation_change():
    a = _fs("1.0", (_member(0, orientation="LONG"),))
    b = _fs("1.0", (_member(0, orientation="SHORT"),))
    assert compute_feature_set_content_hash(a.ordered_members) != compute_feature_set_content_hash(
        b.ordered_members
    )


def test_content_hash_stable_on_metadata_only():
    a = _fs("1.0", (_member(0),))
    b = _fs("1.0", (_member(0),))
    # consumer_profile is metadata, not part of member identity.
    b2 = FeatureSetVersion(
        feature_set_id="FS1",
        version="1.0",
        ordered_members=(_member(0),),
        consumer_profile="different",
    )
    assert compute_feature_set_content_hash(a.ordered_members) == compute_feature_set_content_hash(
        b2.ordered_members
    )


def test_content_hash_changes_on_order_change():
    a = _fs("1.0", (_member(0, name="f1"), _member(1, name="f2")))
    b = _fs("1.0", (_member(0, name="f2"), _member(1, name="f1")))
    assert compute_feature_set_content_hash(a.ordered_members) != compute_feature_set_content_hash(
        b.ordered_members
    )


# ---- (c) FeatureSetDiff classification ----
def test_diff_metadata_only_no_retrain():
    a = _fs("1.0", (_member(0),))
    b = FeatureSetVersion(
        feature_set_id="FS1",
        version="1.0",  # same version — only consumer_profile differs
        ordered_members=(_member(0),),
        consumer_profile="changed",
        label_definition_ref="L1",
        data_revision_ref="DR1",
    )
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.METADATA_ONLY
    assert not diff.retrain_required
    assert diff.to_event("FS1") is None


def test_diff_evidence_only_no_retrain():
    a = _fs("1.0", (_member(0),))
    b = FeatureSetVersion(
        feature_set_id="FS1",
        version="1.1",
        ordered_members=(_member(0),),
        source_library_versions=("LIB2",),
        label_definition_ref="L1",
        data_revision_ref="DR1",
    )
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.EVIDENCE_ONLY
    assert not diff.retrain_required


def test_diff_membership_change_triggers_retrain():
    a = _fs("1.0", (_member(0),))
    b = _fs("1.0", (_member(0), _member(1, name="f2")))
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE
    assert diff.retrain_required
    ev = diff.to_event("FS1")
    assert ev is not None
    assert ev.diff_category == FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE


def test_diff_treatment_change_triggers_retrain():
    a = _fs("1.0", (_member(0, treatment="T1"),))
    b = _fs("1.0", (_member(0, treatment="T2"),))
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.FEATURE_TRANSFORM_CHANGE
    assert diff.retrain_required


def test_diff_orientation_change_triggers_retrain():
    a = _fs("1.0", (_member(0, orientation="LONG"),))
    b = _fs("1.0", (_member(0, orientation="SHORT"),))
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE
    assert diff.retrain_required


def test_diff_schema_change_triggers_retrain():
    a = _fs("1.0", (_member(0, dtype="float64"),))
    b = _fs("1.0", (_member(0, dtype="float32"),))
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.FEATURE_SCHEMA_CHANGE
    assert diff.retrain_required


def test_diff_label_change_triggers_retrain():
    a = _fs("1.0", (_member(0),), label="L1")
    b = _fs("1.0", (_member(0),), label="L2")
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.LABEL_CHANGE
    assert diff.retrain_required


def test_diff_data_revision_triggers_retrain():
    a = _fs("1.0", (_member(0),), data_rev="DR1")
    b = _fs("1.0", (_member(0),), data_rev="DR2")
    diff = classify_feature_set_diff(a, b)
    assert diff.category == FeatureSetDiffCategory.DATA_REVISION
    assert diff.retrain_required


def test_diff_rejects_different_feature_set_ids():
    a = FeatureSetVersion(feature_set_id="FS1", version="1.0", ordered_members=(_member(0),))
    b = FeatureSetVersion(feature_set_id="FS2", version="1.0", ordered_members=(_member(0),))
    with pytest.raises(ValueError):
        classify_feature_set_diff(a, b)


# ---- (d) cluster label-drift + lineage continuity ----
def test_algorithm_label_separate_from_logical_id():
    graph = SimilarityGraphVersion(graph_version_id="G1", graph_ref="graph://1")
    csv = ClusterSetVersion(cluster_set_version_id="CS1", similarity_graph_version=graph, algorithm="leiden")
    prev = ClusterVersion(
        cluster_version_id="CV1",
        logical_cluster_id="CL_PV_MOM_017",
        cluster_set_version_id="CS1",
        algorithm_cluster_label="cluster 18",
    )
    next_ = ClusterVersion(
        cluster_version_id="CV2",
        logical_cluster_id="CL_PV_MOM_017",  # same logical id
        cluster_set_version_id="CS1",
        algorithm_cluster_label="cluster 22",  # different algorithm label
    )
    drift = classify_cluster_label_drift(prev, next_)
    assert drift.label_changed is True
    assert drift.transition == ClusterConversionType.MIGRATED
    # Logical id continuity preserved.
    assert drift.logical_cluster_id == "CL_PV_MOM_017"


def test_unchanged_label_is_unchanged_transition():
    graph = SimilarityGraphVersion(graph_version_id="G1", graph_ref="graph://1")
    csv = ClusterSetVersion(cluster_set_version_id="CS1", similarity_graph_version=graph, algorithm="leiden")
    prev = ClusterVersion(
        cluster_version_id="CV1",
        logical_cluster_id="CL_PV_MOM_017",
        cluster_set_version_id="CS1",
        algorithm_cluster_label="cluster 18",
    )
    next_ = ClusterVersion(
        cluster_version_id="CV2",
        logical_cluster_id="CL_PV_MOM_017",
        cluster_set_version_id="CS1",
        algorithm_cluster_label="cluster 18",
    )
    drift = classify_cluster_label_drift(prev, next_)
    assert drift.label_changed is False
    assert drift.transition == ClusterConversionType.UNCHANGED


def test_incremental_split_merge_continuity():
    # Split: one prior cluster -> two new logical ids, SPLIT transition.
    split = resolve_incremental_cluster(
        case="split",
        logical_cluster_id="CL_NEW_001",
        algorithm_cluster_label="cluster 5a",
        prev_logical_cluster_ids=("CL_OLD_001",),
    )
    assert split.transition == ClusterConversionType.SPLIT
    assert split.logical_cluster_id == "CL_NEW_001"

    # Merge: several prior clusters -> one new logical id, MERGED transition.
    merge = resolve_incremental_cluster(
        case="merge",
        logical_cluster_id="CL_NEW_002",
        algorithm_cluster_label="cluster 9",
        prev_logical_cluster_ids=("CL_OLD_001", "CL_OLD_002"),
    )
    assert merge.transition == ClusterConversionType.MERGED
    assert merge.logical_cluster_id == "CL_NEW_002"

    # Ambiguous: keep prior logical id for continuity.
    amb = resolve_incremental_cluster(
        case="ambiguous",
        logical_cluster_id="CL_OLD_001",
        algorithm_cluster_label="cluster 3",
    )
    assert amb.transition == ClusterConversionType.MIGRATED
    assert amb.logical_cluster_id == "CL_OLD_001"

    # New / dissolved.
    new = resolve_incremental_cluster(case="new", logical_cluster_id="CL_NEW_003", algorithm_cluster_label="c")
    assert new.transition == ClusterConversionType.NEW
    dissolved = resolve_incremental_cluster(case="dissolved", logical_cluster_id="CL_OLD_009", algorithm_cluster_label="c")
    assert dissolved.transition == ClusterConversionType.DISSOLVED


def test_lineage_edges_built_from_pairs():
    from quant_platform.app.contracts.cluster import ClusterVersionPair

    pairs = (
        ClusterVersionPair(
            logical_cluster_id="CL_PV_MOM_017",
            prev_version_id="CV1",
            next_version_id="CV2",
            transition=ClusterConversionType.MIGRATED,
        ),
        ClusterVersionPair(
            logical_cluster_id="CL_PV_MOM_018",
            prev_version_id="CV3",
            next_version_id="CV4",
            transition=ClusterConversionType.SPLIT,
        ),
    )
    edges = build_lineage_edges(pairs)
    assert len(edges) == 2
    assert all(isinstance(e, ClusterLineageEdge) for e in edges)
    assert edges[0].old_version_id == "CV1" and edges[0].new_version_id == "CV2"
    assert edges[0].transition == ClusterConversionType.MIGRATED


# ---- version chain ----
def test_version_chain_single_active():
    links = (
        VersionChainLink(
            cluster_set_version_id="CS1",
            library_version_id="FLV_102",
            feature_set_version_id="FS1@1.0",
            model_version_id="M1@1",
            status=VersionChainStatus.ACTIVE,
        ),
        VersionChainLink(
            cluster_set_version_id="CS1",
            library_version_id="FLV_103",
            feature_set_version_id="FS1@2.0",
            model_version_id="M1@2",
            status=VersionChainStatus.SUPERSEDED,
        ),
    )
    chain = build_version_chain("CORE_LOW_REDUNDANCY", links)
    assert chain.active.library_version_id == "FLV_102"


def test_version_chain_rejects_multiple_active():
    links = (
        VersionChainLink(
            cluster_set_version_id="CS1",
            library_version_id="FLV_102",
            feature_set_version_id="FS1@1.0",
            status=VersionChainStatus.ACTIVE,
        ),
        VersionChainLink(
            cluster_set_version_id="CS1",
            library_version_id="FLV_103",
            feature_set_version_id="FS1@2.0",
            status=VersionChainStatus.ACTIVE,
        ),
    )
    with pytest.raises(ValueError):
        build_version_chain("CORE_LOW_REDUNDANCY", links)


# ---- model version ----
def test_model_version_holds_weight_in_model():
    label = LabelDefinition(label_definition_id="L1", label_name="fwd_ret", horizon="10d")
    snap = DataSnapshot(snapshot_id="S1")
    split = SplitPlan(split_plan_id="SP1")
    mv = ModelVersion(
        model_id="M1",
        model_version="1",
        model_architecture="lgbm",
        feature_set_version_ref="FS1@1.0",
        label_definition=label,
        data_snapshot=snap,
        split_plan=split,
        weights=(ModelWeight(feature_name="f1", position=0, weight=0.5),),
    )
    assert mv.weights[0].weight == 0.5
    assert mv.label_definition.return_basis == "vwap"
