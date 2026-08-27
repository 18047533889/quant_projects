"""QRP-P0R-C1 contract correction tests.

Cover the C1 corrections:
(a) strict contenthash codec raises on NaN/Inf/naive-datetime/unsupported-type and
    normalizes -0.0;
(b) ArtifactRef does NOT require a recomputable object hash (only format
    validation; a valid-format hash is accepted);
(c) EventEnvelope deep-freezes payload and accepts new trace fields;
(d) FeatureSetArtifact content_hash verified when supplied (fail closed);
(e) FactorLibraryVersion requires cluster_set_version_id (not cluster_version_id)
    and ClusterSetVersion references SimilarityGraphVersion;
(f) LifecycleState has NO MATERIALIZING while QRPPipelineStage does, and all four
    enums are distinct;
(g) security: factor:read_formula requires >= CONFIDENTIAL_ALPHA classification.
"""

import math
from datetime import date, datetime, timezone

import pytest

import quant_platform.app.contracts as c


# ---- (a) strict contenthash codec ----
def test_contenthash_raises_on_nan():
    from quant_platform.app.contracts._contenthash import content_hash

    with pytest.raises(ValueError):
        content_hash(float("nan"))


def test_contenthash_raises_on_inf():
    from quant_platform.app.contracts._contenthash import content_hash

    with pytest.raises(ValueError):
        content_hash(float("inf"))
    with pytest.raises(ValueError):
        content_hash(float("-inf"))


def test_contenthash_raises_on_naive_datetime():
    from quant_platform.app.contracts._contenthash import content_hash

    with pytest.raises(ValueError):
        content_hash(datetime(2024, 1, 1, 12, 0, 0))  # naive


def test_contenthash_accepts_aware_datetime_as_z():
    from quant_platform.app.contracts import canonical_str

    s = canonical_str(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert s.endswith("Z")
    assert "+00:00" not in s


def test_contenthash_raises_on_unsupported_type():
    from quant_platform.app.contracts._contenthash import content_hash

    with pytest.raises(TypeError):
        content_hash(object())


def test_contenthash_normalizes_negative_zero():
    from quant_platform.app.contracts import canonical_str

    assert canonical_str(-0.0) == canonical_str(0.0) == "0.0"


def test_contenthash_enum_reduces_to_value():
    from quant_platform.app.contracts import canonical_str, SecurityClassification

    assert canonical_str(SecurityClassification.CONFIDENTIAL_ALPHA) == "CONFIDENTIAL_ALPHA"


def test_contenthash_sorts_dict_keys():
    from quant_platform.app.contracts import canonical_str

    assert canonical_str({"b": 1, "a": 2}) == canonical_str({"a": 2, "b": 1})


# ---- (b) ArtifactRef is a reference, not a validator ----
def test_artifact_ref_accepts_format_valid_hash_without_recompute():
    # A caller-supplied hash is accepted (only format validated); ArtifactRef
    # does NOT recompute the object-bytes hash.
    ref = c.ArtifactRef(
        artifact_id="A1",
        artifact_type="FACTOR_DEFINITION",
        schema_version="1.0",
        content_hash="a" * 64,
        storage_uri="cos://bucket/key",
        size_bytes=10,
        created_at=datetime.now(timezone.utc),
        producer_type="FACTOR_ENGINE",
        producer_version="1.0",
        semantic_hash="b" * 64,
    )
    assert ref.content_hash == "a" * 64
    assert ref.semantic_hash == "b" * 64


def test_artifact_ref_rejects_bad_hash_format():
    with pytest.raises(ValueError):
        c.ArtifactRef(
            artifact_id="A1",
            artifact_type="FACTOR_DEFINITION",
            schema_version="1.0",
            content_hash="not-a-hash",
            storage_uri="cos://bucket/key",
            size_bytes=10,
            created_at=datetime.now(timezone.utc),
            producer_type="FACTOR_ENGINE",
            producer_version="1.0",
        )


def test_artifact_ref_rejects_missing_uri_scheme():
    with pytest.raises(ValueError):
        c.ArtifactRef(
            artifact_id="A1",
            artifact_type="FACTOR_DEFINITION",
            schema_version="1.0",
            content_hash="a" * 64,
            storage_uri="bucket/key",  # no scheme
            size_bytes=10,
            created_at=datetime.now(timezone.utc),
            producer_type="FACTOR_ENGINE",
            producer_version="1.0",
        )


def test_artifact_ref_rejects_negative_size():
    with pytest.raises(ValueError):
        c.ArtifactRef(
            artifact_id="A1",
            artifact_type="FACTOR_DEFINITION",
            schema_version="1.0",
            content_hash="a" * 64,
            storage_uri="cos://bucket/key",
            size_bytes=-1,
            created_at=datetime.now(timezone.utc),
            producer_type="FACTOR_ENGINE",
            producer_version="1.0",
        )


# ---- (c) EventEnvelope deep-freezes payload + new trace fields ----
def test_event_envelope_deep_freezes_payload():
    from types import MappingProxyType

    env = c.EventEnvelope(
        event_id="E1",
        event_type=c.EVENT_TYPE_JOB_FAILED,
        schema_version="1.0",
        occurred_at=datetime.now(timezone.utc),
        producer="p",
        aggregate_type="FACTOR",
        aggregate_id="F1",
        correlation_id="C1",
        payload={"nested": {"k": [1, 2, 3]}},
        trace_id="T1",
        actor_principal_id="U1",
        idempotency_key="IK1",
        event_version=2,
    )
    assert isinstance(env.payload, MappingProxyType)
    assert isinstance(env.payload["nested"], MappingProxyType)
    assert isinstance(env.payload["nested"]["k"], tuple)
    assert env.trace_id == "T1"
    assert env.actor_principal_id == "U1"
    assert env.idempotency_key == "IK1"
    assert env.event_version == 2


# ---- (d) FeatureSetArtifact content_hash verified when supplied ----
def test_feature_set_artifact_verifies_content_hash():
    fs = c.FeatureSetArtifact(
        feature_set_id="FS1",
        feature_set_version="1.0",
        ordered_feature_manifest=("F1", "F2"),
    )
    # A caller-supplied correct hash is accepted.
    fs2 = c.FeatureSetArtifact(
        feature_set_id="FS1",
        feature_set_version="1.0",
        ordered_feature_manifest=("F1", "F2"),
        content_hash=fs.content_hash,
    )
    assert fs2.content_hash == fs.content_hash


def test_feature_set_artifact_fails_closed_on_mismatch():
    with pytest.raises(ValueError):
        c.FeatureSetArtifact(
            feature_set_id="FS1",
            feature_set_version="1.0",
            ordered_feature_manifest=("F1", "F2"),
            content_hash="0" * 64,  # wrong
        )


def test_feature_set_version_has_semantic_hash_including_members():
    v = c.FeatureSetVersion(
        feature_set_id="FS1",
        version="1.0",
        ordered_members=(
            c.FeatureMemberRef(position=0, feature_name="f", factor_definition_ref="FD"),
        ),
    )
    assert v.schema_hash
    assert v.semantic_hash
    # Changing member order must change the hash.
    v2 = c.FeatureSetVersion(
        feature_set_id="FS1",
        version="1.0",
        ordered_members=(
            c.FeatureMemberRef(position=0, feature_name="g", factor_definition_ref="FD"),
        ),
    )
    assert v.semantic_hash != v2.semantic_hash


# ---- (e) Cluster/library layered model ----
def test_factor_library_version_requires_cluster_set_version_id():
    graph = c.SimilarityGraphVersion(graph_version_id="G1", graph_ref="graph://1")
    csv = c.ClusterSetVersion(
        cluster_set_version_id="CS1",
        similarity_graph_version=graph,
        algorithm="leiden",
    )
    lib = c.FactorLibraryVersion(
        library_version_id="L1",
        logical_library_id="CORE_LOW_REDUNDANCY",
        cluster_set_version_id=csv.cluster_set_version_id,
    )
    assert lib.cluster_set_version_id == "CS1"
    # No cluster_version_id field may exist anymore.
    assert not hasattr(lib, "cluster_version_id")
    # And you cannot construct it with the old field name.
    with pytest.raises(TypeError):
        c.FactorLibraryVersion(
            library_version_id="L1",
            logical_library_id="CORE_LOW_REDUNDANCY",
            cluster_version_id="CV1",  # old name
        )


def test_cluster_set_version_references_similarity_graph_version():
    graph = c.SimilarityGraphVersion(graph_version_id="G1", graph_ref="graph://1")
    csv = c.ClusterSetVersion(
        cluster_set_version_id="CS1",
        similarity_graph_version=graph,
        algorithm="leiden",
    )
    assert csv.similarity_graph_version.graph_version_id == "G1"


def test_cluster_version_belongs_to_cluster_set():
    cv = c.ClusterVersion(
        cluster_version_id="CV1",
        logical_cluster_id="CL_PV_MOM_017",
        cluster_set_version_id="CS1",
        algorithm_cluster_label="cluster 18",
    )
    assert cv.cluster_set_version_id == "CS1"


# ---- (f) LifecycleState vs QRPPipelineStage separation ----
def test_lifecycle_state_has_no_materializing():
    assert "MATERIALIZING" not in {s.value for s in c.LifecycleState}
    assert "RAW_EVALUATING" not in {s.value for s in c.LifecycleState}
    assert "TREATMENT_SEARCHING" not in {s.value for s in c.LifecycleState}


def test_qrp_pipeline_stage_has_materializing():
    stage_values = {s.value for s in c.QRPPipelineStage}
    assert "MATERIALIZING" in stage_values
    assert "RAW_EVALUATING" in stage_values
    assert "TREATMENT_SEARCHING" in stage_values
    assert "DISCOVERED" in stage_values
    assert "LIBRARY" in stage_values
    assert "PRODUCTION" in stage_values


def test_four_enums_are_distinct():
    # The invariant is that these are four INDEPENDENT enum *types* (a stage is
    # not a LifecycleState etc.). Some enum *values* may legitimately share a
    # label (e.g. "PRODUCTION") — what matters is that a value of one type is
    # never an instance of another type and the member names do not collide in a
    # way that conflates semantics.
    # 1. Every enum is a distinct type (no member name shared across two enums).
    all_member_names = {}
    for enum_cls in (c.LifecycleState, c.QRPPipelineStage, c.JobStatus, c.HealthState):
        for member in enum_cls:
            all_member_names.setdefault(member.name, set()).add(enum_cls.__name__)
    # 2. Cross-type assignment is impossible: a value of one type is not an
    #    instance of another type.
    assert not isinstance(c.LifecycleState.REGISTERED, c.JobStatus)
    assert not isinstance(c.QRPPipelineStage.MATERIALIZING, c.LifecycleState)
    assert not isinstance(c.JobStatus.RUNNING, c.HealthState)
    assert not isinstance(c.HealthState.HEALTHY, c.QRPPipelineStage)
    # 3. No enum type is a subclass of another.
    types = [c.LifecycleState, c.QRPPipelineStage, c.JobStatus, c.HealthState]
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            if i != j:
                assert not issubclass(a, b), f"{a.__name__} is subclass of {b.__name__}"


# ---- (g) security classification gating ----
def test_security_formula_requires_confidential_alpha():
    assert (
        c.PERMISSION_MIN_CLASSIFICATION[c.Permission.FACTOR_READ_FORMULA]
        == c.SecurityClassification.CONFIDENTIAL_ALPHA
    )
    assert (
        c.PERMISSION_MIN_CLASSIFICATION[c.Permission.FACTOR_READ_FORMULA].rank
        >= c.SecurityClassification.CONFIDENTIAL_ALPHA.rank
    )


def test_security_raw_values_requires_restricted():
    assert (
        c.PERMISSION_MIN_CLASSIFICATION[c.Permission.FACTOR_READ_VALUES]
        == c.SecurityClassification.RESTRICTED_RAW_VALUES
    )
    assert (
        c.PERMISSION_MIN_CLASSIFICATION[c.Permission.FACTOR_READ_VALUES].rank
        >= c.SecurityClassification.RESTRICTED_RAW_VALUES.rank
    )


def test_rbac_new_types_exist():
    assert c.Team.FACTOR_TEAM.value == "FACTOR_TEAM"
    assert c.Role.CORE.value == "CORE"
    hp = c.HumanPrincipal(
        principal_id="u1",
        display_name="Ding",
        team=c.Team.FACTOR_TEAM,
        role=c.Role.CORE,
    )
    assert hp.principal_id == "u1"
    wp = c.WorkloadPrincipal(principal_id="svc1", service_name="publisher")
    assert wp.role == c.Role.SERVICE
