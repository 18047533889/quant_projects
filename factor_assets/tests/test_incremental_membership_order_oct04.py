"""Public incremental recall ordering and cross-cluster membership guards."""

import pytest

import factor_assets.clustering.incremental as incremental
from factor_assets.contracts.cluster_governance import (
    ClusterScale,
    ClusterVersionArtifact,
)
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact


def _fp(fid, embedding):
    return SimilarityFingerprintArtifact(
        factor_id=fid,
        embedding=tuple(embedding),
        embedding_spec="test-v1",
        snapshot="snapshot-1",
        universe="universe-1",
        window="window-1",
        preprocessing_ref="prep-1",
        mask_policy="pairwise-valid",
        direction="signed",
        aggregation_method="daily-then-time",
        embedding_model_version="embedding-1",
        value_ref=f"values:{fid}",
        profile_ref=f"profile:{fid}",
    )


def _cluster(cid, members):
    return ClusterVersionArtifact(
        logical_cluster_id=cid,
        cluster_set_version_ref="set-1",
        member_factor_ids=tuple(members),
        representative_factor_id=members[0],
        scale=ClusterScale.MICRO_CLUSTER,
    )


def test_exact_batch_result_is_invariant_to_cluster_mapping_insertion_order():
    # Same logical cluster set and query order must yield a stable audit tuple/hash.
    members = {
        "m-a": _fp("m-a", (1.0, 0.0)),
        "m-b": _fp("m-b", (0.0, 1.0)),
    }
    queries = [
        _fp("q-a", (0.9, 0.1)),
        _fp("q-b", (0.1, 0.9)),
        _fp("q-c", (0.7, 0.3)),
    ]
    mapping = {**members, **{query.factor_id: query for query in queries}}
    cluster_a = _cluster("A", ("m-a",))
    cluster_b = _cluster("B", ("m-b",))
    clusters_ab = {"A": cluster_a, "B": cluster_b}
    clusters_ba = {"B": cluster_b, "A": cluster_a}
    snapshot_ab = tuple((cid, cv.member_factor_ids) for cid, cv in clusters_ab.items())
    snapshot_ba = tuple((cid, cv.member_factor_ids) for cid, cv in clusters_ba.items())
    fingerprint_keys = tuple(mapping)

    forward = incremental.incremental_assign(
        queries, clusters_ab, mapping, batch_id="stable-order-probe")
    reversed_mapping = incremental.incremental_assign(
        queries, clusters_ba, mapping, batch_id="stable-order-probe")

    assert forward.assignments == reversed_mapping.assignments
    assert forward.candidates == reversed_mapping.candidates
    assert forward.content_hash == reversed_mapping.content_hash
    assert tuple(candidate.cluster_id for candidate in forward.candidates) == ("A", "B") * len(queries)
    assert tuple((cid, cv.member_factor_ids) for cid, cv in clusters_ab.items()) == snapshot_ab
    assert tuple((cid, cv.member_factor_ids) for cid, cv in clusters_ba.items()) == snapshot_ba
    assert tuple(mapping) == fingerprint_keys


def test_exact_recall_rejects_cross_cluster_duplicate_member_before_member_scan(monkeypatch):
    # If one factor is listed in two logical clusters, treating both as
    # independent candidates creates a false runner-up and false ambiguity.
    member = _fp("shared", (1.0, 0.0))
    queries = [_fp("q-a", (1.0, 0.0)), _fp("q-b", (0.9, 0.1))]
    mapping = {"shared": member, **{query.factor_id: query for query in queries}}
    clusters = {
        "A": _cluster("A", ("shared",)),
        "B": _cluster("B", ("shared",)),
    }
    original_membership = {cid: cv.member_factor_ids for cid, cv in clusters.items()}

    def fail_if_member_scan(*args, **kwargs):
        pytest.fail("invalid overlapping membership reached exact member preparation")

    monkeypatch.setattr(incremental, "_prepare_unit_members", fail_if_member_scan)
    with pytest.raises(ValueError, match="member factor.*multiple logical clusters"):
        incremental.incremental_assign(queries, clusters, mapping, batch_id="overlap-probe")

    assert {cid: cv.member_factor_ids for cid, cv in clusters.items()} == original_membership
    assert tuple(mapping) == ("shared", "q-a", "q-b")
