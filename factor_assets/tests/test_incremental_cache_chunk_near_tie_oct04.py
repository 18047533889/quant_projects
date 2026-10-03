"""Public cached/chunked exact-recall parity at high-dimensional near ties."""

from decimal import Decimal, localcontext
import numpy as np
import pytest

import factor_assets.clustering.incremental as incremental
from factor_assets.contracts.cluster_governance import (
    ClusterScale,
    ClusterVersionArtifact,
)
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact


def _fingerprint(fid, vector):
    return SimilarityFingerprintArtifact(
        factor_id=fid,
        embedding=tuple(map(float, vector)),
        embedding_spec="cache-chunk-near-tie-oct04",
        snapshot="snapshot",
        universe="universe",
        window="window",
        preprocessing_ref="prep",
        mask_policy="pairwise-valid",
        direction="signed",
        aggregation_method="daily-then-time",
        embedding_model_version="model",
        value_ref=f"values:{fid}",
        profile_ref=f"profile:{fid}",
    )


def _cluster(cluster_id, member_ids):
    return ClusterVersionArtifact(
        logical_cluster_id=cluster_id,
        cluster_set_version_ref="set",
        member_factor_ids=tuple(member_ids),
        representative_factor_id=member_ids[0],
        scale=ClusterScale.MICRO_CLUSTER,
    )


def _compare_cached_and_chunked(monkeypatch, query, fingerprints, clusters):
    policy = incremental.IncrementalPolicy(
        affinity_threshold=0.5,
        ambiguity_gap=0.0,
        min_measure_floor=1.0,
    )
    previous = incremental._EXACT_MEMBER_CACHE_BYTES
    try:
        # These matrices are tiny; force the reference branch independently of
        # future production cache-cap changes.
        monkeypatch.setattr(incremental, "_EXACT_MEMBER_CACHE_BYTES", 1 << 30)
        cached = incremental.incremental_assign(
            [query], clusters, fingerprints, policy=policy,
        )
        monkeypatch.setattr(incremental, "_EXACT_MEMBER_CACHE_BYTES", 1)
        chunked = incremental.incremental_assign(
            [query], clusters, fingerprints, policy=policy,
        )
    finally:
        monkeypatch.setattr(incremental, "_EXACT_MEMBER_CACHE_BYTES", previous)
    return cached, chunked


def _candidate_map(result):
    return {candidate.cluster_id: candidate for candidate in result.candidates}


@pytest.mark.parametrize("epsilon", [2.0**-25, 2.0**-26])
def test_public_recall_preserves_256d_near_tie_winner_and_floor_status(
        monkeypatch, epsilon):
    # q and r are orthonormal in R^256. p=q+epsilon*r has exact cosine
    # 1/sqrt(1+epsilon**2) with q, just below the exact-match member's score.
    q = np.full(256, 1.0 / 16.0, dtype=np.float64)
    r = np.concatenate((
        np.full(128, 1.0 / 16.0, dtype=np.float64),
        np.full(128, -1.0 / 16.0, dtype=np.float64),
    ))
    p = q + epsilon * r
    assert np.dot(q, q) == 1.0
    assert np.dot(r, r) == 1.0
    assert np.dot(q, r) == 0.0
    with localcontext() as context:
        context.prec = 100
        exact_epsilon = Decimal.from_float(epsilon)
        assert Decimal(1) / (Decimal(1) + exact_epsilon**2).sqrt() < Decimal(1)

    member_ids = tuple(f"member-{index:03d}" for index in range(257))
    near_ids, exact_ids = member_ids[:256], member_ids[256:]
    fingerprints = {
        fid: _fingerprint(fid, -q)
        for fid in near_ids[:-1]
    }
    fingerprints[near_ids[-1]] = _fingerprint(near_ids[-1], p)
    fingerprints[exact_ids[0]] = _fingerprint(exact_ids[0], q)
    query = _fingerprint("query", q)
    fingerprints[query.factor_id] = query
    clusters = {
        "near": _cluster("near", near_ids),
        "exact": _cluster("exact", exact_ids),
    }

    cached, chunked = _compare_cached_and_chunked(
        monkeypatch, query, fingerprints, clusters,
    )
    cached_candidates, chunked_candidates = _candidate_map(cached), _candidate_map(chunked)
    assert set(cached_candidates) == set(chunked_candidates) == {"near", "exact"}
    for cluster_id in ("near", "exact"):
        left, right = cached_candidates[cluster_id], chunked_candidates[cluster_id]
        assert left.factor_id == right.factor_id
        assert left.evidence_status is right.evidence_status
        assert left.similarity == pytest.approx(right.similarity, rel=0.0, abs=2e-15)

    near = cached_candidates["near"]
    exact = cached_candidates["exact"]
    assert near.factor_id == near_ids[-1]
    assert near.similarity < 1.0
    assert near.evidence_status is incremental.PairwiseEvidenceStatus.MEASURED_LOW
    assert exact.factor_id == exact_ids[0]
    assert exact.similarity == 1.0
    assert exact.evidence_status is incremental.PairwiseEvidenceStatus.APPROXIMATE

    for result in (cached, chunked):
        assignment = result.assignments[0]
        assert assignment.logical_cluster_id == "exact"
        assert assignment.kind is incremental.IncrementalAssignmentKind.RESEARCH_PROPOSED


def test_public_recall_keeps_first_256d_duplicate_across_chunk_boundary(monkeypatch):
    q = np.zeros(256, dtype=np.float64)
    q[0] = 1.0
    orthogonal = np.zeros(256, dtype=np.float64)
    orthogonal[1] = 1.0
    member_ids = tuple(f"member-{index:03d}" for index in range(257))
    first_tie, later_tie = member_ids[255], member_ids[256]
    vectors = [orthogonal.copy() for _ in member_ids]
    vectors[255] = q.copy()
    vectors[256] = q.copy()
    fingerprints = {
        fid: _fingerprint(fid, vector)
        for fid, vector in zip(member_ids, vectors, strict=True)
    }
    query = _fingerprint("query", q)
    fingerprints[query.factor_id] = query
    clusters = {"cluster": _cluster("cluster", member_ids)}

    cached, chunked = _compare_cached_and_chunked(
        monkeypatch, query, fingerprints, clusters,
    )
    for result in (cached, chunked):
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.cluster_id == "cluster"
        assert candidate.factor_id == first_tie
        assert candidate.factor_id != later_tie
        assert candidate.similarity == 1.0
        assert candidate.evidence_status is incremental.PairwiseEvidenceStatus.APPROXIMATE
        assignment = result.assignments[0]
        assert assignment.logical_cluster_id == "cluster"
        assert assignment.kind is incremental.IncrementalAssignmentKind.RESEARCH_PROPOSED
