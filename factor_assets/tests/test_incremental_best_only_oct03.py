"""Regression coverage for exact top-member selection from cached unit rows."""

from __future__ import annotations

import numpy as np
import pytest

from factor_assets.clustering import incremental as inc
from factor_assets.contracts.cluster_governance import ClusterVersionArtifact
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact


def _reference(unit_query, present, unit_matrix, cid, floor):
    """The pre-optimization full exact selector is the compatibility oracle."""
    # Reconstruct ordinary fingerprints is unnecessary: this is precisely the
    # old cached-unit-matrix path, including its score/status ordering.
    scores = inc.unit_cosine_scores(unit_query, unit_matrix)
    result = []
    for fid, score in zip(present, scores):
        score = float(score)
        status = (inc.PairwiseEvidenceStatus.MEASURED_LOW if score < floor
                  else inc.PairwiseEvidenceStatus.APPROXIMATE)
        result.append(inc.IncrementalCandidate(
            factor_id=fid, similarity=score, cluster_id=cid,
            evidence_status=status,
        ))
    result.sort(key=lambda candidate: candidate.similarity, reverse=True)
    return result[0] if result else None


def _assert_same(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is None:
        return
    assert actual.factor_id == expected.factor_id
    assert actual.similarity == expected.similarity
    assert actual.cluster_id == expected.cluster_id
    assert actual.evidence_status == expected.evidence_status


@pytest.mark.parametrize("seed", range(12))
def test_best_only_matches_full_exact_selector_on_seeded_matrices(seed):
    rng = np.random.default_rng(seed)
    count, width = 137, 29
    matrix = rng.normal(size=(count, width))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.normal(size=width)
    query /= np.linalg.norm(query)
    present = [f"member-{i:04d}" for i in range(count)]

    expected = _reference(query, present, matrix, "cluster-x", 0.31)
    actual = inc._best_exact_from_unit_matrix(
        query, present, matrix, "cluster-x", 0.31)
    _assert_same(actual, expected)


def test_best_only_preserves_first_member_on_perfect_ties_and_opposite_vectors():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    members = ["first", "opposite", "tied-later", "orthogonal"]
    result = inc._best_exact_from_unit_matrix(
        query, members, matrix, "ties", 0.0)
    _assert_same(result, _reference(query, members, matrix, "ties", 0.0))
    assert result.factor_id == "first"
    assert result.similarity == 1.0


def test_best_only_does_not_treat_adjacent_float_scores_as_a_tie():
    query = np.array([1.0, 0.0])
    lower = 0.5
    higher = np.nextafter(lower, 1.0)
    matrix = np.array([
        [lower, np.sqrt(1.0 - lower * lower)],
        [higher, np.sqrt(1.0 - higher * higher)],
    ])
    result = inc._best_exact_from_unit_matrix(
        query, ["lower-first", "higher-second"], matrix, "adjacent", -1.0)
    _assert_same(
        result,
        _reference(query, ["lower-first", "higher-second"], matrix,
                   "adjacent", -1.0),
    )
    assert result.factor_id == "higher-second"


def test_best_only_handles_extreme_finite_rows_and_missing_members():
    # These vectors represent the already-normalized cache output; constructing
    # via large finite inputs also protects against assumptions about source
    # embedding magnitude in this selector.
    raw = np.array([[1e150, -1e150], [-1e150, 1e150], [1e150, 1e150]])
    matrix = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    query = np.array([1.0, 0.0])
    present = ["kept-a", "kept-c", "kept-z"]
    # The cache's present list is authoritative; absent catalog members are
    # intentionally not scored or invented by the best-only selector.
    actual = inc._best_exact_from_unit_matrix(
        query, present, matrix, "missing", -1.0)
    _assert_same(actual, _reference(query, present, matrix, "missing", -1.0))
    assert actual.factor_id in present


@pytest.mark.parametrize("floor", [-1.0, 1.0])
def test_best_only_preserves_floor_boundary_status(floor):
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = inc._best_exact_from_unit_matrix(
        query, ["boundary", "low"], matrix, "floor", floor)
    _assert_same(result, _reference(query, ["boundary", "low"], matrix,
                                    "floor", floor))
    assert result.evidence_status is inc.PairwiseEvidenceStatus.APPROXIMATE


def test_best_only_noncontiguous_inputs_match_full_reference():
    rng = np.random.default_rng(901)
    backing = rng.normal(size=(31, 12))
    query_backing = rng.normal(size=12)
    # Strided views exercise the matrix multiply with non-C-contiguous inputs.
    matrix_view = backing[:, ::2]
    query_view = query_backing[::2]
    matrix_view /= np.linalg.norm(matrix_view, axis=1, keepdims=True)
    query_view /= np.linalg.norm(query_view)
    present = [f"strided-{i}" for i in range(len(matrix_view))]
    _assert_same(
        inc._best_exact_from_unit_matrix(
            query_view, present, matrix_view, "strided", 0.2),
        _reference(query_view, present, matrix_view, "strided", 0.2),
    )


def test_best_only_empty_matrix_returns_none_and_rejects_row_id_mismatch():
    query = np.array([1.0, 0.0])
    assert inc._best_exact_from_unit_matrix(
        query, [], np.empty((0, 2)), "empty", 0.0) is None
    with pytest.raises(ValueError, match="member ids and rows"):
        inc._best_exact_from_unit_matrix(
            query, ["only-id"], np.empty((0, 2)), "mismatch", 0.0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_best_only_rejects_invalid_nonwinner_score(invalid):
    # The first row has an unquestionably winning score. The invalid second
    # row must still fail closed instead of being hidden by argmax.
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [invalid, 0.0]])
    with pytest.raises(ValueError, match="unit cosine scores"):
        inc._best_exact_from_unit_matrix(
            query, ["winner", "invalid-nonwinner"], matrix, "invalid", 0.0)


def test_best_only_reports_measured_low_for_below_floor_winner():
    query = np.array([1.0, 0.0])
    matrix = np.array([[0.1, np.sqrt(0.99)], [0.0, 1.0]])
    result = inc._best_exact_from_unit_matrix(
        query, ["below-floor", "zero"], matrix, "low", 0.5)
    assert result.factor_id == "below-floor"
    assert result.similarity == pytest.approx(0.1)
    assert result.evidence_status is inc.PairwiseEvidenceStatus.MEASURED_LOW


def test_best_only_does_not_construct_candidates_for_every_member(monkeypatch):
    rng = np.random.default_rng(20261003)
    members_count, query_count = 4096, 5
    matrix = rng.normal(size=(members_count, 8))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    members = [f"m-{i}" for i in range(members_count)]
    original = inc.IncrementalCandidate
    calls = []

    def counting_candidate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(inc, "IncrementalCandidate", counting_candidate)
    for _ in range(query_count):
        query = rng.normal(size=8)
        query /= np.linalg.norm(query)
        result = inc._best_exact_from_unit_matrix(
            query, members, matrix, "large-cluster", 0.0)
        assert result is not None
    # One returned public candidate per query, independent of member count.
    assert len(calls) == query_count


def _public_fp(fid, embedding):
    return SimilarityFingerprintArtifact(
        factor_id=fid,
        embedding=tuple(float(x) for x in embedding),
        embedding_spec="best-only-test",
        snapshot="2026-10-03",
        universe="test-universe",
        window="all",
        preprocessing_ref="preprocess-v1",
        mask_policy="finite",
        direction="signed",
        aggregation_method="test",
        embedding_model_version="test-v1",
    )


def test_public_incremental_assign_constructs_one_candidate_per_cluster_query(monkeypatch):
    # The public path receives three new queries and two large clusters. Its
    # exact candidate allocation should therefore be six, even as membership
    # is much larger (96 x 3 potential member/query pairs).
    members_per_cluster = 48
    fingerprints = {}
    clusters = {}
    for cluster_index, direction in enumerate(((1.0, 0.0), (0.0, 1.0))):
        cid = f"cluster-{cluster_index}"
        ids = []
        for member_index in range(members_per_cluster):
            fid = f"{cid}-member-{member_index}"
            ids.append(fid)
            fingerprints[fid] = _public_fp(fid, direction)
        clusters[cid] = ClusterVersionArtifact(
            logical_cluster_id=cid,
            cluster_set_version_ref="set-1",
            member_factor_ids=tuple(ids),
            representative_factor_id=ids[0],
        )
    queries = [
        _public_fp(f"query-{i}", (1.0, 0.05 * (i + 1)))
        for i in range(3)
    ]
    fingerprints.update((query.factor_id, query) for query in queries)

    original = inc.IncrementalCandidate
    calls = []

    def counting_candidate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(inc, "IncrementalCandidate", counting_candidate)
    result = inc.incremental_assign(
        queries,
        clusters,
        fingerprints,
        inc.IncrementalPolicy(
            affinity_threshold=0.0, ambiguity_gap=0.0, min_measure_floor=-1.0),
        batch_id="best-only-public-count",
    )
    assert len(result.assignments) == len(queries)
    assert len(calls) == len(clusters) * len(queries)
