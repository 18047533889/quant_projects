"""Batch exact member-scan contracts against the single-query reference."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from factor_assets.clustering import incremental_recall
from factor_assets.clustering.incremental_recall import (
    scan_exact_member_winner,
    scan_exact_member_winners,
)
from factor_assets.similarity.unit_vectors import _unit_rows


def _member(fid, values):
    return SimpleNamespace(factor_id=fid, embedding=np.asarray(values, dtype=np.float64))


def _single(mapping, member_ids, query, **kwargs):
    return scan_exact_member_winner(
        mapping, member_ids, np.asarray(query, dtype=np.float64),
        to_embedding=lambda item: item.embedding,
        normalize_rows=_unit_rows,
        **kwargs,
    )


def _batch(mapping, member_ids, queries, **kwargs):
    unit_queries = np.vstack([np.asarray(query, dtype=np.float64) for query in queries])
    return scan_exact_member_winners(
        mapping, member_ids, unit_queries,
        to_embedding=lambda item: item.embedding,
        normalize_rows=_unit_rows,
        **kwargs,
    )


def _assert_matches_single(mapping, member_ids, queries, **kwargs):
    expected = [
        _single(mapping, member_ids, query, **kwargs) for query in queries
    ]
    actual = _batch(mapping, member_ids, queries, **kwargs)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            assert got is not None
            assert got[0] == want[0]
            assert got[1] == pytest.approx(want[1], rel=0.0, abs=0.0)


def test_batch_matches_single_scan_for_multichunk_exact_and_near_ties():
    mapping = {
        "exact-first": _member("exact-first", [1.0, 0.0, 0.0]),
        "orthogonal": _member("orthogonal", [0.0, 1.0, 0.0]),
        "exact-later": _member("exact-later", [1.0, 0.0, 0.0]),
        "opposite": _member("opposite", [-1.0, 0.0, 0.0]),
    }
    member_ids = list(mapping)
    queries = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

    _assert_matches_single(
        mapping, member_ids, queries,
        max_chunk_rows=2, max_chunk_bytes=2 * 3 * 8 * 3,
    )
    result = _batch(
        mapping, member_ids, queries[:1],
        max_chunk_rows=1, max_chunk_bytes=3 * 8 * 3,
    )
    assert result[0][0] == "exact-first"


def test_batch_preserves_query_order_and_returns_none_for_empty_member_matches():
    mapping = {
        "x": _member("x", [1.0, 0.0]),
        "y": _member("y", [0.0, 1.0]),
    }
    queries = ([0.0, 1.0], [1.0, 0.0], [-1.0, 0.0])

    result = _batch(mapping, ["not-present", "y", "x"], queries,
                    max_chunk_rows=1)
    assert [winner[0] if winner else None for winner in result] == ["y", "x", "y"]
    _assert_matches_single(mapping, ["missing"], queries, max_chunk_rows=1)


def test_empty_query_batch_does_not_prepare_any_member(monkeypatch):
    mapping = {"member": _member("member", [1.0, 0.0])}

    def forbidden(*args, **kwargs):
        pytest.fail("empty query batch must return without preparing members")

    monkeypatch.setattr(incremental_recall, "_rowwise_unit_cosine_scores", forbidden)
    result = scan_exact_member_winners(
        mapping, ["member"], np.empty((0, 2), dtype=np.float64),
        to_embedding=forbidden, normalize_rows=forbidden,
    )
    assert result == []


def test_batch_rejects_miskeyed_present_fingerprint():
    mapping = {"alias": _member("actual", [1.0, 0.0])}
    with pytest.raises(ValueError, match="mapping key"):
        _batch(mapping, ["missing", "alias"], ([1.0, 0.0],))


@pytest.mark.parametrize(
    "bad_embedding, message",
    [
        ([1.0, 0.0, 0.0], "matching the query width"),
        ([np.nan, 0.0], "only finite values"),
        ([np.inf, 0.0], "only finite values"),
        ([-np.inf, 0.0], "only finite values"),
    ],
)
def test_batch_rejects_invalid_member_even_after_provisional_winner(
    bad_embedding, message,
):
    mapping = {
        "provisional-winner": _member("provisional-winner", [1.0, 0.0]),
        "invalid-later": _member("invalid-later", bad_embedding),
    }
    with pytest.raises(ValueError, match=message):
        _batch(mapping, list(mapping), ([1.0, 0.0],),
               max_chunk_rows=1, max_chunk_bytes=2 * 8 * 3)


def test_batch_normalizes_each_member_chunk_once_for_many_queries(monkeypatch):
    width = 5
    member_ids = [f"member-{index}" for index in range(8)]
    mapping = {
        fid: _member(fid, np.arange(1, width + 1, dtype=np.float64) + index)
        for index, fid in enumerate(member_ids)
    }
    queries = [
        np.eye(width, dtype=np.float64)[index]
        for index in range(4)
    ]
    normalized_chunk_shapes = []
    embedding_calls = []
    score_shapes = []
    original_rowwise = incremental_recall._rowwise_unit_cosine_scores

    def count_embedding(item):
        embedding_calls.append(item.factor_id)
        return item.embedding

    def count_normalization(rows):
        normalized_chunk_shapes.append(rows.shape)
        return _unit_rows(rows)

    def count_scores(query, unit_matrix):
        score_shapes.append((query.shape, unit_matrix.shape))
        return original_rowwise(query, unit_matrix)

    monkeypatch.setattr(
        incremental_recall, "_rowwise_unit_cosine_scores", count_scores)
    result = scan_exact_member_winners(
        mapping, member_ids, np.vstack(queries),
        to_embedding=count_embedding,
        normalize_rows=count_normalization,
        max_chunk_rows=3,
        max_chunk_bytes=3 * width * 8 * 3,
        max_batch_queries=4,
    )

    assert len(result) == len(queries)
    assert embedding_calls == member_ids
    assert normalized_chunk_shapes == [(3, width), (3, width), (2, width)]
    assert all(query_shape == (width,) for query_shape, _ in score_shapes)
    assert all(matrix_shape[0] <= 3 for _, matrix_shape in score_shapes)
    assert len(score_shapes) == len(queries) * len(normalized_chunk_shapes)


def test_batch_validates_every_member_before_returning_results(monkeypatch):
    mapping = {
        "best": _member("best", [1.0, 0.0]),
        "later": _member("later", [0.0, 1.0]),
    }
    seen = []

    def track_embedding(item):
        seen.append(item.factor_id)
        return item.embedding

    result = scan_exact_member_winners(
        mapping, list(mapping), np.array([[1.0, 0.0]]),
        to_embedding=track_embedding,
        normalize_rows=_unit_rows,
        max_chunk_rows=1,
    )
    assert result[0][0] == "best"
    assert seen == list(mapping)


def test_batch_validates_query_shape_finiteness_and_count_before_preparing(monkeypatch):
    mapping = {"member": _member("member", [1.0, 0.0])}
    calls = []

    def count_embedding(item):
        calls.append(item.factor_id)
        return item.embedding

    with pytest.raises(ValueError, match="two-dimensional"):
        scan_exact_member_winners(
            mapping, ["member"], np.array([1.0, 0.0]),
            to_embedding=count_embedding, normalize_rows=_unit_rows,
        )
    with pytest.raises(ValueError, match="finite real numeric"):
        scan_exact_member_winners(
            mapping, ["member"], np.array([[np.inf, 0.0]]),
            to_embedding=count_embedding, normalize_rows=_unit_rows,
        )
    with pytest.raises(ValueError, match="exceeds"):
        scan_exact_member_winners(
            mapping, ["member"], np.ones((3, 2)),
            to_embedding=count_embedding, normalize_rows=_unit_rows,
            max_batch_queries=2,
        )
    assert calls == []


def test_near_tie_across_chunks_matches_single_query_reference():
    low = 0.5
    high = np.nextafter(low, 1.0)
    mapping = {
        "lower-first": _member(
            "lower-first", [low, np.sqrt(1.0 - low * low)]),
        "higher-second": _member(
            "higher-second", [high, np.sqrt(1.0 - high * high)]),
    }
    result = _batch(
        mapping, list(mapping), ([1.0, 0.0],),
        max_chunk_rows=1, max_chunk_bytes=2 * 8 * 3,
    )
    expected = _single(
        mapping, list(mapping), [1.0, 0.0],
        max_chunk_rows=1, max_chunk_bytes=2 * 8 * 3,
    )
    assert result == [expected]
    assert result[0][0] == "higher-second"


def test_public_oversized_multiquery_batch_matches_query_major_reference(monkeypatch):
    import factor_assets.clustering.incremental as incremental
    import factor_assets.clustering.incremental_batch_recall as batch_policy
    from factor_assets.contracts.cluster_governance import (
        ClusterScale, ClusterVersionArtifact,
    )
    from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact

    def fingerprint(fid, vector):
        return SimilarityFingerprintArtifact(
            factor_id=fid, embedding=tuple(vector), embedding_spec="batch-v1",
            snapshot="snapshot", universe="universe", window="window",
            preprocessing_ref="prep", mask_policy="pairwise-valid",
            direction="signed", aggregation_method="daily-then-time",
            embedding_model_version="model", value_ref=f"values:{fid}",
            profile_ref=f"profile:{fid}",
        )

    member_ids = tuple(f"member-{index:04d}" for index in range(600))
    members = {
        fid: fingerprint(fid, (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0))
        for index, fid in enumerate(member_ids)
    }
    queries = [fingerprint(f"query-x-{index}", (1.0, 0.0))
               for index in range(257)]
    queries.append(fingerprint("query-y", (0.0, 1.0)))
    fingerprints = {**members, **{query.factor_id: query for query in queries}}
    clusters = {
        "cluster": ClusterVersionArtifact(
            logical_cluster_id="cluster", cluster_set_version_ref="set",
            member_factor_ids=member_ids, representative_factor_id=member_ids[0],
            scale=ClusterScale.MICRO_CLUSTER,
        )
    }
    monkeypatch.setattr(incremental, "_EXACT_MEMBER_CACHE_BYTES", 1)
    real_prepare = incremental.prepare_oversized_batch_winners
    monkeypatch.setattr(
        incremental, "prepare_oversized_batch_winners",
        lambda *args, **kwargs: {},
    )
    reference = incremental.incremental_assign(queries, clusters, fingerprints)

    monkeypatch.setattr(incremental, "prepare_oversized_batch_winners", real_prepare)
    original_batch_scan = batch_policy.scan_exact_member_winners
    query_block_shapes = []

    def observe_query_blocks(*args, **kwargs):
        query_block_shapes.append(args[2].shape)
        return original_batch_scan(*args, **kwargs)

    query_embeddings = []
    original_to_embedding = incremental._to_embedding

    def count_query_embeddings(fp):
        if fp.factor_id.startswith("query-"):
            query_embeddings.append(fp.factor_id)
        return original_to_embedding(fp)

    normalized_queries = []
    original_normalize = incremental._normalize_rows

    def count_query_normalizations(rows):
        if rows.shape == (1, 2):
            normalized_queries.append(rows.shape)
        return original_normalize(rows)

    monkeypatch.setattr(batch_policy, "scan_exact_member_winners", observe_query_blocks)
    monkeypatch.setattr(incremental, "_to_embedding", count_query_embeddings)
    monkeypatch.setattr(incremental, "_normalize_rows", count_query_normalizations)
    batched = incremental.incremental_assign(queries, clusters, fingerprints)
    monkeypatch.setattr(batch_policy, "MAX_BATCH_PREPARATION_BYTES", 1)
    assert query_embeddings == [query.factor_id for query in queries]
    assert len(normalized_queries) == len(queries)
    query_embeddings.clear()
    normalized_queries.clear()
    capped_fallback = incremental.incremental_assign(queries, clusters, fingerprints)

    assert batched.assignments == reference.assignments
    assert batched.candidates == reference.candidates
    assert capped_fallback.assignments == reference.assignments
    assert capped_fallback.candidates == reference.candidates
    assert query_block_shapes == [(256, 2), (2, 2)]
    assert query_embeddings == [query.factor_id for query in queries]
    assert len(normalized_queries) == len(queries)
    assert [candidate.factor_id for candidate in batched.candidates] == [
        candidate.factor_id for candidate in reference.candidates]
    assert all(candidate.evidence_status.name == "APPROXIMATE"
               for candidate in batched.candidates)
    assert len(batched.candidates) == len(queries)
    assert all(candidate.factor_id == member_ids[0]
               for candidate in batched.candidates[:-1])
    assert batched.candidates[-1].factor_id == member_ids[1]
    assert all(candidate.similarity == 1.0 for candidate in batched.candidates)


def test_public_batch_with_all_members_missing_keeps_queries_pending(monkeypatch):
    import factor_assets.clustering.incremental as incremental
    from factor_assets.contracts.cluster_governance import (
        ClusterScale, ClusterVersionArtifact, IncrementalAssignmentKind,
    )
    from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact

    def fingerprint(fid, vector):
        return SimilarityFingerprintArtifact(
            factor_id=fid, embedding=tuple(vector), embedding_spec="batch-v1",
            snapshot="snapshot", universe="universe", window="window",
            preprocessing_ref="prep", mask_policy="pairwise-valid",
            direction="signed", aggregation_method="daily-then-time",
            embedding_model_version="model", value_ref=f"values:{fid}",
            profile_ref=f"profile:{fid}",
        )

    queries = [
        fingerprint("query-zero-1", (0.0, 0.0)),
        fingerprint("query-zero-2", (0.0, 0.0)),
    ]
    clusters = {
        "cluster": ClusterVersionArtifact(
            logical_cluster_id="cluster", cluster_set_version_ref="set",
            member_factor_ids=("absent-1", "absent-2"),
            representative_factor_id="absent-1", scale=ClusterScale.MICRO_CLUSTER,
        )
    }
    # Query IDs are bound, but no requested member has a stored fingerprint.
    monkeypatch.setattr(incremental, "_EXACT_MEMBER_CACHE_BYTES", 1)
    result = incremental.incremental_assign(
        queries, clusters, {query.factor_id: query for query in queries})

    assert tuple(item.factor_id for item in result.assignments) == (
        "query-zero-1", "query-zero-2")
    assert all(item.kind is IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH
               for item in result.assignments)
