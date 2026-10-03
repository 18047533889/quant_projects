"""Independent regressions for bounded incremental batch recall."""
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

import factor_assets.clustering.incremental as incremental
from factor_assets.clustering.incremental_recall import RequestLocalMemberMatrixCache
from factor_assets.contracts._canonical import canonical_digest
from factor_assets.contracts.cluster_governance import (
    ClusterScale,
    ClusterVersionArtifact,
)
from factor_assets.contracts.fingerprint import (
    ANNIndexArtifact,
    ANNIndexCapability,
    SimilarityFingerprintArtifact,
)


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
        representative_factor_id=members[0] if members else "",
        scale=ClusterScale.MICRO_CLUSTER,
    )


def _ann_artifact(mapping, members):
    ordered = tuple(sorted(members))
    member_hash = canonical_digest(tuple(
        (fid, mapping[fid].content_hash) for fid in ordered
    ))
    domain_hash = canonical_digest(incremental._fingerprint_domain(mapping[ordered[0]]))
    return ANNIndexArtifact(
        index_id="test-index",
        backend="annoy",
        capability=ANNIndexCapability.APPROXIMATE,
        index_params={"embedding_dim": len(mapping[ordered[0]].embedding)},
        member_set_hash=member_hash,
        embedding_spec_hash=domain_hash,
    )


def test_exact_batch_collects_and_normalizes_each_cluster_once(monkeypatch):
    members = {
        "m1": _fp("m1", (1.0, 0.0)),
        "m2": _fp("m2", (0.0, 1.0)),
    }
    queries = [_fp("q1", (0.9, 0.1)), _fp("q2", (0.1, 0.9))]
    mapping = {**members, **{query.factor_id: query for query in queries}}
    clusters = {"A": _cluster("A", ("m1",)), "B": _cluster("B", ("m2",))}

    collect_calls = []
    prepare_calls = []
    domain_calls = []
    original_collect = incremental._collect_member_embeddings
    original_prepare = incremental._prepare_unit_members
    original_domain = incremental._fingerprint_domain

    def collect(fingerprint_map, ids):
        collect_calls.append(tuple(ids))
        return original_collect(fingerprint_map, ids)

    def prepare(fingerprint_map, ids):
        prepare_calls.append(tuple(ids))
        return original_prepare(fingerprint_map, ids)

    def domain(fp):
        domain_calls.append(fp.factor_id)
        return original_domain(fp)

    monkeypatch.setattr(incremental, "_collect_member_embeddings", collect)
    monkeypatch.setattr(incremental, "_prepare_unit_members", prepare)
    monkeypatch.setattr(incremental, "_fingerprint_domain", domain)

    result = incremental.incremental_assign(queries, clusters, mapping)

    assert collect_calls == [("m1",), ("m2",)]
    assert prepare_calls == [("m1",), ("m2",)]
    assert len(domain_calls) == len(queries) + len(members)
    assert tuple(a.factor_id for a in result.assignments) == ("q1", "q2")
    assert result.assignments[0].logical_cluster_id == "A"
    assert result.assignments[1].logical_cluster_id == "B"


def test_request_cache_lru_evicts_and_oversized_or_empty_results_are_not_retained():
    cache = RequestLocalMemberMatrixCache(max_bytes=600)
    loads = {"a": 0, "b": 0}

    def loader(key):
        def load():
            loads[key] += 1
            return [key], np.ones((1, 2), dtype=np.float64)
        return load

    cache.get("a", loader("a"))
    cache.get("b", loader("b"))
    assert cache.retained_bytes <= 600
    cache.get("a", loader("a"))
    assert loads == {"a": 2, "b": 1}
    assert cache.retained_bytes <= 600

    oversized = RequestLocalMemberMatrixCache(max_bytes=100)
    oversized_loads = []
    def load_large():
        oversized_loads.append(1)
        return ["large"], np.ones((1, 2), dtype=np.float64)
    oversized.get("large", load_large)
    oversized.get("large", load_large)
    assert len(oversized_loads) == 2
    assert oversized.retained_bytes == 0

    empty = RequestLocalMemberMatrixCache(max_bytes=100)
    assert empty.get("empty", lambda: ([], None)) == ([], None)
    assert empty.retained_bytes == 0


def test_exact_cosine_handles_extreme_finite_scales_against_decimal_oracle():
    query = _fp("q", (1e308, 0.0))
    members = [
        _fp("large", (1e308, 1e308)),
        _fp("tiny", (1e-300, 1e-300)),
        _fp("orthogonal", (0.0, 1e-300)),
    ]
    mapping = {fp.factor_id: fp for fp in [query, *members]}
    matrix = np.vstack([incremental._to_embedding(fp) for fp in members])
    scores = incremental._cosine_scores(incremental._to_embedding(query), matrix)
    by_id = dict(zip((member.factor_id for member in members), scores))

    def decimal_cosine(left, right):
        with localcontext() as ctx:
            ctx.prec = 1000
            a = [Decimal(str(value)) for value in left]
            b = [Decimal(str(value)) for value in right]
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a).sqrt()
            norm_b = sum(y * y for y in b).sqrt()
            return float(dot / (norm_a * norm_b))

    for member in members:
        expected = decimal_cosine(query.embedding, member.embedding)
        assert by_id[member.factor_id] == pytest.approx(expected, abs=2e-15)
    assert by_id["large"] == pytest.approx(2**-0.5)
    assert by_id["tiny"] == pytest.approx(2**-0.5)
    assert by_id["orthogonal"] == pytest.approx(0.0, abs=1e-15)


def test_ann_search_failure_falls_back_exact_for_that_query_only(monkeypatch):
    members = {
        "m1": _fp("m1", (1.0, 0.0)),
        "m2": _fp("m2", (0.0, 1.0)),
    }
    queries = [_fp("q1", (0.9, 0.1)), _fp("q2", (0.1, 0.9))]
    mapping = {**members, **{query.factor_id: query for query in queries}}
    ann = _ann_artifact(mapping, members)

    class FlakyIndex:
        def __init__(self, **_kwargs):
            self.calls = 0

        def build(self, _ids, _matrix):
            return None

        def search(self, _query, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient ANN query failure")
            return [SimpleNamespace(factor_id="m1", similarity_score=0.99)]

    monkeypatch.setattr(incremental, "AnnoyANNIndex", FlakyIndex)
    result = incremental.incremental_assign(
        queries, {"A": _cluster("A", ("m1",)), "B": _cluster("B", ("m2",))},
        mapping, ann_index=ann,
    )
    assert len(result.candidates) == 3
    assert result.candidates[0].factor_id == "m1"
    assert result.candidates[0].similarity == pytest.approx(0.99)
    assert tuple(item.factor_id for item in result.candidates[1:]) == ("m1", "m2")
    assert result.candidates[2].similarity > result.candidates[1].similarity


def test_malformed_ann_result_remains_fail_closed(monkeypatch):
    member = _fp("m", (1.0, 0.0))
    query = _fp("q", (0.8, 0.2))
    mapping = {"m": member, "q": query}
    ann = _ann_artifact(mapping, ("m",))

    class MalformedIndex:
        def __init__(self, **_kwargs):
            pass

        def build(self, _ids, _matrix):
            return None

        def search(self, _query, **_kwargs):
            return [SimpleNamespace(factor_id="unknown-member", similarity_score=0.9)]

    monkeypatch.setattr(incremental, "AnnoyANNIndex", MalformedIndex)
    with pytest.raises(KeyError, match="unknown-member"):
        incremental.incremental_assign(
            [query], {"A": _cluster("A", ("m",))}, mapping, ann_index=ann,
        )



def test_exact_self_and_opposite_cosine_scores_respect_closed_interval():
    query = _fp("q", (1.0, 0.0))
    same = _fp("same", (1.0, 0.0))
    opposite = _fp("opposite", (-1.0, 0.0))
    mapping = {fp.factor_id: fp for fp in (query, same, opposite)}

    result = incremental.incremental_assign(
        [query],
        {"same-cluster": _cluster("same-cluster", ("same",)),
         "opposite-cluster": _cluster("opposite-cluster", ("opposite",))},
        mapping,
    )

    by_id = {candidate.factor_id: candidate for candidate in result.candidates}
    assert by_id["same"].similarity == 1.0
    assert by_id["opposite"].similarity == -1.0


def test_unit_cosine_rejects_scores_outside_float64_roundoff_envelope():
    with pytest.raises(ValueError, match="Float64 envelope"):
        incremental.unit_cosine_scores(
            np.asarray([1.0]), np.asarray([[2.0]]),
        )
