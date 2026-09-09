import pytest

from factor_assets.contracts.cluster_governance import (
    ClusterResolutionSelector, ClusterSetVersionArtifact, ClusterVersionArtifact, ClusterVersionMatch,
    ClusterVersionMatcher,
)
from factor_assets.clustering.refresh_service import (
    ClusterRefreshEvidence, ClusterRefreshPolicy, ClusterRefreshRegistry,
    evaluate_cluster_refresh, execute_cluster_refresh,
)


def _v(cid, version, members):
    return ClusterVersionArtifact(cid, version, tuple(members), tuple(members)[0])


def test_t27_split_and_merge_are_real_many_to_many_edges():
    matcher = ClusterVersionMatcher(.5)
    split = matcher.match(
        {"old": _v("old", "v0", "abcd")},
        {"left": _v("left", "v1", "ab"), "right": _v("right", "v1", "cd")},
    )
    assert {(e.to_cluster_id, e.match) for e in split} == {
        ("left", ClusterVersionMatch.SPLIT), ("right", ClusterVersionMatch.SPLIT)
    }
    merged = matcher.match(
        {"left": _v("left", "v0", "ab"), "right": _v("right", "v0", "cd")},
        {"new": _v("new", "v1", "abcd")},
    )
    assert len(merged) == 2
    assert all(e.match is ClusterVersionMatch.MERGED for e in merged)


def test_t28_resolution_requires_stability_and_lower_vi_wins_order_independently():
    selector = ClusterResolutionSelector((.5, 1.0, 1.5))
    with pytest.raises(ValueError, match="stability evidence"):
        selector.select({.5: {"modularity": .9}, 1.0: {"modularity": .1}})
    scores = {1.0: {"vi": .4}, .5: {"vi": .1}, 1.5: {"vi": .8}}
    assert selector.select(scores) == .5
    assert selector.select(dict(reversed(list(scores.items())))) == .5
    with pytest.raises(ValueError, match="outside declared grid"):
        selector.select({2.0: {"vi": .1}})
    with pytest.raises(ValueError):
        selector.select({.5: {"vi": float("inf")}})


def test_n10_cluster_version_binds_existing_refresh_and_downstream_evidence_refs():
    base = dict(
        cluster_set_version_id="csv1", member_universe_ref="u1",
        similarity_graph_version_ref="g1", algorithm="leiden", backend="igraph",
        backend_version="1", seed=1, resolution=1.0, clustering_policy_ref="p1",
        assignment_artifact_ref="a1", created_at="2026-01-01T00:00:00Z",
        refresh_trigger_ref="platform:event:quarterly-or-quality-trigger",
    )
    before = ClusterSetVersionArtifact(**base, downstream_validation_ref="modeling:trial:before")
    after = ClusterSetVersionArtifact(**base, downstream_validation_ref="modeling:trial:after")
    assert before.content_hash != after.content_hash


def test_n10_trigger_executes_existing_cluster_callback_records_lineage_but_not_production():
    evidence = ClusterRefreshEvidence(
        observed_at="2026-04-01T00:00:00Z", days_since_local_review=31,
        days_since_global_review=91, pending_share=.12, migration_share=.02,
        persistent_quality_failures=0, evidence_refs=("registry:snapshot:1",),
    )
    decision = evaluate_cluster_refresh(evidence, ClusterRefreshPolicy("refresh-policy:v1"))
    assert decision.kind == "GLOBAL_REFRESH"
    previous = {"old": _v("old", "v0", "abcd")}
    def existing_cluster_service():
        cluster_set = ClusterSetVersionArtifact(
            "csv2", "u1", "graph2", "leiden", "igraph", "1", 1, 1.0,
            "policy:v1", "assignment:2", "2026-04-01T00:00:00Z",
            refresh_trigger_ref="registry:snapshot:1",
            downstream_validation_ref="modeling:feature_set_trial:pending",
        )
        return cluster_set, {
            "left": _v("left", "csv2", "ab"), "right": _v("right", "csv2", "cd")
        }
    registry = ClusterRefreshRegistry()
    run = execute_cluster_refresh(
        decision=decision, previous_clusters=previous,
        cluster_builder=existing_cluster_service, registry=registry,
        production_cluster_set_ref="csv-production-1",
    )
    assert {edge.match for edge in run.lineage_edges} == {ClusterVersionMatch.SPLIT}
    assert run.status == "RESEARCH_CANDIDATE_NOT_PRODUCTION"
    assert run.production_cluster_set_ref == "csv-production-1"
    assert registry.runs == (run,)
