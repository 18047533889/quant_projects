import pytest
from factor_assets.clustering.families import (
    ClusterQualityPolicy, LeidenClustering, _cluster_quality, _leiden_finalize,
)
from factor_assets.clustering.certification import ExecutionMode
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.errors import InvalidClusteringContract


POLICY = ClusterQualityPolicy(.8, 'fixture-medoid-floor', '1')


def graph(rows):
    return SparseCorrelationGraph([CorrelationEdge(*row) for row in rows])


def test_balanced_negative_direction_is_not_a_conflict():
    g = graph([('a','b',-.95), ('a','c',.91), ('b','c',-.90)])
    result = _cluster_quality(g, dict.fromkeys('abc',0), POLICY)[0]
    assert result['status'] == 'PASS'
    assert result['direction_conflict_edges'] == ()
    assert result['medoid'] == 'a'


def test_inconsistent_signed_triangle_fails():
    g = graph([('a','b',-.95), ('a','c',-.91), ('b','c',-.90)])
    result = _cluster_quality(g, dict.fromkeys('abc',0), POLICY)[0]
    assert result['status'] == 'FAIL'
    assert result['direction_conflict_edges']


def test_quality_is_invariant_to_graph_edge_order():
    rows = [('a','b',-.95), ('a','c',-.91), ('b','c',-.90)]
    first = _cluster_quality(graph(rows), dict.fromkeys('abc',0), POLICY)
    second = _cluster_quality(graph(rows[::-1]), dict.fromkeys('cba',0), POLICY)
    assert first == second


def test_bridge_connectivity_is_not_cluster_quality_and_missing_is_not_zero():
    g = graph([('a','b',.95), ('b','c',.2), ('c','d',.95)])
    result = _cluster_quality(g, dict.fromkeys('abcd',0), POLICY)[0]
    assert result['high_affinity_components'] == 2
    assert result['status'] == 'INSUFFICIENT'
    assert result['observed_medoid_pairs'] < result['required_medoid_pairs']
    assert result['min_observed_medoid_affinity'] == .2


def test_actual_finalize_production_gate_rejects_uncalibrated_quality():
    g = graph([('a','b',.95)])
    runner = LeidenClustering(g)
    # Exercise the real finalization gate independently of the graph-entry
    # certification gate (which has its own adversarial test suite).
    runner.execution_mode = ExecutionMode.PRODUCTION
    with pytest.raises(InvalidClusteringContract, match='quality gate'):
        _leiden_finalize(runner, {'a':0,'b':0})


def test_real_leiden_public_artifact_carries_medoid_evidence_and_hash():
    pytest.importorskip('igraph')
    g = graph([('a','b',-.95), ('a','c',.91), ('b','c',-.90)])
    runner = LeidenClustering(g, resolution=.1, quality_policy=POLICY)
    artifact = runner.cluster_artifact()
    assert artifact.num_clusters == 1
    assert artifact.representatives == ('a',)
    assert artifact.quality_evidence[0]['status'] == 'PASS'
    assert artifact.to_dict()['quality_evidence'][0]['policy_id'] == POLICY.policy_id
    import json
    from factor_assets.clustering.families import ClusterArtifact
    restored = ClusterArtifact(**json.loads(json.dumps(artifact.to_dict())))
    assert restored.content_hash == artifact.content_hash
    from dataclasses import replace
    changed = replace(artifact, quality_evidence={}, content_hash='')
    assert changed.content_hash != artifact.content_hash
    with pytest.raises(TypeError):
        artifact.quality_evidence[0]['status'] = 'PASS'


def test_real_production_public_gate_requires_quality_in_addition_to_graph_certificate():
    pytest.importorskip('igraph')
    from factor_assets.clustering.certification import (
        CertifiedGraphArtifact, GraphCertificationStatus, graph_summary, compute_graph_content_hash,
    )
    g = graph([('a','b',-.95), ('a','c',-.91), ('b','c',-.90)])
    nodes, edges = graph_summary(g)
    cert = CertifiedGraphArtifact(certification_id='fixture-certified-graph',
        graph_content_hash=compute_graph_content_hash(node_universe=nodes, edge_list=edges),
        node_count=len(nodes), edge_count=len(edges),
        graph_completeness_class='CERTIFIED_ANN_REFINED', allowed_algorithms=('leiden',),
        status=GraphCertificationStatus.CERTIFIED, certified_at='2026-01-01T00:00:00Z',
        frozen_until='2099-01-01T00:00:00Z', certified_by='fixture', evidence_refs=('fixture:evidence',))
    runner = LeidenClustering(g, resolution=.1, quality_policy=POLICY,
                              execution_mode='PRODUCTION', certification=cert)
    with pytest.raises(InvalidClusteringContract, match='quality gate'):
        runner.cluster_artifact()
