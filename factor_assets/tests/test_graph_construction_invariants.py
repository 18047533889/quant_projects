"""Adversarial invariants for sparse correlation graph construction."""

import pytest

from factor_assets.graph.sparse import CorrelationEdge, SparseCorrelationGraph


def test_constructs_100_nodes_including_10_isolated_nodes():
    nodes = {f"F{i:03d}" for i in range(100)}
    connected = sorted(nodes)[:90]
    edges = [
        CorrelationEdge(connected[index], connected[index + 1], 0.5)
        for index in range(0, len(connected), 2)
    ]

    graph = SparseCorrelationGraph(edges, nodes=nodes)

    isolated = nodes - set(connected)
    assert graph.node_count == 100
    assert graph.nodes == nodes
    assert len(isolated) == 10
    assert all(graph.degree(node) == 0 for node in isolated)


def test_exact_reverse_duplicate_is_deduplicated():
    graph = SparseCorrelationGraph([
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "A", 0.8),
    ])

    assert graph.edge_count == 1
    assert graph.get_correlation("A", "B") == 0.8


def test_conflicting_reverse_duplicate_fails_closed():
    with pytest.raises(ValueError, match="Conflicting duplicate edge"):
        SparseCorrelationGraph([
            CorrelationEdge("A", "B", 0.8),
            CorrelationEdge("B", "A", 0.7),
        ])


def test_self_edge_policy_remains_fail_closed():
    with pytest.raises(ValueError, match="Self-loops not allowed"):
        CorrelationEdge("A", "A", 1.0)


def test_subgraph_conserves_requested_existing_nodes_without_edges():
    graph = SparseCorrelationGraph(
        [CorrelationEdge("A", "B", 0.8)],
        nodes={"A", "B", "C", "D"},
    )

    subgraph = graph.subgraph({"B", "C", "D", "missing"})

    assert subgraph.nodes == {"B", "C", "D"}
    assert subgraph.node_count == 3
    assert subgraph.edge_count == 0
