"""Parity and compatibility checks for lineage's call-local graph cache."""

from factor_assets.clustering.lineage import LineageDetector
from factor_assets.graph.sparse import CorrelationEdge, SparseCorrelationGraph


def _legacy_relations(detector, subgraph, members):
    """Reference the pre-cache traversal and confidence arithmetic."""
    from factor_assets.clustering.lineage import ParentChildRelation

    relations = []
    seen_pairs = set()
    for factor_a in members:
        for factor_b, corr in subgraph.neighbors(factor_a):
            if factor_b not in members:
                continue
            pair = tuple(sorted([factor_a, factor_b]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if abs(corr) < detector.min_correlation:
                continue
            degree_a = subgraph.degree(factor_a)
            degree_b = subgraph.degree(factor_b)
            if degree_a > degree_b + detector.degree_threshold:
                parent, child = factor_a, factor_b
            elif degree_b > degree_a + detector.degree_threshold:
                parent, child = factor_b, factor_a
            else:
                parent, child = sorted((factor_a, factor_b))
            confidence = detector._calculate_confidence(parent, child, subgraph)
            if confidence >= 0.3:
                relations.append(ParentChildRelation(parent, child, corr, confidence))
    return relations


def test_cached_lineage_matches_legacy_relation_values_and_order():
    edges = [
        CorrelationEdge("a", "b", 0.91), CorrelationEdge("a", "c", -0.83),
        CorrelationEdge("a", "d", 0.76), CorrelationEdge("b", "c", 0.88),
        CorrelationEdge("c", "d", -0.95), CorrelationEdge("x", "y", 0.99),
    ]
    graph = SparseCorrelationGraph(edges, nodes={"isolated"})
    detector = LineageDetector(graph, min_correlation=0.7, degree_threshold=1)
    members = {"a", "b", "c", "d", "isolated"}
    subgraph = graph.subgraph(members)

    assert detector._find_relations(subgraph, members) == _legacy_relations(
        detector, subgraph, members
    )


def test_cached_lineage_reuses_degrees_and_bounds_neighbor_lru(monkeypatch):
    class SmallCacheDetector(LineageDetector):
        _NEIGHBOR_CACHE_MAX_ENTRIES = 3
        _NEIGHBOR_CACHE_MAX_TOTAL_NEIGHBORS = 4

    edges = [
        CorrelationEdge(f"n{i}", f"n{(i + step) % 8}", 0.9)
        for i in range(8)
        for step in (1, 2)
        if i < (i + step) % 8
    ]
    graph = SparseCorrelationGraph(edges)
    calls = {"neighbors": {}, "degree": {}}
    original_neighbors = SparseCorrelationGraph.neighbors
    original_degree = SparseCorrelationGraph.degree

    def count_neighbors(self, node):
        calls["neighbors"][node] = calls["neighbors"].get(node, 0) + 1
        return original_neighbors(self, node)

    def count_degree(self, node):
        calls["degree"][node] = calls["degree"].get(node, 0) + 1
        return original_degree(self, node)

    monkeypatch.setattr(SparseCorrelationGraph, "neighbors", count_neighbors)
    monkeypatch.setattr(SparseCorrelationGraph, "degree", count_degree)
    detector = SmallCacheDetector(graph, min_correlation=0.7)
    members = graph.nodes
    subgraph = graph.subgraph(members)
    expected = _legacy_relations(detector, subgraph, members)
    calls["neighbors"].clear()
    calls["degree"].clear()
    actual = detector._find_relations(subgraph, members)

    assert actual == expected
    assert set(calls["degree"].values()) == {1}
    assert set(calls["neighbors"]) == members
    # Confidence queries revisit evicted nodes, proving the reuse stays bounded.
    assert sum(calls["neighbors"].values()) > len(members)


def test_confidence_override_keeps_original_signature_and_result():
    class FixedConfidenceDetector(LineageDetector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def _calculate_confidence(self, parent, child, subgraph):
            self.calls.append((parent, child, subgraph))
            return 0.42

    graph = SparseCorrelationGraph([
        CorrelationEdge("a", "b", 0.95),
        CorrelationEdge("a", "c", 0.93),
        CorrelationEdge("b", "c", 0.92),
    ])
    detector = FixedConfidenceDetector(graph, min_correlation=0.7)
    lineage = detector.detect_lineage(graph.nodes, 1)

    assert detector.calls
    assert all(call[2] is not None for call in detector.calls)
    assert all(relation.confidence == 0.42 for relation in lineage.relations)


def test_custom_graph_confidence_semantics_are_not_bypassed():
    class CustomGraph(SparseCorrelationGraph):
        def subgraph(self, node_subset):
            retained_edges = [
                edge for edge in self.to_edge_list()
                if edge.factor_a in node_subset and edge.factor_b in node_subset
            ]
            return CustomGraph(retained_edges, nodes=node_subset)

        def get_correlation(self, factor_a, factor_b):
            # Custom graphs may define confidence lookup differently from adjacency.
            return 0.0

    graph = CustomGraph([CorrelationEdge("a", "b", 0.99)])
    detector = LineageDetector(graph, min_correlation=0.7)
    assert detector.detect_lineage(graph.nodes, 2).relations == ()


def test_reverse_duplicate_edges_match_legacy_first_seen_semantics():
    edge_variants = [
        [
            CorrelationEdge("a", "b", 0.92),
            CorrelationEdge("b", "a", 0.92),
            CorrelationEdge("a", "c", 0.88),
        ],
        [
            CorrelationEdge("b", "a", 0.92),
            CorrelationEdge("a", "b", 0.92),
            CorrelationEdge("c", "a", 0.88),
        ],
    ]
    for edges in edge_variants:
        graph = SparseCorrelationGraph(edges)
        detector = LineageDetector(graph, min_correlation=0.7)
        members = {"a", "b", "c"}
        subgraph = graph.subgraph(members)
        assert detector._find_relations(subgraph, members) == _legacy_relations(
            detector, subgraph, members
        )


def test_custom_duplicate_directional_graph_uses_full_legacy_path():
    class DuplicateDirectionalGraph(SparseCorrelationGraph):
        def subgraph(self, node_subset):
            return DuplicateDirectionalGraph(
                [CorrelationEdge("a", "b", 0.9)], nodes=node_subset
            )

        def neighbors(self, factor_id):
            if factor_id == "a":
                return [("b", 0.91), ("b", 0.99)]
            if factor_id == "b":
                return [("a", 0.99), ("a", 0.91)]
            return []

        def get_correlation(self, factor_a, factor_b):
            return 0.61 if factor_a == "a" else 0.97

    graph = DuplicateDirectionalGraph([CorrelationEdge("a", "b", 0.9)])
    detector = LineageDetector(graph, min_correlation=0.7)
    members = {"a", "b"}
    subgraph = graph.subgraph(members)
    actual = detector._find_relations(subgraph, members)
    assert actual == _legacy_relations(detector, subgraph, members)


def test_instance_confidence_override_keeps_legacy_path():
    from types import MethodType

    graph = SparseCorrelationGraph([CorrelationEdge("a", "b", 0.95)])
    detector = LineageDetector(graph, min_correlation=0.7)
    detector._calculate_confidence = MethodType(
        lambda self, parent, child, subgraph: 0.42, detector
    )
    lineage = detector.detect_lineage(graph.nodes, 3)
    assert lineage.relations
    assert all(relation.confidence == 0.42 for relation in lineage.relations)


def test_instance_graph_method_overrides_keep_legacy_semantics():
    from types import MethodType

    cases = (
        ("neighbors", lambda self, node: []),
        ("degree", lambda self, node: 0),
        ("get_correlation", lambda self, a, b: 0.0),
    )
    for method_name, replacement in cases:
        if method_name == "degree":
            graph = SparseCorrelationGraph([
                CorrelationEdge("z", "a", 0.95),
                CorrelationEdge("z", "b", 0.93),
                CorrelationEdge("z", "c", 0.91),
            ])
        else:
            graph = SparseCorrelationGraph([CorrelationEdge("a", "b", 0.95)])
        detector = LineageDetector(graph, min_correlation=0.7, degree_threshold=1)
        members = graph.nodes
        subgraph = graph.subgraph(members)
        setattr(subgraph, method_name, MethodType(replacement, subgraph))

        actual = detector._find_relations(subgraph, members)
        assert actual == _legacy_relations(detector, subgraph, members)
        if method_name in {"neighbors", "get_correlation"}:
            assert actual == []
