import numpy as np
import pytest

from factor_assets.clustering.families import (
    SCIPY_AVAILABLE,
    HierarchicalClustering,
    hierarchy,
    squareform,
    _correlation_distance_matrix,
)
from factor_assets.errors import InvalidClusteringContract
from factor_assets.graph.sparse import CorrelationEdge, SparseCorrelationGraph


pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")


def _legacy_distance_matrix(graph, factor_ids, missing_distance_policy):
    """Independent copy of the former upper-triangle lookup implementation."""
    distance = np.ones((len(factor_ids), len(factor_ids)))
    for i, factor_a in enumerate(factor_ids):
        distance[i, i] = 0.0
        for j, factor_b in enumerate(factor_ids):
            if i < j:
                correlation = graph.get_correlation(factor_a, factor_b)
                if correlation is None:
                    if missing_distance_policy != "treat_missing_as_max":
                        raise ValueError(
                            f"No observed distance between {factor_a!r} and {factor_b!r}; "
                            "a missing edge is not evidence of low similarity. "
                            "Pass missing_distance_policy='treat_missing_as_max' "
                            "to opt into the research-only behaviour, or supply a "
                            "complete distance artifact."
                        )
                    value = 1.0
                else:
                    value = 1.0 - abs(correlation)
                distance[i, j] = value
                distance[j, i] = value
    return distance


def test_complete_matrix_matches_legacy_and_is_symmetric():
    nodes = ["C", "A", "B"]
    edges = [
        CorrelationEdge("A", "B", -0.9),
        CorrelationEdge("C", "A", 0.2),
        CorrelationEdge("B", "C", -0.5),
        CorrelationEdge("B", "A", -0.9),
    ]
    graph = SparseCorrelationGraph(edges, nodes=nodes)
    factor_ids = sorted(graph.nodes)

    actual = _correlation_distance_matrix(graph, factor_ids, None)
    expected = _legacy_distance_matrix(graph, factor_ids, None)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_array_equal(np.diag(actual), np.zeros(len(nodes)))


def test_first_missing_pair_and_error_text_match_legacy():
    graph = SparseCorrelationGraph(
        [CorrelationEdge("A", "B", 0.5)], nodes=["D", "C", "B", "A"]
    )
    factor_ids = sorted(graph.nodes)
    message = (
        "No observed distance between 'A' and 'C'; a missing edge is not evidence of low similarity. "
        "Pass missing_distance_policy='treat_missing_as_max' to opt into the research-only behaviour, "
        "or supply a complete distance artifact."
    )

    with pytest.raises(ValueError, match="No observed distance between 'A' and 'C'") as old_error:
        _legacy_distance_matrix(graph, factor_ids, None)
    with pytest.raises(ValueError, match="No observed distance between 'A' and 'C'") as new_error:
        _correlation_distance_matrix(graph, factor_ids, None)

    assert str(old_error.value) == message
    assert str(new_error.value) == message


def test_asymmetric_adjacency_uses_ascending_row_and_max_policy():
    graph = SparseCorrelationGraph(
        [CorrelationEdge("A", "B", 0.25)], nodes=["B", "A"]
    )
    graph._adjacency["A"].clear()
    factor_ids = sorted(graph.nodes)

    actual = _correlation_distance_matrix(graph, factor_ids, "treat_missing_as_max")
    expected = _legacy_distance_matrix(graph, factor_ids, "treat_missing_as_max")

    with pytest.raises(ValueError) as old_error:
        _legacy_distance_matrix(graph, factor_ids, None)
    with pytest.raises(ValueError) as new_error:
        _correlation_distance_matrix(graph, factor_ids, None)
    assert str(old_error.value) == str(new_error.value)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] == actual[1, 0] == 1.0


def test_duplicate_row_entries_keep_first_correlation_like_get_correlation():
    graph = SparseCorrelationGraph(
        [CorrelationEdge("A", "B", 0.25)], nodes=["A", "B"]
    )
    graph._adjacency["A"].append(("B", -0.9))

    actual = _correlation_distance_matrix(graph, ["A", "B"], None)
    expected = _legacy_distance_matrix(graph, ["A", "B"], None)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] == pytest.approx(0.75)


def test_present_none_adjacency_value_is_still_treated_as_missing():
    graph = SparseCorrelationGraph(
        [CorrelationEdge("A", "B", 0.25)], nodes=["A", "B"]
    )
    graph._adjacency["A"][0] = ("B", None)
    graph._adjacency["A"].append(("B", 0.9))

    with pytest.raises(ValueError) as old_error:
        _legacy_distance_matrix(graph, ["A", "B"], None)
    with pytest.raises(ValueError) as new_error:
        _correlation_distance_matrix(graph, ["A", "B"], None)
    assert str(old_error.value) == str(new_error.value)
    np.testing.assert_array_equal(
        _correlation_distance_matrix(graph, ["A", "B"], "treat_missing_as_max"),
        _legacy_distance_matrix(graph, ["A", "B"], "treat_missing_as_max"),
    )


def test_overridden_get_correlation_uses_legacy_lookup_semantics():
    class OverrideGraph(SparseCorrelationGraph):
        def get_correlation(self, factor_a, factor_b):
            if {factor_a, factor_b} == {"A", "B"}:
                return -0.125
            return super().get_correlation(factor_a, factor_b)

    graph = OverrideGraph([CorrelationEdge("A", "B", 0.9)])
    factor_ids = sorted(graph.nodes)

    actual = _correlation_distance_matrix(graph, factor_ids, None)
    expected = _legacy_distance_matrix(graph, factor_ids, None)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] == pytest.approx(0.875)


def test_duck_graph_without_neighbors_uses_legacy_lookup_semantics():
    class DuckGraph:
        def get_correlation(self, factor_a, factor_b):
            return 0.75 if (factor_a, factor_b) == ("A", "B") else None

    graph = DuckGraph()
    actual = _correlation_distance_matrix(graph, ["A", "B"], None)
    assert actual[0, 1] == actual[1, 0] == 0.25


def test_public_dendrogram_linkage_matches_legacy_distance_matrix():
    graph = SparseCorrelationGraph([
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("A", "C", -0.4),
        CorrelationEdge("A", "D", 0.2),
        CorrelationEdge("B", "C", 0.7),
        CorrelationEdge("B", "D", -0.3),
        CorrelationEdge("C", "D", 0.8),
    ])
    factor_ids = sorted(graph.nodes)
    expected_distances = _legacy_distance_matrix(graph, factor_ids, None)
    expected_linkage = hierarchy.linkage(
        squareform(expected_distances, checks=False), method="average", metric="euclidean"
    )

    actual = HierarchicalClustering(graph, method="average").build_dendrogram()

    assert actual.factor_ids == factor_ids
    np.testing.assert_array_equal(actual.linkage_matrix, expected_linkage)


def test_constructor_validation_for_policy_and_ward_is_unchanged():
    graph = SparseCorrelationGraph([CorrelationEdge("A", "B", 0.5)])

    with pytest.raises(ValueError, match="missing_distance_policy must be None"):
        HierarchicalClustering(graph, missing_distance_policy="unknown")
    with pytest.raises(InvalidClusteringContract, match="method='ward'"):
        HierarchicalClustering(graph, method="ward")


def test_conflicting_duplicate_edges_still_fail_at_graph_construction():
    with pytest.raises(ValueError, match="Conflicting duplicate edge"):
        SparseCorrelationGraph([
            CorrelationEdge("A", "B", 0.5),
            CorrelationEdge("B", "A", 0.6),
        ])
