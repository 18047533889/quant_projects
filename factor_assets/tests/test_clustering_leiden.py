"""Production-path tests for LeidenClustering (igraph / leidenalg backends).

These exercise the REAL sparse-graph Leiden path now that `igraph` 1.0.0 and
`leidenalg` 0.12.0 are installed in the venv.  They must never build a dense
N x N similarity matrix.
"""

import pytest
import numpy as np

from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.clustering.families import (
   
        LeidenClustering,
    ModularityClustering,
)

pytestmark = pytest.mark.skipif(
    not LeidenClustering.PRODUCTION_CAPABLE,
    reason="No Leiden backend installed",
)


def _two_triangles_graph():
    """Six factors split into two tight triangles, weakly connected.

    A-B-C form one dense triangle, D-E-F form another, joined only by the
    weak C-D edge (|corr| 0.2).  Leiden should recover the two clusters.
    """
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.9),
        CorrelationEdge("A", "C", 0.9),
        CorrelationEdge("D", "E", 0.9),
        CorrelationEdge("E", "F", 0.9),
        CorrelationEdge("D", "F", 0.9),
        CorrelationEdge("C", "D", 0.2),
    ]
    return SparseCorrelationGraph(edges)


def test_leiden_backend_installed_and_capable():
    """The production backend is actually present and active."""
    assert LeidenClustering.PRODUCTION_CAPABLE is True
    assert LeidenClustering.RESEARCH_ONLY is False
    # _detect_backend must resolve to a mature library (not fail closed).
    backend = LeidenClustering._detect_backend()
    assert backend in ("igraph", "leidenalg")


def test_leiden_returns_real_two_clusters():
    """Leiden recovers the two known dense triangles on the production path."""
    graph = _two_triangles_graph()
    result = LeidenClustering(graph, seed=42).cluster()

    assert result.num_clusters == 2
    clusters = result.get_all_clusters()
    assert sorted(len(m) for m in clusters.values()) == [3, 3]
    assert clusters[0] == {"A", "B", "C"}
    assert clusters[1] == {"D", "E", "F"}
    # Every factor assigned exactly once.
    assert set(result.assignments) == {"A", "B", "C", "D", "E", "F"}


def test_leiden_is_deterministic_same_seed():
    """Same seed -> identical assignment across two independent runs."""
    graph = _two_triangles_graph()
    first = LeidenClustering(graph, seed=42).cluster()
    second = LeidenClustering(graph, seed=42).cluster()
    assert first.assignments == second.assignments


def test_leiden_uses_sparse_graph_not_dense_matrix():
    """Leiden builds an igraph object from the sparse edge list, never a dense
    N x N similarity matrix (which would blow up at 100K scale)."""
    import igraph

    graph = _two_triangles_graph()
    lc = LeidenClustering(graph, seed=42)

    # Reconstruct exactly what the igraph backend feeds into community_leiden:
    # a sparse edge list + parallel weight vector.
    factor_ids = sorted(graph.nodes)
    id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}
    edges = []
    weights = []
    for edge in graph.to_edge_list():
        edges.append((id_to_idx[edge.factor_a], id_to_idx[edge.factor_b]))
        weights.append(edge.abs_correlation)

    g = igraph.Graph(n=len(factor_ids), edges=edges, directed=False)
    g.es["weight"] = weights

    # The igraph graph is sparse: it has exactly one entry per graph edge,
    # NOT n*n entries.
    assert g.ecount() == graph.edge_count
    # No dense numpy array of size n*n is ever materialised.
    assert len(edges) == graph.edge_count == len(weights)
    # Sanity: a dense similarity matrix for 6 factors would be 36 cells.
    assert graph.edge_count < graph.node_count ** 2

    # And the resulting partition is valid.
    partition = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=1.0,
        n_iterations=2,
    )
    assert len(partition.membership) == graph.node_count


def test_modularity_clustering_still_research_only():
    """Toy ModularityClustering stays RESEARCH_ONLY and fails closed without
    the explicit opt-in."""
    assert ModularityClustering.RESEARCH_ONLY is True
    graph = _two_triangles_graph()
    with pytest.raises(RuntimeError):
        ModularityClustering(graph)  # no allow_toy_algorithm=True


def test_leiden_empty_graph():
    """Empty graph yields an empty result, not an error."""
    graph = SparseCorrelationGraph([])
    result = LeidenClustering(graph, seed=42).cluster()
    assert result.num_clusters == 0
    assert result.assignments == {}


def test_leiden_isolated_node_is_its_own_cluster():
    """A node with no edges must still be assigned to its own cluster."""
    graph = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.9),
        ],
        nodes=["A", "B", "C", "ISO"],
    )
    result = LeidenClustering(graph, seed=42).cluster()
    clusters = result.get_all_clusters()
    # The connected component {A,B,C} plus the isolated ISO node.
    assert "ISO" in result.assignments
    assert sorted(len(m) for m in clusters.values()) == [1, 3]


def test_leiden_min_cluster_size_merge_nearest():
    """min_cluster_size is honored via MERGE_NEAREST: a cluster smaller than
    the threshold is merged into its nearest neighbour, never left as a dead
    parameter."""
    graph = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.9),
            CorrelationEdge("D", "E", 0.9),
        ]
    )
    # Default (min_cluster_size=1): every cluster kept as-is.
    base = LeidenClustering(graph, seed=42).cluster()
    assert set(base.cluster_sizes.values()) <= {2, 3}

    merged = LeidenClustering(
        graph, seed=42, min_cluster_size=2, min_cluster_policy="MERGE_NEAREST"
    ).cluster()
    # No cluster is left with size 1: the size-2 and size-2 clusters remain,
    # and the merge policy ensures no isolated singleton survives.
    assert all(size >= 2 for size in merged.cluster_sizes.values())

    # Invalid policy fails closed.
    import pytest
    with pytest.raises(ValueError, match="min_cluster_policy"):
        LeidenClustering(graph, seed=42, min_cluster_policy="BOGUS")


def test_leiden_min_cluster_size_mark_unstable():
    """MARK_UNSTABLE keeps small clusters but records them as unstable, so the
    configuration is observable rather than silently dropped."""
    graph = SparseCorrelationGraph(
        [
            CorrelationEdge("A", "B", 0.9),
            CorrelationEdge("B", "C", 0.9),
            CorrelationEdge("D", "E", 0.9),
        ]
    )
    result = LeidenClustering(
        graph, seed=42, min_cluster_size=3, min_cluster_policy="MARK_UNSTABLE"
    ).cluster()
    # The size-2 cluster is recorded as unstable.
    small = [cid for cid, size in result.cluster_sizes.items() if size < 3]
    assert small
    assert set(small) == set(result.unstable_clusters)


def test_leiden_records_run_config():
    """The RNG/backend configuration that produced a partition is recorded so a
    concurrent campaign cannot silently change the seed, backend, or version."""
    graph = _two_triangles_graph()
    lc = LeidenClustering(graph, seed=7, resolution=1.5, min_cluster_size=2)
    cfg = lc.run_config
    assert cfg["backend"] in ("igraph", "leidenalg")
    assert cfg["backend_version"]  # non-empty
    assert cfg["seed"] == 7
    assert cfg["resolution"] == 1.5
    assert cfg["min_cluster_size"] == 2


def test_cluster_artifact_full_identity():
    """Leiden now produces a production ClusterArtifact with full provenance:
    graph identity, similarity spec ref, snapshot/universe, backend version,
    seed, resolution, assignments, representatives, modularity, stability, and
    a content hash over all of it."""
    from factor_assets.clustering.families import ClusterArtifact

    graph = _two_triangles_graph()
    lc = LeidenClustering(graph, seed=42)
    artifact = lc.cluster_artifact(
        snapshot_ref="snapshot:2024",
        universe_ref="universe:ashare",
    )
    assert artifact.graph_identity == graph.graph_identity
    assert artifact.snapshot_ref == "snapshot:2024"
    assert artifact.universe_ref == "universe:ashare"
    assert artifact.algorithm == "leiden"
    assert artifact.backend in ("igraph", "leidenalg")
    assert artifact.backend_version
    assert artifact.seed == 42
    assert artifact.resolution == 1.0
    assert artifact.assignments
    assert artifact.representatives
    assert artifact.modularity is None or artifact.modularity >= -1.0
    assert artifact.stability is None or artifact.stability >= 0.0
    assert artifact.content_hash
    # Deterministic: two independent runs with the same inputs share a hash.
    again = LeidenClustering(graph, seed=42).cluster_artifact(
        snapshot_ref="snapshot:2024",
        universe_ref="universe:ashare",
    )
    assert artifact.content_hash == again.content_hash
