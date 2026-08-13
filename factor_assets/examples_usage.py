"""
Usage examples for factor_assets clustering and similarity.

Demonstrates ANN search, QE-based similarity, and hierarchical clustering.
"""

import sys
sys.path.insert(0, '/home/shw/quant_projects')

import numpy as np
from factor_assets.similarity import (
    QEPairwiseSimilarity,
    SimilarityMethod,
    ANN_AVAILABLE,
)
from factor_assets.clustering import (
    HierarchicalClustering,
    HIERARCHICAL_AVAILABLE,
)
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge


def example_qe_similarity():
    """Example: Using QE-based pairwise similarity."""
    print("\n=== QE Pairwise Similarity Example ===")

    # Create similarity provider (stub mode, no real QE adapter)
    similarity = QEPairwiseSimilarity()

    # In production, would inject real QE adapter:
    # similarity = QEPairwiseSimilarity(qe_adapter=my_qe_adapter)

    # Compute similarity would delegate to QE
    result = similarity.compute_similarity(
        factor_id_a="momentum_20d",
        factor_id_b="momentum_60d",
        method=SimilarityMethod.PEARSON,
        universe_ref="US_LARGE_CAP",
        period_start="2024-01-01",
        period_end="2024-12-31",
    )

    if result:
        print(f"Similarity: {result.similarity_score}")
    else:
        print("No result (QE adapter not configured in stub mode)")

    print("✓ QE similarity interface ready for production adapter")


def example_ann_search():
    """Example: Using ANN for fast similarity search."""
    print("\n=== ANN Fast Similarity Search Example ===")

    try:
        from factor_assets.similarity.ann import (
            FaissANNIndex,
            ANNBackend,
            create_ann_index,
        )
    except ImportError:
        print("⚠ ANN module not available")
        return

    if not ANN_AVAILABLE:
        print("⚠ ANN backends (faiss/annoy) not available")
        print("  Install with: pip install faiss-cpu annoy")
        return

    # Create embeddings for factors (in production, from evidence)
    factor_ids = ["momentum", "reversal", "value", "quality", "size"]
    embeddings = np.random.randn(5, 64)  # 64-dim embeddings

    # Build FAISS index
    try:
        index = create_ann_index(ANNBackend.FAISS, embedding_dim=64, normalize=True)
        index.build(factor_ids, embeddings)

        print(f"✓ Built index with {index.num_factors} factors")

        # Search for similar factors
        query = embeddings[0]  # momentum
        results = index.search(query, k=3, min_similarity=0.5)

        print(f"✓ Found {len(results)} similar factors:")
        for r in results:
            print(f"  - {r.factor_id}: similarity={r.similarity_score:.3f}")

        # Search by factor ID
        results = index.search_by_id("momentum", k=2)
        print(f"✓ Neighbors of 'momentum': {[r.factor_id for r in results]}")
    except ImportError as e:
        print(f"⚠ Cannot build index: {e}")
        print("  This is expected if faiss-cpu is not installed")


def example_hierarchical_clustering():
    """Example: Hierarchical clustering with dendrogram cutting."""
    print("\n=== Hierarchical Clustering Example ===")

    if not HIERARCHICAL_AVAILABLE:
        print("⚠ scipy not available for hierarchical clustering")
        return

    # Create correlation graph
    edges = [
        # Momentum family
        CorrelationEdge("mom_20d", "mom_60d", 0.85),
        CorrelationEdge("mom_60d", "mom_120d", 0.80),
        CorrelationEdge("mom_20d", "mom_120d", 0.70),

        # Value family
        CorrelationEdge("pb_ratio", "pe_ratio", 0.75),
        CorrelationEdge("pe_ratio", "pcf_ratio", 0.80),
        CorrelationEdge("pb_ratio", "pcf_ratio", 0.70),

        # Weak cross-family connection
        CorrelationEdge("mom_120d", "pb_ratio", 0.15),
    ]

    graph = SparseCorrelationGraph(edges)
    print(f"✓ Created graph with {graph.node_count} nodes, {graph.edge_count} edges")

    # Build hierarchical clustering
    clustering = HierarchicalClustering(graph, method='average')
    dendrogram = clustering.build_dendrogram()

    print(f"✓ Built dendrogram with {len(dendrogram.factor_ids)} factors")

    # Cut at specific number of clusters
    result = dendrogram.cut_at_num_clusters(2)
    print(f"✓ Cut into {result.num_clusters} clusters:")

    for cluster_id, size in result.cluster_sizes.items():
        factors = [f for f, c in result.assignments.items() if c == cluster_id]
        print(f"  Cluster {cluster_id}: {size} factors - {factors}")

    # Auto-optimize cluster count
    optimal_k = dendrogram.get_optimal_num_clusters(min_clusters=2, max_clusters=4)
    print(f"✓ Optimal number of clusters: {optimal_k}")

    result_auto = dendrogram.cut_at_num_clusters(optimal_k)
    print(f"✓ Auto-optimized clustering: {result_auto.num_clusters} clusters")


def example_integration():
    """Example: Integrating similarity and clustering."""
    print("\n=== Integration Example ===")

    # 1. Use QE to compute pairwise similarities
    print("1. Compute pairwise similarities via QE")
    qe_sim = QEPairwiseSimilarity()
    print("   ✓ Ready to compute correlations through QE adapter")

    # 2. Build correlation graph from similarities
    print("2. Build correlation graph from pairwise results")
    edges = [
        CorrelationEdge("F1", "F2", 0.9),
        CorrelationEdge("F2", "F3", 0.8),
        CorrelationEdge("F4", "F5", 0.85),
    ]
    graph = SparseCorrelationGraph(edges)
    print(f"   ✓ Graph with {graph.node_count} factors")

    # 3. Cluster to find families
    if HIERARCHICAL_AVAILABLE:
        print("3. Hierarchical clustering to identify families")
        clustering = HierarchicalClustering(graph)
        result = clustering.cluster(auto_optimize=True)
        print(f"   ✓ Found {result.num_clusters} factor families")

    # 4. For fast search, build ANN index from embeddings
    try:
        from factor_assets.similarity.ann import FaissANNIndex
        print("4. Build ANN index for fast neighbor search")

        # In production: embeddings derived from factor evidence
        embeddings = np.random.randn(5, 32)
        factor_ids = ["F1", "F2", "F3", "F4", "F5"]

        index = FaissANNIndex(embedding_dim=32)
        index.build(factor_ids, embeddings)
        print(f"   ✓ ANN index with {index.num_factors} factors ready")
    except ImportError:
        print("4. ANN index (skipped - faiss not installed)")


if __name__ == "__main__":
    print("Factor Assets - Clustering & Similarity Examples")
    print("=" * 50)

    example_qe_similarity()
    example_ann_search()
    example_hierarchical_clustering()
    example_integration()

    print("\n" + "=" * 50)
    print("✓ All examples completed")
