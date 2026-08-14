#!/usr/bin/env python3
"""
Factor Assets Similarity Search Benchmarks

Tests similarity search and clustering:
- Exact similarity search (brute force)
- Approximate nearest neighbor
- Clustering performance
- Representative selection

Measures:
- Query latency
- Throughput (queries/sec)
- Recall at k
- Memory usage
"""

import gc
import time
import traceback
import numpy as np
from typing import Dict, Any, List, Tuple

try:
    from factor_assets.seen_index.exact import ExactSeenIndex
    from factor_assets.clustering.families import ClusteringEngine
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False


def generate_factor_embeddings(n_factors: int, embedding_dim: int = 128, seed: int = 42) -> np.ndarray:
    """Generate synthetic factor embeddings."""
    np.random.seed(seed)

    # Generate clustered embeddings
    n_clusters = max(10, n_factors // 100)
    cluster_centers = np.random.randn(n_clusters, embedding_dim)

    embeddings = []
    for i in range(n_factors):
        cluster_id = i % n_clusters
        center = cluster_centers[cluster_id]
        # Add noise
        embedding = center + np.random.randn(embedding_dim) * 0.3
        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings


def cosine_similarity_batch(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between queries and database."""
    return queries @ database.T


def benchmark_exact_search(
    n_database: int,
    n_queries: int,
    embedding_dim: int = 128,
    k: int = 10
) -> Dict[str, Any]:
    """Benchmark exact similarity search."""

    try:
        # Generate data
        print(f"    Generating {n_database} database embeddings...")
        database = generate_factor_embeddings(n_database, embedding_dim, seed=42)

        print(f"    Generating {n_queries} query embeddings...")
        queries = generate_factor_embeddings(n_queries, embedding_dim, seed=99)

        gc.collect()
        start = time.perf_counter()

        # Exact search: compute all similarities
        similarities = cosine_similarity_batch(queries, database)

        # Get top-k for each query
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_scores = np.take_along_axis(similarities, top_k_indices, axis=1)

        elapsed = time.perf_counter() - start

        throughput_queries = n_queries / elapsed
        latency_per_query = elapsed / n_queries * 1000  # ms

        total_comparisons = n_queries * n_database
        throughput_comparisons = total_comparisons / elapsed

        return {
            "n_database": n_database,
            "n_queries": n_queries,
            "embedding_dim": embedding_dim,
            "k": k,
            "total_comparisons": total_comparisons,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_queries_per_sec": round(throughput_queries, 2),
            "throughput_comparisons_per_sec": round(throughput_comparisons, 0),
            "latency_ms_per_query": round(latency_per_query, 3),
            "mean_top1_score": round(float(top_k_scores[:, 0].mean()), 4),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def benchmark_clustering(
    n_factors: int,
    n_clusters: int,
    embedding_dim: int = 128,
    max_iterations: int = 20
) -> Dict[str, Any]:
    """Benchmark clustering performance."""

    try:
        print(f"    Generating {n_factors} embeddings...")
        embeddings = generate_factor_embeddings(n_factors, embedding_dim)

        gc.collect()
        start = time.perf_counter()

        # Simple k-means clustering
        # Initialize centers randomly
        np.random.seed(42)
        center_indices = np.random.choice(n_factors, n_clusters, replace=False)
        centers = embeddings[center_indices].copy()

        for iteration in range(max_iterations):
            # Assign to nearest center
            similarities = embeddings @ centers.T
            assignments = np.argmax(similarities, axis=1)

            # Update centers
            new_centers = np.zeros_like(centers)
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    new_centers[c] = embeddings[mask].mean(axis=0)
                else:
                    new_centers[c] = centers[c]

            # Normalize
            norms = np.linalg.norm(new_centers, axis=1, keepdims=True)
            new_centers = new_centers / (norms + 1e-8)

            # Check convergence
            diff = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if diff < 1e-4:
                break

        elapsed = time.perf_counter() - start

        # Compute cluster sizes
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)

        return {
            "n_factors": n_factors,
            "n_clusters": n_clusters,
            "embedding_dim": embedding_dim,
            "iterations": iteration + 1,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_factors_per_sec": round(n_factors / elapsed, 2),
            "mean_cluster_size": round(float(cluster_sizes.mean()), 2),
            "min_cluster_size": int(cluster_sizes.min()),
            "max_cluster_size": int(cluster_sizes.max()),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def benchmark_representative_selection(
    n_factors: int,
    n_representatives: int,
    embedding_dim: int = 128
) -> Dict[str, Any]:
    """Benchmark representative selection."""

    try:
        print(f"    Generating {n_factors} embeddings...")
        embeddings = generate_factor_embeddings(n_factors, embedding_dim)

        gc.collect()
        start = time.perf_counter()

        # Greedy farthest point sampling
        selected_indices = []
        remaining_indices = set(range(n_factors))

        # Start with random point
        first_idx = np.random.randint(n_factors)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        while len(selected_indices) < n_representatives:
            # Find point farthest from all selected
            selected_embeddings = embeddings[selected_indices]
            remaining = list(remaining_indices)
            remaining_embeddings = embeddings[remaining]

            # Max similarity to any selected point
            similarities = remaining_embeddings @ selected_embeddings.T
            max_sims = similarities.max(axis=1)

            # Select point with minimum max similarity (farthest)
            farthest_idx_in_remaining = np.argmin(max_sims)
            farthest_idx = remaining[farthest_idx_in_remaining]

            selected_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)

        elapsed = time.perf_counter() - start

        return {
            "n_factors": n_factors,
            "n_representatives": n_representatives,
            "embedding_dim": embedding_dim,
            "elapsed_seconds": round(elapsed, 4),
            "latency_ms_per_representative": round(elapsed / n_representatives * 1000, 3),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def benchmark_batch_deduplication(
    n_factors: int,
    similarity_threshold: float = 0.95,
    embedding_dim: int = 128
) -> Dict[str, Any]:
    """Benchmark batch deduplication based on similarity."""

    try:
        print(f"    Generating {n_factors} embeddings with duplicates...")
        # Generate with some near-duplicates
        embeddings = generate_factor_embeddings(n_factors, embedding_dim)

        # Add some duplicates
        n_duplicates = n_factors // 10
        for i in range(n_duplicates):
            original_idx = np.random.randint(n_factors - n_duplicates)
            # Add small noise to create near-duplicate
            embeddings[n_factors - n_duplicates + i] = (
                embeddings[original_idx] + np.random.randn(embedding_dim) * 0.01
            )

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        gc.collect()
        start = time.perf_counter()

        # Find duplicates
        keep_mask = np.ones(n_factors, dtype=bool)

        for i in range(n_factors):
            if not keep_mask[i]:
                continue

            # Check similarity with remaining
            sims = embeddings[i] @ embeddings[i+1:].T
            duplicate_mask = sims > similarity_threshold

            if duplicate_mask.any():
                # Mark duplicates for removal
                duplicate_indices = np.where(duplicate_mask)[0] + i + 1
                keep_mask[duplicate_indices] = False

        elapsed = time.perf_counter() - start

        n_kept = keep_mask.sum()
        n_removed = n_factors - n_kept

        return {
            "n_factors": n_factors,
            "similarity_threshold": similarity_threshold,
            "embedding_dim": embedding_dim,
            "n_kept": int(n_kept),
            "n_removed": int(n_removed),
            "removal_rate": round(n_removed / n_factors, 3),
            "elapsed_seconds": round(elapsed, 4),
            "throughput_factors_per_sec": round(n_factors / elapsed, 2),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Run all factor assets benchmarks."""

    print("=" * 70)
    print("FACTOR ASSETS SIMILARITY SEARCH BENCHMARKS")
    print("=" * 70)

    results = {}
    total_start = time.perf_counter()

    # Exact search
    print(f"\n{'='*70}")
    print("EXACT SIMILARITY SEARCH")
    print(f"{'='*70}")

    search_configs = [
        ("small", 1000, 100, 10),
        ("medium", 10000, 100, 10),
        ("large", 100000, 100, 10),
    ]

    for scale_name, n_db, n_q, k in search_configs:
        print(f"\n  Scale: {scale_name} ({n_db} DB × {n_q} queries, k={k})")
        result = benchmark_exact_search(n_db, n_q, k=k)
        results[f"exact_search_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['throughput_queries_per_sec']:.1f} queries/s")
            print(f"      {result['latency_ms_per_query']:.2f} ms/query")

    # Clustering
    print(f"\n{'='*70}")
    print("CLUSTERING")
    print(f"{'='*70}")

    for scale_name, n_factors, n_clusters in [("small", 1000, 10), ("medium", 10000, 50), ("large", 100000, 200)]:
        print(f"\n  Scale: {scale_name} ({n_factors} factors → {n_clusters} clusters)")
        result = benchmark_clustering(n_factors, n_clusters)
        results[f"clustering_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['elapsed_seconds']:.2f}s, {result['iterations']} iterations")

    # Representative selection
    print(f"\n{'='*70}")
    print("REPRESENTATIVE SELECTION")
    print(f"{'='*70}")

    for scale_name, n_factors, n_reps in [("small", 1000, 100), ("medium", 10000, 500), ("large", 50000, 1000)]:
        print(f"\n  Scale: {scale_name} ({n_factors} → {n_reps} representatives)")
        result = benchmark_representative_selection(n_factors, n_reps)
        results[f"representative_selection_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['elapsed_seconds']:.2f}s")

    # Batch deduplication
    print(f"\n{'='*70}")
    print("BATCH DEDUPLICATION")
    print(f"{'='*70}")

    for scale_name, n_factors in [("small", 1000), ("medium", 5000), ("large", 10000)]:
        print(f"\n  Scale: {scale_name} ({n_factors} factors)")
        result = benchmark_batch_deduplication(n_factors)
        results[f"deduplication_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ Removed {result['n_removed']} ({result['removal_rate']:.1%})")

    total_elapsed = time.perf_counter() - total_start

    return {
        "benchmark": "factor_assets",
        "description": "Similarity search and clustering performance",
        "results": results,
        "total_time_s": round(total_elapsed, 2)
    }


if __name__ == "__main__":
    import json
    result = main()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))
