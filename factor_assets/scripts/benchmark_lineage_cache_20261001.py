#!/usr/bin/env python3
"""Bounded, reproducible A/B benchmark for the lineage neighborhood cache."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

# Allow direct invocation from any working directory in the project checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factor_assets.clustering.lineage import LineageDetector
from factor_assets.graph.sparse import CorrelationEdge, SparseCorrelationGraph


DEFAULT_SEED = 20261001
MAX_NODES = 2_000
MAX_EDGES = 50_000
MAX_REPETITIONS = 21


class LegacyPathDetector(LineageDetector):
    """Use the original confidence hook to select the unoptimized path."""

    def _calculate_confidence(self, parent, child, subgraph):
        return super()._calculate_confidence(parent, child, subgraph)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--degree-steps", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not 2 <= args.nodes <= MAX_NODES:
        parser.error(f"--nodes must be in [2, {MAX_NODES}]")
    if not 1 <= args.degree_steps < args.nodes:
        parser.error("--degree-steps must be in [1, nodes - 1]")
    if not 1 <= args.repetitions <= MAX_REPETITIONS:
        parser.error(f"--repetitions must be in [1, {MAX_REPETITIONS}]")

    edge_count = sum(
        1
        for i in range(args.nodes)
        for step in range(1, args.degree_steps + 1)
        if i < (i + step) % args.nodes
    )
    if edge_count > MAX_EDGES:
        parser.error(f"generated graph has {edge_count} edges; cap is {MAX_EDGES}")
    return args


def _build_graph(nodes: int, degree_steps: int, seed: int) -> SparseCorrelationGraph:
    rng = random.Random(seed)
    edges = []
    for i in range(nodes):
        for step in range(1, degree_steps + 1):
            j = (i + step) % nodes
            if i < j:
                corr = 0.75 + 0.24 * rng.random()
                edges.append(CorrelationEdge(f"v{i:04}", f"v{j:04}", corr))
    rng.shuffle(edges)
    return SparseCorrelationGraph(edges)


def _timed_call(detector: LineageDetector, subgraph, members):
    started = perf_counter()
    relations = detector._find_relations(subgraph, members)
    return perf_counter() - started, relations


def main() -> None:
    args = _parse_args()

    graph_started = perf_counter()
    graph = _build_graph(args.nodes, args.degree_steps, args.seed)
    graph_build_seconds = perf_counter() - graph_started

    subgraph_started = perf_counter()
    members = graph.nodes
    subgraph = graph.subgraph(members)
    subgraph_build_seconds = perf_counter() - subgraph_started

    optimized = LineageDetector(graph, min_correlation=0.7, degree_threshold=1)
    legacy = LegacyPathDetector(graph, min_correlation=0.7, degree_threshold=1)

    # Warm both routes before collecting the alternating timing samples.
    legacy_relations = legacy._find_relations(subgraph, members)
    optimized_relations = optimized._find_relations(subgraph, members)
    if legacy_relations != optimized_relations:
        raise AssertionError("warmup relation outputs differ")

    legacy_times = []
    optimized_times = []
    outputs_equal = True
    for repetition in range(args.repetitions):
        order = (
            (("legacy", legacy), ("cached", optimized))
            if repetition % 2 == 0
            else (("cached", optimized), ("legacy", legacy))
        )
        for route, detector in order:
            elapsed, relations = _timed_call(detector, subgraph, members)
            if route == "legacy":
                legacy_times.append(elapsed)
                outputs_equal = outputs_equal and relations == legacy_relations
            else:
                optimized_times.append(elapsed)
                outputs_equal = outputs_equal and relations == legacy_relations
    if not outputs_equal:
        raise AssertionError("timed relation outputs differ from legacy output")

    legacy_median = median(legacy_times)
    optimized_median = median(optimized_times)
    result = {
        "seed": args.seed,
        "nodes": len(members),
        "edges": subgraph.edge_count,
        "degree_steps": args.degree_steps,
        "relations": len(legacy_relations),
        "outputs_equal_including_order": outputs_equal,
        "repetitions_per_route": args.repetitions,
        "graph_build_seconds_outside_timing": graph_build_seconds,
        "subgraph_build_seconds_outside_timing": subgraph_build_seconds,
        "legacy_median_seconds": legacy_median,
        "bounded_cache_median_seconds": optimized_median,
        "speedup": legacy_median / optimized_median,
        "neighbor_cache_max_entries": optimized._NEIGHBOR_CACHE_MAX_ENTRIES,
        "neighbor_cache_max_total_refs": optimized._NEIGHBOR_CACHE_MAX_TOTAL_NEIGHBORS,
        "caps": {
            "max_nodes": MAX_NODES,
            "max_edges": MAX_EDGES,
            "max_repetitions": MAX_REPETITIONS,
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
