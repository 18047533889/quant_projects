# Lineage bounded-cache benchmark (2026-10-01)

Reproduce from the repository root with:

    .venv/bin/python factor_assets/scripts/benchmark_lineage_cache_20261001.py

The script defaults to seed 20261001, 500 nodes, 24 forward steps, and five
alternating repetitions per route. It caps runs at 2,000 nodes, 50,000 edges,
and 21 repetitions. Graph and induced-subgraph construction are timed separately
and excluded from route timings. The legacy route is selected by a detector
subclass that forwards the confidence method to its parent, so no copy of the
old traversal is embedded in the benchmark. Both routes receive the same graph
and members; every returned relation list is compared in full, including order.

Captured stdout from the accepted run:

~~~json
{"bounded_cache_median_seconds": 0.07171712597482838, "caps": {"max_edges": 50000, "max_nodes": 2000, "max_repetitions": 21}, "degree_steps": 24, "edges": 11700, "graph_build_seconds_outside_timing": 0.10769202097435482, "legacy_median_seconds": 0.13110803501331247, "neighbor_cache_max_entries": 128, "neighbor_cache_max_total_refs": 16384, "nodes": 500, "outputs_equal_including_order": true, "relations": 11700, "repetitions_per_route": 5, "seed": 20261001, "speedup": 1.8281272880250306, "subgraph_build_seconds_outside_timing": 0.10580600798130035}
~~~

This single seeded synthetic graph is a bounded microbenchmark, not a general
performance guarantee. The cache limits retained neighborhood data to 128
entries and 16,384 total neighbor references; traversal results and relation
outputs retain their existing semantics.
