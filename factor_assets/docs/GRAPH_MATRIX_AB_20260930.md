# Hierarchical graph distance matrix A/B (2026-09-30)

The hierarchical clustering path previously visited every upper-triangle pair
and called `SparseCorrelationGraph.get_correlation`, which linearly scans the
left node's adjacency list. It now builds a temporary neighbor-to-correlation
map for each left row, then fills that row of the same dense symmetric matrix.
The matrix remains O(N²), as required by the clustering backend; temporary
index storage is O(max degree), not a permanent graph-wide cache.

Compatibility is checked against an independent copy of the previous matrix
loop. Tests cover dense matrices and public linkage parity, symmetry and zero
diagonal, exact duplicate graph edges, duplicate adjacency entries (first
value wins), asymmetric/reverse-only rows, present-`None` entries, subclass
overrides and duck graphs, the same first missing pair and exact error text
under the default fail-closed policy, explicit missing-as-maximum behavior,
conflicting duplicate rejection, and constructor validation for unknown
missing policies and Ward. Focused validation: `pytest -q
factor_assets/tests/test_hierarchical_graph_matrix_20260930.py factor_assets/tests/test_clustering_hierarchical.py` — 26 passed.

## Bounded matrix construction benchmark

Compared the prior nested `get_correlation` loop with the new implementation
on deterministic complete graphs. Each timed call includes node extraction
and sorting, matrix allocation, factor-index creation, neighbor-list copies,
per-row dict construction, and matrix fill. Shared graph construction is
outside both timed regions. Each size was run three times; the median is
reported. Matrices were asserted element-identical before timing. Maximum size
was 500 nodes (124,750 edges; a 500 × 500 float64 matrix, approximately 1.91
MiB). The benchmark wrote no result files.

| Nodes | Edges | Previous loop | Indexed rows | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 4,950 | 7.61 ms | 1.64 ms | 4.6× |
| 250 | 31,125 | 114.72 ms | 10.72 ms | 10.7× |
| 500 | 124,750 | 883.52 ms | 43.48 ms | 20.3× |

This synthetic complete-graph measurement isolates matrix construction; it
does not measure SciPy linkage, clustering quality, or end-to-end pipeline
latency. The default missing-edge policy still fails closed at the same first
lexical pair. Only the explicit `treat_missing_as_max` option maps absent
pairs to distance 1.0.
