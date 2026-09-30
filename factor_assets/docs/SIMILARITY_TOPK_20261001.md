# Similarity top-k selection

For `find_similar(..., max_results=k)`, Assets first keeps each neighbor's
strongest absolute admissible score. Equal scores use the existing scope
ordering. It then selects the first k neighbors with `heapq.nsmallest`, using
the same score, neighbor-ID and scope ordering as the previous full sort.
Both cached similarity providers use this helper; their evidence remains
summary-only. This change does not recompute factor correlations.

Small k selection costs O(M log k) after deduplication, rather than O(M log M)
for sorting M unique neighbors. Candidate scanning and deduplication still
need O(C) time and O(M) storage for C qualifying scope measurements.
A zero result limit returns before visiting the cache index. Negative limits
raise ValueError; noninteger limits retain the previous slice-index rejection.

Reproduce the bounded synthetic comparison from the monorepo root:

```sh
.venv/bin/python -m factor_assets.scripts.benchmark_similarity_topk \
  --neighbors 20000 --top-k 10 --repetitions 7
```

On server-c, 60,000 scope measurements for 20,000 neighbors gave medians
0.165937 seconds for full sorting and 0.121593 seconds for top-k selection.
The script alternates order, excludes one warmup, and compares complete
ordered outputs on each call. These numbers cover the dedup/selection helper;
they exclude provider computation, cache filtering and source I/O.
New tests cover strongest scopes, positive/negative score ties, reversed
insertion order, zero/oversized limits and integer index rules.
