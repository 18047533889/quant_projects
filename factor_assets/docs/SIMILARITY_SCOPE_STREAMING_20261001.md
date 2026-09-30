# Scoped similarity lookup and streaming deduplication

When you query `QEPairwiseSimilarity.find_similar` or
`CorrelationSimilarity.find_similar`, FA visits the factor-local directional
key index. It checks method, universe and requested period bounds before
reading result objects from the cache. A narrow query therefore avoids result
reads for mismatched scopes; it still visits indexed keys to apply filters.

FA streams threshold-passing results into its strongest-per-neighbor map.
It retains the highest absolute score for each neighbor, resolves equal-score
scope ties with the existing deterministic scope key, then applies the existing
top-k selection. Output ordering and directional orientation remain unchanged.
This removes the extra list of all passing scoped candidates. Exact
deduplication still needs O(unique matching neighbors) map entries; top-k does
not impose a bound on the complete query's memory.

The dedicated regression suite compares both similarity classes with an
independent full-scan reference across method/scope filters, score thresholds,
ties, result replacement, cache clearing and result limits. A counting cache
checks that narrow queries read only matching results and limit zero reads
none. The combined focused run passed 56 tests. These operation-count checks
establish avoided cache reads; they do not establish end-to-end latency gains.

Implementation: `similarity/exact.py`. Regression tests:
`tests/test_similarity_scope_filter_streaming.py`.
