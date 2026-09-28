from factor_assets.similarity.exact import (
    QEPairwiseSimilarity,
    SimilarityMeasurementStatus,
    SimilarityMethod,
    SimilarityResult,
)


def _scan_reference(similarity, factor_id, **kwargs):
    """Pre-index implementation, retained here as the behavior oracle."""
    threshold = kwargs.get("threshold", 0.7)
    max_results = kwargs.get("max_results", 10)
    method = kwargs.get("method", SimilarityMethod.PEARSON)
    universe_ref = kwargs.get("universe_ref")
    period_start = kwargs.get("period_start")
    period_end = kwargs.get("period_end")
    results = []
    seen_factor_ids = set()
    for key, result in similarity._cache.items():
        fid_a, fid_b, method_val, universe, start, end = key
        if fid_a != factor_id or method_val != method.value:
            continue
        if universe_ref is not None and universe != universe_ref:
            continue
        if period_start is not None and start != period_start:
            continue
        if period_end is not None and end != period_end:
            continue
        if fid_b in seen_factor_ids:
            continue
        if result.is_high_similarity(threshold):
            results.append(result)
            seen_factor_ids.add(fid_b)
    results.sort(key=lambda result: abs(result.similarity_score), reverse=True)
    return results[:max_results]


def _result(a, b, score, *, method=SimilarityMethod.PEARSON,
            universe=None, start=None, end=None,
            status=SimilarityMeasurementStatus.COMPUTED_VALUE):
    return SimilarityResult(
        factor_id_a=a,
        factor_id_b=b,
        similarity_score=score,
        method=method,
        timestamp="2026-09-29T00:00:00Z",
        sample_size=100,
        universe_ref=universe,
        period_start=start,
        period_end=end,
        measurement_status=status,
    )


def test_qe_find_similar_index_matches_scan_reference_across_cache_mutations():
    similarity = QEPairwiseSimilarity()
    inserted = [
        _result("A", "B", 0.8, universe="US", start="s1", end="e1"),
        _result("C", "A", -0.91, universe="US", start="s1", end="e1"),
        _result("A", "D", 0.6, universe="US", start="s1", end="e1"),
        _result("A", "E", -0.9, method=SimilarityMethod.SPEARMAN,
                universe="US", start="s1", end="e1"),
        _result("A", "B", 0.72, universe="EU", start="s1", end="e1"),
        _result("A", "F", 0.88, universe="US", start="s2", end="e2"),
        _result("A", "G", None, universe="US", start="s1", end="e1",
                status=SimilarityMeasurementStatus.UNKNOWN),
        _result("A", "A", 0.99, universe="US", start="s1", end="e1"),
        # Replacing an existing directional key must retain its insertion slot.
        _result("A", "B", 0.77, universe="US", start="s1", end="e1"),
    ]
    for result in inserted:
        similarity.add_result(result)
    assert similarity.count() == 8

    queries = [
        ("A", {}),
        ("A", {"threshold": 0.8}),
        ("A", {"max_results": 2}),
        ("A", {"max_results": 0}),
        ("A", {"method": SimilarityMethod.SPEARMAN}),
        ("A", {"universe_ref": "US"}),
        ("A", {"period_start": "s1"}),
        ("A", {"universe_ref": "US", "period_start": "s1", "period_end": "e1"}),
        ("C", {}),
    ]
    for factor_id, kwargs in queries:
        assert similarity.find_similar(factor_id, **kwargs) == _scan_reference(
            similarity, factor_id, **kwargs
        )

    similarity.clear()
    assert similarity.count() == 0
    assert similarity.find_similar("A") == _scan_reference(similarity, "A") == []
    similarity.add_result(_result("A", "Z", 0.93))
    assert similarity.find_similar("A") == _scan_reference(similarity, "A")
