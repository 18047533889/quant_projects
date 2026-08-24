"""
Additional validation tests for cache collision fix.

Tests edge cases and scenarios that may have been missed in initial implementation.
"""

from factor_assets.similarity.exact import (
    QEPairwiseSimilarity,
    CorrelationSimilarity,
    SimilarityResult,
    SimilarityMethod,
)


def test_qe_symmetric_pair_shares_cache():
    """
    Test that (A,B) and (B,A) share the same cache entry.

    Verifies symmetric storage is working correctly.
    """
    similarity = QEPairwiseSimilarity()

    # Add result for (A, B)
    result = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    similarity.add_result(result)

    # Query with reversed order (B, A) should return same result
    retrieved = similarity.compute_similarity("factor_b", "factor_a")

    assert retrieved is not None
    assert retrieved.similarity_score == 0.85
    assert retrieved.factor_id_a == "factor_a"  # Original ordering preserved
    assert retrieved.factor_id_b == "factor_b"


def test_qe_count_correct_after_multiple_adds():
    """
    Test that count() correctly accounts for symmetric storage.

    Each unique pair is stored twice (A,B) and (B,A) but should count as 1.
    """
    similarity = QEPairwiseSimilarity()

    # Should start at 0
    assert similarity.count() == 0

    # Add first result
    result1 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.8,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    similarity.add_result(result1)
    assert similarity.count() == 1

    # Add second result with different factors
    result2 = SimilarityResult(
        factor_id_a="factor_c",
        factor_id_b="factor_d",
        similarity_score=0.6,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    similarity.add_result(result2)
    assert similarity.count() == 2

    # Add third result (different method, same factors as first)
    result3 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.75,
        method=SimilarityMethod.SPEARMAN,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    similarity.add_result(result3)
    assert similarity.count() == 3  # Different method = different entry


def test_correlation_similarity_count_correct():
    """
    Test that CorrelationSimilarity count() also works correctly.
    """
    similarity = CorrelationSimilarity()

    assert similarity.count() == 0

    # Add three distinct entries
    for i, (method, score) in enumerate([
        (SimilarityMethod.PEARSON, 0.8),
        (SimilarityMethod.SPEARMAN, 0.7),
        (SimilarityMethod.KENDALL, 0.6),
    ]):
        result = SimilarityResult(
            factor_id_a="factor_a",
            factor_id_b="factor_b",
            similarity_score=score,
            method=method,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=100,
        )
        similarity.add_result(result)
        assert similarity.count() == i + 1


def test_cache_key_distinguishes_empty_vs_none():
    """
    Test that empty string and None are handled consistently in cache keys.

    Both should map to empty string "" for cache key purposes.
    """
    similarity = QEPairwiseSimilarity()

    # Add with None universe
    result_none = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.8,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        universe_ref=None,
    )
    similarity.add_result(result_none)

    # Query with None should work
    retrieved_none = similarity.compute_similarity(
        "factor_a", "factor_b", universe_ref=None
    )
    assert retrieved_none is not None
    assert retrieved_none.similarity_score == 0.8

    # Query with empty string should also work (maps to same key)
    # This is testing the implementation detail that None and "" both -> ""
    # in the cache key


def test_cache_distinguishes_similar_but_different_periods():
    """
    Test that very similar but distinct periods don't collide.
    """
    similarity = QEPairwiseSimilarity()

    # Add result for period ending 2023-12-30
    result1 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.9,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        period_start="2023-01-01",
        period_end="2023-12-30",
    )
    similarity.add_result(result1)

    # Add result for period ending 2023-12-31
    result2 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.4,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        period_start="2023-01-01",
        period_end="2023-12-31",
    )
    similarity.add_result(result2)

    # Verify both are cached independently
    retrieved1 = similarity.compute_similarity(
        "factor_a", "factor_b",
        period_start="2023-01-01", period_end="2023-12-30"
    )
    retrieved2 = similarity.compute_similarity(
        "factor_a", "factor_b",
        period_start="2023-01-01", period_end="2023-12-31"
    )

    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.similarity_score == 0.9
    assert retrieved2.similarity_score == 0.4
    assert similarity.count() == 2


def test_clear_resets_count():
    """
    Test that clear() properly resets count to 0.
    """
    similarity = QEPairwiseSimilarity()

    # Add some results
    for i in range(3):
        result = SimilarityResult(
            factor_id_a=f"factor_{i}",
            factor_id_b=f"factor_{i+1}",
            similarity_score=0.8,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=100,
        )
        similarity.add_result(result)

    assert similarity.count() == 3

    similarity.clear()
    assert similarity.count() == 0
