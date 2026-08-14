"""
Test that similarity cache properly distinguishes all parameters.

Verifies that CACHE-001 fix prevents cache collisions when same factor
pair is evaluated with different methods, universes, or time periods.
"""

from factor_assets.similarity.exact import (
    QEPairwiseSimilarity,
    CorrelationSimilarity,
    SimilarityResult,
    SimilarityMethod,
)


def test_qe_pairwise_cache_no_collision_method():
    """
    Test that different methods don't collide in cache.

    Same factor pair with different methods should be cached separately.
    """
    similarity = QEPairwiseSimilarity()

    # Add pearson result
    result_pearson = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.8,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    similarity.add_result(result_pearson)

    # Add spearman result for same factors
    result_spearman = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.6,
        method=SimilarityMethod.SPEARMAN,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    similarity.add_result(result_spearman)

    # Verify both are cached independently
    retrieved_pearson = similarity.compute_similarity(
        "factor_a", "factor_b", method=SimilarityMethod.PEARSON
    )
    retrieved_spearman = similarity.compute_similarity(
        "factor_a", "factor_b", method=SimilarityMethod.SPEARMAN
    )

    assert retrieved_pearson is not None
    assert retrieved_spearman is not None
    assert retrieved_pearson.similarity_score == 0.8
    assert retrieved_spearman.similarity_score == 0.6


def test_qe_pairwise_cache_no_collision_universe():
    """
    Test that different universes don't collide in cache.

    Same factor pair with different universes should be cached separately.
    """
    similarity = QEPairwiseSimilarity()

    # Add result for universe A
    result_univ_a = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.7,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        universe_ref="universe_a",
    )
    similarity.add_result(result_univ_a)

    # Add result for universe B
    result_univ_b = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.5,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        universe_ref="universe_b",
    )
    similarity.add_result(result_univ_b)

    # Verify both are cached independently
    retrieved_a = similarity.compute_similarity(
        "factor_a", "factor_b", universe_ref="universe_a"
    )
    retrieved_b = similarity.compute_similarity(
        "factor_a", "factor_b", universe_ref="universe_b"
    )

    assert retrieved_a is not None
    assert retrieved_b is not None
    assert retrieved_a.similarity_score == 0.7
    assert retrieved_b.similarity_score == 0.5


def test_qe_pairwise_cache_no_collision_period():
    """
    Test that different time periods don't collide in cache.

    Same factor pair with different periods should be cached separately.
    """
    similarity = QEPairwiseSimilarity()

    # Add result for period 1
    result_period_1 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.9,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        period_start="2023-01-01",
        period_end="2023-06-30",
    )
    similarity.add_result(result_period_1)

    # Add result for period 2
    result_period_2 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.4,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        period_start="2023-07-01",
        period_end="2023-12-31",
    )
    similarity.add_result(result_period_2)

    # Verify both are cached independently
    retrieved_1 = similarity.compute_similarity(
        "factor_a", "factor_b",
        period_start="2023-01-01", period_end="2023-06-30"
    )
    retrieved_2 = similarity.compute_similarity(
        "factor_a", "factor_b",
        period_start="2023-07-01", period_end="2023-12-31"
    )

    assert retrieved_1 is not None
    assert retrieved_2 is not None
    assert retrieved_1.similarity_score == 0.9
    assert retrieved_2.similarity_score == 0.4


def test_correlation_similarity_cache_no_collision():
    """
    Test that CorrelationSimilarity also prevents cache collisions.

    Verifies the fix applies to both similarity implementations.
    """
    similarity = CorrelationSimilarity()

    # Add results with different parameters
    result_1 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.8,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        universe_ref="universe_a",
    )
    similarity.add_result(result_1)

    result_2 = SimilarityResult(
        factor_id_a="factor_a",
        factor_id_b="factor_b",
        similarity_score=0.3,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        universe_ref="universe_b",
    )
    similarity.add_result(result_2)

    # Verify both are cached independently
    retrieved_1 = similarity.compute_similarity(
        "factor_a", "factor_b", universe_ref="universe_a"
    )
    retrieved_2 = similarity.compute_similarity(
        "factor_a", "factor_b", universe_ref="universe_b"
    )

    assert retrieved_1 is not None
    assert retrieved_2 is not None
    assert retrieved_1.similarity_score == 0.8
    assert retrieved_2.similarity_score == 0.3

    # Verify cache count is correct (2 unique results, stored symmetrically)
    assert similarity.count() == 2
