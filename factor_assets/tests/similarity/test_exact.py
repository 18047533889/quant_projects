"""
Tests for correlation-based similarity measurement.
"""

import pytest

from factor_assets.similarity import (
    SimilarityMethod,
    SimilarityResult,
    CorrelationSimilarity,
)


def test_similarity_result_creation():
    """Test SimilarityResult creation."""
    result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
        universe_ref="US_500",
        period_start="2024-01-01",
        period_end="2024-12-31",
        confidence=0.95,
    )

    assert result.factor_id_a == "F001"
    assert result.factor_id_b == "F002"
    assert result.similarity_score == 0.85
    assert result.method == SimilarityMethod.PEARSON
    assert result.sample_size == 1000


def test_similarity_result_requires_fields():
    """SimilarityResult must have required fields."""
    with pytest.raises(ValueError, match="factor_id_a"):
        SimilarityResult(
            factor_id_a="",
            factor_id_b="F002",
            similarity_score=0.85,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )

    with pytest.raises(ValueError, match="factor_id_b"):
        SimilarityResult(
            factor_id_a="F001",
            factor_id_b="",
            similarity_score=0.85,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )


def test_similarity_result_score_bounds():
    """Similarity score must be in [-1, 1]."""
    with pytest.raises(ValueError, match="similarity_score must be in"):
        SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=1.5,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )

    with pytest.raises(ValueError, match="similarity_score must be in"):
        SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=-1.5,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )


def test_similarity_result_sample_size_bounds():
    """Sample size must be non-negative."""
    with pytest.raises(ValueError, match="sample_size must be non-negative"):
        SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=0.85,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=-10,
        )


def test_similarity_result_is_high_similarity():
    """Test is_high_similarity property."""
    high_sim = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    assert high_sim.is_high_similarity(threshold=0.7)
    assert not high_sim.is_high_similarity(threshold=0.9)

    # Test with negative correlation
    high_neg_sim = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=-0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    assert high_neg_sim.is_high_similarity(threshold=0.7)


def test_similarity_result_is_significant():
    """Test is_significant property."""
    large_sample = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.5,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
    )
    assert large_sample.is_significant(min_samples=30)
    assert not large_sample.is_significant(min_samples=200)


def test_similarity_result_has_warnings():
    """Test has_warnings property."""
    no_warnings = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    assert not no_warnings.has_warnings

    with_warnings = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
        warnings=("low_sample_size",),
    )
    assert with_warnings.has_warnings


def test_correlation_similarity_add_result():
    """Test adding similarity results."""
    similarity = CorrelationSimilarity()

    result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )

    similarity.add_result(result)
    assert similarity.count() == 1


def test_correlation_similarity_compute_similarity():
    """Test computing similarity."""
    similarity = CorrelationSimilarity()

    result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(result)

    # Retrieve in same order
    retrieved = similarity.compute_similarity("F001", "F002")
    assert retrieved is not None
    assert retrieved.similarity_score == 0.85

    # Retrieve in reverse order (should work due to symmetry)
    retrieved_rev = similarity.compute_similarity("F002", "F001")
    assert retrieved_rev is not None
    assert retrieved_rev.similarity_score == 0.85


def test_correlation_similarity_compute_not_found():
    """Test computing similarity when not cached."""
    similarity = CorrelationSimilarity()

    result = similarity.compute_similarity("F001", "F999")
    assert result is None


def test_correlation_similarity_compute_by_method():
    """Test filtering by similarity method."""
    similarity = CorrelationSimilarity()

    pearson_result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(pearson_result)

    spearman_result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.80,
        method=SimilarityMethod.SPEARMAN,
        timestamp="2024-01-02T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(spearman_result)

    # Query for Pearson
    pearson = similarity.compute_similarity("F001", "F002", method=SimilarityMethod.PEARSON)
    assert pearson is not None
    assert pearson.similarity_score == 0.85

    # Query for Spearman
    spearman = similarity.compute_similarity("F001", "F002", method=SimilarityMethod.SPEARMAN)
    assert spearman is not None
    assert spearman.similarity_score == 0.80


def test_correlation_similarity_most_recent():
    """Test retrieving most recent similarity."""
    similarity = CorrelationSimilarity()

    old_result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.75,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(old_result)

    new_result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-02-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(new_result)

    retrieved = similarity.compute_similarity("F001", "F002")
    assert retrieved is not None
    assert retrieved.similarity_score == 0.85  # Most recent


def test_correlation_similarity_find_similar():
    """Test finding similar factors."""
    similarity = CorrelationSimilarity()

    # Add multiple similarity results
    for i in range(2, 6):
        result = SimilarityResult(
            factor_id_a="F001",
            factor_id_b=f"F{i:03d}",
            similarity_score=0.95 - (i * 0.05),
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )
        similarity.add_result(result)

    # Find factors similar to F001
    similar = similarity.find_similar("F001", threshold=0.7)

    assert len(similar) == 4  # F002, F003, F004, F005 (all >= 0.7)
    assert similar[0].factor_id_b == "F002"  # Highest similarity first
    assert similar[0].similarity_score == 0.85


def test_correlation_similarity_find_similar_max_results():
    """Test limiting find_similar results."""
    similarity = CorrelationSimilarity()

    for i in range(2, 12):
        result = SimilarityResult(
            factor_id_a="F001",
            factor_id_b=f"F{i:03d}",
            similarity_score=0.8,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
        )
        similarity.add_result(result)

    similar = similarity.find_similar("F001", threshold=0.7, max_results=5)
    assert len(similar) == 5


def test_correlation_similarity_find_similar_by_method():
    """Test finding similar factors by method."""
    similarity = CorrelationSimilarity()

    pearson_result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(pearson_result)

    spearman_result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F003",
        similarity_score=0.85,
        method=SimilarityMethod.SPEARMAN,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(spearman_result)

    # Find Pearson similar
    pearson_similar = similarity.find_similar("F001", method=SimilarityMethod.PEARSON)
    assert len(pearson_similar) == 1
    assert pearson_similar[0].factor_id_b == "F002"

    # Find Spearman similar
    spearman_similar = similarity.find_similar("F001", method=SimilarityMethod.SPEARMAN)
    assert len(spearman_similar) == 1
    assert spearman_similar[0].factor_id_b == "F003"


def test_correlation_similarity_count():
    """Test counting similarity results."""
    similarity = CorrelationSimilarity()

    assert similarity.count() == 0

    result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(result)

    assert similarity.count() == 1  # Symmetric pairs count as one


def test_correlation_similarity_clear():
    """Test clearing similarity cache."""
    similarity = CorrelationSimilarity()

    result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(result)

    assert similarity.count() == 1

    similarity.clear()

    assert similarity.count() == 0
    assert similarity.compute_similarity("F001", "F002") is None


def test_correlation_similarity_symmetric():
    """Test that similarity is symmetric."""
    similarity = CorrelationSimilarity()

    result = SimilarityResult(
        factor_id_a="F001",
        factor_id_b="F002",
        similarity_score=0.85,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=1000,
    )
    similarity.add_result(result)

    # Both directions should work
    forward = similarity.compute_similarity("F001", "F002")
    backward = similarity.compute_similarity("F002", "F001")

    assert forward is not None
    assert backward is not None
    assert forward.similarity_score == backward.similarity_score


def test_correlation_similarity_find_similar_filters_universe_and_period():
    similarity = CorrelationSimilarity()
    for factor_id, universe, start, end in [
        ("MATCH", "US", "2024-01-01", "2024-12-31"),
        ("OTHER_UNIVERSE", "EU", "2024-01-01", "2024-12-31"),
        ("OTHER_PERIOD", "US", "2023-01-01", "2023-12-31"),
    ]:
        similarity.add_result(SimilarityResult(
            factor_id_a="F001", factor_id_b=factor_id, similarity_score=0.9,
            method=SimilarityMethod.PEARSON, timestamp="2025-01-01T00:00:00Z",
            sample_size=100, universe_ref=universe, period_start=start, period_end=end,
        ))

    results = similarity.find_similar(
        "F001", universe_ref="US", period_start="2024-01-01", period_end="2024-12-31"
    )
    assert [result.factor_id_b for result in results] == ["MATCH"]


def test_correlation_similarity_find_similar_partial_filter_is_exact():
    similarity = CorrelationSimilarity()
    similarity.add_result(SimilarityResult(
        factor_id_a="F001", factor_id_b="DATED", similarity_score=0.9,
        method=SimilarityMethod.PEARSON, timestamp="2025-01-01T00:00:00Z",
        sample_size=100, universe_ref="US", period_start="2024-01-01", period_end="2024-12-31",
    ))
    similarity.add_result(SimilarityResult(
        factor_id_a="F001", factor_id_b="UNDATED", similarity_score=0.8,
        method=SimilarityMethod.PEARSON, timestamp="2025-01-01T00:00:00Z",
        sample_size=100, universe_ref="US",
    ))

    assert [r.factor_id_b for r in similarity.find_similar("F001", universe_ref="US")] == ["DATED", "UNDATED"]
    assert [r.factor_id_b for r in similarity.find_similar("F001", period_start="2024-01-01")] == ["DATED"]


def test_similarity_method_enum():
    """Test SimilarityMethod enum values."""
    assert SimilarityMethod.PEARSON.value == "pearson"
    assert SimilarityMethod.SPEARMAN.value == "spearman"
    assert SimilarityMethod.KENDALL.value == "kendall"
