"""
Tests for conditional novelty with result identity and reverse cache.
"""

import pytest
from factor_assets.novelty.conditional import (
    NoveltyResult,
    ResultIdentity,
    ResultIdentityCache,
    SimpleConditionalNoveltyAssessor,
)


class TestResultIdentity:
    """Tests for ResultIdentity."""

    def test_from_values(self):
        """Test creating result identity from values."""
        values = b"mock_factor_values_serialized"

        result_id = ResultIdentity.from_values(
            factor_id="factor1",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        assert result_id.factor_id == "factor1"
        assert result_id.result_hash is not None
        assert len(result_id.result_hash) == 64  # SHA256 hex
        assert result_id.universe_ref == "top3000"
        assert result_id.num_observations == 1000

    def test_same_values_same_hash(self):
        """Test that same values produce same hash."""
        values = b"identical_values"

        result_id1 = ResultIdentity.from_values(
            factor_id="factor1",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        result_id2 = ResultIdentity.from_values(
            factor_id="factor2",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        # Different factor IDs but same values -> same result hash
        assert result_id1.result_hash == result_id2.result_hash
        assert result_id1.factor_id != result_id2.factor_id

    def test_different_values_different_hash(self):
        """Test that different values produce different hashes."""
        result_id1 = ResultIdentity.from_values(
            factor_id="factor1",
            values=b"values_a",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        result_id2 = ResultIdentity.from_values(
            factor_id="factor2",
            values=b"values_b",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        assert result_id1.result_hash != result_id2.result_hash


class TestResultIdentityCache:
    """Tests for ResultIdentityCache."""

    def test_add_and_find(self):
        """Test adding and finding equivalent factors."""
        cache = ResultIdentityCache()

        result_id = ResultIdentity.from_values(
            factor_id="factor1",
            values=b"test_values",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        cache.add(result_id)

        equivalents = cache.find_equivalent(result_id.result_hash)
        assert len(equivalents) == 1
        assert equivalents[0] == "factor1"

    def test_multiple_equivalent_factors(self):
        """Test finding multiple factors with same results."""
        cache = ResultIdentityCache()

        values = b"shared_values"

        result_id1 = ResultIdentity.from_values(
            factor_id="factor1",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        result_id2 = ResultIdentity.from_values(
            factor_id="factor2",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        cache.add(result_id1)
        cache.add(result_id2)

        equivalents = cache.find_equivalent(result_id1.result_hash)
        assert len(equivalents) == 2
        assert "factor1" in equivalents
        assert "factor2" in equivalents

    def test_has_equivalent(self):
        """Test checking if cache has equivalent factors."""
        cache = ResultIdentityCache()

        result_id = ResultIdentity.from_values(
            factor_id="factor1",
            values=b"test_values",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        assert not cache.has_equivalent(result_id.result_hash)

        cache.add(result_id)

        assert cache.has_equivalent(result_id.result_hash)

    def test_no_duplicates(self):
        """Test that adding same factor twice doesn't duplicate."""
        cache = ResultIdentityCache()

        result_id = ResultIdentity.from_values(
            factor_id="factor1",
            values=b"test_values",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        cache.add(result_id)
        cache.add(result_id)  # Add again

        equivalents = cache.find_equivalent(result_id.result_hash)
        assert len(equivalents) == 1

    def test_count(self):
        """Test counting unique result identities."""
        cache = ResultIdentityCache()

        result_id1 = ResultIdentity.from_values(
            factor_id="factor1",
            values=b"values_a",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        result_id2 = ResultIdentity.from_values(
            factor_id="factor2",
            values=b"values_b",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        assert cache.count() == 0

        cache.add(result_id1)
        assert cache.count() == 1

        cache.add(result_id2)
        assert cache.count() == 2

    def test_clear(self):
        """Test clearing the cache."""
        cache = ResultIdentityCache()

        result_id = ResultIdentity.from_values(
            factor_id="factor1",
            values=b"test_values",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        cache.add(result_id)
        assert cache.count() == 1

        cache.clear()
        assert cache.count() == 0
        assert not cache.has_equivalent(result_id.result_hash)


class TestSimpleConditionalNoveltyAssessor:
    """Tests for SimpleConditionalNoveltyAssessor."""

    def test_novel_factor(self):
        """Test assessing a novel factor."""
        cache = ResultIdentityCache()
        assessor = SimpleConditionalNoveltyAssessor(cache)

        result_id = ResultIdentity.from_values(
            factor_id="new_factor",
            values=b"new_values",
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        novelty = assessor.assess_novelty(
            factor_id="new_factor",
            result_identity=result_id,
            pool_factor_ids=("factor1", "factor2"),
            pool_ref="active_pool",
        )

        assert novelty.is_novel
        assert novelty.novelty_score == 1.0
        assert len(novelty.similar_factor_ids) == 0
        assert novelty.assessment_method == "result_identity"

    def test_redundant_factor(self):
        """Test assessing a redundant factor."""
        cache = ResultIdentityCache()
        assessor = SimpleConditionalNoveltyAssessor(cache)

        values = b"shared_values"

        # Add existing factor to cache
        existing_id = ResultIdentity.from_values(
            factor_id="existing_factor",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )
        cache.add(existing_id)

        # Assess new factor with same results
        new_id = ResultIdentity.from_values(
            factor_id="new_factor",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        novelty = assessor.assess_novelty(
            factor_id="new_factor",
            result_identity=new_id,
            pool_factor_ids=("existing_factor",),
            pool_ref="active_pool",
        )

        assert not novelty.is_novel
        assert novelty.novelty_score == 0.0
        assert "existing_factor" in novelty.similar_factor_ids
        assert novelty.assessment_method == "result_identity"

    def test_equivalent_not_in_pool(self):
        """Test that equivalents outside pool don't affect novelty."""
        cache = ResultIdentityCache()
        assessor = SimpleConditionalNoveltyAssessor(cache)

        values = b"shared_values"

        # Add factor NOT in pool to cache
        outside_id = ResultIdentity.from_values(
            factor_id="outside_factor",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )
        cache.add(outside_id)

        # Assess new factor with same results but against different pool
        new_id = ResultIdentity.from_values(
            factor_id="new_factor",
            values=values,
            universe_ref="top3000",
            period_start="2023-01-01",
            period_end="2023-12-31",
            num_observations=1000,
        )

        novelty = assessor.assess_novelty(
            factor_id="new_factor",
            result_identity=new_id,
            pool_factor_ids=("other_factor",),  # outside_factor not in pool
            pool_ref="active_pool",
        )

        # Should be novel because equivalent is not in the comparison pool
        assert novelty.is_novel
        assert novelty.novelty_score == 1.0


class TestNoveltyResult:
    """Tests for NoveltyResult validation."""

    def test_valid_novelty_result(self):
        """Test creating valid novelty result."""
        result = NoveltyResult(
            factor_id="factor1",
            is_novel=True,
            novelty_score=0.8,
            similar_factor_ids=(),
            assessment_method="conditional_ic",
            pool_ref="active_pool",
        )

        assert result.factor_id == "factor1"
        assert result.is_novel
        assert result.novelty_score == 0.8

    def test_empty_factor_id_raises(self):
        """Test that empty factor_id raises ValueError."""
        with pytest.raises(ValueError, match="factor_id is required"):
            NoveltyResult(
                factor_id="",
                is_novel=True,
                novelty_score=0.8,
                similar_factor_ids=(),
                assessment_method="test",
                pool_ref="pool",
            )

    def test_novelty_score_bounds_validation(self):
        """Test that novelty_score must be in [0,1]."""
        with pytest.raises(ValueError, match="must be in"):
            NoveltyResult(
                factor_id="factor1",
                is_novel=True,
                novelty_score=1.5,  # Invalid
                similar_factor_ids=(),
                assessment_method="test",
                pool_ref="pool",
            )

    def test_empty_assessment_method_raises(self):
        """Test that empty assessment_method raises ValueError."""
        with pytest.raises(ValueError, match="assessment_method is required"):
            NoveltyResult(
                factor_id="factor1",
                is_novel=True,
                novelty_score=0.8,
                similar_factor_ids=(),
                assessment_method="",
                pool_ref="pool",
            )
