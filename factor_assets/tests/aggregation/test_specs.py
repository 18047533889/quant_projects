"""Tests for aggregation specifications."""

import pytest
from factor_assets.aggregation.specs import (
    WeightingScheme,
    AggregationSpec,
    AggregationResult,
)


def test_weighting_scheme_values():
    """Test weighting scheme enum values."""
    assert WeightingScheme.EQUAL.value == "equal"
    assert WeightingScheme.IC_WEIGHTED.value == "ic_weighted"
    assert WeightingScheme.INVERSE_VARIANCE.value == "inverse_variance"
    assert WeightingScheme.RANK_IC.value == "rank_ic"
    assert WeightingScheme.MEAN_ABSOLUTE_IC.value == "mean_abs_ic"
    assert WeightingScheme.CUSTOM.value == "custom"


def test_aggregation_spec_equal_weighted():
    """Test equal weighted aggregation spec."""
    spec = AggregationSpec(
        spec_id="test_spec_001",
        name="Equal Weight Test",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("factor_1", "factor_2", "factor_3"),
        universe_ref="US_LARGE",
        frequency="daily",
    )

    assert spec.spec_id == "test_spec_001"
    assert spec.name == "Equal Weight Test"
    assert spec.num_factors == 3
    assert spec.is_equal_weighted is True
    assert spec.is_ic_based is False
    assert spec.has_weight_bounds is False


def test_aggregation_spec_ic_weighted():
    """Test IC weighted aggregation spec."""
    spec = AggregationSpec(
        spec_id="test_spec_002",
        name="IC Weight Test",
        weighting_scheme=WeightingScheme.IC_WEIGHTED,
        factor_ids=("factor_a", "factor_b"),
        lookback_periods=252,
        min_ic_threshold=0.01,
    )

    assert spec.weighting_scheme == WeightingScheme.IC_WEIGHTED
    assert spec.is_ic_based is True
    assert spec.lookback_periods == 252
    assert spec.min_ic_threshold == 0.01


def test_aggregation_spec_custom_weights():
    """Test custom weighted aggregation spec."""
    spec = AggregationSpec(
        spec_id="test_spec_003",
        name="Custom Weight Test",
        weighting_scheme=WeightingScheme.CUSTOM,
        factor_ids=("f1", "f2", "f3"),
        custom_weights=(0.5, 0.3, 0.2),
    )

    assert spec.weighting_scheme == WeightingScheme.CUSTOM
    assert spec.custom_weights == (0.5, 0.3, 0.2)
    assert spec.num_factors == 3


def test_aggregation_spec_with_weight_bounds():
    """Test aggregation spec with weight bounds."""
    spec = AggregationSpec(
        spec_id="test_spec_004",
        name="Bounded Weight Test",
        weighting_scheme=WeightingScheme.IC_WEIGHTED,
        factor_ids=("f1", "f2", "f3", "f4"),
        max_weight=0.4,
        min_weight=0.1,
    )

    assert spec.has_weight_bounds is True
    assert spec.max_weight == 0.4
    assert spec.min_weight == 0.1


def test_aggregation_spec_requires_spec_id():
    """Test that spec_id is required."""
    with pytest.raises(ValueError, match="spec_id is required"):
        AggregationSpec(
            spec_id="",
            name="Test",
            weighting_scheme=WeightingScheme.EQUAL,
            factor_ids=("f1",),
        )


def test_aggregation_spec_requires_name():
    """Test that name is required."""
    with pytest.raises(ValueError, match="name is required"):
        AggregationSpec(
            spec_id="test",
            name="",
            weighting_scheme=WeightingScheme.EQUAL,
            factor_ids=("f1",),
        )


def test_aggregation_spec_requires_factor_ids():
    """Test that factor_ids cannot be empty."""
    with pytest.raises(ValueError, match="factor_ids cannot be empty"):
        AggregationSpec(
            spec_id="test",
            name="Test",
            weighting_scheme=WeightingScheme.EQUAL,
            factor_ids=(),
        )


def test_aggregation_spec_custom_requires_weights():
    """Test that CUSTOM scheme requires custom_weights."""
    with pytest.raises(ValueError, match="custom_weights required"):
        AggregationSpec(
            spec_id="test",
            name="Test",
            weighting_scheme=WeightingScheme.CUSTOM,
            factor_ids=("f1", "f2"),
        )


def test_aggregation_spec_custom_weights_length_mismatch():
    """Test that custom_weights length must match factor_ids."""
    with pytest.raises(ValueError, match="custom_weights length must match"):
        AggregationSpec(
            spec_id="test",
            name="Test",
            weighting_scheme=WeightingScheme.CUSTOM,
            factor_ids=("f1", "f2", "f3"),
            custom_weights=(0.5, 0.5),
        )


def test_aggregation_spec_custom_weights_only_for_custom():
    """Test that custom_weights only valid for CUSTOM scheme."""
    with pytest.raises(ValueError, match="custom_weights only valid for CUSTOM"):
        AggregationSpec(
            spec_id="test",
            name="Test",
            weighting_scheme=WeightingScheme.EQUAL,
            factor_ids=("f1", "f2"),
            custom_weights=(0.5, 0.5),
        )


def test_aggregation_spec_weight_bounds_validation():
    """Test that max_weight must be >= min_weight."""
    with pytest.raises(ValueError, match="max_weight must be >= min_weight"):
        AggregationSpec(
            spec_id="test",
            name="Test",
            weighting_scheme=WeightingScheme.EQUAL,
            factor_ids=("f1", "f2"),
            max_weight=0.2,
            min_weight=0.5,
        )


def test_aggregation_result_basic():
    """Test basic aggregation result."""
    spec = AggregationSpec(
        spec_id="spec_001",
        name="Test Spec",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("f1", "f2", "f3"),
    )

    result = AggregationResult(
        result_id="result_001",
        spec=spec,
        composite_factor_id="composite_factor_001",
        timestamp="2026-08-14T00:00:00Z",
    )

    assert result.result_id == "result_001"
    assert result.composite_factor_id == "composite_factor_001"
    assert result.has_warnings is False
    assert result.has_computed_weights is False


def test_aggregation_result_with_computed_weights():
    """Test aggregation result with computed weights."""
    spec = AggregationSpec(
        spec_id="spec_002",
        name="Test Spec",
        weighting_scheme=WeightingScheme.IC_WEIGHTED,
        factor_ids=("f1", "f2", "f3"),
    )

    result = AggregationResult(
        result_id="result_002",
        spec=spec,
        composite_factor_id="composite_002",
        timestamp="2026-08-14T00:00:00Z",
        computed_weights=(0.5, 0.3, 0.2),
        weight_computation_method="ic_weighted",
    )

    assert result.has_computed_weights is True
    assert result.computed_weights == (0.5, 0.3, 0.2)
    assert result.weight_computation_method == "ic_weighted"


def test_aggregation_result_weight_concentration():
    """Test weight concentration (Herfindahl index) computation."""
    spec = AggregationSpec(
        spec_id="spec_003",
        name="Test Spec",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("f1", "f2", "f3"),
    )

    # Equal weights: concentration = 3 * (1/3)^2 = 1/3
    result = AggregationResult(
        result_id="result_003",
        spec=spec,
        composite_factor_id="composite_003",
        timestamp="2026-08-14T00:00:00Z",
        computed_weights=(1/3, 1/3, 1/3),
    )

    concentration = result.weight_concentration
    assert concentration is not None
    assert abs(concentration - 1/3) < 1e-10

    # Concentrated weights: higher concentration
    result2 = AggregationResult(
        result_id="result_004",
        spec=spec,
        composite_factor_id="composite_004",
        timestamp="2026-08-14T00:00:00Z",
        computed_weights=(0.7, 0.2, 0.1),
    )

    concentration2 = result2.weight_concentration
    assert concentration2 is not None
    assert concentration2 > concentration


def test_aggregation_result_with_warnings():
    """Test aggregation result with warnings."""
    spec = AggregationSpec(
        spec_id="spec_005",
        name="Test Spec",
        weighting_scheme=WeightingScheme.IC_WEIGHTED,
        factor_ids=("f1", "f2"),
    )

    result = AggregationResult(
        result_id="result_005",
        spec=spec,
        composite_factor_id="composite_005",
        timestamp="2026-08-14T00:00:00Z",
        warnings=("Insufficient IC data", "Using fallback weights"),
    )

    assert result.has_warnings is True
    assert len(result.warnings) == 2
    assert "Insufficient IC data" in result.warnings


def test_aggregation_result_requires_result_id():
    """Test that result_id is required."""
    spec = AggregationSpec(
        spec_id="spec",
        name="Test",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("f1",),
    )

    with pytest.raises(ValueError, match="result_id is required"):
        AggregationResult(
            result_id="",
            spec=spec,
            composite_factor_id="composite",
            timestamp="2026-08-14T00:00:00Z",
        )


def test_aggregation_result_requires_composite_factor_id():
    """Test that composite_factor_id is required."""
    spec = AggregationSpec(
        spec_id="spec",
        name="Test",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("f1",),
    )

    with pytest.raises(ValueError, match="composite_factor_id is required"):
        AggregationResult(
            result_id="result",
            spec=spec,
            composite_factor_id="",
            timestamp="2026-08-14T00:00:00Z",
        )


def test_aggregation_result_requires_timestamp():
    """Test that timestamp is required."""
    spec = AggregationSpec(
        spec_id="spec",
        name="Test",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("f1",),
    )

    with pytest.raises(ValueError, match="timestamp is required"):
        AggregationResult(
            result_id="result",
            spec=spec,
            composite_factor_id="composite",
            timestamp="",
        )


def test_aggregation_result_weights_length_validation():
    """Test that computed_weights length must match spec."""
    spec = AggregationSpec(
        spec_id="spec",
        name="Test",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=("f1", "f2", "f3"),
    )

    with pytest.raises(ValueError, match="computed_weights length must match"):
        AggregationResult(
            result_id="result",
            spec=spec,
            composite_factor_id="composite",
            timestamp="2026-08-14T00:00:00Z",
            computed_weights=(0.5, 0.5),  # Only 2 weights for 3 factors
        )


def test_aggregation_result_with_quality_metrics():
    """Test aggregation result with quality metrics."""
    spec = AggregationSpec(
        spec_id="spec_006",
        name="Test Spec",
        weighting_scheme=WeightingScheme.IC_WEIGHTED,
        factor_ids=("f1", "f2", "f3", "f4"),
    )

    result = AggregationResult(
        result_id="result_006",
        spec=spec,
        composite_factor_id="composite_006",
        timestamp="2026-08-14T00:00:00Z",
        computed_weights=(0.4, 0.3, 0.2, 0.1),
        effective_num_factors=3.33,
        max_weight_used=0.4,
        min_weight_used=0.1,
        sample_size=1000,
        period_start="2024-01-01",
        period_end="2026-08-13",
    )

    assert result.effective_num_factors == 3.33
    assert result.max_weight_used == 0.4
    assert result.min_weight_used == 0.1
    assert result.sample_size == 1000
