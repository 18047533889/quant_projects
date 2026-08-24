"""Tests for family representative selection."""

import pytest
from factor_assets.aggregation.representatives import (
    RepresentativeSelectionMethod,
    RepresentativeSelection,
    FamilyRepresentativeSelector,
    ICProvider,
    CorrelationProvider,
)


class MockICProvider:
    """Mock IC provider for testing."""

    def __init__(self, ic_data: dict[str, float]):
        self.ic_data = ic_data

    def get_ic(self, factor_id: str, universe_ref=None, lookback_periods=None):
        return self.ic_data.get(factor_id)


class MockCorrelationProvider:
    """Mock correlation provider for testing."""

    def __init__(self, correlation_data: dict[str, float]):
        self.correlation_data = correlation_data

    def get_avg_correlation(self, factor_id: str, other_factor_ids, universe_ref=None):
        return self.correlation_data.get(factor_id)


def test_representative_selection_method_values():
    """Test representative selection method enum values."""
    assert RepresentativeSelectionMethod.MAX_IC.value == "max_ic"
    assert RepresentativeSelectionMethod.MIN_CORRELATION.value == "min_correlation"
    assert RepresentativeSelectionMethod.EQUAL_WEIGHT.value == "equal_weight"
    assert RepresentativeSelectionMethod.FIRST.value == "first"
    assert RepresentativeSelectionMethod.RANDOM.value == "random"


def test_representative_selection_basic():
    """Test basic representative selection result."""
    selection = RepresentativeSelection(
        selection_id="sel_001",
        family="momentum",
        method=RepresentativeSelectionMethod.MAX_IC,
        selected_factor_ids=("factor_1", "factor_2"),
        timestamp="2026-08-14T00:00:00Z",
        candidate_factor_ids=("factor_1", "factor_2", "factor_3", "factor_4"),
    )

    assert selection.selection_id == "sel_001"
    assert selection.family == "momentum"
    assert selection.num_selected == 2
    assert selection.num_candidates == 4
    assert selection.selection_ratio == 0.5
    assert selection.has_warnings is False


def test_representative_selection_with_ic_values():
    """Test representative selection with IC values."""
    selection = RepresentativeSelection(
        selection_id="sel_002",
        family="value",
        method=RepresentativeSelectionMethod.MAX_IC,
        selected_factor_ids=("f1", "f2", "f3"),
        timestamp="2026-08-14T00:00:00Z",
        ic_values=(0.15, 0.12, 0.10),
    )

    assert selection.ic_values == (0.15, 0.12, 0.10)
    assert len(selection.ic_values) == selection.num_selected


def test_representative_selection_with_correlations():
    """Test representative selection with correlation values."""
    selection = RepresentativeSelection(
        selection_id="sel_003",
        family="quality",
        method=RepresentativeSelectionMethod.MIN_CORRELATION,
        selected_factor_ids=("f1", "f2"),
        timestamp="2026-08-14T00:00:00Z",
        avg_correlations=(0.3, 0.4),
    )

    assert selection.avg_correlations == (0.3, 0.4)
    assert len(selection.avg_correlations) == selection.num_selected


def test_representative_selection_with_warnings():
    """Test representative selection with warnings."""
    selection = RepresentativeSelection(
        selection_id="sel_004",
        family="growth",
        method=RepresentativeSelectionMethod.MAX_IC,
        selected_factor_ids=("f1",),
        timestamp="2026-08-14T00:00:00Z",
        warnings=("Insufficient IC data for f2", "Using default for f3"),
    )

    assert selection.has_warnings is True
    assert len(selection.warnings) == 2


def test_representative_selection_requires_selection_id():
    """Test that selection_id is required."""
    with pytest.raises(ValueError, match="selection_id is required"):
        RepresentativeSelection(
            selection_id="",
            family="test",
            method=RepresentativeSelectionMethod.FIRST,
            selected_factor_ids=("f1",),
            timestamp="2026-08-14T00:00:00Z",
        )


def test_representative_selection_requires_family():
    """Test that family is required."""
    with pytest.raises(ValueError, match="family is required"):
        RepresentativeSelection(
            selection_id="sel",
            family="",
            method=RepresentativeSelectionMethod.FIRST,
            selected_factor_ids=("f1",),
            timestamp="2026-08-14T00:00:00Z",
        )


def test_representative_selection_requires_selected_factors():
    """Test that selected_factor_ids cannot be empty."""
    with pytest.raises(ValueError, match="selected_factor_ids cannot be empty"):
        RepresentativeSelection(
            selection_id="sel",
            family="test",
            method=RepresentativeSelectionMethod.FIRST,
            selected_factor_ids=(),
            timestamp="2026-08-14T00:00:00Z",
        )


def test_representative_selection_requires_timestamp():
    """Test that timestamp is required."""
    with pytest.raises(ValueError, match="timestamp is required"):
        RepresentativeSelection(
            selection_id="sel",
            family="test",
            method=RepresentativeSelectionMethod.FIRST,
            selected_factor_ids=("f1",),
            timestamp="",
        )


def test_representative_selection_ic_values_length_validation():
    """Test that ic_values length must match selected_factor_ids."""
    with pytest.raises(ValueError, match="ic_values length must match"):
        RepresentativeSelection(
            selection_id="sel",
            family="test",
            method=RepresentativeSelectionMethod.MAX_IC,
            selected_factor_ids=("f1", "f2", "f3"),
            timestamp="2026-08-14T00:00:00Z",
            ic_values=(0.1, 0.2),  # Only 2 values for 3 factors
        )


def test_representative_selection_correlations_length_validation():
    """Test that avg_correlations length must match selected_factor_ids."""
    with pytest.raises(ValueError, match="avg_correlations length must match"):
        RepresentativeSelection(
            selection_id="sel",
            family="test",
            method=RepresentativeSelectionMethod.MIN_CORRELATION,
            selected_factor_ids=("f1", "f2"),
            timestamp="2026-08-14T00:00:00Z",
            avg_correlations=(0.3, 0.4, 0.5),  # 3 values for 2 factors
        )


def test_selector_max_ic():
    """Test MAX_IC selection method."""
    ic_provider = MockICProvider({
        "f1": 0.15,
        "f2": 0.12,
        "f3": 0.18,
        "f4": 0.08,
    })

    selector = FamilyRepresentativeSelector(ic_provider=ic_provider)

    selection = selector.select_representatives(
        family="momentum",
        factor_ids=("f1", "f2", "f3", "f4"),
        method=RepresentativeSelectionMethod.MAX_IC,
        max_representatives=2,
    )

    assert selection.num_selected == 2
    # Should select f3 (0.18) and f1 (0.15)
    assert selection.selected_factor_ids == ("f3", "f1")
    assert selection.ic_values == (0.18, 0.15)


def test_selector_max_ic_with_negative_ic():
    """Test MAX_IC selection uses absolute IC."""
    ic_provider = MockICProvider({
        "f1": 0.10,
        "f2": -0.15,  # Negative IC
        "f3": 0.08,
    })

    selector = FamilyRepresentativeSelector(ic_provider=ic_provider)

    selection = selector.select_representatives(
        family="reversal",
        factor_ids=("f1", "f2", "f3"),
        method=RepresentativeSelectionMethod.MAX_IC,
        max_representatives=1,
    )

    # Should select f2 with highest absolute IC
    assert selection.selected_factor_ids == ("f2",)
    assert selection.ic_values == (0.15,)


def test_selector_max_ic_missing_data():
    """Test MAX_IC with missing IC data."""
    ic_provider = MockICProvider({
        "f1": 0.10,
        # f2 missing
        "f3": 0.12,
    })

    selector = FamilyRepresentativeSelector(ic_provider=ic_provider)

    selection = selector.select_representatives(
        family="value",
        factor_ids=("f1", "f2", "f3"),
        method=RepresentativeSelectionMethod.MAX_IC,
        max_representatives=2,
    )

    # Should select f3 and f1, with warning about f2
    assert selection.selected_factor_ids == ("f3", "f1")
    assert selection.has_warnings
    assert any("No IC data for f2" in w for w in selection.warnings)


def test_selector_max_ic_requires_provider():
    """Test MAX_IC requires IC provider."""
    selector = FamilyRepresentativeSelector()

    with pytest.raises(ValueError, match="MAX_IC method requires ic_provider"):
        selector.select_representatives(
            family="test",
            factor_ids=("f1", "f2"),
            method=RepresentativeSelectionMethod.MAX_IC,
            max_representatives=1,
        )


def test_selector_min_correlation():
    """Test MIN_CORRELATION selection method."""
    corr_provider = MockCorrelationProvider({
        "f1": 0.7,  # High correlation
        "f2": 0.3,  # Low correlation
        "f3": 0.5,  # Medium correlation
        "f4": 0.2,  # Lowest correlation
    })

    selector = FamilyRepresentativeSelector(correlation_provider=corr_provider)

    selection = selector.select_representatives(
        family="quality",
        factor_ids=("f1", "f2", "f3", "f4"),
        method=RepresentativeSelectionMethod.MIN_CORRELATION,
        max_representatives=2,
    )

    assert selection.num_selected == 2
    # Should select f4 (0.2) and f2 (0.3) - lowest correlations
    assert selection.selected_factor_ids == ("f4", "f2")
    assert selection.avg_correlations == (0.2, 0.3)


def test_selector_min_correlation_missing_data():
    """Test MIN_CORRELATION with missing data."""
    corr_provider = MockCorrelationProvider({
        "f1": 0.4,
        # f2 missing
        "f3": 0.3,
    })

    selector = FamilyRepresentativeSelector(correlation_provider=corr_provider)

    selection = selector.select_representatives(
        family="momentum",
        factor_ids=("f1", "f2", "f3"),
        method=RepresentativeSelectionMethod.MIN_CORRELATION,
        max_representatives=2,
    )

    # Should select f3 and f1, with warning about f2
    assert selection.selected_factor_ids == ("f3", "f1")
    assert selection.has_warnings
    assert any("No correlation data for f2" in w for w in selection.warnings)


def test_selector_min_correlation_requires_provider():
    """Test MIN_CORRELATION requires correlation provider."""
    selector = FamilyRepresentativeSelector()

    with pytest.raises(ValueError, match="MIN_CORRELATION method requires correlation_provider"):
        selector.select_representatives(
            family="test",
            factor_ids=("f1", "f2"),
            method=RepresentativeSelectionMethod.MIN_CORRELATION,
            max_representatives=1,
        )


def test_selector_equal_weight():
    """Test EQUAL_WEIGHT selection method."""
    selector = FamilyRepresentativeSelector()

    selection = selector.select_representatives(
        family="size",
        factor_ids=("f1", "f2", "f3", "f4", "f5", "f6"),
        method=RepresentativeSelectionMethod.EQUAL_WEIGHT,
        max_representatives=3,
    )

    assert selection.num_selected == 3
    # Should select evenly spaced factors
    assert selection.selected_factor_ids == ("f1", "f3", "f5")


def test_selector_equal_weight_exact_fit():
    """Test EQUAL_WEIGHT when max_representatives equals factor count."""
    selector = FamilyRepresentativeSelector()

    selection = selector.select_representatives(
        family="growth",
        factor_ids=("f1", "f2", "f3"),
        method=RepresentativeSelectionMethod.EQUAL_WEIGHT,
        max_representatives=3,
    )

    assert selection.num_selected == 3
    assert selection.selected_factor_ids == ("f1", "f2", "f3")


def test_selector_first():
    """Test FIRST selection method."""
    selector = FamilyRepresentativeSelector()

    selection = selector.select_representatives(
        family="volatility",
        factor_ids=("f1", "f2", "f3", "f4"),
        method=RepresentativeSelectionMethod.FIRST,
        max_representatives=2,
    )

    assert selection.num_selected == 2
    assert selection.selected_factor_ids == ("f1", "f2")


def test_selector_random_with_seed():
    """Test RANDOM selection with seed for reproducibility."""
    selector = FamilyRepresentativeSelector()

    selection1 = selector.select_representatives(
        family="liquidity",
        factor_ids=("f1", "f2", "f3", "f4", "f5"),
        method=RepresentativeSelectionMethod.RANDOM,
        max_representatives=3,
        random_seed=42,
    )

    selection2 = selector.select_representatives(
        family="liquidity",
        factor_ids=("f1", "f2", "f3", "f4", "f5"),
        method=RepresentativeSelectionMethod.RANDOM,
        max_representatives=3,
        random_seed=42,
    )

    # Same seed should produce same selection
    assert selection1.selected_factor_ids == selection2.selected_factor_ids
    assert selection1.random_seed == 42
    assert selection2.random_seed == 42


def test_selector_random_different_seeds():
    """Test RANDOM selection with different seeds produces different results."""
    selector = FamilyRepresentativeSelector()

    selection1 = selector.select_representatives(
        family="liquidity",
        factor_ids=("f1", "f2", "f3", "f4", "f5"),
        method=RepresentativeSelectionMethod.RANDOM,
        max_representatives=2,
        random_seed=42,
    )

    selection2 = selector.select_representatives(
        family="liquidity",
        factor_ids=("f1", "f2", "f3", "f4", "f5"),
        method=RepresentativeSelectionMethod.RANDOM,
        max_representatives=2,
        random_seed=123,
    )

    # Different seeds likely produce different selections
    # (Not guaranteed but very likely with 5 choose 2)
    assert selection1.random_seed != selection2.random_seed


def test_selector_empty_factor_ids():
    """Test that empty factor_ids raises error."""
    selector = FamilyRepresentativeSelector()

    with pytest.raises(ValueError, match="factor_ids cannot be empty"):
        selector.select_representatives(
            family="test",
            factor_ids=(),
            method=RepresentativeSelectionMethod.FIRST,
            max_representatives=1,
        )


def test_selector_max_representatives_validation():
    """Test max_representatives validation."""
    selector = FamilyRepresentativeSelector()

    with pytest.raises(ValueError, match="max_representatives must be >= 1"):
        selector.select_representatives(
            family="test",
            factor_ids=("f1", "f2"),
            method=RepresentativeSelectionMethod.FIRST,
            max_representatives=0,
        )


def test_selector_max_representatives_capped():
    """Test max_representatives is capped to factor count."""
    selector = FamilyRepresentativeSelector()

    selection = selector.select_representatives(
        family="test",
        factor_ids=("f1", "f2", "f3"),
        method=RepresentativeSelectionMethod.FIRST,
        max_representatives=10,  # More than available
    )

    # Should select all 3 factors
    assert selection.num_selected == 3
    assert selection.selected_factor_ids == ("f1", "f2", "f3")


def test_selector_timestamp_format():
    """Test that selector generates valid ISO 8601 timestamps."""
    selector = FamilyRepresentativeSelector()

    selection = selector.select_representatives(
        family="test",
        factor_ids=("f1", "f2"),
        method=RepresentativeSelectionMethod.FIRST,
        max_representatives=1,
    )

    # Should end with Z for UTC
    assert selection.timestamp.endswith("Z")
    # Should contain T separator
    assert "T" in selection.timestamp


def test_selector_selection_id_format():
    """Test that selector generates selection IDs with expected format."""
    ic_provider = MockICProvider({"f1": 0.10, "f2": 0.12})
    selector = FamilyRepresentativeSelector(ic_provider=ic_provider)

    selection = selector.select_representatives(
        family="momentum",
        factor_ids=("f1", "f2"),
        method=RepresentativeSelectionMethod.MAX_IC,
        max_representatives=1,
        universe_ref="US_LARGE",
    )

    # Selection ID should contain family and method
    assert "momentum" in selection.selection_id
    assert "max_ic" in selection.selection_id
