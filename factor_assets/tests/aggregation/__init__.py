"""Tests for aggregation module init."""

from factor_assets.aggregation import (
    WeightingScheme,
    AggregationSpec,
    AggregationResult,
    RepresentativeSelectionMethod,
    RepresentativeSelection,
    FamilyRepresentativeSelector,
)


def test_aggregation_module_exports_specs():
    """Test that aggregation module exports spec types."""
    assert WeightingScheme is not None
    assert AggregationSpec is not None
    assert AggregationResult is not None


def test_aggregation_module_exports_representatives():
    """Test that aggregation module exports representative types."""
    assert RepresentativeSelectionMethod is not None
    assert RepresentativeSelection is not None
    assert FamilyRepresentativeSelector is not None


def test_weighting_scheme_accessible():
    """Test that WeightingScheme enum is accessible."""
    assert hasattr(WeightingScheme, 'EQUAL')
    assert hasattr(WeightingScheme, 'IC_WEIGHTED')
    assert hasattr(WeightingScheme, 'INVERSE_VARIANCE')


def test_representative_method_accessible():
    """Test that RepresentativeSelectionMethod enum is accessible."""
    assert hasattr(RepresentativeSelectionMethod, 'MAX_IC')
    assert hasattr(RepresentativeSelectionMethod, 'MIN_CORRELATION')
    assert hasattr(RepresentativeSelectionMethod, 'EQUAL_WEIGHT')
