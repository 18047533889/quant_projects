"""Tests for graph.edges module."""

import pytest
from factor_assets.graph.sparse import CorrelationEdge
from factor_assets.graph.edges import (
    ThresholdFilter,
    TopKFilter,
    CompositeFilter,
    SignFilter,
)


def test_threshold_filter():
    """ThresholdFilter keeps edges above threshold."""
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.5),
        CorrelationEdge("C", "D", 0.3),
        CorrelationEdge("D", "E", -0.8),
    ]

    filter = ThresholdFilter(0.6)
    filtered = filter.filter(edges)

    assert len(filtered) == 2
    pairs = {edge.canonical_form() for edge in filtered}
    assert ("A", "B") in pairs
    assert ("D", "E") in pairs  # Negative but |corr| >= 0.6


def test_threshold_filter_validation():
    """ThresholdFilter validates threshold range."""
    with pytest.raises(ValueError, match="Threshold must be in"):
        ThresholdFilter(-0.1)

    with pytest.raises(ValueError, match="Threshold must be in"):
        ThresholdFilter(1.5)

    # Valid thresholds
    ThresholdFilter(0.0)
    ThresholdFilter(1.0)
    ThresholdFilter(0.5)


def test_topk_filter():
    """TopKFilter keeps top-K edges per node."""
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("A", "C", 0.8),
        CorrelationEdge("A", "D", 0.7),
        CorrelationEdge("B", "C", 0.6),
    ]

    # Keep top-2 per node
    filter = TopKFilter(2)
    filtered = filter.filter(edges)

    # A should keep B and C (highest correlations)
    # B should keep A and C
    # C should keep A and B
    # D should keep A
    # This results in 4 edges kept: A-B, A-C, A-D, B-C
    assert len(filtered) == 4
    pairs = {edge.canonical_form() for edge in filtered}
    assert ("A", "B") in pairs
    assert ("A", "C") in pairs
    assert ("B", "C") in pairs
    assert ("A", "D") in pairs


def test_topk_filter_k_zero():
    """TopKFilter with k=0 returns empty."""
    edges = [
        CorrelationEdge("A", "B", 0.9),
        CorrelationEdge("B", "C", 0.8),
    ]

    filter = TopKFilter(0)
    filtered = filter.filter(edges)

    assert len(filtered) == 0


def test_topk_filter_validation():
    """TopKFilter validates k."""
    with pytest.raises(ValueError, match="k must be non-negative"):
        TopKFilter(-1)

    # Valid k
    TopKFilter(0)
    TopKFilter(5)


def test_composite_filter():
    """CompositeFilter applies multiple filters."""
    edges = [
        CorrelationEdge("A", "B", 0.95),
        CorrelationEdge("A", "C", 0.85),
        CorrelationEdge("A", "D", 0.75),
        CorrelationEdge("B", "C", 0.65),
        CorrelationEdge("C", "D", 0.55),
        CorrelationEdge("D", "E", 0.45),
    ]

    # First filter by threshold, then top-K
    filter = CompositeFilter(
        ThresholdFilter(0.6),
        TopKFilter(2)
    )

    filtered = filter.filter(edges)

    # After threshold: A-B, A-C, A-D, B-C remain
    # After top-K: should keep strongest connections
    assert len(filtered) <= 4


def test_sign_filter_positive():
    """SignFilter keeps positive correlations."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", -0.6),
        CorrelationEdge("C", "D", 0.7),
        CorrelationEdge("D", "E", -0.5),
    ]

    filter = SignFilter('positive')
    filtered = filter.filter(edges)

    assert len(filtered) == 2
    pairs = {edge.canonical_form() for edge in filtered}
    assert ("A", "B") in pairs
    assert ("C", "D") in pairs


def test_sign_filter_negative():
    """SignFilter keeps negative correlations."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", -0.6),
        CorrelationEdge("C", "D", 0.7),
        CorrelationEdge("D", "E", -0.5),
    ]

    filter = SignFilter('negative')
    filtered = filter.filter(edges)

    assert len(filtered) == 2
    pairs = {edge.canonical_form() for edge in filtered}
    assert ("B", "C") in pairs
    assert ("D", "E") in pairs


def test_sign_filter_both():
    """SignFilter with 'both' keeps all edges."""
    edges = [
        CorrelationEdge("A", "B", 0.8),
        CorrelationEdge("B", "C", -0.6),
    ]

    filter = SignFilter('both')
    filtered = filter.filter(edges)

    assert len(filtered) == 2


def test_sign_filter_validation():
    """SignFilter validates sign parameter."""
    with pytest.raises(ValueError, match="sign must be"):
        SignFilter('invalid')

    # Valid signs
    SignFilter('positive')
    SignFilter('negative')
    SignFilter('both')


def test_empty_edge_list():
    """Filters handle empty edge lists."""
    edges = []

    assert ThresholdFilter(0.5).filter(edges) == []
    assert TopKFilter(3).filter(edges) == []
    assert SignFilter('positive').filter(edges) == []
    assert CompositeFilter(ThresholdFilter(0.5)).filter(edges) == []
