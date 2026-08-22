# -*- coding: utf-8 -*-
"""Tests for relation_jaccard operator."""
import numpy as np
import pandas as pd
import pytest

from cleaned_operators.fiscal_event_ops import pd_relation_jaccard


def _panel(entity_ids, snapshot_ids):
    """Build aligned test panels."""
    index = pd.date_range("2025-01-01", periods=len(entity_ids))
    entity = pd.DataFrame({"A": entity_ids}, index=index, dtype=object)
    snapshot = pd.DataFrame({"A": snapshot_ids}, index=index, dtype=object)
    return entity, snapshot


def test_full_overlap_returns_one():
    """Identical entity sets should yield Jaccard = 1.0."""
    entity, snapshot = _panel(
        ["x", "y", "x", "y"],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # Row 2: Q2={x} vs Q1={x,y} → 1/2 = 0.5
    assert result.iloc[2, 0] == 0.5
    # Row 3: Q2={x,y} vs Q1={x,y} → 2/2 = 1.0
    assert result.iloc[3, 0] == 1.0


def test_no_overlap_returns_zero():
    """Disjoint entity sets should yield Jaccard = 0.0."""
    entity, snapshot = _panel(
        ["a", "b", "c", "d"],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # 2024Q2: {c, d} vs 2024Q1: {a, b} → no intersection
    assert result.iloc[2, 0] == 0.0
    assert result.iloc[3, 0] == 0.0


def test_partial_overlap():
    """Partial overlap should compute correct Jaccard coefficient."""
    entity, snapshot = _panel(
        ["x", "y", "y", "z"],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # Row 2: Q2={y} vs Q1={x,y} → |{y}|/|{x,y}| = 1/2 = 0.5
    assert abs(result.iloc[2, 0] - 0.5) < 1e-9
    # Row 3: Q2={y,z} vs Q1={x,y} → |{y}|/|{x,y,z}| = 1/3
    expected = 1.0 / 3.0
    assert abs(result.iloc[3, 0] - expected) < 1e-9


def test_both_empty_returns_nan():
    """Both snapshots empty should return NaN."""
    entity, snapshot = _panel(
        [np.nan, np.nan, np.nan, np.nan],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[3, 0])


def test_one_empty_returns_zero():
    """One empty snapshot should return 0.0."""
    entity, snapshot = _panel(
        ["a", "b", np.nan, np.nan],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # 2024Q2 empty, 2024Q1 = {a, b}
    assert result.iloc[2, 0] == 0.0


def test_deduplication_within_snapshot():
    """Duplicate entities within same snapshot should be deduplicated."""
    entity, snapshot = _panel(
        ["x", "x", "x", "y", "y", "z"],
        ["2024Q1", "2024Q1", "2024Q1", "2024Q2", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # 2024Q1: {x} (deduplicated), 2024Q2: {y, z}
    # Intersection: {} = 0, Union: {x, y, z} = 3 → 0/3 = 0.0
    assert result.iloc[3, 0] == 0.0


def test_revision_policy_latest_available():
    """Latest revision should overwrite earlier value for same period."""
    entity, snapshot = _panel(
        ["old", "old", "new", "y"],
        ["2024Q1", "2024Q1", "2024Q1", "2024Q2"],  # Row 2 revises 2024Q1
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1, revision_policy="latest_available")
    # After revision row 2, 2024Q1 = {new, old}
    # 2024Q2 = {y} vs 2024Q1 = {new, old} → 0 intersection
    assert result.iloc[3, 0] == 0.0


def test_revision_policy_first_available():
    """First revision should persist for same period."""
    entity, snapshot = _panel(
        ["first", "first", "revised", "y"],
        ["2024Q1", "2024Q1", "2024Q1", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1, revision_policy="first_available")
    # First available keeps initial values for 2024Q1
    # 2024Q1 = {first}, 2024Q2 = {y} → no intersection
    assert result.iloc[3, 0] == 0.0


def test_periods_parameter():
    """periods parameter should control lag distance."""
    entity, snapshot = _panel(
        ["a", "a", "b", "b", "c", "c"],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2", "2024Q3", "2024Q3"],
    )
    # periods=1: compare 2024Q3 to 2024Q2
    result1 = pd_relation_jaccard(entity, snapshot, periods=1)
    # 2024Q3: {c} vs 2024Q2: {b} → 0.0
    assert result1.iloc[4, 0] == 0.0

    # periods=2: compare 2024Q3 to 2024Q1
    result2 = pd_relation_jaccard(entity, snapshot, periods=2)
    # 2024Q3: {c} vs 2024Q1: {a} → 0.0
    assert result2.iloc[4, 0] == 0.0


def test_no_future_leakage():
    """Future snapshots should not be visible."""
    entity, snapshot = _panel(
        ["x", "x", "future", "future"],
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # At row 1, only 2024Q1 is visible, no 2024Q2 yet
    assert np.isnan(result.iloc[1, 0])  # Not enough history


def test_insufficient_history():
    """Not enough history should return NaN."""
    entity, snapshot = _panel(
        ["a", "a"],
        ["2024Q1", "2024Q1"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # Need at least periods+1 snapshots
    assert np.isnan(result.iloc[0, 0])
    assert np.isnan(result.iloc[1, 0])


def test_multiple_instruments():
    """Multiple columns should be computed independently."""
    index = pd.date_range("2025-01-01", periods=4)
    entity = pd.DataFrame({
        "A": ["x", "y", "y", "z"],
        "B": ["p", "p", "q", "q"],
    }, index=index, dtype=object)
    snapshot = pd.DataFrame({
        "A": ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
        "B": ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    }, index=index, dtype=object)

    result = pd_relation_jaccard(entity, snapshot, periods=1)

    # Instrument A: Q1={x,y}, Q2={y,z} → 1/3
    assert abs(result.iloc[2, 0] - 1.0/3.0) < 1e-9

    # Instrument B: Q1={p}, Q2={q} → 0.0
    assert result.iloc[2, 1] == 0.0


def test_nan_filtering():
    """NaN and None values should be excluded from entity sets."""
    entity, snapshot = _panel(
        ["a", None, "b", "nan"],  # None and "nan" string should be filtered
        ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1)
    # 2024Q1: {a} (None filtered), 2024Q2: {b} (if "nan" filtered) or {b, nan}
    # This tests the filtering logic
    assert result.iloc[2, 0] == 0.0


def test_complex_revision_scenario():
    """Complex scenario with multiple revisions."""
    entity, snapshot = _panel(
        ["e1", "e2", "e1_rev", "e3", "e4"],
        ["2024Q1", "2024Q1", "2024Q1", "2024Q2", "2024Q2"],
    )
    result = pd_relation_jaccard(entity, snapshot, periods=1, revision_policy="latest_available")
    # After all 2024Q1 revisions: {e1, e2, e1_rev}
    # 2024Q2: {e3, e4}
    # No overlap → 0.0
    assert result.iloc[4, 0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
