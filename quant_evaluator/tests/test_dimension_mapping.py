"""
Tests for the dimension metric registry mapping
(``quant_evaluator.metrics.dimensions``).

These verify the mapping is consistent with the single MetricRegistry
authority: every id referenced in DIMENSION_METRICS must actually be
registered, and unknown ids must resolve to None (fail-closed).
"""

from __future__ import annotations

import pytest

from quant_evaluator.metrics.dimensions import (
    COMPLEXITY,
    DIMENSION_METRICS,
    DimensionName,
    collect_dimension_metric_ids,
    dimension_for,
)
from quant_evaluator.registry.metrics import MetricRegistry, list_metrics


def _registered() -> set:
    return set(list_metrics())


def test_every_dimension_has_registered_metrics():
    """Each dimension maps to >=1 metric_id actually present in the registry.

    PURITY_EXPOSURE is a documented gap: no exposure metrics are registered in
    the MetricRegistry yet (a sibling owns exposure extension), so it is
    excluded from this assertion and checked separately below.
    """
    registered = _registered()
    for dimension in DimensionName:
        if dimension is DimensionName.PURITY_EXPOSURE:
            continue
        ids = DIMENSION_METRICS[dimension]
        assert len(ids) >= 1, f"{dimension} maps to no metrics"
        assert any(mid in registered for mid in ids), (
            f"{dimension} has no metric_id that is actually registered"
        )


def test_purity_exposure_is_documented_gap():
    """PURITY_EXPOSURE currently maps to no registered metrics (documented gap)."""
    assert DIMENSION_METRICS[DimensionName.PURITY_EXPOSURE] == ()


def test_metric_resolves_to_its_dimension():
    """rank_ic -> PREDICTIVE; a turnover metric -> TRADABILITY."""
    assert dimension_for("rank_ic") is DimensionName.PREDICTIVE
    assert dimension_for("turnover") is DimensionName.TRADABILITY
    assert dimension_for("factor_turnover_rate") is DimensionName.TRADABILITY


def test_unknown_metric_returns_none():
    """An unknown metric_id resolves to None (fail-closed, not silently bucketed)."""
    assert dimension_for("not_a_metric") is None
    assert dimension_for("") is None


def test_consistency_with_registry():
    """Every id in DIMENSION_METRICS is a real registered metric (no invented ids)."""
    registered = _registered()
    for dimension, ids in DIMENSION_METRICS.items():
        for mid in ids:
            assert mid in registered, (
                f"{dimension} references unregistered metric_id {mid!r}"
            )


def test_collect_dimension_metric_ids_filters_to_registered():
    """collect_dimension_metric_ids returns only registered ids."""
    registered = _registered()
    for dimension in DimensionName:
        collected = collect_dimension_metric_ids(dimension)
        assert set(collected) <= registered
        # collected must equal the registered subset of DIMENSION_METRICS
        expected = sorted(
            mid for mid in DIMENSION_METRICS[dimension] if mid in registered
        )
        assert list(collected) == expected


def test_collect_dimension_metric_ids_with_registry_instance():
    """collect_dimension_metric_ids resolves through a MetricRegistry instance."""
    registry = MetricRegistry()
    # A fresh empty registry knows no ids -> every dimension collects nothing.
    for dimension in DimensionName:
        assert collect_dimension_metric_ids(dimension, registry=registry) == ()


def test_complexity_is_separate_penalty_dimension():
    """COMPLEXITY is a separate penalty dimension, not a DimensionName member."""
    assert COMPLEXITY == "complexity"
    assert not any(d.value == COMPLEXITY for d in DimensionName)
