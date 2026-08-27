"""DLIB-FA-018 composite evaluation tests (vwap-to-vwap return basis)."""

import pytest

from factor_assets.aggregation.composite import (
    CompositeEvaluator,
    OrientationType,
    RegimeType,
)
from factor_assets.aggregation.specs import (
    AggregationResult,
    AggregationSpec,
    WeightingScheme,
)
from factor_assets.errors import (
    EvidenceUnavailableError,
    InsufficientObservations,
    NumericalFailure,
)


def make_spec(factor_ids=("F1", "F2")):
    return AggregationSpec(
        spec_id="agg1",
        name="A1",
        weighting_scheme=WeightingScheme.EQUAL,
        factor_ids=factor_ids,
    )


def test_composite_evaluator_vwap_to_vwap_basis():
    """Composite aggregates the same-frequency vwap-to-vwap forward return —
    the global hard return basis (Memory: vwap-to-vwap 收益口径)."""
    spec = make_spec()
    ev = CompositeEvaluator(horizon=2, orientation=OrientationType.LONG)
    value = ev.compose(
        "CF1",
        spec,
        {"F1": [0.01, 0.02, 0.03], "F2": [0.02, 0.01, 0.02]},
        agg_freq="daily",
        n_observations=100,
        evidence_basis_ref="ev1",
    )
    assert value.composite_factor_id == "CF1"
    assert value.horizon == 2
    assert value.orientation is OrientationType.LONG
    assert value.source == "CCCC"
    # F1 horizon-2 mean = (0.01+0.02)/2 = 0.015; F2 = 0.015; composite = 0.015.
    assert value.agg_vwap_to_vwap_return == pytest.approx(0.015)


def test_composite_short_flips_sign():
    spec = make_spec()
    ev = CompositeEvaluator(horizon=1, orientation=OrientationType.SHORT)
    value = ev.compose(
        "CF1",
        spec,
        {"F1": [0.01], "F2": [0.03]},
        agg_freq="daily",
        n_observations=1,
        evidence_basis_ref="ev1",
    )
    # raw = 0.02 -> SHORT flips to -0.02.
    assert value.agg_vwap_to_vwap_return == pytest.approx(-0.02)


def test_composite_missing_factor_fails_closed():
    spec = make_spec(("F1", "F2", "F3"))
    ev = CompositeEvaluator()
    with pytest.raises(EvidenceUnavailableError):
        ev.compose(
            "CF1",
            spec,
            {"F1": [0.01], "F2": [0.02]},  # F3 missing
            agg_freq="daily",
            n_observations=1,
            evidence_basis_ref="ev1",
        )


def test_composite_insufficient_observations_fails_closed():
    spec = make_spec()
    ev = CompositeEvaluator(horizon=5)
    with pytest.raises(InsufficientObservations):
        ev.compose(
            "CF1",
            spec,
            {"F1": [0.01, 0.02], "F2": [0.02, 0.01]},
            agg_freq="daily",
            n_observations=2,
            evidence_basis_ref="ev1",
        )


def test_composite_non_finite_return_fails_closed():
    spec = make_spec()
    ev = CompositeEvaluator(horizon=1)
    with pytest.raises(NumericalFailure):
        ev.compose(
            "CF1",
            spec,
            {"F1": [float("nan")], "F2": [0.02]},
            agg_freq="daily",
            n_observations=1,
            evidence_basis_ref="ev1",
        )


def test_accounting_diagnostics_are_weight_only_not_backtest():
    """FA reports weight-derived accounting diagnostics ONLY — no fabricated
    long/short portfolio backtest (DLIB-FA-018)."""
    spec = make_spec(("F1", "F2", "F3", "F4"))
    result = AggregationResult(
        result_id="r1",
        spec=spec,
        composite_factor_id="CF1",
        timestamp="2024-01-01T00:00:00Z",
        computed_weights=(0.4, 0.3, 0.2, 0.1),
    )
    ev = CompositeEvaluator()
    diag = ev.accounting_diagnostics(result)
    assert diag["available"] is True
    assert diag["weight_concentration_hhi"] == pytest.approx(0.3)
    assert diag["effective_num_factors"] == pytest.approx(1 / 0.3)
    # No fabricated backtest PnL / long-short return in FA diagnostics.
    assert "long_short_pnl" not in diag


def test_composite_regime_weights():
    spec = make_spec()
    ev = CompositeEvaluator(
        horizon=1,
        orientation=OrientationType.LONG,
        regime_weights={"UP": 1.0, "DOWN": 0.5},
    )
    value = ev.compose(
        "CF1",
        spec,
        {"F1": [0.01], "F2": [0.03]},
        agg_freq="daily",
        n_observations=1,
        evidence_basis_ref="ev1",
    )
    assert value.regime is RegimeType.ALL
    # raw 0.02 x regime weights (1.0 + 0.5) / 2 = 0.015.
    assert value.agg_vwap_to_vwap_return == pytest.approx(0.015)