"""Decimal oracles for cross-time quantile mean aggregation."""
from decimal import Decimal, localcontext
import warnings

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics import registry_adapters


def _decimal_mean(values):
    finite = [float(value) for value in np.asarray(values).reshape(-1)
              if np.isfinite(value)]
    if not finite:
        return np.nan
    with localcontext() as context:
        context.prec = 2000
        total = sum((Decimal.from_float(value) for value in finite), Decimal(0))
        return float(total / Decimal(len(finite)))


def _contracts(times=25, assets=10):
    time_axis = AxisRef("time", "int64", times, np.arange(times, dtype=np.int64))
    asset_axis = AxisRef("asset", "str", assets,
                         np.array([f"a{i}" for i in range(assets)]))
    batch = FactorBatch(
        factor_ids=("f0",), time_axis=time_axis, asset_axis=asset_axis,
        values=np.tile(np.arange(assets, dtype=np.float64), (times, 1))[:, :, None],
    )
    label = LabelBundle(
        target_id="temporal-quantile-mean", values=np.zeros((times, assets)),
        horizon=1, decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )
    return batch, label


def _stub_daily(monkeypatch, values, counts=None):
    values = np.asarray(values, dtype=np.float64)
    if counts is None:
        counts = np.ones(values.shape, dtype=np.int32)
    else:
        counts = np.asarray(counts, dtype=np.int32)
    before = counts.copy()

    def compute(*args, **kwargs):
        return values, counts

    monkeypatch.setattr(registry_adapters, "compute_quantile_returns_fast", compute)
    return counts, before


def test_public_full_quantile_mean_avoids_overflow_for_huge_constant():
    batch, label = _contracts()
    label = LabelBundle(
        target_id=label.target_id, values=np.full((25, 10), 1e308), horizon=1,
        decision_time=label.decision_time, label_start_time=label.label_start_time,
        label_end_time=label.label_end_time, asset_axis=label.asset_axis,
    )

    result = registry_adapters.compute_quantile_returns_full_value(
        batch, label, min_periods=20, n_quantiles=1)

    assert result[0, 0] == 1e308


def test_public_quantile_spread_mean_avoids_overflow_for_huge_constant():
    batch, label = _contracts(assets=20)
    label_values = np.zeros((25, 20), dtype=np.float64)
    label_values[:, 10:] = 1e308
    label = LabelBundle(
        target_id=label.target_id, values=label_values, horizon=1,
        decision_time=label.decision_time, label_start_time=label.label_start_time,
        label_end_time=label.label_end_time, asset_axis=label.asset_axis,
    )

    result = registry_adapters.compute_quantile_spread_value(
        batch, label, min_periods=20, n_quantiles=2)

    assert result[0] == 1e308


def test_finite_temporal_mean_matches_decimal_huge_and_cancellation_cases():
    from quant_evaluator.metrics.finite_mean import finite_mean_axis0

    cases = (
        np.full((25, 1), 1e308),
        np.array([[1e308], [3.0], [-1e308], [0.0]]),
    )
    for values in cases:
        mean, counts = finite_mean_axis0(values, min_periods=len(values))
        assert counts[0] == len(values)
        assert mean[0] == _decimal_mean(values)


def test_finite_temporal_mean_filters_nonfinite_values_and_enforces_threshold():
    from quant_evaluator.metrics.finite_mean import finite_mean_axis0

    values = np.array([
        [1e308, 1e308], [np.nan, 3.0], [np.inf, -1e308],
        [-np.inf, 0.0], [1e308, np.nan],
    ])
    means, counts = finite_mean_axis0(values, min_periods=4)
    np.testing.assert_array_equal(counts, [2, 4])
    assert np.isnan(means[0])
    assert means[1] == _decimal_mean([1e308, 3.0, -1e308, 0.0]) == 0.75


@pytest.mark.parametrize("min_periods", [0, -1, True, np.bool_(False), 1.5])
def test_finite_temporal_mean_rejects_invalid_min_periods(min_periods):
    from quant_evaluator.metrics.finite_mean import finite_mean_axis0

    with pytest.raises(ValueError, match="positive integer"):
        finite_mean_axis0(np.array([[1.0], [np.nan]]), min_periods=min_periods)
    with pytest.raises(ValueError, match="positive integer"):
        finite_mean_axis0(np.array([[np.nan]]), min_periods=min_periods)


def test_finite_temporal_mean_accepts_numpy_integer_min_periods():
    from quant_evaluator.metrics.finite_mean import finite_mean_axis0

    means, counts = finite_mean_axis0(np.array([[2.0], [4.0]]), min_periods=np.int64(2))
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(means, [3.0])


def test_finite_temporal_mean_ordinary_seeded_data_keeps_vectorized_path(monkeypatch):
    from quant_evaluator.metrics import finite_mean

    rng = np.random.default_rng(61003)
    values = rng.normal(size=(64, 3, 2))
    values[2, 0, 0] = np.nan
    values[4, 1, 1] = np.inf
    values[7, 2, 0] = -np.inf
    monkeypatch.setattr(
        finite_mean, "stable_finite_mean",
        lambda *_: pytest.fail("ordinary finite means should stay vectorized"),
    )

    means, counts = finite_mean.finite_mean_axis0(values, min_periods=20)
    finite = np.isfinite(values)
    expected_counts = finite.sum(axis=0)
    expected = np.divide(
        np.where(finite, values, 0.0).sum(axis=0), expected_counts,
        out=np.full(expected_counts.shape, np.nan), where=expected_counts > 0,
    )
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(means, expected, rtol=1e-14, atol=1e-15)


def test_temporal_mean_empty_slices_return_nan_without_warnings(monkeypatch):
    from quant_evaluator.metrics.finite_mean import finite_mean_axis0

    values = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        means, counts = finite_mean_axis0(values, min_periods=1)
    np.testing.assert_array_equal(counts, np.zeros(2, dtype=np.int64))
    assert np.isnan(means).all()
    assert caught == []

    batch, label = _contracts(times=2, assets=2)
    _stub_daily(monkeypatch, np.array([[[np.nan]], [[np.inf]]]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = registry_adapters.compute_quantile_returns_full_value(
            batch, label, min_periods=1, n_quantiles=1)
    assert np.isnan(result).all()
    assert caught == []

    _stub_daily(monkeypatch, np.full((2, 2, 1), np.inf))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spread = registry_adapters.compute_quantile_spread_value(
            batch, label, min_periods=1, n_quantiles=2)
    assert np.isnan(spread).all()
    assert caught == []
