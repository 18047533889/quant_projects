import numpy as np
import pytest


@pytest.mark.parametrize("cost_rate", [True, np.bool_(True), np.complex128(.001 + .5j)])
def test_portfolio_series_rejects_boolean_cost_rates(cost_rate):
    from factor_optimizer.research_fitness import portfolio_series

    values = np.tile(np.arange(20, dtype=float), (20, 1))
    with pytest.raises(ValueError, match="cost_rate"):
        portfolio_series(values, np.zeros_like(values), cost_rate=cost_rate)


@pytest.mark.parametrize("periods_per_year", [
    True, np.bool_(True), 0, -1, np.inf, np.complex128(252 + 1j),
])
def test_summarize_rejects_invalid_annualization(periods_per_year):
    from factor_optimizer.research_fitness import summarize

    rng = np.random.default_rng(20260922)
    series = np.column_stack((rng.normal(0, .1, 30),
                              rng.normal(0, .01, 30), np.ones(30)))
    with pytest.raises(ValueError, match="periods_per_year"):
        summarize(series, periods_per_year=periods_per_year)


def test_summarize_accepts_positive_real_annualization():
    from factor_optimizer.research_fitness import summarize

    rng = np.random.default_rng(20260924)
    series = np.column_stack((rng.normal(0, .1, 30),
                              rng.normal(0, .01, 30), np.ones(30)))
    assert summarize(series, periods_per_year=252.0) == summarize(
        series, periods_per_year=np.int64(252))


@pytest.mark.parametrize("indices", [
    tuple(range(-20, 0)),
    (0, 1, 1),
    np.array([[0, 1]]),
    np.array([True] * 40),
    np.array([2, 1], dtype=np.uint8),
    (0, 40),
])
def test_paired_series_rejects_ambiguous_or_invalid_indices(indices):
    from factor_optimizer.research_fitness import paired_series
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    rng = np.random.default_rng(20260923)
    values = rng.normal(size=(40, 20))
    time_axis = AxisRef("time", "int", 40, np.arange(40))
    asset_axis = AxisRef("asset", "str", 20, np.arange(20).astype(str))
    batch = FactorBatch(("x",), time_axis, asset_axis, values[:, :, None])
    labels = LabelBundle("returns", rng.normal(0, .01, values.shape), 1,
        decision_time=tuple(range(40)), label_start_time=tuple(range(1, 41)),
        label_end_time=tuple(range(2, 42)), asset_axis=asset_axis)

    with pytest.raises(ValueError, match="indices"):
        paired_series(values, values, batch, labels, indices)
