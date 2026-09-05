import numpy as np
from quant_evaluator.metrics import portfolio_stats

from factor_engine.reporting.quant_evaluator_adapter import (
    evaluate_report_arrays,
)


def test_report_metrics_share_one_non_missing_long_short_series():
    factors = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    returns = np.array(
        [
            [-0.04, -0.02, 0.02, 0.04],
            [0.10, 0.10, 0.10, 0.10],
            [0.04, 0.02, -0.02, -0.04],
        ]
    )

    result = evaluate_report_arrays(factors, returns, n_quantiles=2, min_assets=1)

    assert result.valid_return_periods == 2
    assert np.isnan(result.long_short_returns[1])
    assert result.long_short_nav.shape == (2,)
    assert np.isclose(result.cumulative_return, result.long_short_nav[-1] - 1.0)
    assert np.isclose(result.win_rate, 1.0)


def test_training_direction_flips_factor_and_all_dependent_outputs():
    factors = np.tile(np.arange(1.0, 11.0), (4, 1))
    returns = np.tile(np.linspace(0.05, -0.05, 10), (4, 1))

    result = evaluate_report_arrays(
        factors,
        returns,
        n_quantiles=10,
        min_assets=1,
        direction_training_periods=2,
        min_ic_periods=2,
    )

    assert result.direction == -1
    assert result.mean_rank_ic > 0
    assert np.all(result.long_short_returns[np.isfinite(result.long_short_returns)] > 0)
    assert np.all(result.quantile_returns[:, -1] > result.quantile_returns[:, 0])


def test_partial_quantile_day_keeps_aligned_nav_and_gap():
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(40, 100))
    returns = rng.normal(scale=0.01, size=(40, 100))
    returns[0, :5] = np.nan

    result = evaluate_report_arrays(
        factors,
        returns,
        n_quantiles=10,
        min_assets=10,
        direction_training_periods=20,
        min_ic_periods=20,
    )

    assert result.quantile_nav.shape == (40, 10)
    assert np.isnan(result.quantile_nav[0]).any()
    assert np.isfinite(result.quantile_nav[-1]).all()


def test_long_short_cost_charges_both_legs():
    net = portfolio_stats.apply_long_short_costs(
        np.array([0.03]),
        np.array([0.01]),
        long_turnover=np.array([0.5]),
        short_turnover=np.array([0.25]),
        cost_rate=0.001,
    )

    assert np.isclose(net[0], 0.03 - 0.01 - 0.5 * 0.001 - 0.25 * 0.001)


def test_batch_evaluation_applies_training_direction_per_factor():
    from factor_engine.reporting import quant_evaluator_adapter as adapter

    ascending = np.tile(np.arange(1.0, 101.0), (30, 1))
    factors = np.stack([ascending, -ascending], axis=-1)
    returns = np.tile(np.linspace(-0.05, 0.05, 100), (30, 1))

    result = adapter.evaluate_report_batch(
        factors,
        returns,
        factor_ids=("positive", "negative"),
        backend="cpu",
        direction_training_periods=20,
        min_ic_periods=20,
    )

    assert result.backend_used == "cpu"
    assert result.factors["positive"].direction == 1
    assert result.factors["negative"].direction == -1
    assert result.factors["positive"].mean_rank_ic > 0
    assert result.factors["negative"].mean_rank_ic > 0


def test_single_factor_entrypoint_delegates_to_batch_authority(monkeypatch):
    from types import SimpleNamespace
    from factor_engine.reporting import quant_evaluator_adapter as adapter

    sentinel = object()
    calls = []

    def batch(factors, returns, **options):
        calls.append((factors.shape, returns.shape, options))
        return SimpleNamespace(factors={"report_factor": sentinel})

    monkeypatch.setattr(adapter, "evaluate_report_batch", batch)
    result = adapter.evaluate_report_arrays(
        np.ones((3, 4)), np.zeros((3, 4)), n_quantiles=2, min_assets=1,
    )

    assert result is sentinel
    assert calls[0][0] == (3, 4, 1)
    assert calls[0][2]["factor_ids"] == ("report_factor",)
