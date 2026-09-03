import numpy as np

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
