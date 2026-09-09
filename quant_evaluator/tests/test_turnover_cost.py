import numpy as np
import pytest

from quant_evaluator.metrics.turnover_cost import compute_turnover_cost


def test_realized_cost_rate_is_averaged_and_reported_in_bps():
    costs = np.array([0.0010, 0.0020, np.nan, 0.0000])
    assert compute_turnover_cost(costs, min_periods=3) == pytest.approx(10.0)


def test_each_factor_has_an_independent_minimum_observation_gate():
    costs = np.array([
        [0.0010, np.nan],
        [0.0030, 0.0020],
        [np.nan, np.nan],
    ])
    actual = compute_turnover_cost(costs, min_periods=2)
    assert actual[0] == pytest.approx(20.0)
    assert np.isnan(actual[1])


@pytest.mark.parametrize("bad", [np.array([-0.001]), np.array([np.inf])])
def test_negative_or_infinite_cost_contributions_fail_closed(bad):
    with pytest.raises(ValueError):
        compute_turnover_cost(bad)


def test_cost_metric_rejects_non_panel_inputs_and_implicit_minimums():
    with pytest.raises(ValueError, match="shape"):
        compute_turnover_cost(np.zeros((2, 2, 1)))
    with pytest.raises(TypeError, match="positive integer"):
        compute_turnover_cost(np.array([0.001]), min_periods=True)
    with pytest.raises(ValueError, match="at least 1"):
        compute_turnover_cost(np.array([0.001]), min_periods=0)
