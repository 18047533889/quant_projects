import numpy as np
import pytest
from quant_evaluator.metrics.portfolio_stats import (
    equal_gross_weights, equal_gross_long_short_returns,
    compute_wealth_curve, compute_aligned_wealth_curve, compute_maximum_drawdown,
    compute_long_short_returns, apply_long_short_costs,
)


def test_all_selected_names_have_equal_absolute_weight_and_full_short_margin():
    longs = np.array([[1, 1, 0], [1, 0, 0]], bool)
    shorts = np.array([[0, 0, 1], [0, 1, 1]], bool)
    weights = equal_gross_weights(longs, shorts)
    np.testing.assert_allclose(np.abs(weights), 1/3)
    np.testing.assert_allclose(np.abs(weights).sum(axis=1), 1)
    ret = np.array([[.06, .03, -.03], [.03, -.03, -.06]])
    pnl = equal_gross_long_short_returns(longs, shorts, ret, cost_rate=.001)
    np.testing.assert_allclose(pnl, [.04-.001, .04-.001*2/3])


def test_missing_selected_returns_do_not_change_positions():
    l = np.array([[1, 0, 0]], bool); s = np.array([[0, 1, 0]], bool)
    assert np.isnan(equal_gross_long_short_returns(l, s, [[.1, np.nan, 0.]])[0])
    assert equal_gross_long_short_returns(l, s, [[.1, -.1, np.nan]])[0] == pytest.approx(.1)
    with pytest.raises(ValueError): equal_gross_weights(l, l)


def test_public_return_api_and_aggregate_costs_have_one_hundred_percent_gross():
    _, _, ls = compute_long_short_returns(np.array([[1.,2.,3.,4.]]), np.array([[-.1,0.,0.,.1]]))
    assert ls[0] == pytest.approx(.1)
    assert apply_long_short_costs(np.array([.1]), np.array([-.1]), long_turnover=np.ones(1),
        short_turnover=np.ones(1), cost_rate=.001)[0] == pytest.approx(.099)


def test_equity_exhaustion_is_absorbing_and_reports_full_loss():
    r = np.array([.1,-1.2,-2.,.1])
    np.testing.assert_allclose(compute_wealth_curve(r), [1.1,0.,0.,0.])
    np.testing.assert_allclose(compute_aligned_wealth_curve(r), [1.1,0.,0.,0.])
    assert compute_maximum_drawdown(r)[0] == 1.
    assert compute_maximum_drawdown(np.array([-.1,0.,.05]))[0] == pytest.approx(.1)
