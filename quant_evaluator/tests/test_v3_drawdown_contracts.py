"""Independent arithmetic cases for V3 QE-11/12/13 risk boundaries."""
import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import (
    compute_maximum_drawdown, compute_sharpe_ratio,
    compute_wealth_curve, compute_aligned_wealth_curve, apply_long_short_costs,
)
from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_series
from quant_evaluator.metrics.risk.drawdown_analysis import drawdown_events
from quant_evaluator.metrics.underwater import (
    compute_time_to_recovery, compute_max_underwater_duration, compute_return_skew,
    compute_rolling_sharpe_tail,
)


@pytest.mark.parametrize("returns, expected", [
    ([-.1], [.1]), ([.1, -1., .5], [0., 1., 1.]),
    ([-1., .2], [1., 1.]),
])
def test_default_is_observed_loss_not_missing(returns, expected):
    values = np.array(returns)
    maximum, series, peak = compute_maximum_drawdown(values)
    np.testing.assert_allclose(-series, expected)
    assert maximum == pytest.approx(max(expected))
    np.testing.assert_allclose(compute_drawdown_series(values)[0], series)
    assert peak == (0 if returns[0] > 0 else -1)


@pytest.mark.parametrize("kernel", [compute_maximum_drawdown, compute_drawdown_series,
                                  compute_wealth_curve, compute_aligned_wealth_curve])
def test_negative_capital_requires_separate_contract(kernel):
    with pytest.raises(ValueError, match="negative-capital"):
        kernel(np.array([-1.2, -2.]))


def test_zero_nav_remains_zero():
    for kernel in (compute_wealth_curve, compute_aligned_wealth_curve):
        np.testing.assert_allclose(kernel(np.array([.1, -1., .5])), [1.1, 0., 0.])


@pytest.mark.parametrize("shape", [(0,), (0, 3), (0, 0), (2, 0)])
def test_empty_return_shapes_do_not_change_metric_type(shape):
    values = np.empty(shape)
    sharpe = compute_sharpe_ratio(values)
    assert np.shape(sharpe) == shape[1:]
    maximum, series, peak = compute_maximum_drawdown(values)
    assert np.shape(maximum) == shape[1:]
    assert series.shape == shape
    assert np.shape(peak) == shape[1:]


@pytest.mark.parametrize("turnover", [-1., np.nan, np.inf])
def test_bad_turnover_cannot_create_cost_rebate(turnover):
    with pytest.raises(ValueError, match="turnover"):
        apply_long_short_costs(np.array([.1]), np.array([.01]),
                              long_turnover=np.array([turnover]),
                              short_turnover=np.array([0.]), cost_rate=.001)


def test_two_events_are_split_before_selecting_trough():
    returns = np.array([-.1, 1 / .9 - 1, -.5, 0., 1.])
    events = drawdown_events(returns)
    assert [(e['trough_idx'], e['recovery_idx']) for e in events] == [(0, 1), (2, 4)]
    assert compute_time_to_recovery(returns, min_periods=1) == 1.5


def test_censored_age_is_not_a_recovery():
    returns = np.array([-.1, 0., 0.])
    assert np.isnan(compute_time_to_recovery(returns, min_periods=1))
    assert drawdown_events(returns)[0]['censored'] is True
    assert compute_max_underwater_duration(returns, min_periods=1) == 3


def test_missing_valuation_cannot_certify_recovery():
    returns = np.array([-.1, np.nan, .5])
    event = drawdown_events(returns)[0]
    assert event['status'] == 'INVALID_VALUATION'
    assert event['end_idx'] == 2
    assert event['duration'] == 3
    assert event['recovery_idx'] == -1
    assert np.isnan(compute_time_to_recovery(returns, min_periods=1))
    assert np.isnan(compute_max_underwater_duration(returns, min_periods=1))


def test_rolling_windows_do_not_join_across_deleted_dates():
    returns = np.array([.1, .2, np.nan, np.nan, .3, .4])
    assert np.isnan(compute_rolling_sharpe_tail(returns, window=3, min_periods=3))


@pytest.mark.parametrize('bias', [True, False])
def test_skew_declared_variant_matches_independent_reference(bias):
    from scipy.stats import skew
    returns = np.array([-.2, .1, .15, .8, np.nan])
    assert compute_return_skew(returns, min_periods=3, bias=bias) == pytest.approx(
        skew(returns, bias=bias, nan_policy='omit'))


@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_unknown_valuation_is_not_zero_risk_but_bankruptcy_is_known(backend):
    from quant_evaluator.metrics.portfolio_stats import compute_calmar_ratio
    if backend == 'cuda':
        cp = pytest.importorskip('cupy')
        from quant_evaluator.kernels.gpu.drawdown import compute_max_drawdown_batch,compute_calmar_batch
        drawdown = lambda r, **kw: cp.asnumpy(compute_max_drawdown_batch(r,**kw))[0]
        calmar = lambda r, **kw: cp.asnumpy(compute_calmar_batch(r,**kw))[0]
    else:
        drawdown = lambda r, **kw: compute_maximum_drawdown(r,**kw)[0]
        calmar = compute_calmar_ratio
    returns = np.array([-.1,np.nan,.5])
    assert np.isnan(drawdown(returns))
    assert np.isnan(calmar(returns,min_periods=1))
    assert drawdown(returns,missing_return_policy='zero_fill') == pytest.approx(.1)
    assert drawdown(np.array([np.nan,-1.,.5])) == 1.
    assert np.isnan(drawdown(np.array([np.nan,np.nan])))
    with pytest.raises(ValueError,match='non-finite|nonfinite'):
        drawdown(returns,missing_return_policy='fail')
