import numpy as np
import pytest
from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown
from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_series, identify_drawdown_periods


@pytest.mark.parametrize('returns', [np.array([-.1, -.1, .1]), np.array([[-.1, .1], [-.1, -.2], [.1, .1]])])
def test_drawdown_includes_initial_capital(returns):
    wealth = np.cumprod(1 + returns, axis=0)
    peaks = np.maximum(1.0, np.maximum.accumulate(wealth, axis=0))
    expected = wealth / peaks - 1
    actual, actual_wealth, actual_peaks = compute_drawdown_series(returns)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_wealth, wealth)
    np.testing.assert_allclose(actual_peaks, peaks)
    maximum, drawdown, _ = compute_maximum_drawdown(returns)
    np.testing.assert_allclose(drawdown, expected)
    np.testing.assert_allclose(maximum, -np.min(expected, axis=0))


def test_initial_loss_is_not_reported_as_zero_drawdown():
    maximum, drawdown, peak = compute_maximum_drawdown(np.array([-.1]))
    assert maximum == pytest.approx(.1)
    np.testing.assert_allclose(drawdown, [-.1])
    assert peak == -1  # Initial capital precedes the first observed return.


def test_initial_capital_peak_is_used_for_drawdown_periods():
    periods = identify_drawdown_periods(np.array([-.01, -.1, .2]), threshold=.05)
    assert len(periods) == 1
    assert periods[0]['peak_idx'] == -1
    assert periods[0]['trough_idx'] == 1
    assert periods[0]['recovery_idx'] == 2
    assert periods[0]['drawdown'] == pytest.approx(.109)


def test_initial_drawdown_ongoing_trough_never_uses_negative_array_index():
    period = identify_drawdown_periods(np.array([-.2, .1]), threshold=.05)[0]
    assert period['peak_idx'] == -1
    assert period['trough_idx'] == 0
    assert period['recovery_idx'] == -1
    assert period['drawdown'] == pytest.approx(.2)


def test_gpu_drawdown_matches_initial_capital_oracle():
    cp = pytest.importorskip('cupy')
    from quant_evaluator.kernels.gpu.drawdown import compute_max_drawdown_batch
    returns = np.array([[-.1, .1], [-.1, -.2], [.1, .1]])
    np.testing.assert_allclose(cp.asnumpy(compute_max_drawdown_batch(returns)), [.19, .2])
