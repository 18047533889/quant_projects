"""Price-ledger oracles independent of the cohort return implementation."""
import numpy as np
import pytest
from quant_evaluator.metrics.probe_portfolio import compute_cohort_pnl
from quant_evaluator.metrics.portfolio_stats import compute_long_short_returns


def cohort(prices, holding=1, cost=0.):
    t = len(prices)
    signals = np.full((t, 2), np.nan)
    signals[0] = [0., 1.]
    returns = np.zeros_like(prices)
    returns[1:] = prices[1:] / prices[:-1] - 1
    return signals, returns, compute_cohort_pnl(signals, returns, prices,
        n_quantiles=2, holding=holding, per_side_cost=cost)


def test_entry_before_jump_is_not_owned():
    prices = np.array([[100., 100.], [100., 110.], [100., 110.]])
    _, _, result = cohort(prices)
    np.testing.assert_equal(result['pnl_net'], [0., 0., 0.])
    assert result['scheduled_exit'][0] == 2
    assert result['matured'][0]


@pytest.mark.parametrize('holding', [1, 2, 10, 20])
def test_h_counts_post_entry_intervals_and_two_cost_events(holding):
    prices = np.ones((holding + 3, 2)) * 100
    for t in range(2, len(prices)):
        prices[t, 1] = prices[t - 1, 1] * 1.01
    _, _, result = cohort(prices, holding, cost=.001)
    expected = np.zeros(len(prices))
    expected[1] -= .001 / holding
    expected[2:holding + 2] += .005 / holding
    expected[holding + 1] -= .001 / holding
    np.testing.assert_allclose(result['pnl_net'], expected, atol=1e-15)
    assert result['entry_cost'].sum() == pytest.approx(.001 / holding)
    assert result['exit_cost'].sum() == pytest.approx(.001 / holding)


def test_tail_is_valued_but_not_forced_liquidated():
    prices = np.ones((4, 2)) * 100
    _, _, result = cohort(prices, holding=5, cost=.001)
    assert not result['matured'][0]
    assert result['scheduled_exit'][0] == 6
    assert result['entry_cost'].sum() > 0
    assert result['exit_cost'].sum() == 0


def test_offsetting_cohorts_do_not_hide_unknown_held_quotes():
    cp = pytest.importorskip('cupy')
    from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
    factors = np.array([[0.,1.],[1.,0.],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]])
    returns = np.zeros((5,2))
    returns[3,0] = np.nan
    cpu = compute_cohort_pnl(factors,returns,np.ones((5,2)),holding=2,n_quantiles=2,require_tradable=False)
    gpu = compute_cohort_pnl_batch_gpu(factors,returns,holding=2,n_quantiles=2,require_tradable=False)
    assert np.isnan(cpu['pnl_net'][3])
    np.testing.assert_allclose(cp.asnumpy(gpu['pnl_net'])[:,0],cpu['pnl_net'],equal_nan=True)


def test_real_gpu_matches_independent_post_entry_ledger():
    cp = pytest.importorskip('cupy')
    from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
    prices = np.array([[100., 100.], [100., 110.], [100., 121.], [100., 121.]])
    signals, returns, cpu = cohort(prices, holding=1, cost=.001)
    gpu = compute_cohort_pnl_batch_gpu(signals, returns, n_quantiles=2, holding=1, per_side_cost=.001)
    np.testing.assert_allclose(cp.asnumpy(gpu['pnl_net'])[:, 0], [0., -.001, .049, 0.], atol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(gpu['pnl_net'])[:, 0], cpu['pnl_net'], atol=1e-15)


def test_shared_mask_and_degenerate_ties():
    factors = np.array([[[0., 1.], [1., 0.], [2., -1.], [3., -2.]]])
    returns = np.array([[.01, .02, .03, .04]])
    shared = np.ones((1, 4), dtype=bool)
    a = compute_long_short_returns(factors, returns, validity_mask=shared)
    b = compute_long_short_returns(factors, returns, validity_mask=np.broadcast_to(shared[:, :, None], factors.shape))
    for x, y in zip(a, b):
        np.testing.assert_equal(x, y)
    assert np.isnan(compute_long_short_returns(np.ones((1, 4)), returns)[2][0])


def test_future_missingness_cannot_replace_selected_top_stock():
    factors = np.array([[0., 1., 2., 3., 4.]])
    returns = np.array([[.01, .02, .03, .04, np.nan]])
    long, short, spread = compute_long_short_returns(factors, returns, missing_return_policy='drop')
    assert np.isnan(long[0])  # old implementation selected stock 3 as a substitute
    assert np.isnan(spread[0])


def test_public_cpu_gpu_portfolio_uses_price_panel_not_forward_labels():
    from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    from quant_evaluator.contracts.errors import InvalidContractError
    pytest.importorskip('cupy')
    times = AxisRef('time', 'int', 30, np.arange(30))
    assets = AxisRef('asset', 'str', 2, np.array(['A', 'B']))
    prices = np.ones((30, 2)) * 100
    for t in range(1, 30):
        prices[t, 1] = prices[t-1, 1] * (1.01 if t % 2 else .998)
    panel = HoldingReturnPanel.from_prices(prices, time_axis=times, asset_axis=assets,
                                           source_ref='fixture:quotes:v1', price_basis='vwap')
    batch = FactorBatch(('up', 'down'), times, assets,
                        np.tile(np.array([[0., 1.], [1., 0.]])[None], (30, 1, 1)))
    labels = LabelBundle('forward_h10', np.full((30, 2), 999.), 10,
                         decision_time=tuple(range(30)), label_start_time=tuple(range(1, 31)),
                         label_end_time=tuple(range(11, 41)), asset_axis=assets)
    spec = PortfolioSpec(holding=1, n_quantiles=2, per_side_cost=.001)
    kwargs = dict(metrics=['sharpe_ratio', 'max_drawdown'], holding_returns=panel, portfolio_spec=spec)
    cpu = evaluate(batch, labels, **kwargs)
    gpu = evaluate(batch, labels, backend='cuda', **kwargs)
    assert cpu.config_hash == gpu.config_hash
    for mid in kwargs['metrics']:
        np.testing.assert_allclose(cpu.artifacts[mid].values, gpu.artifacts[mid].values, atol=1e-10)
    assert cpu.get_metric('sharpe_ratio', 'up').value > cpu.get_metric('sharpe_ratio', 'down').value
    with pytest.raises(InvalidContractError, match='HoldingReturnPanel'):
        evaluate(batch, labels, metrics=['sharpe_ratio'], holding_returns=labels, portfolio_spec=spec)
