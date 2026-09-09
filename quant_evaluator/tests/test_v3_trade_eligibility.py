"""Independent side-permission cases through the public CPU/CUDA facade."""
from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec, TradeEligibilityPanel
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.contracts.errors import InvalidContractError


def inputs():
    t, n = 30, 2
    times = AxisRef('time', 'int', t, np.arange(t))
    assets = AxisRef('asset', 'str', n, np.array(['short', 'long']))
    factors = np.full((t, n, 1), np.nan)
    factors[0, :, 0] = [0., 1.]
    batch = FactorBatch(('f',), times, assets, factors)
    prices = np.full((t, n), 100.)
    prices[2:, 1] = 110.
    returns = HoldingReturnPanel.from_prices(prices, time_axis=times, asset_axis=assets,
        source_ref='fixture:prices', price_basis='close')
    labels = LabelBundle('h10', np.full((t,n), 999.), 10, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1,t+1)), label_end_time=tuple(range(11,t+11)), asset_axis=assets)
    allow = np.ones((t,n), dtype=bool)
    permissions = TradeEligibilityPanel(allow, allow, allow, allow,
        np.broadcast_to(np.arange(t)[:,None], (t,n)), times, assets, 'fixture:permission', 'fixture:universe')
    return batch, labels, dict(metrics=['max_drawdown'], holding_returns=returns,
        portfolio_spec=PortfolioSpec(holding=1,n_quantiles=2,per_side_cost=.001), trade_eligibility=permissions)


@pytest.mark.parametrize('backend', [None, 'cuda'])
def test_no_borrow_does_not_create_naked_long_probe(backend):
    if backend:
        pytest.importorskip('cupy')
    batch, labels, kwargs = inputs()
    permitted = evaluate(batch, labels, backend=backend, **kwargs)
    p = kwargs['trade_eligibility']
    kwargs['trade_eligibility'] = replace(p, borrowable=np.zeros_like(p.borrowable))
    blocked = evaluate(batch, labels, backend=backend, **kwargs)
    assert permitted.artifacts['max_drawdown'].values[0] > 0
    assert blocked.artifacts['max_drawdown'].values[0] == 0
    assert blocked.config_hash != permitted.config_hash
    provenance = blocked.artifacts['max_drawdown'].provenance
    assert provenance['portfolio_purpose'] == 'RESEARCH_PROBE'
    assert provenance['execution_certified'] is False
    assert provenance['trade_eligibility_ref'] == kwargs['trade_eligibility'].content_hash


@pytest.mark.parametrize('backend', [None, 'cuda'])
@pytest.mark.parametrize('side,asset', [('coverable',0),('can_sell',1)])
def test_blocked_exit_never_fabricates_liquidation(backend, side, asset):
    if backend:
        pytest.importorskip('cupy')
    batch, labels, kwargs = inputs()
    p = kwargs['trade_eligibility']
    blocked = np.array(getattr(p,side), copy=True)
    blocked[2,asset] = False
    kwargs['trade_eligibility'] = replace(p, **{side:blocked})
    with pytest.raises(ValueError, match='fill-ledger'):
        evaluate(batch, labels, backend=backend, **kwargs)


def test_future_known_permissions_and_swapped_axes_fail_closed():
    batch, labels, kwargs = inputs()
    p = kwargs['trade_eligibility']
    with pytest.raises(ValueError, match='not known'):
        replace(p, available_time=p.available_time+1)
    kwargs['trade_eligibility'] = replace(p, asset_axis=AxisRef('asset','str',2,np.array(['long','short'])))
    with pytest.raises(InvalidContractError, match='coordinates'):
        evaluate(batch, labels, **kwargs)


def test_real_cpu_cuda_permission_parity():
    pytest.importorskip('cupy')
    batch, labels, kwargs = inputs()
    cpu = evaluate(batch,labels,**kwargs)
    gpu = evaluate(batch,labels,backend='cuda',**kwargs)
    np.testing.assert_allclose(cpu.artifacts['max_drawdown'].values,gpu.artifacts['max_drawdown'].values)
    assert cpu.config_hash == gpu.config_hash
