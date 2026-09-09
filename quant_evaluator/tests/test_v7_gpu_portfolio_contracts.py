import numpy as np
import cupy as cp
import pytest

from quant_evaluator.kernels.gpu.drawdown import compute_sharpe_batch, compute_sortino_batch
from quant_evaluator.kernels.gpu.portfolio import compute_cohort_pnl_batch_gpu
from quant_evaluator.metrics.probe_portfolio._core import compute_cohort_pnl


def _sharpe_oracle(x, annual_rf, frequency):
    excess = np.asarray(x)[np.isfinite(x)] - annual_rf / frequency
    return excess.mean() / excess.std(ddof=1) * np.sqrt(frequency)


def _sortino_oracle(x, annual_rf, frequency):
    excess = np.asarray(x)[np.isfinite(x)] - annual_rf / frequency
    downside = excess[excess < 0]
    return excess.mean() / np.sqrt(np.mean(downside ** 2)) * np.sqrt(frequency)


def test_nonzero_rf_ignores_nonfinite_padding_on_actual_cuda():
    base = np.array([.01, -.02, .03, -.01, .02])
    padded = np.r_[base, np.nan, np.inf, np.nan][:, None]
    kwargs = dict(risk_free_rate=.08, periods_per_year=252, min_periods=2)
    sharpe = cp.asnumpy(compute_sharpe_batch(cp.asarray(padded), **kwargs))[0]
    sortino = cp.asnumpy(compute_sortino_batch(cp.asarray(padded), **kwargs))[0]
    assert np.isclose(sharpe, _sharpe_oracle(base, .08, 252))
    assert np.isclose(sortino, _sortino_oracle(base, .08, 252))


def test_h_greater_than_t_explicit_liquidation_cpu_gpu_oracle():
    T, H, N = 5, 7, 4
    factors = np.tile(np.arange(N, dtype=float), (T, 1))
    returns = np.zeros((T, N))
    vwap = np.ones((T, N))
    kwargs = dict(n_quantiles=2, holding=H, long_weight=1., short_weight=0.,
                  per_side_cost=.001, min_bucket_size=1,
                  terminal_position_policy="liquidate_at_end")
    cpu = compute_cohort_pnl(factors, returns, vwap, require_tradable=True, **kwargs)
    gpu = compute_cohort_pnl_batch_gpu(cp.asarray(factors), cp.asarray(returns),
                                       require_tradable=False, **kwargs)
    gpu_exit = cp.asnumpy(gpu["exit_cost"])[:, 0]
    assert np.isclose(gpu_exit[-1], 3 * .001 / H)
    np.testing.assert_allclose(gpu_exit, cpu["exit_cost"])
    np.testing.assert_allclose(cp.asnumpy(gpu["pnl_net"])[:, 0], cpu["pnl_net"])


def test_ongoing_remains_default_and_terminal_policy_is_validated():
    factors = cp.asarray(np.tile(np.arange(4, dtype=float), (5, 1)))
    returns = cp.zeros((5, 4))
    default = compute_cohort_pnl_batch_gpu(factors, returns, n_quantiles=2,
                                            holding=7, long_weight=1., short_weight=0.,
                                            per_side_cost=.001, require_tradable=False)
    assert cp.asnumpy(default["exit_cost"]).sum() == 0

@pytest.mark.parametrize('T,H',[(1,1),(2,1),(4,1),(4,2),(5,5),(6,5),(8,3),(5,7)])
@pytest.mark.parametrize('long_weight,short_weight',[(1.,0.),(0.,-1.),(.5,-.5)])
def test_terminal_liquidation_every_schedule_boundary(T,H,long_weight,short_weight):
    factors=np.tile(np.arange(8,dtype=float),(T,1)); returns=np.zeros((T,8))
    kwargs=dict(n_quantiles=2,holding=H,long_weight=long_weight,short_weight=short_weight,
                per_side_cost=.001,min_bucket_size=1,terminal_position_policy='liquidate_at_end')
    gpu=compute_cohort_pnl_batch_gpu(cp.asarray(factors),cp.asarray(returns),require_tradable=False,**kwargs)
    expected=np.zeros(T)
    for signal in range(T):
        entry=signal+1
        if entry>=T: continue
        exit_day=entry+H
        if exit_day<T:
            expected[exit_day]+=.001/H
        elif entry<T-1:
            expected[T-1]+=.001/H
    np.testing.assert_allclose(cp.asnumpy(gpu['exit_cost'])[:,0],expected,atol=1e-14)

def test_terminal_liquidation_rejects_unavailable_exit_permission():
    factors=cp.asarray(np.tile(np.arange(8,dtype=float),(5,1)))
    permissions=cp.ones((4,5,8),dtype=cp.bool_)
    permissions[1,-1,:]=False
    with pytest.raises(ValueError,match='exit|liquidation'):
        compute_cohort_pnl_batch_gpu(factors,cp.zeros((5,8)),n_quantiles=2,holding=7,
            long_weight=1.,short_weight=0.,require_tradable=False,
            terminal_position_policy='liquidate_at_end',trade_eligibility=permissions)

@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_public_terminal_policy_reaches_actual_pnl_and_independent_sharpe(backend):
    from quant_evaluator import evaluate
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel,PortfolioSpec
    T,N,H=6,8,5
    times=AxisRef('time','int',T,np.arange(T)); assets=AxisRef('asset','int',N,np.arange(N))
    factors=FactorBatch(('f',),times,assets,np.tile(np.arange(N,dtype=float),(T,1))[:,:,None])
    label=LabelBundle('h1',np.zeros((T,N)),1,decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),label_end_time=tuple(range(1,T+1)),asset_axis=assets)
    holding=HoldingReturnPanel(np.zeros((T,N)),times,assets,'synthetic:zero-price-change','close_to_close')
    spec=PortfolioSpec(holding=H,n_quantiles=2,long_weight=1.,short_weight=0.,per_side_cost=.001,
                       terminal_position_policy='liquidate_at_end')
    result=evaluate(factors,label,metrics=['sharpe_ratio'],holding_returns=holding,portfolio_spec=spec,
                    metric_parameters={'sharpe_ratio':{'min_periods':2}},backend=backend)
    expected=np.zeros(T)
    for signal in range(T-1):
        entry=signal+1; expected[entry]-=.001/H
        exit_day=entry+H
        if exit_day<T: expected[exit_day]-=.001/H
        elif entry<T-1: expected[-1]-=.001/H
    actual=result.artifacts['sharpe_ratio'].values[0]
    assert actual==pytest.approx(_sharpe_oracle(expected,0.,252),rel=1e-10)

@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_public_nonzero_rf_heterogeneous_masks_empty_factor(backend):
    from quant_evaluator import evaluate
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
    T=40; ids=('a','b','empty'); dates=tuple(range(T))
    batch=FactorBatch(ids,AxisRef('t','int',T),AxisRef('n','int',20),np.ones((T,20,3)))
    label=LabelBundle('h1',np.ones((T,20)),1,decision_time=dates,label_start_time=dates,
                      label_end_time=tuple(range(1,T+1)))
    pnl=np.tile(np.array([.004,-.003,.002,-.001]),10)[:,None]*np.array([1.,2.,1.])[None,:]
    pnl[20:,0]=np.nan; pnl[::3,1]=np.nan; pnl[:,2]=np.nan
    artifact=ProbePortfolioArtifact(pnl,time_index=dates,factor_ids=ids)
    params={m:{'risk_free_rate':.05,'min_periods':2} for m in ['sharpe_ratio','sortino_ratio']}
    result=evaluate(batch,label,metrics=list(params),portfolio_returns=artifact,metric_parameters=params,backend=backend)
    for metric,oracle in [('sharpe_ratio',_sharpe_oracle),('sortino_ratio',_sortino_oracle)]:
        for f in range(2):
            assert result.artifacts[metric].values[f]==pytest.approx(oracle(pnl[:,f],.05,252),rel=1e-10)
        assert np.isnan(result.artifacts[metric].values[2])
