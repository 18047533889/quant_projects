import numpy as np
import pytest
from quant_evaluator.metrics.portfolio_stats import compute_sortino_ratio, compute_calmar_ratio


def test_downside_variants_independent_hand_calculation():
    returns = np.array([.1, -.1, .2, -.2])
    # Mean = 0; use periodic MAR=.01 so the numerator is nonzero.
    excess = np.array([.09, -.11, .19, -.21])
    for denominator, n in [('negative',2),('all',4)]:
        expected = -.01 / np.sqrt((.11**2+.21**2)/n)
        actual = compute_sortino_ratio(returns, min_periods=4, mar=.01,
            downside_denominator=denominator,annualization='none')
        assert actual == pytest.approx(expected)
    assert np.isnan(compute_sortino_ratio(np.ones(30)*.1))


def test_cagr_calmar_uses_independent_wealth_not_mean():
    returns = np.array([.5, -.4])
    # Initial NAV1 → 1.5 → .9; max drawdown .4.
    arithmetic = compute_calmar_ratio(returns,periods_per_year=2,min_periods=2,annualization='arithmetic')
    cagr = compute_calmar_ratio(returns,periods_per_year=2,min_periods=2,annualization='cagr')
    assert arithmetic == pytest.approx(.1/.4)
    assert cagr == pytest.approx(-.1/.4)
    assert compute_calmar_ratio(returns,periods_per_year=2,min_periods=2) == pytest.approx(cagr)


@pytest.mark.parametrize('denominator',['negative','all'])
def test_cuda_nonzero_target_and_missing_rows_match_cpu(denominator):
    cp = pytest.importorskip('cupy')
    from quant_evaluator.kernels.gpu.drawdown import compute_sortino_batch, compute_calmar_batch
    returns = np.array([.1,np.nan,-.1,.2,-.2,np.nan])
    args = dict(min_periods=4,mar=.01,downside_denominator=denominator)
    assert cp.asnumpy(compute_sortino_batch(returns,**args))[0] == pytest.approx(compute_sortino_ratio(returns,**args))
    for annualization in ['arithmetic','cagr']:
        args = dict(min_periods=4,periods_per_year=4,annualization=annualization,missing_return_policy='zero_fill')
        assert cp.asnumpy(compute_calmar_batch(returns,**args))[0] == pytest.approx(compute_calmar_ratio(returns,**args))


def test_public_variants_are_preserved_in_configuration():
    from quant_evaluator.tests.test_v3_public_artifacts import inputs
    from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
    from quant_evaluator.runtime.evaluator import evaluate
    batch,labels = inputs()
    portfolio = ProbePortfolioArtifact(np.tile([[.1,-.1],[-.2,.2]],(15,1)),
        time_index=labels.decision_time,factor_ids=batch.factor_ids)
    kwargs = dict(metrics=['sortino_ratio','calmar_ratio'],portfolio_returns=portfolio)
    a = evaluate(batch,labels,**kwargs)
    b = evaluate(batch,labels,**kwargs,metric_parameters={
        'sortino_ratio':{'downside_denominator':'all','mar':.01},
        'calmar_ratio':{'annualization':'cagr'}})
    assert a.config_hash != b.config_hash
    assert b.artifacts['calmar_ratio'].provenance['parameters']['annualization'] == 'cagr'


def test_public_cuda_prebuilt_portfolio_stays_factor_aligned_across_tiles(monkeypatch):
    pytest.importorskip('cupy')
    from quant_evaluator.tests.test_v3_public_artifacts import inputs
    from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
    from quant_evaluator.runtime.evaluator import evaluate
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    batch,labels = inputs()
    pnl = np.tile([[.1,-.03],[-.2,.07]],(15,1))
    pnl[3,0] = np.nan
    portfolio = ProbePortfolioArtifact(pnl,time_index=labels.decision_time,factor_ids=batch.factor_ids)
    kwargs = dict(metrics=['sortino_ratio','calmar_ratio'],portfolio_returns=portfolio,
        metric_parameters={'sortino_ratio':{'downside_denominator':'all','mar':.01},
                           'calmar_ratio':{'annualization':'cagr'}})
    monkeypatch.setattr(DeviceEvaluationSession,'estimate_tile',lambda *args,**kwargs:1)
    cpu = evaluate(batch,labels,**kwargs)
    gpu = evaluate(batch,labels,backend='cuda',**kwargs)
    assert cpu.config_hash == gpu.config_hash
    for mid in kwargs['metrics']:
        np.testing.assert_allclose(cpu.artifacts[mid].values,gpu.artifacts[mid].values,rtol=1e-12)
