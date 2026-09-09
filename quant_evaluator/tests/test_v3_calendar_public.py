from datetime import timedelta
import json
import numpy as np
import pytest
from quant_evaluator.tests.test_calendar_returns import snapshot, instants
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.api.requests import EvaluationBundle
from quant_evaluator.runtime.evaluator import evaluate


def test_public_calendar_metric_keeps_period_counts_and_partial_evidence():
    calendar = snapshot(('2025-01-31','2025-02-03','2025-02-04','2025-03-03'))
    times = instants(calendar.trading_days[1:-1])
    batch = FactorBatch(('f',),AxisRef('time','datetime',2,np.array(times)),
        AxisRef('asset','str',2,np.array(['A','B'])),np.ones((2,2,1)))
    labels = LabelBundle('h1',np.ones((2,2)),1,decision_time=times,
        label_start_time=tuple(t+timedelta(days=1) for t in times),
        label_end_time=tuple(t+timedelta(days=2) for t in times))
    portfolio = ProbePortfolioArtifact(np.array([[-.1],[-.1]]),time_index=times,factor_ids=('f',))
    bundle = evaluate(batch,labels,metrics=['worst_calendar_month'],portfolio_returns=portfolio,calendar_snapshot=calendar)
    metric = bundle.get_metric('worst_calendar_month','f')
    assert metric.value == pytest.approx(-.19)
    assert metric.observation_count == 1
    assert metric.sample_unit == 'included_calendar_period'
    provenance = bundle.artifacts['worst_calendar_month'].provenance
    assert provenance['calendar_snapshot_id'] == calendar.snapshot_id
    assert provenance['partial_policy'] == 'exclude'
    assert provenance['execution_certified'] is False
    restored = EvaluationBundle.from_dict(json.loads(json.dumps(bundle.to_dict(),allow_nan=False)))
    assert restored.get_metric('worst_calendar_month','f').value == metric.value
    assert restored.artifacts['worst_calendar_month'].provenance['period_rows'] == provenance['period_rows']


def test_public_rolling_id_does_not_accept_immature_prefix():
    from quant_evaluator.tests.test_v3_public_artifacts import inputs
    batch,labels = inputs()
    returns = np.zeros((30,2))
    returns[0] = -.1
    portfolio = ProbePortfolioArtifact(returns,time_index=labels.decision_time,factor_ids=batch.factor_ids)
    bundle = evaluate(batch,labels,metrics=['worst_rolling_21d','worst_rolling_63d'],portfolio_returns=portfolio)
    np.testing.assert_allclose(bundle.artifacts['worst_rolling_21d'].values,[-.1,-.1])
    assert np.isnan(bundle.artifacts['worst_rolling_63d'].values).all()
    assert bundle.get_metric('worst_rolling_21d','up').observation_count == 10
    assert bundle.get_metric('worst_rolling_63d','up').observation_count == 0


def test_l20_cuda_calendar_and_rolling_strict_parity_and_telemetry():
    pytest.importorskip('cupy')
    calendar = snapshot(('2024-12-31','2025-01-31','2025-02-03','2025-02-04','2025-03-03','2025-04-01'))
    times = instants(calendar.trading_days[1:-1])
    batch = FactorBatch(('a','b'),AxisRef('time','datetime',4,np.array(times)),
        AxisRef('asset','str',2,np.array(['A','B'])),np.ones((4,2,2)))
    labels = LabelBundle('h1',np.ones((4,2)),1,decision_time=times,
        label_start_time=tuple(t+timedelta(days=1) for t in times),
        label_end_time=tuple(t+timedelta(days=2) for t in times))
    returns = np.array([[-.2,.1],[-.1,np.nan],[-.1,.2],[-.05,-.1]])
    portfolio = ProbePortfolioArtifact(returns,time_index=times,factor_ids=('a','b'))
    metrics = ['worst_calendar_month','worst_calendar_quarter','worst_calendar_year']
    params = {mid:{'partial_policy':'include'} for mid in metrics}
    cpu = evaluate(batch,labels,metrics=metrics,metric_parameters=params,
        portfolio_returns=portfolio,calendar_snapshot=calendar)
    gpu = evaluate(batch,labels,metrics=metrics,metric_parameters=params,
        portfolio_returns=portfolio,calendar_snapshot=calendar,backend='cuda_strict')
    for mid in metrics:
        np.testing.assert_allclose(gpu.artifacts[mid].values,cpu.artifacts[mid].values,
            rtol=0,atol=1e-12,equal_nan=True)
        assert tuple(gpu.artifacts[mid].provenance['observation_counts']) == \
            tuple(cpu.artifacts[mid].provenance['observation_counts'])
        assert gpu.artifacts[mid].provenance['execution_backend'] == 'cuda_strict'
        assert gpu.artifacts[mid].provenance['planning_backend'] == 'cpu'
        assert gpu.artifacts[mid].provenance['no_fallback'] is True
    assert gpu.metadata['calendar_rolling_reduction_backend'] == 'cuda_strict'
    assert gpu.metadata['calendar_rolling_cuda_no_fallback'] is True
    assert gpu.metadata['calendar_rolling_factor_tiles'] >= 3
    exclude_cpu = evaluate(batch,labels,metrics=['worst_calendar_month'],
        portfolio_returns=portfolio,calendar_snapshot=calendar)
    exclude_gpu = evaluate(batch,labels,metrics=['worst_calendar_month'],
        portfolio_returns=portfolio,calendar_snapshot=calendar,backend='cuda_strict')
    np.testing.assert_allclose(exclude_gpu.artifacts['worst_calendar_month'].values,
        exclude_cpu.artifacts['worst_calendar_month'].values,rtol=0,atol=1e-12,equal_nan=True)

    from quant_evaluator.tests.test_v3_public_artifacts import inputs
    rolling_batch,rolling_labels = inputs()
    rolling_returns = np.zeros((30,2)); rolling_returns[0] = -.1; rolling_returns[10,1] = np.nan
    rolling_portfolio = ProbePortfolioArtifact(rolling_returns,
        time_index=rolling_labels.decision_time,factor_ids=rolling_batch.factor_ids)
    rolling_metrics = ['worst_rolling_21d','worst_rolling_63d']
    rolling_cpu = evaluate(rolling_batch,rolling_labels,metrics=rolling_metrics,
        portfolio_returns=rolling_portfolio)
    rolling_gpu = evaluate(rolling_batch,rolling_labels,metrics=rolling_metrics,
        portfolio_returns=rolling_portfolio,backend='cuda_strict')
    for mid in rolling_metrics:
        np.testing.assert_allclose(rolling_gpu.artifacts[mid].values,
            rolling_cpu.artifacts[mid].values,rtol=0,atol=1e-12,equal_nan=True)
        assert rolling_gpu.artifacts[mid].provenance['observation_counts'] == \
            rolling_cpu.artifacts[mid].provenance['observation_counts']


def test_l20_cuda_rolling_builds_direct_holding_probe_without_adapter_gap():
    pytest.importorskip('cupy')
    from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel,PortfolioSpec
    times = AxisRef('time','int',30,np.arange(30)); assets = AxisRef('asset','str',4,np.array(list('ABCD')))
    prices = np.full((30,4),100.0)
    prices[:,0] *= np.cumprod(np.r_[1.,np.full(29,1.01)])
    prices[:,3] *= np.cumprod(np.r_[1.,np.full(29,.995)])
    holding = HoldingReturnPanel.from_prices(prices,time_axis=times,asset_axis=assets,
        source_ref='fixture:vwap',price_basis='vwap')
    values = np.tile(np.arange(4.)[None,:,None],(30,1,2)); values[:,:,1] *= -1
    batch = FactorBatch(('up','down'),times,assets,values)
    labels = LabelBundle('forward',np.ones((30,4)),1,decision_time=tuple(range(30)),
        label_start_time=tuple(range(1,31)),label_end_time=tuple(range(2,32)),asset_axis=assets)
    spec = PortfolioSpec(holding=1,n_quantiles=2)
    kwargs = dict(metrics=['worst_rolling_21d'],holding_returns=holding,portfolio_spec=spec)
    cpu = evaluate(batch,labels,**kwargs); gpu = evaluate(batch,labels,backend='cuda_strict',**kwargs)
    np.testing.assert_allclose(gpu.artifacts['worst_rolling_21d'].values,
        cpu.artifacts['worst_rolling_21d'].values,rtol=0,atol=1e-12,equal_nan=True)
    assert gpu.metadata['calendar_rolling_trajectory_source'] == 'cuda_holding_return_probe'
    assert gpu.metadata['probe_trajectory_factor_tiles'] > 0
    assert gpu.artifacts['worst_rolling_21d'].provenance['trajectory_source'] == 'cuda_holding_return_probe'
