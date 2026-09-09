from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator import evaluate
from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact

@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_four_factor_public_batch_equals_individual_and_metric_reordering(backend):
    if backend=='cuda':
        import cupy as cp
        assert cp.cuda.runtime.getDeviceCount()>0
    rng=np.random.default_rng(7128); t,n=40,60
    x=rng.normal(size=(t,n)); y=x*.2+rng.normal(size=(t,n))
    values=np.stack((x,-x,rng.normal(size=(t,n)),np.full((t,n),np.nan)),axis=-1)
    values[::5,:8,2]=np.nan
    ids=('positive','negative','independent','empty'); dates=tuple(range(t))
    batch=FactorBatch(ids,AxisRef('t','int',t),AxisRef('n','int',n),values)
    labels=LabelBundle('h1',y,1,decision_time=dates,label_start_time=dates,
                       label_end_time=tuple(range(1,t+1)))
    daily=np.column_stack((np.tile([.01,-.002],20),np.tile([-.01,.002],20),
                           rng.normal(0,.01,t),np.full(t,np.nan)))
    portfolio=ProbePortfolioArtifact(daily,time_index=dates,factor_ids=ids)
    metrics=['rank_ic_series','quantile_returns_full','sharpe_ratio']
    batch_result=evaluate(batch,labels,metrics=metrics,portfolio_returns=portfolio,backend=backend)
    reversed_result=evaluate(batch,labels,metrics=list(reversed(metrics)),portfolio_returns=portfolio,backend=backend)
    for metric in metrics:
        np.testing.assert_allclose(batch_result.artifacts[metric].values,
            reversed_result.artifacts[metric].values,equal_nan=True,rtol=1e-11,atol=1e-12)
    for i,fid in enumerate(ids):
        single=replace(batch,factor_ids=(fid,),values=values[:,:,i:i+1])
        single_portfolio=ProbePortfolioArtifact(daily[:,i:i+1],time_index=dates,factor_ids=(fid,))
        result=evaluate(single,labels,metrics=metrics,portfolio_returns=single_portfolio,backend=backend)
        for metric in metrics:
            np.testing.assert_allclose(batch_result.artifacts[metric].values[...,i],
                result.artifacts[metric].values[...,0],equal_nan=True,rtol=1e-11,atol=1e-12)
            assert batch_result.metric_versions[metric]==result.metric_versions[metric]
