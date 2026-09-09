import numpy as np
import pandas as pd
import pytest
from factor_preprocess.neutralization.regularized import ridge_neutralize,lasso_neutralize
from factor_preprocess.regime.adaptive_weights import fit_regime_weights,regime_adaptive_weights
from factor_preprocess.regime.switching import RegimeSwitchingState,regime_switching_transform
from quant_evaluator.metrics.stats.regime_detection import RegimeInference
from factor_preprocess.representation.policy import decide_representation
from factor_assets.selection import (DecisionProvider,DecisionRequest,CandidateEvidence,
    SelectionPolicySpec,MetricRule,UtilityDirection)

def test_regularized_alignment_affine_and_contract():
    v=pd.DataFrame({'date':[1]*4,'asset_id':['d','b','a','c'],'value':[11.,7.,5.,9.]},index=[40,20,10,30])
    e=pd.DataFrame({'date':[1]*4,'asset_id':['a','b','c','d'],'x':[1.,2.,3.,4.]})
    r=ridge_neutralize(v,e,alpha=0,normalize=False,min_observations=2)
    assert r.index.equals(v.index) and np.max(np.abs(r))<1e-10
    e2=e.copy(); e2['x']+=100
    assert np.allclose(r,ridge_neutralize(v,e2,alpha=0,normalize=False,min_observations=2))
    assert r.attrs['fit_diagnostics'][0]['status']=='CONVERGED'
    with pytest.raises(ValueError): ridge_neutralize(v,pd.concat([e,e.iloc[:1]]),min_observations=2)
    with pytest.raises(ValueError): lasso_neutralize(v,e,max_iter=0,min_observations=2)

def test_weights_float_identity_and_label_validation():
    idx=pd.Index([4,8]); f=pd.DataFrame({'date':[1,2],'a':np.array([1,3],dtype=np.int32)},index=idx); labels=pd.Series([0,0],index=idx)
    state=fit_regime_weights(f,labels,method='equal',min_obs_per_regime=2)
    out=regime_adaptive_weights(f,labels,state,check_staleness=False)
    assert out.a.dtype.kind=='f'
    with pytest.raises(ValueError): fit_regime_weights(f,pd.Series([0,0],index=idx[::-1]),min_obs_per_regime=2)
    with pytest.raises(ValueError): fit_regime_weights(f,pd.Series([True,True],index=idx),min_obs_per_regime=2)

def test_actual_fp_consumer_rejects_smoothed_regime_evidence():
    values=pd.DataFrame({'date':pd.to_datetime(['2026-01-03','2026-01-04']),'value':[1.,2.]})
    state=RegimeSwitchingState({0:{}},'none','value',1,pd.Timestamp('2026-01-01'),pd.Timestamp('2026-01-02'))
    smooth=RegimeInference(np.ones((2,1)),np.zeros(2,dtype=int),'SMOOTHED_POSTHOC',2,3,False)
    with pytest.raises(ValueError): regime_switching_transform(values,smooth,state)
    filtered=RegimeInference(np.ones((2,1)),np.zeros(2,dtype=int),'FILTERED_ASOF',2,3,True)
    assert np.allclose(regime_switching_transform(values,filtered,state),[1,2])

def test_fp_uses_canonical_decision_provider():
    policy=SelectionPolicySpec('p','1',(MetricRule('m',UtilityDirection.HIGHER_IS_BETTER,1,0,1),),{},noninferiority_margin=.1)
    base=CandidateEvidence('raw',('e',),'FINAL',{}, {'m':.5},1,'q')
    treated=CandidateEvidence('treated',('e2',),'FINAL',{}, {'m':.49},1,'q',paired_effect=-.01,paired_interval=(-.02,.01),paired_evidence_ref='paired')
    req=DecisionRequest('r','p',policy.content_hash,'ctx','representation','factor','raw',(base,treated),'FINAL','fam')
    provider=DecisionProvider(policy); direct=provider.decide(req); via=decide_representation(provider,req,'treated')
    assert via.content_hash==direct.content_hash and via.relationship==direct.relationship


def test_t130_t131_effective_counts_reasons_and_volatility_boundary():
    idx=pd.RangeIndex(4)
    factors=pd.DataFrame({'date':pd.date_range('2026-01-01',periods=4),
                          'empty':[np.nan]*4,'single':[1.,np.nan,np.nan,np.nan],
                          'constant':[2.]*4,'small':[0.,1e-12,2e-12,3e-12]},index=idx)
    labels=pd.Series([0]*4,index=idx)
    state=fit_regime_weights(factors,labels,method='volatility_inverse',min_obs_per_regime=0)
    assert state.effective_counts[0]==(0,1,4,4)
    assert state.factor_reasons[0]==('INSUFFICIENT_EFFECTIVE_OBSERVATIONS','INSUFFICIENT_EFFECTIVE_OBSERVATIONS',
                                     'ZERO_OR_NONFINITE_VOLATILITY','ESTIMABLE')
    np.testing.assert_array_equal(state.regime_weights[0][:3],[0.,0.,0.])
    assert state.regime_weights[0][3]==pytest.approx(1.)


@pytest.mark.parametrize('bad',[0.5,np.inf,-np.inf])
def test_t134_fractional_and_infinite_labels_reject(bad):
    f=pd.DataFrame({'date':[1,2],'a':[1.,2.]})
    with pytest.raises(ValueError,match='finite non-boolean integers'):
        fit_regime_weights(f,pd.Series([0,bad]),min_obs_per_regime=1)


def test_t135_duplicate_sample_keys_reject():
    idx=pd.Index(['same','same'])
    f=pd.DataFrame({'date':[1,2],'a':[1.,2.]},index=idx)
    with pytest.raises(ValueError,match='duplicate sample keys'):
        fit_regime_weights(f,pd.Series([0,0],index=idx),min_obs_per_regime=1)


def test_t136_exact_half_weight_golden_and_t138_dtype_invariance():
    idx=pd.RangeIndex(4); labels=pd.Series([0]*4,index=idx)
    outputs=[]
    for dtype in (np.int32,np.int64,np.float64):
        f=pd.DataFrame({'date':[1,2,3,4],
                        'a':np.array([0,3,0,3],dtype=dtype),
                        'b':np.array([0,1,0,1],dtype=dtype)},index=idx)
        state=fit_regime_weights(f,labels,method='volatility_inverse',min_obs_per_regime=4)
        future=pd.DataFrame({'date':[5],'a':np.array([2],dtype=dtype),'b':np.array([2],dtype=dtype)})
        outputs.append(regime_adaptive_weights(future,pd.Series([0]),state,check_staleness=False)[['a','b']].to_numpy())
    np.testing.assert_allclose(outputs[2][0],[.5,1.5])
    np.testing.assert_allclose(outputs[0],outputs[1]); np.testing.assert_allclose(outputs[1],outputs[2])


def test_t142_extreme_unknown_value_cannot_change_known_rank():
    train=pd.DataFrame({'date':pd.to_datetime(['2025-01-01']*4),'value':[1.,2.,3.,4.]})
    labels=pd.Series([0]*4)
    from factor_preprocess.regime.switching import fit_regime_switching
    state=fit_regime_switching(train,labels,transform_type='rank',min_obs_per_regime=2)
    base=pd.DataFrame({'date':pd.to_datetime(['2026-01-01']*3),'value':[10.,20.,30.]})
    regimes=pd.Series([0,0,9])
    low=regime_switching_transform(base.assign(value=[10.,20.,-1e99]),regimes,state,unknown_regime_policy='FAIL_NAN')
    high=regime_switching_transform(base.assign(value=[10.,20.,1e99]),regimes,state,unknown_regime_policy='FAIL_NAN')
    np.testing.assert_allclose(low.iloc[:2],high.iloc[:2],equal_nan=True)
    assert np.isnan(low.iloc[2]) and np.isnan(high.iloc[2])
