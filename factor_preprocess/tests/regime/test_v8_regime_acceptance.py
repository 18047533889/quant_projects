import numpy as np
import pandas as pd
import pytest
import json
from factor_preprocess.regime.adaptive_weights import fit_regime_weights,serialize_regime_weights,deserialize_regime_weights,regime_adaptive_weights
from factor_preprocess.regime.switching import fit_regime_switching,regime_switching_transform
from factor_preprocess.regime.causal_detector import CausalRegimeDetector,_strength,_rolling

def _series(n=120,seed=4):
    rng=np.random.default_rng(seed); x=rng.normal(size=n); rho=np.linspace(-.7,.8,n); y=rho*x+rng.normal(size=n)*np.sqrt(1-rho*rho)
    return pd.DataFrame({'date':pd.date_range('2025-01-01',periods=n),'x':x,'y':y})

def test_t132_supervised_fallback_is_explicit_research_evidence():
    n=40; f=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=n),'x':np.arange(n,dtype=float),'y':np.arange(n,dtype=float)+1}); labels=pd.Series(np.zeros(n),index=f.index); target=pd.Series(-np.arange(n,dtype=float),index=f.index)
    state=fit_regime_weights(f,labels,target=target,method='target_corr_over_vol',min_obs_per_regime=10)
    assert state.admission=='RESEARCH_ONLY' and state.fallback_reasons[0]=='NO_POSITIVE_SUPERVISED_EVIDENCE_EQUAL_WEIGHT_FALLBACK'
    restored=deserialize_regime_weights(serialize_regime_weights(state))
    assert restored.admission=='RESEARCH_ONLY' and restored.fallback_reasons==state.fallback_reasons

def test_t139_141_serialization_replay_isolated_and_old_payload_rejected():
    n=40; f=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=n),'x':np.arange(n,dtype=float),'y':np.arange(n,dtype=float)+1}); labels=pd.Series(np.zeros(n),index=f.index)
    state=fit_regime_weights(f,labels,min_obs_per_regime=10); payload=serialize_regime_weights(state); restored=deserialize_regime_weights(payload)
    future=pd.DataFrame({'date':pd.date_range('2026-01-01',periods=2),'x':[2,4],'y':[6,8]}); unknown=pd.Series([9,9])
    before=regime_adaptive_weights(future,unknown,restored,unknown_regime_policy='FALLBACK_GLOBAL'); payload['regime_weights'][0][0]=999; payload['global_unqualified'][0]=999
    pd.testing.assert_frame_equal(before,regime_adaptive_weights(future,unknown,restored,unknown_regime_policy='FALLBACK_GLOBAL'))
    old=serialize_regime_weights(state); old['version']=1
    with pytest.raises(ValueError): deserialize_regime_weights(old)
    bad=serialize_regime_weights(state); bad.pop('admission')
    with pytest.raises(ValueError): deserialize_regime_weights(bad)

@pytest.mark.parametrize('kind',['rank','zscore','none'])
def test_t143_144_unknown_policy_and_minimum_cross_section(kind):
    train=pd.DataFrame({'date':pd.to_datetime(['2025-01-01']*4),'value':[1.,2.,3.,4.]}); labels=pd.Series([0]*4)
    state=fit_regime_switching(train,labels,transform_type=kind,min_obs_per_regime=2); future=pd.DataFrame({'date':pd.to_datetime(['2026-01-01']*2),'value':[10.,1e9]}); unknown=pd.Series([0,9])
    out=regime_switching_transform(future,unknown,state,unknown_regime_policy='FAIL_NAN'); assert np.isnan(out.iloc[1])
    if kind=='rank': assert np.isnan(out.iloc[0]) and set(out.attrs['eligibility_status_by_time'].values())=={'INSUFFICIENT_ELIGIBLE_SAMPLE'}
    with pytest.raises(ValueError): regime_switching_transform(future,unknown,state,unknown_regime_policy='FAIL')

def test_t145_147_chunk_bar_checkpoint_and_prior_fit_rejection():
    all_data=_series(); train=all_data.iloc[:80]; future=all_data.iloc[80:]; checkpoint=CausalRegimeDetector(window=12,min_periods=8).fit(train).checkpoint()
    one=CausalRegimeDetector.from_checkpoint(checkpoint).detect(future); uneven=CausalRegimeDetector.from_checkpoint(checkpoint); pieces=[]
    for chunk in (future.iloc[:3],future.iloc[3:17],future.iloc[17:]): pieces.append(uneven.detect(chunk).regime)
    bars=CausalRegimeDetector.from_checkpoint(checkpoint); bar=pd.concat([bars.detect(future.iloc[i:i+1]).regime for i in range(len(future))])
    np.testing.assert_equal(pd.concat(pieces).to_numpy(),one.regime.to_numpy()); np.testing.assert_equal(bar.to_numpy(),one.regime.to_numpy())
    with pytest.raises(ValueError): CausalRegimeDetector.from_checkpoint(checkpoint).detect(train.tail(5))
    assert len(CausalRegimeDetector.from_checkpoint(checkpoint).detect(train.tail(20),allow_historical_replay=True).regime)==20


def test_t147_rejected_late_revision_preserves_decisions_and_restart_snapshot():
    data=_series(); train=data.iloc[:80]; first=data.iloc[80:90]; late=first.iloc[:3].copy()
    detector=CausalRegimeDetector(window=12,min_periods=8).fit(train)
    decisions=detector.detect(first)
    frozen=(decisions.regime.copy(deep=True),decisions.regime_strength.copy(deep=True),
            decisions.transition_flag.copy(deep=True))
    checkpoint=detector.checkpoint()
    checkpoint_bytes=json.dumps(checkpoint,sort_keys=True,default=str,separators=(',',':')).encode()
    late.loc[late.index[0],'x'] += 999.0
    with pytest.raises(ValueError,match='strictly after checkpoint history'):
        detector.detect(late)
    assert json.dumps(detector.checkpoint(),sort_keys=True,default=str,separators=(',',':')).encode()==checkpoint_bytes
    pd.testing.assert_series_equal(decisions.regime,frozen[0])
    pd.testing.assert_series_equal(decisions.regime_strength,frozen[1])
    pd.testing.assert_series_equal(decisions.transition_flag,frozen[2])
    restarted=CausalRegimeDetector.from_checkpoint(checkpoint)
    assert json.dumps(restarted.checkpoint(),sort_keys=True,default=str,separators=(',',':')).encode()==checkpoint_bytes

def test_t146_150_missing_strength_transition_and_degenerate_boundaries():
    data=_series(); det=CausalRegimeDetector(window=10,min_periods=5).fit(data.iloc[:80]); future=data.iloc[80:].copy(); future.loc[future.index[3:6],'x']=np.nan
    state=det.detect(future); assert np.isfinite(state.regime.iloc[3:6]).any(); assert not state.transition_flag[state.regime.isna()].any()
    sample=np.array([[1.,1.],[2.,2.],[np.nan,9.],[3.,4.],[4.,8.],[5.,16.]])
    rolling=_rolling(sample,5,3); expected=np.corrcoef(sample[[0,1,3],0],sample[[0,1,3],1])[0,1]; assert rolling[4]==pytest.approx(expected)
    b=np.array([-.2,.2]); vals=[_strength(v,int(np.digitize(v,b)),b) for v in (-.9,-.5,-.1,0.,.1,.5,.9)]
    assert all(np.isfinite(vals)) and all(0<=x<=1 for x in vals) and vals[0]>vals[1] and vals[-1]>vals[-2] and vals[3]>vals[2]
    constant=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=30),'x':np.arange(30.),'y':np.arange(30.)})
    with pytest.raises(ValueError): CausalRegimeDetector(window=5,min_periods=5,n_regimes=3).fit(constant)
    first=np.flatnonzero(state.regime.notna().to_numpy())[0]; assert not state.transition_flag.iloc[first]
