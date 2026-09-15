"""Cross-sectional estimators: independent math, true backend semantics and finite scales."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def fixtures():
    idx=pd.date_range("2025-01-01",periods=3)
    x=pd.DataFrame(np.tile(np.arange(1.,13.),(3,1)),index=idx,columns=[f"S{i}" for i in range(12)])
    return x

def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if isinstance(x,pd.DataFrame) else x

def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

def run(name,backend,kw):
    return result(OperatorRegistry.get(name,backend,mode="research").calculate(
        **{k:convert(v) if backend=="polars" else v for k,v in kw.items()}))

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",["cs_isolation_forest_score","cs_factor_bucket_return","cs_empirical_bayes_shrinkage","cs_shrink_to_group_mean"])
def test_cs_estimators_contract_and_oracle(name,backend):
    x=fixtures()
    group=x*0
    group.iloc[:,6:]=1.
    if name=="cs_isolation_forest_score":
        from sklearn.ensemble import IsolationForest
        kw=dict(x=x,n_trees=12,contamination=.2,random_seed=7)
        iso=IsolationForest(n_estimators=12,contamination=.2,random_state=7,bootstrap=False)
        X=(x.iloc[0].to_numpy()/12).reshape(-1,1)
        raw=iso.fit(X).decision_function(X)
        oracle=.5-raw/(2*(np.max(np.abs(raw))+1e-10))
    elif name=="cs_factor_bucket_return":
        kw=dict(factor=x,ret=x*.01,n_buckets=3,ascending=True)
        oracle=np.repeat([.025,.065,.105],4)
    elif name=="cs_empirical_bayes_shrinkage":
        kw=dict(estimate=x,std_err=x*.1,shrinkage_factor=2.)
        row=x.iloc[0].to_numpy()
        oracle=row.mean()+(row-row.mean())/(1+2*(row*.1)**2/np.var(row,ddof=1))
    else:
        kw=dict(x=x,group=group,shrinkage_intensity=.25)
        oracle=x.iloc[0].to_numpy()*.75+np.repeat([3.5,9.5],6)*.25
    got=run(name,backend,kw)
    np.testing.assert_allclose(got,np.tile(oracle,(3,1)),atol=1e-10,rtol=1e-10)
    op=OperatorRegistry.get(name,backend,mode="research")
    assert _parameter_contract(op,tuple(k for k,v in kw.items() if isinstance(v,pd.DataFrame)))[2]
    pd.testing.assert_frame_equal(result(op.calculate(*[convert(v) if backend=="polars" else v for v in kw.values()])),got,check_freq=False)
    short={k:v.iloc[:2] if isinstance(v,pd.DataFrame) else v for k,v in kw.items()}
    pd.testing.assert_frame_equal(run(name,backend,short),got.iloc[:2],check_freq=False)
    panels={k:v for k,v in kw.items() if isinstance(v,pd.DataFrame)}
    pd.testing.assert_frame_equal(run(name,backend,panels),run(name,"pandas_numpy",panels),check_freq=False)
    bad_key=next(k for k,v in kw.items() if not isinstance(v,pd.DataFrame))
    with pytest.raises((ValueError,TypeError)):
        run(name,backend,{**kw,bad_key:np.nan})

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_bayes_scale_and_exact_standard_error(backend):
    x=fixtures()
    for scale in (1e-200,1.,1e200):
        kw=dict(estimate=x*scale,std_err=x*.1*scale,shrinkage_factor=2.)
        got=run("cs_empirical_bayes_shrinkage",backend,kw)/scale
        expected=run("cs_empirical_bayes_shrinkage","pandas_numpy",dict(estimate=x,std_err=x*.1,shrinkage_factor=2.))
        np.testing.assert_allclose(got,expected,rtol=1e-12)
    np.testing.assert_allclose(run("cs_empirical_bayes_shrinkage",backend,dict(estimate=x,std_err=x*0)),x)
    np.testing.assert_allclose(run("cs_empirical_bayes_shrinkage",backend,dict(estimate=x,std_err=x*.1,shrinkage_factor=0)),x)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_group_finite_mean_and_invalid_labels(backend):
    x=fixtures()*0+1e308
    group=x*0+1
    got=run("cs_shrink_to_group_mean",backend,dict(x=x,group=group))
    np.testing.assert_allclose(got,x)
    for label in (np.nan,np.inf,""):
        invalid=group.astype(object)
        invalid.iloc[:,:]=label
        assert run("cs_shrink_to_group_mean",backend,dict(x=x,group=invalid)).isna().all().all()

def test_isolation_dependency_failure_is_explicit(monkeypatch):
    import builtins
    from factor_engine.cleaned_operators.cs_batch1 import CsIsolationForestScore
    original=builtins.__import__
    def reject(name,*args,**kwargs):
        if name=="sklearn.ensemble":
            raise ImportError("test missing dependency")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",reject)
    with pytest.raises(ImportError,match="requires scikit-learn"):
        CsIsolationForestScore().calculate(fixtures())
