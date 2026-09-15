"""Final weighted-tail contracts and independent weighted-risk oracles."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_stratified_mean_spread","ts_weighted_semivariance","ts_weighted_downside_deviation",
       "ts_weighted_expected_shortfall","ts_weighted_drawdown_area")

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if isinstance(x,pd.DataFrame) else x

def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

def run(name,backend,kw):
    return result(OperatorRegistry.get(name,backend,mode="research").calculate(
        **{k:convert(v) if backend=="polars" else v for k,v in kw.items()}))

@pytest.mark.parametrize("name",NAMES)
def test_weighted_final_scalar_topology_oracle(name):
    x=pd.DataFrame({"A":np.arange(100.,10.,-1)},index=pd.date_range("2025-01-01",periods=90))
    if name=="ts_stratified_mean_spread":
        kw=dict(target=x,sorter=x,window=20,quantile=.2,min_periods=5)
        expected=16.
    elif name=="ts_weighted_expected_shortfall":
        kw=dict(x=x,weight=x*0+1,window=20,quantile=.2,side="lower",min_tail_count=2)
        expected=12.5
    elif name=="ts_weighted_drawdown_area":
        kw=dict(x=x,weight=x*0+1,window=20)
        expected=np.mean(1-np.arange(30.,10.,-1)/30.)
    else:
        kw=dict(x=x,weight=x*0+1,window=20,target=50.,min_periods=2)
        expected=np.mean(np.arange(20.,40.)**2)
        if name=="ts_weighted_downside_deviation":
            expected=np.sqrt(expected)
    reference=run(name,"pandas_numpy",kw)
    assert reference.iloc[-1,0]==pytest.approx(expected)
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        assert _parameter_contract(op,tuple(k for k,v in kw.items() if isinstance(v,pd.DataFrame)))[2]
        pd.testing.assert_frame_equal(run(name,backend,kw),reference,check_freq=False,rtol=1e-10,atol=1e-10)
        args=[convert(v) if backend=="polars" else v for v in kw.values()]
        pd.testing.assert_frame_equal(result(op.calculate(*args)),reference,check_freq=False,rtol=1e-10,atol=1e-10)
        short={k:v.iloc[:70] if isinstance(v,pd.DataFrame) else v for k,v in kw.items()}
        pd.testing.assert_frame_equal(run(name,backend,short),reference.iloc[:70],check_freq=False,rtol=1e-10,atol=1e-10)
        panels={k:v for k,v in kw.items() if isinstance(v,pd.DataFrame)}
        pd.testing.assert_frame_equal(run(name,backend,panels),run(name,"pandas_numpy",panels),check_freq=False,rtol=1e-10,atol=1e-10)
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,{**kw,"window":0})

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_weighted_downside_extreme_scale_and_weight_invariance(backend):
    idx=pd.date_range("2025-01-01",periods=30)
    x=pd.DataFrame({"A":np.full(30,-1.)},index=idx)
    w=x*0+1
    for scale in (1e-200,1e200):
        got=run("ts_weighted_downside_deviation",backend,dict(x=x*scale,weight=w,window=20))
        np.testing.assert_allclose(got.iloc[1:]/scale,1.,atol=1e-12)
    for scale in (1e-200,1e200):
        got=run("ts_weighted_semivariance",backend,dict(x=x,weight=w*scale,window=20))
        np.testing.assert_allclose(got.iloc[1:],1.,atol=1e-12)
    huge=run("ts_weighted_semivariance",backend,dict(x=x*1e200,weight=w,window=20))
    assert huge.isna().all().all()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_weighted_negative_future_and_price_gap(backend):
    idx=pd.date_range("2025-01-01",periods=6)
    x=pd.DataFrame({"A":[10.,8.,np.nan,4.,3.,2.]},index=idx)
    w=x*0+1
    w.iloc[2]=1.
    got=run("ts_weighted_drawdown_area",backend,dict(x=x,weight=w,window=6))
    assert got.iloc[-1,0]==pytest.approx((0+.2+0+.25+.5)/5)
    changed=w.copy()
    changed.iloc[-1]=-1.
    bad=run("ts_weighted_drawdown_area",backend,dict(x=x,weight=changed,window=6))
    pd.testing.assert_frame_equal(bad.iloc[:-1],got.iloc[:-1])
    assert bad.iloc[-1].isna().all()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_weighted_support_rejects_impossible_calls(backend):
    x=pd.DataFrame({"A":np.arange(20.)},index=pd.date_range("2025-01-01",periods=20))
    for name in ("ts_weighted_semivariance","ts_weighted_downside_deviation"):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,dict(x=x,weight=x*0+1,window=10,min_periods=11))

def test_final_weighted_polars_avoids_pandas_panel_conversion(monkeypatch):
    idx=pd.date_range("2025-01-01",periods=70)
    x=pd.DataFrame({"A":np.arange(1.,71.)},index=idx)
    weight=x*0+1
    def forbidden(*args,**kwargs):
        raise AssertionError("unexpected Pandas panel conversion")
    for name in NAMES:
        op=OperatorRegistry.get(name,"polars",mode="research")
        values=[convert(x),convert(x if name=="ts_stratified_mean_spread" else weight)]
        with monkeypatch.context() as guard:
            guard.setattr(pl.DataFrame,"to_pandas",forbidden)
            guard.setattr(pd.DataFrame,"__init__",forbidden)
            out=op.calculate(*values)
        assert isinstance(out,pl.DataFrame)
