"""Independent finite-support and history-window checks, after normal bootstrap."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def get(name,backend="pandas_numpy"):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None
    return op
def convert(x,backend):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if backend=="polars" else x
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",["ts_argmax","ts_argmin"])
def test_argext_ignores_infinity_preserves_original_age(name,backend):
    x=pd.DataFrame({"A":[5.,np.inf,3.,np.nan,2.,-np.inf,8.]},
                   index=pd.date_range("2026-01-01",periods=7))
    out=result(get(name,backend).calculate(convert(x,backend),3))
    expected=[]
    for row in range(len(x)):
        v=x.iloc[max(0,row-2):row+1,0].to_numpy()
        valid=np.isfinite(v)
        target=(max if name=="ts_argmax" else min)(v[valid])
        expected.append(len(v)-1-np.flatnonzero(valid&(v==target))[-1])
    pd.testing.assert_frame_equal(out,pd.DataFrame({"A":np.asarray(expected,dtype=float)},index=x.index),check_freq=False)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_cs_bucket_finite_only_and_invalid_direction(backend):
    x=pd.DataFrame([[1.,2.,np.inf,4.],[np.nan,-np.inf,3.,np.nan]],columns=list("ABCD"),
                   index=pd.date_range("2026-01-01",periods=2))
    out=result(get("cs_bucket",backend).calculate(convert(x,backend),4,True))
    expected=pd.DataFrame([[1.,3.,np.nan,4.],[np.nan,np.nan,3.,np.nan]],index=x.index,columns=x.columns)
    pd.testing.assert_frame_equal(out,expected,check_freq=False)
    with pytest.raises((TypeError,ValueError)):
        get("cs_bucket",backend).calculate(convert(x,backend),ascending="False")

def test_historical_bucket_uses_exact_prior_window():
    x=pd.DataFrame({"A":[1.,2.,3.,4.,.5,2.5]},index=pd.date_range("2026-01-01",periods=6))
    op=get("cs_bucket_historical")
    actual=op.calculate(x,window=3,quantiles=(.5,),min_periods=3)
    expected=pd.DataFrame({"A":[np.nan,np.nan,np.nan,2.,1.,1.]},index=x.index)
    pd.testing.assert_frame_equal(actual,expected)
    pd.testing.assert_frame_equal(op.calculate(x.iloc[:5],3,(.5,),3),expected.iloc[:5])
    for params in ({"min_periods":2.5},{"min_periods":4},{"quantiles":(.8,.2)},{"quantiles":(.5,.5)},{"quantiles":(np.nan,)}):
        kw=dict(window=3,quantiles=(.5,),min_periods=3)
        kw.update(params)
        with pytest.raises((TypeError,ValueError)):
            op.calculate(x,**kw)

def test_fixed_bucket_finite_sorted_boundaries():
    x=pd.DataFrame({"A":[-1.,0.,1.,2.,np.inf,np.nan]})
    pd.testing.assert_frame_equal(get("cs_bucket_fixed").calculate(x,[0.,1.]),
                                  pd.DataFrame({"A":[1.,2.,3.,3.,np.nan,np.nan]}))
    for breaks in ([np.nan],[np.inf],[1.,1.],[2.,1.],[]):
        with pytest.raises((TypeError,ValueError)):
            get("cs_bucket_fixed").calculate(x,breaks)
