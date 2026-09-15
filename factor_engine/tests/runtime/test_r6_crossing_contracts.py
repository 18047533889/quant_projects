"""Real crossing geometry: exact event flags, missing adjacency and stable units."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_crossing_speed","ts_crossing_acceleration")
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frame(v):
    return pd.DataFrame({"A":v},index=pd.date_range("2025-01-01",periods=len(v),tz="Asia/Hong_Kong"))
def run(name,backend,x,y,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op.metadata.panel_params==("x","y")
    assert _parameter_contract(op,("x","y"))[2]
    if backend=="polars":
        convert=lambda p:pl.from_pandas(p.rename_axis("date").reset_index())
        out=op.calculate(x=convert(x),y=convert(y),**kw)
        return out.to_pandas().set_index("date").rename_axis(None)
    return op.calculate(x=x,y=y,**kw)
def oracle(name,x,y,window):
    z=x-y;out=np.zeros(len(z))
    for t in range(len(z)):
        if not np.isfinite(z[t]) or (t>0 and not np.isfinite(z[t-1])):
            out[t]=np.nan;continue
        if t==0 or (name==NAMES[1] and t<2):
            continue
        up=z[t-1]<=0<z[t];down=z[t-1]>=0>z[t]
        if not(up or down):
            continue
        v=z[max(0,t-window+1):t+1];v=v[np.isfinite(v)]
        scale=np.std(v)
        if scale<=0 or (name==NAMES[1] and not np.isfinite(z[t-2])):
            out[t]=np.nan;continue
        numerator=z[t]-z[t-1] if name==NAMES[0] else z[t]-2*z[t-1]+z[t-2]
        out[t]=numerator/scale
    return out

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_independent_crossing_defaults_scale_prefix_and_missing(name,backend):
    values=np.tile([-2.,-1.,1.,3.,-4.,-2.,2.],5)
    x=frame(values);y=frame(np.zeros(len(values)))
    x.iloc[12]=np.nan
    got=run(name,backend,x,y,window=5)
    np.testing.assert_allclose(got.A,oracle(name,x.A.to_numpy(),y.A.to_numpy(),5),equal_nan=True,atol=1e-12)
    assert got.iloc[12:14].isna().all().all()
    np.testing.assert_allclose(run(name,backend,x.iloc[:25],y.iloc[:25],window=5),got.iloc[:25],equal_nan=True)
    np.testing.assert_allclose(run(name,backend,x,y),run(name,backend,x,y,window=20),equal_nan=True)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(run(name,backend,x*scale,y*scale,window=5),got,equal_nan=True,atol=1e-12)
    for bad in (True,1,2.5,np.inf):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,x,y,window=bad)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_finite_difference_overflow_and_truly_flat_history(name,backend):
    v=np.tile([-1.,1.,-1.,1.],5)
    x=frame(v*1e308);y=frame(-v*1e308)
    out=run(name,backend,x,y,window=4)
    expected=oracle(name,v,-v,4)
    np.testing.assert_allclose(out.A,expected,atol=1e-12)
    flat=run(name,backend,frame(np.ones(20)),frame(np.ones(20)),window=4)
    assert (flat==0).all().all()

@pytest.mark.parametrize("name",NAMES)
def test_native_no_pandas_materialization_and_misalignment(name,monkeypatch):
    x=pl.DataFrame({"date":pd.date_range("2025-01-01",periods=20),"A":np.tile([-1.,1.],10)})
    y=x.with_columns(pl.lit(0.).alias("A"))
    op=OperatorRegistry.get(name,"polars",mode="research")
    def forbidden(*a,**kw):
        raise AssertionError("crossing native must not materialize pandas")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    assert np.isfinite(op.calculate(x,y)["A"][-1])
    shifted=y.with_columns(pl.col("date")+pl.duration(days=1))
    with pytest.raises((ValueError,TypeError)):
        op.calculate(x,shifted)
