"""Real final vector geometry bindings, not single-input placeholder formulas."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_vector_path_efficiency","ts_vector_turning_coherence",
       "ts_vector_path_curvature","ts_vector_self_intersection_rate")
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def panels(n=70):
    t=np.arange(n,dtype=float)
    index=pd.date_range("2025-01-01",periods=n,tz="Asia/Hong_Kong")
    return (pd.DataFrame({"A":np.sin(t*.7)+t*.1,"B":t%7},index=index),
            pd.DataFrame({"A":np.cos(t*.6)+t*.04,"B":(t*3)%11},index=index))

def run(name,backend,a,b,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None
    assert op.metadata.panel_params==("f1","f2")
    assert _parameter_contract(op,("f1","f2"))[2]
    if backend=="polars":
        convert=lambda x:pl.from_pandas(x.rename_axis("date").reset_index())
        out=op.calculate(f1=convert(a),f2=convert(b),**kw)
        return out.to_pandas().set_index("date").rename_axis(None)
    return op.calculate(f1=a,f2=b,**kw)

def oracle(name,x,y):
    pts=np.column_stack([x,y])
    if name==NAMES[3]:
        segments=[(pts[i],pts[i+1]) for i in range(len(pts)-1)
                  if not np.array_equal(pts[i],pts[i+1])]
        pairs=hits=0
        def cross(a,b,c):
            d,e=b-a,c-a
            return d[0]*e[1]-d[1]*e[0]
        for i,(a,b) in enumerate(segments):
            for j,(c,d) in enumerate(segments):
                if j<=i+1:
                    continue
                if i==0 and j==len(segments)-1 and np.array_equal(a,d):
                    continue
                pairs+=1
                hits+=cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0
        return hits/pairs if pairs else np.nan
    z=(pts-pts.mean(axis=0))/pts.std(axis=0)
    v=np.diff(z,axis=0)
    speeds=np.sqrt(np.sum(v*v,axis=1))
    if name==NAMES[0]:
        return np.linalg.norm(z[-1]-z[0])/(speeds.sum()+1e-12)
    if name==NAMES[1]:
        valid=(speeds[1:]>1e-12)&(speeds[:-1]>1e-12)
        return np.mean(np.sum(v[1:]*v[:-1],axis=1)[valid]/(speeds[1:]*speeds[:-1])[valid])
    acceleration=np.diff(v,axis=0)
    return np.median(np.abs(v[:-1,0]*acceleration[:,1]-v[:-1,1]*acceleration[:,0])/
                     (speeds[:-1]**3+1e-12))

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_independent_geometry_defaults_scale_prefix(name,backend):
    a,b=panels()
    got=run(name,backend,a,b,window=8)
    assert got.iloc[:7].isna().all().all()
    for c in a.columns:
        expected=oracle(name,a[c].iloc[-8:].to_numpy(),b[c].iloc[-8:].to_numpy())
        assert got[c].iloc[-1]==pytest.approx(expected,abs=1e-11)
    np.testing.assert_allclose(run(name,backend,a.iloc[:15],b.iloc[:15],window=8),
                               got.iloc[:15],atol=1e-11,equal_nan=True)
    np.testing.assert_allclose(run(name,backend,a,b),run(name,backend,a,b,window=60),
                               atol=1e-11,equal_nan=True)
    for sx,sy in ((1e-200,1e200),(1e200,1e-200)):
        np.testing.assert_allclose(run(name,backend,a*sx,b*sy,window=8),got,
                                   atol=1e-10,rtol=1e-10,equal_nan=True)
    a.iloc[10]=np.nan
    assert run(name,backend,a,b,window=8).iloc[10:18].isna().all().all()
    for bad in (True,1,2.5,np.nan):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,a,b,window=bad)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_true_crossing_survives_unit_change(backend):
    idx=pd.date_range("2025-01-01",periods=4)
    a=pd.DataFrame({"A":[0.,1.,0.,1.]},index=idx)
    b=pd.DataFrame({"A":[0.,1.,1.,0.]},index=idx)
    for scale in (1.,1e-200,1e200):
        out=run(NAMES[3],backend,a*scale,b*scale,window=4)
        assert out.iloc[-1,0]==1.

@pytest.mark.parametrize("name",NAMES)
def test_native_no_pandas_materialization_and_true_time_axis(name,monkeypatch):
    a,b=panels(12)
    a=pl.from_pandas(a.rename_axis("date").reset_index())
    b=pl.from_pandas(b.rename_axis("date").reset_index())
    op=OperatorRegistry.get(name,"polars",mode="research")
    def forbidden(*args,**kwargs):
        raise AssertionError("native path must not materialize Pandas")
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    out=op.calculate(a,b,window=8)
    assert out.shape==a.shape and out["date"].equals(a["date"])
