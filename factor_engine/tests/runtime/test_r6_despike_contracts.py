"""Independent causal filter oracles and true final Polars bindings."""
from decimal import Decimal, localcontext
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_hampel_filter_causal","ts_median3_causal","ts_rolling_median_causal")
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def median(v):
    ordered=sorted(v);n=len(v)
    return ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2

def oracle(name,values,window=5,n_sigma=3.,replacement="clip",scale_floor=1e-10,min_periods=3):
    out=[]
    with localcontext() as ctx:
        ctx.prec=50
        for t,current in enumerate(values):
            if name==NAMES[0]:
                if not np.isfinite(current):
                    out.append(np.nan);continue
                past=[Decimal(str(v)) for v in values[max(0,t-window):t] if np.isfinite(v)]
                if t<window or len(past)<2:
                    out.append(current);continue
                m=median(past);mad=median([abs(v-m) for v in past])
                if mad==0:
                    span=max(past)-min(past)
                    if span==0:
                        out.append(current);continue
                    robust=Decimal("1.4826")*span/4
                else:
                    robust=Decimal("1.4826")*mad
                threshold=Decimal(str(n_sigma))*max(robust,Decimal(str(scale_floor)))
                delta=Decimal(str(current))-m
                answer=Decimal(str(current))
                if abs(delta)>threshold:
                    answer=m if replacement=="median" else m+max(-threshold,min(threshold,delta))
                out.append(float(answer))
            elif name==NAMES[1]:
                recent=values[max(0,t-2):t+1]
                if t<2:
                    out.append(current if np.isfinite(current) else np.nan)
                else:
                    out.append(float(median([Decimal(str(v)) for v in recent])) if np.isfinite(recent).all() else np.nan)
            else:
                recent=[Decimal(str(v)) for v in values[max(0,t-window+1):t+1] if np.isfinite(v)]
                out.append(float(median(recent)) if len(recent)>=min_periods else np.nan)
    return np.array(out)

def run(name,backend,values,**kw):
    index=pd.date_range("2025-01-01",periods=len(values),tz="Asia/Hong_Kong")
    panel=pd.DataFrame({"A":values},index=index)
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op.metadata.panel_params==("x",)
    assert _parameter_contract(op,("x",))[2]
    if backend=="polars":
        out=op.calculate(x=pl.from_pandas(panel.rename_axis("date").reset_index()),**kw)
        assert out["date"].to_list()==list(index)
        return out["A"].to_numpy()
    return op.calculate(x=panel,**kw).A.to_numpy()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_independent_oracle_defaults_prefix_scale(name,backend):
    values=np.tile([1.,2.,1.,3.,40.,2.,1.,4.,2.,3.],4)
    values[14]=np.nan;values[21]=np.inf
    kw={} if name==NAMES[1] else {"window":5}
    out=run(name,backend,values,**kw)
    np.testing.assert_allclose(out,oracle(name,values),equal_nan=True,rtol=1e-12)
    np.testing.assert_allclose(run(name,backend,values[:25],**kw),out[:25],equal_nan=True)
    defaults={} if name==NAMES[1] else ({"window":20,"n_sigma":3.,"replacement":"clip","scale_floor":1e-10}
                                      if name==NAMES[0] else {"window":5,"min_periods":3})
    np.testing.assert_allclose(run(name,backend,values),run(name,backend,values,**defaults),equal_nan=True)
    for scale in (1e-200,1e200):
        args=dict(kw)
        if name==NAMES[0]:
            args["scale_floor"]=1e-10*scale
        np.testing.assert_allclose(run(name,backend,values*scale,**args)/scale,out,equal_nan=True,rtol=1e-12)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_hampel_past_only_flat_bypass_exact_degeneracy_and_extremes(backend):
    for values in (
        np.array([0.,1.,0.,0.,0.,100.]),
        np.array([-1e308,1e308,-1e308,1e308,0.,1.7e308]),
        np.array([1e308,1e308,1e308,1e308,1e308,-1e308]),
    ):
        for policy in ("clip","median"):
            out=run(NAMES[0],backend,values,window=5,n_sigma=.5,replacement=policy)
            expected=oracle(NAMES[0],values,n_sigma=.5,replacement=policy)
            np.testing.assert_allclose(out,expected,rtol=1e-12,equal_nan=True)
            assert np.isfinite(out).all()
    for name in NAMES[1:]:
        out=run(name,backend,np.full(9,1.7e308))
        assert out[-1]==1.7e308
    assert np.isnan(run(NAMES[1],backend,np.array([np.inf,1.,2.]))[0])

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_strict_parameters(backend):
    v=np.arange(30.)
    for key,bad in (("window",2.5),("window",True),("n_sigma",True),("n_sigma",np.nan),
                    ("scale_floor",0),("scale_floor",np.inf),("replacement","bad")):
        with pytest.raises((ValueError,TypeError)):
            run(NAMES[0],backend,v,**{key:bad})
    for key,bad in (("window",2.5),("min_periods",True),("min_periods",6)):
        with pytest.raises((ValueError,TypeError)):
            run(NAMES[2],backend,v,**{key:bad})

@pytest.mark.parametrize("name",NAMES)
def test_native_does_not_materialize_pandas(name,monkeypatch):
    op=OperatorRegistry.get(name,"polars",mode="research")
    frame=pl.DataFrame({"A":np.arange(30.)})
    def forbidden(*a,**kw):
        raise AssertionError("despike native must not materialize pandas")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    assert np.isfinite(op.calculate(frame)["A"][-1])
