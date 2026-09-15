"""Fresh-load contracts and exact native CPU expectile execution."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import get_physical_spec
from scipy.optimize import brentq, least_squares

def oracle(v,tau):
    return brentq(lambda e: np.dot(np.where(v>=e,tau,1-tau),v-e),
                  float(v.min()),float(v.max()),xtol=1e-13)

def reference(y,x,tau):
    a=np.column_stack([x,np.ones(len(x))])
    def residual(beta):
        r=y-a@beta
        return np.sqrt(np.where(r>=0,tau,1-tau))*r
    def jac(beta):
        r=y-a@beta
        return -np.sqrt(np.where(r>=0,tau,1-tau))[:,None]*a
    result=least_squares(residual,[0.,0.],jac=jac,ftol=1e-13,xtol=1e-13,gtol=1e-13)
    assert np.max(np.abs(jac(result.x).T@residual(result.x)))<1e-8
    return result.x[0]

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def inputs():
    rng=np.random.default_rng(722)
    x=rng.normal(size=80)
    y=3*x+rng.lognormal(size=80)
    index=pd.date_range("2025-01-01",periods=80,tz="Asia/Hong_Kong")
    return pd.DataFrame({"A":x},index=index),pd.DataFrame({"A":y},index=index)

def execute(name,backend,panels,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    names=("y","x") if name.endswith("beta") else ("x",)
    assert op.metadata.panel_params==names
    assert _parameter_contract(op,names)[2]
    if backend=="polars":
        panels=[pl.from_pandas(p.rename_axis("date").reset_index()) for p in panels]
    out=op.calculate(**dict(zip(names,panels)),**kw)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",["ts_expectile","ts_expectile_beta"])
def test_defaults_true_panels_oracles_scale_and_prefix(name,backend):
    x,y=inputs()
    panels=[y,x] if name.endswith("beta") else [x]
    got=execute(name,backend,panels)
    np.testing.assert_allclose(got,execute(name,backend,panels,window=60,tau=.1,n_min=3),equal_nan=True)
    expected=reference(y.A.to_numpy()[-60:],x.A.to_numpy()[-60:],.1) if name.endswith("beta") else oracle(x.A.to_numpy()[-60:],.1)
    assert got.A.iloc[-1]==pytest.approx(expected,abs=1e-8)
    np.testing.assert_allclose(execute(name,backend,[p.iloc[:65] for p in panels]),got.iloc[:65],equal_nan=True)
    for scale in (1e-200,1e200):
        scaled=execute(name,backend,[p*scale for p in panels])
        if not name.endswith("beta"):
            scaled=scaled/scale
        np.testing.assert_allclose(scaled,got,equal_nan=True,atol=1e-10)
    for key,bad in (("window",3.5),("window",True),("n_min",1.5),("n_min",True),("tau",True),("tau",0),("tau",1),("tau",np.inf)):
        with pytest.raises((ValueError,TypeError)):
            execute(name,backend,panels,**{key:bad})

@pytest.mark.parametrize("name",["ts_expectile","ts_expectile_beta"])
def test_no_pandas_panel_conversion_and_honest_kind(name,monkeypatch):
    x,y=inputs()
    panels=[y,x] if name.endswith("beta") else [x]
    panels=[pl.from_pandas(p.rename_axis("date").reset_index()) for p in panels]
    op=OperatorRegistry.get(name,"polars",mode="research")
    spec=get_physical_spec(op)
    assert spec.execution_kind is ExecutionKind.POLARS_NUMPY_KERNEL
    def forbidden(*a,**kw):
        raise AssertionError("native expectile must not convert a Pandas panel")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    out=op.calculate(*panels)
    assert np.isfinite(out["A"][-1])
    assert out["date"].equals(panels[0]["date"])

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_pair_identity_mismatch_rejected(backend):
    x,y=inputs()
    y.index=y.index+pd.Timedelta(days=1)
    with pytest.raises((ValueError,TypeError)):
        execute("ts_expectile_beta",backend,[y,x])
