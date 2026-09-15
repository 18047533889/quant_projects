"""Real conditional covariance and weighted standardized moment reference paths."""
from decimal import Decimal,localcontext
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

COV="ts_cov_if";MOMENT="ts_weighted_standardized_moment"
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frame(v):
    return pd.DataFrame({"A":v},index=pd.date_range("2025-01-01",periods=len(v),tz="Asia/Hong_Kong"))
def run(name,backend,panels,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    names=("x","y","condition") if name==COV else ("x","weight")
    assert op.metadata.panel_params==names
    assert _parameter_contract(op,names)[2]
    if backend=="polars":
        panels=[pl.from_pandas(p.rename_axis("date").reset_index()) for p in panels]
    out=op.calculate(**dict(zip(names,panels)),**kw)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out
def cov_oracle(x,y):
    with localcontext() as ctx:
        ctx.prec=80
        a=[Decimal(str(v)) for v in x];b=[Decimal(str(v)) for v in y]
        ma=sum(a)/len(a);mb=sum(b)/len(b)
        return float(sum((u-ma)*(v-mb) for u,v in zip(a,b))/(len(a)-1))

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_covariance_defaults_mask_prefix_and_safe_products(backend):
    x=frame(np.arange(30,dtype=float));y=frame(np.sin(np.arange(30))+.3*np.arange(30))
    cond=frame(np.tile([0.,1.,1.],10));cond.iloc[12]=np.nan
    out=run(COV,backend,[x,y,cond])
    np.testing.assert_allclose(out,run(COV,backend,[x,y,cond],window=20,min_periods=2),equal_nan=True)
    for t in range(2,len(x)):
        lo=max(0,t-19);mask=cond.A.iloc[lo:t+1].to_numpy()==1.
        if mask.sum()>=2:
            expected=cov_oracle(x.A.iloc[lo:t+1].to_numpy()[mask],y.A.iloc[lo:t+1].to_numpy()[mask])
            assert out.A.iloc[t]==pytest.approx(expected,abs=1e-12)
    np.testing.assert_allclose(run(COV,backend,[x.iloc[:21],y.iloc[:21],cond.iloc[:21]]),out.iloc[:21],equal_nan=True)
    for sx,sy in ((1e200,1e-200),(1e-200,1e200)):
        np.testing.assert_allclose(run(COV,backend,[x*sx,y*sy,cond]),out,equal_nan=True,atol=1e-10)
    constant=frame(np.full(5,1.7e308))
    got=run(COV,backend,[constant,constant,frame(np.ones(5))],window=5)
    assert got.iloc[-1,0]==0.
    v=frame(np.array([-1.,1.,-1.,1.])*1e200)
    assert run(COV,backend,[v,v,frame(np.ones(4))],window=4).isna().all().all()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("order",[3,4])
def test_weighted_moment_exact_weight_support_scale_and_defaults(backend,order):
    x=frame(np.arange(30.)**2);w=frame(np.linspace(1.,2.,30))
    out=run(MOMENT,backend,[x,w],order=order)
    weights=w.A.to_numpy()[-20:];weights=weights/weights.sum()
    values=x.A.to_numpy()[-20:];dev=values-np.dot(weights,values)
    expected=np.dot(weights,dev**order)/np.dot(weights,dev**2)**(order/2)
    assert out.iloc[-1,0]==pytest.approx(expected,abs=1e-12)
    if order==3:
        np.testing.assert_allclose(out,run(MOMENT,backend,[x,w]),equal_nan=True)
    for sx,sw in ((1e-200,1e300),(1e200,1e-300)):
        np.testing.assert_allclose(run(MOMENT,backend,[x*sx,w*sw],order=order),out,equal_nan=True,atol=1e-10)
    for invalid in (-1.,np.nan,np.inf):
        weights=w.copy();weights.iloc[12]=invalid
        assert run(MOMENT,backend,[x,weights],window=5,order=order).iloc[12:17].isna().all().all()
    np.testing.assert_allclose(run(MOMENT,backend,[x.iloc[:22],w.iloc[:22]],order=order),out.iloc[:22],equal_nan=True)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_parameter_feasibility_and_condition_bool(backend):
    x=frame(np.arange(25.));one=frame(np.ones(25))
    for name,key,bad in ((MOMENT,"window",True),(MOMENT,"window",2),(MOMENT,"order",2),
                         (COV,"window",2.5),(COV,"min_periods",1),(COV,"min_periods",21)):
        panels=[x,x,one] if name==COV else [x,one]
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,panels,**{key:bad})
    with pytest.raises((ValueError,TypeError)):
        run(MOMENT,backend,[x,one],window=3,order=4)
    for bad in (2.,np.inf):
        c=one.copy();c.iloc[4]=bad
        with pytest.raises((ValueError,TypeError)):
            run(COV,backend,[x,x,c])
    shifted=x.copy();shifted.index+=pd.Timedelta(days=1)
    with pytest.raises((ValueError,TypeError)):
        run(COV,backend,[x,shifted,one])
