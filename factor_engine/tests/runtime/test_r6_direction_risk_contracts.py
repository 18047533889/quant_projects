"""Actual direction/risk backends, finite-support oracles and no-Pandas execution."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.common.direction_risk_polars import PARAMS
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frames():
    rng=np.random.default_rng(716)
    idx=pd.date_range("2025-01-01",periods=48)
    x=pd.DataFrame(rng.normal(0,.03,(48,2)),index=idx,columns=["A","B"])
    y=.5*x+.8*x.shift(1).fillna(0)+pd.DataFrame(rng.normal(0,.003,(48,2)),index=idx,columns=x.columns)
    return x,y
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

@pytest.mark.parametrize("name",sorted(PARAMS))
def test_final_risk_call_contract_prefix_and_native_parity(name,monkeypatch):
    x,y=frames()
    values=dict(x=x,y=y,stock_return=y,benchmark_return=x,window=20,threshold=.01,
                tolerance=.01,target=.01,min_periods=3,normalize=True,max_lag=2)
    if name in ("ts_current_drawdown_duration","ts_time_under_water"):
        values["x"]=100+x.cumsum()
        values["x"].iloc[12,0]=np.nan
    kwargs={k:values[k] for k in PARAMS[name]}
    pandas_op=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    expected=pandas_op.calculate(**kwargs)
    assert np.isfinite(expected.to_numpy()).any(),name
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        assert op is not None
        inputs={k:convert(v) if backend=="polars" and isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        if backend=="polars":
            def forbidden(*a,**k):
                raise AssertionError("Native risk call must not construct Pandas panels")
            with monkeypatch.context() as m:
                m.setattr(pl.DataFrame,"to_pandas",forbidden)
                actual=op.calculate(**inputs)
        else:
            actual=op.calculate(**inputs)
        pd.testing.assert_frame_equal(result(actual),expected,check_freq=False,atol=1e-10,rtol=1e-10)
        pd.testing.assert_frame_equal(result(op.calculate(*inputs.values())),expected,check_freq=False,atol=1e-10,rtol=1e-10)
        short={k:(convert(v.iloc[:40]) if backend=="polars" else v.iloc[:40]) if isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        pd.testing.assert_frame_equal(result(op.calculate(**short)),expected.iloc[:40],check_freq=False,atol=1e-10,rtol=1e-10)
        panel_names=tuple(k for k,v in kwargs.items() if isinstance(v,pd.DataFrame))
        assert _parameter_contract(op,panel_names)[2],(name,backend)
        with pytest.raises((TypeError,ValueError)):
            op.calculate(**{**inputs,"window":0})

@pytest.mark.parametrize("name",["ts_positive_ratio","ts_negative_ratio","ts_zero_ratio"])
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_ratio_independent_missing_support_and_threshold(name,backend):
    x=pd.DataFrame({"A":[-.1,.1,0.,np.nan,np.inf,-.2]},index=pd.date_range("2025-01-01",periods=6))
    threshold=.05
    op=OperatorRegistry.get(name,backend,mode="research")
    args=convert(x) if backend=="polars" else x
    key="tolerance" if name=="ts_zero_ratio" else "threshold"
    out=result(op.calculate(args,window=3,min_periods=1,**{key:threshold}))
    expected=[]
    for t in range(len(x)):
        v=x.A.iloc[max(0,t-2):t+1].to_numpy()
        v=v[np.isfinite(v)]
        mask=v>threshold if name=="ts_positive_ratio" else v<threshold if name=="ts_negative_ratio" else np.abs(v)<=threshold
        expected.append(np.mean(mask) if len(v) else np.nan)
    pd.testing.assert_frame_equal(out,pd.DataFrame({"A":expected},index=x.index),check_freq=False)
    for params in ({"window":2,"min_periods":3},{key:True},{"window":2.5}):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(args,**params)

@pytest.mark.parametrize("name",["ts_abs_concentration","ts_abs_entropy_normalized","ts_abs_entropy_nats"])
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_mass_large_scale_invariance(name,backend):
    x=pd.DataFrame({"A":[1.,2.,1.]},index=pd.date_range("2025-01-01",periods=3))
    op=OperatorRegistry.get(name,backend,mode="research")
    call=lambda frame:result(op.calculate(convert(frame) if backend=="polars" else frame,3))
    pd.testing.assert_frame_equal(call(x),call(x*5e307),check_freq=False,atol=1e-12,rtol=1e-12)
    weights=np.array([.25,.5,.25])
    expected=np.sum(weights**2) if name=="ts_abs_concentration" else -np.sum(weights*np.log(weights))/(np.log(3) if name.endswith("normalized") else 1)
    assert call(x).iloc[-1,0]==pytest.approx(expected)

@pytest.mark.parametrize("name,sign",[("ts_downside_deviation",-1),("ts_upside_deviation",1)])
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_deviation_large_values_have_finite_rms(name,sign,backend):
    x=pd.DataFrame({"A":sign*np.array([1e200,2e200,3e200])},index=pd.date_range("2025-01-01",periods=3))
    op=OperatorRegistry.get(name,backend,mode="research")
    out=result(op.calculate(convert(x) if backend=="polars" else x,3))
    assert out.iloc[-1,0]==pytest.approx(np.sqrt(14/3)*1e200)
