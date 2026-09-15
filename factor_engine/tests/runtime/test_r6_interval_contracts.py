"""Final interval-geometry contracts and independent prior-history oracle."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.interval_geometry import _NEW_CANONICALS
from factor_engine.runtime.operator_snapshot import _parameter_contract
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x
@pytest.mark.parametrize("name",_NEW_CANONICALS)
def test_final_interval_scalar_binding_numerics_prefix(name):
    rng=np.random.default_rng(51)
    low=pd.DataFrame(rng.normal(size=(45,2)).cumsum(axis=0)+30,index=pd.date_range("2025-01-01",periods=45),columns=["A","B"])
    high=low+2
    values=dict(low=low,high=high,x=low+1,close=low+1,window=20,bins=8,mode="inside")
    ref=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    kwargs={k:values[k] for k in ref.metadata.param_names}
    expected=ref.calculate(**kwargs)
    assert np.isfinite(expected.to_numpy()).any()
    panels=tuple(k for k,v in kwargs.items() if isinstance(v,pd.DataFrame))
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        assert op is not None
        assert _parameter_contract(op,panels)[2],(name,backend)
        inputs={k:convert(v) if backend=="polars" and isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        pd.testing.assert_frame_equal(result(op.calculate(**inputs)),expected,check_freq=False,rtol=1e-10,atol=1e-10)
        pd.testing.assert_frame_equal(result(op.calculate(*inputs.values())),expected,check_freq=False,rtol=1e-10,atol=1e-10)
        short={k:(convert(v.iloc[:35]) if backend=="polars" else v.iloc[:35]) if isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        pd.testing.assert_frame_equal(result(op.calculate(**short)),expected.iloc[:35],check_freq=False,rtol=1e-10,atol=1e-10)
        key="mode" if "mode" in inputs else "window"
        with pytest.raises((TypeError,ValueError)):
            op.calculate(**{**inputs,key:"bad" if key=="mode" else 0})
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_mode_distance_uses_exactly_window_prior_intervals(backend):
    idx=pd.date_range("2025-01-01",periods=3)
    low=pd.DataFrame({"A":[0.,4.,4.]},index=idx)
    high=low+1
    x=pd.DataFrame({"A":[.5,4.5,4.5]},index=idx)
    op=OperatorRegistry.get("ts_interval_occupancy_mode_distance",backend,mode="research")
    args=[convert(a) if backend=="polars" else a for a in (x,low,high)]
    got=result(op.calculate(*args,window=2,bins=2))
    assert got.iloc[2,0]==pytest.approx(.65)
    x.iloc[2,0]=np.inf
    args[0]=convert(x) if backend=="polars" else x
    assert np.isnan(result(op.calculate(*args,window=2,bins=2)).iloc[2,0])
