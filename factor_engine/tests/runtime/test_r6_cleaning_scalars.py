"""Scalar-cleaning final backends, numerical references and conversion bounds."""
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
def frame():
    return pd.DataFrame({"A":[1.,np.nan,3.,np.inf,5.,-np.inf,7.,9.],
                         "B":[np.nan,2.,np.nan,4.,4.,5.,6.,7.]},
                        index=pd.date_range("2025-01-01",periods=8))
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x
@pytest.mark.parametrize("name",["ts_ewm_std","ts_ewm_var"])
@pytest.mark.parametrize("span",[.2,1.,2.5,20.])
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_ewm_fractional_decay_missing_and_prefix(name,span,backend,monkeypatch):
    x=frame()
    op=OperatorRegistry.get(name,backend,mode="research")
    assert _parameter_contract(op,("x",))[2]
    a=span if span<1 else 2/(span+1)
    expected=getattr(x.ewm(alpha=a,adjust=False),name.removeprefix("ts_ewm_"))()
    args=convert(x) if backend=="polars" else x
    with monkeypatch.context() as patch:
        if backend=="polars":
            def forbid(*a,**k):
                raise AssertionError("No Pandas panel conversion in native cleaning kernel")
            patch.setattr(pl.DataFrame,"to_pandas",forbid)
        actual=op.calculate(args,span)
    pd.testing.assert_frame_equal(result(actual),expected,check_freq=False,rtol=1e-12,atol=1e-12)
    pd.testing.assert_frame_equal(result(op.calculate(x=args,span=span)),expected,check_freq=False,rtol=1e-12,atol=1e-12)
    short=convert(x.iloc[:6]) if backend=="polars" else x.iloc[:6]
    pd.testing.assert_frame_equal(result(op.calculate(short,span)),expected.iloc[:6],check_freq=False,rtol=1e-12,atol=1e-12)
    for bad in (0,-1,True,np.nan,np.inf):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(args,bad)

@pytest.mark.parametrize("name",["fillna_const","nonfinite_to_num"])
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_constant_missing_vs_nonfinite(name,backend):
    x=frame()
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None
    assert _parameter_contract(op,("x",))[2]
    key="value" if name=="fillna_const" else "num"
    args=convert(x) if backend=="polars" else x
    expected=x.fillna(2.5) if name=="fillna_const" else x.where(np.isfinite(x),2.5)
    actual=op.calculate(**{"x":args,key:2.5})
    pd.testing.assert_frame_equal(result(actual),expected,check_freq=False)
    pd.testing.assert_frame_equal(result(op.calculate(args,2.5)),expected,check_freq=False)
    for bad in (True,np.nan,np.inf,-np.inf):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(args,bad)
