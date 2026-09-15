"""Actual final six scalar-bearing elementwise kernels; independent row references."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

def get(name, backend):
    op = OperatorRegistry.get(name, backend, mode="research")
    assert op is not None, (name, backend)
    return op

def source():
    return pd.DataFrame([[1.25,-2.55,10.1,30.],[1.,np.nan,5.,20.],
                         [3.,np.inf,-np.inf,8.],[np.nan]*4,[9.,2.,7.,1.]],
                        columns=list("ABCD"), index=pd.date_range("2026-01-01",periods=5))

def convert(x, backend):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if backend=="polars" else x

def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",["lerp","round","truncate","winsorize","winsorize_mean"])
def test_final_numeric_keywords_prefix_contract(name,backend):
    x=source()
    params={"lerp":{"fraction":.25},"round":{"decimals":1},"truncate":{"decimals":1},
            "winsorize":{"lower":.2,"upper":.8},"winsorize_mean":{"trim_pct":.2}}[name]
    panels={"a":x,"b":x+3} if name=="lerp" else {"x":x}
    inputs={k:convert(v,backend) for k,v in panels.items()}
    op=get(name,backend)
    actual=result(op.calculate(**inputs,**params))
    pd.testing.assert_frame_equal(actual,result(op.calculate(*inputs.values(),*params.values())),check_freq=False)
    pd.testing.assert_frame_equal(actual,result(op.calculate(*inputs.values(),**params)),check_freq=False)
    finite=x.replace([np.inf,-np.inf],np.nan)
    if name=="lerp":
        expected=(x+.75).replace([np.inf,-np.inf],np.nan)
    elif name=="round":
        expected=x.round(1)
    elif name=="truncate":
        expected=np.trunc(x*10)/10
    else:
        clipped=finite.clip(lower=finite.quantile(.2,axis=1),upper=finite.quantile(.8,axis=1),axis=0)
        expected=clipped if name=="winsorize" else pd.DataFrame(np.repeat(clipped.mean(axis=1).to_numpy()[:,None],4,axis=1),index=x.index,columns=x.columns)
    pd.testing.assert_frame_equal(actual,expected,check_freq=False,atol=1e-12,rtol=1e-12)
    assert np.isfinite(actual.to_numpy()).any()
    pd.testing.assert_frame_equal(result(op.calculate(**{k:convert(v.iloc[:3],backend) for k,v in panels.items()},**params)),actual.iloc[:3],check_freq=False)
    _,_,verified=_parameter_contract(op,tuple(panels))
    assert verified,(name,backend)
    pd.testing.assert_frame_equal(result(op.calculate(**inputs)),result(op.calculate(*inputs.values())),check_freq=False)
    bad={"lerp":{"fraction":True},"round":{"decimals":1.2},"truncate":{"decimals":19},
         "winsorize":{"lower":.9,"upper":.1},"winsorize_mean":{"trim_pct":.5}}[name]
    with pytest.raises((TypeError,ValueError)):
        op.calculate(**inputs,**bad)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_constant_scalar_default_and_keyword(backend):
    op=get("constant",backend)
    assert op.calculate()==0.
    assert op.calculate(2.5)==op.calculate(c=2.5)==2.5
    _,_,verified=_parameter_contract(op,())
    assert verified
    for bad in (True,np.inf,np.nan,[]):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(bad)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",["round","truncate"])
@pytest.mark.parametrize("decimals",[-18,-1,0,18])
def test_rounding_large_finite_and_negative_decimals(name,backend,decimals):
    x=pd.DataFrame({"A":[1e308,-1e308,125.,-125.,123.456,-123.456]},
                   index=pd.date_range("2026-01-01",periods=6))
    op=get(name,backend)
    actual=result(op.calculate(convert(x,backend),decimals))
    assert np.isfinite(actual.to_numpy()).all()
    assert actual.iloc[0,0]==1e308 and actual.iloc[1,0]==-1e308
    scale=10.**decimals
    expected=x.iloc[2:].round(decimals) if name=="round" else np.trunc(x.iloc[2:]*scale)/scale
    pd.testing.assert_frame_equal(actual.iloc[2:],expected,check_freq=False)
