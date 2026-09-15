"""Final-registry source-safe calls, independent expected values and policy rejection."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.common.safe_kernels import EXTREMES
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def conv(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def out(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x
def call(name,backend,kwargs):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None,(name,backend)
    panels=tuple(k for k,v in kwargs.items() if isinstance(v,pd.DataFrame))
    assert _parameter_contract(op,panels)[2],(name,backend)
    values={k:conv(v) if backend=="polars" and isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
    result=out(op.calculate(**values))
    pd.testing.assert_frame_equal(result,out(op.calculate(*values.values())),check_freq=False)
    shorter={k:(conv(v.iloc[:-1]) if backend=="polars" else v.iloc[:-1]) if isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
    pd.testing.assert_frame_equal(result.iloc[:-1],out(op.calculate(**shorter)),check_freq=False)
    return result,op,values

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",EXTREMES)
def test_extreme_finite_ties_and_real_bar_positions(name,backend):
    values=[1.,3.,np.inf,3.,np.nan,2.]
    x=pd.DataFrame({"A":values},index=pd.date_range("2025-01-01",periods=6))
    actual,op,inputs=call(name,backend,dict(x=x,window=3,min_periods=1))
    expected=[]
    for i in range(6):
        segment=np.array(values[max(0,i-2):i+1])
        positions=np.flatnonzero(np.isfinite(segment))
        target=(max if "argmax" in name else min)(segment[positions])
        position=positions[segment[positions]==target][-1]
        expected.append(len(segment)-1-position if name.endswith("_age") else position)
    np.testing.assert_array_equal(actual.A,expected)
    for bad in (0,1.5,True,4):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(**{**inputs,"min_periods":bad})

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_group_imputation_is_cross_sectional_bounded_and_finite(backend):
    idx=pd.date_range("2025-01-01",periods=3)
    x=pd.DataFrame([[1.,3.,5.,np.nan,50.,np.nan],[10.,np.inf,30.,50.,8.,np.nan],
                    [1e308,1e308,1e308,np.nan,8.,np.nan]],index=idx,columns=list("ABCDEF"))
    group=pd.DataFrame([["g","g","g","g","h","h"]]*3,index=idx,columns=x.columns)
    actual,op,inputs=call("group_impute_median",backend,dict(x=x,group=group,min_group_size=3))
    expected=x.copy()
    expected.iloc[0,3]=3.
    expected.iloc[1,1]=30.
    expected.iloc[2,3]=1e308
    pd.testing.assert_frame_equal(actual,expected,check_freq=False)
    for bad in (0,True,2.5):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(**{**inputs,"min_group_size":bad})

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_ffill_nan_limit_and_lineage_are_enforced(backend):
    x=pd.DataFrame({"A":[1.,np.nan,np.nan,np.nan,2.,np.nan]},index=pd.date_range("2025-01-01",periods=6))
    actual,op,inputs=call("ts_ffill_limited",backend,dict(x=x,max_gap=2,lineage="price"))
    expected=pd.DataFrame({"A":[1.,1.,1.,np.nan,2.,2.]},index=x.index)
    pd.testing.assert_frame_equal(actual,expected,check_freq=False)
    pd.testing.assert_frame_equal(out(op.calculate(inputs["x"],2)),expected,check_freq=False)
    for bad in ("financial","return","event","revision","unknown",2):
        with pytest.raises((TypeError,ValueError)):
            op.calculate(**{**inputs,"lineage":bad})
