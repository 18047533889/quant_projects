"""Gaussian tied-rank and inclusive event-age final-backend contracts."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.special import ndtri
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
from factor_engine.runtime.execution_contract import own_history_requirement
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def run(name,backend,x,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert _parameter_contract(op,op.metadata.panel_params)[2]
    if backend=="polars":
        x=pl.from_pandas(x.rename_axis("date").reset_index())
    result=op.calculate(x,**kw)
    return result.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else result
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_gaussian_ties_independent_probability_oracle(backend):
    x=pd.DataFrame([[1.,2.,2.,4.,np.inf,np.nan]],columns=list("ABCDEF"))
    for method in ("blom","van_der_waerden"):
        ranks=np.array([1.,2.5,2.5,4.])
        p=(ranks-.375)/4.25 if method=="blom" else ranks/5
        out=run("cs_rank_gaussian",backend,x,method=method)
        np.testing.assert_allclose(out.iloc[0,:4],ndtri(p),atol=1e-12)
        assert out.iloc[0,4:].isna().all()
    with pytest.raises((ValueError,TypeError)):
        run("cs_rank_gaussian",backend,x*0,method="invalid")
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_days_since_real_zero_one_unknown_and_inclusive_limit(backend):
    x=pd.DataFrame({"A":[0.,1.,0.,0.,np.nan,0.,1.,0.]},
        index=pd.date_range("2025-01-01",periods=8,tz="Asia/Hong_Kong"))
    np.testing.assert_allclose(run("ts_days_since",backend,x).A,[np.nan,0,1,2,np.nan,np.nan,0,1],equal_nan=True)
    np.testing.assert_allclose(run("ts_days_since",backend,x,max_lookback=1).A,[np.nan,0,1,np.nan,np.nan,np.nan,0,1],equal_nan=True)
    out=run("ts_days_since",backend,x)
    np.testing.assert_allclose(run("ts_days_since",backend,x.iloc[:6]),out.iloc[:6],equal_nan=True)
    for bad in (np.inf,-np.inf,.5):
        invalid=x.copy();invalid.iloc[0]=bad
        with pytest.raises((ValueError,TypeError)):
            run("ts_days_since",backend,invalid)
    for bad in (True,0,1.5):
        with pytest.raises((ValueError,TypeError)):
            run("ts_days_since",backend,x,max_lookback=bad)
def test_unlimited_event_age_does_not_claim_finite_history():
    assert own_history_requirement("ts_days_since",{}).is_full_history
    op = OperatorRegistry.get("ts_days_since","polars",mode="research")
    assert op._physical_spec.stateful is True

@pytest.mark.parametrize("name",["ts_kurt","ts_sma_cn","lqtp_historical_cvar","ts_days_since"])
def test_temporal_native_contract_requires_ordered_rows(name):
    op = OperatorRegistry.get(name,"polars",mode="research")
    spec = op.physical_spec() if callable(getattr(op,"physical_spec",None)) else op._physical_spec
    assert spec.requires_sorted is True
