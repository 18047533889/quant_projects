"""Recursive SMA and historical CVaR real defaults, support and scale tests."""
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
def run(name,backend,pdf,*args,**kwargs):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert _parameter_contract(op,op.metadata.panel_params)[2]
    data=pl.from_pandas(pdf.rename_axis("date").reset_index()) if backend=="polars" else pdf
    out=op.calculate(data,*args,**kwargs)
    return out.to_pandas().set_index("date") if backend=="polars" else out
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_recursive_sma_missing_state_and_default(backend):
    x=pd.DataFrame({"A":[1.,2.,np.nan,4.,np.inf,8.]})
    state=1.;want=[]
    for value in x.A:
        if not np.isfinite(value):
            want.append(np.nan);continue
        state=(2*value+5*state)/7
        want.append(state)
    np.testing.assert_allclose(run("ts_sma_cn",backend,x).A,want,equal_nan=True)
    np.testing.assert_allclose(run("ts_sma_cn",backend,x,7,2).A,want,equal_nan=True)
    for kw in ({"n":True},{"m":2.5},{"n":1,"m":2}):
        with pytest.raises((ValueError,TypeError)):
            run("ts_sma_cn",backend,x,**kw)
    from factor_engine.runtime.execution_contract import own_history_requirement
    assert own_history_requirement("ts_sma_cn",{}).is_full_history
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_cvar_finite_support_and_extreme_values(backend):
    x=pd.DataFrame({"A":[-4.,1.,-2.,np.nan,np.inf,8.]})
    expected=[]
    for i in range(len(x)):
        vals=x.A.iloc[max(0,i-2):i+1].to_numpy()
        vals=vals[np.isfinite(vals)]
        expected.append(-np.mean(vals[vals<=np.quantile(vals,.5)]) if len(vals) else np.nan)
    np.testing.assert_allclose(run("lqtp_historical_cvar",backend,x,3,.5).A,expected,equal_nan=True)
    extreme=pd.DataFrame({"A":[-1e308,1e308]})
    np.testing.assert_allclose(run("lqtp_historical_cvar",backend,extreme,2,1.).A,[1e308,0.])
    assert run("lqtp_historical_cvar",backend,x).shape==x.shape
    for kw in ({"window":True},{"window":2.5},{"q":0},{"q":True},{"q":np.inf}):
        with pytest.raises((ValueError,TypeError)):
            run("lqtp_historical_cvar",backend,x,**kw)
