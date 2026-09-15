"""Fiscal period state: exact visible revisions, scale and full-replay contracts."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
from factor_engine.runtime.execution_contract import execution_contract

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frame(values):
    return pd.DataFrame({"A":values},index=pd.date_range("2025-01-01",periods=len(values),tz="Asia/Hong_Kong"))
def run(name,backend,inputs,**params):
    op=OperatorRegistry.get(name,backend,mode="research")
    names=("x","period_id","revision_id") if name=="revision_delta" else ("x","period_id")
    assert op.metadata.panel_params==names
    assert _parameter_contract(op,names)[2]
    assert execution_contract(name).chunking=="required_full_history"
    if backend=="polars":
        inputs=[pl.from_pandas(p.rename_axis("date").reset_index()) for p in inputs]
    result=op.calculate(**dict(zip(names,inputs)),**params)
    return result.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else result

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_visible_revision_ratio_scale_and_prefix(backend):
    inputs=[frame([1.,2.,1.5,3.,4.]),frame(["2024Q1","2024Q2","2024Q1","2024Q2","2024Q3"]),frame([1,1,2,2,1])]
    got=run("revision_delta",backend,inputs)
    np.testing.assert_allclose(got.A,[np.nan,np.nan,.5,1.,np.nan],equal_nan=True)
    ratio=run("revision_delta",backend,inputs,mode="ratio")
    np.testing.assert_allclose(ratio.A,[np.nan,np.nan,.5,.5,np.nan],equal_nan=True)
    scaled=[inputs[0]*1e-200,*inputs[1:]]
    np.testing.assert_allclose(run("revision_delta",backend,scaled,mode="ratio"),ratio,equal_nan=True)
    np.testing.assert_allclose(run("revision_delta",backend,[p.iloc[:3] for p in inputs]),got.iloc[:3],equal_nan=True)
    with pytest.raises((ValueError,TypeError)):
        run("revision_delta",backend,inputs,mode="anything")

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_exact_period_stability_independent_and_scale(backend):
    periods=frame(["2023Q1","2023Q2","2023Q3","2023Q4","2024Q1","2024Q2","2024Q3","2024Q4"])
    x=frame([1.,2.,4.,7.,8.,10.,12.,16.])
    assert run("period_stability",backend,[x,periods]).iloc[-1,0]==pytest.approx(np.median(abs(x.A-np.median(x.A))))
    expected=np.std(x.A,ddof=1)/abs(np.mean(x.A))
    for scale in (1.,1e-200,1e200):
        assert run("period_stability",backend,[x*scale,periods],method="cv").iloc[-1,0]==pytest.approx(expected)
    missing=periods.copy();missing.iloc[3]="2022Q4"
    assert np.isnan(run("period_stability",backend,[x,missing]).iloc[-1,0])
    assert np.isfinite(run("period_stability",backend,[x,missing],require_consecutive=False).iloc[-1,0])
    for kwargs in (dict(periods=1),dict(periods=True),dict(periods=2.5),dict(require_consecutive=1)):
        with pytest.raises((ValueError,TypeError)):
            run("period_stability",backend,[x,periods],**kwargs)
    with pytest.raises((ValueError,TypeError)):
        run("period_stability",backend,[x,periods.rename(columns={"A":"B"})])
