"""Real panel contracts and both backend oracles for event/state alpha transforms."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
NAMES=("event_rate_pct","event_recency_z","event_cluster_score","state_dwell_pct","state_transition_surprise")

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def panels(name):
    idx=pd.date_range("2025-01-01",periods=16)
    if name.startswith("event"):
        values=np.zeros(16)
        values[[0,2,5,6,9,12,14]]=1.
        return dict(event=pd.DataFrame({"A":values},index=idx),window=4,rate_window=8) if name=="event_cluster_score" else dict(event=pd.DataFrame({"A":values},index=idx),window=8)
    return dict(state=pd.DataFrame({"A":[0.,0.,1.,1.,1.,2.,2.,0.,0.,1.,2.,2.,1.,1.,1.,0.]},index=idx),window=8)

def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if isinstance(x,pd.DataFrame) else x

def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

def run(name,backend,kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    return result(op.calculate(**{k:convert(v) if backend=="polars" else v for k,v in kw.items()}))

@pytest.mark.parametrize("name",NAMES)
def test_event_state_oracle_and_contract(name):
    kw=panels(name)
    field=next(iter(kw))
    expected={"event_rate_pct":3/8,"event_recency_z":-1.5/np.sqrt(.5),
        "event_cluster_score":.5/np.sqrt(1.5),"state_dwell_pct":.25,
        "state_transition_surprise":-2/np.sqrt(14)}[name]
    ref=run(name,"pandas_numpy",kw)
    assert ref.iloc[-1,0]==pytest.approx(expected)
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        schema,_,verified=_parameter_contract(op,(field,))
        assert verified
        assert "window" in schema["required"]
        pd.testing.assert_frame_equal(run(name,backend,kw),ref,check_freq=False,rtol=1e-10,atol=1e-10)
        args=[convert(v) if backend=="polars" else v for v in kw.values()]
        pd.testing.assert_frame_equal(result(op.calculate(*args)),ref,check_freq=False,rtol=1e-10,atol=1e-10)
        short={**kw,field:kw[field].iloc[:12]}
        pd.testing.assert_frame_equal(run(name,backend,short),ref.iloc[:12],check_freq=False,rtol=1e-10,atol=1e-10)
        missing=kw[field].copy()
        missing.iloc[10]=np.nan
        unknown=run(name,backend,{**kw,field:missing})
        assert unknown.iloc[-1].isna().all()
        bad=kw[field].copy()
        bad.iloc[-1]=np.inf
        with pytest.raises(ValueError):
            run(name,backend,{**kw,field:bad})
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,{**kw,"window":1})

def test_event_state_native_no_pandas_conversion(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("unexpected Pandas-panel conversion")
    for name in NAMES:
        kw={k:convert(v) for k,v in panels(name).items()}
        op=OperatorRegistry.get(name,"polars",mode="research")
        with monkeypatch.context() as guard:
            guard.setattr(pl.DataFrame,"to_pandas",forbidden)
            guard.setattr(pd.DataFrame,"__init__",forbidden)
            out=op.calculate(**kw)
        assert isinstance(out,pl.DataFrame)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_state_identity_comparison_without_overflow(backend):
    kw=panels("state_transition_surprise")
    kw["state"]=kw["state"].replace({0.:-1e308,1.:0.,2.:1e308})
    with np.errstate(over="raise",invalid="raise"):
        got=run("state_transition_surprise",backend,kw)
    assert got.iloc[-1,0]==pytest.approx(-2/np.sqrt(14))
