"""Real spread kernels: pure bid/ask oracles, support and extreme prices."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

CS="ohlc_corwin_schultz_spread"
ROLL="ts_roll_effective_spread"
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frame(values):
    return pd.DataFrame({"A":values},index=pd.date_range("2025-01-01",periods=len(values),tz="Asia/Hong_Kong"))
def run(name,backend,panels,**kwargs):
    op=OperatorRegistry.get(name,backend,mode="research")
    names=("high","low") if name==CS else ("price",)
    assert op.metadata.panel_params==names
    assert _parameter_contract(op,names)[2]
    args=dict(zip(names,panels))
    if backend=="polars":
        args={k:pl.from_pandas(p.rename_axis("date").reset_index()) for k,p in args.items()}
    out=op.calculate(**args,**kwargs)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_cs_pure_spread_default_units_gap_and_extreme(backend):
    high,low=frame(np.full(30,102.)),frame(np.full(30,98.))
    result=run(CS,backend,[high,low])
    assert result.iloc[:5].isna().all().all()
    np.testing.assert_allclose(result.iloc[5:],.04,atol=1e-12)
    np.testing.assert_allclose(result,run(CS,backend,[high,low],smooth_window=5),equal_nan=True)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(run(CS,backend,[high*scale,low*scale]),result,equal_nan=True,atol=1e-12)
    # Equal daily high/low ratios give alpha=log(high/low); bounded tanh never overflows.
    extreme=run(CS,backend,[high*0+1e308,low*0+1e-308],smooth_window=1)
    assert extreme.iloc[-1,0]==2.
    high.iloc[10]=np.nan
    gap=run(CS,backend,[high,low],smooth_window=3)
    assert gap.iloc[10:14].isna().all().all()
    np.testing.assert_allclose(run(CS,backend,[high.iloc[:20],low.iloc[:20]],smooth_window=3),gap.iloc[:20],equal_nan=True)
    assert run(CS,backend,[low,low],smooth_window=1).iloc[-1,0]==0.
    assert run(CS,backend,[low-1,low],smooth_window=1).isna().all().all()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_roll_adjacent_covariance_default_scale_and_atomic_price_gate(backend):
    price=frame(100*np.exp(np.tile([-.01,.01],20)))
    result=run(ROLL,backend,[price],window=10,min_periods=4)
    d=np.diff(np.log(price.A.to_numpy()))[-10:]
    expected=2*np.sqrt(-np.cov(d[1:],d[:-1],ddof=1)[0,1])
    assert result.iloc[-1,0]==pytest.approx(expected,abs=1e-12)
    np.testing.assert_allclose(run(ROLL,backend,[price]),run(ROLL,backend,[price],window=20,min_periods=10),equal_nan=True)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(run(ROLL,backend,[price*scale],window=10,min_periods=4),result,equal_nan=True,atol=1e-12)
    np.testing.assert_allclose(run(ROLL,backend,[price.iloc[:25]],window=10,min_periods=4),result.iloc[:25],equal_nan=True)
    invalid=price.copy();invalid.iloc[-1]=0
    assert run(ROLL,backend,[invalid],window=10).isna().all().all()
    # Surface authoring promotion is not a numerical production certificate.
    # The existing DirectUse table can classify this research estimator as
    # extended; this repair must not invent production evidence for its kernel.
    if backend=="polars":
        from factor_engine.backend.polars_backend_kind import get_physical_spec
        assert not get_physical_spec(OperatorRegistry.get(ROLL,backend,mode="research")).is_production_eligible()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_bad_controls_and_axes(backend):
    x=frame(np.arange(30.)+100.)
    for name,key,bad in ((CS,"smooth_window",True),(CS,"smooth_window",1.5),
                         (ROLL,"window",5.5),(ROLL,"min_periods",True),(ROLL,"min_periods",20)):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,[x,x] if name==CS else [x],**{key:bad})
    shifted=x.copy();shifted.index+=pd.Timedelta(days=1)
    with pytest.raises((ValueError,TypeError)):
        run(CS,backend,[x,shifted])

@pytest.mark.parametrize("name",[CS,ROLL])
def test_no_pandas_panel_conversion(name,monkeypatch):
    x=pl.DataFrame({"A":100*np.exp(np.tile([-.01,.01],20))})
    panels=[x*1.02,x*.98] if name==CS else [x]
    op=OperatorRegistry.get(name,"polars",mode="research")
    def forbidden(*a,**kw):
        raise AssertionError("native spread must not convert pandas panels")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    assert np.isfinite(op.calculate(*panels)["A"][-1])
