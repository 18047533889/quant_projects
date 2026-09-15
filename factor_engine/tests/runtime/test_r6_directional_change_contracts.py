"""Final directional-change event-table oracle and stable scale/clock semantics."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
from factor_engine.cleaned_operators.common.directional_change_polars import NAMES

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def panels(relative=False):
    values=np.array([100.,103.,106.,102.,99.,97.,101.,105.,108.,104.,100.,96.,100.])
    idx=pd.date_range("2025-01-01",periods=values.size)
    x=pd.DataFrame({"A":np.exp(values/100) if relative else values},index=idx)
    return x,pd.DataFrame(.02 if relative else 2.,index=idx,columns=["A"])

def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if isinstance(x,pd.DataFrame) else x

def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

def run(name,backend,kw):
    return result(OperatorRegistry.get(name,backend,mode="research").calculate(
        **{k:convert(v) if backend=="polars" else v for k,v in kw.items()}))

@pytest.mark.parametrize("name",sorted(NAMES))
def test_dc_final_event_table(name):
    # Confirmations at 1,3,6,9,12; completed durations 2,3,3,3;
    # overshoots 3,5,7,8 in absolute price units.
    expected={"ts_dc_event_rate":5/13,"ts_dc_overshoot_ratio":4.,
              "ts_dc_duration_asymmetry":-1/11,"ts_dc_overshoot_asymmetry":-3/23}[name]
    for relative in (False,True):
        x,scale=panels(relative)
        for mode in ("adaptive","fixed_absolute"):
            kw=dict(x=x,scale=scale,threshold=1.,window=20,threshold_mode=mode,
                    scale_mode="relative" if relative else "absolute")
            reference=run(name,"pandas_numpy",kw)
            assert reference.iloc[-1,0]==pytest.approx(expected,abs=1e-11)
            for backend in ("pandas_numpy","polars"):
                op=OperatorRegistry.get(name,backend,mode="research")
                assert _parameter_contract(op,("x","scale"))[2]
                pd.testing.assert_frame_equal(run(name,backend,kw),reference,check_freq=False,rtol=1e-10,atol=1e-10)
                args=[convert(v) if backend=="polars" else v for v in kw.values()]
                pd.testing.assert_frame_equal(result(op.calculate(*args)),reference,check_freq=False,rtol=1e-10,atol=1e-10)
                short={k:v.iloc[:10] if isinstance(v,pd.DataFrame) else v for k,v in kw.items()}
                pd.testing.assert_frame_equal(run(name,backend,short),reference.iloc[:10],check_freq=False,rtol=1e-10,atol=1e-10)
                with pytest.raises((ValueError,TypeError)):
                    run(name,backend,{**kw,"threshold":np.nan})

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_dc_absolute_scale_invariance_and_window_membership(backend):
    x,scale=panels()
    for name in NAMES:
        ref=run(name,backend,dict(x=x,scale=scale))
        for multiplier in (1e-200,1e200):
            got=run(name,backend,dict(x=x*multiplier,scale=scale*multiplier))
            np.testing.assert_allclose(got,ref,equal_nan=True,atol=1e-10,rtol=1e-10)
    last={"ts_dc_event_rate":2/7,"ts_dc_overshoot_ratio":4.,"ts_dc_duration_asymmetry":0.,"ts_dc_overshoot_asymmetry":-1/15}
    # Window [6,12] contains confirmations 6,9,12 (three events), not two.
    last["ts_dc_event_rate"]=3/7
    for name,value in last.items():
        got=run(name,backend,dict(x=x,scale=scale,window=7))
        assert got.iloc[-1,0]==pytest.approx(value,abs=1e-11)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_dc_atomic_price_validation_and_nonrepresentable_threshold(backend):
    x,scale=panels()
    invalid=x.copy()
    invalid.iloc[-1]=0.
    for name in NAMES:
        with pytest.raises(ValueError,match="positive price"):
            run(name,backend,dict(x=invalid,scale=scale))
        huge=run(name,backend,dict(x=x,scale=scale*1e200,threshold=1e200))
        assert huge.isna().all().all()

def test_dc_native_has_no_pandas_panel_conversion(monkeypatch):
    x,scale=panels()
    inputs=(convert(x),convert(scale))
    def forbidden(*args,**kwargs):
        raise AssertionError("unexpected Pandas-panel conversion")
    for name in NAMES:
        op=OperatorRegistry.get(name,"polars",mode="research")
        with monkeypatch.context() as guard:
            guard.setattr(pl.DataFrame,"to_pandas",forbidden)
            guard.setattr(pd.DataFrame,"__init__",forbidden)
            got=op.calculate(*inputs)
        assert isinstance(got,pl.DataFrame)

def test_dc_zero_median_asymmetry():
    from factor_engine.cleaned_operators.directional_change import _median_asym
    assert _median_asym([0.,0.,1.],[0.,0.,2.])==0.
