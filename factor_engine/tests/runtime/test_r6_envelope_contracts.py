"""Final registered envelope backends must consume supplied bands identically."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

NAMES=("ts_envelope_compression","ts_envelope_pressure","ts_envelope_boundary_dwell")
def panels(name):
    lower=pd.DataFrame({"A":[0.]*12, "B":[1.]*12})
    upper=pd.DataFrame({"A":[10.,8.,6.,4.,8.,2.,7.,5.,6.,4.,3.,8.],
                        "B":[9.,8.,7.,6.,5.,4.,6.,7.,8.,9.,6.,5.]})
    mid=(upper+lower)/2
    x=lower+(upper-lower)*pd.DataFrame({"A":np.linspace(0,1,12),"B":np.linspace(1,0,12)})
    return {"upper":upper,"lower":lower,"mid":mid} if name.endswith("compression") else {"x":x,"upper":upper,"lower":lower}

@pytest.mark.parametrize("name",NAMES)
@pytest.mark.parametrize("missing",[False,True])
def test_final_envelope_native_parity_bands_keywords_and_prefix(name,missing,monkeypatch):
    inputs=panels(name)
    if missing:
        for panel in inputs.values():
            panel.iloc[3,0]=np.nan
        inputs["upper"].iloc[5,0]=np.inf
        inputs["lower"].iloc[7,1]=inputs["upper"].iloc[7,1]+1
    pandas_op=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    polars_op=OperatorRegistry.get(name,"polars",mode="research")
    params={"window":3}
    if name.endswith("dwell"):
        params["quantile"]=.8
    expected=pandas_op.calculate(**inputs,**params)
    native={k:pl.from_pandas(v) for k,v in inputs.items()}
    def forbidden(*args,**kwargs):
        raise AssertionError("Envelope native kernel must not bridge to pandas/numpy")
    with monkeypatch.context() as m:
        m.setattr(pl.DataFrame,"to_pandas",forbidden)
        m.setattr(pl.DataFrame,"to_numpy",forbidden)
        m.setattr(pl.Series,"to_numpy",forbidden)
        actual=polars_op.calculate(**native,**params)
    pd.testing.assert_frame_equal(actual.to_pandas(),expected,atol=1e-12,rtol=1e-12)
    positional=polars_op.calculate(*native.values(),*params.values()).to_pandas()
    pd.testing.assert_frame_equal(positional,expected,atol=1e-12,rtol=1e-12)
    prefix=polars_op.calculate(**{k:v[:8] for k,v in native.items()},**params).to_pandas()
    pd.testing.assert_frame_equal(prefix,expected.iloc[:8],atol=1e-12,rtol=1e-12)
    assert np.isfinite(actual.to_numpy()).any()
    with pytest.raises((ValueError,TypeError)):
        polars_op.calculate(**native,window=1)

def test_envelope_independent_pressure_dwell_and_compression():
    x=pd.DataFrame({"A":[0.,2.,4.]})
    upper=x*0+10
    lower=x*0
    for backend in ("pandas_numpy","polars"):
        convert=pl.from_pandas if backend=="polars" else lambda x:x
        def value(name,args):
            op=OperatorRegistry.get(name,backend,mode="research")
            result=op.calculate(*(convert(a) for a in args),window=3)
            return result.to_pandas().iat[-1,0] if backend=="polars" else result.iat[-1,0]
        assert value("ts_envelope_pressure",[x,upper,lower]) == pytest.approx((-1-1.2-.6)/6)
        assert value("ts_envelope_boundary_dwell",[x,upper,lower]) == pytest.approx(1/3)
        u=pd.DataFrame({"A":[10.,8.,6.]})
        assert value("ts_envelope_compression",[u,lower,upper/2]) == pytest.approx(2/3)
