"""Final numerical acceptance for seven remaining regression-model operators."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.regression_models import _REMAINING_MODEL_NAMES
from factor_engine.runtime.operator_snapshot import _parameter_contract
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x
def frames():
    rng=np.random.default_rng(615)
    x=pd.DataFrame({"A":rng.normal(size=40).cumsum()+10},index=pd.date_range("2025-01-01",periods=40))
    return x,3*x+2
@pytest.mark.parametrize("name",sorted(_REMAINING_MODEL_NAMES))
def test_model_contract_calls_scale_and_causal_prefix(name,monkeypatch):
    x,y=frames()
    values=dict(x=x,y=y,window=20,min_periods=10,q=.5 if "quantile" in name else 3)
    reference=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    kwargs={k:values[k] for k in reference.metadata.param_names}
    expected=reference.calculate(**kwargs)
    assert np.isfinite(expected.to_numpy()).any()
    if "quantile" in name:
        np.testing.assert_allclose(expected.A.dropna(),3.,atol=1e-7)
    panels=tuple(k for k,v in kwargs.items() if isinstance(v,pd.DataFrame))
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        assert op is not None,(name,backend)
        assert _parameter_contract(op,panels)[2],(name,backend)
        inputs={k:convert(v) if backend=="polars" and isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        with monkeypatch.context() as patch:
            if backend=="polars":
                def forbidden(*args,**kwargs):
                    raise AssertionError("Actual native model cannot convert Pandas panels")
                patch.setattr(pl.DataFrame,"to_pandas",forbidden)
            actual=op.calculate(**inputs)
        pd.testing.assert_frame_equal(result(actual),expected,check_freq=False,atol=1e-7,rtol=1e-7)
        pd.testing.assert_frame_equal(result(op.calculate(*inputs.values())),expected,check_freq=False,atol=1e-7,rtol=1e-7)
        short={k:(convert(v.iloc[:32]) if backend=="polars" else v.iloc[:32]) if isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        pd.testing.assert_frame_equal(result(op.calculate(**short)),expected.iloc[:32],check_freq=False,atol=1e-7,rtol=1e-7)
        if "quantile" not in name:
            large={k:(convert(v*1e200) if backend=="polars" else v*1e200) if isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
            pd.testing.assert_frame_equal(result(op.calculate(**large)),expected,check_freq=False,atol=1e-7,rtol=1e-7)
        for bad in (0,True,2.5):
            with pytest.raises((TypeError,ValueError)):
                op.calculate(**{**inputs,"window":bad})
        with pytest.raises((TypeError,ValueError)):
            op.calculate(**{**inputs,"min_periods":21})

@pytest.mark.parametrize("name",["ts_cumulative_deviation_score","ts_level_shift_score","ts_vol_shift_score","ts_lo_mackinlay_vr","ts_lo_mackinlay_z","ts_variance_ratio_proxy"])
def test_last_row_independent_estimator_reference(name):
    x,_=frames()
    op=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    vals=x.A.to_numpy()[-20:]
    kwargs=dict(window=20,min_periods=10)
    if name in ("ts_lo_mackinlay_vr","ts_lo_mackinlay_z","ts_variance_ratio_proxy"):
        kwargs["q"]=3
        rets=np.diff(vals);n=len(rets);q=3
        qrets=np.array([sum(rets[i:i+q]) for i in range(n-q+1)])
        if name=="ts_variance_ratio_proxy":
            expected=np.var(qrets)/(q*np.var(rets))-1
        else:
            mean=rets.mean()
            vr=(sum((qrets-q*mean)**2)/(q*(n-q+1)*(1-q/n)))/(sum((rets-mean)**2)/(n-1))
            if name=="ts_lo_mackinlay_vr":
                expected=vr
            else:
                dev=rets-mean;den=sum(dev**2)
                theta=sum((2*(q-k)/q)**2*sum(dev[k:]**2*dev[:-k]**2)/den**2 for k in range(1,q))
                expected=(vr-1)/np.sqrt(theta)
    elif name=="ts_cumulative_deviation_score":
        expected=np.max(np.abs(np.cumsum((vals-vals.mean())/vals.std())))
    elif name=="ts_level_shift_score":
        expected=(vals[10:].mean()-vals[:10].mean())/vals.std()
    else:
        expected=np.log(vals[10:].std()/vals[:10].std())
    assert op.calculate(x,**kwargs).iloc[-1,0]==pytest.approx(expected,rel=1e-10,abs=1e-10)
