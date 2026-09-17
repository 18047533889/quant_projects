"""Generic signal entropy is a separate research statistic, not a typed-gate bypass."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

def test_signal_entropy_reference_prefix_and_missingness():
    load_all()
    rng=np.random.default_rng(2020)
    x=pd.DataFrame({"A":rng.normal(size=80)})
    op=OperatorRegistry.get("ts_signal_spectral_entropy","pandas_numpy",mode="any")
    actual=op.calculate(x,window=20).to_numpy()[:,0]
    t=np.arange(20,dtype=float);v=x["A"].to_numpy()[-20:]
    slope,intercept=np.polyfit(t,v,1)
    spectrum=np.fft.rfft((v-slope*t-intercept)*np.hanning(20))
    power=np.abs(spectrum[1:])**2/20
    p=power/power.sum()
    expected=-(p[p>0]*np.log(p[p>0])).sum()/np.log(10)
    assert actual[-1]==pytest.approx(expected,rel=1e-12)
    assert np.isnan(actual[:19]).all()
    np.testing.assert_allclose(op.calculate(x.iloc[:60],window=20).to_numpy()[:,0],actual[:60],equal_nan=True)
    missing=x.copy();missing.iloc[65,0]=np.nan
    assert np.isnan(op.calculate(missing,window=20).to_numpy()[65:,0]).all()

def test_signal_entropy_polars_exact_parity():
    pl=pytest.importorskip("polars");load_all()
    x=pd.DataFrame({"A":np.random.default_rng(9).normal(size=50)})
    expected=OperatorRegistry.get("ts_signal_spectral_entropy","pandas_numpy",mode="any").calculate(x,window=20)
    actual=OperatorRegistry.get("ts_signal_spectral_entropy","polars",mode="any").calculate(pl.from_pandas(x),window=20)
    np.testing.assert_allclose(actual.to_numpy(),expected.to_numpy(),equal_nan=True)

def test_return_entropy_rejects_activity_type():
    from factor_engine.cleaned_operators.spectral_ext import _ts_return_spectral_entropy
    with pytest.raises(ValueError):
        _ts_return_spectral_entropy(pd.DataFrame({"A":np.ones(40)}),20,input_kind="NonNegativeActivity")
