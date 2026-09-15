"""Memory kernels: independent finite sums and ACF paired-sequence oracle."""
import math
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.memory_ext import _fd_discarded_weight_mass
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def panel():
    return pd.DataFrame({"A":np.sin(np.arange(48.)*.41)+np.arange(48.)*.02},
                        index=pd.date_range("2025-01-01",periods=48))
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if isinstance(x,pd.DataFrame) else x
def run(name,backend,x,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None
    assert _parameter_contract(op,("x",))[2]
    out=op.calculate(x=convert(x) if backend=="polars" else x,**kw)
    return out.to_pandas().set_index("date").rename_axis(None) if isinstance(out,pl.DataFrame) else out

def acf_oracle(x,lag,monotone):
    z=x-x.mean()
    rho=np.array([np.dot(z[:len(z)-k],z[k:])/(len(z)-k)/(z@z/len(z)) for k in range(lag+1)])
    pairs=rho[:len(rho)//2*2].reshape(-1,2).sum(axis=1)
    stop=np.flatnonzero(pairs<=0)
    pairs=pairs[:stop[0]] if len(stop) else pairs
    return -1+2*(np.minimum.accumulate(pairs) if monotone else pairs).sum()

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",["ts_autocorrelation_time","ts_autocorrelation_time_initial_positive_sequence"])
def test_acf_pair_oracle_gap_scale(name,backend):
    x=panel()
    got=run(name,backend,x,window=20,max_lag=7)
    expected=acf_oracle(x.A.to_numpy()[-20:],7,name=="ts_autocorrelation_time")
    assert got.iloc[-1,0]==pytest.approx(expected,abs=1e-12)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(run(name,backend,x*scale,window=20,max_lag=7),got,atol=1e-12,equal_nan=True)
    short=run(name,backend,x.iloc[:35],window=20,max_lag=7)
    np.testing.assert_allclose(short,got.iloc[:35],equal_nan=True)
    x.iloc[-6]=np.nan
    gap=run(name,backend,x,window=20,max_lag=3)
    assert gap.iloc[-6,0]!=gap.iloc[-6,0]
    assert gap.iloc[-1,0]==pytest.approx(acf_oracle(x.A.to_numpy()[-5:],3,name=="ts_autocorrelation_time"))
    for invalid in (True,2.5,0,np.nan):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,x,window=20,max_lag=invalid)
    with pytest.raises((ValueError,TypeError)):
        run(name,backend,x,window=20,max_lag=20)
    np.testing.assert_allclose(run(name,backend,panel()),run(name,"pandas_numpy",panel()),equal_nan=True)

@pytest.mark.parametrize("d,c",[(.4,20),(.1,1),(.8,100),(.4,60001)])
def test_exact_discarded_weight_mass(d,c):
    # Generalized-binomial partial-sum identity, evaluated independently by gamma.
    oracle=.5*math.exp(math.lgamma(c+1-d)-math.lgamma(1-d)-math.lgamma(c+1))
    assert _fd_discarded_weight_mass(d,c)==pytest.approx(oracle,rel=2e-10)
    assert _fd_discarded_weight_mass(d,c,_tail_terms=2)==_fd_discarded_weight_mass(d,c)
    assert _fd_discarded_weight_mass(-d,c)==1
    assert _fd_discarded_weight_mass(0,c)==0

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_fractional_convolution_and_gate(backend):
    x=panel()
    d=.4;c=7
    w=[1.]
    for k in range(1,c+1):
        w.append(w[-1]*(k-1-d)/k)
    oracle=np.full(x.shape,np.nan)
    oracle[c:,0]=np.convolve(x.A,w,mode="valid")
    got=run("ts_fractional_difference",backend,x,fd=d,cutoff=c)
    np.testing.assert_allclose(got,oracle,atol=1e-14,equal_nan=True)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(run("ts_fractional_difference",backend,x*scale,fd=d,cutoff=c)/scale,oracle,atol=1e-14,equal_nan=True)
    mass=run("ts_fractional_difference_discarded_weight_mass",backend,x,fd=d,cutoff=c)
    np.testing.assert_allclose(mass,_fd_discarded_weight_mass(d,c))
    assert run("ts_fractional_difference",backend,x,fd=d,cutoff=c,max_discarded_weight_mass=.001).isna().all().all()
    assert run("ts_fractional_difference",backend,x,fd=-d,cutoff=c).isna().all().all()
    x.iloc[20]=np.nan
    out=run("ts_fractional_difference",backend,x,fd=d,cutoff=c)
    assert out.iloc[20:28].isna().all().all()
    assert np.isfinite(out.iloc[28,0])
    for name in ("ts_fractional_difference","ts_fractional_difference_discarded_weight_mass"):
        for invalid in (True,2.5,0,np.nan):
            with pytest.raises((ValueError,TypeError)):
                run(name,backend,x,cutoff=invalid)
        np.testing.assert_allclose(run(name,backend,x),run(name,"pandas_numpy",x),equal_nan=True)

def test_fractional_representable_cancellation():
    from factor_engine.cleaned_operators.memory_ext import _fractional_difference_series
    x=np.array([[-1e308],[-1e308],[1e308]])
    # Integration weights positive; sum is finite after cancellation.
    got=_fractional_difference_series(x,-.9,2)[-1,0]
    assert np.isfinite(got)
    assert got==pytest.approx(-.755e308)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_unreachable_filter_support_does_not_allocate_weights(backend,monkeypatch):
    from factor_engine.cleaned_operators import memory_ext
    def forbidden(*args,**kwargs):
        raise AssertionError("unreachable full support must not allocate weights or scan a tail")
    monkeypatch.setattr(memory_ext,"_fd_weights",forbidden)
    monkeypatch.setattr(memory_ext,"_fd_discarded_weight_mass",forbidden)
    assert run("ts_fractional_difference",backend,panel(),cutoff=10**12).isna().all().all()

def test_declared_memory_history():
    from factor_engine.runtime.execution_contract import _declared_history_extension
    assert _declared_history_extension("ts_autocorrelation_time",{"window":120,"max_lag":20})==119
    assert _declared_history_extension("ts_fractional_difference",{"fd":.4,"cutoff":20})==20
    assert _declared_history_extension("ts_fractional_difference_discarded_weight_mass",{"fd":.4,"cutoff":20})==0
