"""Final-registry structure numerical checks, including independent KM/Bures oracles."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
from factor_engine.cleaned_operators.advanced_structure import _STRUCTURE_TARGETS

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def panel_inputs():
    rng=np.random.default_rng(291)
    ix=pd.date_range("2025-01-01",periods=90)
    return [pd.DataFrame(rng.normal(size=(90,8)),index=ix,columns=list("ABCDEFGH")) for _ in range(3)]

def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index()) if isinstance(x,pd.DataFrame) else x

def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

def run(name,backend,kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    return result(op.calculate(**{k:convert(v) if backend=="polars" else v for k,v in kw.items()}))

@pytest.mark.parametrize("name",sorted(_STRUCTURE_TARGETS))
def test_final_structure_bindings(name):
    a,b,c=panel_inputs()
    values=dict(x=a,y=b,f1=a,f2=b,f3=c,group=a*0+1,recent_window=10,prior_window=20,
        min_pairs=5,window=20,bins=2,lag=1,min_bin_count=2,directions=8,
        reference_window=10,min_peers=5,min_reference_days=3,composition_policy="current")
    ref=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    kw={k:values[k] for k in ref.metadata.param_names}
    expected=run(name,"pandas_numpy",kw)
    assert np.isfinite(expected.to_numpy()).any()
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        assert _parameter_contract(op,tuple(k for k,v in kw.items() if isinstance(v,pd.DataFrame)))[2]
        pd.testing.assert_frame_equal(run(name,backend,kw),expected,check_freq=False,rtol=1e-9,atol=1e-9)
        args=[convert(v) if backend=="polars" else v for v in kw.values()]
        pd.testing.assert_frame_equal(result(op.calculate(*args)),expected,check_freq=False,rtol=1e-9,atol=1e-9)
        short={k:v.iloc[:70] if isinstance(v,pd.DataFrame) else v for k,v in kw.items()}
        pd.testing.assert_frame_equal(run(name,backend,short),expected.iloc[:70],check_freq=False,rtol=1e-9,atol=1e-9)
        default_panels={k:v for k,v in kw.items() if isinstance(v,pd.DataFrame)}
        pd.testing.assert_frame_equal(run(name,backend,default_panels),run(name,"pandas_numpy",default_panels),check_freq=False,rtol=1e-9,atol=1e-9)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_km_linear_independent_oracle(backend):
    x=pd.DataFrame({"A":np.arange(70.)},index=pd.date_range("2025-01-01",periods=70))
    for name,expected in (("ts_kramers_moyal_drift",1.),("ts_kramers_moyal_diffusion",.5)):
        got=run(name,backend,dict(x=x,window=20,bins=2,lag=1,min_bin_count=2))
        # The shared state kernel has expanding max-lookback and 10 finite-history maturity.
        assert got.iloc[:10].isna().all().all()
        np.testing.assert_allclose(got.iloc[10:],expected)
        gated=run(name,backend,dict(x=x,window=20,bins=2,lag=1,min_bin_count=10))
        assert gated.iloc[20:].isna().all().all()
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,dict(x=x,window=20,bins=2,lag=20))

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_bures_independent_oracle_and_scale(backend):
    a,b,_=panel_inputs()
    kw=dict(x=a,y=b,recent_window=10,prior_window=20,min_pairs=5)
    got=run("ts_bures_corr_shift",backend,kw)
    r1=np.corrcoef(a.iloc[-10:,0],b.iloc[-10:,0])[0,1]*.95
    r0=np.corrcoef(a.iloc[-30:-10,0],b.iloc[-30:-10,0])[0,1]*.95
    expected=np.sqrt(max(0,4-2*(np.sqrt((1+r1)*(1+r0))+np.sqrt((1-r1)*(1-r0)))))
    assert got.iloc[-1,0]==pytest.approx(expected,abs=1e-10)
    for scale in (1e-200,1e200):
        scaled=run("ts_bures_corr_shift",backend,{**kw,"x":a*scale,"y":b*scale})
        np.testing.assert_allclose(scaled,got,equal_nan=True,atol=2e-8,rtol=2e-8)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_group_scale_and_invalid_labels(backend):
    a,b,c=panel_inputs()
    group=a*0+1
    kw=dict(f1=a,f2=b,f3=c,group=group,reference_window=10,min_reference_days=3,min_peers=5)
    got=run("group_spd_feature_structure_shift",backend,kw)
    for scale in (1e-200,1e200):
        changed=run("group_spd_feature_structure_shift",backend,{**kw,"f1":a*scale,"f2":b*scale,"f3":c*scale})
        np.testing.assert_allclose(changed,got,equal_nan=True,atol=1e-9,rtol=1e-9)
    for label in (np.nan,np.inf,""):
        invalid=group.copy().astype(object)
        invalid.iloc[:,:]=label
        out=run("group_spd_feature_structure_shift",backend,{**kw,"group":invalid})
        assert out.isna().all().all()
