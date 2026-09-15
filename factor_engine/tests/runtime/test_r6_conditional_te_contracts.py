"""Conditional transfer entropy: real final backend, independent smoothed counts."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAME="ts_conditional_transfer_entropy"
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def panels():
    rng=np.random.default_rng(144)
    s=rng.integers(0,2,120).astype(float)
    c=rng.integers(0,2,120).astype(float)
    t=np.r_[0.,s[:-1]]
    index=pd.date_range("2025-01-01",periods=len(s),tz="Asia/Hong_Kong")
    return tuple(pd.DataFrame({"A":v},index=index) for v in (t,s,c))

def oracle(t,s,c):
    # Independent binary-state conditional entropy difference, Jeffreys=.5.
    past, future, source, condition = t[:-1],t[1:],s[:-1],c[:-1]
    mask=np.isfinite(past)&np.isfinite(future)&np.isfinite(source)&np.isfinite(condition)
    counts=np.full((2,2,2,2),.5)
    for nxt,old,src,cond in zip(future[mask],past[mask],source[mask],condition[mask]):
        counts[int(nxt),int(old),int(src),int(cond)]+=1
    p=counts/counts.sum()
    def entropy(v):
        return -np.sum(v[v>0]*np.log(v[v>0]))
    return entropy(p.sum(axis=2))+entropy(p.sum(axis=0))-entropy(p.sum(axis=(0,2)))-entropy(p)

def run(backend,inputs,**kwargs):
    op=OperatorRegistry.get(NAME,backend,mode="research")
    assert op is not None
    assert op.metadata.panel_params==("target","source","condition")
    assert _parameter_contract(op,op.metadata.panel_params)[2]
    args=dict(zip(op.metadata.panel_params,inputs))
    if backend=="polars":
        args={k:pl.from_pandas(v.rename_axis("date").reset_index()) for k,v in args.items()}
    out=op.calculate(**args,**kwargs)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_default_real_te_oracle_and_scale(backend):
    frames=panels()
    out=run(backend,frames)
    expected=oracle(*(v.A.to_numpy()[-60:] for v in frames))
    assert out.A.iloc[-1] == pytest.approx(expected,abs=1e-12)
    assert expected>.1
    assert out.iloc[:48].isna().all().all()
    np.testing.assert_allclose(run(backend,frames,window=60,bins=2,lag=1),out,equal_nan=True)
    np.testing.assert_allclose(run(backend,tuple(v.iloc[:90] for v in frames)),out.iloc[:90],equal_nan=True)
    for factor in (1e-200,1e200):
        np.testing.assert_allclose(run(backend,tuple((v*2-1)*factor for v in frames)),out,
                                   atol=1e-12,equal_nan=True)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_missing_transition_physical_lag_and_infeasible_domains(backend):
    frames=panels()
    frames[0].iloc[-10]=np.nan
    out=run(backend,frames)
    expected=oracle(*(v.A.to_numpy()[-60:] for v in frames))
    assert out.A.iloc[-1]==pytest.approx(expected,abs=1e-12)
    invalid=[dict(window=20),dict(bins=3),dict(window=120,lag=60),
             dict(window=60.5),dict(lag=1.5),dict(min_transitions=3.5),
             dict(min_transitions=0),dict(min_cells_ratio=-1.),
             dict(min_cells_ratio=np.inf),dict(min_cells_ratio=True)]
    for kwargs in invalid:
        with pytest.raises((ValueError,TypeError)):
            run(backend,frames,**kwargs)
