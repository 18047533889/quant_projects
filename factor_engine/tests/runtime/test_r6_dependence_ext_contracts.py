"""Independent dependence oracles and true final Registry backend parity."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import dependence_ext as source
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_chatterjee_xi","ts_hsic","ts_conditional_mutual_information",
       "ts_distance_correlation_partial_proxy")
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def frames():
    rng=np.random.default_rng(85)
    idx=pd.date_range("2025-01-01",periods=72,tz="Asia/Hong_Kong")
    return tuple(pd.DataFrame(rng.normal(size=(72,2)),index=idx,columns=["A","B"])
                 for _ in range(3))

def run(name,backend,inputs,**kwargs):
    op=OperatorRegistry.get(name,backend,mode="research")
    panels=("x","y","z") if name in NAMES[2:] else ("x","y")
    assert op.metadata.panel_params==panels
    assert _parameter_contract(op,panels)[2]
    args=dict(zip(panels,inputs))
    if backend=="polars":
        args={k:pl.from_pandas(v.rename_axis("date").reset_index()) for k,v in args.items()}
    out=op.calculate(**args,**kwargs)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out

def reference_hsic(x,y):
    def kernel(v):
        d=np.abs(v[:,None]-v[None,:])
        positive=d[np.triu_indices(len(v),1)]
        sigma=np.median(positive[positive>0])
        return np.exp(-d*d/(2*sigma*sigma))
    n=len(x);H=np.eye(n)-np.ones((n,n))/n
    K,L=kernel(x),kernel(y)
    return np.trace(K@H@L@H)/np.sqrt(np.trace(K@H@K@H)*np.trace(L@H@L@H))

def reference_cmi(x,y,z,bins=3):
    from collections import Counter
    def labels(v):
        cuts=np.quantile(v,np.arange(1,bins)/bins)
        return np.searchsorted(cuts,v,side="right")
    a,b,c=map(labels,(x,y,z));n=len(a)
    abc=Counter(zip(a,b,c));ac=Counter(zip(a,c));bc=Counter(zip(b,c));cc=Counter(c)
    information=sum(count/n*np.log(count*cc[k]/(ac[i,k]*bc[j,k]))
                    for (i,j,k),count in abc.items())
    return information/np.log(min(len(set(a)),len(set(b))))

def reference_proxy(x,y,z):
    def distance(v):
        d=np.abs(v[:,None]-v[None,:])
        return d-d.mean(axis=0)[None,:]-d.mean(axis=1)[:,None]+d.mean()
    a,b,c=map(distance,(x,y,z))
    def corr(a,b):
        return np.sqrt(max(np.mean(a*b),0)/np.sqrt(np.mean(a*a)*np.mean(b*b)))
    xy,xz,yz=corr(a,b),corr(a,c),corr(b,c)
    return (xy-xz*yz)/np.sqrt((1-xz*xz)*(1-yz*yz))

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_real_inputs_oracles_defaults_scale_and_prefix(name,backend):
    inputs=frames();x,y,z=(v.A.to_numpy()[-64:] for v in inputs)
    out=run(name,backend,inputs,window=64)
    if name==NAMES[0]:
        order=sorted(range(len(x)),key=lambda i:x[i])
        ranks=np.array([sum(v<=y[i] for v in y) for i in order])
        upper=np.array([sum(v>=yi for v in y) for yi in y])
        expected=1-len(y)*np.abs(np.diff(ranks)).sum()/(2*np.sum(upper*(len(y)-upper)))
    elif name==NAMES[1]:
        expected=reference_hsic(x,y)
    elif name==NAMES[2]:
        expected=reference_cmi(x,y,z)
    else:
        expected=reference_proxy(x,y,z)
    assert out.A.iloc[-1]==pytest.approx(expected,abs=1e-11)
    np.testing.assert_allclose(run(name,backend,tuple(v.iloc[:60] for v in inputs),window=64),
                               out.iloc[:60],atol=1e-11,equal_nan=True)
    np.testing.assert_allclose(run(name,backend,inputs),
                               run(name,backend,inputs,window=60 if name==NAMES[1] else 120),
                               atol=1e-11,equal_nan=True)
    for factor in (1e-200,1e200):
        got=run(name,backend,tuple(v*factor for v in inputs),window=64)
        np.testing.assert_allclose(got,out,atol=1e-10,rtol=1e-10,equal_nan=True)

def test_hsic_exact_dense_oracle_with_ties_and_no_dense_multiplication(monkeypatch):
    x=np.array([0.,0,0,0,1,2,3,5])
    y=np.array([1.,2,1,2,3,5,7,11])
    expected=reference_hsic(x,y)
    def forbidden(*a,**kw):
        raise AssertionError("HSIC must use quadratic mean-centering, not explicit H")
    monkeypatch.setattr(np,"eye",forbidden)
    for factor in (1.,1e-200,1e200):
        assert source._hsic(x*factor,y*factor)==pytest.approx(expected,abs=1e-12)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_cmi_support_strict_bins_and_unknown_inputs(backend):
    inputs=frames()
    out=run(NAMES[2],backend,inputs,window=64,bins=3)
    assert out.iloc[:53].isna().all().all()
    assert out.iloc[53:].notna().all().all()
    # Numeric enum choices canonicalize 3.0 to declared int 3 at the shared gate.
    np.testing.assert_allclose(run(NAMES[2],backend,inputs,window=64,bins=3.0),out,
                               atol=1e-12,equal_nan=True)
    with pytest.raises((ValueError,TypeError)):
        source.TsConditionalMutualInformation()._calculate_series(*inputs,bins=3.0)
    for bad in (True,2.5,4):
        with pytest.raises((ValueError,TypeError)):
            run(NAMES[2],backend,inputs,bins=bad)
    bad_inputs=tuple(v*np.nan for v in inputs)
    for name in NAMES:
        assert run(name,backend,bad_inputs,window=64).isna().all().all()
