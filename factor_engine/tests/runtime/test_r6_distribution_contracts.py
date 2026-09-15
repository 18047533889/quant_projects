"""Independent clipped-U energy and copula oracles through final default backends."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.spatial.distance import cdist,pdist
from scipy.stats import rankdata
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

SHIFT="ts_joint_energy_shift";SCORE="ts_energy_break_score";COPULA="ts_copula_central_asymmetry"
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frames(n=105):
    rng=np.random.default_rng(716)
    raw=rng.normal(size=(n,3))+np.arange(n)[:,None]*np.array([.08,.03,-.04])
    idx=pd.date_range("2025-01-01",periods=n,tz="Asia/Hong_Kong")
    return [pd.DataFrame({"A":raw[:,i]},index=idx) for i in range(3)]
def run(name,backend,inputs,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    names=("x","y") if name==COPULA else ("f1","f2","f3")
    assert op.metadata.panel_params==names
    assert _parameter_contract(op,names)[2]
    if backend=="polars":
        inputs=[pl.from_pandas(p.rename_axis("date").reset_index()) for p in inputs]
    out=op.calculate(**dict(zip(names,inputs)),**kw)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out
def shift_oracle(raw,t,recent,prior):
    old=raw[t-recent-prior:t-recent];new=raw[t-recent:t]
    med=np.median(old,axis=0);mad=1.4826*np.median(np.abs(old-med),axis=0)
    scale=np.where(mad>0,mad,np.std(old,axis=0))
    x=(new-med)/scale;y=(old-med)/scale
    return max(2*cdist(x,y).mean()-2*pdist(x).sum()/(len(x)*(len(x)-1))
               -2*pdist(y).sum()/(len(y)*(len(y)-1)),0.)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",[SHIFT,SCORE])
def test_true_three_features_prior_only_reference_scale_and_defaults(name,backend):
    inputs=frames();r,p,w=5,10,8
    kwargs={"recent_window":r,"prior_window":p}
    if name==SCORE:kwargs["window"]=w
    got=run(name,backend,inputs,**kwargs)
    raw=np.column_stack([v.A.to_numpy() for v in inputs])
    shifts=np.array([shift_oracle(raw,t,r,p) for t in range(r+p,len(raw))])
    expected=shifts[-1]
    if name==SCORE:
        history=shifts[-w-1:-1];med=np.median(history)
        mad=1.4826*np.median(np.abs(history-med))
        scale=mad if mad>0 else np.std(history)
        expected=(expected-med)/scale
    assert got.iloc[-1,0]==pytest.approx(expected,abs=1e-9)
    np.testing.assert_allclose(run(name,backend,[v.iloc[:70] for v in inputs],**kwargs),got.iloc[:70],equal_nan=True,atol=1e-9)
    mutated=[v.copy() for v in inputs]
    for v in mutated:v.iloc[-1]=np.nan
    assert run(name,backend,mutated,**kwargs).iloc[-1,0]==pytest.approx(got.iloc[-1,0],abs=1e-12)
    scaled=[v*s for v,s in zip(inputs,(1e-200,1e200,1e-100))]
    np.testing.assert_allclose(run(name,backend,scaled,**kwargs),got,equal_nan=True,atol=1e-8)
    default={"recent_window":20,"prior_window":60}
    if name==SCORE:default["window"]=60
    np.testing.assert_allclose(run(name,backend,inputs),run(name,backend,inputs,**default),equal_nan=True)
    broken=[v.copy() for v in inputs];broken[0].iloc[50]=np.nan
    assert run(name,backend,broken,**kwargs).iloc[51:66].isna().all().all()
    for key,bad in (("recent_window",True),("prior_window",2.5)):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,inputs,**{key:bad})
    if name==SCORE:
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,inputs,window=2)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_copula_midrank_oracle_feasible_window_and_nuisance_roles(backend):
    inputs=frames(30)[:2];n=20;grid=4
    u=(rankdata(inputs[0].A.to_numpy()[-n:])-0.5)/n
    v=(rankdata(inputs[1].A.to_numpy()[-n:])-0.5)/n
    points=(np.arange(grid)+.5)/grid
    expected=np.mean([(np.mean((u<=a)&(v<=b))-a-b+1-np.mean((u<=1-a)&(v<=1-b)))**2
                       for a in points for b in points])
    out=run(COPULA,backend,inputs,window=n,grid=grid)
    assert out.iloc[-1,0]==pytest.approx(expected,abs=1e-12)
    op=OperatorRegistry.get(COPULA,backend,mode="research")
    assert not op.metadata.param_specs["window"].searchable
    assert not op.metadata.param_specs["grid"].searchable
    for key,bad in (("window",9),("window",True),("grid",3.5)):
        with pytest.raises((ValueError,TypeError)):
            run(COPULA,backend,inputs,**{key:bad})

def test_scale_free_robust_z_and_zero_self_distance():
    from factor_engine.cleaned_operators.distribution_break import _robust_z,_euclidean_pairs
    values=np.array([0.,1.,3.,2.,5.,2.,10.])
    expected=_robust_z(values,5,3)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(_robust_z(values*scale,5,3),expected,equal_nan=True,atol=1e-12)
    pts=np.array([[1e308,-1e308],[-1e308,1e308]])
    distances=_euclidean_pairs(pts,pts)
    assert np.isfinite(distances).all()
    assert (np.diag(distances)==0).all()
