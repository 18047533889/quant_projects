"""Regularized cross-sectional neutralization with explicit solver evidence."""
from numbers import Integral, Real
import numpy as np
import pandas as pd

def _num(name, x, low=0.0, high=None, strict=False):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Real): raise TypeError(f"{name} must be a real number, not bool")
    x=float(x)
    if not np.isfinite(x) or x < low or (strict and x == low) or (high is not None and x > high): raise ValueError(f"invalid {name}")
    return x

def _inputs(values, exposures, keys, value_col):
    if value_col not in values: raise ValueError(f"missing value column {value_col}")
    for f,n in ((values,"values"),(exposures,"exposures")):
        m=[k for k in keys if k not in f]
        if m: raise ValueError(f"{n} missing keys: {m}")
    if exposures.duplicated(keys).any(): raise ValueError("exposure keys must be unique")
    cols=[c for c in exposures if c not in keys]
    if not cols: raise ValueError("No exposure columns found")
    overlap=set(cols)&set(values.columns)
    if overlap: raise ValueError(f"factor/exposure field collision: {sorted(overlap)}")
    tag="__fp_row_ordinal__"
    if tag in values or tag in exposures: raise ValueError(f"reserved field {tag}")
    left=values.copy(); left[tag]=np.arange(len(left))
    return left.merge(exposures,on=keys,how="left",sort=False,validate="many_to_one"),cols,tag

def _cd(X,y,alpha,l1_ratio,max_iter,tol):
    n,p=X.shape; b=np.zeros(p); l1=alpha*l1_ratio; l2=alpha*(1-l1_ratio); conv=False
    for it in range(1,max_iter+1):
        old=b.copy()
        for j in range(p):
            r=y-X@b+X[:,j]*b[j]; rho=X[:,j]@r/n; z=X[:,j]@X[:,j]/n+l2
            b[j]=np.sign(rho)*max(abs(rho)-l1,0)/z if z else 0
        if np.max(np.abs(b-old),initial=0)<=tol: conv=True; break
    r=y-X@b; grad=-(X.T@r)/n+l2*b
    gap=float(np.max(np.where(b!=0,np.abs(grad+l1*np.sign(b)),np.maximum(np.abs(grad)-l1,0)),initial=0))
    obj=float(r@r/(2*n)+l1*np.abs(b).sum()+l2*(b@b)/2)
    return b,it,conv,obj,gap

def _run(values,exposures,*,kind,alpha,date_col,asset_col,value_col,min_observations,add_intercept,normalize,max_iter=1000,tol=1e-4,l1_ratio=0.0):
    alpha=_num("alpha",alpha)
    if isinstance(min_observations,(bool,np.bool_)) or not isinstance(min_observations,Integral) or min_observations<=0: raise ValueError("min_observations must be a positive integer")
    if kind!="ridge":
        if isinstance(max_iter,(bool,np.bool_)) or not isinstance(max_iter,Integral) or max_iter<=0: raise ValueError("max_iter must be a positive integer")
        tol=_num("tol",tol,strict=True); l1_ratio=_num("l1_ratio",l1_ratio,high=1)
    merged,cols,tag=_inputs(values,exposures,[date_col,asset_col],value_col)
    out=np.full(len(values),np.nan); evidence=[]
    for date,g in merged.groupby(date_col,sort=False,dropna=False):
        y=g[value_col].to_numpy(float); X=g[cols].to_numpy(float); valid=np.isfinite(y)&np.isfinite(X).all(1); n=int(valid.sum())
        if n<min_observations:
            evidence.append(dict(date=date,status="INSUFFICIENT_DATA",n_effective=n,n_iter=0,converged=False)); continue
        Xv=X[valid]; yv=y[valid]; xm=Xv.mean(0) if add_intercept else np.zeros(Xv.shape[1]); ym=float(yv.mean()) if add_intercept else 0.; Xc=Xv-xm; yc=yv-ym
        scale=Xc.std(0,ddof=1) if normalize else np.ones(Xc.shape[1]); scale[~np.isfinite(scale)|(scale==0)]=1; Z=Xc/scale
        try:
            if kind=="ridge":
                b=np.linalg.solve(Z.T@Z/n+alpha*np.eye(Z.shape[1]),Z.T@yc/n); it=1; conv=True; r=yc-Z@b; obj=float(r@r/(2*n)+alpha*(b@b)/2); gap=float(np.max(np.abs(-(Z.T@r)/n+alpha*b),initial=0))
            else: b,it,conv,obj,gap=_cd(Z,yc,alpha,l1_ratio,max_iter,tol)
        except np.linalg.LinAlgError:
            evidence.append(dict(date=date,status="NUMERICAL_FAILURE",n_effective=n,n_iter=0,converged=False)); continue
        coef=b/scale; intercept=ym-xm@coef if add_intercept else 0.; residual=y-(X@coef+intercept); residual[~valid]=np.nan; out[g[tag].to_numpy(int)]=residual
        evidence.append(dict(date=date,status="CONVERGED" if conv else "ITERATION_BUDGET_EXHAUSTED",n_effective=n,n_iter=it,converged=conv,objective=obj,stationarity_gap=gap,effective_alpha=alpha,loss_normalization="mean"))
    ans=pd.Series(out,index=values.index,name="residual"); ans.attrs["fit_diagnostics"]=evidence; ans.attrs["objective"]="||y-Xb||^2/(2n)+alpha*(l1_ratio*L1+(1-l1_ratio)*L2/2)"; return ans

def ridge_neutralize(values,exposures,alpha=1.0,date_col="date",asset_col="asset_id",value_col="value",min_observations=10,add_intercept=True,normalize=True):
    return _run(values,exposures,kind="ridge",alpha=alpha,date_col=date_col,asset_col=asset_col,value_col=value_col,min_observations=min_observations,add_intercept=add_intercept,normalize=normalize)
def lasso_neutralize(values,exposures,alpha=1.0,date_col="date",asset_col="asset_id",value_col="value",min_observations=10,add_intercept=True,normalize=True,max_iter=1000,tol=1e-4):
    return _run(values,exposures,kind="lasso",alpha=alpha,date_col=date_col,asset_col=asset_col,value_col=value_col,min_observations=min_observations,add_intercept=add_intercept,normalize=normalize,max_iter=max_iter,tol=tol,l1_ratio=1.)
def elastic_net_neutralize(values,exposures,alpha=1.0,l1_ratio=0.5,date_col="date",asset_col="asset_id",value_col="value",min_observations=10,add_intercept=True,normalize=True,max_iter=1000,tol=1e-4):
    return _run(values,exposures,kind="elastic",alpha=alpha,date_col=date_col,asset_col=asset_col,value_col=value_col,min_observations=min_observations,add_intercept=add_intercept,normalize=normalize,max_iter=max_iter,tol=tol,l1_ratio=l1_ratio)
