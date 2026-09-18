"""Independent bounded numeric oracle for representative R31 formulas."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

OUT=Path("evidence/r31_operator_campaign/oracle_results.json")
def roll(a,w,mp,fn):
 out=np.full_like(a,np.nan,dtype=float)
 for c in range(a.shape[1]):
  for r in range(a.shape[0]):
   z=a[max(0,r-w+1):r+1,c]; z=z[np.isfinite(z)]
   if len(z)>=mp: out[r,c]=fn(z)
 return out
def cond_transition(c,w):
 out=np.full_like(c,np.nan)
 for j in range(c.shape[1]):
  for r in range(c.shape[0]):
   if not np.isfinite(c[r,j]): continue
   prev=None;n=0
   for v in c[max(0,r-w+1):r+1,j]:
    if not np.isfinite(v): prev=None;continue
    cur=bool(v)
    if prev is not None and cur!=prev:n+=1
    prev=cur
   out[r,j]=n
 return out
def streak(a):
 out=np.full_like(a,np.nan)
 for c in range(a.shape[1]):
  n=0
  for r,v in enumerate(a[:,c]):
   if not np.isfinite(v):n=0;continue
   n=n+1 if v>0 else 0;out[r,c]=n
 return out
def median3(a):
 out=np.full_like(a,np.nan)
 for c in range(a.shape[1]):
  for r,v in enumerate(a[:,c]):
   if r<2:out[r,c]=v
   elif np.isfinite(a[r-2:r+1,c]).all():out[r,c]=np.median(a[r-2:r+1,c])
 return out
def regression(y,x,w,mp,kind):
 out=np.full_like(y,np.nan)
 for c in range(y.shape[1]):
  for r in range(y.shape[0]):
   s=max(0,r-w+1); yy=y[s:r+1,c];xx=x[s:r+1,c];ok=np.isfinite(yy)&np.isfinite(xx)
   if ok.sum()<mp or np.var(xx[ok])==0:continue
   X=np.column_stack((np.ones(ok.sum()),xx[ok])); beta=np.linalg.lstsq(X,yy[ok],rcond=None)[0]
   pred=X@beta
   if kind=="slope":out[r,c]=beta[1]
   elif kind=="intercept":out[r,c]=beta[0]
   else:
    den=np.sum((yy[ok]-yy[ok].mean())**2)
    if den>0:out[r,c]=1-np.sum((yy[ok]-pred)**2)/den
 return out
def actual(n,b,frames,p):
 op=OperatorRegistry.get(n,b,mode="any");args=[pl.from_pandas(f) for f in frames] if b=="polars" else frames
 z=op.calculate(*args,**p);return z.to_numpy() if isinstance(z,pl.DataFrame) else z.to_numpy(float)
def main():
 rng=np.random.default_rng(3109);a=rng.normal(size=(40,2)).cumsum(0);a[[6,19],0]=np.nan;a[12,1]=np.nan
 x=pd.DataFrame(a,columns=["A","B"]);other=pd.DataFrame(np.where(np.isfinite(a),.4*a+.2+np.sin(a)*.03,np.nan),columns=x.columns)
 c=np.where(np.isfinite(a),(a>np.nanmedian(a,axis=0)).astype(float),np.nan);condition=pd.DataFrame(c,columns=x.columns)
 cases={
 "ts_mad":([x],{"window":10,"min_periods":5,"scale":1.0},roll(a,10,5,lambda z:np.median(np.abs(z-np.median(z))))),
 "ts_quantile":([x],{"d":10,"q":.3},roll(a,10,1,lambda z:np.quantile(z,.3))),
 "ts_quantile_range":([x],{"window":10,"q_low":.25,"q_high":.75,"min_periods":5},roll(a,10,5,lambda z:np.quantile(z,.75)-np.quantile(z,.25))),
 "ts_upside_deviation":([x],{"window":10,"target":0.,"min_periods":2},roll(a,10,2,lambda z:np.sqrt(np.mean(np.maximum(z,0)**2)))),
 "ts_zero_ratio":([x],{"window":10,"tolerance":.05,"min_periods":3},roll(a,10,3,lambda z:np.mean(np.abs(z)<=.05))),
 "ts_positive_streak":([x],{},streak(a)),
 "ts_transition_count":([condition],{"window":10,"missing_policy":"break"},cond_transition(c,10)),
 "ts_nth_value":([x],{"window":10,"n":3,"order":"largest","min_periods":5},roll(a,10,5,lambda z:np.sort(z)[-3] if len(z)>=3 else np.nan)),
 "ts_ratio":([x],{},np.vstack((np.full((1,2),np.nan),a[1:]/a[:-1]))),
 "ts_median3_causal":([x],{},median3(a)),
 "ts_regression_slope":([x,other],{"window":10,"lag":0,"retval":"slope","min_periods":5,"add_intercept":True},regression(a,other.to_numpy(),10,5,"slope")),
 "ts_regression_intercept":([x,other],{"window":10,"min_periods":5,"add_intercept":True},regression(a,other.to_numpy(),10,5,"intercept")),
 "ts_regression_r2":([x,other],{"window":10,"min_periods":5,"add_intercept":True},regression(a,other.to_numpy(),10,5,"r2"))}
 load_all();rows=[]
 for n,(frames,p,e) in cases.items():
  for b in ("pandas_numpy","polars"):
   o=actual(n,b,frames,p);err=float(np.nanmax(np.abs(o-e)));passed=bool(np.allclose(o,e,equal_nan=True,rtol=1e-10,atol=1e-12));rows.append({"canonical":n,"backend":b,"passed":passed,"max_abs_error":err})
   if not passed:raise AssertionError((n,b,err,np.where(~np.isclose(o,e,equal_nan=True))))
 payload={"status":"PASS","cases":len(cases),"backend_checks":len(rows),"results":rows};OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print({k:payload[k] for k in ("status","cases","backend_checks")})
if __name__=="__main__":main()
