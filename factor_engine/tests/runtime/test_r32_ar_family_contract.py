"""Independent contract tests for the R32 AR family Polars repair."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

SPECS={
 "ts_ar_fitted_value":("forecast",0,0),"ts_ar_forecast":("forecast",0,0),
 "ts_ar_innovation":("innovation",0,0),"ts_ar_in_sample_resid":("innovation",0,0),
 "ts_ar_innovation_z":("innovation_z",0,0),"ts_ar_prior_coeff":("coeff",1,0),
 "ts_ar_prior_forecast":("forecast",1,0),"ts_ar_prior_innovation":("innovation",1,0),
 "ts_ar_prior_innovation_z":("innovation_z",1,0),"ts_ar_coeff_stability":("coeff_stability",1,5)}

def fit(seg,p):
 n=len(seg);X=np.full((n,p+1),np.nan)
 for t in range(p,n):
  lag=seg[t-p:t][::-1]
  if np.isfinite(lag).all():X[t]=np.r_[1.,lag]
 ok=np.isfinite(X).all(1)&np.isfinite(seg)
 if ok.sum()<p+2:return None,X
 return np.linalg.lstsq(X[ok],seg[ok],rcond=None)[0],X

def oracle(v,w,p,policy,stat,fit_lag,k):
 out=np.full(len(v),np.nan)
 for r in range(len(v)):
  end=r-fit_lag
  if end<0 or (policy=="full" and end<w-1):continue
  start=max(0,end-w+1);seg=v[start:end+1];beta,X=fit(seg,p)
  if beta is None or r<p:continue
  lag=v[r-p:r][::-1]
  if not np.isfinite(lag).all():continue
  pred=float(np.r_[1.,lag]@beta);innov=v[r]-pred if np.isfinite(v[r]) else np.nan
  if stat=="forecast":out[r]=pred
  elif stat=="innovation":out[r]=innov
  elif stat=="coeff":out[r]=beta[1]
  elif stat=="innovation_z":
   ok=np.isfinite(X).all(1)&np.isfinite(seg);sd=np.std(seg[ok]-X[ok]@beta)
   if sd>0:out[r]=innov/sd
  else:
   co=[]
   for j in range(k):
    e=r-fit_lag-j
    if e<0:break
    b,_=fit(v[max(0,e-w+1):e+1],p)
    if b is None:break
    co.append(b[1])
   if len(co)>=2:out[r]=np.std(co)
 return out

@pytest.mark.parametrize("name",sorted(SPECS))
def test_ar_family_matches_independent_oracle_and_preserves_time(name):
 load_all();rng=np.random.default_rng(3207);v=np.empty(70);v[:2]=[.2,-.1]
 for i in range(2,len(v)):v[i]=.15+.55*v[i-1]-.2*v[i-2]+rng.normal(0,.05)
 v[[13,31]]=np.nan; dates=pd.date_range("2025-01-01",periods=len(v))
 src=pl.DataFrame({"date":dates,"A":v});stat,lag,k=SPECS[name];params=dict(window=20,order=2,warmup_policy="expanding")
 got=OperatorRegistry.get(name,"polars",mode="any").calculate(src,**params)
 assert got["date"].to_list()==src["date"].to_list()
 np.testing.assert_allclose(got["A"].to_numpy(),oracle(v,20,2,"expanding",stat,lag,k),rtol=1e-12,atol=1e-12,equal_nan=True)

def test_ar_full_empty_and_prior_current_row_exclusion():
 load_all();op=OperatorRegistry.get("ts_ar_prior_coeff","polars",mode="any")
 empty=pl.DataFrame(schema={"date":pl.Date,"A":pl.Float64});assert op.calculate(empty,window=12,order=1,warmup_policy="full").shape==(0,2)
 v=np.sin(np.arange(40)/4);base=pl.DataFrame({"A":v});changed=base.with_columns(pl.when(pl.int_range(pl.len())==30).then(pl.col("A")+999).otherwise(pl.col("A")).alias("A"))
 a=op.calculate(base,window=12,order=1,warmup_policy="expanding")["A"].to_numpy();b=op.calculate(changed,window=12,order=1,warmup_policy="expanding")["A"].to_numpy()
 assert a[30]==b[30]
 full=OperatorRegistry.get("ts_ar_fitted_value","polars",mode="any").calculate(base,window=12,order=1,warmup_policy="full")["A"].to_numpy()
 assert np.isnan(full[:11]).all() and np.isfinite(full[11:]).all()
