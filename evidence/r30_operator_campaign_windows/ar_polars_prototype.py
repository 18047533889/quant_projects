"""Bounded native-Polars prototype for the canonical AR coefficient."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import polars as pl

OUT = Path("evidence/r30_operator_campaign_windows/ar_polars_prototype_results.json")

def oracle(a, w, lag, mp, policy):
    out=np.full_like(a,np.nan,dtype=float); mp=max(3,mp)
    for c in range(a.shape[1]):
      for r in range(a.shape[0]):
        if policy=="full" and r<w-1: continue
        s=max(0,r-w+1)
        if r-s < lag: continue
        y=a[s+lag:r+1,c]; z=a[s:r-lag+1,c]
        ok=np.isfinite(y)&np.isfinite(z)
        if ok.sum()<mp: continue
        y,z=y[ok],z[ok]
        if np.std(z)>0: out[r,c]=np.mean((y-y.mean())*(z-z.mean()))/np.var(z)
    return out

def numpy_current(a,w,lag,mp,policy):
    return oracle(a,w,lag,mp,policy)

def native(df,w,lag,mp,policy):
    pair_w=w-lag; mp=max(3,mp)
    if pair_w < mp:
        return df.select([pl.lit(None,dtype=pl.Float64).alias(c) for c in df.columns])
    exprs=[]
    for c in df.columns:
        y=pl.col(c).cast(pl.Float64).fill_nan(None)
        z=pl.col(c).shift(lag).cast(pl.Float64).fill_nan(None)
        pair=y.is_not_null() & z.is_not_null()
        yp=pl.when(pair).then(y).otherwise(None)
        zp=pl.when(pair).then(z).otherwise(None)
        count=pair.cast(pl.Int64).rolling_sum(pair_w,min_samples=1)
        slope=pl.rolling_cov(yp,zp,window_size=pair_w,min_samples=1,ddof=0)/zp.rolling_var(pair_w,min_samples=1,ddof=0)
        valid=(count>=mp) & zp.rolling_var(pair_w,min_samples=1,ddof=0).is_finite()
        if policy=="full": valid=valid & (pl.int_range(pl.len())>=w-1)
        exprs.append(pl.when(valid).then(slope).otherwise(None).alias(c))
    return df.select(exprs)

def run_case(name,a,w,lag,mp,policy):
    expected=oracle(a,w,lag,mp,policy)
    observed=native(pl.DataFrame({f"c{i}":a[:,i] for i in range(a.shape[1])}),w,lag,mp,policy).to_numpy()
    finite=np.isfinite(expected)&np.isfinite(observed)
    err=float(np.max(np.abs(expected[finite]-observed[finite]))) if finite.any() else 0.0
    same_nan=bool(np.array_equal(np.isnan(expected),np.isnan(observed)))
    passed=bool(same_nan and np.allclose(expected,observed,equal_nan=True,rtol=1e-10,atol=1e-12))
    return {"case":name,"lag":lag,"warmup_policy":policy,"same_nan_mask":same_nan,"max_abs_error":err,"passed":passed}

def bench(fn,repeats=7):
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); times.append(time.perf_counter()-t)
    return float(np.median(times)),times

def main():
    rng=np.random.default_rng(3031); n=80
    x=np.empty((n,2)); x[0]=[.4,-.2]
    for r in range(1,n): x[r]=.3+.72*x[r-1]+rng.normal(0,.08,2)
    xn=x.copy(); xn[[7,14,27,51],0]=np.nan; xn[[3,19,44],1]=np.nan
    large=x+1e12
    cases=[]
    for lag in (1,2):
      for policy in ("expanding","full"):
        cases.append(run_case("finite",x,20,lag,8,policy))
        cases.append(run_case("pairwise_nan",xn,20,lag,8,policy))
        cases.append(run_case("large_offset_1e12",large,20,lag,8,policy))
    b=rng.normal(size=(512,8)).cumsum(axis=0); b[::41,2]=np.nan; b[::53,6]=np.nan
    df=pl.DataFrame({f"c{i}":b[:,i] for i in range(b.shape[1])})
    native(df,30,2,10,"expanding")
    numpy_current(b,30,2,10,"expanding")
    nt,ns=bench(lambda:native(df,30,2,10,"expanding"))
    pt,ps=bench(lambda:numpy_current(b,30,2,10,"expanding"))
    payload={"status":"PASS" if all(c["passed"] for c in cases) else "FAIL","semantic_cases":cases,"benchmark":{"shape":[512,8],"repeats":7,"native_polars_median_seconds":nt,"current_numpy_median_seconds":pt,"speedup_vs_current":pt/nt,"native_samples":ns,"numpy_samples":ps}}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"failed":[c for c in cases if not c["passed"]],"benchmark":payload["benchmark"]},indent=2))
if __name__=="__main__": main()
