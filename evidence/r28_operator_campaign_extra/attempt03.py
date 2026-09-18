"""Targeted R28 attempt02 for four triaged statistical operators."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

BASE=Path(__file__).resolve().parent
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path); assert s and s.loader
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
campaign=loadmod("r28_campaign_base",BASE/"campaign.py")
core=loadmod("r23_campaign_core",BASE.parent/"r23_operator_campaign"/"campaign.py")
DOMAIN="daily_statistical_triage4_v1"

def data(rows=64,assets=8):
    d=campaign.fixture(rows,assets); t=np.arange(rows)[:,None]; a=np.arange(assets)[None,:]
    inc=0.001+0.012*((-1.0)**(t+a)); logp=np.log(100+2*a)+np.cumsum(inc,axis=0)
    d["bounce_price"]=pd.DataFrame(np.exp(logp),columns=d["close"].columns)
    return d

def future(d,cut):
    z={k:v.copy() for k,v in d.items()}; n=len(d["ret"])-cut
    z["weight"].iloc[cut:,:]*=np.resize(np.array([2.0,.35,1.8,.5]),n)[:,None]
    z["ret"].iloc[cut:,:]=d["ret"].iloc[cut:,:]*np.resize(np.array([2.5,-1.8,.4,-2.2]),n)[:,None]
    z["bounce_price"].iloc[cut:,:]*=np.exp(np.cumsum(np.resize(np.array([.03,-.025,.028,-.032]),n)))[:,None]
    return z

def oracle(name,d):
    if name=="ts_mass_concentration":
        w=d["weight"]; n=w.rolling(12,min_periods=8).count(); h=(w*w).rolling(12,min_periods=8).sum()/w.rolling(12,min_periods=8).sum().pow(2)
        return ((h-1/n)/(1-1/n)).to_numpy()
    if name=="vv1_downside_vol_share":
        x=d["ret"].to_numpy(); out=np.full_like(x,np.nan)
        for j in range(x.shape[1]):
            for i in range(x.shape[0]):
                v=x[max(0,i-11):i+1,j]; v=v[np.isfinite(v)]
                dn=v[v<0]; up=v[v>0]
                if v.size>=8 and dn.size>=3 and up.size>=3:
                    sd=np.std(dn,ddof=1); su=np.std(up,ddof=1)
                    if sd+su>1e-12: out[i,j]=sd/(sd+su)
        return out
    if name=="ts_roll_effective_spread":
        x=np.log(d["bounce_price"].to_numpy()); dx=np.diff(x,axis=0,prepend=np.nan); out=np.full_like(x,np.nan)
        for j in range(x.shape[1]):
            for i in range(x.shape[0]):
                v=dx[max(0,i-11):i+1,j]; aa=v[1:]; bb=v[:-1]; ok=np.isfinite(aa)&np.isfinite(bb)
                if ok.sum()>=8:
                    cov=np.cov(aa[ok],bb[ok],ddof=1)[0,1]
                    if cov<0: out[i,j]=2*np.sqrt(-cov)
        return out
    if name=="ts_vol_of_vol":
        r=d["ret"]; inner=r.rolling(5,min_periods=4).std(ddof=0); return np.log(inner+1e-12).rolling(12,min_periods=4).std(ddof=0).to_numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--recipes",required=True); p.add_argument("--ledger",required=True); a=p.parse_args()
    recs=json.loads(Path(a.recipes).read_text())["recipes"]; assert len(recs)==1
    d=data(); cut=55; f=future(d,cut); ledger=Path(a.ledger); load_all(); summary={}
    for rec in recs:
        name=rec["canonical"]; outs={}
        for backend in ("pandas_numpy","polars"):
            op=OperatorRegistry.get(name,backend,mode="any"); fp=core.execution_fingerprint(name,backend,rec,d,op)
            core.emit(ledger,{"canonical":name,"backend":backend,"status":"RUNNING","fingerprint":fp,"fixture_domain":DOMAIN})
            try:
                roles=rec["inputs"]; out=core.array(op.calculate(*[core.as_backend(d[r],backend) for r in roles],**rec["params"])); fut=core.array(op.calculate(*[core.as_backend(f[r],backend) for r in roles],**rec["params"])); exp=oracle(name,d); expf=oracle(name,f)
                prefix=np.allclose(out[:cut],fut[:cut],equal_nan=True); suffix=not np.allclose(out[cut:],fut[cut:],equal_nan=True); oe=np.allclose(out,exp,equal_nan=True,rtol=1e-9,atol=1e-11) and np.allclose(fut,expf,equal_nan=True,rtol=1e-9,atol=1e-11)
                deg=None
                if name=="ts_roll_effective_spread":
                    smooth=d["close"]; const=smooth*0+100
                    deg=bool(np.isnan(core.array(op.calculate(core.as_backend(smooth,backend),**rec["params"]))).all() and np.isnan(core.array(op.calculate(core.as_backend(const,backend),**rec["params"]))).all())
                ok=np.isfinite(out).any() and prefix and suffix and oe and deg is not False; outs[backend]=out
                core.emit(ledger,{"canonical":name,"backend":backend,"status":"EXECUTED_FINITE" if ok else "FAILED_EXECUTION_PENDING_TRIAGE","finite":int(np.isfinite(out).sum()),"future_prefix_invariant":bool(prefix),"future_suffix_changed":bool(suffix),"independent_oracle_equal":bool(oe),"degenerate_paths_nan":deg,"fingerprint":fp,"fixture_domain":DOMAIN})
            except Exception as e: core.emit(ledger,{"canonical":name,"backend":backend,"status":"FAILED_EXECUTION_PENDING_TRIAGE","error_type":type(e).__name__,"error":str(e),"fingerprint":fp,"fixture_domain":DOMAIN})
        eq=set(outs)=={"pandas_numpy","polars"} and np.allclose(outs["pandas_numpy"],outs["polars"],equal_nan=True)
        summary[name]=bool(eq); core.emit(ledger,{"canonical":name,"status":"CANONICAL_PARITY" if eq else "CANONICAL_PARITY_FAILED","pandas_polars_equal":bool(eq),"fixture_domain":DOMAIN})
    print(json.dumps(summary,sort_keys=True))
if __name__=="__main__": main()
