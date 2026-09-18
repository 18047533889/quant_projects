"""Independent bounded R24 campaign for forty daily numeric operators."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

_SHARED_CAMPAIGN = (
    Path(__file__).resolve().parents[1]
    / "r23_operator_campaign"
    / "campaign.py"
)
_SPEC = importlib.util.spec_from_file_location("r23_campaign_core", _SHARED_CAMPAIGN)
assert _SPEC is not None and _SPEC.loader is not None
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)

DOMAIN = "daily_ohlcv_8_asset_v1"

def _df(v, cols): return pd.DataFrame(v, columns=cols)

def fixture(rows=64, assets=8):
    t=np.arange(rows,dtype=float)[:,None]; a=np.arange(assets,dtype=float)[None,:]
    mid=80+0.12*t+2.5*np.sin(t/6+a/5)+a*3
    op=mid*(1+0.004*np.sin(t/3+a)); cl=mid*(1+0.004*np.cos(t/4+a/2))
    hi=np.maximum(op,cl)+1.2+0.05*a; lo=np.minimum(op,cl)-1.0-0.04*a
    vol=8e5+2e4*a+1.2e5*(1.2+np.sin(t/5+a/3)); turn=0.01+0.001*a+0.003*(1.2+np.cos(t/7+a))
    ret=0.012*np.sin(t/4+a/2)+0.003*np.cos(t/9+a); ret[::9,:]=0.0001
    cols=[f"asset_{i:02d}" for i in range(assets)]
    close=_df(cl,cols)
    out={"open":_df(op,cols),"high":_df(hi,cols),"low":_df(lo,cols),"close":close,
         "pre_close":close.shift(1).bfill(),"volume":_df(vol,cols),"turnover":_df(turn,cols),"ret":_df(ret,cols)}
    check(out); return out

def check(d):
    for k,v in d.items(): assert np.isfinite(v.to_numpy(float)).all(),k
    assert (d["open"]>0).all().all() and (d["close"]>0).all().all()
    assert (d["volume"]>0).all().all() and (d["turnover"]>0).all().all()
    assert (d["high"]>=d["open"]).all().all() and (d["high"]>=d["close"]).all().all()
    assert (d["low"]<=d["open"]).all().all() and (d["low"]<=d["close"]).all().all()

def future(d,cut):
    z={k:v.copy() for k,v in d.items()}; s=slice(cut,None)
    z["open"].iloc[s,:]=d["open"].iloc[s,:]*1.035; z["close"].iloc[s,:]=d["close"].iloc[s,:]*0.985
    z["high"].iloc[s,:]=np.maximum(z["open"].iloc[s,:],z["close"].iloc[s,:])+1.7
    z["low"].iloc[s,:]=np.minimum(z["open"].iloc[s,:],z["close"].iloc[s,:])-0.75
    z["pre_close"].iloc[s,:]=d["pre_close"].iloc[s,:]*1.012
    tail = len(d["close"]) - cut
    z["volume"].iloc[s,:]=d["volume"].iloc[s,:]*(1.16+0.01*np.arange(tail)[:,None])
    z["turnover"].iloc[s,:]=d["turnover"].iloc[s,:]*(0.91+0.005*np.arange(tail)[:,None])
    z["ret"].iloc[s,:]=d["ret"].iloc[s,:]*-1.27+0.002*np.sin(np.arange(tail)[:,None])
    check(z); return z

def oracle(name,d):
    o,h,l,c=d["open"].to_numpy(),d["high"].to_numpy(),d["low"].to_numpy(),d["close"].to_numpy(); rng=h-l
    if name=="candle_body_ratio": return np.abs(c-o)/rng
    if name=="candle_upper_shadow_ratio": return (h-np.maximum(o,c))/rng
    if name=="candle_lower_shadow_ratio": return (np.minimum(o,c)-l)/rng
    if name=="candle_body_position": return ((o+c)/2-l)/rng
    if name=="signed_dollar_volume": return np.sign(d["ret"].to_numpy())*c*d["volume"].to_numpy()
    if name=="return_per_turnover": return d["ret"].to_numpy()/d["turnover"].to_numpy()
    if name=="open_close_return": return c/o-1
    if name=="overnight_return": return o/d["pre_close"].to_numpy()-1
    if name in {"true_range","true_range_pct"}:
        prev=np.vstack([np.full((1,c.shape[1]),np.nan),c[:-1]]); tr=np.maximum.reduce([rng,np.abs(h-prev),np.abs(l-prev)]); tr[0]=rng[0]
        return tr if name=="true_range" else tr/prev
    return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--recipes",required=True); p.add_argument("--ledger",required=True); p.add_argument("--rows",type=int,default=64); p.add_argument("--assets",type=int,default=8); a=p.parse_args()
    if not 32<=a.rows<=96 or not 8<=a.assets<=16: raise ValueError("bounded rows/assets")
    core.campaign_protocol_hash=lambda:hashlib.sha256(Path(core.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    recipes=json.loads(Path(a.recipes).read_text())["recipes"]; assert len(recipes)==40
    inputs=fixture(a.rows,a.assets); cut=a.rows-9; changed=future(inputs,cut); ledger=Path(a.ledger); load_all(); summary={}
    for rec in sorted(recipes,key=lambda r:r["canonical"]):
        name=rec["canonical"]; outs={}; fps={}
        for backend in ("pandas_numpy","polars"):
            op=OperatorRegistry.get(name,backend,mode="any")
            if op is None: core.emit(ledger,{"canonical":name,"backend":backend,"status":"UNTESTED_NO_SLOT","fixture_domain":DOMAIN}); continue
            fp=core.execution_fingerprint(name,backend,rec,inputs,op); fps[backend]=fp
            core.emit(ledger,{"canonical":name,"backend":backend,"status":"RUNNING","fingerprint":fp,"rows":a.rows,"assets":a.assets,"params":rec["params"],"fixture_domain":DOMAIN})
            try:
                roles=rec["inputs"]; actual=core.array(op.calculate(*[core.as_backend(inputs[r],backend) for r in roles],**rec["params"])); fut=core.array(op.calculate(*[core.as_backend(changed[r],backend) for r in roles],**rec["params"])); outs[backend]=actual
                finite=int(np.isfinite(actual.astype(float)).sum()); prefix=bool(actual.shape==fut.shape and np.allclose(actual[:cut],fut[:cut],equal_nan=True,rtol=1e-9,atol=1e-11)); suffix=bool(not np.allclose(actual[cut:],fut[cut:],equal_nan=True,rtol=1e-9,atol=1e-11))
                exp=oracle(name,inputs); expf=oracle(name,changed); oe=None if exp is None else bool(np.allclose(actual,exp,equal_nan=True,rtol=1e-9,atol=1e-11) and np.allclose(fut,expf,equal_nan=True,rtol=1e-9,atol=1e-11))
                status,reason=core.execution_status(finite,prefix,None)
                if status=="EXECUTED_FINITE" and not suffix: status,reason="FAILED_EXECUTION_PENDING_TRIAGE","legal future perturbation did not change suffix"
                if status=="EXECUTED_FINITE" and oe is False: status,reason="FAILED_EXECUTION_PENDING_TRIAGE","independent oracle mismatch"
                core.emit(ledger,{"canonical":name,"backend":backend,"status":status,"reason":reason,"shape":list(actual.shape),"finite":finite,"future_prefix_invariant_all_inputs":prefix,"future_suffix_changed":suffix,"independent_oracle_equal":oe,"fingerprint":fp,"params":rec["params"],"fixture_domain":DOMAIN})
            except Exception as e: core.emit(ledger,{"canonical":name,"backend":backend,"status":"FAILED_EXECUTION_PENDING_TRIAGE","reason":"triage required","error_type":type(e).__name__,"error":str(e)[:500],"fingerprint":fp,"params":rec["params"],"fixture_domain":DOMAIN})
            gc.collect()
        if set(outs)=={"pandas_numpy","polars"}:
            eq=bool(outs["pandas_numpy"].shape==outs["polars"].shape and np.allclose(outs["pandas_numpy"],outs["polars"],equal_nan=True,rtol=1e-9,atol=1e-11)); summary[name]=eq
            core.emit(ledger,{"canonical":name,"status":"CANONICAL_PARITY" if eq else "CANONICAL_PARITY_FAILED","pandas_polars_equal":eq,"backend_fingerprints":fps,"fixture_domain":DOMAIN})
    print(json.dumps({"selected":sorted(summary),"parity_pass":sum(summary.values()),"count":len(summary),"rows":a.rows,"assets":a.assets},sort_keys=True))
if __name__=="__main__": main()
