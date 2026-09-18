"""Independent bounded R30 campaign for thirty liquidity/event operators."""
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

DOMAIN = "daily_liquidity_event_8_asset_v1"

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
    out.update({
      "working_capital":_df(420+2.5*t+14*np.sin(t/7+a),cols),
      "earnings":_df(55+0.35*t+6*np.sin(t/5+a/3),cols),
      "operating_cash_flow":_df(62+0.42*t+7*np.cos(t/6+a/4),cols),
      "roa":_df(0.06+0.006*np.sin(t/8+a/5),cols),
      "revenue":_df(500+4.2*t+20*np.cos(t/9+a/3),cols),
      "earnings_yield":_df(0.045+0.004*np.sin(t/6+a/4),cols),
      "fcf_yield":_df(0.04+0.003*np.cos(t/7+a/5),cols),
      "own_pe":_df(16+0.03*t+0.8*np.sin(t/5+a/3),cols),
      "market_pe":_df(14+0.025*t+0.6*np.cos(t/6+a/4),cols),
      "own_metric":_df(2.4+0.02*a+0.08*np.sin(t/7),cols),
      "reference_metric":_df(2.1+0.015*a+0.05*np.cos(t/8),cols),
      "value_metric":_df(1.8+0.01*t+0.12*np.sin(t/6+a/5),cols),
      "amihud":_df(2e-6+2e-7*(1.2+np.sin(t/8+a/3)),cols),
      "amount":_df((8e5+2e4*a+1.2e5*(1.2+np.sin(t/5+a/3)))*cl,cols),
      "state":_df(((t.astype(int)//4+a.astype(int))%3).astype(float),cols),
      "event":_df((((t.astype(int)+2*a.astype(int))%7)==0).astype(float),cols),
      "signed_event":_df(np.where(((t.astype(int)+a.astype(int))%6)==0,1.0,np.where(((t.astype(int)+a.astype(int))%9)==0,-1.0,0.0)),cols),
      "mark":_df(np.where(((t.astype(int)+a.astype(int))%6)==0,0.4+0.02*a,np.where(((t.astype(int)+a.astype(int))%9)==0,-0.3-0.01*a,0.0)),cols),
      "x":_df(20+0.08*t+1.7*np.sin(2*np.pi*t/5+a/4)+0.5*np.cos(2*np.pi*t/20+a/3),cols),
      "y":_df(11+0.05*t+1.1*np.cos(2*np.pi*t/7+a/5)+0.3*np.sin(t/9),cols),
      "weight":_df(1.0+0.2*a+0.4*(1.1+np.sin(t/6+a/4)),cols),
      "signed_volume":_df(vol*np.where(((t.astype(int)+a.astype(int))%5)<3,1.0,-1.0),cols),
      "up_event":_df((((t.astype(int)+a.astype(int))%11)==0).astype(float),cols),
      "down_event":_df((((t.astype(int)+2*a.astype(int))%13)==0).astype(float),cols),
      "limit_up_event":_df((((t.astype(int)+a.astype(int))%11)==0).astype(float),cols),
      "limit_down_event":_df((((t.astype(int)+2*a.astype(int))%13)==0).astype(float),cols),
      "known_status":_df(np.ones((rows,assets),dtype=float),cols),
      "is_suspend":_df((((t.astype(int)//3+a.astype(int))%9)==0).astype(float),cols),
    })
    check(out); return out

def check(d):
    for k,v in d.items(): assert np.isfinite(v.to_numpy(float)).all(),k
    assert (d["open"]>0).all().all() and (d["close"]>0).all().all()
    assert (d["volume"]>0).all().all() and (d["turnover"]>0).all().all()
    assert (d["high"]>=d["open"]).all().all() and (d["high"]>=d["close"]).all().all()
    assert (d["low"]<=d["open"]).all().all() and (d["low"]<=d["close"]).all().all()

def future(d,cut):
    z={k:v.copy() for k,v in d.items()}; s=slice(cut,None)
    tail = len(d["close"]) - cut
    swing=np.resize(np.array([1.32,0.71,1.38,0.66,1.44,0.62],dtype=float),tail)[:,None]
    z["close"].iloc[s,:]=d["close"].iloc[s,:]*swing
    z["open"].iloc[s,:]=d["open"].iloc[s,:]*swing*np.where(swing>1.0,0.97,1.03)
    z["high"].iloc[s,:]=np.maximum(z["open"].iloc[s,:],z["close"].iloc[s,:])*1.015
    z["low"].iloc[s,:]=np.minimum(z["open"].iloc[s,:],z["close"].iloc[s,:])*0.985
    z["pre_close"].iloc[s,:]=d["pre_close"].iloc[s,:]*1.012
    asset_shock=np.linspace(0.35,1.4,d["volume"].shape[1],dtype=float)[None,:]
    asset_shock[0,0]=24.0
    z["volume"].iloc[s,:]=d["volume"].iloc[s,:]*(1.16+0.01*np.arange(tail)[:,None])*asset_shock
    z["turnover"].iloc[s,:]=d["turnover"].iloc[s,:]*(0.91+0.005*np.arange(tail)[:,None])*asset_shock
    z["ret"].iloc[s,:]=d["ret"].iloc[s,:]*-1.27+0.002*np.sin(np.arange(tail)[:,None])
    for k in ("working_capital","operating_cash_flow","roa","revenue","earnings_yield","fcf_yield","own_pe","market_pe","own_metric","reference_metric","value_metric","amihud"):
        z[k].iloc[s,:]=d[k].iloc[s,:]*(1.08+0.02*np.arange(tail)[:,None])
    z["amount"].iloc[s,:]=d["amount"].iloc[s,:]*(1.08+0.02*np.arange(tail)[:,None])*asset_shock
    z["earnings"].iloc[s,:]=d["earnings"].iloc[s,:]*np.resize(np.array([1.2,-0.8,-1.1,1.3]),tail)[:,None]
    z["state"].iloc[s,:]=np.resize(np.array([2,0,2,1,0],dtype=float),tail)[:,None]
    z["event"].iloc[s,:]=np.resize(np.array([1,1,0,1,0],dtype=float),tail)[:,None]
    z["signed_event"].iloc[s,:]=np.resize(np.array([1,-1,1,0,-1],dtype=float),tail)[:,None]
    z["mark"].iloc[s,:]=np.resize(np.array([0.8,-0.7,0.6,0,-0.9],dtype=float),tail)[:,None]
    z["x"].iloc[s,:]=d["x"].iloc[s,:]*np.resize(np.array([1.2,0.82,1.25,0.78],dtype=float),tail)[:,None]
    z["y"].iloc[s,:]=d["y"].iloc[s,:]*np.resize(np.array([0.75,1.3,0.8,1.25],dtype=float),tail)[:,None]
    z["weight"].iloc[s,:]=d["weight"].iloc[s,:]*np.resize(np.array([2.0,0.4,1.8,0.5],dtype=float),tail)[:,None]
    z["signed_volume"].iloc[s,:]=d["signed_volume"].iloc[s,:]*np.resize(np.array([-1.4,1.8,-0.7,1.5],dtype=float),tail)[:,None]
    for k in ("event","up_event","down_event","limit_up_event","limit_down_event","is_suspend"):
        z[k].iloc[s,:]=np.resize(np.array([1,0,1,1,0],dtype=float),tail)[:,None]
    check(z); return z

def edge_fixtures(d):
    constant={k:v.copy() for k,v in d.items()}
    for k,value in (("open",100.0),("high",101.0),("low",99.0),("close",100.0),("pre_close",100.0)):
        constant[k].iloc[:,:]=value
    constant["volume"].iloc[:,:]=1_000_000.0
    constant["turnover"].iloc[:,:]=0.02
    constant["ret"].iloc[:,:]=0.0
    for k in constant:
        if k not in {"open","high","low","close","pre_close","volume","turnover","ret"}: constant[k].iloc[:,:]=float(constant[k].iloc[0,0])
    check(constant)
    missing={k:v.copy() for k,v in d.items()}
    for k in missing:
        missing[k].iloc[len(d[k])//2,2]=np.nan
    return {"constant_price":constant,"single_missing_bar":missing}

def oracle(name,d):
    x=d["x"]; r=d["ret"]; w=d["weight"]
    if name=="ts_lower_partial_moment": return np.maximum(20.0-x,0).pow(2).rolling(12,min_periods=8).mean().to_numpy()
    if name=="ts_upper_partial_moment": return np.maximum(x-20.0,0).pow(2).rolling(12,min_periods=8).mean().to_numpy()
    if name=="ts_mass_concentration":
        return ((w*w).rolling(12,min_periods=8).sum()/w.rolling(12,min_periods=8).sum().pow(2)).to_numpy()
    if name=="ts_semivariance_balance":
        up=r.clip(lower=0).pow(2).rolling(12,min_periods=8).mean(); dn=(-r.clip(upper=0)).pow(2).rolling(12,min_periods=8).mean(); return ((up-dn)/(up+dn)).to_numpy()
    if name=="vv1_downside_vol_share":
        up=r.clip(lower=0).pow(2).rolling(12,min_periods=8).mean().pow(.5); dn=(-r.clip(upper=0)).pow(2).rolling(12,min_periods=8).mean().pow(.5); return (dn/(dn+up)).to_numpy()
    if name=="lf1_herfindahl_volume":
        shares=d["volume"].div(d["volume"].sum(axis=1),axis=0); h=(shares*shares).sum(axis=1); return np.repeat(h.to_numpy()[:,None],d["volume"].shape[1],axis=1)
    if name=="lf1_own_volume_share": return (d["volume"]/d["volume"].rolling(12,min_periods=8).sum()).to_numpy()
    if name=="ofi_volume_imbalance": return (d["signed_volume"].rolling(12,min_periods=8).sum()/d["signed_volume"].abs().rolling(12,min_periods=8).sum()).to_numpy()
    if name=="ashare_limit_asymmetry": return (d["up_event"].rolling(12,min_periods=1).mean()-d["down_event"].rolling(12,min_periods=1).mean()).to_numpy()
    return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--recipes",required=True); p.add_argument("--ledger",required=True); p.add_argument("--rows",type=int,default=64); p.add_argument("--assets",type=int,default=8); a=p.parse_args()
    if not 32<=a.rows<=96 or not 8<=a.assets<=16: raise ValueError("bounded rows/assets")
    core.campaign_protocol_hash=lambda:hashlib.sha256(Path(core.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    recipes=json.loads(Path(a.recipes).read_text())["recipes"]; assert len(recipes)==30
    inputs=fixture(a.rows,a.assets); cut=a.rows-9; changed=future(inputs,cut); edges=edge_fixtures(inputs); ledger=Path(a.ledger); load_all(); summary={}
    for rec in sorted(recipes,key=lambda r:r["canonical"]):
        name=rec["canonical"]; outs={}; edge_outs={}; fps={}
        for backend in ("pandas_numpy","polars"):
            op=OperatorRegistry.get(name,backend,mode="any")
            if op is None: core.emit(ledger,{"canonical":name,"backend":backend,"status":"UNTESTED_NO_SLOT","fixture_domain":DOMAIN}); continue
            fp=core.execution_fingerprint(name,backend,rec,inputs,op); fps[backend]=fp
            core.emit(ledger,{"canonical":name,"backend":backend,"status":"RUNNING","fingerprint":fp,"rows":a.rows,"assets":a.assets,"params":rec["params"],"fixture_domain":DOMAIN})
            try:
                roles=rec["inputs"]; actual=core.array(op.calculate(*[core.as_backend(inputs[r],backend) for r in roles],**rec["params"])); fut=core.array(op.calculate(*[core.as_backend(changed[r],backend) for r in roles],**rec["params"])); outs[backend]=actual
                edge_outs[backend]={label:core.array(op.calculate(*[core.as_backend(ed[r],backend) for r in roles],**rec["params"])) for label,ed in edges.items()}
                finite=int(np.isfinite(actual.astype(float)).sum()); prefix=bool(actual.shape==fut.shape and np.allclose(actual[:cut],fut[:cut],equal_nan=True,rtol=1e-9,atol=1e-11)); suffix=bool(not np.allclose(actual[cut:],fut[cut:],equal_nan=True,rtol=1e-9,atol=1e-11))
                exp=oracle(name,inputs); expf=oracle(name,changed); oe=None if exp is None else bool(np.allclose(actual,exp,equal_nan=True,rtol=1e-9,atol=1e-11) and np.allclose(fut,expf,equal_nan=True,rtol=1e-9,atol=1e-11))
                status,reason=core.execution_status(finite,prefix,None)
                if status=="EXECUTED_FINITE" and not suffix: status,reason="FAILED_EXECUTION_PENDING_TRIAGE","legal future perturbation did not change suffix"
                if status=="EXECUTED_FINITE" and oe is False: status,reason="FAILED_EXECUTION_PENDING_TRIAGE","independent oracle mismatch"
                core.emit(ledger,{"canonical":name,"backend":backend,"status":status,"reason":reason,"shape":list(actual.shape),"finite":finite,"future_prefix_invariant_all_inputs":prefix,"future_suffix_changed":suffix,"independent_oracle_equal":oe,"edge_constant_finite":int(np.isfinite(edge_outs[backend]["constant_price"].astype(float)).sum()),"edge_missing_finite":int(np.isfinite(edge_outs[backend]["single_missing_bar"].astype(float)).sum()),"fingerprint":fp,"params":rec["params"],"fixture_domain":DOMAIN})
            except Exception as e: core.emit(ledger,{"canonical":name,"backend":backend,"status":"FAILED_EXECUTION_PENDING_TRIAGE","reason":"triage required","error_type":type(e).__name__,"error":str(e)[:500],"fingerprint":fp,"params":rec["params"],"fixture_domain":DOMAIN})
            gc.collect()
        if set(outs)=={"pandas_numpy","polars"}:
            eq=bool(outs["pandas_numpy"].shape==outs["polars"].shape and np.allclose(outs["pandas_numpy"],outs["polars"],equal_nan=True,rtol=1e-9,atol=1e-11)); summary[name]=eq
            edge_eq={label:bool(np.allclose(edge_outs["pandas_numpy"][label],edge_outs["polars"][label],equal_nan=True,rtol=1e-9,atol=1e-11)) for label in edges}
            core.emit(ledger,{"canonical":name,"status":"CANONICAL_PARITY" if eq and all(edge_eq.values()) else "CANONICAL_PARITY_FAILED","pandas_polars_equal":eq,"edge_parity":edge_eq,"backend_fingerprints":fps,"fixture_domain":DOMAIN})
    print(json.dumps({"selected":sorted(summary),"parity_pass":sum(summary.values()),"count":len(summary),"rows":a.rows,"assets":a.assets},sort_keys=True))
if __name__=="__main__": main()
