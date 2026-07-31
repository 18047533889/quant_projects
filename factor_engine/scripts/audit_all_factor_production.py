#!/usr/bin/env python3
"""Audit every retained Factor DSL canonical against production invariants.

Runtime-only mode is the evidence bootstrap: real Pandas reference execution,
shape preservation, determinism and prefix causality.  Strict mode additionally
requires evidence-backed production admission.
"""
from __future__ import annotations

import argparse,inspect,sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

FE_ROOT=Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent),str(FE_ROOT)):
    if p not in sys.path:sys.path.insert(0,p)

PANEL_NAMES=frozenset({
"x","y","z","a","b","w","g","left","right","numerator","denominator","ret","returns","benchmark_ret","market_ret","benchmark","market","open","high","low","close","price","volume","amount","vwap","turnover","weight","weights","signal","fallback","condition","group","industry","sector","fiscal_quarter","period_id","quarter","revision_id","decision_time","available_time","available_at","exposure","exposures","control","controls","factor","target","mask","event","value","values","v1","v2","sort_col","float_shares","flow","balance","earnings","cashflow","assets","working_capital","base","dollar_volume","scale_base","fundamental_x","fundamental_y","fundamental_scale","current_assets","current_liabilities","inventory","total_debt","total_equity","short_debt","long_debt","cash","total_assets","operating_income","revenue","gross_profit","net_income","operating_cash_flow","research_development","capex","invested_capital","nopat","receivables","cost_of_goods_sold","interest_expense","market_cap",
})
SCALAR_VALUES:dict[str,Any]={
"window":20,"d":20,"n":3,"m":2,"span":20,"period":20,"periods":4,"lag":1,"lags":1,"k":3,"q":0.2,"quantile":0.2,"threshold":0.01,"run":2,"hump":0.02,"min_periods":5,"min_obs":8,"ddof":1,"ann_factor":252,"decimals":2,"to":1.0,"lower":-2.0,"upper":2.0,"eps":1e-8,"epsilon":1e-8,"alpha":0.2,
"fast":12,"slow":26,"fast_period":12,"slow_period":26,"fast_window":12,"slow_window":26,"signal_span":9,"signal_window":9,"signal_period":9,"side":"lower","order":"largest","add_intercept":True,"clip":3.0,"limit":3,"max_gap":3,"max_periods":8,"max_lookback":60,"power":2.0,"exponent":2.0,"p":0.5,"c":1.0,"fraction":0.5,"buckets":5,"top":3,"asc":True,"annualization":252,"annualization_factor":252,"periods_per_year":4,"method":"std","interpolation":"linear","center":True,"ascending":True,"inclusive":True,"offset":0,"require_consecutive":True,"trim_pct":0.1,"sign_policy":"strict","denominator_policy":"signed","aggr_func":"sum","lo":-2.0,"hi":2.0,
"left_window":3,"right_window":3,"history_window":60,"points":3,"std_dev":2.0,"skipna":True,"revision_policy":"latest_available","missing_group_policy":"raise","short_window":7,"medium_window":14,"long_window":28,"ema_window":20,"atr_window":14,"tenkan_window":9,"kijun_window":26,"senkou_b_window":52,"er_window":10,"vol_window":20,"baseline_window":40,"price_window":20,"volume_window":20,"turnover_window":20,"adl_window":60,"impulse_window":20,"flag_window":10,"growth_periods":4,"compare_periods":1,"window_periods":8,"average_periods":2,"short_periods":1,"long_periods":4,"window_days":252,"max_days":504,"body_window":10,"shadow_window":10,
"tolerance":0.03,"min_depth":0.05,"min_spacing":5,"max_spacing":40,"shoulder_tolerance":0.05,"head_min_prominence":0.05,"max_neckline_slope":0.02,"slope_threshold":0.001,"parallel_tolerance":0.001,"min_impulse":0.08,"max_retracement":0.5,"max_width":0.12,"volume_decay_threshold":0.0,"multiplier":2.0,"volume_scale":1_000_000.0,"acceleration":0.02,"maximum":0.2,"penetration":0.3,"short_weight":4.0,"medium_weight":2.0,"long_weight":1.0,
"pattern":"high_wave","body_factor":1.0,"shadow_factor":1.0,
}
SPECIAL_SCALARS={
("cs_rank_gaussian","method"):"blom",("cs_regression","mode"):0,("cs_quantile","p"):0.5,("group_percentile","p"):0.5,("group_percentile","side"):"top",("group_percentile","missing_group_policy"):"raise",("revision_delta","mode"):"absolute",("period_change","mode"):"absolute",("period_lag","revision_policy"):"latest_available",("period_stability","method"):"std",("ts_nth_value","order"):"largest",("ts_mad","scale"):1.0,("ts_product","skipna"):True,("MACD_line","signal"):9,("MACD_signal","signal"):9,("MACD_hist","signal"):9,("fillna_const","value"):0.0,("group_winsorize","a"):0.05,("winsorize","lower"):0.05,("winsorize","upper"):0.95,
}
SPECIAL_POSITIONAL={"cs_multi_resid":("target","exposure","control"),"cs_neutralize":("target","exposure","control")}
SPECIAL_KWARGS={"cs_multi_resid":{"add_intercept":True,"min_obs":8},"cs_neutralize":{"add_intercept":True,"min_obs":8}}


def _panels(rows:int=160,cols:int=6)->dict[str,pd.DataFrame]:
    dates=pd.date_range("2020-01-01",periods=rows,freq="B");assets=[f"A{i}" for i in range(cols)];t=np.arange(rows,dtype=float)[:,None];j=np.arange(cols,dtype=float)[None,:]
    close=pd.DataFrame(50+0.15*t+0.7*j+np.sin(t/5+j/3),index=dates,columns=assets);open_=close*(1+0.002*np.cos(t/4+j));high=pd.DataFrame(np.maximum(open_,close)*1.01,index=dates,columns=assets);low=pd.DataFrame(np.minimum(open_,close)*0.99,index=dates,columns=assets);volume=pd.DataFrame(1_000_000+5000*t+10000*j,index=dates,columns=assets);amount=volume*close;ret=close.pct_change().fillna(0);market=pd.DataFrame(np.repeat(ret.mean(axis=1).to_numpy()[:,None],cols,axis=1),index=dates,columns=assets)
    group=pd.DataFrame(np.tile(np.array(["G0","G1","G2","G0","G1","G2"],object)[:cols],(rows,1)),index=dates,columns=assets);condition=volume.gt(volume.rolling(5,min_periods=1).mean())
    # One visible report period per ~10 business days. Values can revise inside a
    # period because numeric panels change daily; this intentionally exercises
    # revision-aware period transforms.
    period=pd.DataFrame(np.repeat((np.arange(rows)//10)[:,None],cols,axis=1),index=dates,columns=assets);quarter=pd.DataFrame(np.repeat((((np.arange(rows)//10)%4)+1)[:,None],cols,axis=1),index=dates,columns=assets);revision=pd.DataFrame(np.repeat((np.arange(rows)//5)[:,None],cols,axis=1),index=dates,columns=assets);decision=pd.DataFrame(np.repeat(dates.to_numpy()[:,None],cols,axis=1),index=dates,columns=assets);available=decision-pd.Timedelta(days=3);weights=volume.div(volume.sum(axis=1),axis=0);control=volume.pct_change().fillna(0);zero=pd.DataFrame(0.0,index=dates,columns=assets);float_shares=pd.DataFrame(np.repeat((100_000_000+np.arange(rows,dtype=float)[:,None]*10_000),cols,axis=1),index=dates,columns=assets);turnover=volume/float_shares
    panels={
        "x":close,"y":open_,"z":control,"a":close,"b":open_,"w":weights,"g":group,"left":close,"right":open_,"numerator":close,"denominator":open_.abs()+1,"ret":ret,"returns":ret,"benchmark_ret":market,"market_ret":market,"benchmark":market,"market":market,"open":open_,"high":high,"low":low,"close":close,"price":(high+low+close)/3,"volume":volume,"amount":amount,"vwap":amount/volume,"turnover":turnover,"weight":weights,"weights":weights,"signal":ret,"fallback":zero,"condition":condition,"mask":condition,"event":condition,"group":group,"industry":group,"sector":group,"fiscal_quarter":quarter,"period_id":period,"quarter":quarter,"revision_id":revision,"decision_time":decision,"available_time":available,"available_at":available,"exposure":market,"exposures":market,"control":control,"controls":control,"factor":market,"target":ret,"sort_col":volume,"value":close,"values":close,"v1":close,"v2":open_,"float_shares":float_shares,"flow":close,"balance":open_.abs()+10,"earnings":close,"cashflow":open_,"assets":close.abs()+100,"working_capital":close,"base":close.abs()+100,"dollar_volume":amount,"scale_base":close.abs()+100,
    }
    for name in PANEL_NAMES:
        if name not in panels and name not in {"period_id","group","industry","sector"}:panels[name]=close*(1+0.001*(len(panels)%17))+10
    return panels

def _value(canonical,name,panels):
    key=str(name)
    if (canonical,key) in SPECIAL_SCALARS:return SPECIAL_SCALARS[(canonical,key)]
    if key in panels:return panels[key]
    if key in SCALAR_VALUES:return SCALAR_VALUES[key]
    if key.startswith("window") or key.endswith("_window"):return 20
    if key.endswith("_periods"):return 4
    if key.endswith("_days"):return 60
    if key.endswith("_id"):return panels["period_id"]
    if key in PANEL_NAMES:return panels["x"]
    raise KeyError(f"unclassified public parameter {canonical}.{key}")
def _build_call(canonical,op,panels):
    from cleaned_operators.registry import OperatorRegistry
    names=tuple(str(x) for x in (OperatorRegistry._catalog.get(canonical,{}).get("param_names") or ())) or ("x",)
    if canonical in SPECIAL_POSITIONAL:return [_value(canonical,n,panels) for n in SPECIAL_POSITIONAL[canonical]],dict(SPECIAL_KWARGS.get(canonical,{}) or {})
    return [_value(canonical,n,panels) for n in names if n!="..."],dict(SPECIAL_KWARGS.get(canonical,{}) or {})
def _slice(v,rows):return v.iloc[:rows] if isinstance(v,(pd.DataFrame,pd.Series)) else v
def _to_frame(v,t):
    if isinstance(v,pd.DataFrame):return v
    if isinstance(v,pd.Series):
        if v.index.equals(t.index):return pd.DataFrame(np.repeat(v.to_numpy()[:,None],t.shape[1],axis=1),index=t.index,columns=t.columns)
        raise TypeError("Series result does not share time index")
    arr=np.asarray(v)
    if arr.shape==t.shape:return pd.DataFrame(arr,index=t.index,columns=t.columns)
    if np.isscalar(v):return pd.DataFrame(v,index=t.index,columns=t.columns)
    raise TypeError(f"unsupported result shape/type {type(v).__name__} {arr.shape}")
def _equal(a,b):
    if a.shape!=b.shape or not a.index.equals(b.index) or not a.columns.equals(b.columns):return False
    try:return bool(np.allclose(a.to_numpy(float),b.to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-8))
    except (TypeError,ValueError):return a.astype(object).where(pd.notna(a),None).equals(b.astype(object).where(pd.notna(b),None))
def _delta(a,b):
    try:
        av,bv=a.to_numpy(float),b.to_numpy(float);m=np.isfinite(av)&np.isfinite(bv);return 0.0 if not m.any() else float(np.max(np.abs(av[m]-bv[m])))
    except Exception:return None
def _impl(op):
    cls=type(op);source=inspect.getsourcefile(cls)
    try:source=str(Path(source).resolve().relative_to(FE_ROOT)) if source else "?"
    except Exception:source=str(source or "?")
    return f"{cls.__module__}.{cls.__name__}@{source}"

def audit(*,require_admission:bool=True)->list[str]:
    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.registry import OperatorRegistry
    from backend.operator_capability import production_eligible_backends
    load_all();errors=[];panels=_panels();template=panels["x"];prefix_rows=110
    for canonical in sorted(factor_production_targets()):
        op=OperatorRegistry.get(canonical,"pandas_numpy")
        if op is None:errors.append(f"{canonical}: no pandas semantic reference");continue
        catalog=OperatorRegistry._catalog.get(canonical,{});policy=infer_operator_policy(op,canonical=canonical)
        if str(catalog.get("status"))!="production":errors.append(f"{canonical}: lifecycle status is not production")
        if not policy.pit_safe or policy.lag<0:errors.append(f"{canonical}: PIT policy is not causal")
        if not policy.shape_preserving:errors.append(f"{canonical}: shape_preserving=False")
        if require_admission:
            spec=build_operator_spec(canonical)
            if spec is None or not spec.allow_in_production:errors.append(f"{canonical}: OperatorSpec.allow_in_production=False");continue
            if not production_eligible_backends(canonical):errors.append(f"{canonical}: no production eligible backend");continue
        try:
            args,kwargs=_build_call(canonical,op,panels);r1=_to_frame(op.calculate(*args,**kwargs),template);r2=_to_frame(op.calculate(*args,**kwargs),template)
            if not r1.index.equals(template.index) or not r1.columns.equals(template.columns):errors.append(f"{canonical}: output axes changed [{_impl(op)}]");continue
            if not _equal(r1,r2):errors.append(f"{canonical}: non-deterministic repeated evaluation [{_impl(op)}]");continue
            p_args=[_slice(x,prefix_rows) for x in args];p_kwargs={k:_slice(v,prefix_rows) for k,v in kwargs.items()};p_template=template.iloc[:prefix_rows];prefix=_to_frame(op.calculate(*p_args,**p_kwargs),p_template);historical=r1.iloc[:prefix_rows]
            if not _equal(historical,prefix):
                d=_delta(historical,prefix);errors.append(f"{canonical}: prefix invariance / causality violation"+(f" (max_abs_delta={d:.6g})" if d is not None else "")+f" [{_impl(op)}]")
        except Exception as exc:errors.append(f"{canonical}: {type(exc).__name__}: {exc} [{_impl(op)}]")
    return errors

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--runtime-only",action="store_true");args=parser.parse_args();errors=audit(require_admission=not args.runtime_only)
    if errors:
        print(f"factor production audit FAILED ({len(errors)} issues)",file=sys.stderr)
        for error in errors:print(f"- {error}",file=sys.stderr)
        return 1
    from cleaned_operators.production_hardening import factor_production_targets
    print(f"factor production {'runtime' if args.runtime_only else 'admission'} audit passed ({len(factor_production_targets())} canonicals)");return 0
if __name__=="__main__":raise SystemExit(main())
