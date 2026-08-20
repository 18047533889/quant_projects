# -*- coding: utf-8 -*-
"""Additional causal chart patterns with explicit bounded windows."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata,SeriesOperator,register_operator
from cleaned_operators.price_volume.structure_patterns_v2 import ts_nth_pivot_high,ts_nth_pivot_low,ts_pivot_high_spacing,ts_pivot_low_spacing,ts_consolidation_width,ts_consolidation_volume_decay,_seq_features,_between

def _pi(v,n,m=1):
    if isinstance(v,bool):raise ValueError(f"{n} must be integer")
    v=int(v)
    if v<m:raise ValueError(f"{n} must be >= {m}")
    return v
def _pf(v,n,lo=0):
    v=float(v)
    if not np.isfinite(v) or v<lo:raise ValueError(f"{n} invalid")
    return v
def _register(name,params,fn,desc):
    meta=OperatorMetadata(name=name,category="chart_pattern",description=desc,param_names=list(params),return_type="series",tags=["pit_safe","causal","bounded_history","production_extension"])
    def calc(self,*a,**k):return fn(*a,**k)
    cls=type(f"ChartExtra_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":calc,"__module__":__name__})
    register_operator(name=name,category="chart_pattern",business_category="technical_structure",canonical=name,source="structure_patterns_extra_v2",backend="pandas_numpy",status="production")(cls)
def _similar(values,tol):
    stack=np.stack([v.to_numpy(float) for v in values])
    valid=np.isfinite(stack)
    count=valid.sum(axis=0)
    mx=np.max(np.where(valid,stack,-np.inf),axis=0)
    mn=np.min(np.where(valid,stack,np.inf),axis=0)
    abs_sum=np.where(valid,np.abs(stack),0.0).sum(axis=0)
    mid=np.divide(abs_sum,count,out=np.full(count.shape,np.nan,dtype=float),where=count>0)
    complete=count==len(values)
    denom=mid*float(tol)
    ratio=np.full(count.shape,np.nan,dtype=float)
    np.divide(mx-mn,denom,out=ratio,where=complete&(denom>0))
    score=np.where(complete,np.clip(1-ratio,0,1),np.nan)
    return pd.DataFrame(score,index=values[0].index,columns=values[0].columns)
def pattern_triple_top(high,low,left_window,right_window,history_window,tolerance,min_depth,min_spacing,max_spacing):
    tol=_pf(tolerance,"tolerance",1e-12)
    ok,prices,positions=_seq_features(high,low,left_window,right_window,history_window,5,(True,False,True,False,True))
    hs=[prices[0],prices[2],prices[4]];trough=prices[3]
    depth=((sum(hs)/3)/trough.replace(0,np.nan)-1).ge(_pf(min_depth,"min_depth",0))
    spacing=positions[4]-positions[2]
    return (_similar(hs,tol)*depth.astype(float)*_between(spacing,_pi(min_spacing,"min_spacing"),_pi(max_spacing,"max_spacing")))*ok.astype(float)
def pattern_triple_bottom(high,low,left_window,right_window,history_window,tolerance,min_depth,min_spacing,max_spacing):
    tol=_pf(tolerance,"tolerance",1e-12)
    ok,prices,positions=_seq_features(high,low,left_window,right_window,history_window,5,(False,True,False,True,False))
    ls=[prices[0],prices[2],prices[4]];peak=prices[3]
    depth=(peak/(sum(ls)/3).replace(0,np.nan)-1).ge(_pf(min_depth,"min_depth",0))
    spacing=positions[4]-positions[2]
    return (_similar(ls,tol)*depth.astype(float)*_between(spacing,_pi(min_spacing,"min_spacing"),_pi(max_spacing,"max_spacing")))*ok.astype(float)
def pattern_123_bull(high,low,left_window,right_window,history_window,min_swing):
    ok,prices,_=_seq_features(high,low,left_window,right_window,history_window,3,(False,True,False))
    l2,h1,l1=prices[0],prices[1],prices[2]
    return ((l1>l2)&((h1/l1.replace(0,np.nan)-1)>=_pf(min_swing,"min_swing",0))).astype(float)*ok.astype(float)
def pattern_123_bear(high,low,left_window,right_window,history_window,min_swing):
    ok,prices,_=_seq_features(high,low,left_window,right_window,history_window,3,(True,False,True))
    h2,l1,h1=prices[0],prices[1],prices[2]
    return ((h1<h2)&((h1/l1.replace(0,np.nan)-1)>=_pf(min_swing,"min_swing",0))).astype(float)*ok.astype(float)
def _quadratic_score(close,window,up=True):
    w=_pi(window,"window",5);x=np.linspace(-1,1,w);out=pd.DataFrame(np.nan,index=close.index,columns=close.columns)
    for col in close.columns:
        arr=close[col].to_numpy(float);res=np.full(len(arr),np.nan)
        for t in range(w-1,len(arr)):
            y=arr[t-w+1:t+1]
            if not np.isfinite(y).all() or np.nanmean(np.abs(y))==0:continue
            yn=y/np.nanmean(np.abs(y));coef=np.polyfit(x,yn,2);fit=np.polyval(coef,x);tot=np.sum((yn-yn.mean())**2);r2=1-np.sum((yn-fit)**2)/tot if tot>1e-12 else 0
            curvature=coef[0] if up else -coef[0];res[t]=max(0.0,curvature)*max(0.0,r2)
        out[col]=res
    return out
def pattern_rounding_bottom(close,window,min_fit):return _quadratic_score(close,window,True).where(_quadratic_score(close,window,True)>=_pf(min_fit,"min_fit",0),0.0)
def pattern_rounding_top(close,window,min_fit):return _quadratic_score(close,window,False).where(_quadratic_score(close,window,False)>=_pf(min_fit,"min_fit",0),0.0)
def pattern_cup(close,window,min_depth,max_edge_diff,min_fit):
    w=_pi(window,"window",5);score=_quadratic_score(close,w,True);left=close.shift(w-1);edge=(close-left).abs()/((close.abs()+left.abs())/2).replace(0,np.nan);center=close.shift(w//2);depth=((left+close)/2/center.replace(0,np.nan)-1);return score*edge.le(_pf(max_edge_diff,"max_edge_diff",0)).astype(float)*depth.ge(_pf(min_depth,"min_depth",0)).astype(float)*score.ge(_pf(min_fit,"min_fit",0)).astype(float)
def pattern_cup_handle(close,high,low,cup_window,handle_window,min_depth,max_edge_diff,min_fit,max_handle_retracement):
    cw=_pi(cup_window,"cup_window",5);hw=_pi(handle_window,"handle_window",2);cup=pattern_cup(close.shift(hw),cw,min_depth,max_edge_diff,min_fit);base=close.shift(hw);trough=low.rolling(hw,min_periods=hw).min();retr=(base-trough)/base.abs().replace(0,np.nan);return cup*retr.le(_pf(max_handle_retracement,"max_handle_retracement",0)).astype(float)
def _pennant(close,high,low,volume,impulse_window,pennant_window,min_impulse,max_width,volume_decay_threshold,bull):
    iw=_pi(impulse_window,"impulse_window",2);pw=_pi(pennant_window,"pennant_window",3);imp=close.shift(pw)/close.shift(pw+iw)-1;width=ts_consolidation_width(high,low,pw)/close.abs().replace(0,np.nan);v=ts_consolidation_volume_decay(volume,pw);range_now=(high-low).rolling(max(2,pw//2),min_periods=max(2,pw//2)).mean();range_old=(high-low).shift(max(2,pw//2)).rolling(max(2,pw//2),min_periods=max(2,pw//2)).mean();compress=(range_now<range_old);direction=imp.ge(_pf(min_impulse,"min_impulse",0)) if bull else imp.le(-_pf(min_impulse,"min_impulse",0));return (direction&width.le(_pf(max_width,"max_width",0))&v.ge(_pf(volume_decay_threshold,"volume_decay_threshold"))&compress).astype(float)
def pattern_bull_pennant(close,high,low,volume,impulse_window,pennant_window,min_impulse,max_width,volume_decay_threshold):return _pennant(close,high,low,volume,impulse_window,pennant_window,min_impulse,max_width,volume_decay_threshold,True)
def pattern_bear_pennant(close,high,low,volume,impulse_window,pennant_window,min_impulse,max_width,volume_decay_threshold):return _pennant(close,high,low,volume,impulse_window,pennant_window,min_impulse,max_width,volume_decay_threshold,False)
def _retest(close,window,max_wait,tolerance,break_up):
    w=_pi(window,"window",2);wait=_pi(max_wait,"max_wait");tol=_pf(tolerance,"tolerance",0);level=(close.shift(1).rolling(w,min_periods=w).max() if break_up else close.shift(1).rolling(w,min_periods=w).min());event=(close>level if break_up else close<level);out=pd.DataFrame(0.0,index=close.index,columns=close.columns)
    for col in close.columns:
        last_level=np.nan;age=wait+1
        for t in range(len(close)):
            if bool(event.iloc[t][col]) and np.isfinite(level.iloc[t][col]):last_level=float(level.iloc[t][col]);age=0;continue
            age+=1
            if age<=wait and np.isfinite(last_level):
                px=float(close.iloc[t][col]);rel=px/last_level-1 if last_level else np.nan
                ok=(-tol<=rel<=tol) and ((px>=last_level*(1-tol)) if break_up else (px<=last_level*(1+tol)))
                out.iat[t,out.columns.get_loc(col)]=1.0 if ok else 0.0
    return out
def pattern_breakout_retest(close,window,max_wait,tolerance):return _retest(close,window,max_wait,tolerance,True)
def pattern_breakdown_retest(close,window,max_wait,tolerance):return _retest(close,window,max_wait,tolerance,False)

_SPECS=[
("pattern_triple_top",["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"],pattern_triple_top),
("pattern_triple_bottom",["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"],pattern_triple_bottom),
("pattern_123_bull",["high","low","left_window","right_window","history_window","min_swing"],pattern_123_bull),
("pattern_123_bear",["high","low","left_window","right_window","history_window","min_swing"],pattern_123_bear),
("pattern_rounding_bottom",["close","window","min_fit"],pattern_rounding_bottom),
("pattern_rounding_top",["close","window","min_fit"],pattern_rounding_top),
("pattern_cup",["close","window","min_depth","max_edge_diff","min_fit"],pattern_cup),
("pattern_cup_handle",["close","high","low","cup_window","handle_window","min_depth","max_edge_diff","min_fit","max_handle_retracement"],pattern_cup_handle),
("pattern_bull_pennant",["close","high","low","volume","impulse_window","pennant_window","min_impulse","max_width","volume_decay_threshold"],pattern_bull_pennant),
("pattern_bear_pennant",["close","high","low","volume","impulse_window","pennant_window","min_impulse","max_width","volume_decay_threshold"],pattern_bear_pennant),
("pattern_breakout_retest",["close","window","max_wait","tolerance"],pattern_breakout_retest),
("pattern_breakdown_retest",["close","window","max_wait","tolerance"],pattern_breakdown_retest),
]
for n,p,f in _SPECS:_register(n,p,f,n.replace("_"," "))
import cleaned_operators.operator_surface as _surface
_surface.extend_extended_only({x[0] for x in _SPECS})
