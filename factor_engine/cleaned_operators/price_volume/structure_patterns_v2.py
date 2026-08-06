# -*- coding: utf-8 -*-
"""Causal chart-structure primitives and interpretable pattern scores.

Patterns are built from *confirmed* pivots.  A pivot discovered at t-right is
only available at its confirmation timestamp, never written back to the pivot
bar.  All searches are bounded by ``history_window``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.price_volume.technical_structure_repairs import _events, _bounded_line

_EPS=1e-12


def _pi(v,name,minimum=1):
    if isinstance(v,bool): raise ValueError(f"{name} must be integer")
    v=int(v)
    if v<minimum: raise ValueError(f"{name} must be >= {minimum}")
    return v


def _pf(v,name,minimum=None):
    v=float(v)
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    if minimum is not None and v<minimum: raise ValueError(f"{name} must be >= {minimum}")
    return v


def _register(name,params,fn,desc,*,category="price_structure",tags=()):
    meta=OperatorMetadata(name=name,category=category,description=desc,param_names=list(params),return_type="series",
        tags=["pit_safe","causal","bounded_history","production_extension",*tags])
    def _calculate_series(self,*args,**kwargs): return fn(*args,**kwargs)
    cls=type(f"StructureV2_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category=category,business_category="technical_structure",canonical=name,
        source="structure_patterns_v2",backend="pandas_numpy",status="production")(cls)


def _recent_events(frame,left,right,history,*,high):
    left,right,history=_pi(left,"left_window"),_pi(right,"right_window"),_pi(history,"history_window")
    prices,positions=_events(frame,left,right,high=high)
    rows,cols=prices.shape
    out=[[[] for _ in range(cols)] for _ in range(rows)]
    active=[[] for _ in range(cols)]
    for t in range(rows):
        cutoff=t-history+1
        for c in range(cols):
            active[c]=[e for e in active[c] if e[0]>=cutoff]
            if np.isfinite(prices[t,c]) and np.isfinite(positions[t,c]):
                active[c].append((t,float(positions[t,c]),float(prices[t,c])))
            out[t][c]=list(active[c])
    return out


def _event_stat(frame,left,right,history,n,*,high,which):
    n=_pi(n,"n")
    events=_recent_events(frame,left,right,history,high=high)
    arr=np.full(frame.shape,np.nan,float)
    for t,row in enumerate(events):
        for c,ev in enumerate(row):
            if len(ev)<n: continue
            confirm,pivot,price=ev[-n]
            if which=="price": arr[t,c]=price
            elif which=="pivot_age": arr[t,c]=t-pivot
            elif which=="confirmation_age": arr[t,c]=t-confirm
    return pd.DataFrame(arr,index=frame.index,columns=frame.columns)


def ts_nth_pivot_high(high,left_window,right_window,history_window,n): return _event_stat(high,left_window,right_window,history_window,n,high=True,which="price")
def ts_nth_pivot_low(low,left_window,right_window,history_window,n): return _event_stat(low,left_window,right_window,history_window,n,high=False,which="price")
def ts_nth_pivot_high_age(high,left_window,right_window,history_window,n): return _event_stat(high,left_window,right_window,history_window,n,high=True,which="pivot_age")
def ts_nth_pivot_low_age(low,left_window,right_window,history_window,n): return _event_stat(low,left_window,right_window,history_window,n,high=False,which="pivot_age")


def _event_count(frame,left,right,history,high):
    events=_recent_events(frame,left,right,history,high=high)
    return pd.DataFrame([[float(len(x)) for x in row] for row in events],index=frame.index,columns=frame.columns)

def ts_pivot_high_count(high,left_window,right_window,history_window): return _event_count(high,left_window,right_window,history_window,True)
def ts_pivot_low_count(low,left_window,right_window,history_window): return _event_count(low,left_window,right_window,history_window,False)


def _spacing(frame,left,right,history,high):
    events=_recent_events(frame,left,right,history,high=high); arr=np.full(frame.shape,np.nan,float)
    for t,row in enumerate(events):
        for c,ev in enumerate(row):
            if len(ev)>=2: arr[t,c]=ev[-1][1]-ev[-2][1]
    return pd.DataFrame(arr,index=frame.index,columns=frame.columns)
def ts_pivot_high_spacing(high,left_window,right_window,history_window): return _spacing(high,left_window,right_window,history_window,True)
def ts_pivot_low_spacing(low,left_window,right_window,history_window): return _spacing(low,left_window,right_window,history_window,False)


def ts_swing_amplitude(high,low,left_window,right_window,history_window):
    h=ts_nth_pivot_high(high,left_window,right_window,history_window,1)
    l=ts_nth_pivot_low(low,left_window,right_window,history_window,1)
    return (h-l).abs()

def ts_swing_amplitude_pct(high,low,close,left_window,right_window,history_window):
    return ts_swing_amplitude(high,low,left_window,right_window,history_window)/close.abs().replace(0,np.nan)

def ts_swing_duration(high,low,left_window,right_window,history_window):
    ha=ts_nth_pivot_high_age(high,left_window,right_window,history_window,1)
    la=ts_nth_pivot_low_age(low,left_window,right_window,history_window,1)
    return (ha-la).abs()

def ts_swing_velocity(high,low,left_window,right_window,history_window):
    amp=ts_swing_amplitude(high,low,left_window,right_window,history_window)
    dur=ts_swing_duration(high,low,left_window,right_window,history_window)
    return amp/dur.replace(0,np.nan)


def _true_range(high,low,close):
    prev=close.shift(1)
    return pd.DataFrame(np.maximum.reduce([(high-low).to_numpy(float),(high-prev).abs().to_numpy(float),(low-prev).abs().to_numpy(float)]),index=high.index,columns=high.columns)

def ts_swing_amplitude_atr(high,low,close,left_window,right_window,history_window,atr_window):
    atr=_true_range(high,low,close).rolling(_pi(atr_window,"atr_window"),min_periods=_pi(atr_window,"atr_window")).mean()
    return ts_swing_amplitude(high,low,left_window,right_window,history_window)/atr.replace(0,np.nan)


def ts_channel_width(high,low,left_window,right_window,history_window,points):
    r=_bounded_line(high,left_window,right_window,history_window,points,high=True,output="level")
    s=_bounded_line(low,left_window,right_window,history_window,points,high=False,output="level")
    return r-s

def ts_channel_width_pct(close,high,low,left_window,right_window,history_window,points):
    return ts_channel_width(high,low,left_window,right_window,history_window,points)/close.abs().replace(0,np.nan)

def ts_channel_width_atr(high,low,close,left_window,right_window,history_window,points,atr_window):
    atr=_true_range(high,low,close).rolling(_pi(atr_window,"atr_window"),min_periods=_pi(atr_window,"atr_window")).mean()
    return ts_channel_width(high,low,left_window,right_window,history_window,points)/atr.replace(0,np.nan)

def ts_channel_width_slope(high,low,left_window,right_window,history_window,points,window):
    width=ts_channel_width(high,low,left_window,right_window,history_window,points)
    w=_pi(window,"window",2); x=np.arange(w,dtype=float); xb=x.mean(); den=np.sum((x-xb)**2)
    return width.rolling(w,min_periods=w).apply(lambda a: float(np.sum((x-xb)*(a-a.mean()))/den),raw=True)

def ts_line_convergence(high,low,left_window,right_window,history_window,points):
    rs=_bounded_line(high,left_window,right_window,history_window,points,high=True,output="slope")
    ss=_bounded_line(low,left_window,right_window,history_window,points,high=False,output="slope")
    return ss-rs

def ts_line_parallelism(high,low,left_window,right_window,history_window,points):
    rs=_bounded_line(high,left_window,right_window,history_window,points,high=True,output="slope")
    ss=_bounded_line(low,left_window,right_window,history_window,points,high=False,output="slope")
    return -(rs-ss).abs()


def _fit_r2(frame,left,right,history,points,high):
    k=_pi(points,"points",2); events=_recent_events(frame,left,right,history,high=high); arr=np.full(frame.shape,np.nan,float)
    for t,row in enumerate(events):
        for c,ev in enumerate(row):
            if len(ev)<k: continue
            selected=ev[-k:]; x=np.asarray([z[1] for z in selected],float); y=np.asarray([z[2] for z in selected],float)
            xb,yb=x.mean(),y.mean(); den=np.sum((x-xb)**2)
            if den<=_EPS: continue
            slope=np.sum((x-xb)*(y-yb))/den; fit=yb+slope*(x-xb)
            ss_tot=np.sum((y-yb)**2); ss_res=np.sum((y-fit)**2)
            arr[t,c]=1.0-ss_res/ss_tot if ss_tot>_EPS else 1.0
    return pd.DataFrame(arr,index=frame.index,columns=frame.columns)
def ts_resistance_fit_r2(high,left_window,right_window,history_window,points): return _fit_r2(high,left_window,right_window,history_window,points,True)
def ts_support_fit_r2(low,left_window,right_window,history_window,points): return _fit_r2(low,left_window,right_window,history_window,points,False)


def ts_pattern_symmetry(high,low,left_window,right_window,history_window):
    h1=ts_nth_pivot_high(high,left_window,right_window,history_window,1); h2=ts_nth_pivot_high(high,left_window,right_window,history_window,2)
    l1=ts_nth_pivot_low(low,left_window,right_window,history_window,1); l2=ts_nth_pivot_low(low,left_window,right_window,history_window,2)
    hs=(h1-h2).abs()/((h1.abs()+h2.abs())/2).replace(0,np.nan)
    ls=(l1-l2).abs()/((l1.abs()+l2.abs())/2).replace(0,np.nan)
    return 1.0/(1.0+hs+ls)


def ts_impulse_return(close,window): return close/close.shift(_pi(window,"window"))-1.0
def ts_impulse_strength(close,window,vol_window):
    ret=ts_impulse_return(close,window); rv=close.pct_change(fill_method=None).rolling(_pi(vol_window,"vol_window",2),min_periods=_pi(vol_window,"vol_window",2)).std()
    return ret/(rv*np.sqrt(_pi(window,"window"))).replace(0,np.nan)
def ts_impulse_volume(volume,window,baseline_window):
    w=_pi(window,"window"); b=_pi(baseline_window,"baseline_window")
    recent=volume.rolling(w,min_periods=w).mean(); base=volume.shift(w).rolling(b,min_periods=b).mean()
    return recent/base.replace(0,np.nan)-1.0

def ts_consolidation_width(high,low,window):
    w=_pi(window,"window",2); return high.rolling(w,min_periods=w).max()-low.rolling(w,min_periods=w).min()
def ts_consolidation_slope(close,window):
    w=_pi(window,"window",2); x=np.arange(w,dtype=float); xb=x.mean(); den=np.sum((x-xb)**2)
    return close.rolling(w,min_periods=w).apply(lambda a: float(np.sum((x-xb)*(a-a.mean()))/den),raw=True)
def ts_consolidation_volume_decay(volume,window): return -ts_consolidation_slope(volume,window)


def _closeness(a,b,tol):
    tol=_pf(tol,"tolerance",0.0)
    if tol<=0: raise ValueError("tolerance must be > 0")
    rel=(a-b).abs()/((a.abs()+b.abs())/2).replace(0,np.nan)
    return (1.0-rel/tol).clip(lower=0.0,upper=1.0)

def _between(v,lo,hi): return v.ge(lo).astype(float)*v.le(hi).astype(float)


def pattern_double_top(high,low,left_window,right_window,history_window,tolerance,min_depth,min_spacing,max_spacing):
    h1=ts_nth_pivot_high(high,left_window,right_window,history_window,1); h2=ts_nth_pivot_high(high,left_window,right_window,history_window,2)
    l1=ts_nth_pivot_low(low,left_window,right_window,history_window,1)
    depth=((h1+l1*0)/l1.replace(0,np.nan)-1.0).clip(lower=0.0)
    spacing=ts_pivot_high_spacing(high,left_window,right_window,history_window)
    return _closeness(h1,h2,tolerance)*depth.ge(_pf(min_depth,"min_depth",0)).astype(float)*_between(spacing,_pi(min_spacing,"min_spacing"),_pi(max_spacing,"max_spacing"))

def pattern_double_bottom(high,low,left_window,right_window,history_window,tolerance,min_depth,min_spacing,max_spacing):
    l1=ts_nth_pivot_low(low,left_window,right_window,history_window,1); l2=ts_nth_pivot_low(low,left_window,right_window,history_window,2)
    h1=ts_nth_pivot_high(high,left_window,right_window,history_window,1)
    depth=(h1/l1.replace(0,np.nan)-1.0).clip(lower=0.0); spacing=ts_pivot_low_spacing(low,left_window,right_window,history_window)
    return _closeness(l1,l2,tolerance)*depth.ge(_pf(min_depth,"min_depth",0)).astype(float)*_between(spacing,_pi(min_spacing,"min_spacing"),_pi(max_spacing,"max_spacing"))


def pattern_head_shoulders(high,low,left_window,right_window,history_window,shoulder_tolerance,head_min_prominence,max_neckline_slope):
    r=ts_nth_pivot_high(high,left_window,right_window,history_window,1); h=ts_nth_pivot_high(high,left_window,right_window,history_window,2); l=ts_nth_pivot_high(high,left_window,right_window,history_window,3)
    shoulders=_closeness(l,r,shoulder_tolerance); base=((l.abs()+r.abs())/2).replace(0,np.nan); prom=(h/base-1.0)
    neckline=_bounded_line(low,left_window,right_window,history_window,2,high=False,output="slope").abs()
    return shoulders*prom.ge(_pf(head_min_prominence,"head_min_prominence",0)).astype(float)*neckline.le(_pf(max_neckline_slope,"max_neckline_slope",0)).astype(float)

def pattern_inverse_head_shoulders(high,low,left_window,right_window,history_window,shoulder_tolerance,head_min_prominence,max_neckline_slope):
    r=ts_nth_pivot_low(low,left_window,right_window,history_window,1); h=ts_nth_pivot_low(low,left_window,right_window,history_window,2); l=ts_nth_pivot_low(low,left_window,right_window,history_window,3)
    shoulders=_closeness(l,r,shoulder_tolerance); base=((l.abs()+r.abs())/2).replace(0,np.nan); prom=(base/h.abs().replace(0,np.nan)-1.0)
    neckline=_bounded_line(high,left_window,right_window,history_window,2,high=True,output="slope").abs()
    return shoulders*prom.ge(_pf(head_min_prominence,"head_min_prominence",0)).astype(float)*neckline.le(_pf(max_neckline_slope,"max_neckline_slope",0)).astype(float)


def _slopes(high,low,left,right,history,points):
    rs=_bounded_line(high,left,right,history,points,high=True,output="slope"); ss=_bounded_line(low,left,right,history,points,high=False,output="slope")
    return rs,ss

def pattern_sym_triangle(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.lt(-th)&ss.gt(th)).astype(float)
def pattern_ascending_triangle(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.abs().le(th)&ss.gt(th)).astype(float)
def pattern_descending_triangle(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.lt(-th)&ss.abs().le(th)).astype(float)
def pattern_rising_wedge(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.gt(th)&ss.gt(th)&ss.gt(rs)).astype(float)
def pattern_falling_wedge(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.lt(-th)&ss.lt(-th)&rs.lt(ss)).astype(float)
def pattern_rectangle(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.abs().le(th)&ss.abs().le(th)).astype(float)
def pattern_rising_channel(high,low,left_window,right_window,history_window,points,slope_threshold,parallel_tolerance):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0); pt=_pf(parallel_tolerance,"parallel_tolerance",0)
    return (rs.gt(th)&ss.gt(th)&(rs-ss).abs().le(pt)).astype(float)
def pattern_falling_channel(high,low,left_window,right_window,history_window,points,slope_threshold,parallel_tolerance):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0); pt=_pf(parallel_tolerance,"parallel_tolerance",0)
    return (rs.lt(-th)&ss.lt(-th)&(rs-ss).abs().le(pt)).astype(float)
def pattern_broadening(high,low,left_window,right_window,history_window,points,slope_threshold):
    rs,ss=_slopes(high,low,left_window,right_window,history_window,points); th=_pf(slope_threshold,"slope_threshold",0)
    return (rs.gt(th)&ss.lt(-th)).astype(float)


def pattern_bull_flag(close,high,low,volume,impulse_window,flag_window,min_impulse,max_retracement,max_width,volume_decay_threshold):
    iw=_pi(impulse_window,"impulse_window",2); fw=_pi(flag_window,"flag_window",2)
    impulse=close.shift(fw)/close.shift(fw+iw)-1.0
    peak=close.shift(fw); trough=low.rolling(fw,min_periods=fw).min(); retr=(peak-trough)/peak.abs().replace(0,np.nan)
    width=ts_consolidation_width(high,low,fw)/close.abs().replace(0,np.nan)
    vdec=ts_consolidation_volume_decay(volume,fw)
    return (impulse.ge(_pf(min_impulse,"min_impulse",0))&retr.le(_pf(max_retracement,"max_retracement",0))&width.le(_pf(max_width,"max_width",0))&vdec.ge(_pf(volume_decay_threshold,"volume_decay_threshold"))).astype(float)

def pattern_bear_flag(close,high,low,volume,impulse_window,flag_window,min_impulse,max_retracement,max_width,volume_decay_threshold):
    iw=_pi(impulse_window,"impulse_window",2); fw=_pi(flag_window,"flag_window",2)
    impulse=close.shift(fw)/close.shift(fw+iw)-1.0
    base=close.shift(fw); rebound=high.rolling(fw,min_periods=fw).max(); retr=(rebound-base)/base.abs().replace(0,np.nan)
    width=ts_consolidation_width(high,low,fw)/close.abs().replace(0,np.nan); vdec=ts_consolidation_volume_decay(volume,fw)
    return (impulse.le(-_pf(min_impulse,"min_impulse",0))&retr.le(_pf(max_retracement,"max_retracement",0))&width.le(_pf(max_width,"max_width",0))&vdec.ge(_pf(volume_decay_threshold,"volume_decay_threshold"))).astype(float)


_SPECS=[
("ts_nth_pivot_high",["high","left_window","right_window","history_window","n"],ts_nth_pivot_high,"Nth most recent confirmed pivot high."),
("ts_nth_pivot_low",["low","left_window","right_window","history_window","n"],ts_nth_pivot_low,"Nth most recent confirmed pivot low."),
("ts_nth_pivot_high_age",["high","left_window","right_window","history_window","n"],ts_nth_pivot_high_age,"Age of nth confirmed pivot high."),
("ts_nth_pivot_low_age",["low","left_window","right_window","history_window","n"],ts_nth_pivot_low_age,"Age of nth confirmed pivot low."),
("ts_pivot_high_count",["high","left_window","right_window","history_window"],ts_pivot_high_count,"Count of confirmed pivot highs in bounded history."),
("ts_pivot_low_count",["low","left_window","right_window","history_window"],ts_pivot_low_count,"Count of confirmed pivot lows in bounded history."),
("ts_pivot_high_spacing",["high","left_window","right_window","history_window"],ts_pivot_high_spacing,"Spacing between latest two confirmed pivot highs."),
("ts_pivot_low_spacing",["low","left_window","right_window","history_window"],ts_pivot_low_spacing,"Spacing between latest two confirmed pivot lows."),
("ts_swing_amplitude",["high","low","left_window","right_window","history_window"],ts_swing_amplitude,"Absolute amplitude between latest confirmed high and low pivots."),
("ts_swing_amplitude_pct",["high","low","close","left_window","right_window","history_window"],ts_swing_amplitude_pct,"Latest swing amplitude normalized by close."),
("ts_swing_duration",["high","low","left_window","right_window","history_window"],ts_swing_duration,"Bar distance between latest opposite confirmed pivots."),
("ts_swing_velocity",["high","low","left_window","right_window","history_window"],ts_swing_velocity,"Latest swing amplitude per bar."),
("ts_swing_amplitude_atr",["high","low","close","left_window","right_window","history_window","atr_window"],ts_swing_amplitude_atr,"Latest swing amplitude normalized by ATR."),
("ts_channel_width",["high","low","left_window","right_window","history_window","points"],ts_channel_width,"Projected resistance minus support width."),
("ts_channel_width_pct",["close","high","low","left_window","right_window","history_window","points"],ts_channel_width_pct,"Projected channel width normalized by close."),
("ts_channel_width_atr",["high","low","close","left_window","right_window","history_window","points","atr_window"],ts_channel_width_atr,"Projected channel width normalized by ATR."),
("ts_channel_width_slope",["high","low","left_window","right_window","history_window","points","window"],ts_channel_width_slope,"Rolling slope of projected channel width."),
("ts_line_convergence",["high","low","left_window","right_window","history_window","points"],ts_line_convergence,"Support slope minus resistance slope; positive means convergence."),
("ts_line_parallelism",["high","low","left_window","right_window","history_window","points"],ts_line_parallelism,"Negative absolute support/resistance slope difference."),
("ts_resistance_fit_r2",["high","left_window","right_window","history_window","points"],ts_resistance_fit_r2,"R-squared of recent confirmed resistance pivots."),
("ts_support_fit_r2",["low","left_window","right_window","history_window","points"],ts_support_fit_r2,"R-squared of recent confirmed support pivots."),
("ts_pattern_symmetry",["high","low","left_window","right_window","history_window"],ts_pattern_symmetry,"Continuous symmetry score of recent pivot highs/lows."),
("ts_impulse_return",["close","window"],ts_impulse_return,"Return over a configurable impulse window."),
("ts_impulse_strength",["close","window","vol_window"],ts_impulse_strength,"Impulse return normalized by recent close-to-close volatility."),
("ts_impulse_volume",["volume","window","baseline_window"],ts_impulse_volume,"Impulse-period volume versus a prior baseline."),
("ts_consolidation_width",["high","low","window"],ts_consolidation_width,"High-low width of a consolidation window."),
("ts_consolidation_slope",["close","window"],ts_consolidation_slope,"OLS slope inside a consolidation window."),
("ts_consolidation_volume_decay",["volume","window"],ts_consolidation_volume_decay,"Positive score when volume trends down through consolidation."),
("pattern_double_top",["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"],pattern_double_top,"Continuous confirmed double-top structure score."),
("pattern_double_bottom",["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"],pattern_double_bottom,"Continuous confirmed double-bottom structure score."),
("pattern_head_shoulders",["high","low","left_window","right_window","history_window","shoulder_tolerance","head_min_prominence","max_neckline_slope"],pattern_head_shoulders,"Confirmed head-and-shoulders structure score."),
("pattern_inverse_head_shoulders",["high","low","left_window","right_window","history_window","shoulder_tolerance","head_min_prominence","max_neckline_slope"],pattern_inverse_head_shoulders,"Confirmed inverse head-and-shoulders structure score."),
("pattern_sym_triangle",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_sym_triangle,"Symmetrical triangle structure."),
("pattern_ascending_triangle",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_ascending_triangle,"Ascending triangle structure."),
("pattern_descending_triangle",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_descending_triangle,"Descending triangle structure."),
("pattern_rising_wedge",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_rising_wedge,"Rising wedge structure."),
("pattern_falling_wedge",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_falling_wedge,"Falling wedge structure."),
("pattern_rectangle",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_rectangle,"Flat support/resistance rectangle structure."),
("pattern_rising_channel",["high","low","left_window","right_window","history_window","points","slope_threshold","parallel_tolerance"],pattern_rising_channel,"Rising parallel channel structure."),
("pattern_falling_channel",["high","low","left_window","right_window","history_window","points","slope_threshold","parallel_tolerance"],pattern_falling_channel,"Falling parallel channel structure."),
("pattern_broadening",["high","low","left_window","right_window","history_window","points","slope_threshold"],pattern_broadening,"Broadening/megaphone structure."),
("pattern_bull_flag",["close","high","low","volume","impulse_window","flag_window","min_impulse","max_retracement","max_width","volume_decay_threshold"],pattern_bull_flag,"Bull flag: prior positive impulse plus bounded consolidation."),
("pattern_bear_flag",["close","high","low","volume","impulse_window","flag_window","min_impulse","max_retracement","max_width","volume_decay_threshold"],pattern_bear_flag,"Bear flag: prior negative impulse plus bounded consolidation."),
]
for _name,_params,_fn,_desc in _SPECS: _register(_name,_params,_fn,_desc,category="chart_pattern" if _name.startswith("pattern_") else "price_structure")
