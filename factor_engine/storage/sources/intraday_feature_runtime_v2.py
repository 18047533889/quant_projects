# -*- coding: utf-8 -*-
"""Session-clock and per-execution cache for intraday→daily aggregation."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from . import intraday_feature_extension as base


def _hm(text:str)->int:
    h,m=(int(x) for x in str(text).split(":"))
    if not (0<=h<=23 and 0<=m<=59):raise ValueError(f"invalid HH:MM {text!r}")
    return h*60+m
def _session_segments(dataset:str,session_open:str,session_close:str):
    if dataset=="ashare_stock_minute":return [(_hm("09:30"),_hm("11:30")),(_hm("13:00"),_hm("15:00"))]
    return [(_hm(session_open),_hm(session_close))]
def _effective_minutes(dataset,session_open,session_close,cutoff):
    end=_hm(cutoff);total=0
    for start,stop in _session_segments(dataset,session_open,session_close):
        if end>start:total+=max(0,min(end,stop)-start)
    return max(1,total)
def _ordinal(ts:pd.Series,dataset,session_open,session_close):
    # Session *labels* need a one-slot separator at the boundary so an observed
    # 11:30 and 13:00 can never land in the same N-minute bucket. Coverage uses
    # _effective_minutes separately and therefore keeps the true 240/390-minute
    # trading-session denominator.
    minute=ts.dt.hour*60+ts.dt.minute;out=pd.Series(np.nan,index=ts.index,dtype=float);offset=0
    for start,stop in _session_segments(dataset,session_open,session_close):
        mask=(minute>=start)&(minute<=stop);out.loc[mask]=offset+(minute.loc[mask]-start);offset+=stop-start+1
    return out
def _clock_bars(grp:pd.DataFrame,bar_minutes:int,dataset:str,session_open:str,session_close:str):
    grp=grp.sort_values("timestamp").copy();ordv=_ordinal(grp["timestamp"],dataset,session_open,session_close);grp=grp[ordv.notna()].copy();grp["_slot"]=(ordv[ordv.notna()]//max(1,int(bar_minutes))).astype(int)
    rows=[]
    for _,g in grp.groupby("_slot",sort=True):
        rows.append({"timestamp":g["timestamp"].iloc[-1],"open":float(g["open"].iloc[0]),"high":float(g["high"].max()),"low":float(g["low"].min()),"close":float(g["close"].iloc[-1]),"volume":float(pd.to_numeric(g["volume"],errors="coerce").fillna(0).sum()),"amount":float(pd.to_numeric(g["amount"],errors="coerce").fillna(0).sum())})
    return pd.DataFrame(rows)
def _cache(source,name):
    attr=f"_intraday_{name}_cache";cache=getattr(source,attr,None)
    if cache is None:cache={};setattr(source,attr,cache)
    return cache
def _grouped_bars(source,dataset,session_open,session_close,cutoff,bar_minutes,min_coverage,history_days):
    key=(dataset,session_open,session_close,cutoff,int(bar_minutes),float(min_coverage),int(history_days));cache=_cache(source,"bars")
    if key in cache:return cache[key]
    src=base._child(source,dataset,history_days);frame_key=(dataset,int(history_days));frame_cache=_cache(source,"frame")
    if frame_key not in frame_cache:frame_cache[frame_key]=base._wide_frame(src)
    frame=frame_cache[frame_key].copy();frame=base._hhmm_filter(frame,session_open,cutoff);frame["date"]=frame["timestamp"].dt.normalize()
    usable=_effective_minutes(dataset,session_open,session_close,cutoff);expected=max(1,int(math.ceil(usable/bar_minutes)));min_bars=max(2,int(math.ceil(expected*min_coverage)));grouped=[]
    for (date,inst),grp in frame.groupby(["date","instrument"],sort=True):
        bar=_clock_bars(grp,bar_minutes,dataset,session_open,session_close);coverage=len(bar)/expected
        if len(bar)>=min_bars and coverage>=min_coverage:grouped.append((pd.Timestamp(date),str(inst),bar))
    cache[key]=(src,grouped,expected);return cache[key]
def load_intraday_feature(source,params:dict[str,Any]):
    feature=str(params.get("feature") or "").strip()
    if not feature:raise ValueError("intraday_feature requires feature")
    bar_minutes=max(1,int(params.get("bar_minutes",5)));min_coverage=float(params.get("min_coverage",0.8));history_days=max(0,int(params.get("history_days",0)))
    if not 0<min_coverage<=1:raise ValueError("min_coverage must be in (0,1]")
    dataset,session_open,session_close,_=base._dataset_and_session(source,params);cutoff=str(params.get("cutoff_time") or "session_close");cutoff=session_close if cutoff=="session_close" else cutoff
    if _hm(cutoff)<_hm(session_open) or _hm(cutoff)>_hm(session_close):raise ValueError("cutoff_time must lie inside configured regular session")
    src,grouped,_=_grouped_bars(source,dataset,session_open,session_close,cutoff,bar_minutes,min_coverage,history_days)
    profile_features={"profile_zscore","profile_deviation","abnormal_volume_profile","abnormal_return_profile","abnormal_vol_profile"};values={}
    if feature in profile_features:
        if history_days<2:raise ValueError("profile features require history_days >= 2")
        values=base._profile_scores(grouped,feature,history_days)
    else:
        prev={}
        for date,inst,bar in grouped:
            values[(date,inst)]=base._calc(feature,bar,params,prev_close=prev.get(inst,np.nan));prev[inst]=float(bar["close"].iloc[-1])
    if hasattr(source,"_record_dependency"):source._record_dependency(dataset,kind="minute_to_daily",snapshot_id=getattr(src,"data_snapshot_id",None),feature=feature,bar_minutes=bar_minutes,cutoff_time=cutoff,min_coverage=min_coverage,join_policy="exact_date")
    anchor=source._anchor_index()
    if not values:return pd.Series(np.nan,index=anchor,name=f"intraday_{feature}")
    idx=pd.MultiIndex.from_tuples(list(values),names=["timestamp","instrument"]);series=pd.Series(list(values.values()),index=idx,name=f"intraday_{feature}").sort_index();return source._align_exact_by_instrument(anchor,series)
def install_intraday_feature_runtime():
    from .lqtp_logical_source_v2 import LQTPLogicalDataSource
    if getattr(LQTPLogicalDataSource,"_intraday_feature_v2_installed",False):return
    original=LQTPLogicalDataSource._minute_daily
    def patched(self,field,transform,params):
        if transform=="intraday_feature":return load_intraday_feature(self,dict(params))
        return original(self,field,transform,params)
    LQTPLogicalDataSource._minute_daily=patched;LQTPLogicalDataSource._intraday_feature_v2_installed=True
