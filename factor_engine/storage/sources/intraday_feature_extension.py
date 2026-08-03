# -*- coding: utf-8 -*-
"""Runtime implementation for ``intraday_feature`` SourceRef transforms.

The extension is installed lazily by ``api.intraday_daily``.  It preserves the
normal minute_at/range/bar behavior and only intercepts transform=
``intraday_feature``.  Non-A-share minute datasets must be explicitly pinned via
``minute_dataset`` (DataAccessSource params, function kwarg, or environment) so a
US strategy can never silently read the A-share minute mirror.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from runtime.ashare_intraday import (
    ashare_limit_prices,
    limit_touch_fraction,
    return_path_features,
    trade_structure_features,
)

from .data_access_source import DataAccessSource, MissingDataDependencyError

_EPS=1e-12


def _dataset_and_session(source, params: dict[str, Any]):
    inner_params=dict(getattr(source.inner,"params",{}) or {})
    dataset=str(params.get("minute_dataset") or inner_params.get("minute_dataset") or os.environ.get("FACTOR_ENGINE_MINUTE_DATASET","")).strip()
    primary=str(getattr(source.inner,"dataset","") or "").lower()
    if not dataset:
        if primary.startswith("ashare_") or primary.startswith("a_share"):
            dataset="ashare_stock_minute"
        else:
            raise MissingDataDependencyError(
                "intraday→daily features require an explicit minute_dataset for non-A-share sources; "
                "set data-source params.minute_dataset or FACTOR_ENGINE_MINUTE_DATASET"
            )
    if dataset=="ashare_stock_minute":
        session_open=str(params.get("session_open") or "09:30")
        session_close=str(params.get("session_close") or "15:00")
        session_minutes=int(params.get("session_minutes") or 240)
    else:
        session_open=str(params.get("session_open") or inner_params.get("session_open") or os.environ.get("FACTOR_ENGINE_SESSION_OPEN","")).strip()
        session_close=str(params.get("session_close") or inner_params.get("session_close") or os.environ.get("FACTOR_ENGINE_SESSION_CLOSE","")).strip()
        raw_minutes=params.get("session_minutes") or inner_params.get("session_minutes") or os.environ.get("FACTOR_ENGINE_SESSION_MINUTES","")
        if not session_open or not session_close or not raw_minutes:
            raise MissingDataDependencyError(
                f"minute dataset {dataset!r} requires explicit session_open/session_close/session_minutes"
            )
        session_minutes=int(raw_minutes)
    if session_minutes<=0: raise ValueError("session_minutes must be positive")
    return dataset,session_open,session_close,session_minutes


def _child(source,dataset: str,history_days: int):
    start=getattr(source.inner,"start_date",None); end=getattr(source.inner,"end_date",None)
    if start and history_days>0:
        start=(pd.Timestamp(start)-pd.Timedelta(days=max(7,history_days*3))).strftime("%Y-%m-%d")
    return DataAccessSource(dataset=dataset,start_date=start,end_date=end,instrument_filter=getattr(source.inner,"instrument_filter",None))


def _load_field(src: DataAccessSource, candidates: tuple[str,...], *, required=True):
    last=None
    for name in candidates:
        try: return src.load_column(name)
        except Exception as exc: last=exc
    if required: raise MissingDataDependencyError(f"minute dataset missing required field candidates={candidates}: {last}")
    return None


def _wide_frame(src: DataAccessSource):
    fields={
        "open":_load_field(src,("Open","open"),required=False),
        "high":_load_field(src,("High","high"),required=False),
        "low":_load_field(src,("Low","low"),required=False),
        "close":_load_field(src,("Close","close"),required=True),
        "volume":_load_field(src,("Volume","volume"),required=False),
        "amount":_load_field(src,("Amount","amount","DollarVolume","dollar_volume"),required=False),
    }
    close=fields["close"]
    base=close.rename("close").reset_index(); base.columns=["timestamp","instrument","close"]
    for key,series in fields.items():
        if key=="close" or series is None: continue
        temp=series.rename(key).reset_index(); temp.columns=["timestamp","instrument",key]
        base=base.merge(temp,on=["timestamp","instrument"],how="left",sort=False)
    base["timestamp"]=pd.to_datetime(base["timestamp"])
    for key in ("open","high","low"):
        if key not in base: base[key]=base["close"]
    if "volume" not in base: base["volume"]=1.0
    if "amount" not in base: base["amount"]=base["close"].abs()*base["volume"]
    return base


def _hhmm_filter(frame,session_open,cutoff):
    hh=frame["timestamp"].dt.strftime("%H:%M")
    return frame[(hh>=session_open)&(hh<=cutoff)].copy()


def _bars(grp: pd.DataFrame,bar_minutes: int):
    grp=grp.sort_values("timestamp").copy(); n=max(1,int(bar_minutes)); grp["_bar"]=np.arange(len(grp))//n
    rows=[]
    for _,g in grp.groupby("_bar",sort=True):
        volume=float(pd.to_numeric(g["volume"],errors="coerce").fillna(0).sum()); amount=float(pd.to_numeric(g["amount"],errors="coerce").fillna(0).sum())
        rows.append({"timestamp":g["timestamp"].iloc[-1],"open":float(g["open"].iloc[0]),"high":float(g["high"].max()),"low":float(g["low"].min()),"close":float(g["close"].iloc[-1]),"volume":volume,"amount":amount})
    return pd.DataFrame(rows)


def _returns(bar):
    c=np.asarray(bar["close"],float); out=np.full(len(c),np.nan)
    if len(c)>1:
        with np.errstate(divide="ignore",invalid="ignore"): out[1:]=np.log(c[1:]/c[:-1])
    return out

def _rv(r):
    x=np.asarray(r,float); x=x[np.isfinite(x)]; return float(np.sum(x*x)) if len(x) else np.nan

def _trend(values):
    y=np.asarray(values,float); mask=np.isfinite(y); y=y[mask]
    if len(y)<3:return np.nan,np.nan
    x=np.arange(len(y),dtype=float); xb=x.mean(); yb=y.mean(); den=np.sum((x-xb)**2)
    slope=np.sum((x-xb)*(y-yb))/den; fit=yb+slope*(x-xb); tot=np.sum((y-yb)**2); res=np.sum((y-fit)**2); r2=1-res/tot if tot>_EPS else 1.0
    return float(slope),float(r2)

def _share_stats(values):
    a=np.asarray(values,float); a=np.where(np.isfinite(a)&(a>0),a,0.0); total=a.sum()
    if total<=0:return np.nan,np.nan
    p=a/total; hhi=float(np.sum(p*p)); nz=p[p>0]; ent=float(-np.sum(nz*np.log(nz))/math.log(max(2,len(p))))
    return hhi,ent

def _max_drawdown(c):
    a=np.asarray(c,float); a=a[np.isfinite(a)]
    if len(a)<2:return np.nan
    peak=np.maximum.accumulate(a); return float(np.min(a/np.where(peak==0,np.nan,peak)-1.0))
def _max_runup(c):
    a=np.asarray(c,float); a=a[np.isfinite(a)]
    if len(a)<2:return np.nan
    trough=np.minimum.accumulate(a); return float(np.max(a/np.where(trough==0,np.nan,trough)-1.0))
def _corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[m],b[m])[0,1]) if m.sum()>=3 and np.std(a[m])>_EPS and np.std(b[m])>_EPS else np.nan


def _calc(feature,bar,params,*,prev_close=np.nan):
    r=_returns(bar); finite=r[np.isfinite(r)]; close=np.asarray(bar["close"],float); high=np.asarray(bar["high"],float); low=np.asarray(bar["low"],float); volume=np.asarray(bar["volume"],float); amount=np.asarray(bar["amount"],float)
    n=len(bar); rv=_rv(r); bv=float(np.pi/2*np.nansum(np.abs(finite[1:])*np.abs(finite[:-1]))) if len(finite)>1 else np.nan
    if feature=="realized_variance":return rv
    if feature=="realized_vol":return float(np.sqrt(rv)) if np.isfinite(rv) else np.nan
    if feature=="realized_skew":return float(np.sqrt(len(finite))*np.sum(finite**3)/(np.sum(finite**2)**1.5)) if len(finite)>=3 and np.sum(finite**2)>_EPS else np.nan
    if feature=="realized_kurtosis":return float(len(finite)*np.sum(finite**4)/(np.sum(finite**2)**2)) if len(finite)>=4 and np.sum(finite**2)>_EPS else np.nan
    if feature=="realized_quarticity":return float(len(finite)/3*np.sum(finite**4)) if len(finite)>=4 else np.nan
    if feature=="upside_semivariance":return float(np.sum(finite[finite>0]**2)) if len(finite) else np.nan
    if feature=="downside_semivariance":return float(np.sum(finite[finite<0]**2)) if len(finite) else np.nan
    if feature=="bipower_variation":return bv
    if feature=="jump_variation":return max(0.0,rv-bv) if np.isfinite(rv) and np.isfinite(bv) else np.nan
    if feature=="jump_ratio":return max(0.0,rv-bv)/rv if np.isfinite(rv) and np.isfinite(bv) and rv>_EPS else np.nan
    if feature=="signed_jump":
        th=float(params.get("threshold",0)); x=finite[np.abs(finite)>=th]; return float(np.sum(x)) if len(x) else 0.0
    if feature=="max_abs_return":return float(np.max(np.abs(finite))) if len(finite) else np.nan
    if feature=="tail_return_sum":
        q=float(params.get("q",0.95));
        if not 0<q<1: raise ValueError("q must be in (0,1)")
        th=np.quantile(np.abs(finite),q) if len(finite) else np.nan; return float(np.sum(finite[np.abs(finite)>=th])) if len(finite) else np.nan
    if feature=="jump_count":return float(np.sum(np.abs(finite)>=float(params.get("threshold",0.01))))
    if feature in {"return","open_to_close_return"}:return float(close[-1]/bar["open"].iloc[0]-1.0) if n else np.nan
    minutes=max(1,int(params.get("minutes",30))); bars=max(1,int(math.ceil(minutes/max(1,int(params.get("bar_minutes",1))))))
    if feature=="first_nmin_return":return float(close[min(n-1,bars-1)]/bar["open"].iloc[0]-1.0) if n else np.nan
    if feature=="last_nmin_return":return float(close[-1]/close[max(0,n-bars-1)]-1.0) if n>1 else np.nan
    split=str(params.get("split_time","12:00")); mask=(bar["timestamp"].dt.strftime("%H:%M")<=split).to_numpy();
    if feature=="morning_return":return float(close[np.where(mask)[0][-1]]/bar["open"].iloc[0]-1.0) if mask.any() else np.nan
    if feature=="afternoon_return":
        idx=np.where(~mask)[0]; return float(close[-1]/close[max(0,idx[0]-1)]-1.0) if len(idx) else np.nan
    if feature=="morning_afternoon_reversal":
        idx=np.where(~mask)[0]; mr=float(close[np.where(mask)[0][-1]]/bar["open"].iloc[0]-1.0) if mask.any() else np.nan; ar=float(close[-1]/close[max(0,idx[0]-1)]-1.0) if len(idx) else np.nan; return -mr*ar if np.isfinite(mr) and np.isfinite(ar) else np.nan
    slope,r2=_trend(np.log(np.where(close>0,close,np.nan)))
    if feature=="trend_slope":return slope
    if feature=="trend_r2":return r2
    if feature=="path_length":return return_path_features(close)["path_length"]
    if feature=="path_efficiency":return return_path_features(close)["path_efficiency"]
    if feature=="reversal_count":return return_path_features(close)["reversal_count"]
    if feature=="return_autocorr":
        lag=max(1,int(params.get("lag",1))); return _corr(finite[lag:],finite[:-lag]) if len(finite)>lag else np.nan
    if feature=="max_drawdown":return _max_drawdown(close)
    if feature=="max_runup":return _max_runup(close)
    if feature=="time_of_high":return float(np.nanargmax(high)/max(1,n-1)) if n else np.nan
    if feature=="time_of_low":return float(np.nanargmin(low)/max(1,n-1)) if n else np.nan
    if feature=="close_location":return float((close[-1]-np.nanmin(low))/(np.nanmax(high)-np.nanmin(low))) if n and np.nanmax(high)>np.nanmin(low) else np.nan
    if feature=="opening_range":return float(np.nanmax(high[:bars])-np.nanmin(low[:bars])) if n else np.nan
    if feature=="opening_range_position":
        hi,lo=np.nanmax(high[:bars]),np.nanmin(low[:bars]); return float((close[-1]-lo)/(hi-lo)) if hi>lo else np.nan
    if feature=="opening_drive":return float(close[min(n-1,bars-1)]/bar["open"].iloc[0]-1.0) if n else np.nan
    if feature=="gap_continuation":
        if not np.isfinite(prev_close) or prev_close==0:return np.nan
        gap=bar["open"].iloc[0]/prev_close-1.0; day=close[-1]/bar["open"].iloc[0]-1.0; return float(np.sign(gap)*day)
    if feature=="gap_fill_ratio":
        if not np.isfinite(prev_close) or prev_close==0:return np.nan
        op=float(bar["open"].iloc[0]); gap=op-prev_close
        if abs(gap)<=_EPS:return 1.0
        best=np.nanmin(low) if gap>0 else np.nanmax(high); filled=(op-best)/gap if gap>0 else (best-op)/(-gap); return float(np.clip(filled,0,1))
    if feature in {"limit_up_touch_fraction","limit_down_touch_fraction","limit_up_close","limit_down_close"}:
        instrument=str(params.get("instrument") or "")
        trade_date=params.get("trade_date") or bar["timestamp"].iloc[-1]
        upper,lower=ashare_limit_prices(prev_close,instrument,trade_date,is_st=bool(params.get("is_st",False)))
        if feature=="limit_up_touch_fraction":return limit_touch_fraction(bar,upper,direction="up")
        if feature=="limit_down_touch_fraction":return limit_touch_fraction(bar,lower,direction="down")
        if feature=="limit_up_close":return float(close[-1]>=upper-_EPS) if np.isfinite(upper) else np.nan
        return float(close[-1]<=lower+_EPS) if np.isfinite(lower) else np.nan
    if feature=="closing_return":return float(close[-1]/close[max(0,n-bars-1)]-1.0) if n>1 else np.nan
    if feature=="closing_ramp":return _trend(np.log(np.where(close[-bars:]>0,close[-bars:],np.nan)))[0] if n else np.nan
    vwap=np.nansum(amount)/np.nansum(volume) if np.nansum(volume)>_EPS else np.nan
    if feature=="vwap":return float(vwap)
    if feature=="close_to_vwap":return float(close[-1]/vwap-1.0) if np.isfinite(vwap) and vwap!=0 else np.nan
    if feature=="high_to_vwap":return float(np.nanmax(high)/vwap-1.0) if np.isfinite(vwap) and vwap!=0 else np.nan
    if feature=="low_to_vwap":return float(np.nanmin(low)/vwap-1.0) if np.isfinite(vwap) and vwap!=0 else np.nan
    bar_vwap=np.where(volume>0,amount/volume,np.nan); dev=close/bar_vwap-1.0
    if feature=="vwap_slope":return _trend(bar_vwap)[0]
    if feature=="vwap_deviation_mean":return float(np.nanmean(dev))
    if feature=="vwap_deviation_std":return float(np.nanstd(dev,ddof=1)) if n>1 else np.nan
    if feature=="vwap_cross_count":
        s=np.sign(dev); return float(np.sum((s[1:]*s[:-1])<0)) if n>1 else 0.0
    total_v=np.nansum(volume); total_a=np.nansum(amount)
    if feature=="volume_first_share":return float(np.nansum(volume[:bars])/total_v) if total_v>_EPS else np.nan
    if feature=="volume_last_share":return float(np.nansum(volume[-bars:])/total_v) if total_v>_EPS else np.nan
    if feature=="volume_peak_time":return float(np.nanargmax(volume)/max(1,n-1)) if n else np.nan
    vh,ve=_share_stats(volume); ah,ae=_share_stats(amount)
    if feature=="volume_hhi":return vh
    if feature=="volume_entropy":return ve
    if feature=="turnover_hhi":return ah
    if feature=="turnover_entropy":return ae
    if feature=="volume_profile_slope":return _trend(volume)[0]
    if feature=="volume_profile_skew":
        if total_v<=_EPS:return np.nan
        p=volume/total_v; x=np.arange(n,dtype=float); mu=np.sum(x*p); sd=np.sqrt(np.sum(((x-mu)**2)*p)); return float(np.sum(((x-mu)/sd)**3*p)) if sd>_EPS else 0.0
    aligned_r=np.nan_to_num(r,nan=0.0)
    if feature=="return_volume_corr":return _corr(aligned_r,volume)
    if feature=="abs_return_volume_corr":return _corr(np.abs(aligned_r),volume)
    structure=trade_structure_features(aligned_r,volume,amount)
    if feature in structure:return structure[feature]
    if feature=="volume_weighted_return":return float(np.nansum(aligned_r*volume)/total_v) if total_v>_EPS else np.nan
    if feature in {"price_impact","amihud"}:return float(np.nanmean(np.abs(aligned_r)/np.where(amount>0,amount,np.nan)))
    if feature=="turnover_per_volatility":return float(total_a/np.sqrt(rv)) if np.isfinite(rv) and rv>_EPS else np.nan
    raise MissingDataDependencyError(f"unsupported intraday feature {feature!r}")


def _profile_vector(feature,bar):
    r=np.nan_to_num(_returns(bar),nan=0.0); volume=np.asarray(bar["volume"],float)
    if feature=="abnormal_volume_profile":
        s=np.nansum(volume); return volume/s if s>_EPS else np.full(len(volume),np.nan)
    if feature=="abnormal_return_profile" or feature=="profile_deviation":return r
    if feature=="abnormal_vol_profile":return np.abs(r)
    return np.asarray([_rv(r)],float)


def _profile_scores(rows,feature,history_days):
    out={}; hist=defaultdict(lambda:deque(maxlen=history_days))
    for date,inst,bar in rows:
        vec=_profile_vector(feature,bar); prior=hist[inst]
        if len(prior)>=history_days:
            maxlen=max(len(x) for x in prior); cur=vec
            if feature=="profile_zscore":
                vals=np.asarray([float(x[0]) for x in prior],float); sd=np.nanstd(vals,ddof=1); out[(date,inst)]=(float(cur[0])-np.nanmean(vals))/sd if sd>_EPS else 0.0
            else:
                if len(cur)==maxlen and all(len(x)==maxlen for x in prior):
                    mean=np.nanmean(np.vstack(prior),axis=0); out[(date,inst)]=float(np.nanmean(np.abs(cur-mean)))
                else: out[(date,inst)]=np.nan
        else: out[(date,inst)]=np.nan
        prior.append(vec)
    return out


def load_intraday_feature(source,params: dict[str,Any]):
    feature=str(params.get("feature") or "").strip()
    if not feature: raise ValueError("intraday_feature requires feature")
    bar_minutes=max(1,int(params.get("bar_minutes",5))); min_coverage=float(params.get("min_coverage",0.8))
    if not 0<min_coverage<=1: raise ValueError("min_coverage must be in (0,1]")
    history_days=max(0,int(params.get("history_days",0)))
    dataset,session_open,session_close,session_minutes=_dataset_and_session(source,params)
    cutoff=str(params.get("cutoff_time") or "session_close"); cutoff=session_close if cutoff=="session_close" else cutoff
    if cutoff<session_open or cutoff>session_close: raise ValueError("cutoff_time must lie inside configured regular session")
    src=_child(source,dataset,history_days); frame=_wide_frame(src); frame=_hhmm_filter(frame,session_open,cutoff); frame["date"]=frame["timestamp"].dt.normalize()
    expected=max(1,int(math.ceil(session_minutes/bar_minutes))); min_bars=max(2,int(math.floor(expected*min_coverage)))
    grouped=[]
    for (date,inst),grp in frame.groupby(["date","instrument"],sort=True):
        bar=_bars(grp,bar_minutes)
        if len(bar)>=min_bars: grouped.append((pd.Timestamp(date),str(inst),bar))
    profile_features={"profile_zscore","profile_deviation","abnormal_volume_profile","abnormal_return_profile","abnormal_vol_profile"}
    values={}
    if feature in profile_features:
        if history_days<2: raise ValueError("profile features require history_days >= 2")
        values=_profile_scores(grouped,feature,history_days)
    else:
        prev={}
        for date,inst,bar in grouped:
            values[(date,inst)]=_calc(feature,bar,params,prev_close=prev.get(inst,np.nan)); prev[inst]=float(bar["close"].iloc[-1])
    if hasattr(source,"_record_dependency"):
        source._record_dependency(dataset,kind="minute_to_daily",snapshot_id=getattr(src,"data_snapshot_id",None),feature=feature,bar_minutes=bar_minutes,cutoff_time=cutoff,min_coverage=min_coverage,join_policy="exact_date")
    idx=pd.MultiIndex.from_tuples(list(values),names=["timestamp","instrument"]); series=pd.Series(list(values.values()),index=idx,name=f"intraday_{feature}").sort_index()
    return source._align_exact_by_instrument(source._anchor_index(),series)


def install_intraday_feature_runtime():
    """Compatibility no-op; v2 uses an explicit logical-source dispatcher."""
    return None
